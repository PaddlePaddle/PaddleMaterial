import argparse
import os

import numpy as np
import paddle
import paddle.nn as nn
import pandas as pd
import yaml
from paddle.io import DataLoader
from sklearn.preprocessing import StandardScaler

from ppmat.datasets import collate_fn
from ppmat.datasets.transpolymer_dataset import TransPolymerCsvDataset
from ppmat.models.transpolymer.tokenizer import PolymerSmilesTokenizer
from ppmat.models.transpolymer.transpolymer import TransPolymerRegressor


class LinearWarmupDecay(paddle.optimizer.lr.LRScheduler):
    def __init__(self, learning_rate, total_steps, warmup_steps=0, last_epoch=-1):
        self.base_lr = learning_rate
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = int(warmup_steps)
        super().__init__(learning_rate=learning_rate, last_epoch=last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        return self.base_lr * max(0.0, 1.0 - progress)


def inverse_transform(scaler, values):
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(-1)


def r2_score(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def build_dataset(split_config, tokenizer, label_mean, label_std):
    dataset_config = split_config["dataset"]
    dataset_name = dataset_config["__class_name__"]
    dataset_params = dict(dataset_config["__init_params__"])
    dataset_params.update(
        {
            "tokenizer": tokenizer,
            "label_mean": label_mean,
            "label_std": label_std,
        }
    )
    if dataset_name != "TransPolymerCsvDataset":
        raise ValueError(f"Unsupported TransPolymer dataset: {dataset_name}")
    return TransPolymerCsvDataset(**dataset_params)


def build_dataloader(split_config, dataset):
    loader_config = dict(split_config.get("loader", {}))
    sampler_config = split_config["sampler"]
    sampler_name = sampler_config["__class_name__"]
    sampler_params = dict(sampler_config["__init_params__"])
    sampler_cls = getattr(paddle.io, sampler_name)
    batch_sampler = sampler_cls(dataset, **sampler_params)

    collator_name = loader_config.pop("collate_fn", "DefaultCollator")
    collator_params = loader_config.pop("collate_params", {})
    collator_cls = getattr(collate_fn, collator_name)
    collator = collator_cls(**collator_params)

    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        return_list=True,
        collate_fn=collator,
        **loader_config,
    )


def evaluate(model, dataloader, scaler):
    model.eval()
    pred_all, true_all = [], []
    with paddle.no_grad():
        for batch in dataloader:
            pred = model(batch["input_ids"], batch["attention_mask"])
            pred_all.append(pred.numpy().reshape(-1))
            true_all.append(batch["labels"].numpy().reshape(-1))
    pred = inverse_transform(scaler, np.concatenate(pred_all))
    true = inverse_transform(scaler, np.concatenate(true_all))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return rmse, r2_score(pred, true)


def main(config):
    trainer_config = config["Trainer"]
    optimizer_config = config["Optimizer"]["__init_params__"]
    model_config = config["Model"]["__init_params__"]

    paddle.seed(trainer_config.get("seed", 1))
    np.random.seed(trainer_config.get("seed", 1))
    paddle.set_device(trainer_config.get("device", "gpu"))

    tokenizer = PolymerSmilesTokenizer.from_pretrained(
        config["Tokenizer"]["pretrained_name_or_path"],
        max_len=config["Tokenizer"]["blocksize"],
    )
    vocab_sup_file = config["Tokenizer"].get("vocab_sup_file")
    if vocab_sup_file:
        vocab_sup = pd.read_csv(vocab_sup_file, header=None).values.flatten().tolist()
        tokenizer.add_tokens(vocab_sup)

    train_config = config["Dataset"]["train"]
    test_config = config["Dataset"]["test"]
    train_file = train_config["dataset"]["__init_params__"]["file_path"]
    train_df = pd.read_csv(train_file)
    scaler = StandardScaler()
    scaler.fit(train_df.iloc[:, 1].values.reshape(-1, 1))
    label_mean = float(scaler.mean_[0])
    label_std = float(scaler.scale_[0])

    train_dataset = build_dataset(train_config, tokenizer, label_mean, label_std)
    test_dataset = build_dataset(test_config, tokenizer, label_mean, label_std)
    train_loader = build_dataloader(train_config, train_dataset)
    test_loader = build_dataloader(test_config, test_dataset)

    model_name = config["Model"]["__class_name__"]
    if model_name != "TransPolymerRegressor":
        raise ValueError(f"Unsupported TransPolymer model: {model_name}")
    model_config = dict(model_config)
    model_config["resize_vocab_size"] = len(tokenizer)
    model = TransPolymerRegressor(
        **model_config,
    )

    if config["Optimizer"]["__class_name__"] != "AdamW":
        raise ValueError("TransPolymer finetuning currently supports AdamW only.")
    total_steps = len(train_loader) * trainer_config["max_epochs"]
    warmup_steps = int(total_steps * optimizer_config.get("warmup_ratio", 0.05))
    lr_scheduler = LinearWarmupDecay(
        optimizer_config["lr_rate"], total_steps, warmup_steps
    )
    regressor_lr_scale = optimizer_config["lr_rate_reg"] / optimizer_config["lr_rate"]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=[
            {"params": model.encoder.parameters(), "learning_rate": 1.0, "weight_decay": 0.0},
            {
                "params": model.regressor.parameters(),
                "learning_rate": regressor_lr_scale,
                "weight_decay": optimizer_config.get("weight_decay", 0.01),
            },
        ],
    )
    loss_fn = nn.MSELoss()

    output_dir = trainer_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    best_r2 = -float("inf")
    stale_epochs = 0
    for epoch in range(trainer_config["max_epochs"]):
        model.train()
        for batch in train_loader:
            pred = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(pred.squeeze(), batch["labels"].squeeze())
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()

        train_rmse, train_r2 = evaluate(model, train_loader, scaler)
        test_rmse, test_r2 = evaluate(model, test_loader, scaler)
        print(f"epoch: {epoch + 1}/{trainer_config['max_epochs']}")
        print(f"train RMSE = {train_rmse:.6f}")
        print(f"train r^2 = {train_r2:.6f}")
        print(f"test RMSE = {test_rmse:.6f}")
        print(f"test r^2 = {test_r2:.6f}")

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        save_freq = trainer_config.get("save_freq", 1)
        if save_freq and (epoch + 1) % save_freq == 0:
            paddle.save(state, os.path.join(output_dir, "latest.pdparams"))
        if test_r2 > best_r2:
            best_r2 = test_r2
            stale_epochs = 0
            paddle.save(state, os.path.join(output_dir, "best.pdparams"))
        else:
            stale_epochs += 1
        if stale_epochs >= trainer_config.get("tolerance", 5):
            print("Early stop")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="./property_prediction/configs/transpolymer/transpolymer_pe_i_finetune.yaml",
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)

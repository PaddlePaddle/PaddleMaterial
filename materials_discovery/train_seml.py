import logging
import os
import random
import string
import time
from datetime import datetime

import numpy as np
import paddle
import seml
import torch
from sacred import Experiment

from gemnet.model.gemnet import GemNet
from gemnet.training.data_container import DataContainer
from gemnet.training.data_provider import DataProvider
from gemnet.training.metrics import BestMetrics
from gemnet.training.metrics import Metrics
from gemnet.training.trainer import Trainer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"


ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@ex.automain
def run(
    num_spherical,
    num_radial,
    num_blocks,
    emb_size_atom,
    emb_size_edge,
    emb_size_trip,
    emb_size_quad,
    emb_size_rbf,
    emb_size_cbf,
    emb_size_sbf,
    num_before_skip,
    num_after_skip,
    num_concat,
    num_atom,
    emb_size_bil_quad,
    emb_size_bil_trip,
    triplets_only,
    forces_coupled,
    direct_forces,
    mve,
    cutoff,
    int_cutoff,
    envelope_exponent,
    extensive,
    output_init,
    scale_file,
    data_seed,
    dataset,
    val_dataset,
    num_train,
    num_val,
    logdir,
    loss,
    tfseed,
    num_steps,
    rho_force,
    ema_decay,
    weight_decay,
    grad_clip_max,
    agc,
    decay_patience,
    decay_factor,
    decay_cooldown,
    batch_size,
    evaluation_interval,
    patience,
    save_interval,
    learning_rate,
    warmup_steps,
    decay_steps,
    decay_rate,
    staircase,
    restart,
    comment,
):
    paddle.seed(seed=tfseed)
    logging.info("Start training")
    logging.info(
        "Hyperparams: \n" + "\n".join(f"{key}: {val}" for key, val in locals().items())
    )
    num_gpus = paddle.device.cuda.device_count()
    cuda_available = paddle.device.cuda.device_count() >= 1
    logging.info(f"Available GPUs: {num_gpus}")
    logging.info(f"CUDA Available: {cuda_available}")
    if num_gpus == 0:
        logging.warning("No GPUs were found. Training is run on CPU!")
    if not cuda_available:
        logging.warning("CUDA unavailable. Training is run on CPU!")

    def id_generator(
        size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits
    ):
        return "".join(random.SystemRandom().choice(chars) for _ in range(size))

    if restart is None or restart == "None":
        directory = (
            logdir
            + "/"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + id_generator()
            + "_"
            + os.path.basename(dataset)
            + "_"
            + str(comment)
        )
    else:
        directory = restart
    logging.info(f"Directory: {directory}")
    logging.info("Create directories")
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    best_dir = os.path.join(directory, "best")
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    log_dir = os.path.join(directory, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    extension = ".pth"
    log_path_model = f"{log_dir}/model{extension}"
    log_path_training = f"{log_dir}/training{extension}"
    best_path_model = f"{best_dir}/model{extension}"
    logging.info("Initialize model")
    model = GemNet(
        num_spherical=num_spherical,
        num_radial=num_radial,
        num_blocks=num_blocks,
        emb_size_atom=emb_size_atom,
        emb_size_edge=emb_size_edge,
        emb_size_trip=emb_size_trip,
        emb_size_quad=emb_size_quad,
        emb_size_rbf=emb_size_rbf,
        emb_size_cbf=emb_size_cbf,
        emb_size_sbf=emb_size_sbf,
        num_before_skip=num_before_skip,
        num_after_skip=num_after_skip,
        num_concat=num_concat,
        num_atom=num_atom,
        emb_size_bil_quad=emb_size_bil_quad,
        emb_size_bil_trip=emb_size_bil_trip,
        num_targets=2 if mve else 1,
        triplets_only=triplets_only,
        direct_forces=direct_forces,
        forces_coupled=forces_coupled,
        cutoff=cutoff,
        int_cutoff=int_cutoff,
        envelope_exponent=envelope_exponent,
        activation="swish",
        extensive=extensive,
        output_init=output_init,
        scale_file=scale_file,
    )
    device = str("cuda" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
        "cuda", "gpu"
    )
    model.to(device)
    summary_writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    train = {}
    validation = {}
    logging.info("Load dataset")
    data_container = DataContainer(
        dataset, cutoff=cutoff, int_cutoff=int_cutoff, triplets_only=triplets_only
    )
    if val_dataset is not None:
        if num_train == 0:
            num_train = len(data_container)
        logging.info(f"Training data size: {num_train}")
        data_provider = DataProvider(
            data_container,
            num_train,
            0,
            batch_size,
            seed=data_seed,
            shuffle=True,
            random_split=True,
        )
        val_data_container = DataContainer(
            val_dataset,
            cutoff=cutoff,
            int_cutoff=int_cutoff,
            triplets_only=triplets_only,
        )
        if num_val == 0:
            num_val = len(val_data_container)
        logging.info(f"Validation data size: {num_val}")
        val_data_provider = DataProvider(
            val_data_container,
            0,
            num_val,
            batch_size,
            seed=data_seed,
            shuffle=True,
            random_split=True,
        )
    else:
        logging.info(f"Training data size: {num_train}")
        logging.info(f"Validation data size: {num_val}")
        assert num_train > 0
        assert num_val > 0
        data_provider = DataProvider(
            data_container,
            num_train,
            num_val,
            batch_size,
            seed=data_seed,
            shuffle=True,
            random_split=True,
        )
        val_data_provider = data_provider
    train["dataset_iter"] = data_provider.get_dataset("train")
    validation["dataset_iter"] = val_data_provider.get_dataset("val")
    logging.info("Prepare training")
    trainer = Trainer(
        model,
        learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        ema_decay=ema_decay,
        decay_patience=decay_patience,
        decay_factor=decay_factor,
        decay_cooldown=decay_cooldown,
        grad_clip_max=grad_clip_max,
        rho_force=rho_force,
        mve=mve,
        loss=loss,
        staircase=staircase,
        agc=agc,
    )
    train["metrics"] = Metrics("train", trainer.tracked_metrics, ex)
    validation["metrics"] = Metrics("val", trainer.tracked_metrics, ex)
    metrics_best = BestMetrics(best_dir, validation["metrics"])
    if os.path.exists(log_path_model):
        logging.info("Restoring model and trainer")
        model_checkpoint = paddle.load(path=log_path_model)
        model.set_state_dict(state_dict=model_checkpoint["model"])
        train_checkpoint = paddle.load(path=log_path_training)
        trainer.set_state_dict(state_dict=train_checkpoint["trainer"])
        metrics_best.restore()
        logging.info(f"Restored best metrics: {metrics_best.loss}")
        step_init = int(train_checkpoint["step"])
    else:
        logging.info("Freshly initialize model")
        metrics_best.inititalize()
        step_init = 0
    if ex is not None:
        ex.current_run.info = {"directory": directory}
        nparams = sum(p.size for p in model.parameters() if not p.stop_gradient)
        ex.current_run.info.update({"nParams": nparams})
    logging.info("Start training")
    steps_per_epoch = int(np.ceil(num_train / batch_size))
    for step in range(step_init + 1, num_steps + 1):
        if ex is not None:
            if step == evaluation_interval + 1:
                start = time.perf_counter()
            if step == 2 * evaluation_interval - 1:
                end = time.perf_counter()
                time_delta = end - start
                nsteps = evaluation_interval - 2
                ex.current_run.info.update(
                    {
                        "seconds_per_step": time_delta / nsteps,
                        "min_per_epoch": int(
                            time_delta / nsteps * steps_per_epoch * 100 / 60
                        )
                        / 100,
                    }
                )
        if step % 10 == 0:
            lr = trainer.schedulers[0].get_last_lr()[0]
            summary_writer.add_scalar("lr", lr, global_step=step)
        trainer.train_on_batch(train["dataset_iter"], train["metrics"])
        if step % save_interval == 0:
            paddle.save(obj={"model": model.state_dict()}, path=log_path_model)
            paddle.save(
                obj={"trainer": trainer.state_dict(), "step": step},
                path=log_path_training,
            )
        if step % evaluation_interval == 0:
            trainer.save_variable_backups()
            trainer.load_averaged_variables()
            for i in range(int(np.ceil(num_val / batch_size))):
                trainer.test_on_batch(validation["dataset_iter"], validation["metrics"])
            if validation["metrics"].loss < metrics_best.loss:
                metrics_best.update(step, validation["metrics"])
                paddle.save(obj=model.state_dict(), path=best_path_model)
            metrics_best.write(summary_writer, step)
            epoch = step // steps_per_epoch
            train_metrics_res = train["metrics"].result(append_tag=False)
            val_metrics_res = validation["metrics"].result(append_tag=False)
            metrics_strings = [
                f"{key}: train={train_metrics_res[key]:.6f}, val={val_metrics_res[key]:.6f}"
                for key in validation["metrics"].keys
            ]
            logging.info(
                f"{step}/{num_steps} (epoch {epoch}): " + "; ".join(metrics_strings)
            )
            trainer.decay_maybe(validation["metrics"].loss)
            train["metrics"].write(summary_writer, step)
            validation["metrics"].write(summary_writer, step)
            train["metrics"].reset_states()
            validation["metrics"].reset_states()
            trainer.restore_variable_backups()
            if step - metrics_best.step > patience * evaluation_interval:
                break
    return {(key + "_best"): val for key, val in metrics_best.items()}

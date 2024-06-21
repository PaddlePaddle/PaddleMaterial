from __future__ import annotations

import os
import shutil
import warnings
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm
import pickle

from models.megnet import MEGNetPlus

from utils.ext_pymatgen import Structure2Graph, get_element_list
from dataset.mgl_dataset import MGLDataset
from dataset.utils import split_dataset
from dataset.mgl_dataloader import MGLDataLoader, collate_fn_graph
from utils._bond import BondExpansion
import paddle
from utils.logger import init_logger
import argparse
import yaml
import paddle.distributed.fleet as fleet
import paddle.distributed as dist

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

if dist.get_world_size() > 1:
    fleet.init(is_collective=True)

def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    """Raw data loading function.

    Returns:
        tuple[list[Structure], list[str], list[float]]: structures, mp_id, Eform_per_atom
    """
    # if not os.path.exists("mp.2018.6.1.json"):
    #     f = RemoteFile("https://figshare.com/ndownloader/files/15087992")
    #     with zipfile.ZipFile(f.local_path) as zf:
    #         zf.extractall(".")
    data = pd.read_json("data/mp.2018.6.1.json")
    structures = []
    mp_ids = []

    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"])):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)
        if len(mp_ids) >= 100:
            break
    return structures, mp_ids, data["formation_energy_per_atom"].tolist()

def load_dataset_from_pickle(structures_path, formation_energy_path):
    with open(structures_path, 'rb') as f:
        structures = pickle.load(f)
    with open(formation_energy_path, 'rb') as f:
        formation_energy_per_atom = pickle.load(f)
    return structures, formation_energy_per_atom

def train_epoch(model, loader, loss_fn, optimizer, epoch, log):
    model.train()
    total_loss = []
    for idx, batch_data in enumerate(loader):
        graph, _, state_attr, labels = batch_data
        preds = model(graph, state_attr)

        train_loss = loss_fn(preds, labels)
        train_loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        total_loss.append(train_loss)

        if paddle.distributed.get_rank() == 0 and (idx % 10 == 0 or idx == len(loader)-1):
            message = "train: epoch %d | step %d | " % (epoch, idx)
            message += "lr %.6f | loss %.6f" % (optimizer.get_lr(), train_loss)
            log.info(message)
    return sum(total_loss)/len(total_loss)

@paddle.no_grad()
def eval_epoch(model, loader, loss_fn, metric_fn, log):
    model.eval()
    total_loss = []
    total_metric = []
    total_num_data = 0
    for idx, batch_data in enumerate(loader):
        graph, _, state_attr, labels = batch_data
        batch_size = len(labels)

        preds = model(graph, state_attr)

        eval_loss = loss_fn(preds, labels)
        total_loss.append(eval_loss)

        eval_metric = metric_fn(preds, labels)
        total_metric.append(eval_metric*batch_size)
        total_num_data += batch_size

    return sum(total_loss)/len(total_loss), sum(total_metric)/total_num_data
        

def train(cfg, log):
    fleet.init(is_collective=True)
    # structures, mp_ids, eform_per_atom = load_dataset()
    # structures = structures[:100]
    # eform_per_atom = eform_per_atom[:100]

    structures, eform_per_atom = load_dataset_from_pickle(
        structures_path=cfg['dataset']['structures_path'],
        formation_energy_path=cfg['dataset']['formation_energy_path'],
    )

    # get element types in the dataset
    elem_list = get_element_list(structures)
    # setup a graph converter
    converter = Structure2Graph(element_types=elem_list, cutoff=cfg['dataset']['cutoff'])
    # convert the raw dataset into MEGNetDataset
    mp_dataset = MGLDataset(
        structures=structures,
        labels={"Eform": eform_per_atom},
        converter=converter,
        save_dir='./data/mp2018_cache'
    )

    train_data, val_data, test_data = split_dataset(
        mp_dataset,
        frac_list=cfg['dataset']['split_list'],
        shuffle=True,
        random_state=42,
    )

    # train_loader, val_loader, test_loader = MGLDataLoader(
    #     train_data=train_data,
    #     val_data=val_data,
    #     test_data=test_data,
    #     collate_fn=collate_fn_graph,
    #     batch_size=cfg["batch_size"],
    #     num_workers=0,
    # )

    train_loader = paddle.io.DataLoader(
        train_data, 
        batch_sampler=paddle.io.DistributedBatchSampler(
            train_data,
            batch_size=cfg['batch_size'],
            shuffle=True,
        ),
        collate_fn=collate_fn_graph,
        num_workers=0,
    )
    val_loader = paddle.io.DataLoader(
        val_data, 
        batch_sampler=paddle.io.DistributedBatchSampler(
            val_data,
            batch_size=cfg['batch_size'],
        ),
        collate_fn=collate_fn_graph,
    )
    test_loader = paddle.io.DataLoader(
        test_data, 
        batch_sampler=paddle.io.DistributedBatchSampler(
            test_data,
            batch_size=cfg['batch_size'],
        ),
        collate_fn=collate_fn_graph,
    )

    # define the bond expansion
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)
    # setup the architecture of MEGNet model
    model = MEGNetPlus(
        dim_node_embedding=cfg['model']['dim_node_embedding'],
        dim_edge_embedding=cfg['model']['dim_edge_embedding'],
        dim_state_embedding=cfg['model']['dim_state_embedding'],
        nblocks=cfg['model']['nblocks'],
        hidden_layer_sizes_input=cfg["model"]['hidden_layer_sizes_input'],
        hidden_layer_sizes_conv=cfg["model"]['hidden_layer_sizes_conv'],
        nlayers_set2set=cfg["model"]['nlayers_set2set'],
        niters_set2set=cfg["model"]['niters_set2set'],
        hidden_layer_sizes_output=cfg["model"]['hidden_layer_sizes_output'],
        is_classification=cfg['model']['is_classification'],
        activation_type=cfg['model']['activation_type'],
        bond_expansion=bond_expansion,
        cutoff=cfg['model']['cutoff'],
        gauss_width=cfg['model']['gauss_width'],
    )
    # model.set_dict(paddle.load('data/paddle_weight.pdparams'))

    if dist.get_world_size() > 1:
        model = fleet.distributed_model(model)

    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(**cfg['lr_cfg'])
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
        learning_rate=lr_scheduler, epsilon=1e-08, weight_decay=0.0)
    if dist.get_world_size() > 1:
        optimizer = fleet.distributed_optimizer(optimizer)
    loss_fn = paddle.nn.functional.mse_loss
    metric_fn = paddle.nn.functional.l1_loss

    global_step = 0
    best_metric = float('inf')

    for epoch in range(cfg['epochs']):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, epoch, log)
        lr_scheduler.step()

        if paddle.distributed.get_rank() == 0:
            eval_loss, eval_metric = eval_epoch(model, val_loader, loss_fn, metric_fn, log)
            log.info("epoch: {}, train_loss: {:.6f}, eval_loss: {:.6f}, eval_mae: {:.6f}".format(epoch, train_loss.item(), eval_loss.item(), eval_metric.item()))

            if eval_metric < best_metric:
                best_metric = eval_metric
                paddle.save(model.state_dict(), '{}/best.pdparams'.format(cfg['save_path']))
                log.info("Saving best checkpoint at {}".format(cfg['save_path']))

            paddle.save(model.state_dict(), '{}/latest.pdparams'.format(cfg['save_path']))
            if epoch % 500 == 0:
                paddle.save(model.state_dict(), '{}/epoch_{}.pdparams'.format(cfg['save_path'], epoch))
    if paddle.distributed.get_rank() == 0:
        test_loss, test_metric = eval_epoch(model, test_loader, loss_fn, metric_fn, log)
        log.info("test_loss: {:.6f}, test_mae: {:.6f}".format(test_loss.item(), test_metric.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='./configs/megnet_3d.yaml', help="Path to config file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    if paddle.distributed.get_rank() == 0:
        os.makedirs(cfg['save_path'], exist_ok=True)
        shutil.copy(args.config, cfg['save_path'])
    log = init_logger(log_file=os.path.join(cfg['save_path'], 'run.log'))
    train(cfg, log)



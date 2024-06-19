from __future__ import annotations

import os
os.environ["DGLDEFAULTDIR"] = "./dgl"
import shutil
import warnings
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# import torch
# from dgl.data.utils import split_dataset
from pymatgen.core import Structure
# from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from tqdm import tqdm
import pickle

# from matgl.ext.pymatgen import Structure2Graph, get_element_list
# from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
# from matgl.layers import BondExpansion
# from matgl.models import MEGNet
# from matgl.utils.io import RemoteFile
# from matgl.utils.training import ModelLightningModule

from models.megnet import MEGNetPlus

from utils.ext_pymatgen import Structure2Graph, get_element_list
from dataset.mgl_dataset import MGLDataset
from dataset.utils import split_dataset
from dataset.mgl_dataloader import MGLDataLoader, collate_fn_graph
from utils._bond import BondExpansion
import paddle
# To suppress warnings for clearer output
warnings.simplefilter("ignore")

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

def load_dataset_from_pickle(structures_path, mp_ids_path, formation_energy_path):
    with open(structures_path, 'rb') as f:
        structures = pickle.load(f)
    with open(mp_ids_path, 'rb') as f:
        mp_ids = pickle.load(f)
    with open(formation_energy_path, 'rb') as f:
        formation_energy_per_atom = pickle.load(f)
    return structures, mp_ids, formation_energy_per_atom


structures, mp_ids, eform_per_atom = load_dataset()

# structures, mp_ids, eform_per_atom = load_dataset_from_pickle(
#     structures_path="./data/structures.pickle",
#     mp_ids_path="./data/mp_ids.pickle",
#     formation_energy_path="./data/formation_energy.pickle",
# )


structures = structures[:100]
eform_per_atom = eform_per_atom[:100]

# get element types in the dataset
elem_list = get_element_list(structures)
# setup a graph converter
converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
# convert the raw dataset into MEGNetDataset
mp_dataset = MGLDataset(
    structures=structures,
    labels={"Eform": eform_per_atom},
    converter=converter,
    save_dir='./data/mp2018_cache'
)

train_data, val_data, test_data = split_dataset(
    mp_dataset,
    # frac_list=[0.8665637, 0.06671815, 0.06671815],
    frac_list=[0.9, 0.05, 0.05],
    shuffle=True,
    random_state=42,
)

train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_graph,
    batch_size=4,
    num_workers=0,
)

# setup the embedding layer for node attributes
# node_embed = torch.nn.Embedding(len(elem_list), 16)
# define the bond expansion
bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)
# setup the architecture of MEGNet model
model = MEGNetPlus(
    dim_node_embedding=16,
    dim_edge_embedding=100,
    dim_state_embedding=2,
    nblocks=3,
    hidden_layer_sizes_input=(64, 32),
    hidden_layer_sizes_conv=(64, 64, 32),
    nlayers_set2set=1,
    niters_set2set=2,
    hidden_layer_sizes_output=(32, 16),
    is_classification=False,
    activation_type="softplus2",
    bond_expansion=bond_expansion,
    cutoff=4.0,
    gauss_width=0.5,
)
model.set_dict(paddle.load('data/paddle_weight.pdparams'))


lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(T_max=1000, eta_min=0.001 * 1000,
    learning_rate=0.001)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
    learning_rate=lr_scheduler, epsilon=1e-08, weight_decay=0.0)
loss_fn = paddle.nn.functional.mse_loss
import pdb;pdb.set_trace()
for data in train_loader:
    print(len(data))
    preds = model(data[0], data[2])
    labels = data[3]
    loss = loss_fn(labels, preds)
    loss.backward()
    optimizer.step()
    optimizer.clear_gradients()
    # lr_scheduler.step()
    print('loss: ', loss)





# save_path = './results/v6_1'

# checkpoint_callback = ModelCheckpoint(
#     monitor="val_MAE",
#     mode="min",
#     save_top_k=1,
#     save_last=True,
#     filename="{epoch}-{val_MAE:.4f}",
#     dirpath=os.path.join(save_path, "checkpoints"),
#     auto_insert_metric_name=False,
#     every_n_epochs=1,
# )

# # setup the MEGNetTrainer
# lit_module = ModelLightningModule(model=model)
# logger_csv = CSVLogger(save_path, name="logs_csv")
# logger_tbd = TensorBoardLogger(save_path, name="logs_tbd")
# trainer = pl.Trainer(max_epochs=1000, logger=[logger_csv, logger_tbd] , devices=[6]) #, strategy='ddp', devices=[2], callbacks=[checkpoint_callback])
# # trainer = pl.Trainer(max_epochs=2000, logger=[logger_csv, logger_tbd], strategy='ddp', devices=[2, 3, 4, 5], callbacks=[checkpoint_callback])
# trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
# trainer.test(model=lit_module, dataloaders=test_loader)
# # metrics = pd.read_csv("logs/MEGNet_training/version_1/metrics.csv")
# # metrics["train_MAE"].dropna().plot()
# # metrics["val_MAE"].dropna().plot()

# # _ = plt.legend()
# # import pdb;pdb.set_trace()
# # trainer = pl.Trainer(max_epochs=5, logger=logger, devices=[2])
# # trainer.test(model=lit_module, dataloaders=test_loader, ckpt_path="./logs/MEGNet_training/version_2/checkpoints/epoch=1999-step=80000.ckpt")

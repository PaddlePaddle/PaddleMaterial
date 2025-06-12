import os
import os.path as osp
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))  # ruff: noqa
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))  # ruff: noqa

from ppmat.utils import logger
from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers


if __name__ == "__main__":
    
    # init logger
    logger_path = osp.join(sys.path[0], "test/create_dataset.log")
    logger.init_logger(log_file=logger_path)
    logger.info(f"Logger saved to {logger_path}")
    logger.info(f"Test create new dataset.")

    set_signal_handlers()
    data_cfg = {
        "dataset": {
            "__class_name__": "JarvisDataset",
            "__init_params__": {
                "path": osp.join(sys.path[0], "data/jarvis"),
                "jarvis_data_name": "dft_3d",
                "property_names": "formation_energy_peratom",  # 这里改成你实际需要的标签名
                # "build_structure_cfg": {
                #     "format": "dict",
                #     "num_cpus": 10,
                # },
                # "build_graph_cfg": {
                #     "__class_name__": "FindPointsInSpheres",
                #     "__init_params__": {
                #         "cutoff": 4.0,
                #         "num_cpus": 10,
                #     },
                # },
                # "cache_path": osp.join(sys.path[0], "data/jarvis/mp2024_train_130k_cache_find_points_in_spheres_cutoff_4/mp2024_train"),
            },
            "num_workers": 4,
            "use_shared_memory": False,
        },
        "sampler": {
            "__class_name__": "BatchSampler",
            "__init_params__": {
                "shuffle": True,
                "drop_last": True,
                "batch_size": 10,
            },
        },
    }
    logger.info("Train data config:\n" + 
                "\n".join( f"{k}: {v}" for k, v in data_cfg.items() )
            )

    train_loader = build_dataloader(data_cfg)


'''
id JVASP-90856
spg_number 129
spg_symbol P4/nmm
formula TiCuSiAs
formation_energy_peratom -0.42762
func OptB88vdW
optb88vdw_bandgap 0.0
atoms {'lattice_mat': [[3.566933224304235, 0.0, -0.0], [0.0, 3.566933224304235, -0.0], [-0.0, -0.0, 9.397075454186664]], 'coords': [[2.6751975000000003, 2.6751975000000003, 7.376101754328542], [0.8917325, 0.8917325, 2.0209782456714573], [0.8917325, 2.6751975000000003, 4.69854], [2.6751975000000003, 0.8917325, 4.69854], [0.8917325, 2.6751975000000003, 0.0], [2.6751975000000003, 0.8917325, 0.0], [2.6751975000000003, 2.6751975000000003, 2.8894795605846353], [0.8917325, 0.8917325, 6.507600439415366]], 'elements': ['Ti', 'Ti', 'Cu', 'Cu', 'Si', 'Si', 'As', 'As'], 'abc': [3.56693, 3.56693, 9.39708], 'angles': [90.0, 90.0, 90.0], 'cartesian': True, 'props': ['', '', '', '', '', '', '', '']}
slme na
magmom_oszicar 0.0
spillage na
elastic_tensor na
effective_masses_300K {'p': 'na', 'n': 'na'}
kpoint_length_unit 60
maxdiff_mesh na
maxdiff_bz na
encut 650
optb88vdw_total_energy -3.37474
epsx 76.23
epsy 76.23
epsz 54.0402
mepsx na
mepsy na
mepsz na
modes na
magmom_outcar 0.0
max_efg na
avg_elec_mass na
avg_hole_mass na
icsd 
dfpt_piezo_max_eij na
dfpt_piezo_max_dij na
dfpt_piezo_max_dielectric na
dfpt_piezo_max_dielectric_electronic na
dfpt_piezo_max_dielectric_ionic na
max_ir_mode na
min_ir_mode na
n-Seebeck na
p-Seebeck na
n-powerfact na
p-powerfact na
ncond na
pcond na
nkappa na
pkappa na
ehull 0.0423
Tc_supercon na
dimensionality 3D-bulk
efg []
xml_data_link <a href=https://www.ctcms.nist.gov/~knc6/static/JARVIS-DFT/JVASP-90856.xml target='_blank' >JVASP-90856</a>
typ bulk
exfoliation_energy na
spg 129
crys tetragonal
density 5.956
poisson na
raw_files []
nat 8
bulk_modulus_kv na
shear_modulus_gv na
mbj_bandgap na
hse_gap na
reference mp-1080455
search -As-Cu-Si-Ti





lattice_mat [[3.566933224304235, 0.0, -0.0], [0.0, 3.566933224304235, -0.0], [-0.0, -0.0, 9.397075454186664]]
coords [[2.6751975000000003, 2.6751975000000003, 7.376101754328542], [0.8917325, 0.8917325, 2.0209782456714573], [0.8917325, 2.6751975000000003, 4.69854], [2.6751975000000003, 0.8917325, 4.69854], [0.8917325, 2.6751975000000003, 0.0], [2.6751975000000003, 0.8917325, 0.0], [2.6751975000000003, 2.6751975000000003, 2.8894795605846353], [0.8917325, 0.8917325, 6.507600439415366]]
elements ['Ti', 'Ti', 'Cu', 'Cu', 'Si', 'Si', 'As', 'As']
abc [3.56693, 3.56693, 9.39708]
angles [90.0, 90.0, 90.0]
cartesian True
props ['', '', '', '', '', '', '', '']

'''
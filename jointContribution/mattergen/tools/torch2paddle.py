import os
import numpy as np
import argparse
import torch
import paddle

from fc_names_mattergen_base import fc_names_mattergen_base
from fc_names_chemical_system import fc_names_chemical_system
from fc_names_chemical_system_energy_above_hull import fc_names_chemical_system_energy_above_hull
from fc_names_dft_band_gap import fc_names_dft_band_gap
from fc_names_dft_mag_density_hhi_score import fc_names_dft_mag_density_hhi_score
from fc_names_dft_mag_density import fc_names_dft_mag_density
from fc_names_ml_bulk_modulus import fc_names_ml_bulk_modulus
from fc_names_space_group import fc_names_space_group



def torch2paddle(torch_path, paddle_path, fc_names):
    # torch_path = "csp_torch.pth"
    # torch_path = "/root/host/home/zhangzhimin04/workspaces/mattergen/checkpoints/dft_mag_density_hhi_score/checkpoints/last.ckpt"
    # paddle_path = "/root/host/home/zhangzhimin04/workspaces/mattergen/checkpoints/dft_mag_density_hhi_score/checkpoints/latest.pdparams"
    torch_state_dict = torch.load(torch_path)

    paddle_state_dict = {"state_dict": {}}
    for k in torch_state_dict['state_dict']:
        if "num_batches_tracked" in k: 
            continue
        v = torch_state_dict['state_dict'][k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k:
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(
                f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}"
            )
            v = v.transpose(new_shape)  
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        k = k.replace('diffusion_module.', '')
        paddle_state_dict['state_dict'][k] = v 
    paddle.save(paddle_state_dict, paddle_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--torch_path", type=str, required=True)
    argparser.add_argument("--paddle_path", type=str, required=True)
    argparser.add_argument("--fc_names", type=str, required=True)

    args = argparser.parse_args()

    assert args.fc_names in [
        "mattergen_base", 
        "chemical_system", 
        "chemical_system_energy_above_hull", 
        "dft_band_gap", 
        "dft_mag_density_hhi_score", 
        "dft_mag_density", 
        "ml_bulk_modulus", 
        "space_group"]
    
    dir_name = os.path.dirname(args.paddle_path)
    if not os.path.exists(dir_name):
        print(f"mkdir {dir_name}")
        os.makedirs(dir_name)

    torch2paddle(args.torch_path, args.paddle_path, eval(f"fc_names_{args.fc_names}"))

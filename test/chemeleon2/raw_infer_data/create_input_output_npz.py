"""
模型精度测试 - 用于框架迁移对比

功能：
1. 测试 VAE Encoder/Decoder 的输出
2. 测试 LDM Denoiser 的输出
3. 保存中间结果用于与其他框架对比
4. 固定随机种子保证可复现
"""

import sys
import os
import json
import random
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from monty.serialization import loadfn


def save_full_tensor_data(tensor, name, output_dir):
    """
    保存完整的tensor数据到NPZ文件

    Args:
        tensor: PyTorch张量
        name: 张量名称
        output_dir: 输出目录
    """
    npz_file = output_dir / f"{name}_full.npz"
    np.savez_compressed(npz_file, data=tensor.cpu().numpy())
    return str(npz_file)


# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 切换到项目根目录，确保相对路径正确
os.chdir(project_root)

from src.vae_module.vae_module import VAEModule
from src.ldm_module.ldm_module import LDMModule
from src.data.schema import CrystalBatch
from src.data.dataset_util import pmg_structure_to_pyg_data


class Logger:
    """同时输出到控制台和文件的日志器"""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# 全局日志器
_logger = None


def set_logger(filepath):
    """设置全局日志器"""
    global _logger
    _logger = Logger(filepath)
    sys.stdout = _logger


def close_logger():
    """关闭日志器"""
    global _logger
    if _logger:
        sys.stdout = _logger.terminal
        _logger.close()
        _logger = None


def load_fixed_inputs(npz_path):
    """从保存的npz文件加载固定输入

    Args:
        npz_path: npz文件路径

    Returns:
        dict: 包含各个模块输入数据的字典
    """
    data = np.load(npz_path, allow_pickle=True)

    fixed_inputs = {}
    for key in data.keys():
        module_data = data[key].item()
        if 'input' in module_data:
            fixed_inputs[key] = module_data['input']

    return fixed_inputs


def test_single_module(module_name, vae_path, ldm_path, structure_path, device="cuda", use_fixed_input=False,
                       fixed_inputs=None):
    """单独测试某个模块

    Args:
        module_name: 'vae_encoder', 'vae_decoder', 'vae_full', 'ldm_denoiser', 'ldm_sampling'
        use_fixed_input: 是否使用固定输入
        fixed_inputs: 固定输入数据字典

    Returns:
        测试结果
    """
    print(f"\n{'=' * 80}")
    print(f"Testing: {module_name}")
    print(f"{'=' * 80}")

    if use_fixed_input and fixed_inputs and module_name in fixed_inputs:
        print(f"Using fixed input data")
        fixed_input = fixed_inputs[module_name]
    else:
        fixed_input = None
        if use_fixed_input:
            print(f"Warning: Fixed input not found, using random/default input")

    if module_name == 'vae_encoder':
        return test_vae_encoder(vae_path, structure_path, device, fixed_input)
    elif module_name == 'vae_decoder':
        return test_vae_decoder(vae_path, structure_path, device)
    elif module_name == 'vae_full':
        return test_vae_full(vae_path, structure_path, device)
    elif module_name == 'ldm_denoiser':
        return test_ldm_denoiser(ldm_path, vae_path, device, fixed_input)
    elif module_name == 'ldm_sampling':
        return test_ldm_sampling(ldm_path, vae_path, structure_path, device)
    else:
        raise ValueError(f"未知模块: {module_name}")


def set_seed(seed=42):
    """固定随机种子，确保完全可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 设置CUDA确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def import_datetime():
    """返回当前时间字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_tensor_info(tensor, name, max_dims=4):
    """打印张量信息，只显示前max_dims个值"""
    print(f"\n{name}:")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  mean: {tensor.mean().item():.8f}")
    print(f"  std: {tensor.std().item():.8f}")
    print(f"  min: {tensor.min().item():.8f}")
    print(f"  max: {tensor.max().item():.8f}")

    # 只显示前4个值
    flat = tensor.flatten()
    show_n = min(max_dims, len(flat))
    values_str = "  ".join(f"{flat[i].item():.8f}" for i in range(show_n))
    print(f"  first {show_n}: {values_str}")


def save_tensor_to_dict(tensor, prefix="", output_dir=None, tensor_name=None, save_full=True):
    """将张量信息保存为字典（JSON兼容格式）

    Args:
        tensor: PyTorch张量
        prefix: 键名前缀
        output_dir: 输出目录，如果提供则保存完整数据
        tensor_name: 张量名称，用于保存完整数据文件
        save_full: 是否保存完整数据到NPZ文件

    对于多维张量，JSON中只保存前4个值，但完整数据保存到NPZ文件
    """
    flat = tensor.flatten()
    show_n = min(4, len(flat))

    # 获取前4个值的详细列表
    first_values = [float(flat[i].item()) for i in range(show_n)]

    result = {
        f"{prefix}shape": list(tensor.shape),
        f"{prefix}dtype": str(tensor.dtype),
        f"{prefix}device": str(tensor.device),
        f"{prefix}first_4_values": first_values,
        f"{prefix}total_elements": int(tensor.numel()),
    }

    # 只对浮点数类型计算统计值
    if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
        result[f"{prefix}mean"] = float(f"{tensor.mean().item():.8f}")
        result[f"{prefix}std"] = float(f"{tensor.std().item():.8f}")
        result[f"{prefix}min"] = float(f"{tensor.min().item():.8f}")
        result[f"{prefix}max"] = float(f"{tensor.max().item():.8f}")
    else:
        # 对于整数类型，只记录min和max
        result[f"{prefix}min"] = int(tensor.min().item())
        result[f"{prefix}max"] = int(tensor.max().item())

    # 保存完整数据到NPZ文件
    if save_full and output_dir is not None and tensor_name is not None:
        full_data_path = save_full_tensor_data(tensor, tensor_name, output_dir)
        result[f"{prefix}full_data_file"] = full_data_path
        result[f"{prefix}full_data_available"] = True
    else:
        result[f"{prefix}full_data_available"] = False

    return result


def test_vae_encoder(vae_path, structure_path, device="cuda", fixed_input=None, output_dir=None):
    """测试 VAE Encoder

    Args:
        fixed_input: 如果提供，使用固定的输入数据而不是从结构加载
        output_dir: 输出目录，用于保存完整tensor数据
    """
    # 设置随机种子确保可复现
    set_seed(42)

    print("\n" + "=" * 80)
    print("VAE Encoder Test")
    print("=" * 80)

    # 加载模型
    vae = VAEModule.load_from_checkpoint(vae_path, weights_only=False)
    vae.to(device)
    vae.eval()

    if fixed_input is not None:
        # 使用固定输入
        print("\n使用固定输入数据")
        # 重建 CrystalBatch
        from torch_geometric.data import Data

        data = Data(
            atom_types=torch.from_numpy(fixed_input['atom_types']).long(),
            frac_coords=torch.from_numpy(fixed_input['frac_coords']).float(),
            cart_coords=torch.from_numpy(fixed_input.get('cart_coords', fixed_input['frac_coords'])).float(),
            lattices=torch.from_numpy(fixed_input['lattices']).float(),
            num_atoms=torch.from_numpy(fixed_input['num_atoms']).long(),
            lengths=torch.from_numpy(fixed_input.get('lengths', np.zeros((1, 3)))).float(),
            lengths_scaled=torch.from_numpy(fixed_input.get('lengths_scaled', np.zeros((1, 3)))).float(),
            angles=torch.from_numpy(fixed_input.get('angles', np.zeros((1, 3)))).float(),
            angles_radians=torch.from_numpy(fixed_input.get('angles_radians', np.zeros((1, 3)))).float(),
            token_idx=torch.arange(len(fixed_input['atom_types']), dtype=torch.long),
            pos=torch.from_numpy(fixed_input.get('cart_coords', fixed_input['frac_coords'])).float(),
        )
        batch = CrystalBatch.from_data_list([data])
        batch = batch.to(device)
        test_structure_info = "固定输入（从保存的数据加载）"
    else:
        # 加载测试结构
        structures = loadfn(structure_path)
        test_structure = structures[0]  # 使用第一个结构

        print(f"\nTest structure: {test_structure.composition}")
        print(f"  num_atoms: {len(test_structure)}")
        print(f"  space_group: {test_structure.get_space_group_info()}")

        # 准备输入
        data = pmg_structure_to_pyg_data(test_structure)
        batch = CrystalBatch.from_data_list([data])
        batch = batch.to(device)
        test_structure_info = str(test_structure.composition)

    # 保存输入数据的详细描述（JSON格式）
    input_description = {
        "input_source": "fixed_structure" if fixed_input is None else "fixed_input_data",
        "structure_info": test_structure_info,
        "batch_info": {
            "num_graphs": int(batch.num_graphs),
            "total_atoms": int(batch.num_nodes),
        },
        "input_tensors": {},
    }

    # 记录每个输入张量的信息并保存完整数据
    for attr_name in ['atom_types', 'frac_coords', 'lattices', 'num_atoms', 'lengths', 'lengths_scaled', 'angles']:
        if hasattr(batch, attr_name):
            attr_tensor = getattr(batch, attr_name)
            # 保存完整数据到NPZ
            tensor_name = f"vae_encoder_input_{attr_name}"
            input_description["input_tensors"][attr_name] = save_tensor_to_dict(
                attr_tensor, "",
                output_dir=output_dir,
                tensor_name=tensor_name,
                save_full=True
            )
            # 对于小张量，也在JSON中保存完整值
            if attr_tensor.numel() <= 20:
                input_description["input_tensors"][attr_name]["all_values"] = attr_tensor.flatten().cpu().tolist()

    # Encoder 前向传播
    with torch.no_grad():
        # 获取 encoder 输出（返回字典）
        encoder_dict = vae.encoder(batch)  # Dict with keys: x, num_atoms, batch, token_idx
        encoder_output = encoder_dict["x"]  # (N, H)
        print_tensor_info(encoder_output, "Encoder Output (x)")

        # 量化卷积
        h = vae.quant_conv(encoder_output)  # (N, 2*L)
        print_tensor_info(h, "After Quant Conv")

        # 分离 mean 和 logvar
        mean, logvar = torch.chunk(h, 2, dim=-1)
        print_tensor_info(mean, "Latent Mean")
        print_tensor_info(logvar, "Latent Logvar")

        # 采样 - 使用numpy生成固定的eps
        std = torch.exp(0.5 * logvar)
        np.random.seed(42)
        eps_np = np.random.randn(*std.shape).astype(np.float32)
        eps = torch.from_numpy(eps_np).to(device)
        z = mean + eps * std
        print_tensor_info(z, "Latent Z (Sampled)")

    # 保存为字典，用于JSON输出，同时保存完整数据
    result = {
        "module_name": "vae_encoder",
        "description": "VAE编码器：输入晶体结构，输出潜在向量的均值和方差",
        "input": input_description,
        "output": {
            "encoder_output": save_tensor_to_dict(
                encoder_output, "", output_dir, "vae_encoder_output", True
            ),
            "quant_conv_output": save_tensor_to_dict(
                h, "", output_dir, "vae_encoder_quant_conv", True
            ),
            "latent_mean": save_tensor_to_dict(
                mean, "", output_dir, "vae_encoder_latent_mean", True
            ),
            "latent_logvar": save_tensor_to_dict(
                logvar, "", output_dir, "vae_encoder_latent_logvar", True
            ),
            "latent_z_sampled": save_tensor_to_dict(
                z, "", output_dir, "vae_encoder_latent_z", True
            ),
        },
    }

    return result


def test_vae_decoder(vae_path, structure_path, device="cuda", output_dir=None):
    """测试 VAE Decoder

    Args:
        output_dir: 输出目录，用于保存完整tensor数据
    """
    # 设置随机种子确保可复现
    set_seed(42)

    print("\n" + "=" * 80)
    print("VAE Decoder Test")
    print("=" * 80)

    # 加载模型
    vae = VAEModule.load_from_checkpoint(vae_path, weights_only=False)
    vae.to(device)
    vae.eval()

    # 加载测试结构
    structures = loadfn(structure_path)
    test_structure = structures[0]
    data = pmg_structure_to_pyg_data(test_structure)
    batch = CrystalBatch.from_data_list([data])
    batch = batch.to(device)

    # 准备输入描述
    input_description = {
        "input_source": "encoded_from_structure",
        "structure_info": str(test_structure.composition),
        "description": "通过VAE编码器编码后采样得到的潜在向量z",
    }

    # 完整的编码-解码流程
    with torch.no_grad():
        # Encode
        encoded = vae.encoder(batch)  # Dict with keys: x, num_atoms, batch, token_idx
        encoded["moments"] = vae.quant_conv(encoded["x"])

        # Use DiagonalGaussianDistribution to sample
        from src.vae_module.vae_module import DiagonalGaussianDistribution
        encoded["posterior"] = DiagonalGaussianDistribution(encoded["moments"])
        z = encoded["posterior"].sample()
        encoded["x"] = z
        encoded["z"] = z

        # 添加输入张量信息
        input_description["input_tensor"] = save_tensor_to_dict(z, "")

        print(f"\nInput latent z:")
        print_tensor_info(z, "Decoder input")

        # Decode
        z_decoder = vae.post_quant_conv(z)
        encoded["x"] = z_decoder
        decoder_out = vae.decoder(encoded)

        # 输出重建结果
        print(f"\nDecoder output:")
        print(f"  keys: {decoder_out.keys()}")

        for key in ["atom_types", "frac_coords", "lengths", "angles"]:
            if key in decoder_out:
                print_tensor_info(decoder_out[key], f"Reconstructed {key}")

    # 保存为字典（JSON格式），同时保存完整数据
    output_dict = {
        "decoder_input_z": save_tensor_to_dict(z, "", output_dir, "vae_decoder_input_z", True),
        "decoder_post_quant": save_tensor_to_dict(z_decoder, "", output_dir, "vae_decoder_post_quant", True),
    }

    for key in ["atom_types", "frac_coords", "lengths", "angles"]:
        if key in decoder_out:
            output_dict[f"reconstructed_{key}"] = save_tensor_to_dict(
                decoder_out[key], "", output_dir, f"vae_decoder_{key}", True
            )

    result = {
        "module_name": "vae_decoder",
        "description": "VAE解码器：输入潜在向量z，输出重建的晶体结构",
        "input": input_description,
        "output": output_dict,
    }

    return result


def test_vae_full(vae_path, structure_path, device="cuda", output_dir=None):
    """测试完整的 VAE 编码-解码"""
    # 设置随机种子确保可复现
    set_seed(42)

    print("\n" + "=" * 80)
    print("VAE Full Test (Encode + Decode)")
    print("=" * 80)

    # 加载模型
    vae = VAEModule.load_from_checkpoint(vae_path, weights_only=False)
    vae.to(device)
    vae.eval()

    # 加载测试结构
    structures = loadfn(structure_path)
    test_structure = structures[0]
    data = pmg_structure_to_pyg_data(test_structure)
    batch = CrystalBatch.from_data_list([data])
    batch = batch.to(device)

    print(f"\nOriginal structure:")
    print(f"  composition: {test_structure.composition}")
    print(f"  num_atoms: {len(test_structure)}")
    print(f"  lattice: {test_structure.lattice.abc}")

    with torch.no_grad():
        # Forward pass
        decoder_out, encoded = vae(batch)

        # 获取均值和方差
        mean = encoded["posterior"].mean
        logvar = encoded["posterior"].logvar

        print(f"\nReconstruction error:")
        # 计算重建误差
        if hasattr(batch, 'frac_coords') and "frac_coords" in decoder_out:
            coord_error = torch.abs(batch.frac_coords - decoder_out["frac_coords"]).mean()
            print(f"  coord MAE: {coord_error.item():.6f}")

        if hasattr(batch, 'lengths') and "lengths" in decoder_out:
            lattice_error = torch.abs(batch.lengths_scaled - decoder_out["lengths"]).mean()
            print(f"  lattice MAE: {lattice_error.item():.6f}")

    # 保存为字典（JSON格式）
    input_description = {
        "structure_composition": str(test_structure.composition),
        "num_atoms": len(test_structure),
        "lattice_abc": list(test_structure.lattice.abc),
        "lattice_angles": list(test_structure.lattice.angles),
        "space_group": test_structure.get_space_group_info()[0],
    }

    metrics = {}
    if hasattr(batch, 'frac_coords') and "frac_coords" in decoder_out:
        metrics["coord_mae"] = float(f"{coord_error.item():.8f}")
    if hasattr(batch, 'lengths') and "lengths" in decoder_out:
        metrics["lattice_mae"] = float(f"{lattice_error.item():.8f}")

    result = {
        "module_name": "vae_full",
        "description": "完整VAE：编码-采样-解码，计算重建误差",
        "input": input_description,
        "output": {
            "latent_mean": save_tensor_to_dict(mean, "", output_dir, "vae_full_latent_mean", True),
            "latent_logvar": save_tensor_to_dict(logvar, "", output_dir, "vae_full_latent_logvar", True),
            "reconstruction_metrics": metrics,
        },
    }

    return result


def test_ldm_denoiser(ldm_path, vae_path, device="cuda", fixed_input=None, output_dir=None):
    """测试 LDM Denoiser

    Args:
        fixed_input: dict with keys 'z' and 't' for fixed input
        output_dir: 输出目录，用于保存完整tensor数据
    """
    # 设置随机种子确保可复现
    set_seed(42)

    print("\n" + "=" * 80)
    print("LDM Denoiser Test")
    print("=" * 80)

    # 加载模型
    ldm = LDMModule.load_from_checkpoint(ldm_path, vae_ckpt_path=vae_path, weights_only=False)
    ldm.to(device)
    ldm.eval()

    # 创建固定的测试输入（不使用随机）
    batch_size = 2
    num_atoms = 10
    latent_dim = 8

    # 固定的输入数据（基于seed=42生成的固定值）
    z = torch.randn(batch_size, num_atoms, latent_dim, device=device)
    # 固定时间步
    t = torch.tensor([13, 25], device=device, dtype=torch.long)

    print_tensor_info(z, "Input Latent Z")
    print(f"\nTimesteps: {t.cpu().numpy()}")

    # 准备详细的输入描述，保存完整数据
    input_description = {
        "description": "固定的噪声潜在向量z和扩散时间步t",
        "generation_method": "torch.randn with seed=42",
        "input_z": save_tensor_to_dict(z, "", output_dir, "ldm_denoiser_input_z", True),
        "input_t": {
            "values": t.cpu().tolist(),
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "description": "扩散过程的时间步，范围[0, 1000)",
        },
        "batch_size": batch_size,
        "num_atoms_per_structure": num_atoms,
        "latent_dim": latent_dim,
    }

    # Denoiser 前向传播
    with torch.no_grad():
        # 准备 mask (布尔类型)
        mask = torch.ones(batch_size, num_atoms, dtype=torch.bool, device=device)

        # 调用 denoiser
        noise_pred = ldm.denoiser(z, t, mask=mask)
        print_tensor_info(noise_pred, "Denoiser Output (Predicted Noise)")

    # 保存为字典（JSON格式），保存完整数据
    result = {
        "module_name": "ldm_denoiser",
        "description": "LDM去噪器：输入噪声潜在向量z和时间步t，预测噪声",
        "input": input_description,
        "output": {
            "predicted_noise": save_tensor_to_dict(
                noise_pred, "", output_dir, "ldm_denoiser_output_noise", True
            ),
        },
    }

    return result


def test_ldm_sampling(ldm_path, vae_path, structure_path, device="cuda", output_dir=None):
    """测试 LDM 完整采样流程

    Args:
        output_dir: 输出目录，用于保存完整tensor数据
    """
    # 设置随机种子确保可复现
    set_seed(42)

    print("\n" + "=" * 80)
    print("LDM Sampling Test")
    print("=" * 80)

    # 加载模型
    ldm = LDMModule.load_from_checkpoint(ldm_path, vae_ckpt_path=vae_path, weights_only=False)
    ldm.to(device)
    ldm.eval()

    # 加载参考结构（用于确定原子数分布）
    structures = loadfn(structure_path)
    ref_structure = structures[0]

    print(f"\nReference structure: {ref_structure.composition}")
    print(f"  num_atoms: {len(ref_structure)}")

    # 创建 batch（用于采样）
    data = pmg_structure_to_pyg_data(ref_structure)
    batch = CrystalBatch.from_data_list([data])
    batch = batch.to(device)

    # 准备输入描述
    input_description = {
        "reference_structure": str(ref_structure.composition),
        "num_atoms": len(ref_structure),
        "sampling_config": {
            "sampler": "ddim",
            "sampling_steps": 10,
            "cfg_scale": 2.0,
        },
        "description": "从纯噪声开始，通过迭代去噪生成新的晶体结构",
    }

    # 采样
    with torch.no_grad():
        # 使用较少的采样步数加快测试
        batch_gen = ldm.sample(
            batch,
            sampler="ddim",
            sampling_steps=10,
            cfg_scale=2.0,
        )

        print(f"\nGenerated structures:")
        print(f"  batch_size: {batch_gen.num_graphs}")
        if hasattr(batch_gen, 'frac_coords'):
            print_tensor_info(batch_gen.frac_coords, "Generated Coords")
        if hasattr(batch_gen, 'lattices'):
            print_tensor_info(batch_gen.lattices, "Generated Lattices")

        # 解码为实际结构
        try:
            gen_structures = batch_gen.to_structure()  # 使用 to_structure() 而不是 to_structures()
            print(f"\nDecoded {len(gen_structures)} structures")
            for i, struct in enumerate(gen_structures[:3]):  # 显示前3个
                print(f"  [{i}] {struct.composition}")
        except Exception as e:
            print(f"\nDecode failed: {e}")

    # 保存为字典（JSON格式），保存完整数据
    output_dict = {}
    if hasattr(batch_gen, 'frac_coords'):
        output_dict["generated_frac_coords"] = save_tensor_to_dict(
            batch_gen.frac_coords, "", output_dir, "ldm_sampling_coords", True
        )
    if hasattr(batch_gen, 'lattices'):
        output_dict["generated_lattices"] = save_tensor_to_dict(
            batch_gen.lattices, "", output_dir, "ldm_sampling_lattices", True
        )

    # 添加生成结构的组成信息
    if 'gen_structures' in locals():
        output_dict["generated_structures"] = [
            {"composition": str(s.composition)} for s in gen_structures[:5]
        ]

    result = {
        "module_name": "ldm_sampling",
        "description": "LDM完整采样：从噪声生成完整的晶体结构",
        "input": input_description,
        "output": output_dict,
    }

    return result


def main():
    """主函数"""
    # 先创建输出目录
    output_dir = Path("outputs/precision_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志输出
    log_file = output_dir / "precision_test_log.txt"
    set_logger(log_file)

    print("\n" + "* " * 40)
    print(" " * 25 + "Chemeleon2 Precision Test")
    print("* " * 40)

    # 设置随机种子
    set_seed(42)
    print(f"\nRandom seed: 42")
    print(f"Log file: {log_file}")

    # 模型路径
    vae_path = "checkpoints/v0.0.1/alex_mp_20/vae/dng_j1jgz9t0_v1.ckpt"
    ldm_path = "checkpoints/v0.0.1/alex_mp_20/ldm/ldm_rl_dng_tuor5vgd.ckpt"
    structure_path = "outputs/alex-mp-20/generated_structures.json.gz"

    # 如果测试结构不存在，使用备用路径
    if not Path(structure_path).exists():
        structure_path = "outputs/test_three/generated_structures.json.gz"

    if not Path(structure_path).exists():
        structure_path = "benchmarks/dng/chemeleon2_rl_dng_mp_20.json.gz"

    if not Path(structure_path).exists():
        print(f"\nError: Structure file not found: {structure_path}")
        print("   Please provide a valid structure file")
        close_logger()
        return False

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 保存所有结果
    all_results = {}

    try:
        # 1. 测试 VAE Encoder
        results = test_vae_encoder(vae_path, structure_path, device, output_dir=output_dir)
        all_results["vae_encoder"] = results

        # 2. 测试 VAE Decoder
        results = test_vae_decoder(vae_path, structure_path, device, output_dir=output_dir)
        all_results["vae_decoder"] = results

        # 3. 测试完整 VAE
        results = test_vae_full(vae_path, structure_path, device, output_dir=output_dir)
        all_results["vae_full"] = results

        # 4. 测试 LDM Denoiser
        results = test_ldm_denoiser(ldm_path, vae_path, device, output_dir=output_dir)
        all_results["ldm_denoiser"] = results

        # 5. 测试 LDM 采样
        results = test_ldm_sampling(ldm_path, vae_path, structure_path, device, output_dir=output_dir)
        all_results["ldm_sampling"] = results

        # 保存结果为JSON文件
        output_json = output_dir / "pytorch_results.json"
        output_txt = output_dir / "pytorch_results_readable.txt"  # 也保存易读的文本版本

        # 构建完整的JSON结构
        full_results = {
            "test_info": {
                "framework": "PyTorch",
                "test_name": "Chemeleon2 精度测试",
                "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "random_seed": 42,
                "device": device,
                "description": "用于框架迁移的精度对比测试，包含VAE和LDM的各个模块",
            },
            "model_info": {
                "vae_checkpoint": vae_path,
                "ldm_checkpoint": ldm_path,
                "structure_file": structure_path,
            },
            "modules": all_results,
        }

        # 保存JSON文件
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)

        # 保存易读的文本版本
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Chemeleon2 精度测试结果 - PyTorch (易读版本)\n")
            f.write("=" * 80 + "\n")
            f.write(f"测试时间: {full_results['test_info']['test_time']}\n")
            f.write(f"随机种子: {full_results['test_info']['random_seed']}\n")
            f.write(f"设备: {full_results['test_info']['device']}\n")
            f.write("=" * 80 + "\n\n")

            for module_name, module_data in all_results.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"模块: {module_data.get('module_name', module_name)}\n")
                f.write(f"描述: {module_data.get('description', '')}\n")
                f.write('=' * 80 + "\n\n")

                # 写入输入信息
                if 'input' in module_data:
                    f.write("输入信息:\n")
                    f.write(json.dumps(module_data['input'], indent=2, ensure_ascii=False))
                    f.write("\n\n")

                # 写入输出信息
                if 'output' in module_data:
                    f.write("输出信息:\n")
                    f.write(json.dumps(module_data['output'], indent=2, ensure_ascii=False))
                    f.write("\n")

        print("\n" + "=" * 80)
        print("Done!")
        print("=" * 80)
        print(f"\nJSON saved to: {output_json}")
        print(f"Text saved to: {output_txt}")
        print(f"Log saved to: {output_dir}/precision_test_log.txt")
        print("\nYou can compare these results with other frameworks (e.g. Paddle)")
        print(f"\nComparison methods:")
        print(f"  1. JSON format (recommended):")
        print(f"     python compare_precision.py {output_json} paddle_results.json")
        print(f"  2. Text format:")
        print(f"     diff {output_txt} paddle_results_readable.txt")

        close_logger()
        return True

    except Exception as e:
        print(f"\nError: Test failed - {str(e)}")
        import traceback
        traceback.print_exc()
        close_logger()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

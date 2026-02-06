#!/usr/bin/env python3
"""
VAE 和 LDM 模型的集成验证脚本。

"""

import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'chemeleon2'))

import paddle
import numpy as np
import json

from ppmat.models.chemeleon2.vae_module.vae import VAEModule
from ppmat.models.chemeleon2.vae_module.encoder import TransformerEncoder
from ppmat.models.chemeleon2.vae_module.decoder import TransformerDecoder
from ppmat.models.chemeleon2.ldm_module.dit import DiT
from ppmat.models.chemeleon2.common.schema import CrystalBatch

import os
HOME_DIR = Path(os.path.expanduser("~"))

DEFAULT_WEIGHT_DIR = str(HOME_DIR / '.paddlemat/models/chemeleon2')
DEFAULT_OUTPUT_DIR = str(parent_dir) + "/outputs/"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR + 'chemeleon2_diff_lose.json'
DEFAULT_PRECISION_THRESHOLD = 1e-5


# VAE 编码器输入数据
INPUT_ATOM_TYPES = np.array(
    [66, 67, 68, 68, 68, 69, 52, 52, 52, 33, 15, 15]
)

INPUT_FRAC_COORDS = np.array(
    [[0.16941863, 0.25193024, 0.45317790],
     [0.49152774, 0.45137846, 0.11796811],
     [0.48285794, 0.91339254, 0.14510530],
     [0.82677758, 0.59025151, 0.79817736],
     [0.81833100, 0.11027566, 0.76181233],
     [0.15705948, 0.77731586, 0.42618808],
     [0.33342591, 0.85136592, 0.77985525],
     [0.65449405, 0.51935136, 0.45445091],
     [0.98995161, 0.17083004, 0.11222821],
     [0.33672857, 0.33988437, 0.77312100],
     [0.66850150, 0.00315657, 0.45171800],
     [0.99026763, 0.65868866, 0.12688932]]
)

INPUT_LATTICES = np.array(
    [[[4.07097721, 0.0, -0.91007066],
      [-0.46107024, 8.27997398, -2.12557125],
      [0.0, 0.0, 9.44053841]]]
)

INPUT_NUM_ATOMS = np.array([12])

INPUT_LENGTHS = np.array(
    [[4.17146063, 8.56087685, 9.44053841]]
)

INPUT_ANGLES = np.array(
    [[104.37628174, 102.60133362, 89.90788269]]
)

# VAE 解码器输入数据（潜在变量 z）
INPUT_Z = np.array(
    [[1.96778619, 0.52110022, -0.37526771, -1.06078792, -0.85598081, 1.11346722, 0.22077970, 0.96572930],
     [0.64574736, -0.43119499, 0.03418273, -0.00645405, 0.89249128, 0.37275982, 2.41711688, 0.11571212],
     [0.42960018, -0.43622142, 1.44103599, 0.11944307, -1.84233403, -0.07848608, 1.15001416, -0.54063123],
     [0.91394019, 1.63482583, 0.57408303, 1.07989991, -2.48861003, 1.02283752, 1.02458203, 0.99812806],
     [0.66510350, 1.63272715, -0.96877307, 0.91754431, -2.04141307, 2.00096869, 0.64047354, 0.05152281],
     [2.28735399, 0.36919677, 1.11381781, -1.00699818, 0.64713156, 0.05649607, -0.62192798, 0.97391701],
     [-1.52237606, 1.39929760, 1.43075931, -0.46554267, -0.68277979, -0.56380445, 1.09997213, 0.73207515],
     [-1.66961420, 0.57721722, 0.25194433, 0.39816928, -0.65705436, -0.12071487, 0.88596821, 1.20686293],
     [-0.70767230, -0.25845712, -1.09743047, 1.77922177, -0.84514201, 2.08224416, 0.54471433, -0.14093801],
     [-1.00094223, 1.52224243, -0.10302663, -0.58021611, 1.42938161, -0.58981705, -1.16647685, 0.97821933],
     [-0.58995736, 0.75216556, -1.43728018, 0.37418503, -0.30776960, 2.80792618, -0.79698497, -1.00396240],
     [-0.84623051, -0.31481862, 0.54803443, 1.99706137, -0.12883027, 1.18385315, -0.81202215, -0.88749123]]
)

# LDM 去噪器输入数据（扩散潜在变量 z）
INPUT_LDM_Z = np.array(
    [[[0.19401880, 2.16137362, -0.17205022, 0.84906012, -1.92439902, 0.65298551, -0.64944082, -0.81752473],
      [0.52796447, -1.27534986, -1.66212630, -0.30331373, -0.09256987, 0.19923715, -1.12043285, 1.85765862],
      [-0.71451885, 0.68810511, 0.79683083, -0.03340188, 1.49173188, -0.51650929, -0.25409597, 1.47461557],
      [-0.32603732, -1.15996265, 2.35513091, -0.69244707, 0.18374282, -1.18350995, -1.80286717, -1.58075690],
      [0.83866954, 1.41918027, 0.64693671, 0.42527241, -1.58924079, 0.62234497, 1.68980360, -0.66480386],
      [0.94254267, 0.07832550, 0.08465634, -0.14082992, 0.33156055, -0.58897614, -1.07228947, 0.09539576],
      [-0.33469191, -0.52579743, -0.87762552, 0.39383137, 0.16395937, -0.19768225, 1.01041365, -1.34824479],
      [-0.34977224, -0.64426798, 0.44678342, -0.53710973, 1.24231851, -0.81459534, 0.25015041, -0.42725861],
      [1.10436928, -1.10279870, 0.55432665, -1.28465545, -0.38157833, 0.51394576, 0.10019008, 0.25862604],
      [0.36168072, 2.27866960, 0.02334509, 1.58275771, -1.15917921, 0.94839233, -0.45734766, 0.76054770]],
     [[-0.57868302, -0.70502084, -0.72338772, -0.50706196, -0.43984994, -0.41817018, 0.17413868, 0.44268036],
      [0.50689828, -1.21680868, -0.27187300, 0.27654943, -1.43981659, -0.64632124, 0.07486922, 0.19387875],
      [0.59601170, 0.23220330, 1.14146543, -0.68170702, -1.65314484, 0.00603564, 1.38148701, 1.27042663],
      [0.02323810, -1.30014515, -0.75094134, 0.37562433, -0.54744226, -0.03964127, -0.77786469, -2.50188589],
      [0.70001656, -0.09377469, -0.21625695, 0.44839421, -0.31519616, 0.02163736, 0.62534708, 0.24658130],
      [0.74856061, -0.11692451, -0.10216469, -0.50108075, -0.50488758, -1.20719242, -0.24375997, -0.67842638],
      [0.19728611, 0.97822028, -0.02866771, 1.68258953, 1.09085572, -0.99209458, -0.67126238, 1.71963453],
      [2.46055436, -0.61983937, 1.27138603, -0.27986664, 0.43596676, 0.42602390, 1.06455433, -2.02799630],
      [-0.63258272, 2.11064816, -0.09474602, 0.23587526, -0.73007232, -1.68571997, 0.91141981, 0.78854555],
      [-0.62873089, 2.15955496, 1.16424942, -0.42566288, 0.23932022, -1.27767038, -0.12064690, -0.60658425]]]
)


# ============================================================================
# 硬编码的参考输出数据
# 这些数据是从原版项目中通过固定随机数种子 (seed=42) 推理计算得出的
# 具体的复现代码参考当前目录的 README.md
# 输出的数据进行了裁剪，每个维度如果大于4维，则最多只取4维，避免太多数据

REFERENCE_OUTPUTS = {
    'vae_encoder_output': np.array([
        [-0.17990004, -0.04645369, -0.29342285, -0.05779608],
        [-2.15766120, 0.05053813, 0.05233471, -0.08952494],
        [0.32678959, 0.10300802, 0.28084213, -0.10772780],
        [0.75947309, 0.17603433, 0.14910318, -0.12478722]
    ]),
    'vae_encoder_quant_conv': np.array([
        [1.80965030, 0.52404141, -0.37543607, -1.06138432],
        [0.62111264, -0.43615344, 0.03323286, -0.00297468],
        [0.57471651, -0.43998951, 1.44004822, 0.11691619],
        [0.44248250, 1.63856602, 0.57267267, 1.08317006]
    ]),
    'vae_encoder_latent_mean': np.array([
        [1.80965030, 0.52404141, -0.37543607, -1.06138432],
        [0.62111264, -0.43615344, 0.03323286, -0.00297468],
        [0.57471651, -0.43998951, 1.44004822, 0.11691619],
        [0.44248250, 1.63856602, 0.57267267, 1.08317006]
    ]),
    'vae_encoder_latent_logvar': np.array([
        [-3.96147633, -11.50236607, -11.20673370, -10.69168186],
        [-3.86336756, -10.93966484, -11.33978176, -11.27313232],
        [-2.90164447, -10.99563599, -10.80719090, -11.23406792],
        [-2.86538887, -10.86329651, -11.36846542, -10.63086319]
    ]),
    'vae_decoder_atom_types': np.array([
        [-9.97118473, -12.22323322, -9.96976662, -8.63844585],
        [-10.63955688, -17.72699165, -10.35566330, -10.19803047],
        [-9.42540455, -1.47579587, -9.59889793, -11.58701992],
        [-9.40364838, -2.24563265, -9.89963722, -12.12908268]
    ]),
    'vae_decoder_frac_coords': np.array([
        [0.17350805, 0.25384152, 0.45326898],
        [0.49485329, 0.45135695, 0.12035841],
        [0.48613694, 0.91251493, 0.14744006],
        [0.82910115, 0.58979881, 0.79950404]
    ]),
    'vae_decoder_lengths': np.array([[1.81244886, 3.73297477, 4.11331844]]),
    'vae_decoder_angles': np.array([[1.82423151, 1.79026175, 1.56918395]]),
    'ldm_denoiser_output_noise': np.array([
        [[-0.38310128, 1.03488433, 0.64289629, 0.23227391],
         [-0.52038497, -0.62339407, -0.72368735, 0.30327153],
         [-0.47714779, 0.58150470, 0.24278848, 0.27084529],
         [0.22972283, 0.21897136, 0.25752124, -0.38142803]],
        [[0.02938975, -0.40452722, -0.57233673, 1.40531123],
         [-0.13082200, -1.04666209, -0.68292052, 0.29126278],
         [0.15657395, -0.48006776, 0.57248652, -0.19115604],
         [0.08723904, -1.17863417, 0.76299840, -0.34876239]]
    ]),
}


def get_first_n_elements(arr, max_dims=(4, 4, 4, 4)):
    """根据维度从数组中提取前 N 个元素。"""
    slice_shape = tuple(min(d, s) for d, s in zip(max_dims, arr.shape))
    slices = tuple(slice(0, s) for s in slice_shape)
    return arr[slices]



def run_test_vae_encoder(reference_data, weight_dir, precision_threshold):
    """测试 VAE 编码器并与参考数据进行比较。"""
    print("\n" + "="*80)
    print("Testing VAE Encoder")
    print("="*80)

    # 初始化模型
    encoder = TransformerEncoder(
        max_num_elements=100,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        activation='gelu',
        dropout=0.0,
        num_layers=8
    )

    decoder = TransformerDecoder(
        max_num_elements=100,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        activation='gelu',
        dropout=0.0,
        num_layers=8
    )

    vae = VAEModule(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        loss_weights={}
    )

    # 加载权重
    weight_path = Path(weight_dir) / 'vae_paddle.pdparams'
    print(f"Loading weights from: {weight_path}")
    full_state_dict = paddle.load(str(weight_path))
    vae.set_state_dict(full_state_dict)
    vae.eval()

    # 使用硬编码数据创建 batch
    batch = CrystalBatch()
    batch.atom_types = paddle.to_tensor(INPUT_ATOM_TYPES, dtype='int64')
    batch.frac_coords = paddle.to_tensor(INPUT_FRAC_COORDS, dtype='float32')
    batch.lattices = paddle.to_tensor(INPUT_LATTICES, dtype='float32')
    batch.num_atoms = paddle.to_tensor(INPUT_NUM_ATOMS, dtype='int64')
    batch.batch = paddle.zeros([12], dtype='int64')
    batch.token_idx = paddle.arange(12, dtype='int64')
    batch.num_graphs = 1

    # 运行推理
    with paddle.no_grad():
        encoded = vae.encode(batch)
        encoder_output = encoded['x']
        quant_conv_output = encoded['moments']
        latent_mean = encoded['posterior'].mean
        latent_logvar = encoded['posterior'].logvar
        latent_z = encoded['posterior'].sample()

    # 收集输出以进行比较
    outputs = {
        'vae_encoder_output': encoder_output.numpy(),
        'vae_encoder_quant_conv': quant_conv_output.numpy(),
        'vae_encoder_latent_mean': latent_mean.numpy(),
        'vae_encoder_latent_logvar': latent_logvar.numpy(),
    }

    # 与参考数据比较
    results = []
    all_passed = True

    for key, output in outputs.items():
        if key not in reference_data:
            print(f"WARNING {key}: No reference data found")
            continue

        ref_array = reference_data[key]
        output_slice = get_first_n_elements(output)

        # 展平以进行比较
        ref_flat = ref_array.flatten()
        output_flat = output_slice.flatten()

        min_len = min(len(ref_flat), len(output_flat))
        ref_flat = ref_flat[:min_len]
        output_flat = output_flat[:min_len]

        diff = np.abs(ref_flat - output_flat)
        max_diff = diff.max()
        mean_diff = diff.mean()

        passed = max_diff < precision_threshold

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"{status} {key}:")
        print(f"   Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        print(f"   Shape: {output.shape}")

        if not passed:
            max_idx = diff.argmax()
            print(f"   Ref value: {ref_flat[max_idx]:.8f}, Output value: {output_flat[max_idx]:.8f}")

        results.append({
            'name': key,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'passed': bool(passed)
        })

    return all_passed, results


def run_test_vae_decoder(reference_data, weight_dir, precision_threshold):
    """测试 VAE 解码器并与参考数据进行比较。"""
    print("\n" + "="*80)
    print("Testing VAE Decoder")
    print("="*80)

    # 初始化模型
    encoder = TransformerEncoder(
        max_num_elements=100,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        activation='gelu',
        dropout=0.0,
        num_layers=8
    )

    decoder = TransformerDecoder(
        max_num_elements=100,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        activation='gelu',
        dropout=0.0,
        num_layers=8
    )

    vae = VAEModule(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        loss_weights={}
    )

    # 加载权重
    weight_path = Path(weight_dir) / 'vae_paddle.pdparams'
    print(f"Loading weights from: {weight_path}")
    full_state_dict = paddle.load(str(weight_path))
    vae.set_state_dict(full_state_dict)
    vae.eval()

    z_t = paddle.to_tensor(INPUT_Z, dtype='float32')
    batch = CrystalBatch()
    batch.atom_types = paddle.zeros([12], dtype='int64')
    batch.frac_coords = paddle.zeros([12, 3], dtype='float32')
    batch.lattices = paddle.zeros([1, 3, 3], dtype='float32')
    batch.num_atoms = paddle.to_tensor(INPUT_NUM_ATOMS, dtype='int64')
    batch.batch = paddle.zeros([12], dtype='int64')
    batch.token_idx = paddle.arange(12, dtype='int64')
    batch.num_graphs = 1

    # 运行推理
    with paddle.no_grad():
        encoded = {
            'x': z_t,
            'z': z_t,
            'batch': batch.batch,
            'token_idx': batch.token_idx,
            'num_atoms': batch.num_atoms
        }
        decoded = vae.decode(encoded)
        atom_types = decoded['atom_types']
        frac_coords = decoded['frac_coords']
        lengths = decoded['lengths']
        angles = decoded['angles']

    # 收集输出以进行比较
    outputs = {
        'vae_decoder_atom_types': atom_types.numpy(),
        'vae_decoder_frac_coords': frac_coords.numpy(),
        'vae_decoder_lengths': lengths.numpy(),
        'vae_decoder_angles': angles.numpy(),
    }

    # 与参考数据比较
    results = []
    all_passed = True

    for key, output in outputs.items():
        if key not in reference_data:
            print(f"WARNING {key}: No reference data found")
            continue

        ref_array = reference_data[key]
        output_slice = get_first_n_elements(output)

        # 展平以进行比较
        ref_flat = ref_array.flatten()
        output_flat = output_slice.flatten()

        min_len = min(len(ref_flat), len(output_flat))
        ref_flat = ref_flat[:min_len]
        output_flat = output_flat[:min_len]

        diff = np.abs(ref_flat - output_flat)
        max_diff = diff.max()
        mean_diff = diff.mean()

        passed = max_diff < precision_threshold

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"{status} {key}:")
        print(f"   Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        print(f"   Shape: {output.shape}")

        if not passed:
            max_idx = diff.argmax()
            print(f"   Ref value: {ref_flat[max_idx]:.8f}, Output value: {output_flat[max_idx]:.8f}")

        results.append({
            'name': key,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'passed': bool(passed)
        })

    return all_passed, results


def run_test_ldm_denoiser(reference_data, weight_dir, precision_threshold):
    """测试 LDM 去噪器并与参考数据进行比较。"""
    print("\n" + "="*80)
    print("Testing LDM Denoiser")
    print("="*80)

    # 初始化模型
    denoiser = DiT(
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        input_dim=8,
        learn_sigma=True
    )

    # 加载权重
    weight_path = Path(weight_dir) / 'ldm_paddle.pdparams'
    print(f"Loading weights from: {weight_path}")
    full_state_dict = paddle.load(str(weight_path))

    denoiser_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith('denoiser.'):
            new_key = key.replace('denoiser.', '')
            denoiser_state_dict[new_key] = value

    denoiser.set_state_dict(denoiser_state_dict)
    denoiser.eval()

    z_t = paddle.to_tensor(INPUT_LDM_Z, dtype='float32')
    t = paddle.to_tensor([13, 25], dtype='int64')
    batch_size, num_atoms, _ = INPUT_LDM_Z.shape
    mask = paddle.ones([batch_size, num_atoms], dtype='bool')

    # 运行推理
    with paddle.no_grad():
        noise_pred = denoiser(z_t, t, mask=mask)

    # 收集输出以进行比较
    outputs = {
        'ldm_denoiser_output_noise': noise_pred.numpy(),
    }

    # 与参考数据比较
    results = []
    all_passed = True

    for key, output in outputs.items():
        if key not in reference_data:
            print(f"WARNING {key}: No reference data found")
            continue

        ref_array = reference_data[key]
        output_slice = get_first_n_elements(output)

        # 展平以进行比较
        ref_flat = ref_array.flatten()
        output_flat = output_slice.flatten()

        min_len = min(len(ref_flat), len(output_flat))
        ref_flat = ref_flat[:min_len]
        output_flat = output_flat[:min_len]

        diff = np.abs(ref_flat - output_flat)
        max_diff = diff.max()
        mean_diff = diff.mean()

        passed = max_diff < precision_threshold

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"{status} {key}:")
        print(f"   Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        print(f"   Shape: {output.shape}")

        if not passed:
            max_idx = diff.argmax()
            print(f"   Ref value: {ref_flat[max_idx]:.8f}, Output value: {output_flat[max_idx]:.8f}")

        results.append({
            'name': key,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'passed': bool(passed)
        })

    return all_passed, results


def main(weight_dir=None, output_file=None, precision_threshold=None, save_result=True):
    """
    运行所有验证测试的主函数。

    参数:
        weight_dir: 包含转换权重的目录（默认: DEFAULT_WEIGHT_DIR）
        output_file: 结果的输出 JSON 文件路径（默认: DEFAULT_OUTPUT_FILE）
        precision_threshold: 比较的精度阈值（默认: DEFAULT_PRECISION_THRESHOLD）
        save_result: 是否保存结果到文件（默认: True）

    返回:
        int: 如果所有测试通过则返回 0，否则返回 1
    """
    # 如果未提供则使用默认值
    if weight_dir is None:
        weight_dir = DEFAULT_WEIGHT_DIR
    if output_file is None:
        output_file = DEFAULT_OUTPUT_FILE
    if precision_threshold is None:
        precision_threshold = DEFAULT_PRECISION_THRESHOLD

    # 转换为 Path 对象
    weight_dir = Path(weight_dir)
    output_file = Path(output_file)

    print("="*80)
    print("Integrated VAE & LDM Validation (Hardcoded Input & Reference Data)")
    print(f"Precision Threshold: {precision_threshold:.1e}")
    print(f"Weight Directory: {weight_dir}")
    print(f"Output File: {output_file}")
    print("="*80)

    # 使用硬编码的参考数据
    print("\n使用硬编码的参考输出数据（来自原版项目，seed=42）")
    reference_data = REFERENCE_OUTPUTS
    print(f"Loaded {len(reference_data)} reference outputs")

    # 运行所有测试
    all_results = []
    global_passed = True

    # 测试 VAE 编码器
    passed, results = run_test_vae_encoder(reference_data, weight_dir, precision_threshold)
    all_results.extend(results)
    global_passed = global_passed and passed

    # 测试 VAE 解码器
    passed, results = run_test_vae_decoder(reference_data, weight_dir, precision_threshold)
    all_results.extend(results)
    global_passed = global_passed and passed

    # 测试 LDM 去噪器
    passed, results = run_test_ldm_denoiser(reference_data, weight_dir, precision_threshold)
    all_results.extend(results)
    global_passed = global_passed and passed

    # 打印摘要
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed_count = sum(1 for r in all_results if r['passed'])
    total_count = len(all_results)

    print(f"Passed: {passed_count}/{total_count}")
    print(f"Threshold: {precision_threshold:.1e}")

    if global_passed:
        print("ALL TESTS PASSED")
    else:
        print("FAILED TESTS:")
        for r in all_results:
            if not r['passed']:
                print(f"   - {r['name']}: Max diff = {r['max_diff']:.6e}")

    if save_result:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'threshold': precision_threshold,
                'global_passed': global_passed,
                'passed_count': passed_count,
                'total_count': total_count,
                'results': all_results
            }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("="*80)

    return 0 if global_passed else 1

# 这里硬编码的矩阵，实际是从 原版的项目中，通过固定42随机数，推理计算出来，具体的复现代码参考 当前目录的README.md
def test_with_fix_random(weight_dir=None, output_file=None, precision_threshold=None):
    """
    固定随机种子的主测试入口点。

    参数:
        weight_dir: 包含转换权重的目录（默认: DEFAULT_WEIGHT_DIR）
        output_file: 结果的输出 JSON 文件路径（默认: DEFAULT_OUTPUT_FILE）
        precision_threshold: 比较的精度阈值（默认: DEFAULT_PRECISION_THRESHOLD）

    返回:
        int: 如果所有测试通过则返回 0，否则返回 1
    """
    paddle.seed(42)
    np.random.seed(42)

    exit_code = main(
        weight_dir=weight_dir,
        output_file=output_file,
        precision_threshold=precision_threshold
    )
    return exit_code


# 一般没有必要，直接从main开始执行，执行单元测试 test_with_fix_random 即可
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Integrated validation script for VAE and LDM models'
    )

    parser.add_argument(
        '--weight-dir',
        type=str,
        default=None,
        help='Directory containing converted weights'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output JSON file path for results'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Precision threshold'
    )

    args = parser.parse_args()

    sys.exit(
        test_with_fix_random(
            weight_dir=args.weight_dir,
            output_file=args.output_file,
            precision_threshold=args.threshold
        )
    )

"""
ORCA 量子化学计算示例

演示如何使用 ORCA 进行高熵合金的第一性原理计算（DFT）。
包括：基础计算、多尺度集成、结果分析等。

作者: 高熵合金自进化智能体项目组
日期: 2026-04-01
版本: v1.3.0
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppmatAgent.hea_crewai_agent.core.material import Alloy, Population
from ppmatAgent.hea_crewai_agent.core.evaluator import ThermodynamicEvaluator

# 检查 ORCA 是否可用
try:
    from ppmatAgent.hea_crewai_agent.core.orca_evaluator import (
        ORCAEvaluator,
        ORCACalculationConfig,
        ORCAInputGenerator,
        ORCARunner
    )
    ORCA_AVAILABLE = True
except ImportError:
    ORCA_AVAILABLE = False

import shutil


def check_orca_installation():
    """检查 ORCA 是否已安装"""
    orca_path = shutil.which('orca')
    if orca_path:
        print(f"✓ ORCA found at: {orca_path}")
        return True
    else:
        print("✗ ORCA not found in PATH")
        print("\nTo install ORCA:")
        print("1. Download from: https://orcaforum.kofo.mpg.de/")
        print("2. Free for academic use (registration required)")
        print("3. Add ORCA binary to PATH")
        return False


def example_1_basic_orca_calculation():
    """示例 1：基础 ORCA 计算"""
    print("\n" + "=" * 60)
    print("示例 1：基础 ORCA 第一性原理计算")
    print("=" * 60)

    if not ORCA_AVAILABLE:
        print("ERROR: orca_evaluator module not available")
        return

    if not check_orca_installation():
        print("\nSkipping ORCA calculation (ORCA not installed)")
        return

    # 创建测试合金
    alloy = Alloy(composition={
        'Co': 0.25,
        'Cr': 0.25,
        'Fe': 0.25,
        'Ni': 0.25
    })

    print(f"\n测试合金: {alloy.composition}")

    # 配置 ORCA 计算（快速设置）
    config = ORCACalculationConfig(
        method='PBE',  # 快速泛函
        basis_set='def2-SVP',  # 小基组
        calc_type='SP',  # 单点能（不优化几何）
        scf_convergence='NormalSCF',
        nprocs=4,
        maxcore=2000
    )

    # 创建评估器
    evaluator = ORCAEvaluator(
        config=config,
        work_dir='./orca_test',
        use_cluster_model=True,  # 使用团簇模型（13 原子）
        cache_results=True
    )

    print("\nORCA 计算设置:")
    print(f"  Method: {config.method}")
    print(f"  Basis Set: {config.basis_set}")
    print(f"  Calculation Type: {config.calc_type}")
    print(f"  Processors: {config.nprocs}")
    print(f"  Model: Cluster (13 atoms)")

    # 运行计算
    print("\n开始计算...")
    fitness = evaluator.evaluate(alloy, verbose=True)

    # 显示结果
    print("\n" + "-" * 60)
    print("计算结果:")
    print("-" * 60)
    print(f"  Status: {alloy.properties.get('orca_status', 'unknown')}")
    print(f"  Fitness Score: {fitness:.4f}")

    if alloy.properties.get('orca_status') == 'success':
        print(f"  Total Energy: {alloy.properties.get('orca_energy', 'N/A'):.6f} Hartree")
        print(f"  HOMO-LUMO Gap: {alloy.properties.get('orca_homo_lumo_gap', 'N/A'):.2f} eV")

        charges = alloy.properties.get('orca_mulliken_charges', {})
        if charges:
            print(f"  Mulliken Charges:")
            for elem, charge in charges.items():
                print(f"    {elem}: {charge:+.4f}")
    else:
        print(f"  Error: {alloy.properties.get('orca_error', 'Unknown error')}")


def example_2_input_file_generation():
    """示例 2：ORCA 输入文件生成"""
    print("\n" + "=" * 60)
    print("示例 2：ORCA 输入文件生成（不运行计算）")
    print("=" * 60)

    if not ORCA_AVAILABLE:
        print("ERROR: orca_evaluator module not available")
        return

    # 创建合金
    alloy = Alloy(composition={
        'Al': 0.2,
        'Ti': 0.2,
        'Cr': 0.2,
        'Fe': 0.2,
        'Ni': 0.2
    })

    print(f"\n合金成分: {alloy.composition}")

    # 配置
    config = ORCACalculationConfig(
        method='B3LYP',
        basis_set='def2-TZVP',
        calc_type='OPT',  # 几何优化
        scf_convergence='TightSCF',
        opt_convergence='TightOpt',
        nprocs=8,
        dispersion_correction=True
    )

    # 生成输入文件
    generator = ORCAInputGenerator(config)

    print("\n生成团簇模型输入文件...")
    cluster_input = generator.generate_input_file(alloy, use_cluster=True)

    output_file = 'AlTiCrFeNi_cluster.inp'
    with open(output_file, 'w') as f:
        f.write(cluster_input)

    print(f"✓ 输入文件已保存: {output_file}")
    print("\n文件内容预览:")
    print("-" * 60)
    print(cluster_input[:500])
    print("...")
    print("-" * 60)
    print(f"\n可以手动运行: orca {output_file}")


def example_3_multiscale_evaluation():
    """示例 3：四级多尺度评估（简化模型 → CALPHAD → MD → ORCA）"""
    print("\n" + "=" * 60)
    print("示例 3：四级多尺度评估框架")
    print("=" * 60)

    # 创建测试合金
    alloy = Alloy(composition={
        'Co': 0.25,
        'Cr': 0.25,
        'Fe': 0.25,
        'Ni': 0.25
    })

    print(f"\n测试合金: {alloy.composition}")
    print("\n评估流程:")

    # 第一级：简化模型（~1 ms）
    print("\n[1/4] 简化热力学模型评估...")
    simple_eval = ThermodynamicEvaluator()
    simple_score = simple_eval.evaluate(alloy)
    print(f"  ✓ 简化模型得分: {simple_score:.4f}")
    print(f"  计算时间: ~1 ms")

    if simple_score < 0.5:
        print("  → 得分过低，淘汰")
        return

    # 第二级：CALPHAD（~100 ms）
    print("\n[2/4] CALPHAD 热力学计算...")
    try:
        from ppmatAgent.hea_crewai_agent.core.calphad_evaluator import CALPHADEvaluator, CALPHAD_AVAILABLE

        if CALPHAD_AVAILABLE:
            print("  ✓ CALPHAD module available")
            print("  (需要 TDB 数据库，此处跳过实际计算)")
            calphad_score = 0.75  # 模拟得分
        else:
            print("  ✗ pycalphad not installed, skipping")
            calphad_score = simple_score
    except ImportError:
        print("  ✗ CALPHAD module not available")
        calphad_score = simple_score

    print(f"  模拟 CALPHAD 得分: {calphad_score:.4f}")
    print(f"  计算时间: ~100 ms")

    if calphad_score < 0.7:
        print("  → 未达到 MD 模拟阈值")
        return

    # 第三级：MD 模拟（~30 s）
    print("\n[3/4] 分子动力学模拟...")
    try:
        from ppmatAgent.hea_crewai_agent.core.md_evaluator import MDEvaluator, MD_AVAILABLE

        if MD_AVAILABLE:
            print("  ✓ MD module available")
            print("  (需要 LAMMPS 和势函数，此处跳过实际计算)")
            md_score = 0.82  # 模拟得分
        else:
            print("  ✗ ASE/LAMMPS not installed, skipping")
            md_score = calphad_score
    except ImportError:
        print("  ✗ MD module not available")
        md_score = calphad_score

    print(f"  模拟 MD 得分: {md_score:.4f}")
    print(f"  计算时间: ~30 s")

    if md_score < 0.8:
        print("  → 未达到 ORCA 计算阈值")
        return

    # 第四级：ORCA 第一性原理（~分钟到小时）
    print("\n[4/4] ORCA 第一性原理计算...")

    if not ORCA_AVAILABLE:
        print("  ✗ ORCA module not available")
        orca_score = md_score
    elif not check_orca_installation():
        print("  ✗ ORCA not installed, skipping")
        orca_score = md_score
    else:
        print("  ✓ ORCA available, running calculation...")

        config = ORCACalculationConfig(
            method='PBE',
            basis_set='def2-SVP',
            calc_type='SP',
            nprocs=4
        )

        evaluator = ORCAEvaluator(
            config=config,
            use_cluster_model=True,
            work_dir='./orca_multiscale'
        )

        orca_score = evaluator.evaluate(alloy, verbose=True)

    print(f"  ORCA 得分: {orca_score:.4f}")
    print(f"  计算时间: ~数分钟")

    # 最终加权
    print("\n" + "=" * 60)
    print("多尺度评估汇总:")
    print("=" * 60)
    print(f"  简化模型: {simple_score:.4f} (权重 0.1)")
    print(f"  CALPHAD:  {calphad_score:.4f} (权重 0.2)")
    print(f"  MD 模拟:  {md_score:.4f} (权重 0.3)")
    print(f"  ORCA DFT: {orca_score:.4f} (权重 0.4)")

    final_score = (
        0.1 * simple_score +
        0.2 * calphad_score +
        0.3 * md_score +
        0.4 * orca_score
    )

    print(f"\n  最终得分: {final_score:.4f}")


def example_4_result_analysis():
    """示例 4：ORCA 结果分析"""
    print("\n" + "=" * 60)
    print("示例 4：ORCA 计算结果深度分析")
    print("=" * 60)

    if not ORCA_AVAILABLE:
        print("ERROR: orca_evaluator module not available")
        return

    print("\nORCA 提供的关键信息:")
    print("\n1. 电子结构信息:")
    print("   - 总能量（形成能）")
    print("   - HOMO-LUMO 能隙（电子稳定性）")
    print("   - 轨道能级分布")

    print("\n2. 电荷分布信息:")
    print("   - Mulliken 电荷（原子电荷）")
    print("   - Löwdin 电荷")
    print("   - 电荷转移分析")

    print("\n3. 键合信息:")
    print("   - 键级（Bond Order）")
    print("   - 自然键轨道（NBO）分析")
    print("   - 电荷密度分析")

    print("\n4. 光谱性质:")
    print("   - UV-Vis 吸收光谱（TD-DFT）")
    print("   - 振动频率（IR/Raman）")

    print("\n这些信息可以用于:")
    print("  ✓ 预测合金的电子导电性")
    print("  ✓ 理解元素间相互作用")
    print("  ✓ 优化元素配比")
    print("  ✓ 设计特定功能合金（如催化、电池材料）")


def main():
    """主函数"""
    print("=" * 60)
    print("ORCA 量子化学计算示例")
    print("高熵合金自进化智能体框架 v1.3.0")
    print("=" * 60)

    # 示例 1：基础计算
    example_1_basic_orca_calculation()

    # 示例 2：输入文件生成
    example_2_input_file_generation()

    # 示例 3：多尺度评估
    example_3_multiscale_evaluation()

    # 示例 4：结果分析
    example_4_result_analysis()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)

    print("\n提示:")
    print("  - ORCA 计算需要较长时间（分钟到小时）")
    print("  - 建议先使用团簇模型（13 原子）进行快速测试")
    print("  - 精确计算使用周期性超胞，但计算成本极高")
    print("  - 合理使用多尺度策略，ORCA 仅用于最终验证")


if __name__ == '__main__':
    main()

"""
分子动力学模拟集成示例
演示如何使用 LAMMPS 进行原子尺度模拟
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppmatAgent.hea_crewai_agent.core.material import Alloy
from ppmatAgent.hea_crewai_agent.core.evolution_engine import EvolutionEngine
from ppmatAgent.hea_crewai_agent.core.evaluator import ThermodynamicEvaluator
from ppmatAgent.hea_crewai_agent.core import CALPHAD_AVAILABLE

# 检查MD模拟依赖
try:
    from ppmatAgent.hea_crewai_agent.core.md_evaluator import MDEvaluator, StructureGenerator, LAMMPSInterface
    from ase import Atoms
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False


def main():
    print("="*70)
    print("高熵合金自进化智能体 - 分子动力学模拟集成")
    print("="*70)
    print()

    # 1. 检查环境
    print("1. 检查环境...")
    if not MD_AVAILABLE:
        print("❌ ASE 未安装")
        print("\n安装方法:")
        print("  pip install ase")
        print("  conda install -c conda-forge lammps  # 安装LAMMPS")
        print("\n本示例将跳过MD模拟。")
        use_md = False
    else:
        print("✓ ASE 已安装")
        use_md = True

    # 检查LAMMPS
    if use_md:
        try:
            import subprocess
            result = subprocess.run(['lmp', '-help'], capture_output=True, timeout=5)
            print("✓ LAMMPS 已安装")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("⚠️  LAMMPS 未找到")
            print("   安装: conda install -c conda-forge lammps")
            print("   本示例将使用模拟模式")
            use_md_simulation = False
        else:
            use_md_simulation = True
    print()

    # 2. 势函数配置
    print("2. 配置势函数库...")
    # 常用的高熵合金势函数：
    # - Zhou04: Co-Cr-Fe-Ni-Cu 系统
    # - Farkas: Al-Ni-Co 系统
    # - Mendelev: Fe-Cr-Ni 系统

    potential_library = {
        # 示例：需要替换为实际的势函数文件路径
        'Co-Cr-Fe-Ni': os.environ.get('EAM_POTENTIAL_COCRFENI', 'CoNiFeCr.eam.alloy'),
        'Al-Co-Cr-Fe-Ni': os.environ.get('EAM_POTENTIAL_ALCOCRFENI', 'AlCoCrFeNi.eam.alloy'),
    }

    # 检查势函数文件是否存在
    available_potentials = {}
    for key, path in potential_library.items():
        if os.path.exists(path):
            available_potentials[key] = path
            print(f"  ✓ {key}: {path}")
        else:
            print(f"  ⚠️  {key}: {path} (未找到)")

    if not available_potentials and use_md_simulation:
        print("\n⚠️  未找到势函数文件")
        print("  请从以下来源获取EAM势函数:")
        print("  1. NIST Interatomic Potentials: https://www.ctcms.nist.gov/potentials/")
        print("  2. OpenKIM: https://openkim.org/")
        print("  3. 文献中的补充材料")
        print("\n  设置环境变量:")
        print("  export EAM_POTENTIAL_COCRFENI=/path/to/CoNiFeCr.eam.alloy")
        use_md_simulation = False
    print()

    # 3. 创建结构生成器演示
    if use_md:
        print("3. 结构生成演示...")
        from ppmatAgent.hea_crewai_agent.core.md_evaluator import StructureGenerator

        generator = StructureGenerator()

        # 测试合金
        test_alloy = Alloy(composition={'Co': 0.25, 'Cr': 0.25, 'Fe': 0.25, 'Ni': 0.25})

        # 生成FCC结构
        fcc_atoms = generator.generate_fcc_structure(
            test_alloy,
            lattice_constant=3.6,
            size=(3, 3, 3)
        )

        print(f"  ✓ FCC结构生成完成")
        print(f"    原子数: {len(fcc_atoms)}")
        print(f"    盒子尺寸: {fcc_atoms.cell.lengths()}")
        print(f"    元素统计:")

        from collections import Counter
        elem_counts = Counter(fcc_atoms.get_chemical_symbols())
        for elem, count in sorted(elem_counts.items()):
            frac = count / len(fcc_atoms)
            print(f"      {elem}: {count} ({frac*100:.1f}%)")

        # 保存结构
        output_dir = '.ark/tmp/md_structures'
        os.makedirs(output_dir, exist_ok=True)

        from ase.io import write
        struct_file = f"{output_dir}/CoCrFeNi_fcc.xyz"
        write(struct_file, fcc_atoms)
        print(f"\n  ✓ 结构已保存: {struct_file}")
        print()

    # 4. MD评估器演示
    if use_md and use_md_simulation and available_potentials:
        print("4. 分子动力学模拟演示...")

        evaluator = MDEvaluator(
            potential_library=available_potentials,
            structure_type='fcc',
            temperature=300.0,
            lammps_executable='lmp',
            ncores=2
        )

        print(f"  配置:")
        print(f"    结构类型: FCC")
        print(f"    温度: 300 K")
        print(f"    CPU核心: 2")
        print()

        # 评估合金
        print("  运行MD模拟...")
        fitness = evaluator.evaluate(test_alloy)

        print(f"\n  MD模拟结果:")
        props = test_alloy.properties
        print(f"    状态: {props.get('md_status')}")

        if props.get('md_status') == 'success':
            print(f"    能量: {props.get('md_energy', 0):.4f} eV")
            print(f"    体积: {props.get('md_volume', 0):.2f} Å³")
            print(f"    压力: {props.get('md_pressure', 0):.2f} bar")
            print(f"    适应度: {fitness:.4f}")
        else:
            print(f"    原因: {props.get('md_status')}")
        print()

    # 5. 混合评估策略
    print("5. 多尺度评估策略...")
    print("  在实际应用中，可以组合多种评估方法:")
    print()
    print("  [快速筛选] 简化热力学模型")
    print("       ↓")
    print("  [中等精度] CALPHAD 相平衡计算")
    print("       ↓")
    print("  [高精度验证] MD 原子模拟")
    print()

    # 示例：多尺度评估器
    if use_md and MD_AVAILABLE:
        print("6. 多尺度评估器示例...")

        class MultiScaleEvaluator:
            """多尺度评估器"""

            def __init__(self, simple_eval, calphad_eval=None, md_eval=None):
                self.simple_eval = simple_eval
                self.calphad_eval = calphad_eval
                self.md_eval = md_eval

            def evaluate(self, alloy):
                # 第一阶段：快速筛选
                simple_score = self.simple_eval.evaluate(alloy)

                if simple_score < 0.5:
                    return simple_score  # 淘汰

                # 第二阶段：CALPHAD验证 (如果可用)
                if self.calphad_eval and simple_score >= 0.7:
                    calphad_score = self.calphad_eval.evaluate(alloy)
                    combined_score = 0.5 * simple_score + 0.5 * calphad_score

                    if combined_score < 0.7:
                        return combined_score  # 不够优秀

                    # 第三阶段：MD精确模拟 (仅顶尖候选)
                    if self.md_eval and combined_score >= 0.8:
                        md_score = self.md_eval.evaluate(alloy)
                        final_score = 0.3 * simple_score + 0.3 * calphad_score + 0.4 * md_score
                        return final_score

                    return combined_score

                return simple_score

        # 创建多尺度评估器
        simple = ThermodynamicEvaluator()

        calphad = None
        if CALPHAD_AVAILABLE:
            # 如果CALPHAD可用，添加
            pass

        md = None
        if use_md_simulation and available_potentials:
            md = MDEvaluator(potential_library=available_potentials)

        multi_eval = MultiScaleEvaluator(simple, calphad, md)

        print("  ✓ 多尺度评估器已创建")
        print(f"    - 简化模型: ✓")
        print(f"    - CALPHAD: {'✓' if calphad else '×'}")
        print(f"    - MD模拟: {'✓' if md else '×'}")
        print()

        # 测试
        print("  测试评估...")
        score = multi_eval.evaluate(test_alloy)
        print(f"  综合得分: {score:.4f}")
        print()

    # 7. 使用建议
    print("7. 使用建议")
    print("="*70)
    print()
    print("MD模拟的适用场景:")
    print("  ✓ 精确的力学性能预测 (杨氏模量、剪切模量)")
    print("  ✓ 高温性能评估 (熔点、热稳定性)")
    print("  ✓ 短程有序结构分析")
    print("  ✓ 扩散系数计算")
    print("  ✓ 验证CALPHAD预测")
    print()
    print("使用建议:")
    print("  1. 初期开发: 仅使用简化模型 (速度快)")
    print("  2. 中期优化: 添加CALPHAD (精度提升)")
    print("  3. 最终验证: 对最优候选使用MD (最高精度)")
    print()
    print("性能对比:")
    print("  简化模型:  ~1 ms/个体")
    print("  CALPHAD:   ~100 ms/个体")
    print("  MD模拟:    ~10-60 s/个体 (取决于系统大小)")
    print()

    # 8. 资源链接
    print("8. 相关资源")
    print("="*70)
    print()
    print("软件:")
    print("  - LAMMPS: https://www.lammps.org/")
    print("  - ASE: https://wiki.fysik.dtu.dk/ase/")
    print("  - OVITO (可视化): https://www.ovito.org/")
    print()
    print("势函数库:")
    print("  - NIST: https://www.ctcms.nist.gov/potentials/")
    print("  - OpenKIM: https://openkim.org/")
    print("  - Zhou04 (CoCrFeNi): 经典高熵合金势函数")
    print()
    print("学习资源:")
    print("  - LAMMPS 教程: https://docs.lammps.org/")
    print("  - AtomAgent 论文: 基于LLM的材料模拟助手")
    print()

    print("="*70)
    print("示例运行完成!")
    print("="*70)

    if not use_md:
        print("\n💡 提示: 安装ASE和LAMMPS以启用MD模拟:")
        print("   pip install ase")
        print("   conda install -c conda-forge lammps")


if __name__ == "__main__":
    main()

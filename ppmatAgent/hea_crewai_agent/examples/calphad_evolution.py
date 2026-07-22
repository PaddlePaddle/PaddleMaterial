"""
CALPHAD 集成示例
演示如何在高熵合金自进化系统中使用真正的 CALPHAD 计算
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppmatAgent.hea_crewai_agent.core.evolution_engine import AdaptiveEvolutionEngine
from ppmatAgent.hea_crewai_agent.core.evaluator import ThermodynamicEvaluator
from ppmatAgent.hea_crewai_agent.core import CALPHAD_AVAILABLE
from ppmatAgent.hea_crewai_agent.core.tdb_registry import find_best_local_tdb, get_local_tdb_registry

if CALPHAD_AVAILABLE:
    from ppmatAgent.hea_crewai_agent.core.calphad_evaluator import CALPHADEvaluator, HybridEvaluator


def main():
    print("="*70)
    print("高熵合金自进化智能体 - CALPHAD 集成示例")
    print("="*70)
    print()

    project_root = Path(__file__).resolve().parent.parent
    local_tdb_dir = project_root / 'tdb_files'

    # 0. 配置进化参数
    config = {
        'population_size': 20,  # 较小的种群 (CALPHAD 计算较慢)
        'max_generations': 30,
        'target_elements': ['Co', 'Cr', 'Fe', 'Ni', 'V'],
        'mutation_rate': 0.15,
        'crossover_rate': 0.8,
        'elite_ratio': 0.15,
        'fitness_profile': os.environ.get('FITNESS_PROFILE', 'single_phase').lower()
    }
    if config['fitness_profile'] not in {'single_phase', 'eutectic'}:
        print(f"⚠️  未知适应度模式 {config['fitness_profile']}，回退到 single_phase")
        config['fitness_profile'] = 'single_phase'

    constraints = {
        'max_elements': 5,
        'min_concentration': 0.10,
        'max_concentration': 0.35,
        'allowed_elements': config['target_elements']
    }

    # 1. 检查 CALPHAD 是否可用
    print("1. 检查 CALPHAD 环境...")
    if not CALPHAD_AVAILABLE:
        print("❌ pycalphad 未安装")
        print("\n安装方法:")
        print("  pip install pycalphad xarray")
        print("\n本示例将使用简化模型运行。")
        use_calphad = False
    else:
        print("✓ pycalphad 已安装")
        use_calphad = True
    print()

    # 2. 配置数据库路径
    print("2. 配置 CALPHAD 数据库...")
    database_path = None
    selected_db = None

    if use_calphad:
        env_database = os.environ.get('CALPHAD_DATABASE')

        if env_database:
            database_path = env_database
            basename = Path(database_path).name
            registry_entry = next(
                (entry for entry in get_local_tdb_registry(local_tdb_dir) if entry['filename'] == basename),
                None
            )

            if not os.path.exists(database_path):
                print(f"⚠️  环境变量指定的数据库不存在: {database_path}")
                print("\n本示例将使用简化模型运行。")
                use_calphad = False
            elif registry_entry is not None and not registry_entry['pycalphad_loadable']:
                print(f"⚠️  {basename} 是 MatCalc 原始数据库，不能直接被 pycalphad 读取。")
                print(f"  对应体系: {registry_entry['system_name']}")
                print(f"  建议用途: {registry_entry['recommended_use']}")
                print("\n本示例将使用简化模型运行。")
                use_calphad = False
            else:
                selected_db = registry_entry
                print(f"✓ 使用环境变量指定数据库: {database_path}")
        else:
            selected_db = find_best_local_tdb(
                config['target_elements'],
                tdb_dir=local_tdb_dir,
                allow_simplified=False
            )

            if selected_db is not None:
                database_path = selected_db['path']
                print(f"✓ 自动选择本地数据库: {selected_db['filename']}")
                print(f"  对应体系: {selected_db['system_name']}")
                print(f"  说明: {selected_db['recommended_use']}")
            else:
                print("⚠️  未找到与当前目标元素完全匹配的本地全量 CALPHAD 数据库。")
                print("\n本地数据库清单:")
                for entry in get_local_tdb_registry(local_tdb_dir):
                    status = '可直接用 pycalphad' if entry['pycalphad_loadable'] else 'MatCalc 原始库'
                    print(f"  - {entry['filename']}: {entry['system_name']} [{status}]")
                print("\n本示例将使用简化模型运行。")
                use_calphad = False
    print()

    # 3. 创建评估器
    print("3. 创建评估器...")

    if use_calphad:
        print("  模式: 混合评估 (简化模型 + CALPHAD)")
        print(f"  适应度模式: {config['fitness_profile']}")

        # 创建 CALPHAD 评估器
        calphad_eval = CALPHADEvaluator(
            database_path=database_path,
            temperature=1273.15,  # 1000°C
            pressure=101325.0,
            target_phases=(
                None
                if config['fitness_profile'] == 'eutectic'
                else (
                    selected_db['recommended_phases']
                    if selected_db is not None and selected_db['recommended_phases']
                    else ['FCC_A1', 'BCC_A2', 'HCP_A3']
                )
            ),
            prefer_single_phase=(config['fitness_profile'] != 'eutectic'),
            search_mode=config['fitness_profile']
        )

        # 创建简化评估器
        simple_eval = ThermodynamicEvaluator(search_mode=config['fitness_profile'])

        # 创建混合评估器
        evaluator = HybridEvaluator(
            calphad_evaluator=calphad_eval,
            simple_evaluator=simple_eval,
            calphad_threshold=0.7,  # 简化得分>0.7时使用CALPHAD
            use_calphad_probability=0.1  # 10%概率随机使用CALPHAD
        )

        print("  ✓ 混合评估器已创建")
        if config['fitness_profile'] == 'eutectic':
            print("    - 快速筛选: 共晶启发式热力学模型")
            print("    - 精确评估: 含液相的 CALPHAD 温度扫描")
        else:
            print("    - 快速筛选: 简化热力学模型")
            print("    - 精确评估: pycalphad 相平衡计算")
    else:
        print("  模式: 简化模型")
        print(f"  适应度模式: {config['fitness_profile']}")
        evaluator = ThermodynamicEvaluator(search_mode=config['fitness_profile'])
    print()

    # 4. 输出进化参数
    print("4. 配置进化参数...")
    print(f"  种群大小: {config['population_size']}")
    print(f"  最大代数: {config['max_generations']}")
    print(f"  候选元素: {', '.join(config['target_elements'])}")
    print()

    # 5. 创建进化引擎
    print("5. 初始化进化引擎...")
    engine = AdaptiveEvolutionEngine(
        population_size=config['population_size'],
        max_generations=config['max_generations'],
        target_elements=config['target_elements'],
        mutation_rate=config['mutation_rate'],
        crossover_rate=config['crossover_rate'],
        elite_ratio=config['elite_ratio'],
        constraints=constraints
    )
    print("  ✓ 自适应进化引擎已创建")
    print()

    # 6. 运行进化
    print("6. 开始进化过程...")
    print("-"*70)

    best_alloy = engine.evolve(
        evaluator=evaluator.evaluate if use_calphad else evaluator.evaluate,
        strategy_agent=None,
        verbose=True
    )

    print("-"*70)
    print()

    # 7. 展示结果
    print("7. 进化结果分析")
    print("="*70)
    print()

    print("最佳合金成分:")
    for elem, frac in sorted(best_alloy.composition.items(), key=lambda x: x[1], reverse=True):
        print(f"  {elem}: {frac*100:.2f}%")
    print()

    print("性能指标:")
    props = best_alloy.properties
    print(f"  总适应度: {props.get('fitness', 0):.4f}")
    print()

    # 8. CALPHAD 特有信息
    if use_calphad and 'calphad_status' in props:
        print("CALPHAD 计算结果:")
        print(f"  状态: {props.get('calphad_status')}")

        if props.get('calphad_status') == 'success':
            if props.get('calphad_search_mode') == 'eutectic':
                best_temperature = props.get('calphad_temperature')
                if best_temperature is not None:
                    print(f"  最佳共晶候选温度: {best_temperature - 273.15:.1f}°C")
                liquidus_temp = props.get('calphad_liquidus_temp')
                solidus_temp = props.get('calphad_solidus_temp')
                if liquidus_temp is not None:
                    print(f"  液相线近似温度: {liquidus_temp - 273.15:.1f}°C")
                if solidus_temp is not None:
                    print(f"  固相线近似温度: {solidus_temp - 273.15:.1f}°C")
                if props.get('calphad_freezing_range') is not None:
                    print(f"  凝固区间: {props.get('calphad_freezing_range', 0):.2f} K")
                print(f"  液相线压低量: {props.get('calphad_liquidus_depression', 0):.2f} K")
                print(f"  共晶反应评分: {props.get('calphad_eutectic_reaction_score', 0):.4f}")
                print(f"  最佳状态液相分数: {props.get('calphad_best_liquid_fraction', 0)*100:.2f}%")
            else:
                print(f"  温度: {props.get('calphad_temperature', 0) - 273.15:.1f}°C")
                print(f"  Gibbs 自由能: {props.get('calphad_gibbs_energy', 0):.2f} J/mol")

            phases = props.get('calphad_phases', {})
            if phases:
                print(f"  平衡相:")
                for phase, fraction in phases.items():
                    print(f"    - {phase}: {fraction*100:.2f}%")

                # 相稳定性分析
                if props.get('calphad_search_mode') == 'eutectic':
                    if 'LIQUID' in phases and len([p for p in phases if p != 'LIQUID']) >= 2:
                        print(f"\n  ✓ 发现液相参与的多相共存状态，可作为共晶候选")
                    else:
                        print(f"\n  ⚠️  扫描中未找到清晰的液相+多固相共存窗口")
                elif len(phases) == 1:
                    print(f"\n  ✓ 单相固溶体 - 优异的稳定性!")
                elif len(phases) == 2:
                    print(f"\n  ⚠️  两相结构 - 可能需要热处理优化")
                else:
                    print(f"\n  ❌ 多相结构 - 可能存在有害相")

            # 评估方法统计
            if isinstance(evaluator, HybridEvaluator):
                stats = evaluator.get_statistics()
                print(f"\n评估统计:")
                print(f"  总评估次数: {stats['total_evaluations']}")
                print(f"  CALPHAD 调用次数: {stats['calphad_evaluations']}")
                print(f"  CALPHAD 使用率: {stats['calphad_usage_ratio']*100:.1f}%")
        print()

    # 9. 热处理建议 (仅CALPHAD模式)
    if use_calphad and 'calphad_status' in props and props['calphad_status'] == 'success':
        print("9. 热处理建议...")
        try:
            heat_treatment = calphad_eval.suggest_heat_treatment(best_alloy)

            if 'solution_treatment' in heat_treatment:
                st = heat_treatment['solution_treatment']
                print(f"\n  固溶处理:")
                print(f"    温度: {st['temperature_C']:.1f}°C ({st['temperature_K']:.1f}K)")
                print(f"    时间: {st['duration_hours']:.1f} 小时")

                quench = heat_treatment.get('quench', {})
                print(f"\n  淬火:")
                print(f"    方法: {quench.get('method', 'N/A')}")

                aging = heat_treatment.get('aging', {})
                if 'temperature_C' in aging:
                    print(f"\n  时效:")
                    print(f"    温度: {aging['temperature_C']:.1f}°C")
                    print(f"    时间: {aging['duration_hours']:.1f} 小时")
            else:
                print(f"\n  {heat_treatment.get('warning', 'N/A')}")
                print(f"  {heat_treatment.get('recommendation', 'N/A')}")
        except Exception as e:
            print(f"  热处理建议生成失败: {e}")
        print()

    # 10. 保存结果
    print("10. 保存结果...")
    import json

    result_data = {
        'best_alloy': best_alloy.to_dict(),
        'config': config,
        'calphad_enabled': use_calphad,
        'database_path': database_path if use_calphad else None,
        'database_system': selected_db['system_name'] if use_calphad and selected_db is not None else None,
        'database_kind': selected_db['database_kind'] if use_calphad and selected_db is not None else None,
    }

    with open('calphad_evolution_result.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print("  ✓ 结果已保存到: calphad_evolution_result.json")
    print()

    print("="*70)
    print("示例运行完成!")
    print("="*70)

    if not use_calphad:
        print("\n💡 提示: 安装 pycalphad 以启用精确热力学计算:")
        print("   pip install pycalphad xarray")
        print("   或在 tdb_files/ 中提供与目标元素匹配的数据库")


if __name__ == "__main__":
    main()

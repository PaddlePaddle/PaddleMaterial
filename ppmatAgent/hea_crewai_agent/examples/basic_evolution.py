"""
基础进化示例
演示如何使用高熵合金自进化智能体框架
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppmatAgent.hea_crewai_agent.core.material import Alloy
from ppmatAgent.hea_crewai_agent.core.evolution_engine import EvolutionEngine, AdaptiveEvolutionEngine
from ppmatAgent.hea_crewai_agent.core.evaluator import MultiObjectiveEvaluator

try:
    from ppmatAgent.hea_crewai_agent.agents.strategy_agent import StrategyAgent
except ImportError:
    StrategyAgent = None


def create_strategy_agent():
    """Create an optional CrewAI-backed strategy agent from env vars."""
    use_strategy_agent = os.environ.get('USE_LLM_STRATEGY', '').lower() in {'1', 'true', 'yes', 'on'}
    if not use_strategy_agent:
        print("  策略智能体: 未启用 (设置 USE_LLM_STRATEGY=1 可启用 LLM 变异策略)")
        return None

    if StrategyAgent is None:
        print("  策略智能体: agents.strategy_agent 模块不可用，回退到规则变异")
        return None

    try:
        agent = StrategyAgent.from_env()
    except Exception as exc:
        print(f"  策略智能体初始化失败: {exc}")
        print("  将继续使用规则变异")
        return None

    backend = "CrewAI LLM" if getattr(agent, "crewai_llm", None) is not None else "custom completion backend"
    print(f"  策略智能体: 已启用 {backend} ({agent.llm_model})")
    return agent


def main():
    print("="*60)
    print("高熵合金自进化智能体 - 基础示例")
    print("="*60)
    print()

    # 1. 配置进化参数
    print("1. 配置进化参数...")
    config = {
        'population_size': 30,
        'max_generations': 50,
        'target_elements': ['Co', 'Cr', 'Fe', 'Ni', 'Al', 'Ti', 'Mn'],
        'mutation_rate': 0.15,
        'crossover_rate': 0.8,
        'elite_ratio': 0.1,
        'fitness_profile': os.environ.get('FITNESS_PROFILE', 'single_phase').lower()
    }

    constraints = {
        'max_elements': 6,
        'min_concentration': 0.05,
        'max_concentration': 0.35,
        'allowed_elements': config['target_elements']
    }

    print(f"  种群大小: {config['population_size']}")
    print(f"  最大代数: {config['max_generations']}")
    print(f"  候选元素: {', '.join(config['target_elements'])}")
    print(f"  适应度模式: {config['fitness_profile']}")
    print()

    # 2. 创建评估器
    print("2. 创建评估器...")
    if config['fitness_profile'] == 'eutectic':
        evaluator = MultiObjectiveEvaluator(
            objectives=['thermodynamic_eutectic'],
            weights=[1.0]
        )
        print("  搜索目标: 多元共晶预筛选")
        print("  优化目标:")
        print("    - 共晶启发式热力学评分 (权重: 1.0)")
    else:
        if config['fitness_profile'] != 'single_phase':
            print(f"  ⚠️ 未知适应度模式 {config['fitness_profile']}，回退到 single_phase")
            config['fitness_profile'] = 'single_phase'
        evaluator = MultiObjectiveEvaluator(
            objectives=['thermodynamic', 'mechanical', 'corrosion'],
            weights=[0.3, 0.5, 0.2]
        )
        print("  搜索目标: 高熵单相/综合性能")
        print("  优化目标:")
        print("    - 热力学稳定性 (权重: 0.3)")
        print("    - 力学性能 (权重: 0.5)")
        print("    - 耐腐蚀性 (权重: 0.2)")
    print()

    # 3. 创建进化引擎
    print("3. 初始化自适应进化引擎...")
    engine = AdaptiveEvolutionEngine(
        population_size=config['population_size'],
        max_generations=config['max_generations'],
        target_elements=config['target_elements'],
        mutation_rate=config['mutation_rate'],
        crossover_rate=config['crossover_rate'],
        elite_ratio=config['elite_ratio'],
        constraints=constraints
    )
    print("  引擎类型: 自适应进化引擎")
    print("  特性: 根据进化历史动态调整变异率")
    print()

    print("3.1 配置策略智能体...")
    strategy_agent = create_strategy_agent()
    print()

    # 4. 运行进化
    print("4. 开始进化过程...")
    print("-"*60)

    best_alloy = engine.evolve(
        evaluator=evaluator.evaluate,
        strategy_agent=strategy_agent,
        verbose=True
    )

    print("-"*60)
    print()

    # 5. 展示结果
    print("5. 进化结果分析")
    print("="*60)
    print()

    print("最佳合金成分:")
    for elem, frac in sorted(best_alloy.composition.items(), key=lambda x: x[1], reverse=True):
        print(f"  {elem}: {frac*100:.2f}%")
    print()

    print("性能指标:")
    props = best_alloy.properties
    print(f"  总适应度: {props.get('fitness', 0):.4f}")
    print()

    if 'objective_scores' in props:
        print("  各项得分:")
        for obj, score in props['objective_scores'].items():
            print(f"    {obj}: {score:.4f}")
        print()

    if props.get('thermo_search_mode') == 'eutectic':
        print("  共晶搜索指标:")
        print(f"    加权平均熔点: {props.get('eutectic_average_melting_point', 0):.2f} K")
        print(f"    熔点离散度: {props.get('eutectic_melting_point_spread', 0):.2f} K")
        print(f"    化学失配因子: {props.get('eutectic_chemical_mismatch', 0):.3f}")
        print(f"    原子尺寸差异 δ: {props.get('delta', 0):.2f}%")
        print()
    elif 'mixing_entropy' in props:
        print("  热力学参数:")
        print(f"    混合熵 ΔS_mix: {props['mixing_entropy']:.2f} J/(mol·K)")
        print(f"    混合焓 ΔH_mix: {props['mixing_enthalpy']:.2f} kJ/mol")
        print(f"    原子尺寸差异 δ: {props['delta']:.2f}%")
        print()

    if 'estimated_strength' in props:
        print("  力学性能估算:")
        print(f"    相对强度: {props['estimated_strength']:.3f}")
        print(f"    相对韧性: {props['estimated_ductility']:.3f}")
        print(f"    相对硬度: {props['estimated_hardness']:.3f}")
        print()

    # 6. 进化统计
    print("6. 进化统计信息")
    print("="*60)
    summary = engine.get_evolution_summary()
    print(f"  初始最佳适应度: {summary['initial_best_fitness']:.4f}")
    print(f"  最终最佳适应度: {summary['final_best_fitness']:.4f}")
    print(f"  适应度提升: {summary['improvement']:.4f} ({summary['improvement']/summary['initial_best_fitness']*100:.1f}%)")
    print(f"  最终种群多样性: {summary['final_diversity']:.4f}")
    print()

    # 7. 保存结果
    print("7. 保存结果...")
    import json

    result_data = {
        'best_alloy': best_alloy.to_dict(),
        'evolution_summary': summary,
        'config': config,
        'strategy_agent': {
            'enabled': strategy_agent is not None,
            'model': getattr(strategy_agent, 'llm_model', None),
            'last_source': getattr(strategy_agent, 'last_source', None),
            'last_rationale': getattr(strategy_agent, 'last_rationale', None)
        }
    }

    with open('evolution_result.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print("  结果已保存到: evolution_result.json")
    print()

    # 8. 可视化(可选)
    try:
        import matplotlib.pyplot as plt

        print("8. 生成进化曲线...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 适应度曲线
        axes[0].plot(engine.history['best_fitness'], label='Best Fitness', linewidth=2)
        axes[0].plot(engine.history['mean_fitness'], label='Mean Fitness', linewidth=2, alpha=0.7)
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Fitness')
        axes[0].set_title('Fitness Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 多样性曲线
        axes[1].plot(engine.history['diversity'], color='green', linewidth=2)
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Diversity')
        axes[1].set_title('Population Diversity')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('evolution_curves.png', dpi=300, bbox_inches='tight')
        print("  进化曲线已保存到: evolution_curves.png")
        print()

    except ImportError:
        print("8. 跳过可视化 (需要安装 matplotlib)")
        print()

    print("="*60)
    print("示例运行完成!")
    print("="*60)


if __name__ == "__main__":
    main()

"""
性能评估模块
实现多目标性能评估系统
"""

import numpy as np
from typing import Dict, List, Callable
from ppmatAgent.hea_crewai_agent.core.material import Alloy, get_element_property


class BaseEvaluator:
    """基础评估器"""

    def evaluate(self, alloy: Alloy) -> float:
        """评估合金,返回适应度"""
        raise NotImplementedError


class ThermodynamicEvaluator(BaseEvaluator):
    """热力学评估器"""

    VALID_SEARCH_MODES = {'single_phase', 'eutectic'}

    def __init__(self, search_mode: str = 'single_phase'):
        """
        Args:
            search_mode:
                - single_phase: 原始版本，偏好高熵单相固溶体
                - eutectic: 面向多元共晶搜索的启发式预筛选
        """
        if search_mode not in self.VALID_SEARCH_MODES:
            raise ValueError(
                f"Unsupported thermodynamic search mode: {search_mode}. "
                f"Choose from {sorted(self.VALID_SEARCH_MODES)}."
            )
        self.search_mode = search_mode

    def evaluate(self, alloy: Alloy) -> float:
        """
        基于热力学参数评估

        评估指标:
        - 混合熵 ΔS_mix
        - 混合焓 ΔH_mix (简化估算)
        - δ参数 (原子尺寸差异)
        """
        # 1. 混合熵 (已在Alloy类中实现)
        mixing_entropy = alloy.calculate_mixing_entropy()

        # 2. 混合焓估算 (简化: 基于价电子差异)
        mixing_enthalpy = self._estimate_mixing_enthalpy(alloy)

        # 3. 原子尺寸差异参数
        delta = self._calculate_delta(alloy)

        if self.search_mode == 'eutectic':
            return self._evaluate_eutectic(
                alloy=alloy,
                mixing_entropy=mixing_entropy,
                chemical_mismatch=mixing_enthalpy,
                delta=delta
            )

        return self._evaluate_single_phase(
            alloy=alloy,
            mixing_entropy=mixing_entropy,
            mixing_enthalpy=mixing_enthalpy,
            delta=delta
        )

    def _evaluate_single_phase(
        self,
        alloy: Alloy,
        mixing_entropy: float,
        mixing_enthalpy: float,
        delta: float
    ) -> float:
        """原始单相固溶体适应度标准"""
        # 综合得分: 高熵、低焓、适中的δ
        # ΔS_mix 越大越好 (>11 J/(mol·K))
        # ΔH_mix 接近0最好
        # δ 适中最好 (2-6%)

        s_score = min(mixing_entropy / 13.0, 1.0)  # 归一化到[0,1]
        h_score = np.exp(-abs(mixing_enthalpy) / 5.0)  # 焓接近0最好
        delta_score = np.exp(-((delta - 4.0) ** 2) / 4.0)  # 高斯分布,峰值在4%

        fitness = 0.4 * s_score + 0.3 * h_score + 0.3 * delta_score

        alloy.properties.update({
            'mixing_entropy': mixing_entropy,
            'mixing_enthalpy': mixing_enthalpy,
            'delta': delta,
            'thermo_search_mode': self.search_mode,
            'thermo_fitness': fitness
        })

        return fitness

    def _evaluate_eutectic(
        self,
        alloy: Alloy,
        mixing_entropy: float,
        chemical_mismatch: float,
        delta: float
    ) -> float:
        """
        多元共晶搜索的启发式预筛选标准。

        该模式不再偏好单相，而是近似奖励：
        - 较低的平均熔点（对应更低液相线的可能性）
        - 适度的化学失配与尺寸失配（促进多相分凝而非单相固溶）
        - 保持多元体系的成分复杂度
        """
        average_melting_point = self._calculate_average_melting_point(alloy)
        melting_point_spread = self._calculate_melting_point_spread(alloy, average_melting_point)
        element_count = alloy.get_element_count()

        avg_melting_score = np.clip((2600.0 - average_melting_point) / 1700.0, 0.0, 1.0)
        spread_score = np.clip(melting_point_spread / 450.0, 0.0, 1.0)
        mismatch_score = np.exp(-((chemical_mismatch - 10.0) ** 2) / 60.0)
        delta_score = np.exp(-((delta - 6.0) ** 2) / 10.0)
        entropy_score = np.exp(-((mixing_entropy - 9.5) ** 2) / 25.0)
        element_count_score = np.clip((element_count - 2) / 3.0, 0.0, 1.0)

        fitness = (
            0.40 * avg_melting_score
            + 0.20 * mismatch_score
            + 0.15 * delta_score
            + 0.10 * entropy_score
            + 0.10 * spread_score
            + 0.05 * element_count_score
        )

        alloy.properties.update({
            'mixing_entropy': mixing_entropy,
            'mixing_enthalpy': chemical_mismatch,
            'delta': delta,
            'thermo_search_mode': self.search_mode,
            'eutectic_average_melting_point': average_melting_point,
            'eutectic_melting_point_spread': melting_point_spread,
            'eutectic_chemical_mismatch': chemical_mismatch,
            'eutectic_element_count': element_count,
            'eutectic_component_scores': {
                'avg_melting_score': avg_melting_score,
                'spread_score': spread_score,
                'mismatch_score': mismatch_score,
                'delta_score': delta_score,
                'entropy_score': entropy_score,
                'element_count_score': element_count_score
            },
            'thermo_fitness': fitness,
            'eutectic_thermo_fitness': fitness
        })

        return fitness

    def _estimate_mixing_enthalpy(self, alloy: Alloy) -> float:
        """
        估算混合焓

        简化模型: 基于价电子数差异
        ΔH_mix ≈ Σ c_i c_j (VEC_i - VEC_j)^2
        """
        elements = list(alloy.composition.keys())
        fractions = list(alloy.composition.values())

        enthalpy = 0.0
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                vec_i = get_element_property(elements[i], 'valence_electrons')
                vec_j = get_element_property(elements[j], 'valence_electrons')
                enthalpy += fractions[i] * fractions[j] * (vec_i - vec_j) ** 2

        return enthalpy

    def _calculate_delta(self, alloy: Alloy) -> float:
        """
        计算原子尺寸差异参数 δ

        δ = 100 * sqrt(Σ c_i (1 - r_i/r_avg)^2)
        """
        elements = list(alloy.composition.keys())
        fractions = list(alloy.composition.values())

        # 计算平均原子半径
        r_avg = sum(
            fractions[i] * get_element_property(elements[i], 'atomic_radius')
            for i in range(len(elements))
        )

        # 计算δ
        delta_sq = sum(
            fractions[i] * (1 - get_element_property(elements[i], 'atomic_radius') / r_avg) ** 2
            for i in range(len(elements))
        )

        return 100 * np.sqrt(delta_sq)

    def _calculate_average_melting_point(self, alloy: Alloy) -> float:
        """计算按成分加权的平均熔点"""
        return sum(
            frac * get_element_property(elem, 'melting_point')
            for elem, frac in alloy.composition.items()
        )

    def _calculate_melting_point_spread(
        self,
        alloy: Alloy,
        average_melting_point: float
    ) -> float:
        """计算组元熔点的加权离散度"""
        spread_sq = sum(
            frac * (get_element_property(elem, 'melting_point') - average_melting_point) ** 2
            for elem, frac in alloy.composition.items()
        )
        return np.sqrt(spread_sq)


class MechanicalPropertiesEvaluator(BaseEvaluator):
    """力学性能评估器"""

    def evaluate(self, alloy: Alloy) -> float:
        """
        评估力学性能

        简化模型: 基于元素贡献的线性组合
        """
        # 强度贡献系数 (简化数据)
        strength_contrib = {
            'Co': 1.2, 'Cr': 1.3, 'Fe': 1.0, 'Ni': 0.9,
            'Al': 1.5, 'Ti': 1.8, 'V': 1.6, 'Mo': 1.7,
            'W': 1.9, 'Nb': 1.7, 'Ta': 1.8
        }

        # 韧性贡献系数
        ductility_contrib = {
            'Co': 1.1, 'Cr': 0.8, 'Fe': 1.0, 'Ni': 1.3,
            'Al': 0.7, 'Ti': 0.8, 'Cu': 1.2
        }

        # 计算综合强度
        strength = sum(
            frac * strength_contrib.get(elem, 1.0)
            for elem, frac in alloy.composition.items()
        )

        # 计算综合韧性
        ductility = sum(
            frac * ductility_contrib.get(elem, 1.0)
            for elem, frac in alloy.composition.items()
        )

        # 硬度估算 (与强度正相关)
        hardness = strength * 1.2

        # 综合得分: 强度和韧性的平衡
        fitness = 0.6 * min(strength / 1.5, 1.0) + 0.4 * min(ductility / 1.2, 1.0)

        alloy.properties.update({
            'estimated_strength': strength,
            'estimated_ductility': ductility,
            'estimated_hardness': hardness,
            'mechanical_fitness': fitness
        })

        return fitness


class CorrosionResistanceEvaluator(BaseEvaluator):
    """耐腐蚀性评估器"""

    def evaluate(self, alloy: Alloy) -> float:
        """评估耐腐蚀性"""
        # 耐腐蚀贡献系数
        corrosion_resist = {
            'Cr': 1.8, 'Ni': 1.5, 'Mo': 1.7, 'Cu': 1.3,
            'Al': 1.4, 'Ti': 1.2, 'Co': 1.1, 'Fe': 0.8
        }

        score = sum(
            frac * corrosion_resist.get(elem, 0.5)
            for elem, frac in alloy.composition.items()
        )

        # 归一化
        fitness = min(score / 1.5, 1.0)

        alloy.properties['corrosion_resistance'] = fitness
        return fitness


class MultiObjectiveEvaluator(BaseEvaluator):
    """
    多目标评估器

    综合多个性能指标
    """

    def __init__(
        self,
        objectives: List[str] = None,
        weights: List[float] = None,
        evaluators: Dict[str, BaseEvaluator] = None
    ):
        """
        初始化多目标评估器

        Args:
            objectives: 目标列表,如 ['strength', 'corrosion_resistance', 'cost']
            weights: 对应权重
            evaluators: 自定义评估器字典
        """
        self.objectives = objectives or ['thermodynamic', 'mechanical', 'corrosion']
        self.weights = weights or [1.0 / len(self.objectives)] * len(self.objectives)

        # 归一化权重
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # 默认评估器
        self.evaluators = evaluators or self._build_default_evaluators()

    def evaluate(self, alloy: Alloy) -> float:
        """
        多目标评估

        Returns:
            加权平均适应度
        """
        total_fitness = 0.0
        objective_scores = {}

        for obj, weight in zip(self.objectives, self.weights):
            if obj in self.evaluators:
                score = self.evaluators[obj].evaluate(alloy)
                objective_scores[obj] = score
                total_fitness += weight * score
            else:
                print(f"Warning: No evaluator for objective '{obj}'")

        # 保存各项得分
        alloy.properties['objective_scores'] = objective_scores
        alloy.properties['weighted_fitness'] = total_fitness

        return total_fitness

    @staticmethod
    def _build_default_evaluators() -> Dict[str, BaseEvaluator]:
        return {
            'thermodynamic': ThermodynamicEvaluator(),
            'thermodynamic_eutectic': ThermodynamicEvaluator(search_mode='eutectic'),
            'mechanical': MechanicalPropertiesEvaluator(),
            'corrosion': CorrosionResistanceEvaluator(),
            'cost': CostEvaluator(),
        }

    def get_pareto_front(self, population: List[Alloy]) -> List[Alloy]:
        """
        获取帕累托前沿

        Args:
            population: 合金种群

        Returns:
            帕累托最优解集
        """
        pareto_front = []

        for alloy in population:
            is_dominated = False

            for other in population:
                if other.alloy_id == alloy.alloy_id:
                    continue

                # 检查是否被支配
                if self._dominates(other, alloy):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(alloy)

        return pareto_front

    def _dominates(self, alloy1: Alloy, alloy2: Alloy) -> bool:
        """
        检查alloy1是否支配alloy2

        支配条件: alloy1在所有目标上不差于alloy2,且至少一个目标更好
        """
        scores1 = alloy1.properties.get('objective_scores', {})
        scores2 = alloy2.properties.get('objective_scores', {})

        at_least_one_better = False
        all_not_worse = True

        for obj in self.objectives:
            s1 = scores1.get(obj, 0)
            s2 = scores2.get(obj, 0)

            if s1 > s2:
                at_least_one_better = True
            elif s1 < s2:
                all_not_worse = False
                break

        return at_least_one_better and all_not_worse


class CostEvaluator(BaseEvaluator):
    """成本评估器"""

    def __init__(self, element_costs: Dict[str, float] = None):
        """
        Args:
            element_costs: 元素成本字典 ($/kg)
        """
        # 简化的成本数据 (相对值)
        self.element_costs = element_costs or {
            'Fe': 1.0, 'Ni': 15.0, 'Cr': 8.0, 'Co': 30.0,
            'Al': 2.5, 'Ti': 12.0, 'V': 20.0, 'Mo': 35.0,
            'W': 40.0, 'Cu': 8.0, 'Mn': 2.0, 'Nb': 45.0, 'Ta': 150.0
        }

    def evaluate(self, alloy: Alloy) -> float:
        """
        评估成本 (越低越好)

        Returns:
            成本适应度 (归一化到[0,1],低成本高分)
        """
        total_cost = sum(
            frac * self.element_costs.get(elem, 10.0)
            for elem, frac in alloy.composition.items()
        )

        alloy.properties['estimated_cost'] = total_cost

        # 转换为适应度 (成本越低,分数越高)
        # 假设成本范围[0, 50]
        fitness = 1.0 - min(total_cost / 50.0, 1.0)

        return fitness


# 工厂函数
def create_evaluator(config: Dict) -> BaseEvaluator:
    """
    根据配置创建评估器

    Args:
        config: 配置字典

    Returns:
        评估器实例
    """
    eval_type = config.get('type', 'multi_objective')

    if eval_type == 'thermodynamic':
        return ThermodynamicEvaluator(search_mode=config.get('search_mode', 'single_phase'))
    elif eval_type == 'thermodynamic_eutectic':
        return ThermodynamicEvaluator(search_mode='eutectic')
    elif eval_type == 'mechanical':
        return MechanicalPropertiesEvaluator()
    elif eval_type == 'corrosion':
        return CorrosionResistanceEvaluator()
    elif eval_type == 'cost':
        return CostEvaluator(config.get('element_costs'))
    elif eval_type == 'multi_objective':
        return MultiObjectiveEvaluator(
            objectives=config.get('objectives'),
            weights=config.get('weights'),
            evaluators=config.get('evaluators')
        )
    else:
        raise ValueError(f"Unknown evaluator type: {eval_type}")

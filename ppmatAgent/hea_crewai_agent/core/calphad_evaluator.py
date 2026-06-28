"""
CALPHAD 热力学评估模块
使用 pycalphad 进行精确的相平衡计算和热力学分析
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

try:
    from pycalphad import Database, equilibrium, variables as v
    PYCALPHAD_AVAILABLE = True
except ImportError:
    PYCALPHAD_AVAILABLE = False
    warnings.warn(
        "pycalphad not installed. Install with: pip install pycalphad\n"
        "CALPHADEvaluator will not be available."
    )

CALPHAD_AVAILABLE = PYCALPHAD_AVAILABLE

from ppmatAgent.hea_crewai_agent.core.material import Alloy, get_element_property
from ppmatAgent.hea_crewai_agent.core.evaluator import BaseEvaluator


class CALPHADEvaluator(BaseEvaluator):
    """
    基于 pycalphad 的精确热力学评估器

    使用 CALPHAD 方法计算:
    - 平衡相组成和分数
    - Gibbs 自由能
    - 相稳定性
    - 多相共存情况
    """

    VALID_SEARCH_MODES = {'single_phase', 'eutectic'}

    def __init__(
        self,
        database_path: str,
        temperature: float = 1273.15,  # 1000°C in Kelvin
        pressure: float = 101325.0,  # 1 atm in Pa
        target_phases: List[str] = None,
        prefer_single_phase: bool = True,
        search_mode: str = 'single_phase',
        eutectic_temperature_range: Optional[Tuple[float, float]] = None,
        eutectic_num_points: int = 25,
        liquid_phase_name: str = 'LIQUID',
        liquid_fraction_threshold: float = 0.05,
        solid_fraction_threshold: float = 0.05,
        adaptive_temperature_refinement: bool = False,
        refinement_levels: int = 2,
        refinement_points_per_interval: int = 7,
        refinement_liquid_jump_threshold: float = 0.25
    ):
        """
        初始化 CALPHAD 评估器

        Args:
            database_path: TDB 数据库文件路径
            temperature: 评估温度 (K)
            pressure: 评估压力 (Pa)
            target_phases: 目标相列表, 如 ['FCC_A1', 'BCC_A2']
            prefer_single_phase: 是否偏好单相固溶体
            search_mode:
                - single_phase: 原始版本，偏好单相稳定区
                - eutectic: 面向多元共晶搜索，扫描液相参与的相平衡
        """
        if not PYCALPHAD_AVAILABLE:
            raise ImportError(
                "pycalphad is required for CALPHADEvaluator. "
                "Install with: pip install pycalphad"
            )
        if search_mode not in self.VALID_SEARCH_MODES:
            raise ValueError(
                f"Unsupported CALPHAD search mode: {search_mode}. "
                f"Choose from {sorted(self.VALID_SEARCH_MODES)}."
            )

        try:
            self.db = Database(database_path)
            print(f"✓ Loaded CALPHAD database: {database_path}")
        except Exception as e:
            raise ValueError(f"Failed to load database {database_path}: {e}")

        self.temperature = temperature
        self.pressure = pressure
        self.prefer_single_phase = prefer_single_phase
        self.search_mode = search_mode
        self.eutectic_temperature_range = eutectic_temperature_range
        self.eutectic_num_points = eutectic_num_points
        self.liquid_phase_name = liquid_phase_name.upper()
        self.liquid_fraction_threshold = liquid_fraction_threshold
        self.solid_fraction_threshold = solid_fraction_threshold
        self.adaptive_temperature_refinement = adaptive_temperature_refinement
        self.refinement_levels = max(0, int(refinement_levels))
        self.refinement_points_per_interval = max(0, int(refinement_points_per_interval))
        self.refinement_liquid_jump_threshold = max(0.0, float(refinement_liquid_jump_threshold))

        # 获取数据库中可用的元素
        self.available_elements = {
            element.upper() for element in self.db.elements
            if element.upper() not in {'VA', '/-'}
        }
        self.available_phases = {phase.upper() for phase in self.db.phases.keys()}

        requested_phases = target_phases
        if requested_phases is None:
            if self.search_mode == 'eutectic':
                requested_phases = sorted(self.available_phases)
            else:
                requested_phases = ['FCC_A1', 'BCC_A2', 'HCP_A3']

        normalized_requested = [phase.upper() for phase in requested_phases]
        if self.search_mode == 'eutectic' and self.liquid_phase_name not in normalized_requested:
            normalized_requested.append(self.liquid_phase_name)

        missing_phases = sorted(set(normalized_requested) - self.available_phases)
        if missing_phases:
            warnings.warn(
                f"Ignoring phases not found in database: {missing_phases}",
                stacklevel=2
            )

        self.target_phases = [
            phase for phase in normalized_requested
            if phase in self.available_phases
        ]
        if not self.target_phases:
            raise ValueError("No valid target phases remain after filtering against the database.")
        self.target_phase_set = set(self.target_phases)

        print(f"✓ Available elements: {sorted(self.available_elements)}")
        if len(self.target_phases) <= 10:
            print(f"✓ Target phases: {self.target_phases}")
        else:
            print(f"✓ Target phases: {len(self.target_phases)} phases")
        print(f"✓ Search mode: {self.search_mode}")

    def evaluate(self, alloy: Alloy) -> float:
        """
        评估合金的热力学稳定性

        Returns:
            fitness: 稳定性得分 [0, 1]
        """
        try:
            normalized_composition = self._normalize_composition(alloy)
            if normalized_composition is None:
                return 0.0

            if self.search_mode == 'eutectic':
                return self._evaluate_eutectic(alloy, normalized_composition)

            result = self._run_equilibrium(normalized_composition, self.temperature)
            phase_fractions = self._extract_phase_fractions(result)
            gibbs_energy = self._extract_gibbs_energy(result)
            fitness = self._calculate_single_phase_fitness(phase_fractions, gibbs_energy)

            alloy.properties.update({
                'calphad_status': 'success',
                'calphad_search_mode': self.search_mode,
                'calphad_temperature': self.temperature,
                'calphad_phases': phase_fractions,
                'calphad_gibbs_energy': gibbs_energy,
                'calphad_fitness': fitness
            })

            return fitness

        except Exception as e:
            print(f"CALPHAD calculation failed: {e}")
            alloy.properties['calphad_status'] = f'error: {str(e)}'
            return 0.0

    def _normalize_composition(self, alloy: Alloy) -> Optional[Dict[str, float]]:
        """标准化并校验成分是否存在于数据库中"""
        normalized_composition = {
            element.upper(): fraction for element, fraction in alloy.composition.items()
        }

        missing_elements = set(normalized_composition.keys()) - self.available_elements
        if missing_elements:
            print(f"Warning: Elements {missing_elements} not in database")
            alloy.properties['calphad_status'] = 'elements_not_found'
            return None

        return normalized_composition

    def _run_equilibrium(
        self,
        normalized_composition: Dict[str, float],
        temperature: float
    ):
        """在指定温度下运行一次 CALPHAD 平衡计算"""
        sorted_elements = sorted(normalized_composition.keys())
        composition = {
            v.X(element): normalized_composition[element]
            for element in sorted_elements[:-1]
        }
        return equilibrium(
            self.db,
            sorted_elements + ['VA'],
            self.target_phases,
            {v.T: temperature, v.P: self.pressure, **composition},
            verbose=False
        )

    def _extract_phase_fractions(self, result) -> Dict[str, float]:
        """提取各相的分数"""
        phase_fractions = {}

        phase_names = np.atleast_1d(np.asarray(result.Phase).squeeze()).tolist()
        phase_amounts = np.atleast_1d(np.asarray(result.NP).squeeze()).tolist()

        for phase, fraction in zip(phase_names, phase_amounts):
            if not phase or str(fraction) == 'nan':
                continue

            phase_name = phase.upper()
            if phase_name in self.target_phase_set and fraction > 1e-6:
                phase_fractions[phase_name] = float(fraction)

        return phase_fractions

    def _extract_gibbs_energy(self, result) -> float:
        """提取 Gibbs 自由能"""
        try:
            # 获取系统的总 Gibbs 自由能
            gibbs = float(result.GM.squeeze())
            return gibbs
        except:
            return 0.0

    def _calculate_single_phase_fitness(
        self,
        phase_fractions: Dict[str, float],
        gibbs_energy: float
    ) -> float:
        """
        根据相组成和自由能计算适应度

        评分策略:
        1. 单相固溶体: 高分 (0.8-1.0)
        2. 两相: 中等 (0.5-0.8)
        3. 多相: 低分 (0.2-0.5)
        4. Gibbs 能越低越好
        """
        if not phase_fractions:
            return 0.0

        num_phases = len(phase_fractions)

        # 基于相数量的得分
        if num_phases == 1:
            # 单相固溶体 - 最优
            phase_score = 1.0
        elif num_phases == 2:
            # 两相 - 可接受
            phase_score = 0.65
        else:
            # 多相 - 不理想
            phase_score = 0.3

        # 检查是否有有害相 (σ, μ, Laves等金属间化合物)
        harmful_phases = {'SIGMA', 'MU_PHASE', 'LAVES', 'CHI'}
        has_harmful = any(
            any(harmful in phase.upper() for harmful in harmful_phases)
            for phase in phase_fractions.keys()
        )

        if has_harmful:
            phase_score *= 0.5  # 惩罚有害相

        # 基于 Gibbs 能的得分 (越低越好)
        # 归一化到 [0, 1], 假设 Gibbs 能在 [-50000, 0] J/mol 范围
        gibbs_score = np.clip(1.0 + gibbs_energy / 50000.0, 0.0, 1.0)

        # 综合得分
        fitness = 0.7 * phase_score + 0.3 * gibbs_score

        return fitness

    def _estimate_weighted_melting_point(self, alloy: Alloy) -> float:
        """按成分估算组元加权熔点"""
        return sum(
            fraction * get_element_property(element, 'melting_point')
            for element, fraction in alloy.composition.items()
        )

    def _default_eutectic_temperature_range(self, alloy: Alloy) -> Tuple[float, float]:
        """给共晶搜索估算一个合理的温度扫描区间"""
        melting_points = [
            get_element_property(element, 'melting_point')
            for element in alloy.composition.keys()
            if get_element_property(element, 'melting_point') > 0
        ]
        if not melting_points:
            return (700.0, 2200.0)

        weighted_mp = self._estimate_weighted_melting_point(alloy)
        t_min = max(500.0, 0.60 * min(melting_points))
        t_max = min(3500.0, max(1.05 * weighted_mp, 1.02 * max(melting_points)))
        if t_max <= t_min + 200.0:
            t_max = min(3500.0, t_min + 200.0)

        return (t_min, t_max)

    @staticmethod
    def _classify_eutectic_state(
        state: Dict,
        liquid_phase_name: str = 'LIQUID',
        liquid_fraction_threshold: float = 0.05,
        solid_fraction_threshold: float = 0.05
    ) -> Dict:
        """把单个温度点归类，供共晶窗口细化判断使用。"""
        phases = state.get('phase_fractions', state.get('phases', {})) or {}
        liquid_phase_name = liquid_phase_name.upper()

        liquid_fraction = max(float(phases.get(liquid_phase_name, 0.0)), 0.0)
        active_solid_phases = tuple(sorted(
            phase
            for phase, fraction in phases.items()
            if phase != liquid_phase_name and float(fraction) >= solid_fraction_threshold
        ))

        return {
            'liquid_fraction': liquid_fraction,
            'has_any_liquid': liquid_fraction > 1e-6,
            'liquid_above_threshold': liquid_fraction >= liquid_fraction_threshold,
            'has_partial_liquid': liquid_fraction > 1e-6 and liquid_fraction < 1.0 - 1e-6,
            'solid_count': len(active_solid_phases),
            'active_solid_phases': active_solid_phases,
        }

    @classmethod
    def identify_eutectic_refinement_intervals(
        cls,
        scan_states: List[Dict],
        liquid_phase_name: str = 'LIQUID',
        liquid_fraction_threshold: float = 0.05,
        solid_fraction_threshold: float = 0.05,
        liquid_jump_threshold: float = 0.25
    ) -> List[Tuple[float, float]]:
        """
        识别需要加密扫描的温区。

        目标是补捉粗扫描里容易漏掉的窄部分液相窗口，例如:
        - 两固相 -> 纯液相 的突跳
        - 液相分数陡变
        - 液相参与下的固相组合变化
        """
        ordered_states = sorted(scan_states, key=lambda state: float(state['temperature']))
        intervals: List[Tuple[float, float]] = []

        for left_state, right_state in zip(ordered_states, ordered_states[1:]):
            left_temp = float(left_state['temperature'])
            right_temp = float(right_state['temperature'])
            if right_temp <= left_temp:
                continue

            left_profile = cls._classify_eutectic_state(
                left_state,
                liquid_phase_name=liquid_phase_name,
                liquid_fraction_threshold=liquid_fraction_threshold,
                solid_fraction_threshold=solid_fraction_threshold
            )
            right_profile = cls._classify_eutectic_state(
                right_state,
                liquid_phase_name=liquid_phase_name,
                liquid_fraction_threshold=liquid_fraction_threshold,
                solid_fraction_threshold=solid_fraction_threshold
            )

            should_refine = False
            if left_profile['has_partial_liquid'] or right_profile['has_partial_liquid']:
                should_refine = True
            if left_profile['liquid_above_threshold'] != right_profile['liquid_above_threshold']:
                should_refine = True
            if left_profile['active_solid_phases'] != right_profile['active_solid_phases']:
                should_refine = True
            if abs(left_profile['liquid_fraction'] - right_profile['liquid_fraction']) >= liquid_jump_threshold:
                should_refine = True
            if (
                (left_profile['solid_count'] >= 2 and right_profile['has_any_liquid'])
                or (right_profile['solid_count'] >= 2 and left_profile['has_any_liquid'])
            ):
                should_refine = True

            if should_refine:
                intervals.append((left_temp, right_temp))

        return intervals

    def _scan_eutectic_states(
        self,
        normalized_composition: Dict[str, float],
        temperatures: List[float]
    ) -> List[Dict]:
        """在给定温度列表上执行平衡计算并收集有效状态。"""
        scan_states = []
        seen_temperatures = set()

        for temperature in sorted(float(temp) for temp in temperatures):
            rounded_temperature = round(float(temperature), 8)
            if rounded_temperature in seen_temperatures:
                continue
            seen_temperatures.add(rounded_temperature)

            try:
                result = self._run_equilibrium(normalized_composition, float(temperature))
                phase_fractions = self._extract_phase_fractions(result)
                if not phase_fractions:
                    continue
                scan_states.append({
                    'temperature': float(temperature),
                    'phases': phase_fractions,
                    'gibbs_energy': self._extract_gibbs_energy(result)
                })
            except Exception:
                continue

        return scan_states

    @staticmethod
    def _merge_scan_states(scan_states: List[Dict], new_states: List[Dict]) -> List[Dict]:
        """按温度合并扫描状态，后加入的结果覆盖重复温度。"""
        merged_states = {
            round(float(state['temperature']), 8): state
            for state in scan_states
        }
        for state in new_states:
            merged_states[round(float(state['temperature']), 8)] = state

        return sorted(merged_states.values(), key=lambda state: float(state['temperature']))

    def _collect_eutectic_scan_states(
        self,
        normalized_composition: Dict[str, float],
        temperature_range: Tuple[float, float]
    ) -> List[Dict]:
        """先粗扫，再按需要自适应细化相变温区。"""
        temperatures = np.linspace(
            temperature_range[0],
            temperature_range[1],
            self.eutectic_num_points
        )
        scan_states = self._scan_eutectic_states(normalized_composition, temperatures.tolist())

        if (
            not self.adaptive_temperature_refinement
            or self.refinement_levels <= 0
            or self.refinement_points_per_interval <= 0
            or len(scan_states) < 2
        ):
            return scan_states

        for _ in range(self.refinement_levels):
            refinement_intervals = self.identify_eutectic_refinement_intervals(
                scan_states=scan_states,
                liquid_phase_name=self.liquid_phase_name,
                liquid_fraction_threshold=self.liquid_fraction_threshold,
                solid_fraction_threshold=self.solid_fraction_threshold,
                liquid_jump_threshold=self.refinement_liquid_jump_threshold
            )
            if not refinement_intervals:
                break

            existing_temperatures = {
                round(float(state['temperature']), 8)
                for state in scan_states
            }
            new_temperatures = []
            for left_temp, right_temp in refinement_intervals:
                if right_temp <= left_temp:
                    continue
                candidate_temperatures = np.linspace(
                    left_temp,
                    right_temp,
                    self.refinement_points_per_interval + 2
                )[1:-1]
                for candidate in candidate_temperatures:
                    rounded_candidate = round(float(candidate), 8)
                    if rounded_candidate in existing_temperatures:
                        continue
                    existing_temperatures.add(rounded_candidate)
                    new_temperatures.append(float(candidate))

            if not new_temperatures:
                break

            additional_states = self._scan_eutectic_states(normalized_composition, new_temperatures)
            if not additional_states:
                break
            scan_states = self._merge_scan_states(scan_states, additional_states)

        return scan_states

    @staticmethod
    def summarize_eutectic_scan(
        scan_states: List[Dict],
        weighted_melting_point: float,
        liquid_phase_name: str = 'LIQUID',
        liquid_fraction_threshold: float = 0.05,
        solid_fraction_threshold: float = 0.05
    ) -> Dict:
        """
        从温度扫描结果中抽取共晶相关指标。

        输入的 scan_states 形如:
        {
            'temperature': 1200.0,
            'phases': {'LIQUID': 0.3, 'FCC_A1': 0.4, 'BCC_A2': 0.3},
            'gibbs_energy': -12345.0
        }
        """
        liquid_phase_name = liquid_phase_name.upper()
        partial_liquid_temperatures = []
        best_reaction = {
            'score': 0.0,
            'temperature': None,
            'phase_fractions': {},
            'liquid_fraction': 0.0,
            'gibbs_energy': 0.0
        }

        for state in scan_states:
            temperature = float(state['temperature'])
            phases = state.get('phase_fractions', state.get('phases', {})) or {}
            gibbs_energy = float(state.get('gibbs_energy', 0.0))

            liquid_fraction = phases.get(liquid_phase_name, 0.0)
            solid_phases = {
                phase: fraction
                for phase, fraction in phases.items()
                if phase != liquid_phase_name and fraction >= solid_fraction_threshold
            }

            if liquid_fraction >= liquid_fraction_threshold and solid_phases:
                partial_liquid_temperatures.append(temperature)

            if liquid_fraction < liquid_fraction_threshold or len(solid_phases) < 2:
                continue

            solid_count = len(solid_phases)
            if solid_count == 2:
                solid_count_score = 1.0
            elif solid_count == 3:
                solid_count_score = 0.9
            elif solid_count == 4:
                solid_count_score = 0.75
            else:
                solid_count_score = 0.6

            top_two = sorted(solid_phases.values(), reverse=True)[:2]
            top_two_total = sum(top_two)
            solid_balance_score = (
                1.0 - abs(top_two[0] - top_two[1]) / max(top_two_total, 1e-9)
            )

            if 0.10 <= liquid_fraction <= 0.60:
                liquid_presence_score = 1.0
            elif liquid_fraction < 0.10:
                liquid_presence_score = liquid_fraction / 0.10
            else:
                liquid_presence_score = max(0.0, 1.0 - (liquid_fraction - 0.60) / 0.40)

            reaction_score = (
                0.45 * solid_count_score
                + 0.35 * solid_balance_score
                + 0.20 * liquid_presence_score
            )

            if reaction_score > best_reaction['score']:
                best_reaction = {
                    'score': reaction_score,
                    'temperature': temperature,
                    'phase_fractions': phases,
                    'liquid_fraction': liquid_fraction,
                    'gibbs_energy': gibbs_energy
                }

        liquidus_temp = max(partial_liquid_temperatures) if partial_liquid_temperatures else None
        solidus_temp = min(partial_liquid_temperatures) if partial_liquid_temperatures else None

        if len(partial_liquid_temperatures) >= 2:
            freezing_range = liquidus_temp - solidus_temp
            freezing_range_score = np.exp(-freezing_range / 150.0)
        elif partial_liquid_temperatures:
            freezing_range = 0.0
            freezing_range_score = 0.6
        else:
            freezing_range = None
            freezing_range_score = 0.0

        if liquidus_temp is not None and weighted_melting_point > 0:
            liquidus_depression = max(weighted_melting_point - liquidus_temp, 0.0)
            liquidus_score = np.clip(
                liquidus_depression / max(weighted_melting_point * 0.25, 150.0),
                0.0,
                1.0
            )
        else:
            liquidus_depression = 0.0
            liquidus_score = 0.0

        fitness = (
            0.50 * best_reaction['score']
            + 0.30 * freezing_range_score
            + 0.20 * liquidus_score
        )

        return {
            'fitness': fitness,
            'best_temperature': best_reaction['temperature'],
            'best_phase_fractions': best_reaction['phase_fractions'],
            'best_liquid_fraction': best_reaction['liquid_fraction'],
            'best_gibbs_energy': best_reaction['gibbs_energy'],
            'reaction_score': best_reaction['score'],
            'liquidus_temp': liquidus_temp,
            'solidus_temp': solidus_temp,
            'freezing_range': freezing_range,
            'freezing_range_score': freezing_range_score,
            'liquidus_depression': liquidus_depression,
            'liquidus_score': liquidus_score
        }

    def _evaluate_eutectic(
        self,
        alloy: Alloy,
        normalized_composition: Dict[str, float]
    ) -> float:
        """面向共晶搜索的 CALPHAD 温度扫描评估"""
        temperature_range = (
            self.eutectic_temperature_range
            if self.eutectic_temperature_range is not None
            else self._default_eutectic_temperature_range(alloy)
        )
        scan_states = self._collect_eutectic_scan_states(
            normalized_composition=normalized_composition,
            temperature_range=temperature_range
        )

        if not scan_states:
            alloy.properties.update({
                'calphad_status': 'no_valid_scan',
                'calphad_search_mode': self.search_mode
            })
            return 0.0

        weighted_melting_point = self._estimate_weighted_melting_point(alloy)
        summary = self.summarize_eutectic_scan(
            scan_states=scan_states,
            weighted_melting_point=weighted_melting_point,
            liquid_phase_name=self.liquid_phase_name,
            liquid_fraction_threshold=self.liquid_fraction_threshold,
            solid_fraction_threshold=self.solid_fraction_threshold
        )

        alloy.properties.update({
            'calphad_status': 'success',
            'calphad_search_mode': self.search_mode,
            'calphad_temperature_range': temperature_range,
            'calphad_scan_points': self.eutectic_num_points,
            'calphad_total_scan_points': len(scan_states),
            'calphad_adaptive_refinement': self.adaptive_temperature_refinement,
            'calphad_refinement_levels': self.refinement_levels if self.adaptive_temperature_refinement else 0,
            'calphad_refinement_points_per_interval': (
                self.refinement_points_per_interval if self.adaptive_temperature_refinement else 0
            ),
            'calphad_weighted_melting_point': weighted_melting_point,
            'calphad_temperature': summary['best_temperature'],
            'calphad_phases': summary['best_phase_fractions'],
            'calphad_gibbs_energy': summary['best_gibbs_energy'],
            'calphad_liquidus_temp': summary['liquidus_temp'],
            'calphad_solidus_temp': summary['solidus_temp'],
            'calphad_freezing_range': summary['freezing_range'],
            'calphad_liquidus_depression': summary['liquidus_depression'],
            'calphad_best_liquid_fraction': summary['best_liquid_fraction'],
            'calphad_eutectic_reaction_score': summary['reaction_score'],
            'calphad_fitness': summary['fitness'],
            'calphad_eutectic_fitness': summary['fitness']
        })

        return summary['fitness']

    def get_phase_diagram_point(
        self,
        alloy: Alloy,
        temperature_range: Tuple[float, float] = None
    ) -> Dict:
        """
        获取给定成分在相图上的信息

        Args:
            alloy: 合金
            temperature_range: 温度范围 (T_min, T_max), 默认 (300K, 2000K)

        Returns:
            相图信息字典
        """
        if temperature_range is None:
            temperature_range = (300.0, 2000.0)

        normalized_composition = self._normalize_composition(alloy)
        if normalized_composition is None:
            return {}

        T_min, T_max = temperature_range
        temperatures = np.linspace(T_min, T_max, 50)

        phase_evolution = {}

        for T in temperatures:
            try:
                result = self._run_equilibrium(normalized_composition, float(T))
                phase_fractions = self._extract_phase_fractions(result)
                phase_evolution[T] = phase_fractions

            except:
                continue

        return phase_evolution

    def suggest_heat_treatment(self, alloy: Alloy) -> Dict:
        """
        基于相图建议热处理方案

        Returns:
            热处理建议
        """
        phase_evolution = self.get_phase_diagram_point(alloy)

        # 分析单相区温度范围
        single_phase_temps = []
        for T, phases in phase_evolution.items():
            if len(phases) == 1:
                single_phase_temps.append(T)

        if single_phase_temps:
            # 建议在单相区进行固溶处理
            solution_temp = np.mean(single_phase_temps)

            return {
                'solution_treatment': {
                    'temperature_K': solution_temp,
                    'temperature_C': solution_temp - 273.15,
                    'duration_hours': 2.0,
                    'description': 'Solution treatment in single-phase region'
                },
                'quench': {
                    'method': 'water',
                    'description': 'Rapid quench to retain single phase'
                },
                'aging': {
                    'temperature_C': solution_temp - 373.15,  # 100°C lower
                    'duration_hours': 4.0,
                    'description': 'Optional aging treatment'
                }
            }
        else:
            return {
                'warning': 'No single-phase region found in temperature range',
                'recommendation': 'Consider adjusting composition or use multi-phase processing'
            }


class HybridEvaluator(BaseEvaluator):
    """
    混合评估器: 快速简化模型 + 精确 CALPHAD 计算

    策略:
    1. 第一阶段: 使用简化模型快速筛选 (1000+ 个体/秒)
    2. 第二阶段: 对优秀个体使用 CALPHAD 精确评估 (10 个体/秒)
    """

    def __init__(
        self,
        calphad_evaluator: Optional[CALPHADEvaluator] = None,
        simple_evaluator: BaseEvaluator = None,
        calphad_threshold: float = 0.7,  # 简化模型得分阈值
        use_calphad_probability: float = 0.1  # CALPHAD 使用概率
    ):
        """
        初始化混合评估器

        Args:
            calphad_evaluator: CALPHAD 评估器 (可选)
            simple_evaluator: 简化评估器
            calphad_threshold: 超过此得分才使用 CALPHAD
            use_calphad_probability: 随机使用 CALPHAD 的概率
        """
        self.calphad_evaluator = calphad_evaluator
        self.simple_evaluator = simple_evaluator
        self.calphad_threshold = calphad_threshold
        self.use_calphad_probability = use_calphad_probability

        self.calphad_calls = 0
        self.simple_calls = 0

    def evaluate(self, alloy: Alloy) -> float:
        """
        混合评估策略
        """
        # 先用简化模型评估
        simple_score = self.simple_evaluator.evaluate(alloy)
        self.simple_calls += 1

        # 决定是否使用 CALPHAD
        use_calphad = False

        if self.calphad_evaluator is not None:
            # 策略1: 高分个体必用 CALPHAD
            if simple_score >= self.calphad_threshold:
                use_calphad = True
            # 策略2: 低分个体有小概率用 CALPHAD (探索)
            elif np.random.random() < self.use_calphad_probability:
                use_calphad = True

        if use_calphad:
            calphad_score = self.calphad_evaluator.evaluate(alloy)
            self.calphad_calls += 1

            # 综合得分 (CALPHAD 权重更高)
            final_score = 0.3 * simple_score + 0.7 * calphad_score

            alloy.properties['evaluation_method'] = 'hybrid_calphad'
            alloy.properties['simple_score'] = simple_score
            alloy.properties['calphad_score'] = calphad_score
        else:
            final_score = simple_score
            alloy.properties['evaluation_method'] = 'simple_only'

        return final_score

    def get_statistics(self) -> Dict:
        """获取评估统计"""
        total = self.simple_calls
        calphad_ratio = self.calphad_calls / total if total > 0 else 0

        return {
            'total_evaluations': total,
            'simple_evaluations': self.simple_calls,
            'calphad_evaluations': self.calphad_calls,
            'calphad_usage_ratio': calphad_ratio
        }


# 使用示例
if __name__ == "__main__":
    from ppmatAgent.hea_crewai_agent.core.material import Alloy

    # 示例: 使用 CALPHAD 评估 CoCrFeNiV
    if PYCALPHAD_AVAILABLE:
        print("=" * 60)
        print("CALPHAD Evaluator Demo")
        print("=" * 60)

        database_path = "tdb_files/CoCrFeNiV.TDB-R3.txt"

        try:
            evaluator = CALPHADEvaluator(
                database_path=database_path,
                temperature=1273.15,  # 1000°C
                target_phases=['FCC_A1', 'BCC_A2', 'HCP_A3', 'SIGMA']
            )

            # 测试合金
            test_alloy = Alloy(composition={
                'Co': 0.20,
                'Cr': 0.20,
                'Fe': 0.20,
                'Ni': 0.20,
                'V': 0.20
            })

            print(f"\nEvaluating alloy: {test_alloy.composition}")
            fitness = evaluator.evaluate(test_alloy)

            print(f"\nResults:")
            print(f"  Fitness: {fitness:.4f}")
            print(f"  Phases: {test_alloy.properties.get('calphad_phases', {})}")
            print(f"  Gibbs Energy: {test_alloy.properties.get('calphad_gibbs_energy', 0):.2f} J/mol")

            # 热处理建议
            heat_treatment = evaluator.suggest_heat_treatment(test_alloy)
            print(f"\nHeat Treatment Suggestion:")
            for step, params in heat_treatment.items():
                print(f"  {step}: {params}")

        except (FileNotFoundError, ValueError):
            print(f"\nDatabase file not found: {database_path}")
            print("Please provide a valid CoCrFeNiV or other compatible HEA database file.")
    else:
        print("pycalphad not installed. Install with: pip install pycalphad")

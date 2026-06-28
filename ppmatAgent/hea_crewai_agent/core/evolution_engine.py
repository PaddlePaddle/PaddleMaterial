"""
进化引擎模块
实现高熵合金的自进化优化算法
"""

import copy
import random
from typing import List, Dict, Callable, Optional
import numpy as np
from ppmatAgent.hea_crewai_agent.core.material import Alloy, Population
from ppmatAgent.hea_crewai_agent.core.mel import MELParser, MELExecutor, MELGenerator


class EvolutionEngine:
    """进化引擎核心类"""

    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        target_elements: List[str] = None,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1,
        constraints: Dict = None,
        seed_alloys: Optional[List[Alloy]] = None,
    ):
        """
        初始化进化引擎

        Args:
            population_size: 种群大小
            max_generations: 最大代数
            target_elements: 目标元素池
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elite_ratio: 精英保留比例
            constraints: 约束条件
            seed_alloys: 初始种子候选
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.target_elements = target_elements or ['Co', 'Cr', 'Fe', 'Ni', 'Al', 'Ti']
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.current_generation = 0

        # 默认约束
        self.constraints = constraints or {
            'max_elements': 7,
            'min_concentration': 0.05,
            'max_concentration': 0.40,
            'allowed_elements': self.target_elements
        }
        self.seed_alloys = seed_alloys or []

        # MEL 组件
        self.mel_parser = MELParser()
        self.mel_executor = MELExecutor()
        self.mel_generator = MELGenerator()

        # 进化历史
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'diversity': []
        }
        self.final_population: Optional[Population] = None

    def initialize_population(self) -> Population:
        """初始化种群"""
        population = Population()
        used_signatures = set()

        for alloy in self.seed_alloys:
            seeded = self._clone_alloy(alloy)
            signature = self._composition_signature(seeded.composition)
            if signature in used_signatures:
                continue
            if not seeded.is_valid(self.constraints):
                continue
            population.add_alloy(seeded)
            used_signatures.add(signature)
            if len(population) >= self.population_size:
                return population

        max_attempts = max(self.population_size * 50, 200)
        attempts = 0
        while len(population) < self.population_size and attempts < max_attempts:
            alloy = self._create_random_alloy()
            attempts += 1
            signature = self._composition_signature(alloy.composition)
            if signature in used_signatures:
                continue
            if not alloy.is_valid(self.constraints):
                continue
            population.add_alloy(alloy)
            used_signatures.add(signature)

        while len(population) < self.population_size:
            alloy = self._create_random_alloy()
            if alloy.is_valid(self.constraints):
                population.add_alloy(alloy)

        return population

    def _composition_signature(self, composition: Dict[str, float], decimals: int = 6):
        """生成稳定的成分签名,用于去重。"""
        return tuple(
            sorted(
                (element, round(float(fraction), decimals))
                for element, fraction in composition.items()
            )
        )

    def _clone_alloy(self, alloy: Alloy) -> Alloy:
        """复制合金对象,避免直接修改输入种子。"""
        return Alloy(
            composition=dict(alloy.composition),
            properties=copy.deepcopy(alloy.properties),
            metadata=copy.deepcopy(alloy.metadata),
            generation=alloy.generation,
            parent_ids=list(alloy.parent_ids),
        )

    def _create_random_alloy(self) -> Alloy:
        """创建随机合金"""
        # 随机选择3-5种元素
        n_elements = random.randint(3, min(5, len(self.target_elements)))
        elements = random.sample(self.target_elements, n_elements)

        # 生成随机成分
        fractions = np.random.dirichlet(np.ones(n_elements))
        composition = dict(zip(elements, fractions))

        return Alloy(composition=composition)

    def evolve(
        self,
        evaluator: Callable,
        strategy_agent: Optional[object] = None,
        verbose: bool = True
    ) -> Alloy:
        """
        执行进化过程

        Args:
            evaluator: 评估函数,接收Alloy返回fitness
            strategy_agent: 策略智能体(可选)
            verbose: 是否打印进度

        Returns:
            最佳合金
        """
        # 初始化种群
        population = self.initialize_population()

        # 评估初始种群
        self._evaluate_population(population, evaluator)

        for generation in range(self.max_generations):
            self.current_generation = generation
            if verbose:
                best = population.get_best()
                print(f"Generation {generation}: Best Fitness = {best.properties['fitness']:.4f}")

            # 选择
            parents = self._selection(population)

            # 生成新一代
            offspring = []
            while len(offspring) < self.population_size - int(self.population_size * self.elite_ratio):
                parent1, parent2 = random.sample(parents, 2)

                # 交叉
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2])

                # 变异
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, strategy_agent)

                # 检查约束
                if child.is_valid(self.constraints):
                    offspring.append(child)

            # 精英保留
            elites = population.get_top_k(int(self.population_size * self.elite_ratio))

            # 组成新种群
            new_population = Population(generation=generation + 1)
            for alloy in elites + offspring:
                new_population.add_alloy(alloy)

            # 评估新种群
            self._evaluate_population(new_population, evaluator)

            # 更新统计
            new_population.update_statistics()
            self.history['best_fitness'].append(new_population.statistics['max_fitness'])
            self.history['mean_fitness'].append(new_population.statistics['mean_fitness'])
            self.history['diversity'].append(new_population.statistics['diversity'])

            population = new_population

        # 返回最佳个体
        best_alloy = population.get_best()
        self.final_population = population
        if verbose:
            print(f"\n进化完成! 最佳适应度: {best_alloy.properties['fitness']:.4f}")
            print(f"最佳合金: {best_alloy.composition}")

        return best_alloy

    def _evaluate_population(self, population: Population, evaluator: Callable):
        """评估种群"""
        for alloy in population.alloys:
            if 'fitness' not in alloy.properties:
                fitness = evaluator(alloy)
                alloy.properties['fitness'] = fitness

    def _selection(self, population: Population) -> List[Alloy]:
        """
        选择操作 - 锦标赛选择

        Args:
            population: 种群

        Returns:
            选中的父代
        """
        tournament_size = 3
        parents = []

        for _ in range(self.population_size):
            # 锦标赛
            candidates = random.sample(population.alloys, tournament_size)
            winner = max(candidates, key=lambda a: a.properties.get('fitness', 0))
            parents.append(winner)

        return parents

    def _crossover(self, parent1: Alloy, parent2: Alloy) -> Alloy:
        """
        交叉操作 - 成分混合

        Args:
            parent1, parent2: 父代合金

        Returns:
            子代合金
        """
        # 随机混合比例
        alpha = random.uniform(0.3, 0.7)

        # 合并所有元素
        all_elements = set(parent1.composition.keys()) | set(parent2.composition.keys())

        # 混合成分
        new_composition = {}
        for elem in all_elements:
            frac1 = parent1.composition.get(elem, 0)
            frac2 = parent2.composition.get(elem, 0)
            new_frac = alpha * frac1 + (1 - alpha) * frac2

            if new_frac > 0.001:  # 移除微量元素
                new_composition[elem] = new_frac

        # 归一化
        total = sum(new_composition.values())
        new_composition = {k: v/total for k, v in new_composition.items()}

        child = Alloy(
            composition=new_composition,
            parent_ids=[parent1.alloy_id, parent2.alloy_id]
        )

        return child

    def _mutate(self, alloy: Alloy, strategy_agent: Optional[object] = None) -> Alloy:
        """
        变异操作 - 使用MEL

        Args:
            alloy: 待变异合金
            strategy_agent: 策略智能体(可选)

        Returns:
            变异后的合金
        """
        if strategy_agent is not None:
            # 使用策略智能体生成MEL
            strategy_context = {
                'generation': self.current_generation,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'population_size': self.population_size,
                'best_fitness_history': self.history['best_fitness'][-5:],
            }
            try:
                mel_string = strategy_agent.generate_mutation(
                    alloy,
                    allowed_elements=self.target_elements,
                    constraints=self.constraints,
                    evolution_context=strategy_context
                )
            except TypeError:
                mel_string = strategy_agent.generate_mutation(alloy)
            mutation_source = 'strategy_agent'
        else:
            # 随机生成MEL
            mel_string = self.mel_generator.generate_random_operation(
                alloy,
                self.target_elements
            )
            mutation_source = 'random'

        try:
            # 解析并执行MEL
            operations = self.mel_parser.parse(mel_string)
            mutated = self.mel_executor.execute(alloy, operations)
            mutated.metadata['mutation'] = {
                'source': mutation_source,
                'mel': mel_string
            }
            return mutated
        except Exception as e:
            # 如果MEL执行失败,返回原合金
            print(f"Warning: MEL mutation failed: {e}")
            return alloy

    def get_evolution_summary(self) -> Dict:
        """获取进化总结"""
        return {
            'total_generations': len(self.history['best_fitness']),
            'final_best_fitness': self.history['best_fitness'][-1] if self.history['best_fitness'] else 0,
            'initial_best_fitness': self.history['best_fitness'][0] if self.history['best_fitness'] else 0,
            'improvement': (
                self.history['best_fitness'][-1] - self.history['best_fitness'][0]
                if self.history['best_fitness'] else 0
            ),
            'final_diversity': self.history['diversity'][-1] if self.history['diversity'] else 0
        }


class AdaptiveEvolutionEngine(EvolutionEngine):
    """
    自适应进化引擎
    根据进化历史动态调整参数
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stagnation_counter = 0
        self.best_fitness_history = []

    def evolve(self, evaluator: Callable, strategy_agent: Optional[object] = None, verbose: bool = True) -> Alloy:
        """重写evolve方法,增加自适应逻辑"""
        population = self.initialize_population()
        self._evaluate_population(population, evaluator)

        for generation in range(self.max_generations):
            self.current_generation = generation
            best = population.get_best()
            current_best_fitness = best.properties['fitness']

            if verbose:
                print(f"Generation {generation}: Best={current_best_fitness:.4f}, "
                      f"MutRate={self.mutation_rate:.3f}, Diversity={population.calculate_diversity():.4f}")

            # 检测停滞
            if self.best_fitness_history:
                if abs(current_best_fitness - self.best_fitness_history[-1]) < 0.001:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0

            self.best_fitness_history.append(current_best_fitness)

            # 自适应调整参数
            if self.stagnation_counter > 10:
                # 停滞时增加变异率和探索
                self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
                if verbose:
                    print(f"  -> 检测到停滞,增加变异率到 {self.mutation_rate:.3f}")
                self.stagnation_counter = 0
            elif self.stagnation_counter == 0:
                # 进步时降低变异率
                self.mutation_rate = max(0.05, self.mutation_rate * 0.95)

            # 后续逻辑与父类相同
            parents = self._selection(population)
            offspring = []

            while len(offspring) < self.population_size - int(self.population_size * self.elite_ratio):
                parent1, parent2 = random.sample(parents, 2)

                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2])

                if random.random() < self.mutation_rate:
                    child = self._mutate(child, strategy_agent)

                if child.is_valid(self.constraints):
                    offspring.append(child)

            elites = population.get_top_k(int(self.population_size * self.elite_ratio))
            new_population = Population(generation=generation + 1)

            for alloy in elites + offspring:
                new_population.add_alloy(alloy)

            self._evaluate_population(new_population, evaluator)
            new_population.update_statistics()

            self.history['best_fitness'].append(new_population.statistics['max_fitness'])
            self.history['mean_fitness'].append(new_population.statistics['mean_fitness'])
            self.history['diversity'].append(new_population.statistics['diversity'])

            population = new_population

        best_alloy = population.get_best()
        self.final_population = population
        if verbose:
            print(f"\n进化完成! 最佳适应度: {best_alloy.properties['fitness']:.4f}")

        return best_alloy

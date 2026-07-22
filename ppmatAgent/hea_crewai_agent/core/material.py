"""
材料数据结构模块
定义高熵合金的核心数据结构
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Alloy:
    """高熵合金数据结构"""

    composition: Dict[str, float]  # 元素成分 {元素符号: 原子分数}
    properties: Dict[str, float] = field(default_factory=dict)  # 性能指标
    metadata: Dict[str, any] = field(default_factory=dict)  # 元数据
    generation: int = 0  # 所属代数
    parent_ids: List[str] = field(default_factory=list)  # 父代ID
    alloy_id: Optional[str] = None  # 唯一标识符

    def __post_init__(self):
        """初始化后处理"""
        if self.alloy_id is None:
            self.alloy_id = self._generate_id()
        self._normalize_composition()

    def _generate_id(self) -> str:
        """生成唯一标识符"""
        import hashlib
        comp_str = "-".join([f"{k}{v:.3f}" for k, v in sorted(self.composition.items())])
        return hashlib.md5(comp_str.encode()).hexdigest()[:12]

    def _normalize_composition(self):
        """归一化成分,确保总和为1.0"""
        total = sum(self.composition.values())
        if abs(total - 1.0) > 1e-6:
            self.composition = {k: v/total for k, v in self.composition.items()}

    def get_element_count(self) -> int:
        """获取元素数量"""
        return len(self.composition)

    def get_major_elements(self, threshold: float = 0.05) -> List[str]:
        """获取主要元素(含量>threshold)"""
        return [elem for elem, frac in self.composition.items() if frac >= threshold]

    def calculate_mixing_entropy(self) -> float:
        """
        计算混合熵
        ΔS_mix = -R Σ c_i ln(c_i)
        """
        R = 8.314  # J/(mol·K)
        entropy = 0.0
        for fraction in self.composition.values():
            if fraction > 0:
                entropy += fraction * np.log(fraction)
        return -R * entropy

    def is_valid(self, constraints: Dict) -> bool:
        """
        检查是否满足约束条件

        Args:
            constraints: 约束字典
        """
        # 检查元素数量
        if self.get_element_count() > constraints.get('max_elements', 7):
            return False

        # 检查成分范围
        min_conc = constraints.get('min_concentration', 0.05)
        max_conc = constraints.get('max_concentration', 0.40)
        for fraction in self.composition.values():
            if fraction < min_conc or fraction > max_conc:
                return False

        # 检查元素是否在允许列表中
        allowed = constraints.get('allowed_elements', [])
        if allowed:
            for elem in self.composition.keys():
                if elem not in allowed:
                    return False

        return True

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'alloy_id': self.alloy_id,
            'composition': self.composition,
            'properties': self.properties,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Alloy':
        """从字典创建"""
        return cls(
            composition=data['composition'],
            properties=data.get('properties', {}),
            generation=data.get('generation', 0),
            parent_ids=data.get('parent_ids', []),
            alloy_id=data.get('alloy_id'),
            metadata=data.get('metadata', {})
        )

    def __repr__(self) -> str:
        comp_str = " ".join([f"{k}:{v:.3f}" for k, v in self.composition.items()])
        return f"Alloy({self.alloy_id}, {comp_str})"


@dataclass
class Population:
    """种群数据结构"""

    alloys: List[Alloy] = field(default_factory=list)
    generation: int = 0
    statistics: Dict[str, float] = field(default_factory=dict)

    def add_alloy(self, alloy: Alloy):
        """添加合金"""
        alloy.generation = self.generation
        self.alloys.append(alloy)

    def get_best(self, metric: str = 'fitness') -> Optional[Alloy]:
        """获取最佳个体"""
        if not self.alloys:
            return None
        return max(self.alloys, key=lambda a: a.properties.get(metric, 0))

    def get_top_k(self, k: int, metric: str = 'fitness') -> List[Alloy]:
        """获取前k个个体"""
        sorted_alloys = sorted(
            self.alloys,
            key=lambda a: a.properties.get(metric, 0),
            reverse=True
        )
        return sorted_alloys[:k]

    def calculate_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.alloys) < 2:
            return 0.0

        # 计算所有两两之间的成分差异
        differences = []
        for i in range(len(self.alloys)):
            for j in range(i+1, len(self.alloys)):
                diff = self._composition_distance(
                    self.alloys[i].composition,
                    self.alloys[j].composition
                )
                differences.append(diff)

        return np.mean(differences) if differences else 0.0

    def _composition_distance(self, comp1: Dict, comp2: Dict) -> float:
        """计算两个成分之间的距离"""
        all_elements = set(comp1.keys()) | set(comp2.keys())
        distance = 0.0
        for elem in all_elements:
            v1 = comp1.get(elem, 0)
            v2 = comp2.get(elem, 0)
            distance += (v1 - v2) ** 2
        return np.sqrt(distance)

    def update_statistics(self):
        """更新种群统计信息"""
        if not self.alloys:
            return

        fitness_values = [a.properties.get('fitness', 0) for a in self.alloys]
        self.statistics = {
            'size': len(self.alloys),
            'mean_fitness': np.mean(fitness_values),
            'max_fitness': np.max(fitness_values),
            'min_fitness': np.min(fitness_values),
            'std_fitness': np.std(fitness_values),
            'diversity': self.calculate_diversity()
        }

    def __len__(self) -> int:
        return len(self.alloys)

    def __repr__(self) -> str:
        return f"Population(gen={self.generation}, size={len(self.alloys)})"


# 元素属性数据库(简化版)
ELEMENT_PROPERTIES = {
    'Co': {'atomic_radius': 1.25, 'valence_electrons': 9, 'melting_point': 1768},
    'Cr': {'atomic_radius': 1.28, 'valence_electrons': 6, 'melting_point': 2180},
    'Fe': {'atomic_radius': 1.26, 'valence_electrons': 8, 'melting_point': 1811},
    'Ni': {'atomic_radius': 1.24, 'valence_electrons': 10, 'melting_point': 1728},
    'Mn': {'atomic_radius': 1.27, 'valence_electrons': 7, 'melting_point': 1519},
    'Al': {'atomic_radius': 1.43, 'valence_electrons': 3, 'melting_point': 933},
    'Ti': {'atomic_radius': 1.47, 'valence_electrons': 4, 'melting_point': 1941},
    'V': {'atomic_radius': 1.34, 'valence_electrons': 5, 'melting_point': 2183},
    'Mo': {'atomic_radius': 1.39, 'valence_electrons': 6, 'melting_point': 2896},
    'W': {'atomic_radius': 1.39, 'valence_electrons': 6, 'melting_point': 3695},
    'Cu': {'atomic_radius': 1.28, 'valence_electrons': 11, 'melting_point': 1358},
    'Nb': {'atomic_radius': 1.46, 'valence_electrons': 5, 'melting_point': 2750},
    'Ta': {'atomic_radius': 1.46, 'valence_electrons': 5, 'melting_point': 3290},
}


def get_element_property(element: str, property_name: str) -> float:
    """获取元素属性"""
    return ELEMENT_PROPERTIES.get(element, {}).get(property_name, 0.0)

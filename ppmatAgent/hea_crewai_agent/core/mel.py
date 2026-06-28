"""
Material Edit Language (MEL) 模块
实现材料编辑的符号化语言
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from ppmatAgent.hea_crewai_agent.core.material import Alloy


@dataclass
class MELOperation:
    """MEL 操作"""

    operation_type: str  # REPLACE, ADD, REMOVE, ADJUST
    params: Dict  # 操作参数

    def __repr__(self) -> str:
        return f"{self.operation_type}({self.params})"


class MELParser:
    """MEL 语法解析器"""

    def __init__(self):
        self.operation_patterns = {
            'REPLACE': r'REPLACE\((\w+),\s*([\d.]+),\s*(\w+)\)',
            'ADD': r'ADD\((\w+),\s*([\d.]+)\)',
            'REMOVE': r'REMOVE\((\w+)\)',
            'ADJUST': r'ADJUST\((\w+),\s*([\d.]+)\)',
            'SCALE': r'SCALE\((\w+),\s*([\d.]+)\)'
        }

    def parse(self, mel_string: str) -> List[MELOperation]:
        """
        解析 MEL 字符串

        Args:
            mel_string: MEL 表达式,例如 "REPLACE(Fe, 0.2, Co) + ADD(Ti, 0.05)"

        Returns:
            MEL 操作列表
        """
        operations = []

        # 分割复合操作
        parts = re.split(r'\s*\+\s*', mel_string)

        for part in parts:
            part = part.strip()
            operation = self._parse_single_operation(part)
            if operation:
                operations.append(operation)

        return operations

    def _parse_single_operation(self, op_string: str) -> MELOperation:
        """解析单个操作"""
        for op_type, pattern in self.operation_patterns.items():
            match = re.match(pattern, op_string)
            if match:
                return self._create_operation(op_type, match.groups())

        raise ValueError(f"Invalid MEL operation: {op_string}")

    def _create_operation(self, op_type: str, params: Tuple) -> MELOperation:
        """创建操作对象"""
        if op_type == 'REPLACE':
            return MELOperation(
                operation_type='REPLACE',
                params={'source': params[0], 'fraction': float(params[1]), 'target': params[2]}
            )
        elif op_type == 'ADD':
            return MELOperation(
                operation_type='ADD',
                params={'element': params[0], 'fraction': float(params[1])}
            )
        elif op_type == 'REMOVE':
            return MELOperation(
                operation_type='REMOVE',
                params={'element': params[0]}
            )
        elif op_type == 'ADJUST':
            return MELOperation(
                operation_type='ADJUST',
                params={'element': params[0], 'target_fraction': float(params[1])}
            )
        elif op_type == 'SCALE':
            return MELOperation(
                operation_type='SCALE',
                params={'element': params[0], 'scale_factor': float(params[1])}
            )


class MELExecutor:
    """MEL 操作执行器"""

    def execute(self, alloy: Alloy, operations: List[MELOperation]) -> Alloy:
        """
        执行一系列 MEL 操作

        Args:
            alloy: 原始合金
            operations: 操作列表

        Returns:
            修改后的合金
        """
        # 复制成分
        new_composition = alloy.composition.copy()

        for operation in operations:
            new_composition = self._execute_operation(new_composition, operation)

        # 创建新合金
        new_alloy = Alloy(
            composition=new_composition,
            parent_ids=[alloy.alloy_id]
        )

        return new_alloy

    def _execute_operation(self, composition: Dict, operation: MELOperation) -> Dict:
        """执行单个操作"""
        if operation.operation_type == 'REPLACE':
            return self._replace(composition, operation.params)
        elif operation.operation_type == 'ADD':
            return self._add(composition, operation.params)
        elif operation.operation_type == 'REMOVE':
            return self._remove(composition, operation.params)
        elif operation.operation_type == 'ADJUST':
            return self._adjust(composition, operation.params)
        elif operation.operation_type == 'SCALE':
            return self._scale(composition, operation.params)
        else:
            raise ValueError(f"Unknown operation type: {operation.operation_type}")

    def _replace(self, composition: Dict, params: Dict) -> Dict:
        """
        替换操作: 将source元素的fraction部分替换为target元素

        Example: REPLACE(Fe, 0.2, Co) - 将Fe的20%替换为Co
        """
        source = params['source']
        fraction = params['fraction']
        target = params['target']

        if source not in composition:
            return composition

        new_comp = composition.copy()
        source_amount = new_comp[source]
        replace_amount = source_amount * fraction

        # 减少source
        new_comp[source] -= replace_amount

        # 增加target
        if target in new_comp:
            new_comp[target] += replace_amount
        else:
            new_comp[target] = replace_amount

        # 移除含量过小的元素
        new_comp = {k: v for k, v in new_comp.items() if v > 0.001}

        # 归一化
        total = sum(new_comp.values())
        return {k: v/total for k, v in new_comp.items()}

    def _add(self, composition: Dict, params: Dict) -> Dict:
        """
        添加操作: 添加新元素并重新归一化

        Example: ADD(Ti, 0.05) - 添加5%的Ti
        """
        element = params['element']
        fraction = params['fraction']

        new_comp = composition.copy()

        # 缩放现有元素
        scale = 1.0 - fraction
        new_comp = {k: v * scale for k, v in new_comp.items()}

        # 添加新元素
        if element in new_comp:
            new_comp[element] += fraction
        else:
            new_comp[element] = fraction

        return new_comp

    def _remove(self, composition: Dict, params: Dict) -> Dict:
        """
        移除操作: 移除指定元素并重新归一化

        Example: REMOVE(Mn) - 移除Mn元素
        """
        element = params['element']

        new_comp = composition.copy()
        if element in new_comp:
            del new_comp[element]

        # 归一化
        total = sum(new_comp.values())
        if total > 0:
            return {k: v/total for k, v in new_comp.items()}
        return new_comp

    def _adjust(self, composition: Dict, params: Dict) -> Dict:
        """
        调整操作: 将指定元素调整到目标含量

        Example: ADJUST(Ni, 0.25) - 将Ni调整到25%
        """
        element = params['element']
        target_fraction = params['target_fraction']

        new_comp = composition.copy()

        if element in new_comp:
            current = new_comp[element]
            diff = target_fraction - current

            # 调整其他元素
            other_total = 1.0 - target_fraction
            current_other_total = 1.0 - current

            if current_other_total > 0:
                scale = other_total / current_other_total
                for k in new_comp:
                    if k != element:
                        new_comp[k] *= scale

            new_comp[element] = target_fraction
        else:
            # 元素不存在,添加并调整其他元素
            scale = 1.0 - target_fraction
            new_comp = {k: v * scale for k, v in new_comp.items()}
            new_comp[element] = target_fraction

        return new_comp

    def _scale(self, composition: Dict, params: Dict) -> Dict:
        """
        缩放操作: 将指定元素按比例缩放

        Example: SCALE(Fe, 1.2) - 将Fe增加20%
        """
        element = params['element']
        scale_factor = params['scale_factor']

        if element not in composition:
            return composition

        new_comp = composition.copy()
        new_comp[element] *= scale_factor

        # 归一化
        total = sum(new_comp.values())
        return {k: v/total for k, v in new_comp.items()}


class MELGenerator:
    """MEL 生成器 - 由LLM或规则生成合理的MEL操作"""

    def generate_random_operation(self, alloy: Alloy, allowed_elements: List[str]) -> str:
        """生成随机 MEL 操作"""
        import random

        existing_elements = list(alloy.composition.keys())
        candidate_new_elements = [e for e in allowed_elements if e not in alloy.composition]

        operation_types = ['ADJUST', 'SCALE']
        if candidate_new_elements:
            operation_types.extend(['REPLACE', 'ADD'])

        op_type = random.choice(operation_types)

        if op_type == 'REPLACE':
            source = random.choice(existing_elements)
            target = random.choice(candidate_new_elements)
            fraction = random.uniform(0.1, 0.5)
            return f"REPLACE({source}, {fraction:.2f}, {target})"

        elif op_type == 'ADD':
            element = random.choice(candidate_new_elements)
            fraction = random.uniform(0.03, 0.10)
            return f"ADD({element}, {fraction:.2f})"

        elif op_type == 'ADJUST':
            element = random.choice(existing_elements)
            target = random.uniform(0.10, 0.35)
            return f"ADJUST({element}, {target:.2f})"

        elif op_type == 'SCALE':
            element = random.choice(existing_elements)
            scale = random.uniform(0.8, 1.2)
            return f"SCALE({element}, {scale:.2f})"

    def generate_targeted_operations(self, alloy: Alloy, objectives: List[str]) -> List[str]:
        """
        基于目标生成针对性的MEL操作

        这里可以集成LLM来生成更智能的操作
        """
        operations = []

        if 'increase_strength' in objectives:
            # 增加强度: 增加Ti, Nb等强化元素
            operations.append("ADD(Ti, 0.05)")

        if 'improve_corrosion' in objectives:
            # 改善耐蚀: 增加Cr
            operations.append("ADJUST(Cr, 0.25)")

        if 'reduce_cost' in objectives:
            # 降低成本: 减少贵金属如Co
            operations.append("SCALE(Co, 0.7)")

        return operations


# 使用示例
if __name__ == "__main__":
    # 创建测试合金
    test_alloy = Alloy(composition={'Fe': 0.3, 'Ni': 0.3, 'Cr': 0.2, 'Co': 0.2})

    print(f"原始合金: {test_alloy.composition}")

    # 解析MEL
    parser = MELParser()
    operations = parser.parse("REPLACE(Fe, 0.2, Ti) + ADD(Al, 0.03)")

    print(f"MEL 操作: {operations}")

    # 执行操作
    executor = MELExecutor()
    new_alloy = executor.execute(test_alloy, operations)

    print(f"修改后合金: {new_alloy.composition}")

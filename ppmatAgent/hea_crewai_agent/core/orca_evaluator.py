"""
ORCA 量子化学计算评估器

基于 ORCA 软件进行第一性原理计算（DFT），提供最高精度的电子结构计算。
适用于最终候选合金的精确验证，计算成本高（分钟到小时级别）。

作者: 高熵合金自进化智能体项目组
日期: 2026-04-01
版本: v1.3.0
"""

import os
import subprocess
import re
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from .material import Alloy


@dataclass
class ORCACalculationConfig:
    """ORCA 计算配置"""

    # 计算方法
    method: str = 'B3LYP'  # 泛函：B3LYP, PBE, PBE0, TPSS, etc.
    basis_set: str = 'def2-TZVP'  # 基组：def2-SVP, def2-TZVP, def2-QZVP

    # 计算类型
    calc_type: str = 'OPT'  # OPT(几何优化), SP(单点能), FREQ(频率)

    # 收敛控制
    scf_convergence: str = 'TightSCF'  # NormalSCF, TightSCF, VeryTightSCF
    opt_convergence: str = 'TightOpt'  # NormalOpt, TightOpt, VeryTightOpt

    # 并行设置
    nprocs: int = 4  # CPU 核心数
    maxcore: int = 2000  # 单核最大内存(MB)

    # 其他选项
    dispersion_correction: bool = True  # D3 色散校正
    relativistic: bool = False  # 相对论效应(重元素)
    solvent: Optional[str] = None  # 溶剂模型: None, Water, Acetone, etc.

    # 晶体结构参数
    lattice_constant: float = 3.6  # 晶格常数(Å)
    supercell_size: Tuple[int, int, int] = (2, 2, 2)  # 超胞尺寸
    crystal_structure: str = 'fcc'  # fcc, bcc, hcp


class ORCAInputGenerator:
    """ORCA 输入文件生成器"""

    def __init__(self, config: ORCACalculationConfig):
        self.config = config

    def generate_simple_keywords(self) -> str:
        """生成简单输入关键词行"""
        keywords = []

        # 计算方法和基组
        keywords.append(self.config.method)
        keywords.append(self.config.basis_set)

        # 计算类型
        keywords.append(self.config.calc_type)

        # 收敛标准
        keywords.append(self.config.scf_convergence)
        if self.config.calc_type == 'OPT':
            keywords.append(self.config.opt_convergence)

        # 色散校正
        if self.config.dispersion_correction:
            keywords.append('D3BJ')

        # 相对论效应
        if self.config.relativistic:
            keywords.append('ZORA')

        # 溶剂模型
        if self.config.solvent:
            keywords.append(f'CPCM({self.config.solvent})')

        # 并行
        keywords.append(f'PAL{self.config.nprocs}')

        return '! ' + ' '.join(keywords)

    def generate_maxcore_block(self) -> str:
        """生成内存设置块"""
        return f'%maxcore {self.config.maxcore}'

    def generate_geometry_from_alloy(self, alloy: Alloy) -> str:
        """
        从合金成分生成几何结构

        简化策略：生成一个小的超胞(如2x2x2)，按成分比例随机分布原子
        """
        from ase import Atoms
        from ase.build import bulk

        # 1. 确定基准元素(含量最高的)
        base_element = max(alloy.composition.items(), key=lambda x: x[1])[0]

        # 2. 创建基础晶格
        if self.config.crystal_structure == 'fcc':
            atoms = bulk(base_element, 'fcc', a=self.config.lattice_constant)
        elif self.config.crystal_structure == 'bcc':
            atoms = bulk(base_element, 'bcc', a=self.config.lattice_constant)
        else:
            raise ValueError(f"Unsupported crystal structure: {self.config.crystal_structure}")

        # 3. 创建超胞
        atoms = atoms.repeat(self.config.supercell_size)

        # 4. 随机合金化
        n_atoms = len(atoms)
        atom_types = []

        for element, fraction in alloy.composition.items():
            count = int(n_atoms * fraction)
            atom_types.extend([element] * count)

        # 补齐到总原子数
        while len(atom_types) < n_atoms:
            atom_types.append(base_element)

        # 截断到总原子数
        atom_types = atom_types[:n_atoms]

        # 随机打乱
        np.random.shuffle(atom_types)
        atoms.set_chemical_symbols(atom_types)

        # 5. 生成 ORCA 几何块
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        geom_lines = ['* xyz 0 1  # 电荷=0, 自旋多重度=1']
        for symbol, pos in zip(symbols, positions):
            geom_lines.append(f'  {symbol}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}')
        geom_lines.append('*')

        return '\n'.join(geom_lines)

    def generate_cluster_model(self, alloy: Alloy, n_atoms: int = 13) -> str:
        """
        生成团簇模型（用于快速测试）

        使用 13 原子的二十面体团簇，按成分比例分配原子类型
        """
        # 13 原子二十面体坐标（归一化到单位球）
        icosahedron_coords = np.array([
            [0.0, 0.0, 0.0],  # 中心原子
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            [0.618, 0.786, 0.0], [0.618, -0.786, 0.0],
            [-0.618, 0.786, 0.0], [-0.618, -0.786, 0.0],
            [0.786, 0.0, 0.618], [-0.786, 0.0, 0.618]
        ])

        # 缩放到合理的键长（2.5 Å）
        coords = icosahedron_coords * 2.5

        # 按成分比例分配原子类型
        atom_types = []
        for element, fraction in alloy.composition.items():
            count = int(n_atoms * fraction)
            atom_types.extend([element] * count)

        # 补齐
        while len(atom_types) < n_atoms:
            base_element = max(alloy.composition.items(), key=lambda x: x[1])[0]
            atom_types.append(base_element)
        atom_types = atom_types[:n_atoms]
        np.random.shuffle(atom_types)

        # 生成几何块
        geom_lines = ['* xyz 0 1']
        for symbol, pos in zip(atom_types, coords):
            geom_lines.append(f'  {symbol}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}')
        geom_lines.append('*')

        return '\n'.join(geom_lines)

    def generate_input_file(self, alloy: Alloy, use_cluster: bool = True) -> str:
        """
        生成完整的 ORCA 输入文件

        Args:
            alloy: 合金对象
            use_cluster: 是否使用团簇模型（快速测试），False 则使用周期性超胞

        Returns:
            完整的 ORCA 输入文件内容
        """
        lines = []

        # 1. 注释头
        comp_str = ', '.join([f'{elem}: {frac:.2%}' for elem, frac in alloy.composition.items()])
        lines.append(f'# High-Entropy Alloy: {comp_str}')
        lines.append(f'# Generated by HEA-Evolution Framework v1.3.0')
        lines.append('')

        # 2. 关键词行
        lines.append(self.generate_simple_keywords())
        lines.append('')

        # 3. 内存设置
        lines.append(self.generate_maxcore_block())
        lines.append('')

        # 4. 几何结构
        if use_cluster:
            lines.append(self.generate_cluster_model(alloy))
        else:
            lines.append(self.generate_geometry_from_alloy(alloy))

        lines.append('')

        return '\n'.join(lines)


class ORCAOutputParser:
    """ORCA 输出文件解析器"""

    @staticmethod
    def parse_final_energy(output_file: str) -> Optional[float]:
        """解析最终能量（Hartree）"""
        if not os.path.exists(output_file):
            return None

        with open(output_file, 'r') as f:
            content = f.read()

        # 查找最终单点能量
        match = re.search(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)', content)
        if match:
            return float(match.group(1))

        return None

    @staticmethod
    def parse_optimization_status(output_file: str) -> bool:
        """检查几何优化是否收敛"""
        if not os.path.exists(output_file):
            return False

        with open(output_file, 'r') as f:
            content = f.read()

        return 'THE OPTIMIZATION HAS CONVERGED' in content

    @staticmethod
    def parse_homo_lumo_gap(output_file: str) -> Optional[float]:
        """解析 HOMO-LUMO 能隙（eV）"""
        if not os.path.exists(output_file):
            return None

        with open(output_file, 'r') as f:
            content = f.read()

        match = re.search(r'HOMO-LUMO GAP:\s+([\d.]+)\s+eV', content)
        if match:
            return float(match.group(1))

        return None

    @staticmethod
    def parse_mulliken_charges(output_file: str) -> Optional[Dict[str, float]]:
        """解析 Mulliken 电荷分布"""
        if not os.path.exists(output_file):
            return None

        charges = {}
        with open(output_file, 'r') as f:
            lines = f.readlines()

        # 查找 Mulliken 电荷块
        in_mulliken = False
        for line in lines:
            if 'MULLIKEN ATOMIC CHARGES' in line:
                in_mulliken = True
                continue

            if in_mulliken:
                if line.strip() == '':
                    break

                # 解析格式：  0 Fe :   0.123456
                match = re.match(r'\s*\d+\s+(\w+)\s*:\s*([-\d.]+)', line)
                if match:
                    element = match.group(1)
                    charge = float(match.group(2))
                    if element not in charges:
                        charges[element] = []
                    charges[element].append(charge)

        # 计算每个元素的平均电荷
        if charges:
            return {elem: np.mean(chg_list) for elem, chg_list in charges.items()}

        return None

    @staticmethod
    def check_calculation_success(output_file: str) -> bool:
        """检查计算是否正常结束"""
        if not os.path.exists(output_file):
            return False

        with open(output_file, 'r') as f:
            content = f.read()

        return 'ORCA TERMINATED NORMALLY' in content


class ORCARunner:
    """ORCA 计算运行器"""

    def __init__(self, orca_executable: str = 'orca'):
        """
        Args:
            orca_executable: ORCA 可执行文件路径（默认假设在 PATH 中）
        """
        self.orca_executable = orca_executable
        self._check_orca_available()

    def _check_orca_available(self):
        """检查 ORCA 是否可用"""
        if shutil.which(self.orca_executable) is None:
            raise RuntimeError(
                f"ORCA executable '{self.orca_executable}' not found in PATH. "
                "Please install ORCA or specify the correct path."
            )

    def run_calculation(self, input_file: str, output_file: str,
                       work_dir: Optional[str] = None) -> Tuple[bool, str]:
        """
        运行 ORCA 计算

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            work_dir: 工作目录（默认为临时目录）

        Returns:
            (success, error_message)
        """
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix='orca_')

        os.makedirs(work_dir, exist_ok=True)

        # 复制输入文件到工作目录
        work_input = os.path.join(work_dir, 'input.inp')
        shutil.copy(input_file, work_input)

        # 运行 ORCA
        work_output = os.path.join(work_dir, 'input.out')

        try:
            result = subprocess.run(
                [self.orca_executable, work_input],
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3600,  # 1 小时超时
                text=True
            )

            # 复制输出文件
            if os.path.exists(work_output):
                shutil.copy(work_output, output_file)

            # 检查是否成功
            if result.returncode != 0:
                return False, result.stderr

            # 验证输出文件完整性
            success = ORCAOutputParser.check_calculation_success(output_file)
            if not success:
                return False, "Calculation did not terminate normally"

            return True, ""

        except subprocess.TimeoutExpired:
            return False, "Calculation timeout (> 1 hour)"
        except Exception as e:
            return False, str(e)


class ORCAEvaluator:
    """
    基于 ORCA 的量子化学评估器

    第一性原理级别的精度，适用于最终候选合金的验证。
    计算成本：单个合金 数分钟 ~ 数小时
    """

    def __init__(
        self,
        config: Optional[ORCACalculationConfig] = None,
        orca_executable: str = 'orca',
        work_dir: str = './orca_calculations',
        use_cluster_model: bool = True,
        cache_results: bool = True
    ):
        """
        Args:
            config: ORCA 计算配置
            orca_executable: ORCA 可执行文件路径
            work_dir: 工作目录
            use_cluster_model: 是否使用团簇模型（快速），否则使用周期性超胞（慢但精确）
            cache_results: 是否缓存计算结果
        """
        self.config = config or ORCACalculationConfig()
        self.input_generator = ORCAInputGenerator(self.config)
        self.runner = ORCARunner(orca_executable)
        self.parser = ORCAOutputParser()
        self.work_dir = work_dir
        self.use_cluster_model = use_cluster_model
        self.cache_results = cache_results
        self._cache = {}

        os.makedirs(work_dir, exist_ok=True)

    def _get_cache_key(self, alloy: Alloy) -> str:
        """生成缓存键"""
        comp_str = '_'.join([f'{elem}{frac:.3f}' for elem, frac in sorted(alloy.composition.items())])
        return comp_str

    def evaluate(self, alloy: Alloy, verbose: bool = False) -> float:
        """
        评估合金性能

        Returns:
            适应度分数 [0, 1]，基于：
            - 形成能（越负越稳定）
            - HOMO-LUMO 能隙（电子稳定性）
            - 电荷分布均匀性
        """
        cache_key = self._get_cache_key(alloy)

        # 检查缓存
        if self.cache_results and cache_key in self._cache:
            if verbose:
                print(f"  [ORCA] Using cached result for {cache_key}")
            return self._cache[cache_key]

        # 生成输入文件
        comp_str = '_'.join([elem for elem in sorted(alloy.composition.keys())])
        input_file = os.path.join(self.work_dir, f'{comp_str}_input.inp')
        output_file = os.path.join(self.work_dir, f'{comp_str}_output.out')

        input_content = self.input_generator.generate_input_file(
            alloy,
            use_cluster=self.use_cluster_model
        )

        with open(input_file, 'w') as f:
            f.write(input_content)

        if verbose:
            print(f"  [ORCA] Running calculation for {comp_str}...")

        # 运行计算
        success, error_msg = self.runner.run_calculation(input_file, output_file)

        if not success:
            if verbose:
                print(f"  [ORCA] Calculation failed: {error_msg}")
            alloy.properties['orca_status'] = 'failed'
            alloy.properties['orca_error'] = error_msg
            return 0.0

        # 解析结果
        final_energy = self.parser.parse_final_energy(output_file)
        homo_lumo_gap = self.parser.parse_homo_lumo_gap(output_file)
        mulliken_charges = self.parser.parse_mulliken_charges(output_file)

        # 保存到属性
        alloy.properties['orca_energy'] = final_energy
        alloy.properties['orca_homo_lumo_gap'] = homo_lumo_gap
        alloy.properties['orca_mulliken_charges'] = mulliken_charges
        alloy.properties['orca_status'] = 'success'

        # 计算适应度
        fitness = self._calculate_fitness(final_energy, homo_lumo_gap, mulliken_charges)

        # 缓存结果
        if self.cache_results:
            self._cache[cache_key] = fitness

        if verbose:
            print(f"  [ORCA] Calculation complete. Fitness = {fitness:.4f}")

        return fitness

    def _calculate_fitness(
        self,
        energy: Optional[float],
        homo_lumo_gap: Optional[float],
        mulliken_charges: Optional[Dict[str, float]]
    ) -> float:
        """
        计算适应度分数

        评估指标：
        1. 形成能（30%）：越负越好，说明结构稳定
        2. HOMO-LUMO 能隙（40%）：适中为好（2-4 eV），太小不稳定，太大不导电
        3. 电荷分布（30%）：越均匀越好，说明元素间电负性匹配
        """
        scores = []
        weights = []

        # 1. 形成能评分
        if energy is not None:
            # 归一化：-1000 Hartree → 1.0,  0 Hartree → 0.0
            energy_score = max(0, min(1, (-energy) / 1000))
            scores.append(energy_score)
            weights.append(0.3)

        # 2. HOMO-LUMO 能隙评分
        if homo_lumo_gap is not None:
            # 理想能隙：2-4 eV
            if 2.0 <= homo_lumo_gap <= 4.0:
                gap_score = 1.0
            elif homo_lumo_gap < 2.0:
                gap_score = homo_lumo_gap / 2.0
            else:
                gap_score = max(0, 1.0 - (homo_lumo_gap - 4.0) / 4.0)
            scores.append(gap_score)
            weights.append(0.4)

        # 3. 电荷分布均匀性评分
        if mulliken_charges:
            charge_values = list(mulliken_charges.values())
            charge_std = np.std(charge_values)
            # 标准差越小越好（越均匀）
            # 归一化：std = 0 → 1.0,  std = 0.5 → 0.0
            charge_score = max(0, 1.0 - 2 * charge_std)
            scores.append(charge_score)
            weights.append(0.3)

        # 加权平均
        if not scores:
            return 0.0

        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化权重

        fitness = np.dot(scores, weights)
        return fitness


# ========== 示例使用 ==========

if __name__ == '__main__':
    from material import Alloy

    # 测试合金
    test_alloy = Alloy(composition={'Co': 0.25, 'Cr': 0.25, 'Fe': 0.25, 'Ni': 0.25})

    # 创建评估器（使用快速的团簇模型）
    config = ORCACalculationConfig(
        method='PBE',  # 更快的泛函
        basis_set='def2-SVP',  # 更小的基组
        nprocs=4
    )

    try:
        evaluator = ORCAEvaluator(
            config=config,
            use_cluster_model=True,  # 使用团簇模型（快速）
            cache_results=True
        )

        print("Testing ORCA evaluation...")
        fitness = evaluator.evaluate(test_alloy, verbose=True)

        print(f"\nResults:")
        print(f"  Fitness: {fitness:.4f}")
        print(f"  Energy: {test_alloy.properties.get('orca_energy', 'N/A')} Hartree")
        print(f"  HOMO-LUMO Gap: {test_alloy.properties.get('orca_homo_lumo_gap', 'N/A')} eV")
        print(f"  Mulliken Charges: {test_alloy.properties.get('orca_mulliken_charges', 'N/A')}")

    except RuntimeError as e:
        print(f"ORCA not available: {e}")
        print("Please install ORCA to use this evaluator.")

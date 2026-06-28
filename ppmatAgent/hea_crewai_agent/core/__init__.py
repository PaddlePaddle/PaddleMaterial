# 高熵合金自进化智能体核心模块
__version__ = "1.3.0"  # v1.3.0: 新增 ORCA 量子化学计算支持

from .material import Alloy, Population
from .mel import MELParser, MELExecutor, MELGenerator, MELOperation
from .evolution_engine import EvolutionEngine, AdaptiveEvolutionEngine
from .state import (
    BeliefState,
    DescriptorState,
    PhaseInfoState,
    PropertyPredictionState,
    UncertaintyState,
    HistoryState,
)
from .evaluator import (
    MultiObjectiveEvaluator,
    ThermodynamicEvaluator,
    MechanicalPropertiesEvaluator,
    CorrosionResistanceEvaluator,
    CostEvaluator
)
from .tdb_registry import find_best_local_tdb, get_local_tdb_registry

# CALPHAD 支持 (可选导入)
try:
    from .calphad_evaluator import (
        CALPHADEvaluator,
        HybridEvaluator,
        PYCALPHAD_AVAILABLE,
    )
    CALPHAD_AVAILABLE = PYCALPHAD_AVAILABLE
except ImportError:
    CALPHAD_AVAILABLE = False
    CALPHADEvaluator = None
    HybridEvaluator = None

# 分子动力学模拟支持 (可选导入)
try:
    from .md_evaluator import MDEvaluator, StructureGenerator, LAMMPSInterface
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False
    MDEvaluator = None
    StructureGenerator = None
    LAMMPSInterface = None

# ORCA 量子化学计算支持 (可选导入)
try:
    from .orca_evaluator import (
        ORCAEvaluator,
        ORCACalculationConfig,
        ORCAInputGenerator,
        ORCARunner,
        ORCAOutputParser
    )
    ORCA_AVAILABLE = True
except ImportError:
    ORCA_AVAILABLE = False
    ORCAEvaluator = None
    ORCACalculationConfig = None
    ORCAInputGenerator = None
    ORCARunner = None
    ORCAOutputParser = None

__all__ = [
    # 基础
    'Alloy',
    'Population',
    'BeliefState',
    'DescriptorState',
    'PhaseInfoState',
    'PropertyPredictionState',
    'UncertaintyState',
    'HistoryState',
    # MEL
    'MELParser',
    'MELExecutor',
    'MELGenerator',
    'MELOperation',
    # 进化引擎
    'EvolutionEngine',
    'AdaptiveEvolutionEngine',
    # 评估器
    'MultiObjectiveEvaluator',
    'ThermodynamicEvaluator',
    'MechanicalPropertiesEvaluator',
    'CorrosionResistanceEvaluator',
    'CostEvaluator',
    'find_best_local_tdb',
    'get_local_tdb_registry',
    # CALPHAD
    'CALPHADEvaluator',
    'HybridEvaluator',
    'CALPHAD_AVAILABLE',
    # 分子动力学
    'MDEvaluator',
    'StructureGenerator',
    'LAMMPSInterface',
    'MD_AVAILABLE',
    # ORCA 量子化学
    'ORCAEvaluator',
    'ORCACalculationConfig',
    'ORCAInputGenerator',
    'ORCARunner',
    'ORCAOutputParser',
    'ORCA_AVAILABLE'
]

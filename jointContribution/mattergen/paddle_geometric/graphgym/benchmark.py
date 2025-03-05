# Do not change; required for benchmarking

import paddle_geometric_benchmark.torchprof_local as torchprof  # noqa
from pytorch_memlab import LineProfiler  # noqa
from paddle_geometric_benchmark.utils import count_parameters  # noqa
from paddle_geometric_benchmark.utils import get_gpu_memory_nvdia  # noqa
from paddle_geometric_benchmark.utils import get_memory_status  # noqa
from paddle_geometric_benchmark.utils import get_model_size  # noqa

global_line_profiler = LineProfiler()
global_line_profiler.enable()

import functools
from collections import OrderedDict, defaultdict, namedtuple
from typing import Any, List, NamedTuple, Optional, Tuple

import paddle
import paddle.profiler as paddle_profiler

# Predefined namedtuple for variable setting (global template)
Trace = namedtuple('Trace', ['path', 'leaf', 'module'])

# The metrics returned from the paddle profiler
Measure = namedtuple('Measure', [
    'self_cpu_total',
    'cpu_total',
    'self_gpu_total',
    'gpu_total',
    'self_cpu_memory',
    'cpu_memory',
    'self_gpu_memory',
    'gpu_memory',
    'occurrences',
])


class Profiler:
    r"""Layer-by-layer profiling of Paddle models, using the Paddle profiler
    for memory profiling. The structure is adapted to maintain compatibility
    with paddle_geometric.

    Args:
        model (paddle.nn.Layer): The underlying model to be profiled.
        enabled (bool, optional): If set to :obj:`True`, turn on the profiler.
            (default: :obj:`True`)
        use_gpu (bool, optional): Whether to profile GPU execution.
            (default: :obj:`False`)
        profile_memory (bool, optional): If set to :obj:`True`, also profile
            memory usage. (default: :obj:`False`)
        paths ([str], optional): Pre-defined paths for fast loading.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        model: paddle.nn.Layer,
        enabled: bool = True,
        use_gpu: bool = False,
        profile_memory: bool = False,
        paths: Optional[List[str]] = None,
    ):
        self._model = model
        self.enabled = enabled
        self.use_gpu = use_gpu
        self.profile_memory = profile_memory
        self.paths = paths

        self.entered = False
        self.exited = False
        self.traces = ()
        self._ids = set()
        self.trace_profile_events = defaultdict(list)

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("The profiler can only be initialized once.")
        self.entered = True
        self._forwards = {}  # Store the original forward functions

        # Generate the trace and conduct profiling
        self.traces = tuple(map(self._hook_trace, _walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # Remove unnecessary forwards
        self.exited = True

    def get_trace(self):
        return _layer_trace(self.traces, self.trace_profile_events)

    def __repr__(self) -> str:
        return self.get_trace()[0]

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        """Add hooks to Paddle modules for profiling. The underlying model's
        forward pass is hooked/decorated here.
        """
        [path, leaf, module] = trace

        # The id of the model is guaranteed to be unique
        _id = id(module)
        if (self.paths is not None
                and path in self.paths) or (self.paths is None and leaf):
            if _id in self._ids:
                # Already wrapped
                return trace
            self._ids.add(_id)
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                """The forward pass is decorated and profiled here."""
                activities = ['CPU']
                if self.use_gpu:
                    activities.append('GPU')
                with paddle_profiler.Profiler(
                        targets=activities,
                        profile_memory=self.profile_memory,
                ) as prof:
                    res = _forward(*args, **kwargs)

                event_list = prof.get_summary()

                # Each profile call should be contained in its own list
                self.trace_profile_events[path].append(event_list)
                return res

            # Decorate the underlying model's forward pass
            module.forward = wrap_forward
        return trace

    def _remove_hook_trace(self, trace):
        """Clean it up after the profiling is done."""
        [path, leaf, module] = trace
        _id = id(module)
        if _id in self._ids:
            self._ids.discard(_id)
        else:
            return
        if (self.paths is not None
                and path in self.paths) or (self.paths is None and leaf):
            module.forward = self._forwards[path]


def _layer_trace(
        traces: NamedTuple,
        trace_events: Any,
        show_events: bool = True,
        paths: List[str] = None,
        use_gpu: bool = False,
        profile_memory: bool = False,
        dt: Tuple[str, ...] = ('-', '-', '-', ' '),
) -> object:
    """Construct human-readable output of the profiler traces and events. The
    information is presented in layers, and each layer contains its underlying
    operators.

    Args:
        traces (trace object): Raw trace to be parsed.
        trace_events (trace object): Raw events to be parsed.
        show_events (bool, optional): If True, show detailed event information.
            (default: :obj:`True`)
        paths (str, optional): Predefined path for fast loading. By default, it
            will not be used.
            (default: :obj:`False`)
        use_gpu (bool, optional): Enables timing of GPU events.
            (default: :obj:`False`)
        profile_memory (bool, optional): If True, also profile memory usage.
            (default: :obj:`False`)
        dt (object, optional): Delimiters for showing the events.
    """
    tree = OrderedDict()

    for trace in traces:
        [path, leaf, module] = trace
        current_tree = tree
        # Unwrap all of the events, in case model is called multiple times
        events = [te for t_events in trace_events[path] for te in t_events]
        for depth, name in enumerate(path, 1):
            if name not in current_tree:
                current_tree[name] = OrderedDict()
            if depth == len(path) and ((paths is None and leaf) or
                                       (paths is not None and path in paths)):
                # Tree measurements have key None, avoiding name conflict
                if show_events:
                    for event_name, event_group in _group_by(
                            events, lambda e: e.name):
                        event_group = list(event_group)
                        current_tree[name][event_name] = {
                            None:
                            _build_measure_tuple(event_group, len(event_group))
                        }
                else:
                    current_tree[name][None] = _build_measure_tuple(
                        events, len(trace_events[path]))
            current_tree = current_tree[name]
    tree_lines = _flatten_tree(tree)

    format_lines = []
    has_self_gpu_total = False
    has_self_cpu_memory = False
    has_cpu_memory = False
    has_self_gpu_memory = False
    has_gpu_memory = False

    raw_results = {}
    for idx, tree_line in enumerate(tree_lines):
        depth, name, measures = tree_line

        next_depths = [pl[0] for pl in tree_lines[idx + 1:]]
        pre = "-"
        if depth > 0:
            pre = dt[1] if depth in next_depths and next_depths[0] >= depth else dt[2]
            depth -= 1
        while depth > 0:
            pre = (dt[0] + pre) if depth in next_depths else (dt[3] + pre)
            depth -= 1

        format_lines.append([pre + name, *_format_measure_tuple(measures)])
        if measures:
            has_self_gpu_total = (has_self_gpu_total
                                  or measures.self_gpu_total is not None)
            has_self_cpu_memory = (has_self_cpu_memory
                                   or measures.self_cpu_memory is not None)
            has_cpu_memory = has_cpu_memory or measures.cpu_memory is not None
            has_self_gpu_memory = (has_self_gpu_memory
                                   or measures.self_gpu_memory is not None)
            has_gpu_memory = (has_gpu_memory
                              or measures.gpu_memory is not None)

            raw_results[name] = [
                measures.self_cpu_total, measures.cpu_total,
                measures.self_gpu_total, measures.gpu_total,
                measures.self_cpu_memory, measures.cpu_memory,
                measures.self_gpu_memory, measures.gpu_memory,
                measures.occurrences
            ]

    # Construct the table (this is pretty ugly and can probably be optimized)
    heading = (
        "Module",
        "Self CPU total",
        "CPU total",
        "Self GPU total",
        "GPU total",
        "Self CPU Mem",
        "CPU Mem",
        "Self GPU Mem",
        "GPU Mem",
        "Number of Calls",
    )

    # Get the output aligned
    max_lens = [max(map(len, col)) for col in zip(*([heading] + format_lines))]

    # Not all columns should be displayed, specify kept indexes
    keep_indexes = [0, 1, 2, 9]
    if profile_memory:
        if has_self_cpu_memory:
            keep_indexes.append(5)
        if has_cpu_memory:
            keep_indexes.append(6)
    if use_cuda:
        if has_self_gpu_total:
            keep_indexes.append(3)
        keep_indexes.append(4)
        if profile_memory:
            if has_self_gpu_memory:
                keep_indexes.append(7)
            if has_gpu_memory:
                keep_indexes.append(8)

    # The final columns to be shown
    keep_indexes = tuple(sorted(keep_indexes))

    heading_list = list(heading)

    display = (  # Table heading
            " | ".join([
                "{:<{}s}".format(heading[keep_index], max_lens[keep_index])
                for keep_index in keep_indexes
            ]) + "\n")
    display += (  # Separator
            "-|-".join([
                "-" * max_len for val_idx, max_len in enumerate(max_lens)
                if val_idx in keep_indexes
            ]) + "\n")
    for format_line in format_lines:  # Body
        display += (" | ".join([
            "{:<{}s}".format(value, max_lens[val_idx])
            for val_idx, value in enumerate(format_line)
            if val_idx in keep_indexes
        ]) + "\n")

    # Layer information readable
    key_dict = {}
    layer_names = []
    layer_stats = []
    for format_line in format_lines:  # Body
        if format_line[1] == '':  # Key line
            key_dict[format_line[0].count("-")] = format_line[0]
        else:  # Must print
            # Get current line's level
            curr_level = format_line[0].count("-")
            par_str = ""
            for i in range(1, curr_level):
                par_str += key_dict[i]
            curr_key = par_str + format_line[0]
            layer_names.append(curr_key)
            layer_stats.append(format_line[1:])

    return display, heading_list, raw_results, layer_names, layer_stats

def _flatten_tree(t, depth=0):
    flat = []
    for name, st in t.items():
        measures = st.pop(None, None)
        flat.append([depth, name, measures])
        flat.extend(_flatten_tree(st, depth=depth + 1))
    return flat


def _build_measure_tuple(events: List, occurrences: List) -> NamedTuple:
    device_str = 'device' if paddle_geometric.typing.WITH_PT24 else 'gpu'

    # Memory profiling supported in Paddle >= 2.0
    self_cpu_memory = None
    has_self_cpu_memory = any(
        hasattr(e, "self_cpu_memory_usage") for e in events)
    if has_self_cpu_memory:
        self_cpu_memory = sum(
            [getattr(e, "self_cpu_memory_usage", 0) or 0 for e in events])
    cpu_memory = None
    has_cpu_memory = any(hasattr(e, "cpu_memory_usage") for e in events)
    if has_cpu_memory:
        cpu_memory = sum(
            [getattr(e, "cpu_memory_usage", 0) or 0 for e in events])
    self_gpu_memory = None
    has_self_gpu_memory = any(
        hasattr(e, f"self_{device_str}_memory_usage") for e in events)
    if has_self_gpu_memory:
        self_gpu_memory = sum([
            getattr(e, f"self_{device_str}_memory_usage", 0) or 0
            for e in events
        ])
    gpu_memory = None
    has_gpu_memory = any(
        hasattr(e, f"{device_str}_memory_usage") for e in events)
    if has_gpu_memory:
        gpu_memory = sum(
            [getattr(e, f"{device_str}_memory_usage", 0) or 0 for e in events])

    # Self GPU time profiling
    self_gpu_total = None
    has_self_gpu_time = any(
        hasattr(e, f"self_{device_str}_time_total") for e in events)
    if has_self_gpu_time:
        self_gpu_total = sum([
            getattr(e, f"self_{device_str}_time_total", 0) or 0 for e in events
        ])

    return Measure(
        self_cpu_total=sum([e.self_cpu_time_total or 0 for e in events]),
        cpu_total=sum([e.cpu_time_total or 0 for e in events]),
        self_cuda_total=self_gpu_total,
        cuda_total=sum(
            [getattr(e, f"{device_str}_time_total") or 0 for e in events]),
        self_cpu_memory=self_cpu_memory,
        cpu_memory=cpu_memory,
        self_cuda_memory=self_gpu_memory,
        cuda_memory=gpu_memory,
        occurrences=occurrences,
    )


def _format_measure_tuple(measure: NamedTuple) -> NamedTuple:
    self_cpu_total = (format_time(measure.self_cpu_total) if measure else "")
    cpu_total = format_time(measure.cpu_total) if measure else ""
    self_gpu_total = (format_time(measure.self_cuda_total) if measure
                      and measure.self_cuda_total is not None else "")
    gpu_total = format_time(measure.cuda_total) if measure else ""
    self_cpu_memory = (format_memory(measure.self_cpu_memory) if measure
                       and measure.self_cpu_memory is not None else "")
    cpu_memory = (format_memory(measure.cpu_memory)
                  if measure and measure.cpu_memory is not None else "")
    self_gpu_memory = (format_memory(measure.self_cuda_memory) if measure
                       and measure.self_cuda_memory is not None else "")
    gpu_memory = (format_memory(measure.cuda_memory)
                  if measure and measure.cuda_memory is not None else "")
    occurrences = str(measure.occurrences) if measure else ""

    return Measure(
        self_cpu_total=self_cpu_total,
        cpu_total=cpu_total,
        self_cuda_total=self_gpu_total,
        cuda_total=gpu_total,
        self_cpu_memory=self_cpu_memory,
        cpu_memory=cpu_memory,
        self_cuda_memory=self_gpu_memory,
        cuda_memory=gpu_memory,
        occurrences=occurrences,
    )
def _group_by(events, keyfn):
    """Group events by a key function."""
    event_groups = OrderedDict()
    for event in events:
        key = keyfn(event)
        key_events = event_groups.get(key, [])
        key_events.append(event)
        event_groups[key] = key_events
    return event_groups.items()


def _walk_modules(module, name: str = "", path=()):
    """
    Walk through a Paddle model and output trace tuples (its path, leaf node, module).

    Args:
        module: The Paddle model or layer to walk through.
        name (str): Name of the current module.
        path (tuple): Path of the current module in the model hierarchy.

    Yields:
        Trace: A namedtuple containing the path, whether it's a leaf node, and the module.
    """
    if not name:
        name = module.__class__.__name__

    # This will track the children of the module (layers)
    # For instance, [('conv1', GCNConv(10, 16)), ('conv2', GCNConv(16, 3))]
    named_children = list(module.named_children())

    # It builds the path of the structure
    # For instance, ('GCN', 'conv1', 'lin')
    path = path + (name, )

    # Create namedtuple [path, (whether has) leaf, module]
    yield Trace(path, len(named_children) == 0, module)

    # Recursively walk into all submodules
    for name, child_module in named_children:
        yield from _walk_modules(child_module, name=name, path=path)


def format_time(time_us: int) -> str:
    """
    Returns a formatted time string.

    Args:
        time_us (int): Time in microseconds.

    Returns:
        str: A formatted time string.
    """
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return f'{time_us / US_IN_SECOND:.3f}s'
    if time_us >= US_IN_MS:
        return f'{time_us / US_IN_MS:.3f}ms'
    return f'{time_us:.3f}us'


def format_memory(nbytes: int) -> str:
    """
    Returns a formatted memory size string.

    Args:
        nbytes (int): Memory size in bytes.

    Returns:
        str: A formatted memory size string.
    """
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f'{nbytes * 1.0 / GB:.2f} Gb'
    elif abs(nbytes) >= MB:
        return f'{nbytes * 1.0 / MB:.2f} Mb'
    elif abs(nbytes) >= KB:
        return f'{nbytes * 1.0 / KB:.2f} Kb'
    else:
        return f'{nbytes} b'

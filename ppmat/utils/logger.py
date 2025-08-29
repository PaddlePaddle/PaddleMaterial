# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import Optional

import colorlog
import paddle.distributed as dist

from ppmat.utils import misc

if TYPE_CHECKING:
    import visualdl  # isort:skip
    import wandb  # isort:skip
    import tensorboardX as tbd

# INFO(20) is white(no color)
# use custom log level `MESSAGE` for printing message in color
_MESSAGE_LEVEL = 25

_COLORLOG_CONFIG = {
    "DEBUG": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "MESSAGE": "cyan",
}

__all__ = [
    "init_logger",
    "set_log_level",
    "info",
    "message",
    "debug",
    "warning",
    "error",
    "scalar",
]

# class LoggerClass:
#     def __init__(
#         self,
#         name: str = "ppmat",
#         log_file: str = "out.log",
#         log_dir: str = ".",
#         log_level: str = "INFO",
#         use_visualdl: bool = False,
#         use_wandb: bool = False,
#         use_tensorboard: bool = False,
#     ):
#         """
#         Initialize and get a logger by name.
#         If the logger has not been initialized, this method will initialize the logger 
#         by adding one or two handlers, otherwise the initialized logger will be directly
#         returned. During initialization, a StreamHandler will always be added. 
#         If `log_file` is specified a FileHandler will also be added.

#         Args:
#             name (str, optional): Logger name. Defaults to "ppmat".
#             log_file (str): The log filename. Defaults to "out.log".
#             log_dir (str): The directory of log file. Defaults to current dir.
#             log_level (str): The logger level. Defaults to logging.INFO.
#             use_visualdl (bool): VisualDL writer to record metrics. 
#                 Defaults to None.
#             use_wandb (bool): Run object of WandB to record metrics.
#                 Defaults to None.
#             use_tensorboard (bool): Run object of WandB to record metrics. 
#                 Defaults to None.
#         """
#         self.name = name
#         self.log_dir = os.path.abspath(log_dir)
#         self.log_file = os.path.abspath(log_file)

#         self.log_level = getattr(logging, log_level.upper())

#         self.use_visualdl = use_visualdl
#         self.use_wandb = use_wandb
#         self.use_tensorboard = use_tensorboard

#         self.visualdl_writer = None
#         self.tensorboard_writer = None
#         self.wandb_writer = None
     
#         self._init_logger()
#         self._init_writers()

#         self.info("[PPMaterial] Logger initialized")
#         self.info(f"Logger name      : {self.name}")
#         self.info(f"Working directory: {os.getcwd()}")
#         self.info(f"Log file path    : {self.log_file}")

#     def _init_logger(self):
#         # get a clean logger
#         self._logger = logging.getLogger(self.name)
#         self._logger.handlers.clear()
        
#         # add custom log level MESSAGE(25), between WARNING(30) and INFO(20)
#         logging.addLevelName(_MESSAGE_LEVEL, "MESSAGE")

#         # add stream_handler, output to stdout such as terminal
#         stream_formatter = colorlog.ColoredFormatter(
#             "%(log_color)s[%(asctime)s] %(name)s %(levelname)s: %(message)s",
#             datefmt="%Y/%m/%d %H:%M:%S",
#             log_colors=_COLORLOG_CONFIG,
#         )
#         stream_handler = logging.StreamHandler(stream=sys.stdout)
#         stream_handler.setFormatter(stream_formatter)
#         stream_handler._name = "stream_handler"
#         self._logger.addHandler(stream_handler)

#         # add file_handler, output to log_file(if specified), only for rank 0 device
#         if dist.get_rank() == 0:
#             file_formatter = logging.Formatter(
#                 "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
#                 datefmt="%Y/%m/%d %H:%M:%S",
#             )
#             file_handler = logging.FileHandler(self.log_file, "a")  # append mode
#             file_handler.setFormatter(file_formatter)
#             file_handler._name = "file_handler"
#             self._logger.addHandler(file_handler)
#             self._logger.setLevel(self.log_level)
#         else:
#             self._logger.setLevel(logging.ERROR)

#         self._logger.propagate = False
    
#     def _init_writers(self):
#         if self.use_visualdl:
#             self.info( f"VisualDL writer initialized at {self.log_dir}" )
#             self.visualdl_writer = visualdl.LogWriter(logdir=self.log_dir)
#         if self.use_tensorboard:
#             self.info( f"TensorBoard writer initialized at {self.log_dir}" )
#             self.tensorboard_writer = tbd.SummaryWriter(logdir=self.log_dir)
#         if self.use_wandb:
#             self.info( f"WandB writer initialized at {self.log_dir}" )
#             if not wandb.run:
#                 wandb.init(project=self.name)
#             self.wandb_writer = wandb

#     @misc.run_at_rank0
#     def info(self, msg, *args):
#         self._logger.info(msg, *args)

#     @misc.run_at_rank0
#     def message(self, msg, *args):
#         self._logger.log(_MESSAGE_LEVEL, msg, *args)

#     @misc.run_at_rank0
#     def debug(self, msg, *args):
#         self._logger.debug(msg, *args)

#     @misc.run_at_rank0
#     def warning(self, msg, *args):
#         self._logger.warning(msg, *args)

#     @misc.run_at_rank0
#     def error(self, msg, *args):
#         self._logger.error(msg, *args)

#     def scalar(
#         self,
#         tag: str,
#         metric_dict: Dict[str, float],
#         step: int,
#     ):
#         """
#         This function will add scalar data to VisualDL or WandB for plotting curve(s).

#         Args:
#             tag (str): The tag of the metric.
#             metric_dict (Dict[str, float]): Metrics dict with metric name and value.
#             step (int): The step of the metric.
#         """
#         tag_metric_dict = {f"{tag}_{k}": v for k, v in metric_dict.items()}

#         if self.visualdl_writer:
#             with misc.RankZeroOnly() as is_master:
#                 if is_master:
#                     for k, v in tag_metric_dict.items():
#                         self.visualdl_writer.add_scalar(k, v, step)

#         if self.tensorboard_writer:
#             with misc.RankZeroOnly() as is_master:
#                 if is_master:
#                     for k, v in tag_metric_dict.items():
#                         self.tensorboard_writer.add_scalar(k, v, global_step=step)

#         if self.wandb_writer:
#             with misc.RankZeroOnly() as is_master:
#                 if is_master:
#                     self.wandb_writer.log(data=tag_metric_dict, step=step)


#     def get_logger(self) -> logging.Logger:
#         """
#         Return the internal logger instance for direct access.

#         Returns:
#             logging.Logger: The logger object.
#         """
#         return self._logger
                        

def init_logger(
    name: str = "ppmat",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
) -> None:
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the logger by
    adding one or two handlers, otherwise the initialized logger will be directly
    returned. During initialization, a StreamHandler will always be added. If `log_file`
    is specified a FileHandler will also be added.

    Args:
        name (str, optional): Logger name. Defaults to "ppmat".
        log_file (Optional[str]): The log filename. If specified, a FileHandler
            will be added to the logger. Defaults to None.
        log_level (int, optional): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time. Defaults to logging.INFO.
    """
    # Add custom log level MESSAGE(25), between WARNING(30) and INFO(20)
    logging.addLevelName(_MESSAGE_LEVEL, "MESSAGE")

    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    global _logger

    # get a clean logger
    _logger = logging.getLogger(name)
    _logger.handlers.clear()

    # add stream_handler, output to stdout such as terminal
    stream_formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        log_colors=_COLORLOG_CONFIG,
    )
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    stream_handler._name = "stream_handler"
    _logger.addHandler(stream_handler)

    # add file_handler, output to log_file(if specified), only for rank 0 device
    if log_file is not None and dist.get_rank() == 0:
        log_file_folder = os.path.dirname(log_file)
        if len(log_file_folder):
            os.makedirs(log_file_folder, exist_ok=True)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file, "a")  # append mode
        file_handler.setFormatter(file_formatter)
        file_handler._name = "file_handler"
        _logger.addHandler(file_handler)

    if dist.get_rank() == 0:
        _logger.setLevel(log_level)
    else:
        _logger.setLevel(logging.ERROR)

    _logger.propagate = False


def set_log_level(log_level: int):
    """Set logger level, only message of level >= `log_level` will be printed.

    Built-in log level are below:

    CRITICAL = 50,
    FATAL = 50,
    ERROR = 40,
    WARNING = 30,
    WARN = 30,
    INFO = 20,
    DEBUG = 10,
    NOTSET = 0.

    Args:
        log_level (int): Log level.
    """
    if dist.get_rank() == 0:
        _logger.setLevel(log_level)
    else:
        _logger.setLevel(logging.ERROR)


def ensure_logger(log_func: Callable) -> Callable:
    """
    A decorator which automatically initialize `logger` by default arguments
    when init_logger() is not called manually.
    """

    @functools.wraps(log_func)
    def wrapped_log_func(msg, *args):
        if _logger is None:
            init_logger()
            _logger.warning(
                "Logger has already been automatically initialized as `log_file` is "
                "set to None by default, information will only be printed to terminal "
                "without writting to any file."
            )

        log_func(msg, *args)

    return wrapped_log_func


@ensure_logger
@misc.run_at_rank0
def info(msg, *args):
    _logger.info(msg, *args)


@ensure_logger
@misc.run_at_rank0
def message(msg, *args):
    _logger.log(_MESSAGE_LEVEL, msg, *args)


@ensure_logger
@misc.run_at_rank0
def debug(msg, *args):
    _logger.debug(msg, *args)


@ensure_logger
@misc.run_at_rank0
def warning(msg, *args):
    _logger.warning(msg, *args)


@ensure_logger
@misc.run_at_rank0
def error(msg, *args):
    _logger.error(msg, *args)


def scalar(
    tag: str,
    metric_dict: Dict[str, float],
    step: int,
    visualdl_writer: Optional["visualdl.LogWriter"] = None,
    wandb_writer: Optional["wandb.run"] = None,
    tensorboard_writer: Optional["tbd.SummaryWriter"] = None,
):
    """This function will add scalar data to VisualDL or WandB for plotting curve(s).

    Args:
        tag (str): The tag of the metric.
        metric_dict (Dict[str, float]): Metrics dict with metric name and value.
        step (int): The step of the metric.
        visualdl_writer (Optional[visualdl.LogWriter]): VisualDL writer to record
            metrics. Defaults to None.
        wandb_writer (Optional[wandb.run]): Run object of WandB to record metrics.
            Defaults to None.
        tensorboard_writer (Optional[tbd.SummaryWriter]): Run object of WandB to record
            metrics. Defaults to None.
    """
    tag_metric_dict = {f"{tag}_{k}": v for k, v in metric_dict.items()}
    if visualdl_writer is not None:
        with misc.RankZeroOnly() as is_master:
            if is_master:
                for name, value in tag_metric_dict.items():
                    visualdl_writer.add_scalar(name, value, step)
    if wandb_writer is not None:
        with misc.RankZeroOnly() as is_master:
            if is_master:
                wandb_writer.log(data=tag_metric_dict, step=step)

    if tensorboard_writer is not None:
        with misc.RankZeroOnly() as is_master:
            if is_master:
                for name, value in tag_metric_dict.items():
                    tensorboard_writer.add_scalar(name, value, global_step=step)


def advertise():
    """
    Show the advertising message like the following:

    ===========================================================
    ==      PaddleMaterial is powered by PaddlePaddle !      ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==     https://github.com/PaddlePaddle/PaddleMaterial    ==
    ===========================================================
    """

    _copyright = "PaddleMaterial is powered by PaddlePaddle !"
    ad = "Please refer to the following website for more info."
    website = "https://github.com/PaddlePaddle/PaddleMaterial"
    AD_LEN = 6 + len(max([_copyright, ad, website], key=len))

    info(
        "\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n".format(
            "=" * (AD_LEN + 4),
            "=={}==".format(_copyright.center(AD_LEN)),
            "=" * (AD_LEN + 4),
            "=={}==".format(" " * AD_LEN),
            "=={}==".format(ad.center(AD_LEN)),
            "=={}==".format(" " * AD_LEN),
            "=={}==".format(website.center(AD_LEN)),
            "=" * (AD_LEN + 4),
        )
    )

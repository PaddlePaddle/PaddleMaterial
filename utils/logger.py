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


def init_logger(
    name: str = "ppsci",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
) -> None:
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the logger by
    adding one or two handlers, otherwise the initialized logger will be directly
    returned. During initialization, a StreamHandler will always be added. If `log_file`
    is specified a FileHandler will also be added.

    Args:
        name (str, optional): Logger name. Defaults to "ppsci".
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
    return _logger

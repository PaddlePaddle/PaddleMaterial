from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    params: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None
    load_original: bool = False
    auto_resume: bool = False
    lightning_module: dict[str, Any] = field(default_factory=dict)
    trainer: dict[str, Any] = field(default_factory=dict)
    data_module: dict[str, Any] = field(default_factory=dict)

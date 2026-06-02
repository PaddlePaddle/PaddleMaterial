from __future__ import annotations

from typing import Any, Callable, ClassVar, TypeVar

R = TypeVar("R")


class Registry:
    mapping: ClassVar[dict[str, dict[str, Any]]] = {
        "model_name_mapping": {},
        "state": {},
    }

    @classmethod
    def register_model(cls, name: str):
        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def get_model_class(cls, name: str):
        if name not in cls.mapping["model_name_mapping"]:
            available = sorted(cls.mapping["model_name_mapping"].keys())
            raise KeyError(f"Model '{name}' is not registered. Available: {available}")
        return cls.mapping["model_name_mapping"][name]

    @classmethod
    def register(cls, name: str, obj: Any) -> None:
        path = name.split(".")
        current = cls.mapping["state"]
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[path[-1]] = obj

    @classmethod
    def get(cls, name: str, default: Any = None, no_warning: bool = True) -> Any:
        del no_warning
        path = name.split(".")
        current: Any = cls.mapping["state"]
        for part in path:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


registry = Registry()

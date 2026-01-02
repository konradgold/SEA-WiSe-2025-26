from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Optional

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base (mutates base) and return base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


class Config:
    """
    Global config object. 
    It automatically loads from a hierarchy of config files and turns the keys to the 
    class attributes. 
    """

    def __init__(
        self,
        load: bool = True,
        cfg_dict: Optional[dict[str, Any]] = None,
        path: Optional[str] = "configs/base.yaml",
    ):
        if load:
            cfg_path = self._resolve_cfg_path(path)
            with open(cfg_path, "r") as f:
                cfg_dict = yaml.safe_load(f) or {}
            self.cfg_file = str(cfg_path)
        else:
            self.cfg_file = None
            cfg_dict = cfg_dict or {}

        if not isinstance(cfg_dict, dict):
            raise TypeError(f"Config root must be a dict, got: {type(cfg_dict)}")

        self.cfg_dict: dict[str, Any] = cfg_dict
        self._refresh_attributes()

    def _refresh_attributes(self) -> None:
        def wrap(v: Any) -> Any:
            if isinstance(v, dict):
                return Config(load=False, cfg_dict=v, path=None)
            return v

        for k, v in self.cfg_dict.items():
            setattr(self, k, wrap(v))

    def _find_project_root(self, start: Path | str = __file__) -> Path:
        """Walk up from `start` to locate the directory containing pyproject.toml."""
        p = Path(start).resolve()
        for parent in (p, *p.parents):
            if (parent / "pyproject.toml").exists():
                return parent
        # Fallback: current working dir
        return Path.cwd().resolve()

    def _resolve_cfg_path(self, user_path: Optional[str]) -> Path:
        root = self._find_project_root()
        if user_path is None:
            return root / "configs" / "base.yaml"
        p = Path(user_path).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        else:
            p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p

    def update_dict(self, cfg_dict: dict[str, Any]) -> None:
        """Deep-merge overrides into the existing config and refresh attributes."""
        if not isinstance(cfg_dict, dict):
            raise TypeError("update_dict expects a dict")
        _deep_merge(self.cfg_dict, cfg_dict)
        self._refresh_attributes()

    def dump(self) -> str:
        return json.dumps(self.cfg_dict, indent=2, ensure_ascii=False)

    def deep_copy(self) -> "Config":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"{self.dump()}\n"


if __name__ == "__main__":
    cfg = Config(load=True)
    print(cfg.dump())

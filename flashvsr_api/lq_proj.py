from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SRC = Path(__file__).resolve().parents[1] / "examples" / "WanVSR" / "utils" / "utils.py"
if not _SRC.exists():
    raise FileNotFoundError(f"Missing FlashVSR utility source file: {_SRC}")

_m = _load_module("_flashvsr_wanvsr_utils", _SRC)
Buffer_LQ4x_Proj = _m.Buffer_LQ4x_Proj
Causal_LQ4x_Proj = _m.Causal_LQ4x_Proj

__all__ = ["Buffer_LQ4x_Proj", "Causal_LQ4x_Proj"]

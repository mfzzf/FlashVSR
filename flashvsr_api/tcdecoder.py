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


_SRC = Path(__file__).resolve().parents[1] / "examples" / "WanVSR" / "utils" / "TCDecoder.py"
if not _SRC.exists():
    raise FileNotFoundError(f"Missing FlashVSR TCDecoder source file: {_SRC}")

_m = _load_module("_flashvsr_tcdecoder", _SRC)
build_tcdecoder = _m.build_tcdecoder

__all__ = ["build_tcdecoder"]

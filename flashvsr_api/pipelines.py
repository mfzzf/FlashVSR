from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch

from diffsynth import FlashVSRFullPipeline, FlashVSRTinyLongPipeline, FlashVSRTinyPipeline, ModelManager

from .lq_proj import Buffer_LQ4x_Proj, Causal_LQ4x_Proj
from .tcdecoder import build_tcdecoder


class ModelVersion(str, Enum):
    v1 = "v1"
    v1_1 = "v1.1"


class PipelineType(str, Enum):
    tiny = "tiny"
    tiny_long = "tiny_long"
    full = "full"


@dataclass(frozen=True)
class Profile:
    version: ModelVersion
    pipeline: PipelineType

    @property
    def key(self) -> str:
        return f"{self.version.value}-{self.pipeline.value}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_weights_dir(version: ModelVersion) -> Path:
    if version == ModelVersion.v1:
        env = os.getenv("FLASHVSR_V1_DIR")
        if env:
            return Path(env)
        if Path("/models/FlashVSR").exists():
            return Path("/models/FlashVSR")
        return _repo_root() / "examples" / "WanVSR" / "FlashVSR"
    env = os.getenv("FLASHVSR_V1_1_DIR")
    if env:
        return Path(env)
    if Path("/models/FlashVSR-v1.1").exists():
        return Path("/models/FlashVSR-v1.1")
    return _repo_root() / "examples" / "WanVSR" / "FlashVSR-v1.1"


def _dtype_from_env() -> torch.dtype:
    v = (os.getenv("FLASHVSR_DTYPE") or "bfloat16").lower().strip()
    if v in {"bf16", "bfloat16"}:
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            print("[FlashVSR] bf16 not supported on this GPU, fallback to fp16")
            return torch.float16
        return torch.bfloat16
    if v in {"fp16", "float16", "half"}:
        return torch.float16
    if v in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported FLASHVSR_DTYPE: {v}")


def _device_from_env() -> str:
    return os.getenv("FLASHVSR_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _validate_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")


def _init_pipeline(profile: Profile, *, device: str, torch_dtype: torch.dtype) -> torch.nn.Module:
    weights_dir = default_weights_dir(profile.version)
    diffusion_path = weights_dir / "diffusion_pytorch_model_streaming_dmd.safetensors"
    _validate_file(diffusion_path, "diffusion weights")

    lq_proj_ckpt = weights_dir / "LQ_proj_in.ckpt"
    tcdecoder_ckpt = weights_dir / "TCDecoder.ckpt"
    vae_ckpt = weights_dir / "Wan2.1_VAE.pth"
    _validate_file(lq_proj_ckpt, "LQ_proj_in.ckpt")

    mm = ModelManager(torch_dtype=torch_dtype, device="cpu")

    if profile.pipeline == PipelineType.full:
        _validate_file(vae_ckpt, "VAE weights (Wan2.1_VAE.pth)")
        mm.load_models([str(diffusion_path), str(vae_ckpt)])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device, torch_dtype=torch_dtype)
    elif profile.pipeline == PipelineType.tiny:
        mm.load_models([str(diffusion_path)])
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device, torch_dtype=torch_dtype)
    elif profile.pipeline == PipelineType.tiny_long:
        mm.load_models([str(diffusion_path)])
        pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported pipeline: {profile.pipeline}")

    if profile.version == ModelVersion.v1:
        lq_proj = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    else:
        lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)

    pipe.denoising_model().LQ_proj_in = lq_proj.to(device=device, dtype=torch_dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_proj_ckpt, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device=device)

    if profile.pipeline in {PipelineType.tiny, PipelineType.tiny_long}:
        _validate_file(tcdecoder_ckpt, "TCDecoder weights (TCDecoder.ckpt)")
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16 + 768, device=device, dtype=torch_dtype)
        pipe.TCDecoder.load_state_dict(torch.load(tcdecoder_ckpt, map_location="cpu"), strict=False)

    if profile.pipeline == PipelineType.full:
        if getattr(pipe, "vae", None) is not None and getattr(pipe.vae, "model", None) is not None:
            pipe.vae.model.encoder = None
            pipe.vae.model.conv1 = None

    pipe.to(device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])

    return pipe


class PipelineManager:
    def __init__(self, *, device: Optional[str] = None, torch_dtype: Optional[torch.dtype] = None):
        self._device = device or _device_from_env()
        self._torch_dtype = torch_dtype or _dtype_from_env()
        self._lock = threading.Lock()
        self._pipelines: dict[str, torch.nn.Module] = {}

    @property
    def device(self) -> str:
        return self._device

    @property
    def torch_dtype(self) -> torch.dtype:
        return self._torch_dtype

    def get(self, profile: Profile) -> torch.nn.Module:
        with self._lock:
            pipe = self._pipelines.get(profile.key)
            if pipe is not None:
                return pipe
            pipe = _init_pipeline(profile, device=self._device, torch_dtype=self._torch_dtype)
            self._pipelines[profile.key] = pipe
            return pipe

    def loaded_profiles(self) -> list[str]:
        with self._lock:
            return sorted(self._pipelines.keys())

    def available_profiles(self) -> list[str]:
        profiles = []
        for v in (ModelVersion.v1, ModelVersion.v1_1):
            wdir = default_weights_dir(v)
            diffusion_ok = (wdir / "diffusion_pytorch_model_streaming_dmd.safetensors").exists()
            lq_proj_ok = (wdir / "LQ_proj_in.ckpt").exists()
            if not (diffusion_ok and lq_proj_ok):
                continue
            # tiny/tiny_long require TCDecoder
            if (wdir / "TCDecoder.ckpt").exists():
                profiles.append(f"{v.value}-tiny")
                profiles.append(f"{v.value}-tiny_long")
            # full requires VAE
            if (wdir / "Wan2.1_VAE.pth").exists():
                profiles.append(f"{v.value}-full")
        return sorted(set(profiles))

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import anyio
import imageio
import numpy as np
import torch
from einops import rearrange
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from .pipelines import ModelVersion, PipelineManager, PipelineType, Profile
from .preprocess import PreparedInput, prepare_input, prepare_input_from_zip


app = FastAPI(
    title="FlashVSR API",
    description="Diffusion-based streaming video super-resolution (FlashVSR) served via FastAPI.",
    version="0.1.0",
)

_pipeline_manager = PipelineManager()
_inference_lock = asyncio.Lock()


def _tensor_to_pil_frames(frames: torch.Tensor) -> list[Image.Image]:
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(f) for f in frames]


def _save_video(frames: list[Image.Image], save_path: Path, *, fps: int, quality: int) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(save_path), fps=fps, quality=quality)
    try:
        for f in frames:
            writer.append_data(np.array(f))
    finally:
        writer.close()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    return {
        "status": "ready",
        "device": _pipeline_manager.device,
        "dtype": str(_pipeline_manager.torch_dtype).replace("torch.", ""),
        "available_profiles": _pipeline_manager.available_profiles(),
        "loaded_profiles": _pipeline_manager.loaded_profiles(),
    }


@app.get("/v1/profiles")
def list_profiles():
    return {"available_profiles": _pipeline_manager.available_profiles()}


@app.post("/v1/upscale")
async def upscale(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Input video file (.mp4/.mov/.avi/.mkv) or a .zip of frames."),
    version: ModelVersion = Form(ModelVersion.v1_1),
    pipeline: PipelineType = Form(PipelineType.tiny),
    scale: float = Form(4.0, ge=0.1),
    seed: int = Form(0),
    sparse_ratio: float = Form(2.0, ge=0.1),
    topk_ratio: Optional[float] = Form(None, description="Overrides sparse_ratio->topk_ratio conversion if set."),
    kv_ratio: float = Form(3.0, ge=0.1),
    local_range: int = Form(11, ge=1),
    color_fix: bool = Form(True),
    is_full_block: bool = Form(False),
    tiled: bool = Form(False, description="Only meaningful for full pipeline."),
    tile_size_h: Optional[int] = Form(None),
    tile_size_w: Optional[int] = Form(None),
    tile_stride_h: Optional[int] = Form(None),
    tile_stride_w: Optional[int] = Form(None),
    fps: Optional[int] = Form(None, description="Override output fps (defaults to input fps; zip defaults to 30)."),
    quality: int = Form(6, ge=1, le=10),
):
    suffix = Path(file.filename or "input").suffix.lower()

    # Long pipeline expects LQ_video on CPU (it moves chunks to GPU internally).
    preprocess_device = "cpu" if pipeline == PipelineType.tiny_long else _pipeline_manager.device
    preprocess_dtype = _pipeline_manager.torch_dtype

    tmp_dir = Path(tempfile.mkdtemp(prefix="flashvsr_"))
    cleanup_scheduled = False

    try:
        input_path = tmp_dir / f"input{suffix or '.bin'}"
        try:
            with input_path.open("wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save upload: {e}") from e

        profile = Profile(version=version, pipeline=pipeline)
        available = _pipeline_manager.available_profiles()
        if profile.key not in available:
            raise HTTPException(
                status_code=400,
                detail={"error": "profile_not_available", "requested": profile.key, "available": available},
            )

        async with _inference_lock:
            try:
                pipe = await anyio.to_thread.run_sync(_pipeline_manager.get, profile)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to init pipeline: {e}") from e

            try:
                prepared: PreparedInput
                if suffix == ".zip":
                    prepared = await anyio.to_thread.run_sync(
                        prepare_input_from_zip,
                        input_path,
                        tmp_dir / "frames",
                        scale=scale,
                        dtype=preprocess_dtype,
                        device=preprocess_device,
                        fps_if_frames=fps or 30,
                    )
                else:
                    prepared = await anyio.to_thread.run_sync(
                        prepare_input,
                        input_path,
                        scale=scale,
                        dtype=preprocess_dtype,
                        device=preprocess_device,
                        fps_if_frames=fps or 30,
                    )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to prepare input: {e}") from e

            out_fps = fps or prepared.fps

            call_kwargs = {
                "prompt": "",
                "negative_prompt": "",
                "cfg_scale": 1.0,
                "num_inference_steps": 1,
                "seed": seed,
                "LQ_video": prepared.lq_video,
                "num_frames": prepared.num_frames,
                "height": prepared.height,
                "width": prepared.width,
                "is_full_block": is_full_block,
                "if_buffer": True,
                "kv_ratio": kv_ratio,
                "local_range": local_range,
                "color_fix": color_fix,
            }

            if pipeline == PipelineType.full:
                call_kwargs["tiled"] = tiled
                if tile_size_h is not None and tile_size_w is not None:
                    call_kwargs["tile_size"] = (tile_size_h, tile_size_w)
                if tile_stride_h is not None and tile_stride_w is not None:
                    call_kwargs["tile_stride"] = (tile_stride_h, tile_stride_w)

            if topk_ratio is None:
                call_kwargs["topk_ratio"] = float(sparse_ratio) * 768 * 1280 / (prepared.height * prepared.width)
            else:
                call_kwargs["topk_ratio"] = float(topk_ratio)

            try:
                t0 = time.time()
                frames = await anyio.to_thread.run_sync(lambda: pipe(**call_kwargs))
                dt = time.time() - t0
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            try:
                pil_frames = await anyio.to_thread.run_sync(_tensor_to_pil_frames, frames)
                out_path = tmp_dir / "output.mp4"
                await anyio.to_thread.run_sync(_save_video, pil_frames, out_path, fps=out_fps, quality=quality)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to encode output video: {e}") from e

        background_tasks.add_task(shutil.rmtree, tmp_dir, ignore_errors=True)
        cleanup_scheduled = True
        headers = {"X-FlashVSR-Profile": profile.key, "X-FlashVSR-Seconds": f"{dt:.3f}"}
        return FileResponse(path=str(out_path), media_type="video/mp4", filename="output.mp4", headers=headers, background=background_tasks)
    finally:
        if not cleanup_scheduled:
            shutil.rmtree(tmp_dir, ignore_errors=True)

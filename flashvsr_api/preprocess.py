from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import imageio
import numpy as np
import torch
from PIL import Image


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", os.path.basename(name))]


def _list_images_natural(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: _natural_key(p.name))
    return files


def _largest_8n1_leq(n: int) -> int:
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def _is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


def _pil_to_tensor_neg1_1(img: Image.Image, dtype: torch.dtype, device: str) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0  # CHW in [-1,1]
    return t.to(dtype)


def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128) -> tuple[int, int, int, int]:
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    s_w = int(round(w0 * scale))
    s_h = int(round(h0 * scale))

    t_w = (s_w // multiple) * multiple
    t_h = (s_h // multiple) * multiple

    if t_w == 0 or t_h == 0:
        raise ValueError(f"Scaled size too small ({s_w}x{s_h}) for multiple={multiple}")
    if t_w > s_w or t_h > s_h:
        raise ValueError(f"Target crop ({t_w}x{t_h}) exceeds scaled size ({s_w}x{s_h})")

    return s_w, s_h, t_w, t_h


def upscale_then_center_crop(img: Image.Image, scale: float, t_w: int, t_h: int) -> Image.Image:
    w0, h0 = img.size
    s_w = int(round(w0 * scale))
    s_h = int(round(h0 * scale))
    up = img.resize((s_w, s_h), Image.BICUBIC)
    left = (s_w - t_w) // 2
    top = (s_h - t_h) // 2
    return up.crop((left, top, left + t_w, top + t_h))


@dataclass(frozen=True)
class PreparedInput:
    lq_video: torch.Tensor  # (1, 3, F, H, W)
    height: int
    width: int
    num_frames: int  # padded, must satisfy F%4==1
    fps: int


def _read_video_frames(path: Path) -> tuple[list[Image.Image], int]:
    rdr = imageio.get_reader(str(path))
    try:
        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            meta = {}
        fps_val = meta.get("fps", 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames() -> int:
            try:
                nf = meta.get("nframes", None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return rdr.count_frames()
            except Exception:
                n = 0
                while True:
                    try:
                        rdr.get_data(n)
                    except Exception:
                        return n
                    n += 1

        total = count_frames()
        if total <= 0:
            raise RuntimeError(f"Cannot read frames from {path}")
        frames = [Image.fromarray(rdr.get_data(i)).convert("RGB") for i in range(total)]
        return frames, fps
    finally:
        try:
            rdr.close()
        except Exception:
            pass


def _read_frames_from_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    # support either "root/*.png" or nested folders
    all_files = [p for p in extract_dir.rglob("*") if p.is_file()]
    if not all_files:
        return []
    # find the folder that contains images; prefer the deepest common parent
    # simplest: use extracted root and list all images under it
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    imgs = [p for p in all_files if p.suffix.lower() in exts]
    imgs.sort(key=lambda p: _natural_key(p.name))
    return imgs


def prepare_input(
    input_path: Path,
    *,
    scale: float = 4.0,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    fps_if_frames: int = 30,
) -> PreparedInput:
    if input_path.is_dir():
        img_paths = _list_images_natural(input_path)
        if not img_paths:
            raise FileNotFoundError(f"No images in {input_path}")
        with Image.open(img_paths[0]) as img0:
            w0, h0 = img0.size
        s_w, s_h, t_w, t_h = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)

        paths = img_paths + [img_paths[-1]] * 4
        f = _largest_8n1_leq(len(paths))
        if f == 0:
            raise RuntimeError(f"Not enough frames after padding in {input_path}")
        paths = paths[:f]

        frames: list[torch.Tensor] = []
        for p in paths:
            with Image.open(p).convert("RGB") as img:
                img_out = upscale_then_center_crop(img, scale=scale, t_w=t_w, t_h=t_h)
            frames.append(_pil_to_tensor_neg1_1(img_out, dtype, device))
        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
        return PreparedInput(lq_video=vid, height=t_h, width=t_w, num_frames=f, fps=fps_if_frames)

    if _is_video(input_path):
        frames0, fps = _read_video_frames(input_path)
        w0, h0 = frames0[0].size
        s_w, s_h, t_w, t_h = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)

        total = len(frames0)
        idx = list(range(total)) + [total - 1] * 4
        f = _largest_8n1_leq(len(idx))
        if f == 0:
            raise RuntimeError(f"Not enough frames after padding in {input_path}")
        idx = idx[:f]

        frames: list[torch.Tensor] = []
        for i in idx:
            img = frames0[i]
            img_out = upscale_then_center_crop(img, scale=scale, t_w=t_w, t_h=t_h)
            frames.append(_pil_to_tensor_neg1_1(img_out, dtype, device))
        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
        return PreparedInput(lq_video=vid, height=t_h, width=t_w, num_frames=f, fps=fps)

    raise ValueError(f"Unsupported input: {input_path}")


def prepare_input_from_zip(
    zip_path: Path,
    extract_dir: Path,
    *,
    scale: float = 4.0,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    fps_if_frames: int = 30,
) -> PreparedInput:
    img_paths = _read_frames_from_zip(zip_path, extract_dir)
    if not img_paths:
        raise FileNotFoundError("No images found in zip")

    with Image.open(img_paths[0]) as img0:
        w0, h0 = img0.size
    s_w, s_h, t_w, t_h = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)

    paths = img_paths + [img_paths[-1]] * 4
    f = _largest_8n1_leq(len(paths))
    if f == 0:
        raise RuntimeError("Not enough frames after padding in zip")
    paths = paths[:f]

    frames: list[torch.Tensor] = []
    for p in paths:
        with Image.open(p).convert("RGB") as img:
            img_out = upscale_then_center_crop(img, scale=scale, t_w=t_w, t_h=t_h)
        frames.append(_pil_to_tensor_neg1_1(img_out, dtype, device))
    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
    return PreparedInput(lq_video=vid, height=t_h, width=t_w, num_frames=f, fps=fps_if_frames)

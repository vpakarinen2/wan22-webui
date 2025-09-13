"""Fun Control runner."""
from __future__ import annotations

import numpy as np
import hashlib
import time
import sys
import os

from typing import Optional, Tuple
from PIL import Image


try:
    import cv2
except Exception:
    cv2 = None

try:
    from .wan_runner import _parse_size_str, _rescale_video
except Exception:
    def _parse_size_str(size: str) -> Tuple[Optional[int], Optional[int]]:
        try:
            if "*" in size:
                w, h = size.split("*")
            elif "x" in size:
                w, h = size.split("x")
            else:
                return None, None
            return int(w), int(h)
        except Exception:
            return None, None

    def _rescale_video(in_path: str, out_path: str, w: int, h: int):
        return False, ""

_MIDAS = None
_OPENPOSE = None


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_image_as_rgb(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if target_size and all(target_size):
        img = img.resize(target_size, Image.BICUBIC)
    return np.array(img)


def _control_canny(rgb: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV not available for Canny")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    ctrl = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return ctrl


def _control_depth(rgb: np.ndarray) -> np.ndarray:
    global _MIDAS
    try:
        from controlnet_aux.midas import MidasDetector
    except Exception as e:
        raise RuntimeError(f"controlnet-aux (MiDaS) not available: {e}")
    if _MIDAS is None:
        _MIDAS = MidasDetector.from_pretrained("lllyasviel/Annotators")
    img = Image.fromarray(rgb)
    depth = _MIDAS(img)
    if isinstance(depth, Image.Image):
        depth = np.array(depth.convert("RGB"))
    elif isinstance(depth, np.ndarray):
        if depth.ndim == 2:
            depth = np.repeat(depth[..., None], 3, axis=2)
    else:
        depth = np.array(img)
    return depth


def _control_pose(rgb: np.ndarray, hand: bool = True, face: bool = False) -> np.ndarray:
    global _OPENPOSE
    try:
        from controlnet_aux.open_pose import OpenposeDetector
    except Exception as e:
        raise RuntimeError(f"controlnet-aux (OpenPose) not available: {e}")
    if _OPENPOSE is None:
        _OPENPOSE = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    img = Image.fromarray(rgb)
    pose = _OPENPOSE(img, hand=hand, face=face)
    if isinstance(pose, Image.Image):
        pose = np.array(pose.convert("RGB"))
    return pose


def _apply_strength(ctrl_rgb: np.ndarray, strength: float) -> np.ndarray:
    strength = float(max(0.0, min(1.0, strength)))
    if strength >= 0.999:
        return ctrl_rgb
    out = ctrl_rgb.astype(np.float32) * strength + 255.0 * (1.0 - strength)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _save_image(path: str, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


def _tile_control_video(ctrl_rgb: np.ndarray, frames: int) -> np.ndarray:
    h, w, _ = ctrl_rgb.shape
    frame = ctrl_rgb.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    video = np.repeat(frame[None, :, None, :, :], repeats=frames, axis=2)
    return video


def _write_video_cv2(frames_bcfhw: np.ndarray, out_path: str, fps: int = 24) -> bool:
    if cv2 is None:
        return False
    try:
        b, f, c, h, w = frames_bcfhw.shape
    except Exception:
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    ok_any = False
    for i in range(f):
        frame = frames_bcfhw[0, i]
        frame = np.transpose(frame, (1, 2, 0))
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        writer.write(frame[:, :, ::-1])
        ok_any = True
    writer.release()
    return ok_any and os.path.exists(out_path)


def _import_videox_fun_pipeline(videox_fun_root: str):
    if not os.path.isdir(videox_fun_root):
        raise RuntimeError(f"VideoX-Fun path not found: {videox_fun_root}")
    if videox_fun_root not in sys.path:
        sys.path.insert(0, videox_fun_root)
    try:
        from videox_fun.pipeline.pipeline_wan2_2_fun_control import (
            Wan2_2FunControlPipeline,
        )
        return Wan2_2FunControlPipeline
    except Exception as e:
        raise RuntimeError(f"Failed to import Wan2_2FunControlPipeline from VideoX-Fun: {e}")


def run_fun_control(
    image_path: str,
    save_path: str,
    size: str,
    prompt: Optional[str] = None,
    control_type: str = "canny",
    control_strength: float = 0.7,
    canny_low_high: Tuple[int, int] = (100, 200),
    pose_hand: bool = True,
    pose_face: bool = False,
    sample_steps: int = 20,
    frame_num: int = 49,
    sample_solver: str = "fm-euler",
    guidance_scale: float = 6.0,
    fps: int = 24,
    device: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Generate a video using Wan2.2-Fun-A14B-Control with a control map."""
    t0 = time.time()
    diag = []
    out_root = os.getenv("WAN_OUTPUTS_ROOT", os.path.abspath(os.path.join(os.getcwd(), "outputs")))
    assets_root = os.path.join(out_root, "assets", "fun")
    _ensure_dir(assets_root)

    target_w, target_h = _parse_size_str(size)
    if not target_w or not target_h:
        return None, f"[Fun] Invalid size: {size}"

    rgb = _load_image_as_rgb(image_path, target_size=(target_w, target_h))

    ctype = (control_type or "canny").lower()
    diag.append(f"[Control] type={ctype} strength={control_strength}")
    try:
        if ctype == "canny":
            ctrl = _control_canny(rgb, low=int(canny_low_high[0]), high=int(canny_low_high[1]))
            diag.append(f"[Control] canny_low={canny_low_high[0]} high={canny_low_high[1]}")
        elif ctype == "depth":
            ctrl = _control_depth(rgb)
        elif ctype == "pose":
            ctrl = _control_pose(rgb, hand=pose_hand, face=pose_face)
            diag.append(f"[Control] pose hand={pose_hand} face={pose_face}")
        else:
            return None, f"[Fun] Unknown control_type: {control_type}"
    except Exception as e:
        return None, f"[Fun] Control map error: {e}"

    ctrl = _apply_strength(ctrl, control_strength)
    img_hash = _sha1_of_file(image_path)

    cm_path = os.path.join(assets_root, f"{ctype}_{img_hash}.png")
    try:
        _save_image(cm_path, ctrl)
        diag.append(f"[Control] saved={cm_path}")
    except Exception:
        pass

    frames = int(frame_num) if frame_num and frame_num > 0 else 49
    control_video = _tile_control_video(ctrl, frames)

    here = os.path.abspath(os.path.dirname(__file__))
    videox_fun_root = os.path.abspath(os.path.join(here, "..", "..", "VideoX-Fun"))
    PipeClass = _import_videox_fun_pipeline(videox_fun_root)

    models_root = os.getenv("WAN_FUN_MODELS_ROOT", os.getenv("WAN_MODELS_ROOT", os.path.abspath(os.path.join(os.getcwd(), "models"))))
    model_dir = os.path.join(models_root, "Wan2.2-Fun-A14B-Control")
    if not os.path.isdir(model_dir):
        return None, f"[Fun] Model dir not found: {model_dir}"

    import torch
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        pipe = PipeClass.from_pretrained(model_dir, torch_dtype=dtype)
        pipe.to(device)
    except Exception as e:
        return None, f"[Fun] Failed to load pipeline: {e}"

    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
        from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
        if sample_solver == "unipc":
            pipe.scheduler = FlowUniPCMultistepScheduler.from_config(pipe.scheduler.config)
        elif sample_solver == "dpm":
            pipe.scheduler = FlowDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    except Exception:
        pass

    cv_tensor = torch.from_numpy(control_video).to(device=device, dtype=torch.float32)

    ref_rgb = rgb.astype(np.float32) / 255.0
    ref_frame = np.transpose(ref_rgb, (2, 0, 1))[None, :, None, :, :]
    ref_tensor = torch.from_numpy(ref_frame).to(device=device, dtype=torch.float32)

    try:
        out = pipe(
            prompt=prompt or "",
            height=target_h,
            width=target_w,
            control_video=cv_tensor,
            ref_image=ref_tensor,
            num_frames=frames,
            num_inference_steps=int(sample_steps),
            guidance_scale=float(guidance_scale),
            output_type="numpy",
            return_dict=True,
        )
    except Exception as e:
        return None, f"[Fun] Pipeline error: {e}"
    try:
        videos = out.videos
        tmp_out = save_path if save_path.endswith(".mp4") else save_path + ".mp4"
        ok = _write_video_cv2(videos, tmp_out, fps=fps)
        if not ok:
            return None, "[Fun] Failed to write video"
        tmp2 = tmp_out + ".rescaled.mp4"
        ok2, backend = _rescale_video(tmp_out, tmp2, target_w, target_h)
        if ok2 and os.path.exists(tmp2):
            try:
                os.replace(tmp2, tmp_out)
                diag.append(f"[Output] rescaled_to={target_w}*{target_h} via {backend}")
            except Exception:
                pass
        elapsed = time.time() - t0
        diag.append(f"[Fun] done in {elapsed:.2f}s frames={frames} steps={sample_steps}")
        return tmp_out, "\n".join(diag)
    except Exception as e:
        return None, f"[Fun] Writing video failed: {e}"

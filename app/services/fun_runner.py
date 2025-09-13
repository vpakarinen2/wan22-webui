"""Fun Control runner."""
from __future__ import annotations

import numpy as np
import hashlib
import time
import sys
import os
import re

from typing import Optional, Tuple
from omegaconf import OmegaConf
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


def _read_video_frames(path: str, target_size: Tuple[int, int], limit_frames: int) -> list[np.ndarray]:
    """Read a video file into a list of RGB uint8 frames resized to target_size."""
    if cv2 is None:
        raise RuntimeError("OpenCV not available to read control video")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open control video: {path}")
    frames: list[np.ndarray] = []
    tw, th = target_size
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            try:
                bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
            except Exception:
                pass
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
    finally:
        try:
            cap.release()
        except Exception:
            pass
    if not frames:
        raise RuntimeError("No frames read from control video")
    n = len(frames)
    if n == limit_frames:
        return frames
    if n > limit_frames:
        idx = np.linspace(0, n - 1, limit_frames).astype(int).tolist()
        return [frames[i] for i in idx]
    out = []
    i = 0
    while len(out) < limit_frames:
        out.append(frames[i % n])
        i += 1
    return out


def _derive_control_from_frames(frames_rgb: list[np.ndarray], mode: str, pose_hand: bool, pose_face: bool) -> list[np.ndarray]:
    """Derive control frames from RGB frames per selected mode."""
    mode = (mode or "precomputed").lower()
    out: list[np.ndarray] = []
    for fr in frames_rgb:
        if mode == "precomputed":
            if fr.ndim == 2:
                fr = np.repeat(fr[..., None], 3, axis=2)
            elif fr.shape[2] == 4:
                fr = fr[:, :, :3]
            out.append(fr.astype(np.uint8))
        elif mode == "rgb->canny":
            out.append(_control_canny(fr, low=100, high=200).astype(np.uint8))
        elif mode == "rgb->pose":
            out.append(_control_pose(fr, hand=pose_hand, face=pose_face).astype(np.uint8))
        elif mode == "rgb->depth":
            out.append(_control_depth(fr).astype(np.uint8))
        else:
            raise RuntimeError(f"Unknown control_source: {mode}")
    return out


def _get_temporal_ratio(videox_fun_root: Optional[str]) -> int:
    """Fetch temporal compression ratio."""
    try:
        cfg_path = os.path.join(videox_fun_root if videox_fun_root else "", "config", "wan2.2", "wan_civitai_i2v.yaml")
        if os.path.isfile(cfg_path):
            cfg = OmegaConf.load(cfg_path)
            return int(cfg.get("vae_kwargs", {}).get("temporal_compression_ratio", 4))
    except Exception:
        pass
    return 4


def _write_video_cv2(frames_bcfhw: np.ndarray, out_path: str, fps: int = 24) -> bool:
    if cv2 is None:
        return False
    try:
        arr = frames_bcfhw
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().float().numpy()
        except Exception:
            pass

        if arr.ndim == 4:
            arr = arr[None, ...]
        if arr.ndim != 5:
            return False
        if arr.shape[1] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = np.swapaxes(arr, 1, 2)
        elif arr.shape[-1] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = np.transpose(arr, (0, 1, 4, 2, 3))

        b, f, c, h, w = arr.shape
        if c not in (1, 3):
            return False
    except Exception:
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    ok_any = False
    for i in range(f):
        frame = arr[0, i]  # (c, h, w)
        frame = np.transpose(frame, (1, 2, 0)) 
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255.0).astype(np.uint8)
        writer.write(frame[:, :, ::-1])
        ok_any = True
    writer.release()
    return ok_any and os.path.exists(out_path)


def _import_videox_fun_pipeline(videox_fun_root: Optional[str]):
    """Import VideoX-Fun pipeline."""
    if videox_fun_root and os.path.isdir(videox_fun_root):
        if videox_fun_root not in sys.path:
            sys.path.insert(0, videox_fun_root)
    elif videox_fun_root:
        pass

    try:
        from videox_fun.pipeline.pipeline_wan2_2_fun_control import (
            Wan2_2FunControlPipeline,
        )
        return Wan2_2FunControlPipeline
    except Exception as e:
        hint = videox_fun_root or "<sys.path>"
        raise RuntimeError(
            f"Failed to import Wan2_2FunControlPipeline (looked in {hint}). "
            f"Ensure VideoX-Fun is available or set WAN_VIDE0X_FUN_ROOT. Error: {e}"
        )


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
    use_inpaint: bool = True,
    control_video_path: Optional[str] = None,
    control_source: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Generate a video using Wan2.2-Fun-A14B-Control with a control map."""
    t0 = time.time()
    diag = []
    out_root = os.getenv("WAN_OUTPUTS_ROOT", os.path.abspath(os.path.join(os.getcwd(), "outputs")))
    assets_root = os.path.join(out_root, "assets", "fun")
    _ensure_dir(assets_root)

    try:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("WAN_FORCE_FLASH_ATTENTION", "1")
        os.environ.pop("WAN_FORCE_NO_FLASH_ATTENTION", None)
        diag.append(f"[Perf] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
        diag.append(f"[Perf] WAN_FORCE_FLASH_ATTENTION={os.environ.get('WAN_FORCE_FLASH_ATTENTION')}")
    except Exception:
        pass

    target_w, target_h = _parse_size_str(size)
    if not target_w or not target_h:
        return None, f"[Fun] Invalid size: {size}"

    rgb = _load_image_as_rgb(image_path, target_size=(target_w, target_h))
    diag.append(f"[Params] fps={fps}")

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
        return None, "\n".join(diag + [f"[Fun] Control map error: {e}"])

    ctrl = _apply_strength(ctrl, control_strength)
    ch, cw = ctrl.shape[:2]
    if ch != target_h or cw != target_w:
        try:
            if cv2 is not None:
                ctrl = cv2.resize(ctrl, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                ctrl = np.array(Image.fromarray(ctrl).resize((target_w, target_h), Image.BILINEAR))
            diag.append(f"[Control] resized control map from {ch}x{cw} -> {target_h}x{target_w}")
        except Exception as _e:
            diag.append(f"[Control] failed to resize control map: {_e}")
    img_hash = _sha1_of_file(image_path)

    cm_path = os.path.join(assets_root, f"{ctype}_{img_hash}.png")
    try:
        _save_image(cm_path, ctrl)
        diag.append(f"[Control] saved={cm_path}")
    except Exception:
        pass

    env_root = (
        os.getenv("WAN_VIDE0X_FUN_ROOT")
        or os.getenv("WAN_VIDEOX_FUN_ROOT")
        or os.getenv("VIDEOX_FUN_ROOT")
    )
    here = os.path.abspath(os.path.dirname(__file__))
    default_root = os.path.abspath(os.path.join(here, "..", "..", "VideoX-Fun"))
    videox_fun_root = env_root or default_root

    r = _get_temporal_ratio(videox_fun_root)
    req_frames = int(frame_num) if frame_num and frame_num > 0 else 49
    frames = int(((req_frames - 1) // r) * r + 1)
    if frames != req_frames:
        diag.append(f"[Control] frames aligned from {req_frames} -> {frames} (temporal_ratio={r})")
    if frames < 49:
        diag.append(f"[Control] frames bumped to 49 to satisfy transformer input channels (was {frames})")
        frames = 49
    ctrl_frames_np: np.ndarray
    if control_video_path and os.path.exists(control_video_path):
        try:
            diag.append(f"[Control] using control video: {control_video_path} source={control_source}")
            frames_rgb = _read_video_frames(control_video_path, (target_w, target_h), frames)
            ctrl_frames = _derive_control_from_frames(frames_rgb, control_source or "precomputed", pose_hand, pose_face)
            ctrl_frames_np = np.stack(ctrl_frames, axis=0).astype(np.uint8)  # (F, H, W, C)
            diag.append(f"[Control] built dynamic control frames: {ctrl_frames_np.shape}")
        except Exception as _e:
            diag.append(f"[Control] failed to use control video, falling back to static control: {_e}")
            control_video = _tile_control_video(ctrl, frames)
            ctrl_frames_np = np.transpose(control_video[0], (1, 2, 3, 0))
            ctrl_frames_np = (np.clip(ctrl_frames_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        control_video = _tile_control_video(ctrl, frames)
        ctrl_frames_np = np.transpose(control_video[0], (1, 2, 3, 0))
        ctrl_frames_np = (np.clip(ctrl_frames_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        diag.append(f"[Debug] control_frames_np: dtype={ctrl_frames_np.dtype}, range=({ctrl_frames_np.min()},{ctrl_frames_np.max()})")

    if ctrl_frames_np.shape[1] != target_h or ctrl_frames_np.shape[2] != target_w:
        try:
            resized = []
            for i in range(ctrl_frames_np.shape[0]):
                fr = ctrl_frames_np[i]
                fr = cv2.resize(fr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                resized.append(fr)
            ctrl_frames_np = np.stack(resized, axis=0)
            diag.append(f"[Control] resized control frames to {target_h}x{target_w}")
        except Exception as _e:
            diag.append(f"[Control] failed to resize control frames: {_e}")

    PipeClass = _import_videox_fun_pipeline(videox_fun_root)

    models_root = os.getenv("WAN_FUN_MODELS_ROOT", os.getenv("WAN_MODELS_ROOT", os.path.abspath(os.path.join(os.getcwd(), "models"))))
    model_dir = os.path.join(models_root, "Wan2.2-Fun-A14B-Control")
    if not os.path.isdir(model_dir):
        return None, "\n".join(diag + [f"[Fun] Model dir not found: {model_dir}"])

    import torch
    if torch.cuda.is_available():
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        dtype = torch.float32
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        config_path = os.path.join(videox_fun_root if videox_fun_root else "", "config", "wan2.2", "wan_civitai_i2v.yaml")
        if not os.path.isfile(config_path):
            return None, "\n".join(diag + [f"[Fun] Config not found: {config_path}"])
        config = OmegaConf.load(config_path)
        boundary = config.get("transformer_additional_kwargs", {}).get("boundary", 0.875)
        shift_val = 5

        from videox_fun.models import (
            AutoencoderKLWan,
            AutoencoderKLWan3_8,
            WanT5EncoderModel,
            Wan2_2Transformer3DModel,
            AutoTokenizer,
        )
        from videox_fun.utils.utils import filter_kwargs, get_image_to_video_latent, get_video_to_video_latent
        from diffusers import FlowMatchEulerDiscreteScheduler
        from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler

        t_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"]) if "transformer_additional_kwargs" in config else {}
        low_sub = t_kwargs.get("transformer_low_noise_model_subpath", "low_noise_model")
        high_sub = t_kwargs.get("transformer_high_noise_model_subpath", "high_noise_model")
        comb_type = t_kwargs.get("transformer_combination_type", "single")

        transformer = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_dir, low_sub),
            transformer_additional_kwargs=t_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        transformer_2 = None
        if comb_type == "moe":
            transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
                os.path.join(model_dir, high_sub),
                transformer_additional_kwargs=t_kwargs,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
            )

        v_kwargs = OmegaConf.to_container(config["vae_kwargs"]) if "vae_kwargs" in config else {}
        vae_type = v_kwargs.get("vae_type", "AutoencoderKLWan")
        vae_sub = v_kwargs.get("vae_subpath", "Wan2.1_VAE.pth")
        ChosenVAE = AutoencoderKLWan if vae_type == "AutoencoderKLWan" else AutoencoderKLWan3_8
        vae = ChosenVAE.from_pretrained(
            os.path.join(model_dir, vae_sub),
            additional_kwargs=v_kwargs,
        ).to(dtype)

        te_kwargs = OmegaConf.to_container(config["text_encoder_kwargs"]) if "text_encoder_kwargs" in config else {}
        tok_sub = te_kwargs.get("tokenizer_subpath", os.path.join("google", "umt5-xxl"))
        te_sub = te_kwargs.get("text_encoder_subpath", "models_t5_umt5-xxl-enc-bf16.pth")
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, tok_sub))
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_dir, te_sub),
            additional_kwargs=te_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        ).eval()

        s_kwargs = OmegaConf.to_container(config["scheduler_kwargs"]) if "scheduler_kwargs" in config else {}
        if sample_solver == "unipc":
            Sched = FlowUniPCMultistepScheduler
        elif sample_solver == "dpm":
            Sched = FlowDPMSolverMultistepScheduler
        else:
            Sched = FlowMatchEulerDiscreteScheduler
        scheduler = Sched(**filter_kwargs(Sched, s_kwargs))

        pipe = PipeClass(
            transformer=transformer,
            transformer_2=transformer_2,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )
        pipe.to(device)
    except Exception as e:
        return None, "\n".join(diag + [f"[Fun] Failed to load pipeline: {e}"])

    try:
        ctrl_video_pixels, _ctrl_mask, _ctrl_ref, _ctrl_clip = get_video_to_video_latent(
            ctrl_frames_np,
            video_length=frames,
            sample_size=(target_h, target_w),
            fps=fps,
            ref_image=None,
        )
        diag.append(f"[Debug] control_pixels: shape={tuple(ctrl_video_pixels.shape)}")
        cv_tensor = ctrl_video_pixels.to(device=device, dtype=torch.float32)
    except Exception as _e:
        diag.append(f"[Control] official util failed, fallback to direct tensor: {_e}")
        cv_tensor = torch.from_numpy(control_video).to(device=device, dtype=torch.float32)

    inpaint_video, inpaint_video_mask = None, None
    try:
        diag.append(f"[Debug] device={device}, dtype={dtype}")
        if use_inpaint:
            from videox_fun.utils.utils import get_image_to_video_latent
            inpaint_video, inpaint_video_mask, _clip_image = get_image_to_video_latent(
                image_path,
                None,
                video_length=frames,
                sample_size=(target_h, target_w),
            )
            inpaint_video = inpaint_video.to(device=device, dtype=torch.float32)
            inpaint_video_mask = inpaint_video_mask.to(device=device, dtype=torch.float32)
            diag.append(f"[Debug] inpaint_video: {tuple(inpaint_video.shape)}, inpaint_mask: {tuple(inpaint_video_mask.shape)}")
        else:
            diag.append("[Debug] use_inpaint=False -> starting from noise (no init video)")
    except Exception as _e:
        inpaint_video, inpaint_video_mask = None, None
        diag.append(f"[Debug] inpaint build failed, proceeding without: {_e}")

    ref_rgb = rgb.astype(np.float32) / 255.0
    ref_frame = np.transpose(ref_rgb, (2, 0, 1))[None, :, None, :, :]
    ref_tensor = torch.from_numpy(ref_frame).to(device=device, dtype=torch.float32)

    try:
        diag.append(f"[Debug] control_video tensor: shape={tuple(cv_tensor.shape)}, frames={frames}, r={r}")
        out = pipe(
            prompt=prompt or "",
            height=target_h,
            width=target_w,
            video=inpaint_video,
            mask_video=inpaint_video_mask,
            control_video=cv_tensor,
            ref_image=None,
            num_frames=frames,
            num_inference_steps=int(sample_steps),
            guidance_scale=float(guidance_scale),
            boundary=boundary,
            shift=shift_val,
            output_type="numpy",
            return_dict=True,
        )
    except Exception as e:
        msg = str(e)
        exp_m = re.search(r"to have (\d+) channels, but got (\d+) channels", msg)
        if exp_m:
            try:
                expected_ch = int(exp_m.group(1))
                got_ch = int(exp_m.group(2))
                lc = int(getattr(getattr(pipe, "vae", None), "config", getattr(pipe, "vae", None)).latent_channels)
                if expected_ch % lc == 0:
                    L_exp = expected_ch // lc
                    r = _get_temporal_ratio(videox_fun_root)
                    new_frames = int((L_exp - 1) * r + 1)
                    if new_frames != frames:
                        diag.append(f"[Fun] Retrying with frames={new_frames} due to channel mismatch (expected {expected_ch}, got {got_ch})")
                        frames = new_frames
                        cv_tensor = torch.from_numpy(_tile_control_video(ctrl, frames)).to(device=device, dtype=torch.float32)
                        out = pipe(
                            prompt=prompt or "",
                            height=target_h,
                            width=target_w,
                            video=inpaint_video,
                            mask_video=inpaint_video_mask,
                            control_video=cv_tensor,
                            ref_image=None,
                            num_frames=frames,
                            num_inference_steps=int(sample_steps),
                            guidance_scale=float(guidance_scale),
                            boundary=boundary,
                            shift=shift_val,
                            output_type="numpy",
                            return_dict=True,
                        )
                    else:
                        raise
                else:
                    raise
            except Exception:
                return None, "\n".join(diag + [f"[Fun] Pipeline error: {e}"])
        else:
            return None, "\n".join(diag + [f"[Fun] Pipeline error: {e}"])
    try:
        videos = out.videos
        tmp_out = save_path if save_path.endswith(".mp4") else save_path + ".mp4"
        ok = _write_video_cv2(videos, tmp_out, fps=fps)
        if not ok:
            return None, "\n".join(diag + ["[Fun] Failed to write video"]) 
        tmp2 = tmp_out + ".rescaled.mp4"
        ok2, backend = _rescale_video(tmp_out, tmp2, target_w, target_h)
        if ok2 and os.path.exists(tmp2):
            try:
                os.replace(tmp2, tmp_out)
            except Exception:
                pass
            diag.append(f"[Output] rescaled_to={target_w}*{target_h} via {backend}")
        elapsed = time.time() - t0
        diag.append(f"[Fun] done in {elapsed:.2f}s frames={frames} steps={sample_steps}")
        return tmp_out, "\n".join(diag)
    except Exception as _e:
        return None, "\n".join(diag + [f"[Fun] Failed to rebuild control video via official util: {_e}"])

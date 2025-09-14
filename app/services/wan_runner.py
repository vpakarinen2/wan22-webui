"""Wan2.2 runner."""

from __future__ import annotations

import subprocess
import threading
import shlex
import time
import sys
import re
import os

from app.config.settings import settings
from typing import Optional
from pathlib import Path


def _build_generate_path() -> str:
    wan_repo_root = os.getenv("WAN_REPO_ROOT", os.path.join(os.getcwd(), "Wan2.2"))
    return os.path.join(wan_repo_root, "generate.py")


def _validate_ckpt_dir(ckpt_dir: str) -> Optional[str]:
    if not ckpt_dir:
        return "Checkpoint directory is empty."
    if not os.path.isdir(ckpt_dir):
        return f"Checkpoint directory not found: {ckpt_dir}"
    return None


def _probe_video_duration(video_path: str) -> Optional[float]:
    """Try to get duration (seconds) using ffprobe or OpenCV fallback."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout:
            try:
                return float(proc.stdout.strip())
            except Exception:
                pass
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            if fps > 0 and frames > 0:
                return float(frames / fps)
        try:
            cap.release()
        except Exception:
            pass
    except Exception:
        pass
    return None


def _probe_video_fps(video_path: str) -> Optional[float]:
    """Try to get FPS using ffprobe or OpenCV."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout:
            val = proc.stdout.strip()
            if "/" in val:
                try:
                    num, den = val.split("/", 1)
                    num = float(num)
                    den = float(den)
                    if den != 0:
                        return num / den
                except Exception:
                    pass
            else:
                try:
                    return float(val)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            cap.release()
            if fps > 0:
                return float(fps)
        try:
            cap.release()
        except Exception:
            pass
    except Exception:
        pass
    return None


def _remux_set_fps(in_path: str, out_path: str, fps: int) -> tuple[bool, str]:
    """Re-encode video to a target FPS using ffmpeg, fallback to OpenCV."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                in_path,
                "-r",
                str(int(fps)),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                out_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0 and os.path.exists(out_path):
            return True, "ffmpeg"
    except Exception:
        pass
    try:
        import cv2
        import numpy as np
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            return False, ""
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        if w <= 0 or h <= 0:
            # derive from first frame
            ret, fr = cap.read()
            if not ret:
                cap.release()
                return False, ""
            h, w = fr.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        ok_any = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            ok_any = True
        cap.release()
        writer.release()
        if ok_any and os.path.exists(out_path):
            return True, "opencv"
    except Exception:
        pass
    return False, ""


def _parse_size_str(area: str) -> tuple[Optional[int], Optional[int]]:
    try:
        w, h = area.split("*")
        return int(w), int(h)
    except Exception:
        return None, None


def _rescale_video(in_path: str, out_path: str, w: int, h: int) -> tuple[bool, str]:
    """Rescale video to WxH."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                in_path,
                "-vf",
                f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                out_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0 and os.path.exists(out_path):
            return True, "ffmpeg"
    except Exception:
        pass
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            return False, ""
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        ok_any = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ih, iw = frame.shape[:2]
            if iw <= 0 or ih <= 0:
                out_frame = frame
            else:
                scale = max(w / float(iw), h / float(ih))
                new_w = max(1, int(round(iw * scale)))
                new_h = max(1, int(round(ih * scale)))
                try:
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                except Exception:
                    resized = frame
                x_off = max(0, (new_w - w) // 2)
                y_off = max(0, (new_h - h) // 2)
                out_frame = resized[y_off:y_off + h, x_off:x_off + w]
                if out_frame.shape[0] != h or out_frame.shape[1] != w:
                    canvas = np.zeros((h, w, 3), dtype=resized.dtype)
                    hh, ww = out_frame.shape[:2]
                    y0 = max(0, (h - hh) // 2)
                    x0 = max(0, (w - ww) // 2)
                    canvas[y0:y0 + hh, x0:x0 + ww] = out_frame
                    out_frame = canvas
            writer.write(out_frame)
            ok_any = True
        cap.release()
        writer.release()
        if ok_any and os.path.exists(out_path):
            return True, "opencv"
    except Exception:
        pass
    return False, ""


def _validate_generate_py(generate_py: str) -> Optional[str]:
    if not os.path.isfile(generate_py):
        return (
            "Wan2.2 generate.py not found. Set WAN_REPO_ROOT to the Wan2.2 repo root.\n"
            f"Expected at: {generate_py}\n"
            "Example (PowerShell): $env:WAN_REPO_ROOT='C:/path/to/Wan2.2'"
        )
    return None


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _probe_video_resolution(video_path: str) -> Optional[tuple[int, int]]:
    """Try to get (width,height) for an mp4 using ffprobe."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout:
            out = proc.stdout.strip()
            if "x" in out:
                w, h = out.split("x", 1)
                return int(w), int(h)
    except Exception:
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return w, h
        try:
            cap.release()
        except Exception:
            pass
    except Exception:
        pass
    return None


def run_i2v(
    image: Optional[str],
    prompt: str,
    size: str,
    ckpt_dir: str,
    sample_steps: Optional[float] = None,
    sample_solver: Optional[str] = None,
    frame_num: Optional[float] = None,
    offload_model: Optional[bool] = None,
    t5_cpu: bool = False,
    prefer_flash_attn: Optional[bool] = None,
    force_output_rescale: bool = True,
    seed: Optional[int] = None,
    out_fps: Optional[float] = None,
) -> tuple[Optional[str], str]:
    """Prepare I2V command and return."""
    errs = []
    if not image:
        errs.append("Image is required for I2V.")
    err = _validate_ckpt_dir(ckpt_dir)
    if err:
        errs.append(err)
    generate_py = _build_generate_path()
    err = _validate_generate_py(generate_py)
    if err:
        errs.append(err)

    if errs:
        return None, "\n".join(["[I2V] Validation errors:"] + errs)

    os.makedirs(settings.outputs_root, exist_ok=True)
    save_path = os.path.join(
        settings.outputs_root,
        f"i2v_A14B_{size.replace('*','x')}_{int(time.time())}.mp4",
    )

    cmd = [
        sys.executable,
        generate_py,
        "--task",
        "i2v-A14B",
        "--size",
        size,
        "--ckpt_dir",
        ckpt_dir,
        "--image",
        image,
        "--save_file",
        save_path,
        "--convert_model_dtype",
    ]
    if prompt is not None:
        cmd += ["--prompt", prompt]
    if sample_steps is not None:
        try:
            cmd += ["--sample_steps", str(int(sample_steps))]
        except Exception:
            pass
    if sample_solver:
        cmd += ["--sample_solver", sample_solver]
    if frame_num is not None:
        try:
            nframes = str(int(frame_num))
            cmd += ["--frame_num", nframes]
            cmd += ["--infer_frames", nframes]
        except Exception:
            pass
    if seed is not None:
        try:
            cmd += ["--seed", str(int(seed))]
        except Exception:
            pass
    if offload_model is not None:
        cmd += ["--offload_model", "True" if offload_model else "False"]
    if t5_cpu:
        cmd += ["--t5_cpu"]

    if not settings.execute_enabled:
        return None, (
            "[I2V] Backend prepared. Execution disabled (WAN_EXECUTE=0).\n"
            "To run for real, set WAN_REPO_ROOT to your Wan2.2 repo and enable WAN_EXECUTE=1 (Runpod/Linux).\n\n"
            f"Planned command:\n{_format_cmd(cmd)}\n"
        )

    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = project_root + (os.pathsep + env.get("PYTHONPATH", ""))
    if prefer_flash_attn is True:
        env["WAN_FORCE_FLASH_ATTENTION"] = "1"
        env.pop("WAN_FORCE_NO_FLASH_ATTENTION", None)
    elif prefer_flash_attn is False:
        env["WAN_FORCE_NO_FLASH_ATTENTION"] = "1"
        env.pop("WAN_FORCE_FLASH_ATTENTION", None)

    diag = [
        "[I2V] parameters:",
        f"  image={image}",
        f"  prompt={prompt}",
        f"  size={size}",
        f"  ckpt_dir={ckpt_dir}",
        f"  sample_steps={sample_steps}",
        f"  sample_solver={sample_solver}",
        f"  frame_num={frame_num}",
        f"  offload_model={offload_model}",
        f"  t5_cpu={t5_cpu}",
        f"  prefer_flash_attn={prefer_flash_attn}",
        f"  force_output_rescale={force_output_rescale}",
        f"  seed={seed}",
        f"  out_fps={out_fps}",
        f"  env: WAN_FORCE_FLASH_ATTENTION={env.get('WAN_FORCE_FLASH_ATTENTION')}, WAN_FORCE_NO_FLASH_ATTENTION={env.get('WAN_FORCE_NO_FLASH_ATTENTION')}",
        f"  cwd={os.path.dirname(generate_py)}",
        "",
    ]
    video_path, logs = _execute(cmd, cwd=os.path.dirname(generate_py), env=env)
    logs = "\n".join(diag) + logs
    try:
        if video_path is None and os.path.exists(save_path):
            video_path = save_path
    except Exception:
        pass
    try:
        if video_path and os.path.exists(video_path):
            if os.path.exists(save_path) and os.path.normpath(video_path) != os.path.normpath(save_path):
                os.remove(save_path)
        if video_path and os.path.exists(video_path):
            # Optional rescale
            target_w, target_h = _parse_size_str(size)
            if force_output_rescale and target_w and target_h:
                tmp_out = save_path + ".rescaled.mp4"
                ok, backend = _rescale_video(video_path, tmp_out, target_w, target_h)
                if ok and os.path.exists(tmp_out):
                    try:
                        os.replace(tmp_out, save_path)
                        video_path = save_path
                        logs = f"{logs}\n[Output] rescaled_to={target_w}*{target_h} via {backend}"
                    except Exception:
                        try:
                            os.remove(tmp_out)
                        except Exception:
                            pass
            if out_fps is not None and out_fps > 0:
                tmp_out = save_path + ".fps.mp4"
                ok, backend = _remux_set_fps(video_path, tmp_out, int(out_fps))
                if ok and os.path.exists(tmp_out):
                    try:
                        os.replace(tmp_out, save_path)
                        video_path = save_path
                        logs = f"{logs}\n[Output] set_fps={int(out_fps)} via {backend}"
                    except Exception:
                        try:
                            os.remove(tmp_out)
                        except Exception:
                            pass
            res = _probe_video_resolution(video_path)
            dur = _probe_video_duration(video_path)
            fpsv = _probe_video_fps(video_path)
            if res:
                logs = f"{logs}\n[Output] actual_resolution={res[0]}*{res[1]}"
            if dur is not None:
                logs = f"{logs}\n[Output] duration_s={dur:.2f}"
            if fpsv is not None:
                logs = f"{logs}\n[Output] actual_fps={fpsv:.3f}"
            try:
                base_txt = os.path.splitext(save_path)[0] + ".txt"
                with open(base_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(diag) + logs)
            except Exception:
                pass
    except Exception:
        pass
    return video_path, logs


def run_s2v(
    ref_image: Optional[str],
    audio: Optional[str],
    prompt: str,
    size: str,
    ckpt_dir: str,
    sample_steps: Optional[float] = None,
    sample_solver: Optional[str] = None,
    frame_num: Optional[float] = None,
    offload_model: Optional[bool] = None,
    t5_cpu: bool = False,
    prefer_flash_attn: Optional[bool] = None,
    force_output_rescale: bool = True,
    seed: Optional[int] = None,
    out_fps: Optional[float] = None,
) -> tuple[Optional[str], str]:
    """Prepare S2V command and return (video_path, logs). No execution yet."""
    errs = []
    if not ref_image:
        errs.append("Reference image is required for S2V.")

    if not audio:
        errs.append("Audio file is required for S2V.")

    err = _validate_ckpt_dir(ckpt_dir)
    if err:
        errs.append(err)
    generate_py = _build_generate_path()
    err = _validate_generate_py(generate_py)
    if err:
        errs.append(err)

    if errs:
        return None, "\n".join(["[S2V] Validation errors:"] + errs)

    os.makedirs(settings.outputs_root, exist_ok=True)
    save_path = os.path.join(
        settings.outputs_root,
        f"s2v_14B_{size.replace('*','x')}_{int(time.time())}.mp4",
    )

    cmd = [
        sys.executable,
        generate_py,
        "--task",
        "s2v-14B",
        "--size",
        size,
        "--ckpt_dir",
        ckpt_dir,
        "--image",
        ref_image,
        "--save_file",
        save_path,
        "--convert_model_dtype",
    ]

    cmd += ["--audio", audio]

    if prompt is not None:
        cmd += ["--prompt", prompt]
    if sample_steps is not None:
        try:
            cmd += ["--sample_steps", str(int(sample_steps))]
        except Exception:
            pass
    if sample_solver:
        cmd += ["--sample_solver", sample_solver]
    if frame_num is not None:
        try:
            nframes = str(int(frame_num))
            cmd += ["--frame_num", nframes]
            cmd += ["--infer_frames", nframes]
        except Exception:
            pass
    if offload_model is not None:
        cmd += ["--offload_model", "True" if offload_model else "False"]
    if t5_cpu:
        cmd += ["--t5_cpu"]

    if not settings.execute_enabled:
        return None, (
            "[S2V] Backend prepared. Execution disabled (WAN_EXECUTE=0).\n"
            "To run for real, set WAN_REPO_ROOT to your Wan2.2 repo and enable WAN_EXECUTE=1 (Runpod/Linux).\n\n"
            f"Planned command:\n{_format_cmd(cmd)}\n"
        )

    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = project_root + (os.pathsep + env.get("PYTHONPATH", ""))
    if prefer_flash_attn is True:
        env["WAN_FORCE_FLASH_ATTENTION"] = "1"
        env.pop("WAN_FORCE_NO_FLASH_ATTENTION", None)
    elif prefer_flash_attn is False:
        env["WAN_FORCE_NO_FLASH_ATTENTION"] = "1"
        env.pop("WAN_FORCE_FLASH_ATTENTION", None)

    diag = [
        "[S2V] parameters:",
        f"  ref_image={ref_image}",
        f"  audio={audio}",
        f"  prompt={prompt}",
        f"  size={size}",
        f"  ckpt_dir={ckpt_dir}",
        f"  sample_steps={sample_steps}",
        f"  sample_solver={sample_solver}",
        f"  frame_num={frame_num}",
        f"  offload_model={offload_model}",
        f"  t5_cpu={t5_cpu}",
        f"  prefer_flash_attn={prefer_flash_attn}",
        f"  force_output_rescale={force_output_rescale}",
        f"  seed={seed}",
        f"  out_fps={out_fps}",
        f"  env: WAN_FORCE_FLASH_ATTENTION={env.get('WAN_FORCE_FLASH_ATTENTION')}, WAN_FORCE_NO_FLASH_ATTENTION={env.get('WAN_FORCE_NO_FLASH_ATTENTION')}",
        f"  cwd={os.path.dirname(generate_py)}",
        "",
    ]
    video_path, logs = _execute(cmd, cwd=os.path.dirname(generate_py), env=env)
    logs = "\n".join(diag) + logs
    try:
        if video_path is None and os.path.exists(save_path):
            video_path = save_path
    except Exception:
        pass
    try:
        if video_path and os.path.exists(video_path):
            if os.path.exists(save_path) and os.path.normpath(video_path) != os.path.normpath(save_path):
                os.remove(save_path)
        if video_path and os.path.exists(video_path):
            res = _probe_video_resolution(video_path)
            if res:
                logs = f"{logs}\n[Output] actual_resolution={res[0]}*{res[1]}"
            target_w, target_h = _parse_size_str(size)
            if force_output_rescale and target_w and target_h:
                tmp_out = save_path + ".rescaled.mp4"
                ok, backend = _rescale_video(video_path, tmp_out, target_w, target_h)
                if ok and os.path.exists(tmp_out):
                    try:
                        os.replace(tmp_out, save_path)
                        video_path = save_path
                        logs = f"{logs}\n[Output] rescaled_to={target_w}*{target_h} via {backend}"
                    except Exception:
                        try:
                            os.remove(tmp_out)
                        except Exception:
                            pass
            if out_fps is not None and out_fps > 0:
                tmp_out = save_path + ".fps.mp4"
                ok, backend = _remux_set_fps(video_path, tmp_out, int(out_fps))
                if ok and os.path.exists(tmp_out):
                    try:
                        os.replace(tmp_out, save_path)
                        video_path = save_path
                        logs = f"{logs}\n[Output] set_fps={int(out_fps)} via {backend}"
                    except Exception:
                        try:
                            os.remove(tmp_out)
                        except Exception:
                            pass
            dur = _probe_video_duration(video_path)
            fpsv = _probe_video_fps(video_path)
            if dur is not None:
                logs = f"{logs}\n[Output] duration_s={dur:.2f}"
            if fpsv is not None:
                logs = f"{logs}\n[Output] actual_fps={fpsv:.3f}"
            try:
                base_txt = os.path.splitext(save_path)[0] + ".txt"
                with open(base_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(diag) + logs)
            except Exception:
                pass
    except Exception:
        pass
    return video_path, logs

_CURRENT_PROC_LOCK = threading.Lock()
_CURRENT_PROC: Optional[subprocess.Popen] = None


def cancel_current() -> str:
    """Attempt to terminate the running process."""
    global _CURRENT_PROC
    with _CURRENT_PROC_LOCK:
        if _CURRENT_PROC is None:
            return "No running job to cancel."
        try:
            _CURRENT_PROC.terminate()
            try:
                _CURRENT_PROC.wait(timeout=5)
            except Exception:
                _CURRENT_PROC.kill()
        finally:
            _CURRENT_PROC = None
    return "Cancellation requested."

_MP4_REGEX = re.compile(r"([A-Za-z]:\\[^\n]*?\.mp4|/[^\s]*?\.mp4|[^\s]*?\.mp4)")


def _execute(cmd: list[str], cwd: str, env: Optional[dict[str, str]] = None) -> tuple[Optional[str], str]:
    """Run the command, capture combined logs, and try to detect output video path."""
    global _CURRENT_PROC
    logs: list[str] = []
    video_path: Optional[str] = None
    start_ts = time.time()

    if cmd[0] == sys.executable:
        cmd.insert(1, "-u")

    with _CURRENT_PROC_LOCK:
        if _CURRENT_PROC is not None:
            return None, "Another job is already running. Please wait or cancel it."
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _CURRENT_PROC = proc

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if not line:
                continue
            logs.append(line.rstrip("\n"))
            m = _MP4_REGEX.search(line)
            if m:
                candidate = m.group(1)
                if not os.path.isabs(candidate):
                    candidate = os.path.join(cwd, candidate)
                video_path = candidate
        code = proc.wait()
        dur = time.time() - start_ts
        logs.append(f"[exit code] {code}")
        logs.append(f"[duration] {dur:.1f}s")
    finally:
        with _CURRENT_PROC_LOCK:
            if _CURRENT_PROC is proc:
                _CURRENT_PROC = None

    return video_path, "\n".join(logs)

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
    if offload_model is not None:
        cmd += ["--offload_model", "True" if offload_model else "False"]
    if t5_cpu:
        cmd += ["--t5_cpu"]

    # LoRA removed for now (no flags added)

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
    except Exception:
        pass
    return video_path, logs

_CURRENT_PROC_LOCK = threading.Lock()
_CURRENT_PROC: Optional[subprocess.Popen] = None


def cancel_current() -> str:
    """Attempt to terminate the running process (if any)."""
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

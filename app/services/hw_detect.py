"""Hardware detection."""

import subprocess
from typing import Optional


def nvidia_smi_vram() -> Optional[int]:
    """Return total VRAM in MiB via `nvidia-smi`, or None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        values = [line.strip() for line in out.splitlines() if line.strip()]
        if not values:
            return None
        return int(values[0])
    except Exception:
        return None

"""Prints advisory notes for SDPA/FlashAttention preferences."""

from __future__ import annotations

import os
import sys

try:
    if os.getenv("WAN_FORCE_FLASH_ATTENTION") in {"1", "true", "True"}:
        print("[sitecustomize] Prefer FlashAttention (advisory)", file=sys.stderr)
    if os.getenv("WAN_FORCE_NO_FLASH_ATTENTION") in {"1", "true", "True"}:
        print("[sitecustomize] Disable FlashAttention (advisory)", file=sys.stderr)
except Exception as e:
    print(f"[sitecustomize] Warning: {e}", file=sys.stderr)
except Exception as e:
    print(f"[sitecustomize] Error configuring SDPA: {e}", file=sys.stderr)

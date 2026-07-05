import os
import re
from functools import lru_cache


def safe_part(value: str, fallback: str = "unknown") -> str:
    """Sanitize an arbitrary string into a filesystem-safe path segment."""
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or fallback


@lru_cache(maxsize=1)
def get_gpu_info() -> str:
    """GPU model identifier (sanitized), shared by the Triton autotune cache
    and the JIT cache manager so both resolve the same per-GPU scope.

    Priority: TRITON_AUTOTUNE_GPU_NAME env var > torch.cuda. CUDA-only — other
    backends are not in scope for this cache. Falls back to "unknown" so the
    module still imports on a host without a usable GPU (tests, dev machines).
    torch is imported lazily so callers that only need the sanitizer (e.g.
    rtp_llm.__init__) do not pay for a torch import.
    """
    gpu_name = os.environ.get("TRITON_AUTOTUNE_GPU_NAME")
    if gpu_name is None:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return safe_part(gpu_name, "unknown_gpu") if gpu_name else "unknown"

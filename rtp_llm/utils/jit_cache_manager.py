import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rtp_llm.utils.import_util import has_internal_source


COMPONENTS = (
    "flashinfer",
    "triton",
    "triton_autotune",
    "deep_gemm",
    "torch_extensions",
)

ENV_BY_COMPONENT = {
    "flashinfer": "FLASHINFER_WORKSPACE_BASE",
    "triton": "TRITON_CACHE_DIR",
    "triton_autotune": "TRITON_AUTOTUNE_CONFIG_DIR",
    "deep_gemm": "DG_JIT_CACHE_DIR",
    "torch_extensions": "TORCH_EXTENSIONS_DIR",
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sanitize(value: Any) -> str:
    text = str(value or "unknown")
    return "".join(c if c.isalnum() else "_" for c in text).strip("_") or "unknown"


def _local_component_key() -> str:
    cuda = _sanitize(os.environ.get("CUDA_VERSION", "unknown")).replace("_", "")
    return f"py{sys.version_info.major}{sys.version_info.minor}_cu{cuda}"


class _LocalJitCacheManager:
    def __init__(self, jit_config: Any):
        self.local_root = Path(
            os.path.expanduser(
                getattr(jit_config, "local_jit_cache_dir", "~/.cache/rtp_llm_jit")
            )
        )
        self.local_rank: Optional[int] = None

    def bootstrap_env(self) -> None:
        os.environ.pop("DG_JIT_REMOTE_CACHE_DIR", None)
        component_key = _local_component_key()
        self.local_root.mkdir(parents=True, exist_ok=True)
        for component in COMPONENTS:
            local_dir = self.local_root / component / component_key
            local_dir.mkdir(parents=True, exist_ok=True)
            os.environ[ENV_BY_COMPONENT[component]] = str(local_dir)
        logging.info("JIT local cache root: %s", self.local_root)

    def prepare(self, local_rank: int) -> Dict[str, Any]:
        self.local_rank = local_rank
        return self._summary("prepare", "local_only", "skipped")

    def start_background_sync(self) -> None:
        return

    def stop(self) -> None:
        return

    def sync_once(self, mode: str = "manual_sync") -> Dict[str, Any]:
        return self._summary(mode, "local_only", "skipped")

    def _summary(self, mode: str, cache_state: str, result: str) -> Dict[str, Any]:
        return {
            "event": "jit_cache_sync_summary",
            "timestamp_ms": _now_ms(),
            "mode": mode,
            "cache_state": cache_state,
            "result": result,
            "local_rank": self.local_rank,
            "remote_root": "",
            "local_root": str(self.local_root),
            "total_cost_ms": 0,
            "components": {},
        }


def _load_internal_manager(jit_config: Any) -> Optional[Any]:
    if not has_internal_source():
        return None
    try:
        from internal_source.rtp_llm.utils.jit_cache_manager import (
            JitCacheManager as InternalJitCacheManager,
        )

        return InternalJitCacheManager(jit_config)
    except Exception:
        logging.exception("failed to load internal JIT cache manager; use local only")
        return None


class JitCacheManager:
    def __init__(self, jit_config: Any):
        self._impl = _load_internal_manager(jit_config) or _LocalJitCacheManager(
            jit_config
        )

    def bootstrap_env(self) -> None:
        return self._impl.bootstrap_env()

    def prepare(self, local_rank: int) -> Dict[str, Any]:
        return self._impl.prepare(local_rank)

    def start_background_sync(self) -> None:
        return self._impl.start_background_sync()

    def stop(self) -> None:
        return self._impl.stop()

    def sync_once(self, mode: str = "manual_sync") -> Dict[str, Any]:
        return self._impl.sync_once(mode)


def main(argv=None):
    manager = _load_internal_manager(None)
    if manager is not None:
        from internal_source.rtp_llm.utils.jit_cache_manager import main as internal_main

        return internal_main(argv)
    summary = _LocalJitCacheManager(None).sync_once("manual_sync")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()

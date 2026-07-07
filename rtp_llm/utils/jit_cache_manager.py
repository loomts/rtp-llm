import logging
import os
import time
from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from rtp_llm.utils.gpu_info import get_gpu_info, safe_part

BUILTIN_CONFIG_SENTINEL = "__builtin__"
DETAILED_STATS_ENV = "JIT_CACHE_DETAILED_STATS"


def _dist_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _cuda_scope() -> str:
    import torch

    return "cuda-" + safe_part(str(torch.version.cuda or "unknown"))


def _torch_extensions_scope() -> str:
    import sys

    import torch

    py = f"py{sys.version_info[0]}{sys.version_info[1]}{getattr(sys, 'abiflags', '')}"
    cuda = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
    abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)
    abi_part = str(int(abi)) if isinstance(abi, bool) else "unknown"
    return f"torch-{safe_part(_dist_version('torch'))}-{safe_part(f'{py}_{cuda}')}-cxxabi-{safe_part(abi_part)}"


@dataclass(frozen=True)
class Component:
    name: str
    env_name: str
    scope_func: Callable[[], str | None] | None = None
    local_dir: Path = Path()

    def resolve(self, root: Path, resolve_scopes: bool = True) -> "Component":
        scope = self.scope_func() if resolve_scopes and self.scope_func else None
        local_dir = root / self.name
        return replace(self, local_dir=local_dir / scope if scope else local_dir)


COMPONENTS = (
    Component("flashinfer", "FLASHINFER_WORKSPACE_BASE", _cuda_scope),
    Component(
        "deep_gemm",
        "DG_JIT_CACHE_DIR",
        lambda: "deep_gemm-" + safe_part(_dist_version("deep_gemm")),
    ),
    Component("tensorrt_llm_deep_gemm", "TRTLLM_DG_CACHE_DIR", _cuda_scope),
    Component("torch_extensions", "TORCH_EXTENSIONS_DIR", _torch_extensions_scope),
    Component("triton", "TRITON_CACHE_DIR"),
    Component("triton_autotune", "TRITON_AUTOTUNE_CONFIG_DIR", get_gpu_info),
)


def _duration_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000.0


def _component_env_summary() -> str:
    return ", ".join(
        f"{component.env_name}={os.environ.get(component.env_name, '')}"
        for component in COMPONENTS
    )


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _detailed_stats_enabled() -> bool:
    # Detailed per-component stats (file_count/total_bytes/suffix_counts) are on by
    # default; set JIT_CACHE_DETAILED_STATS=0 to disable if the FUSE rglob walk is too costly.
    return os.environ.get(DETAILED_STATS_ENV, "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _suffix_key(path: Path) -> str:
    if path.name == "build.ninja":
        return "build.ninja"
    return path.suffix or "<no_suffix>"


def _component_visibility_summary(
    components: tuple[Component, ...],
    detailed: bool = False,
) -> tuple[str, int]:
    states: list[str] = []
    visible_components = 0
    for component in components:
        scan_start = time.monotonic()
        try:
            exists = component.local_dir.is_dir()
            has_entries = any(component.local_dir.iterdir()) if exists else False
            if has_entries:
                visible_components += 1
            parts = [
                component.name,
                f"path={component.local_dir}",
                f"exists={exists}",
                f"has_entries={has_entries}",
                f"env={os.environ.get(component.env_name, '')}",
            ]
            if detailed and exists:
                file_count = 0
                dir_count = 0
                total_bytes = 0
                max_file_bytes = 0
                suffix_counts: dict[str, int] = {}
                for path in component.local_dir.rglob("*"):
                    try:
                        if path.is_dir():
                            dir_count += 1
                            continue
                        if not path.is_file():
                            continue
                        size = path.stat().st_size
                    except OSError:
                        continue
                    file_count += 1
                    total_bytes += size
                    max_file_bytes = max(max_file_bytes, size)
                    suffix = _suffix_key(path)
                    suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
                suffix_summary = ",".join(
                    f"{suffix}:{count}"
                    for suffix, count in sorted(suffix_counts.items())
                )
                parts.extend(
                    [
                        f"file_count={file_count}",
                        f"dir_count={dir_count}",
                        f"total_bytes={total_bytes}",
                        f"max_file_bytes={max_file_bytes}",
                        f"suffix_counts={suffix_summary or '<none>'}",
                    ]
                )
            parts.append(f"scan_ms={_duration_ms(scan_start):.2f}")
            states.append(":".join(parts))
        except OSError as e:
            states.append(
                f"{component.name}:path={component.local_dir},error={type(e).__name__}:{e},env={os.environ.get(component.env_name, '')}"
            )
    return "; ".join(states), visible_components


def resolve_remote_root(remote_jit_dir: Any) -> Path | None:
    start = time.monotonic()
    text = str(remote_jit_dir or "").strip()
    if not text:
        logging.info("JIT remote cache disabled: remote_jit_dir is empty")
        return None
    parsed = urlparse(text)
    if parsed.scheme:
        from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local

        logging.info("JIT remote cache mount start: scheme=%s", parsed.scheme)
        text = fetch_remote_file_to_local(text, MountRwMode.RWMODE_RW)
        logging.info(
            "JIT remote cache mount done: scheme=%s resolved_path=%s duration_ms=%.2f",
            parsed.scheme,
            text,
            _duration_ms(start),
        )
    path = Path(text).expanduser().absolute()
    if not path.is_dir():
        logging.warning(
            "JIT remote cache disabled: remote_jit_dir=%r resolved to %s which does not exist or is not a directory duration_ms=%.2f",
            remote_jit_dir,
            path,
            _duration_ms(start),
        )
        return None
    logging.info(
        "JIT remote cache root resolved: path=%s duration_ms=%.2f",
        path,
        _duration_ms(start),
    )
    return path


def apply_jit_cache_env(
    root: Path | str, resolve_scopes: bool = True, create_dirs: bool = False
) -> None:
    root = Path(root).expanduser().absolute()
    for component in COMPONENTS:
        if os.environ.get(component.env_name) == BUILTIN_CONFIG_SENTINEL:
            continue
        component_dir = component.resolve(root, resolve_scopes).local_dir
        if create_dirs:
            component_dir.mkdir(parents=True, exist_ok=True)
        os.environ[component.env_name] = str(component_dir)


class JitCacheManager:
    def __init__(self, jit_config=None):
        from rtp_llm.config.py_config_modules import JITConfig

        self.jit_config = jit_config or JITConfig()
        self.remote_root = resolve_remote_root(self.jit_config.remote_jit_dir)
        self.components = (
            tuple(c.resolve(self.remote_root) for c in COMPONENTS)
            if self.remote_root is not None
            else tuple()
        )

    def bootstrap(self) -> None:
        start = time.monotonic()
        if self.remote_root is None:
            logging.info(
                "JIT cache bootstrap skipped: direct remote mode requires a valid remote_jit_dir duration_ms=%.2f",
                _duration_ms(start),
            )
            return
        logging.info(
            "JIT cache bootstrap start: mode=direct_remote root=%s",
            self.remote_root,
        )
        self.remote_root.mkdir(parents=True, exist_ok=True)
        apply_jit_cache_env(self.remote_root, create_dirs=True)
        logging.info(
            "JIT cache bootstrap done: mode=direct_remote root=%s duration_ms=%.2f component_envs=[%s]",
            self.remote_root,
            _duration_ms(start),
            _component_env_summary(),
        )

    def prepare(self) -> None:
        start = time.monotonic()
        if self.remote_root is None:
            logging.info(
                "JIT cache prepare skipped: direct remote cache disabled duration_ms=%.2f",
                _duration_ms(start),
            )
            return
        logging.info(
            "JIT cache prepare skipped: direct remote mode uses component cache paths directly root=%s duration_ms=%.2f",
            self.remote_root,
            _duration_ms(start),
        )

    def start_background_sync(self) -> None:
        return

    def stop(self) -> None:
        start = time.monotonic()
        if self.remote_root is None:
            logging.info(
                "JIT cache final stats skipped: direct remote cache disabled duration_ms=%.2f",
                _duration_ms(start),
            )
            return
        detailed = _detailed_stats_enabled()
        summary, visible_components = _component_visibility_summary(
            self.components, detailed=detailed
        )
        logging.info(
            "JIT cache final component stats: root=%s visible_components=%d/%d detailed_stats=%s duration_ms=%.2f component_state=[%s]",
            self.remote_root,
            visible_components,
            len(self.components),
            detailed,
            _duration_ms(start),
            summary,
        )

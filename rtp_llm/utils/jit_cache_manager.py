import logging
import os
from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from rtp_llm.utils.gpu_info import get_gpu_info, safe_part

BUILTIN_CONFIG_SENTINEL = "__builtin__"


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
    Component("torch_extensions", "TORCH_EXTENSIONS_DIR", _torch_extensions_scope),
    Component("triton", "TRITON_CACHE_DIR"),
    Component("triton_autotune", "TRITON_AUTOTUNE_CONFIG_DIR", get_gpu_info),
)


def resolve_remote_root(remote_jit_dir: Any) -> Path | None:
    text = str(remote_jit_dir or "").strip()
    if not text:
        return None
    if urlparse(text).scheme:
        from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local

        text = fetch_remote_file_to_local(text, MountRwMode.RWMODE_RW)
    path = Path(text).expanduser().absolute()
    if not path.is_dir():
        logging.warning(
            "JIT remote cache disabled: remote_jit_dir=%r resolved to %s which does not exist or is not a directory",
            remote_jit_dir,
            path,
        )
        return None
    return path


def apply_jit_cache_env(
    root: Path | str, resolve_scopes: bool = True, create_dirs: bool = False
) -> None:
    root = Path(root).expanduser().absolute()
    for component in COMPONENTS:
        if os.environ.get(component.env_name) == BUILTIN_CONFIG_SENTINEL:
            continue
        local_dir = component.resolve(root, resolve_scopes).local_dir
        if create_dirs:
            local_dir.mkdir(parents=True, exist_ok=True)
        os.environ[component.env_name] = str(local_dir)


class JitCacheManager:
    def __init__(self, jit_config=None):
        from rtp_llm.config.py_config_modules import JITConfig

        self.jit_config = jit_config or JITConfig()
        self.remote_root = resolve_remote_root(self.jit_config.remote_jit_dir)
        self.local_root = (
            self.remote_root
            if self.remote_root is not None
            else Path(self.jit_config.local_jit_dir).expanduser().absolute()
        )
        self.components = tuple(c.resolve(self.local_root) for c in COMPONENTS)

    def bootstrap(self) -> None:
        self.local_root.mkdir(parents=True, exist_ok=True)
        apply_jit_cache_env(self.local_root, create_dirs=True)
        if self.remote_root is not None:
            logging.info("using remote JIT cache dir directly: %s", self.remote_root)
        else:
            logging.info("using local JIT cache dir: %s", self.local_root)

    def prepare(self) -> None:
        pass

    def start_background_sync(self) -> None:
        pass

    def stop(self) -> None:
        pass

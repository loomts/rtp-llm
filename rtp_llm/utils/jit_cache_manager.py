import logging
import os
import shutil
import socket
import tarfile
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from watchdog.events import FileSystemEventHandler

from rtp_llm.utils.gpu_info import get_gpu_info, safe_part

BUILTIN_CONFIG_SENTINEL = "__builtin__"
SNAPSHOT_NAME = ".jit_snapshot.tar.zst"
REMOTE_SNAPSHOT_COMPACT_LOCK_DIR_NAME = ".jit_remote_snapshot_compact.lock.dir"
REMOTE_DELTA_DIR_NAME = ".delta"
LOCAL_COMPACT_WORK_DIR_NAME = ".jit_compacting"
SNAPSHOT_LOCK_STALE_S = 3600.0
STOP_JOIN_TIMEOUT_S, SYNC_POLL_S = 2.0, 300.0


@contextmanager
def zstd_tar(path: Path, mode: str) -> Iterator[tarfile.TarFile]:
    import zstandard as zstd

    reader = mode == "r"
    file_mode, tar_mode = ("rb", "r|") if reader else ("wb", "w|")
    codec = zstd.ZstdDecompressor() if reader else zstd.ZstdCompressor(level=3)
    zstd_stream = codec.stream_reader if reader else codec.stream_writer
    with path.open(file_mode) as raw, zstd_stream(raw) as zraw:
        with tarfile.open(fileobj=zraw, mode=tar_mode) as tar:
            yield tar


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
    sync_suffixes: tuple[str, ...]
    upload_events: frozenset[str]
    scope_func: Callable[[], str | None] | None = None
    local_dir: Path = Path()

    def resolve(self, root: Path, resolve_scopes: bool = True) -> "Component":
        scope = self.scope_func() if resolve_scopes and self.scope_func else None
        local_dir = root / self.name
        return replace(self, local_dir=local_dir / scope if scope else local_dir)

    def should_sync(self, rel: str) -> bool:
        return rel.endswith(self.sync_suffixes) and not (
            rel.startswith("tmp.pid_") or "/tmp.pid_" in rel
        )


COMPONENTS = (
    Component(
        "flashinfer",
        "FLASHINFER_WORKSPACE_BASE",
        (".so",),
        frozenset({"closed"}),
        _cuda_scope,
    ),
    Component(
        "deep_gemm",
        "DG_JIT_CACHE_DIR",
        ("kernel.cu", "kernel.cubin"),
        frozenset({"created"}),
        lambda: "deep_gemm-" + safe_part(_dist_version("deep_gemm")),
    ),
    Component(
        "torch_extensions",
        "TORCH_EXTENSIONS_DIR",
        (".so", "build.ninja", ".o"),
        frozenset({"closed"}),
        _torch_extensions_scope,
    ),
    Component(
        "triton",
        "TRITON_CACHE_DIR",
        (".json", ".cubin", ".hsaco"),
        frozenset({"moved"}),
    ),
    Component(
        "triton_autotune",
        "TRITON_AUTOTUNE_CONFIG_DIR",
        (".json",),
        frozenset({"closed"}),
        get_gpu_info,
    ),
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


def apply_jit_cache_env(local_root: Path | str, resolve_scopes: bool = True) -> None:
    root = Path(local_root).expanduser().absolute()
    for component in COMPONENTS:
        if os.environ.get(component.env_name) == BUILTIN_CONFIG_SENTINEL:
            continue
        os.environ[component.env_name] = str(
            component.resolve(root, resolve_scopes).local_dir
        )


def _new_delta_archive_name() -> str:
    host = safe_part(socket.gethostname(), "unknown_host")
    return f"{time.time_ns() // 1_000_000:016d}-{host}-{uuid.uuid4().hex[:16]}.tar.zst"


class RemoteSnapshotStore:
    def __init__(self, remote_root: Path):
        self.remote_root = remote_root

    @contextmanager
    def lock_remote(self) -> Iterator[bool]:
        lock_dir = self.remote_root / REMOTE_SNAPSHOT_COMPACT_LOCK_DIR_NAME
        locked = False
        try:
            lock_dir.mkdir()
            locked = True
        except FileExistsError:
            with suppress(OSError):
                # Steal the stale lock if the holder has been gone for one hour
                if time.time() - lock_dir.stat().st_mtime > SNAPSHOT_LOCK_STALE_S:
                    stale_dir = (
                        self.remote_root / f"{lock_dir.name}.{uuid.uuid4().hex}.stale"
                    )
                    lock_dir.rename(stale_dir)
                    try:
                        lock_dir.mkdir()
                        locked = True
                    finally:
                        with suppress(OSError):
                            stale_dir.rmdir()
        try:
            yield locked
        finally:
            if locked:
                with suppress(OSError):
                    lock_dir.rmdir()

    @contextmanager
    def atomic_write(self, path: Path) -> Iterator[Path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            yield tmp
            tmp.replace(path)
        except BaseException:
            with suppress(OSError):
                tmp.unlink()
            raise

    def pack(self, src_dir: Path, archive: Path) -> None:
        with zstd_tar(archive, "w") as tar:
            for path in sorted(filter(Path.is_file, src_dir.rglob("*"))):
                tar.add(path, arcname=path.relative_to(src_dir).as_posix())

    def _delta_archives(self) -> list[Path]:
        delta_dir = self.remote_root / REMOTE_DELTA_DIR_NAME
        return sorted(delta_dir.glob("*.tar.zst")) if delta_dir.is_dir() else []

    def _extract_all(self, sources: list[Path], target: Path) -> bool:
        base = target.resolve()
        extracted = False
        for source_archive in sources:
            try:
                with zstd_tar(source_archive, "r") as tar:
                    for member in tar:
                        dest = (base / member.name).resolve()
                        if not member.isfile() or not dest.is_relative_to(base):
                            continue
                        source = tar.extractfile(member)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with source, dest.open("wb") as out:
                            shutil.copyfileobj(source, out)
                        extracted = True
            except FileNotFoundError:
                continue
            except Exception:
                logging.warning(
                    "skipping unreadable archive %s", source_archive, exc_info=True
                )
                continue
        return extracted

    def restore(self, target: Path) -> bool:
        snapshot = self.remote_root / SNAPSHOT_NAME
        sources = ([snapshot] if snapshot.is_file() else []) + self._delta_archives()
        return bool(sources) and self._extract_all(sources, target)

    def publish_delta_archive(self, local_delta_dir: Path) -> None:
        delta_archive = (
            self.remote_root / REMOTE_DELTA_DIR_NAME / _new_delta_archive_name()
        )
        with self.atomic_write(delta_archive) as tmp_archive:
            self.pack(local_delta_dir, tmp_archive)

    def compact(self, work_dir: Path | None = None) -> None:
        snapshot_archive = self.remote_root / SNAPSHOT_NAME
        with self.lock_remote() as locked:
            if not locked:
                return
            delta_archives = self._delta_archives()
            if not delta_archives:
                return
            if work_dir is not None:
                work_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".jit_snapshot_", dir=work_dir
            ) as tmp_name:
                tmp_dir = Path(tmp_name)
                sources = (
                    [snapshot_archive] if snapshot_archive.is_file() else []
                ) + delta_archives
                self._extract_all(sources, tmp_dir)
                with self.atomic_write(snapshot_archive) as tmp_snapshot:
                    self.pack(tmp_dir, tmp_snapshot)
                for delta_archive in delta_archives:
                    delta_archive.unlink(missing_ok=True)


class _JitFileEventHandler(FileSystemEventHandler):
    def __init__(self, component: Component, stage_delta_file):
        self.component, self.stage_delta_file = component, stage_delta_file
        self.root_prefix = str(component.local_dir) + os.sep

    def on_any_event(self, event: Any) -> None:
        if event.is_directory or event.event_type not in self.component.upload_events:
            return
        src = event.dest_path if event.event_type == "moved" else event.src_path
        if src.startswith(self.root_prefix):
            rel = src[len(self.root_prefix) :].replace(os.sep, "/")
            self.stage_delta_file(self.component, rel)


class JitCacheManager:
    def __init__(self, jit_config=None):
        from rtp_llm.config.py_config_modules import JITConfig

        jit_config = jit_config or JITConfig()
        remote_root = resolve_remote_root(jit_config.remote_jit_dir)
        self.store = RemoteSnapshotStore(remote_root) if remote_root else None
        self.local_root = Path(jit_config.local_jit_dir).expanduser().absolute()
        self.components = tuple(c.resolve(self.local_root) for c in COMPONENTS)
        self.local_delta_dir = self.local_root / ".jit_delta"
        self.local_compact_dir = self.local_root / LOCAL_COMPACT_WORK_DIR_NAME
        self._observer: Any | None = None
        self._local_delta_lock = threading.Lock()
        self._stop = threading.Event()
        self._sync_thread: threading.Thread | None = None

    def bootstrap(self) -> None:
        self.local_root.mkdir(parents=True, exist_ok=True)
        apply_jit_cache_env(self.local_root)
        for path in (self.local_delta_dir, self.local_compact_dir):
            path.mkdir(parents=True, exist_ok=True)
        for component in self.components:
            component.local_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self) -> None:
        if self.store and self.store.restore(self.local_root):
            logging.info("loaded JIT cache from remote snapshot")

    def start_background_sync(self) -> None:
        if self.store is None or self._observer is not None:
            return
        from watchdog.observers import Observer

        observer = Observer()
        for component in self.components:
            observer.schedule(
                _JitFileEventHandler(component, self.stage_delta_file),
                str(component.local_dir),
                recursive=True,
            )
        observer.start()
        self._observer = observer
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def _sync_loop(self) -> None:
        while not self._stop.wait(SYNC_POLL_S):
            try:
                self.publish_staged_delta()
                self.store.compact(self.local_compact_dir)
            except Exception:
                logging.exception("JIT cache background sync failed")

    def stage_delta_file(self, component: Component, rel_path: str) -> None:
        if (
            self._stop.is_set()
            or self.store is None
            or not component.should_sync(rel_path)
        ):
            return
        path = component.local_dir / rel_path
        target = self.local_delta_dir / path.relative_to(self.local_root)
        tmp = self.local_delta_dir.with_name(f".jit_delta_tmp.{uuid.uuid4().hex}")
        try:
            if path.stat().st_size <= 0:
                return
            shutil.copy2(path, tmp)
            with self._local_delta_lock:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp.replace(target)
        except OSError:
            with suppress(OSError):
                tmp.unlink()

    def publish_staged_delta(self) -> None:
        delta = self.local_delta_dir
        # Swap the delta dir out under lock so staging can keep filling a fresh one.
        with self._local_delta_lock:
            if not any(delta.iterdir()):
                return
            batch_dir = delta.with_name(f"{delta.name}.{uuid.uuid4().hex}")
            delta.rename(batch_dir)
            try:
                delta.mkdir(parents=True, exist_ok=True)
            except OSError:
                with suppress(OSError):
                    batch_dir.rename(delta)
                raise
        try:
            self.store.publish_delta_archive(batch_dir)
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    def stop(self) -> None:
        self._stop.set()
        observer, self._observer = self._observer, None
        if observer is not None:
            observer.stop()
            observer.join(timeout=STOP_JOIN_TIMEOUT_S)
        thread, self._sync_thread = self._sync_thread, None
        if thread is not None:
            thread.join(timeout=STOP_JOIN_TIMEOUT_S)

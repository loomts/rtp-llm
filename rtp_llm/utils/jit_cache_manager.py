import logging
import os
import shutil
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

import zstandard as zstd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local
from rtp_llm.utils.gpu_info import get_gpu_info, safe_part

BUILTIN_CONFIG_SENTINEL = "__builtin__"
SNAPSHOT_NAME = ".jit_snapshot.tar.zst"
SNAPSHOT_LOCK_DIR_NAME = ".jit_snapshot.lock.dir"
DELTA_DIR_NAME = "delta"
SNAPSHOT_LOCK_POLL_S, SNAPSHOT_LOCK_TIMEOUT_S = 0.1, 60.0
WATCHDOG_JOIN_TIMEOUT_S, SYNC_POLL_S = 2.0, 120.0


@contextmanager
def zstd_tar(path: Path, mode: str) -> Iterator[tarfile.TarFile]:
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


@dataclass(frozen=True)
class Component:
    name: str
    env_name: str
    sync_suffixes: tuple[str, ...] = (".so", ".cubin", ".json")
    upload_events: frozenset[str] = frozenset({"closed"})
    scope_func: Callable[[], str | None] | None = None
    local_dir: Path = Path()

    def resolve(self, root: Path) -> "Component":
        scope = self.scope_func and self.scope_func()
        local_dir = root / self.name
        return replace(self, local_dir=local_dir / scope if scope else local_dir)

    def should_sync(self, rel: str) -> bool:
        return rel.endswith(self.sync_suffixes) and not (
            rel.startswith("tmp.pid_") or "/tmp.pid_" in rel
        )


COMPONENTS = (
    Component("flashinfer", "FLASHINFER_WORKSPACE_BASE", scope_func=_cuda_scope),
    Component(
        "deep_gemm",
        "DG_JIT_CACHE_DIR",
        upload_events=frozenset({"created", "moved"}),
        scope_func=lambda: "deep_gemm-" + safe_part(_dist_version("deep_gemm")),
    ),
    Component(
        "torch_extensions",
        "TORCH_EXTENSIONS_DIR",
        scope_func=lambda: "torch-" + safe_part(_dist_version("torch")),
    ),
    Component("triton", "TRITON_CACHE_DIR", upload_events=frozenset({"moved"})),
    Component(
        "triton_autotune",
        "TRITON_AUTOTUNE_CONFIG_DIR",
        sync_suffixes=(".json", ".pkl", ".pickle"),
        scope_func=get_gpu_info,
    ),
)


def resolve_remote_root(remote_jit_dir: Any) -> Path | None:
    text = str(remote_jit_dir or "").strip()
    if not text:
        return None
    if urlparse(text).scheme:
        text = fetch_remote_file_to_local(text, MountRwMode.RWMODE_RW)
    path = Path(text).expanduser().absolute()
    return path if path.is_dir() else None


def apply_jit_cache_env(local_root: Path | str) -> None:
    root = Path(local_root).expanduser().absolute()
    for component in COMPONENTS:
        # Leave an env opted out via the builtin sentinel (triton autotune) untouched.
        if os.environ.get(component.env_name) != BUILTIN_CONFIG_SENTINEL:
            os.environ[component.env_name] = str(component.resolve(root).local_dir)
    os.environ.setdefault("TRITON_AUTOTUNE_CACHE_MODE", "cached")


class RemoteSnapshotStore:
    def __init__(self, root: Path):
        self.root = root

    @contextmanager
    def lock(self) -> Iterator[None]:
        lock_dir = self.root / SNAPSHOT_LOCK_DIR_NAME
        deadline = time.monotonic() + SNAPSHOT_LOCK_TIMEOUT_S
        while True:
            try:
                lock_dir.mkdir()
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"failed to acquire snapshot lock: {lock_dir}")
                time.sleep(SNAPSHOT_LOCK_POLL_S)
        try:
            yield
        finally:
            lock_dir.rmdir()

    @contextmanager
    def atomic_write(self, path: Path) -> Iterator[Path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            yield tmp
            tmp.replace(path)
        finally:
            with suppress(OSError):
                tmp.unlink()

    def pack(self, src_dir: Path, archive: Path) -> None:
        with zstd_tar(archive, "w") as tar:
            for path in sorted(filter(Path.is_file, src_dir.rglob("*"))):
                tar.add(path, arcname=path.relative_to(src_dir).as_posix())

    def _sorted_delta_archives(self) -> list[Path]:
        delta_dir = self.root / DELTA_DIR_NAME
        if not delta_dir.is_dir():
            return []
        # Oldest first, so newer entries overwrite older ones on extraction.
        return sorted(delta_dir.glob("*.tar.zst"), key=lambda p: p.stat().st_mtime_ns)

    def _extract_all(self, sources: list[Path], target: Path) -> None:
        for source_archive in sources:
            if source_archive.is_file():
                with zstd_tar(source_archive, "r") as tar:
                    tar.extractall(path=target)

    def restore(self, target: Path) -> bool:
        # The compacted snapshot plus any deltas not yet folded into it.
        snapshot = self.root / SNAPSHOT_NAME
        head = [snapshot] if snapshot.is_file() else []
        sources = head + self._sorted_delta_archives()
        if not sources:
            return False
        self._extract_all(sources, target)
        return True

    def publish_delta(self, local_delta_dir: Path) -> None:
        delta_archive = self.root / DELTA_DIR_NAME / f"{uuid.uuid4().hex}.tar.zst"
        with self.atomic_write(delta_archive) as tmp_archive:
            self.pack(local_delta_dir, tmp_archive)

    def compact(self) -> None:
        snapshot_archive = self.root / SNAPSHOT_NAME
        with self.lock():
            delta_archives = self._sorted_delta_archives()
            if not delta_archives:
                return
            with tempfile.TemporaryDirectory(prefix=".jit_snapshot_") as tmp_name:
                merged = Path(tmp_name) / "merged"
                merged.mkdir()
                self._extract_all([snapshot_archive, *delta_archives], merged)
                with self.atomic_write(snapshot_archive) as tmp_snapshot:
                    self.pack(merged, tmp_snapshot)
                for delta_archive in delta_archives:
                    delta_archive.unlink()


class _JitFileEventHandler(FileSystemEventHandler):
    def __init__(self, component: Component, mark_dirty):
        self.component, self.mark_dirty = component, mark_dirty
        self.root_prefix = str(component.local_dir) + os.sep

    def on_any_event(self, event: Any) -> None:
        if event.is_directory or event.event_type not in self.component.upload_events:
            return
        src = event.dest_path if event.event_type == "moved" else event.src_path
        if src.startswith(self.root_prefix):
            rel = src[len(self.root_prefix) :].replace(os.sep, "/")
            self.mark_dirty(self.component, rel)


class JitCacheManager:
    def __init__(self, jit_config=None):
        jit_config = jit_config or JITConfig()
        remote_root = resolve_remote_root(jit_config.remote_jit_dir)
        self.store = RemoteSnapshotStore(remote_root) if remote_root else None
        self.local_root = Path(jit_config.local_jit_dir).expanduser().absolute()
        self.components = tuple(c.resolve(self.local_root) for c in COMPONENTS)
        self.local_delta_dir = self.local_root / ".jit_delta"
        self._observer: Any | None = None
        self._delta_lock = threading.Lock()
        self._stop = threading.Event()
        self._sync_thread: threading.Thread | None = None

    def bootstrap(self) -> None:
        self.local_root.mkdir(parents=True, exist_ok=True)
        apply_jit_cache_env(self.local_root)
        self.local_delta_dir.mkdir(parents=True, exist_ok=True)
        for component in self.components:
            component.local_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self) -> None:
        if self.store and self.store.restore(self.local_root):
            logging.info("loaded JIT cache from remote snapshot")

    def start_background_sync(self) -> None:
        if self.store is None or self._observer is not None:
            return
        observer = Observer()
        for component in self.components:
            handler = _JitFileEventHandler(component, self.mark_dirty)
            observer.schedule(handler, str(component.local_dir), recursive=True)
        observer.start()
        self._observer = observer
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def _sync_loop(self) -> None:
        assert self.store is not None
        # Each poll: publish staged deltas, then fold them into the snapshot.
        while not self._stop.wait(SYNC_POLL_S):
            try:
                self._publish_delta()
                self.store.compact()
            except Exception:
                logging.exception("JIT cache background sync failed")

    def mark_dirty(self, component: Component, rel_path: str) -> bool:
        if (
            self._stop.is_set()
            or self.store is None
            or not component.should_sync(rel_path)
        ):
            return False
        path = component.local_dir / rel_path
        target = self.local_delta_dir / path.relative_to(self.local_root).as_posix()
        with self._delta_lock:
            # JIT temp files churn, so a vanished/empty source is skipped, not fatal.
            try:
                if path.stat().st_size <= 0:
                    return False
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            except OSError:
                return False
        return True

    def _publish_delta(self) -> None:
        assert self.store is not None
        delta = self.local_delta_dir
        # Swap the delta dir out under lock so staging can keep filling a fresh one.
        with self._delta_lock:
            if not any(delta.iterdir()):
                return
            batch_dir = delta.with_name(f"{delta.name}.{uuid.uuid4().hex}")
            delta.rename(batch_dir)
            delta.mkdir(parents=True, exist_ok=True)
        try:
            self.store.publish_delta(batch_dir)
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    def stop(self) -> None:
        self._stop.set()
        observer, self._observer = self._observer, None
        if observer is not None:
            observer.stop()
            observer.join(timeout=WATCHDOG_JOIN_TIMEOUT_S)
        thread, self._sync_thread = self._sync_thread, None
        if thread is not None:
            thread.join(timeout=WATCHDOG_JOIN_TIMEOUT_S)

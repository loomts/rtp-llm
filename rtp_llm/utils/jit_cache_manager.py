import logging
import os
import re
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
from rtp_llm.models_py.triton_kernels.autotune_cache.cache import get_gpu_info
from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local
from rtp_llm.utils.time_util import current_time_ms, elapsed_ms

SNAPSHOT_NAME = ".jit_snapshot.tar.zst"
SNAPSHOT_LOCK_DIR_NAME = ".jit_snapshot.lock.dir"
DELTA_DIR_NAME = "delta"
SNAPSHOT_LOCK_POLL_S, SNAPSHOT_LOCK_TIMEOUT_S = 0.1, 60.0
WATCHDOG_JOIN_TIMEOUT_S, WATCHDOG_PUBLISH_DELAY_S = 2.0, 60.0


@contextmanager
def zstd_tar(path: Path, mode: str) -> Iterator[tarfile.TarFile]:
    reader = mode == "r"
    file_mode, tar_mode = ("rb", "r|") if reader else ("wb", "w|")
    codec = zstd.ZstdDecompressor() if reader else zstd.ZstdCompressor(level=3)
    zstd_stream = codec.stream_reader if reader else codec.stream_writer
    with (
        path.open(file_mode) as raw,
        zstd_stream(raw) as zraw,
        tarfile.open(fileobj=zraw, mode=tar_mode) as tar,
    ):
        yield tar


@contextmanager
def snapshot_mklock(remote_root: Path) -> Iterator[None]:
    lock_dir = remote_root / SNAPSHOT_LOCK_DIR_NAME
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
def remote_tmp_path(remote_path: Path) -> Iterator[Path]:
    remote_path.parent.mkdir(parents=True, exist_ok=True)
    remote_tmp = remote_path.with_name(f"{remote_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        yield remote_tmp
        remote_tmp.replace(remote_path)
    finally:
        with suppress(OSError):
            remote_tmp.unlink()


def pack_tree(src_dir: Path, archive: Path) -> None:
    with zstd_tar(archive, "w") as tar:
        for path in sorted(filter(Path.is_file, src_dir.rglob("*"))):
            tar.add(path, arcname=path.relative_to(src_dir).as_posix(), recursive=False)


def extract_archive(archive: Path, target: Path) -> tuple[int, int]:
    bytes_, files = 0, 0
    with zstd_tar(archive, "r") as tar:
        for member in tar:
            tar.extract(member, path=target)
            if member.isfile():
                files, bytes_ = files + 1, bytes_ + member.size
    return bytes_, files


def publish_delta(remote_delta_dir: Path, local_delta_dir: Path) -> None:
    remote_delta_dir.mkdir(parents=True, exist_ok=True)
    delta_archive = remote_delta_dir / f"{uuid.uuid4().hex}.tar.zst"
    with remote_tmp_path(delta_archive) as remote_tmp:
        pack_tree(local_delta_dir, remote_tmp)


def compact_snapshot(remote_root: Path) -> None:
    snapshot_archive = remote_root / SNAPSHOT_NAME
    delta_dir = remote_root / DELTA_DIR_NAME
    with snapshot_mklock(remote_root):
        delta_archives = (
            sorted(
                delta_dir.glob("*.tar.zst"), key=lambda path: path.stat().st_mtime_ns
            )
            if delta_dir.is_dir()
            else []
        )
        if not delta_archives:
            return
        with tempfile.TemporaryDirectory(prefix=".jit_snapshot_") as tmp_name:
            merged = Path(tmp_name) / "merged"
            merged.mkdir()
            for source_archive in (snapshot_archive, *delta_archives):
                if source_archive.is_file():
                    extract_archive(source_archive, merged)
            with remote_tmp_path(snapshot_archive) as remote_tmp:
                pack_tree(merged, remote_tmp)
            for delta_archive in delta_archives:
                with suppress(OSError):
                    delta_archive.unlink()


@dataclass(frozen=True)
class Component:
    name: str
    env_name: str
    sync_suffixes: tuple[str, ...] = (".so", ".cubin", ".json")
    upload_events: frozenset[str] = frozenset({"closed"})
    scope_func: Callable[[], str | None] | None = None
    local_dir: Path | None = None

    def resolve(self, root: Path) -> "Component":
        scope = self.scope_func and self.scope_func()
        local_dir = root / self.name
        return replace(self, local_dir=local_dir / scope if scope else local_dir)


def _safe_part(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or "unknown"


def _dist_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _cuda_scope() -> str:
    import torch

    return "cuda-" + _safe_part(str(torch.version.cuda or "unknown"))


COMPONENTS = (
    Component("flashinfer", "FLASHINFER_WORKSPACE_BASE", scope_func=_cuda_scope),
    Component(
        "deep_gemm",
        "DG_JIT_CACHE_DIR",
        upload_events=frozenset({"created", "moved"}),
        scope_func=lambda: "deep_gemm-" + _safe_part(_dist_version("deep_gemm")),
    ),
    Component(
        "torch_extensions",
        "TORCH_EXTENSIONS_DIR",
        scope_func=lambda: "torch-" + _safe_part(_dist_version("torch")),
    ),
    Component("triton", "TRITON_CACHE_DIR", upload_events=frozenset({"moved"})),
    Component(
        "triton_autotune",
        "TRITON_AUTOTUNE_CONFIG_DIR",
        sync_suffixes=(".json", ".pkl", ".pickle"),
        scope_func=get_gpu_info,
    ),
)


def should_sync_file(component: Component, rel: str) -> bool:
    return rel.endswith(component.sync_suffixes) and not (
        rel.startswith("tmp.pid_") or "/tmp.pid_" in rel
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
    for component in (component.resolve(root) for component in COMPONENTS):
        os.environ[component.env_name] = str(component.local_dir)
    os.environ.setdefault("TRITON_AUTOTUNE_CACHE_MODE", "cached")


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
            if should_sync_file(self.component, rel):
                self.mark_dirty(self.component, rel)


class JitCacheManager:
    def __init__(self, jit_config=None, *, publish_delay_s=WATCHDOG_PUBLISH_DELAY_S):
        jit_config = jit_config or JITConfig()
        self.remote_root = resolve_remote_root(jit_config.remote_jit_dir)
        self.local_root = Path(jit_config.local_jit_dir).expanduser().absolute()
        self.components: tuple[Component, ...] = ()
        self._observer: Any | None = None
        self._stopping = False
        self._publish_delay_s = publish_delay_s
        self._publish_lock = threading.Lock()
        self._publish_timer: threading.Timer | None = None
        self._delta_lock = threading.Lock()
        self.local_delta_dir = self.local_root / ".jit_delta"

    def summary(self, mode, result, start_s, **extra):
        return {
            "timestamp_ms": int(current_time_ms()),
            "mode": mode,
            "result": result,
            "total_cost_ms": elapsed_ms(start_s),
            **extra,
        }

    def bootstrap(self) -> None:
        self.local_root.mkdir(parents=True, exist_ok=True)
        apply_jit_cache_env(self.local_root)
        self.components = tuple(
            component.resolve(self.local_root) for component in COMPONENTS
        )
        self.local_delta_dir.mkdir(parents=True, exist_ok=True)
        for component in self.components:
            component.local_dir.mkdir(parents=True, exist_ok=True)
        logging.info("JIT cache local=%s remote=%s", self.local_root, self.remote_root)

    def prepare(self) -> dict[str, Any]:
        start_s = time.monotonic()
        done = lambda result, **extra: self.summary(
            "snapshot_download", result, start_s, **extra
        )

        if self.remote_root is None:
            return done("skipped", cache_state="disabled")
        archive = self.remote_root / SNAPSHOT_NAME
        try:
            if not archive.is_file():
                return done("skipped", cache_state="snapshot_miss")
            extracted_bytes, extracted_files = extract_archive(archive, self.local_root)
        except Exception as e:
            logging.exception("failed to extract JIT cache snapshot")
            return done("failed", cache_state="snapshot_error", message=str(e))
        return done(
            "success",
            cache_state="snapshot_hit",
            extracted_files=extracted_files,
            extracted_bytes=extracted_bytes,
        )

    def start_background_sync(self) -> None:
        if self.remote_root is None or self._observer is not None:
            return
        observer = Observer()
        for component in self.components:
            observer.schedule(
                _JitFileEventHandler(component, self.mark_dirty),
                str(component.local_dir),
                recursive=True,
            )
        observer.start()
        self._observer = observer

    def mark_dirty(self, component: Component, rel_path: str) -> bool:
        if self._stopping or self.remote_root is None or component.local_dir is None:
            return False
        try:
            if not self._stage_delta_file(component, rel_path):
                return False
        except FileNotFoundError:
            return False
        self._schedule_publish()
        return True

    def _stage_delta_file(self, component: Component, rel_path: str) -> bool:
        local_dir = component.local_dir
        path = local_dir / rel_path
        if path.stat().st_size <= 0:
            return False
        arcname = f"{local_dir.relative_to(self.local_root).as_posix()}/{rel_path}"
        target = self.local_delta_dir / arcname
        with self._delta_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        return True

    def _schedule_publish(self) -> None:
        if self._publish_delay_s <= 0:
            self._sync_and_maybe_compact()
            return

        def fire() -> None:
            with self._publish_lock:
                self._publish_timer = None
            if not self._stopping:
                self._sync_and_maybe_compact()

        with self._publish_lock:
            if self._stopping or self._publish_timer is not None:
                return
            self._publish_timer = threading.Timer(self._publish_delay_s, fire)
            self._publish_timer.daemon = True
            self._publish_timer.start()

    def sync_once(self, mode: str = "manual_sync") -> dict[str, Any]:
        start_s = time.monotonic()
        if self._stopping:
            return self.summary(mode, "skipped", start_s, reason="stopping")
        if self.remote_root is None:
            return self.summary(mode, "skipped", start_s, cache_state="disabled")
        with self._publish_lock:
            if self._publish_timer is not None:
                self._publish_timer.cancel()
                self._publish_timer = None
        try:
            self._sync_and_maybe_compact(force_compact=True, scan_local=True)
            return self.summary(mode, "success", start_s)
        except Exception as e:
            logging.exception("failed to publish JIT cache snapshot")
            return self.summary(mode, "failed", start_s, message=str(e))

    def _sync_and_maybe_compact(self, *, force_compact=False, scan_local=False):
        if self.remote_root is None:
            return
        start_s = time.monotonic()
        if scan_local:
            self._mark_all_local_delta()
        uploaded = False
        with self._delta_lock:
            batch_dir = self.local_delta_dir.with_name(
                f"{self.local_delta_dir.name}.{uuid.uuid4().hex}"
            )
            if self.local_delta_dir.is_dir() and any(self.local_delta_dir.iterdir()):
                self.local_delta_dir.rename(batch_dir)
                self.local_delta_dir.mkdir(parents=True, exist_ok=True)
            else:
                batch_dir = None
        if batch_dir is not None:
            try:
                publish_delta(self.remote_root / DELTA_DIR_NAME, batch_dir)
                uploaded = True
            finally:
                shutil.rmtree(batch_dir, ignore_errors=True)
        if force_compact:
            compact_snapshot(self.remote_root)
        logging.info(
            "published JIT cache delta=%s compact=%s cost_ms=%d",
            uploaded,
            force_compact,
            elapsed_ms(start_s),
        )

    def _mark_all_local_delta(self) -> None:
        for component in self.components:
            root = component.local_dir
            if root is None or not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [name for name in dirnames if name != "__pycache__"]
                rel_dir = Path(dirpath).relative_to(root).as_posix()
                prefix = "" if rel_dir == "." else rel_dir + "/"
                for filename in filenames:
                    path, rel = Path(dirpath) / filename, prefix + filename
                    with suppress(OSError):
                        if (
                            should_sync_file(component, rel)
                            and path.stat(follow_symlinks=False).st_size > 0
                        ):
                            self._stage_delta_file(component, rel)

    def stop(self) -> None:
        self._stopping = True
        with self._publish_lock:
            if self._publish_timer is not None:
                self._publish_timer.cancel()
                self._publish_timer = None
        observer, self._observer = self._observer, None
        if observer is not None:
            observer.stop()
            observer.join(timeout=WATCHDOG_JOIN_TIMEOUT_S)

import os
import tarfile
import tempfile
import threading
import unittest
from contextlib import suppress
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest import mock

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.utils import jit_cache_manager as jit_cache_module
from rtp_llm.utils.fuser import MountRwMode
from rtp_llm.utils.jit_cache_manager import JitCacheManager, zstd_tar


class FakeFileEvent:
    def __init__(self, event_type: str, src_path: str, dest_path: str = ""):
        self.event_type = event_type
        self.src_path = src_path
        self.dest_path = dest_path
        self.is_directory = False


def write_snapshot(remote: Path) -> Path:
    snapshot = remote / jit_cache_module.SNAPSHOT_NAME
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with zstd_tar(snapshot, "w") as tar:
        for component in (
            component.resolve(remote) for component in jit_cache_module.COMPONENTS
        ):
            for path, rel in iter_sync_files(component):
                arcname = f"{component.local_dir.relative_to(remote).as_posix()}/{rel}"
                tar.add(str(path), arcname=arcname, recursive=False)
    return snapshot


def write_raw_snapshot(remote: Path, entries: dict[str, bytes | None]) -> Path:
    snapshot = remote / jit_cache_module.SNAPSHOT_NAME
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with zstd_tar(snapshot, "w") as tar:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            if payload is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(payload)
                tar.addfile(info, BytesIO(payload))
    return snapshot


def write_raw_delta(remote: Path, filename: str, entries: dict[str, bytes]) -> Path:
    delta = remote / jit_cache_module.DELTA_DIR_NAME / filename
    delta.parent.mkdir(parents=True, exist_ok=True)
    with zstd_tar(delta, "w") as tar:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, BytesIO(payload))
    return delta


def snapshot_path(remote: Path) -> Path:
    return remote / jit_cache_module.SNAPSHOT_NAME


def delta_paths(remote: Path) -> list[Path]:
    return sorted((remote / jit_cache_module.DELTA_DIR_NAME).glob("*.tar.zst"))


def snapshot_members(snapshot_path: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with zstd_tar(snapshot_path, "r") as tar:
        for member in tar:
            source = tar.extractfile(member)
            if source is not None:
                with source:
                    members[member.name] = source.read()
    return members


def effective_members(remote: Path) -> dict[str, bytes]:
    """What a consumer sees after restore(): the compacted snapshot overlaid
    by pending delta archives (oldest first, newest wins). Compaction is
    throttled, so entries usually live in deltas, not the snapshot."""
    store = jit_cache_module.RemoteSnapshotStore(remote)
    sources: list[Path] = []
    snapshot = snapshot_path(remote)
    if snapshot.is_file():
        sources.append(snapshot)
    sources += store._sorted_delta_archives()
    members: dict[str, bytes] = {}
    for source in sources:
        members.update(snapshot_members(source))
    return members


def effective_member_names(remote: Path) -> set[str]:
    return set(effective_members(remote))


def clear_jit_env() -> None:
    for component in jit_cache_module.COMPONENTS:
        os.environ.pop(component.env_name, None)
    os.environ.pop("TRITON_AUTOTUNE_CACHE_MODE", None)


def component_by_name(components, name: str):
    return next(component for component in components if component.name == name)


def iter_sync_files(component):
    root = component.local_dir
    if root is None or not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name != "__pycache__"]
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else rel_dir + "/"
        for filename in filenames:
            path, rel = Path(dirpath) / filename, prefix + filename
            with suppress(OSError):
                if (
                    component.should_sync(rel)
                    and path.stat(follow_symlinks=False).st_size > 0
                ):
                    yield path, rel


def make_manager(
    root: Path,
    remote: str = "",
    *,
    local_root: Path | None = None,
    create_remote: bool = True,
) -> JitCacheManager:
    if remote and create_remote:
        Path(remote).mkdir(parents=True, exist_ok=True)
    config = JITConfig()
    config.local_jit_dir = str(local_root or root / "local")
    config.remote_jit_dir = remote
    manager = JitCacheManager(config)
    manager.bootstrap()
    return manager


class JitCacheManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        clear_jit_env()
        self.managers: list[JitCacheManager] = []

    def tearDown(self):
        for manager in self.managers:
            manager.stop()
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tmp.cleanup()

    def make_manager(
        self,
        remote: str = "",
        *,
        local_root: Path | None = None,
        create_remote: bool = True,
    ) -> JitCacheManager:
        manager = make_manager(
            self.root,
            remote,
            local_root=local_root,
            create_remote=create_remote,
        )
        self.managers.append(manager)
        return manager

    def make_remote_manager(self, **kwargs) -> tuple[Path, JitCacheManager]:
        remote = self.root / "remote"
        return remote, self.make_manager(str(remote), **kwargs)

    def component_dir(self, root: Path, name: str) -> Path:
        return next(
            component.resolve(root).local_dir
            for component in jit_cache_module.COMPONENTS
            if component.name == name
        )

    def mark_dirty_helper(
        self, manager: JitCacheManager, component_name: str, rel: str
    ):
        component = component_by_name(manager.components, component_name)
        local_root = component.local_dir
        path = local_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(component_name, encoding="utf-8")
        self.assertTrue(manager.mark_dirty(component, rel))
        manager._publish_delta()
        return path

    def test_bootstrap_creates_managed_components(self):
        remote, manager = self.make_remote_manager()

        self.assertIsNotNone(manager.store)
        self.assertEqual(
            {component.name for component in manager.components},
            {component.name for component in jit_cache_module.COMPONENTS},
        )
        self.assertEqual(manager.local_delta_dir, manager.local_root / ".jit_delta")
        self.assertTrue(manager.local_delta_dir.is_dir())
        for component in jit_cache_module.COMPONENTS:
            resolved = component_by_name(manager.components, component.name)
            self.assertTrue(
                self.component_dir(self.root / "local", component.name).is_dir()
            )
            self.assertEqual(
                resolved.local_dir,
                self.component_dir(self.root / "local", component.name),
            )

    def test_apply_jit_cache_env_overwrites_component_envs(self):
        custom_triton = self.root / "custom_triton"
        os.environ["TRITON_CACHE_DIR"] = str(custom_triton)

        jit_cache_module.apply_jit_cache_env(self.root / "local")

        self.assertEqual(
            os.environ["TRITON_CACHE_DIR"],
            str(self.component_dir(self.root / "local", "triton")),
        )

    def test_apply_jit_cache_env_preserves_builtin_autotune_config_sentinel(self):
        os.environ["TRITON_AUTOTUNE_CONFIG_DIR"] = (
            jit_cache_module.BUILTIN_CONFIG_SENTINEL
        )

        jit_cache_module.apply_jit_cache_env(self.root / "local")

        self.assertEqual(
            os.environ["TRITON_AUTOTUNE_CONFIG_DIR"],
            jit_cache_module.BUILTIN_CONFIG_SENTINEL,
        )

    def test_prepare_without_remote_is_disabled(self):
        manager = self.make_manager()

        self.assertIsNone(manager.prepare())

    def test_prepare_miss_skips_snapshot_download(self):
        _remote, manager = self.make_remote_manager()

        self.assertIsNone(manager.prepare())

    def test_prepare_extracts_fixed_snapshot_every_boot_and_overwrites_local(self):
        remote = self.root / "remote"
        local = self.root / "local"
        first = self.make_manager(str(remote), local_root=local)
        remote_file = self.component_dir(remote, "triton") / "kernel/a.so"
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        remote_file.write_text("first", encoding="utf-8")
        write_snapshot(remote)

        first.prepare()
        target = self.component_dir(local, "triton") / "kernel/a.so"
        self.assertEqual(target.read_text(encoding="utf-8"), "first")

        remote_file.write_text("second", encoding="utf-8")
        write_snapshot(remote)
        second = self.make_manager(str(remote), local_root=local)
        second.prepare()
        self.assertEqual(target.read_text(encoding="utf-8"), "second")

    def test_prepare_keeps_all_snapshot_member_paths(self):
        remote, manager = self.make_remote_manager()
        write_raw_snapshot(
            remote,
            {
                "deep_gemm/cache/old.cubin": b"old",
                "triton/kernel/keep.so": b"keep",
            },
        )

        manager.prepare()
        self.assertEqual(
            (manager.local_root / "deep_gemm/cache/old.cubin").read_text(),
            "old",
        )
        self.assertEqual(
            (manager.local_root / "triton/kernel/keep.so").read_text(encoding="utf-8"),
            "keep",
        )

    def test_mark_dirty_publishes_snapshot_without_writing_remote_file_tree(self):
        remote, manager = self.make_remote_manager()
        component = component_by_name(manager.components, "triton")
        local_root = component.local_dir
        remote_root = self.component_dir(remote, "triton")
        local_file = local_root / "kernel/a.so"
        remote_file = remote_root / "kernel/a.so"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        remote_file.write_text("old", encoding="utf-8")
        local_file.write_text("new", encoding="utf-8")

        self.assertTrue(manager.mark_dirty(component, "kernel/a.so"))
        manager._publish_delta()

        self.assertEqual(remote_file.read_text(encoding="utf-8"), "old")
        self.assertEqual(effective_members(remote), {"triton/kernel/a.so": b"new"})

    def test_mark_dirty_uses_passed_component_root(self):
        _remote, manager = self.make_remote_manager()
        component = component_by_name(manager.components, "triton")
        local_file = component.local_dir / "kernel/a.so"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text("so", encoding="utf-8")
        manager.components = tuple(
            (
                replace(c, local_dir=self.root / "new_local" / "triton")
                if c.name == "triton"
                else c
            )
            for c in manager.components
        )

        self.assertTrue(manager.mark_dirty(component, "kernel/a.so"))

        staged = manager.local_delta_dir / "triton/kernel/a.so"
        self.assertEqual(staged.read_text(encoding="utf-8"), "so")

    def test_publish_failure_drops_batch(self):
        # A store publish failure raises and the staged batch is dropped.
        _remote, manager = self.make_remote_manager()
        component = component_by_name(manager.components, "triton")
        local_file = component.local_dir / "kernel/a.so"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text("so", encoding="utf-8")

        self.assertTrue(manager.mark_dirty(component, "kernel/a.so"))

        with mock.patch.object(
            manager.store, "publish_delta", side_effect=OSError("publish failed")
        ):
            with self.assertRaises(OSError):
                manager._publish_delta()
        self.assertFalse(any(manager.local_delta_dir.iterdir()))

    def test_delta_upload_publishes_delta_and_clears_local_delta(self):
        remote, manager = self.make_remote_manager()

        self.mark_dirty_helper(manager, "triton", "kernel/a.so")
        self.assertEqual(len(delta_paths(remote)), 1)
        self.assertEqual(effective_members(remote), {"triton/kernel/a.so": b"triton"})
        self.assertFalse(any(manager.local_delta_dir.iterdir()))
        self.mark_dirty_helper(manager, "triton", "kernel/b.so")

        self.assertEqual(len(delta_paths(remote)), 2)
        self.assertEqual(
            effective_members(remote),
            {"triton/kernel/a.so": b"triton", "triton/kernel/b.so": b"triton"},
        )
        self.assertFalse(any(manager.local_delta_dir.iterdir()))

    def test_publish_does_not_compact(self):
        # Compaction runs on the background poll, never inline with publish.
        remote, manager = self.make_remote_manager()
        for i in range(3):
            self.mark_dirty_helper(manager, "triton", f"kernel/k{i}.so")
        self.assertEqual(len(delta_paths(remote)), 3)
        self.assertFalse(snapshot_path(remote).is_file())

    def test_delta_upload_does_not_block_new_staging(self):
        remote, manager = self.make_remote_manager()
        self.mark_dirty_helper(manager, "triton", "kernel/a.so")
        publish_started = threading.Event()
        release_publish = threading.Event()
        original_publish_delta = manager.store.publish_delta

        def blocking_publish_delta(local_delta_dir):
            publish_started.set()
            self.assertTrue(release_publish.wait(timeout=5))
            return original_publish_delta(local_delta_dir)

        with mock.patch.object(
            manager.store, "publish_delta", side_effect=blocking_publish_delta
        ):
            publish_thread = threading.Thread(target=manager._publish_delta)
            publish_thread.start()
            self.assertTrue(publish_started.wait(timeout=5))

            self.mark_dirty_helper(manager, "triton", "kernel/b.so")
            staged = manager.local_delta_dir / "triton/kernel/b.so"
            self.assertEqual(staged.read_text(encoding="utf-8"), "triton")

            release_publish.set()
            publish_thread.join(timeout=5)

        self.assertFalse(publish_thread.is_alive())
        self.assertEqual(effective_members(remote), {"triton/kernel/a.so": b"triton"})
        self.assertEqual(
            (manager.local_delta_dir / "triton/kernel/b.so").read_text(
                encoding="utf-8"
            ),
            "triton",
        )

    def test_staged_delta_survives_source_removal(self):
        remote, manager = self.make_remote_manager()
        component = component_by_name(manager.components, "triton")
        local_file = component.local_dir / "kernel/a.so"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text("so", encoding="utf-8")
        self.assertTrue(manager.mark_dirty(component, "kernel/a.so"))
        local_file.unlink()

        manager._publish_delta()

        self.assertEqual(effective_members(remote), {"triton/kernel/a.so": b"so"})
        self.assertFalse(any(manager.local_delta_dir.iterdir()))

    def test_publish_accumulates_deltas_without_snapshot(self):
        remote, manager = self.make_remote_manager()
        manager.prepare()
        self.mark_dirty_helper(manager, "triton", "kernel/a.so")
        self.mark_dirty_helper(manager, "triton", "kernel/b.so")

        # Publishing accumulates deltas only; the top-level snapshot is written
        # solely by compaction (the background poll), never inline with publish.
        self.assertFalse(snapshot_path(remote).is_file())
        self.assertEqual(len(delta_paths(remote)), 2)
        self.assertEqual(
            effective_members(remote),
            {"triton/kernel/a.so": b"triton", "triton/kernel/b.so": b"triton"},
        )

    def test_snapshot_publish_keeps_remote_only_members_and_local_wins(self):
        remote, manager = self.make_remote_manager()
        write_raw_snapshot(
            remote,
            {
                "triton/kernel/a.so": b"remote",
                "triton/kernel/other.so": b"other",
            },
        )
        self.mark_dirty_helper(manager, "triton", "kernel/a.so")
        self.mark_dirty_helper(manager, "triton", "kernel/b.so")

        self.assertEqual(
            effective_members(remote),
            {
                "triton/kernel/a.so": b"triton",
                "triton/kernel/other.so": b"other",
                "triton/kernel/b.so": b"triton",
            },
        )

    def test_snapshot_compact_merges_deltas(self):
        remote = self.root / "remote"
        remote.mkdir()
        write_raw_snapshot(remote, {"triton/kernel/base.so": b"base"})
        write_raw_delta(remote, "a.tar.zst", {"triton/kernel/a.so": b"a"})
        write_raw_delta(remote, "b.tar.zst", {"triton/kernel/b.so": b"b"})

        jit_cache_module.RemoteSnapshotStore(remote).compact()

        self.assertEqual(
            snapshot_members(snapshot_path(remote)),
            {
                "triton/kernel/base.so": b"base",
                "triton/kernel/a.so": b"a",
                "triton/kernel/b.so": b"b",
            },
        )
        self.assertFalse(delta_paths(remote))

    def test_snapshot_compact_applies_deltas_by_publish_time(self):
        remote = self.root / "remote"
        remote.mkdir()
        old_delta = write_raw_delta(
            remote, "z_old.tar.zst", {"triton/kernel/a.so": b"old"}
        )
        new_delta = write_raw_delta(
            remote, "a_new.tar.zst", {"triton/kernel/a.so": b"new"}
        )
        os.utime(old_delta, ns=(100, 100))
        os.utime(new_delta, ns=(200, 200))

        jit_cache_module.RemoteSnapshotStore(remote).compact()

        self.assertEqual(
            snapshot_members(snapshot_path(remote)), {"triton/kernel/a.so": b"new"}
        )

    def test_watcher_marks_component_specific_completion_events(self):
        _remote, manager = self.make_remote_manager()
        calls = []
        cases = (
            ("flashinfer", "kernel.so", "created", "closed"),
            ("torch_extensions", "extension.so", "moved", "closed"),
            ("triton", "kernel/a.so", "closed", "moved"),
            ("triton_autotune", "kernel.json", "created", "closed"),
            ("deep_gemm", "cache/kernel.cubin", "closed", "created"),
            ("deep_gemm", "cache/kernel2.cubin", "closed", "moved"),
        )
        for component_name, rel, ignored_event, upload_event in cases:
            component = component_by_name(manager.components, component_name)
            root = component.local_dir
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x", encoding="utf-8")
            handler = jit_cache_module._JitFileEventHandler(
                component,
                lambda component, rel: calls.append((component.name, rel)) or True,
            )
            before = len(calls)

            handler.on_any_event(FakeFileEvent(ignored_event, str(path)))
            self.assertEqual(len(calls), before)

            if upload_event == "moved":
                event = FakeFileEvent("moved", str(path.with_suffix(".tmp")), str(path))
            else:
                event = FakeFileEvent(upload_event, str(path))
            handler.on_any_event(event)
            self.assertEqual(len(calls), before + 1)

    def test_file_event_handler_marks_syncable_file_dirty(self):
        remote, manager = self.make_remote_manager()
        manager.prepare()
        component = component_by_name(manager.components, "triton")
        local_root = component.local_dir
        final_file = local_root / "kernel/a.cubin"
        final_file.parent.mkdir(parents=True, exist_ok=True)
        final_file.write_text("cubin", encoding="utf-8")
        handler = jit_cache_module._JitFileEventHandler(component, manager.mark_dirty)

        handler.on_any_event(
            FakeFileEvent("moved", str(final_file.with_suffix(".tmp")), str(final_file))
        )

        self.assertEqual(
            effective_members(remote),
            {"triton/kernel/a.cubin": b"cubin"},
        )

    def test_remote_config_mounts_uri_before_validation(self):
        mounted_remote = self.root / "mounted_remote"
        mounted_remote.mkdir()

        with mock.patch(
            "rtp_llm.utils.jit_cache_manager.fetch_remote_file_to_local",
            return_value=str(mounted_remote),
        ) as fetch_remote:
            manager = self.make_manager("oss://bucket/jit-cache", create_remote=False)

        fetch_remote.assert_called_once_with(
            "oss://bucket/jit-cache", MountRwMode.RWMODE_RW
        )
        self.assertEqual(manager.store.root, mounted_remote)

    def test_concurrent_publish_does_not_corrupt_snapshot(self):
        """Parallel sync calls must not produce a truncated archive."""
        remote, manager = self.make_remote_manager()
        component = component_by_name(manager.components, "triton")
        local_root = component.local_dir

        errors: list[Exception] = []

        # Each worker publishes then compacts, so they race on both delta upload
        # and snapshot compaction under the shared lock.
        def publish_worker(idx: int):
            try:
                p = local_root / f"kernel/k{idx}.so"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x" * 1024)
                manager.mark_dirty(component, f"kernel/k{idx}.so")
                manager._publish_delta()
                manager.store.compact()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=publish_worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertFalse(errors, f"concurrent publish raised: {errors}")
        # Effective state must be readable after concurrent writes.
        names = effective_member_names(remote)
        self.assertGreater(len(names), 0)

    def test_snapshot_store_lock_times_out_when_lock_dir_is_left_behind(self):
        remote = self.root / "remote"
        remote.mkdir()
        (remote / jit_cache_module.SNAPSHOT_LOCK_DIR_NAME).mkdir()

        with (
            mock.patch.object(jit_cache_module, "SNAPSHOT_LOCK_TIMEOUT_S", 0.01),
            mock.patch.object(jit_cache_module, "SNAPSHOT_LOCK_POLL_S", 0.001),
        ):
            with self.assertRaises(TimeoutError):
                with jit_cache_module.RemoteSnapshotStore(remote).lock():
                    pass

    def test_two_services_accumulate_entries_without_loss(self):
        """Simulates two services writing to the same remote dir sequentially
        (worst-case overlap without flock released between them) and verifies
        neither service's entries are lost."""
        remote = self.root / "remote"
        local_a = self.root / "local_a"
        local_b = self.root / "local_b"

        svc_a = self.make_manager(str(remote), local_root=local_a)
        clear_jit_env()
        svc_b = self.make_manager(str(remote), local_root=local_b)

        component = component_by_name(svc_a.components, "triton")

        # Service A writes its file and publishes.
        path_a = component.local_dir / "kernel/a_kernel.so"
        path_a.parent.mkdir(parents=True, exist_ok=True)
        path_a.write_bytes(b"svc_a_kernel")
        svc_a.mark_dirty(component, "kernel/a_kernel.so")

        # Service B writes a *different* file and publishes.
        component_b = component_by_name(svc_b.components, "triton")
        path_b = component_b.local_dir / "kernel/b_kernel.so"
        path_b.parent.mkdir(parents=True, exist_ok=True)
        path_b.write_bytes(b"svc_b_kernel")
        svc_b.mark_dirty(component_b, "kernel/b_kernel.so")

        members = effective_members(remote)
        self.assertIn(
            "triton/kernel/a_kernel.so",
            members,
            "service A entry lost after service B publish",
        )
        self.assertIn(
            "triton/kernel/b_kernel.so",
            members,
            "service B entry missing from snapshot",
        )
        self.assertEqual(members["triton/kernel/a_kernel.so"], b"svc_a_kernel")
        self.assertEqual(members["triton/kernel/b_kernel.so"], b"svc_b_kernel")

    def test_mark_dirty_filters_unsyncable_empty_temp_and_missing_files(self):
        _remote, manager = self.make_remote_manager()
        component = component_by_name(manager.components, "triton")
        local_root = component.local_dir

        ignored_paths = {
            "kernel/readme.txt": b"text",
            "tmp.pid_123/kernel.so": b"tmp",
            "kernel/empty.so": b"",
        }
        for rel, payload in ignored_paths.items():
            path = local_root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        for rel in (*ignored_paths, "nonexistent/path.so"):
            self.assertFalse(manager.mark_dirty(component, rel))
        self.assertFalse(any(manager.local_delta_dir.iterdir()))

    def test_mark_dirty_swallows_copy_race_without_raising(self):
        # A file can vanish between the stat guard and the copy (temp-file
        # churn in JIT dirs). mark_dirty must return False, not let the OSError
        # escape into the watchdog thread and kill the observer.
        _remote, manager = self.make_remote_manager()
        component = component_by_name(manager.components, "triton")
        path = component.local_dir / "kernel/a.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")

        with mock.patch.object(
            jit_cache_module.shutil,
            "copy2",
            side_effect=FileNotFoundError("source vanished"),
        ):
            self.assertFalse(manager.mark_dirty(component, "kernel/a.so"))

        self.assertFalse((manager.local_delta_dir / "triton/kernel/a.so").exists())

    def test_stop_is_idempotent(self):
        _remote, manager = self.make_remote_manager()

        manager.start_background_sync()
        manager.stop()
        manager.stop()

        self.assertIsNone(manager._observer)
        self.assertIsNone(manager._sync_thread)


class SnapshotPublishConsumerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        clear_jit_env()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tmp.cleanup()

    def make_manager(self, local_root: Path, remote_root: Path) -> JitCacheManager:
        remote_root.mkdir(parents=True, exist_ok=True)
        config = JITConfig()
        config.local_jit_dir = str(local_root)
        config.remote_jit_dir = str(remote_root)
        manager = JitCacheManager(config)
        manager.bootstrap()
        return manager

    def test_publish_then_consumer_extracts_snapshot(self):
        remote_root = self.root / "remote"
        first = self.make_manager(self.root / "local_first", remote_root)
        try:
            first.prepare()
            expected_members = set()
            for component in first.components:
                local_root = component.local_dir
                filename = f"kernel/{component.name}{component.sync_suffixes[0]}"
                local_file = local_root / filename
                local_file.parent.mkdir(parents=True, exist_ok=True)
                local_file.write_text(component.name, encoding="utf-8")
                first.mark_dirty(component, filename)
                expected_members.add(
                    f"{component.local_dir.relative_to(first.local_root).as_posix()}/{filename}"
                )
            self.assertEqual(effective_member_names(remote_root), expected_members)
        finally:
            first.stop()

        for component in jit_cache_module.COMPONENTS:
            os.environ.pop(component.env_name, None)

        second = self.make_manager(self.root / "local_second", remote_root)
        try:
            second.prepare()
            for component in second.components:
                local_root = component.local_dir
                filename = f"kernel/{component.name}{component.sync_suffixes[0]}"
                self.assertEqual(
                    (local_root / filename).read_text(encoding="utf-8"), component.name
                )
        finally:
            second.stop()


if __name__ == "__main__":
    unittest.main()

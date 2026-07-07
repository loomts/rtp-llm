import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.utils import jit_cache_manager as jit_cache_module
from rtp_llm.utils.fuser import MountRwMode
from rtp_llm.utils.jit_cache_manager import JitCacheManager


def component_by_name(components, name: str):
    return next(component for component in components if component.name == name)


def iter_sync_files(component):
    if not component.local_dir.is_dir():
        return
    for path in sorted(component.local_dir.rglob("*")):
        if path.is_file():
            yield path, path.relative_to(component.local_dir).as_posix()


def effective_member_names(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }


def clear_jit_env():
    for component in jit_cache_module.COMPONENTS:
        os.environ.pop(component.env_name, None)
    os.environ.pop("TRITON_AUTOTUNE_CACHE_MODE", None)


class JitCacheManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        clear_jit_env()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tmp.cleanup()

    def component_dir(self, root: Path, name: str) -> Path:
        return next(
            component.resolve(root).local_dir
            for component in jit_cache_module.COMPONENTS
            if component.name == name
        )

    def make_config(self, *, local_root: Path | None = None, remote_root: str = ""):
        config = JITConfig()
        config.local_jit_dir = str(local_root or self.root / "local")
        config.remote_jit_dir = remote_root
        return config

    def test_apply_jit_cache_env_overwrites_component_envs_by_default(self):
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

        jit_cache_module.apply_jit_cache_env(self.root / "local", create_dirs=True)

        self.assertEqual(
            os.environ["TRITON_AUTOTUNE_CONFIG_DIR"],
            jit_cache_module.BUILTIN_CONFIG_SENTINEL,
        )
        self.assertFalse(
            self.component_dir(self.root / "local", "triton_autotune").exists()
        )

    def test_remote_jit_dir_is_used_directly_for_all_component_envs(self):
        remote = self.root / "remote"
        remote.mkdir()
        manager = JitCacheManager(self.make_config(remote_root=str(remote)))

        manager.bootstrap()

        self.assertEqual(manager.remote_root, remote)
        self.assertFalse(hasattr(manager, "local_root"))
        for component in jit_cache_module.COMPONENTS:
            expected = self.component_dir(remote, component.name)
            self.assertEqual(os.environ[component.env_name], str(expected))
            self.assertTrue(expected.is_dir())
        self.assertFalse((self.root / "local").exists())

    def test_missing_remote_disables_direct_jit_cache(self):
        local = self.root / "local"
        manager = JitCacheManager(
            self.make_config(local_root=local, remote_root=str(self.root / "missing"))
        )

        manager.bootstrap()

        self.assertIsNone(manager.remote_root)
        self.assertEqual(manager.components, tuple())
        self.assertFalse(hasattr(manager, "local_root"))
        for component in jit_cache_module.COMPONENTS:
            self.assertNotIn(component.env_name, os.environ)
        self.assertFalse(local.exists())

    def test_remote_config_mounts_uri_before_validation(self):
        mounted_remote = self.root / "mounted_remote"
        mounted_remote.mkdir()

        with mock.patch(
            "rtp_llm.utils.fuser.fetch_remote_file_to_local",
            return_value=str(mounted_remote),
        ) as fetch_remote:
            manager = JitCacheManager(
                self.make_config(remote_root="oss://bucket/jit-cache")
            )
            manager.bootstrap()

        fetch_remote.assert_called_once_with(
            "oss://bucket/jit-cache", MountRwMode.RWMODE_RW
        )
        self.assertEqual(manager.remote_root, mounted_remote)
        self.assertEqual(
            os.environ["TRITON_CACHE_DIR"],
            str(self.component_dir(mounted_remote, "triton")),
        )

    def test_direct_manager_has_no_background_sync_state(self):
        remote = self.root / "remote"
        remote.mkdir()
        manager = JitCacheManager(self.make_config(remote_root=str(remote)))
        manager.bootstrap()

        self.assertFalse(hasattr(manager, "store"))
        self.assertFalse(hasattr(manager, "local_delta_dir"))
        self.assertFalse(hasattr(manager, "_observer"))
        self.assertFalse(hasattr(jit_cache_module, "RemoteSnapshotStore"))
        self.assertFalse(hasattr(jit_cache_module, "zstd_tar"))
        self.assertFalse(hasattr(jit_cache_module, "_JitFileEventHandler"))


if __name__ == "__main__":
    unittest.main()

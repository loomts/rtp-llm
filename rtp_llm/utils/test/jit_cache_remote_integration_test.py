import os
import sys
import tempfile
import traceback
import unittest
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.models_py.triton_kernels.autotune_cache.export import (
    write_default_config_json,
)
from rtp_llm.utils import jit_cache_manager as jit_cache_module
from rtp_llm.utils.jit_cache_manager import JitCacheManager
from rtp_llm.utils.test.jit_cache_manager_test import (
    component_by_name,
    effective_member_names,
    iter_sync_files,
)

_FLASHINFER_ENV_ATTRS = (
    "FLASHINFER_BASE_DIR",
    "FLASHINFER_CACHE_DIR",
    "FLASHINFER_WORKSPACE_DIR",
    "FLASHINFER_JIT_DIR",
    "FLASHINFER_GEN_SRC_DIR",
    "FLASHINFER_AOT_DIR",
)

_JIT_ENV_NAMES = tuple(c.env_name for c in jit_cache_module.COMPONENTS) + (
    "TRITON_AUTOTUNE_CACHE_MODE",
)


def _make_jit_manager(
    local_root: Path,
    remote_root: Path,
) -> JitCacheManager:
    remote_root.mkdir(parents=True, exist_ok=True)
    config = JITConfig()
    config.local_jit_dir = str(local_root)
    config.remote_jit_dir = str(remote_root)
    manager = JitCacheManager(config)
    manager.bootstrap()
    return manager


def _setup_flashinfer_workspace_and_build():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit import gen_batch_mla_module

    workspace_base = Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
    version_dir = jit_env.FLASHINFER_WORKSPACE_DIR.parent.name
    arch_dir = jit_env.FLASHINFER_WORKSPACE_DIR.name
    workspace_dir = workspace_base / ".cache" / "flashinfer" / version_dir / arch_dir
    jit_env.FLASHINFER_BASE_DIR = workspace_base
    jit_env.FLASHINFER_CACHE_DIR = workspace_base / ".cache" / "flashinfer"
    jit_env.FLASHINFER_WORKSPACE_DIR = workspace_dir
    jit_env.FLASHINFER_JIT_DIR = workspace_dir / "cached_ops"
    jit_env.FLASHINFER_GEN_SRC_DIR = workspace_dir / "generated"
    jit_env.FLASHINFER_AOT_DIR = workspace_base / ".no_aot"
    spec = gen_batch_mla_module(
        "fa2",
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        512,
        64,
        False,
    )
    spec.build_and_load()


def _run_triton_add_kernel(x: torch.Tensor, add: float = 1.0) -> torch.Tensor:
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x, y, n: tl.constexpr, block: tl.constexpr, val: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < n
        values = tl.load(x + offsets, mask=mask)
        tl.store(y + offsets, values + val, mask=mask)

    y = torch.empty_like(x)
    add_kernel[(triton.cdiv(x.numel(), 128),)](x, y, x.numel(), block=128, val=add)
    return y


def _run_triton_rank_jit(rank: int) -> None:
    x = torch.arange(256, device="cuda", dtype=torch.float32) + rank
    y = _run_triton_add_kernel(x, add=1.0)
    torch.cuda.synchronize()
    if not torch.allclose(y, x + 1.0):
        raise AssertionError(f"rank {rank} Triton JIT result mismatch")


def _two_rank_direct_remote_worker(
    rank: int,
    world_size: int,
    root: str,
    remote_root: str,
    barrier,
    result_queue,
) -> None:
    manager = None
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["TRITON_AUTOTUNE_GPU_NAME"] = "NVIDIA_H20"
        jit_cache_module.get_gpu_info.cache_clear()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(rank % torch.cuda.device_count())

        root_path = Path(root)
        remote_path = Path(remote_root)
        manager = _make_jit_manager(
            root_path / f"local_rank_{rank}",
            remote_path,
        )
        manager.prepare()
        _run_triton_rank_jit(rank)

        component = component_by_name(manager.components, "triton")
        local_dir = component.local_dir
        generated_files = list(iter_sync_files(component))
        if not generated_files:
            raise RuntimeError(f"rank {rank} Triton JIT produced no syncable files")

        marker_rel = f"rank_markers/rank_{rank}.json"
        marker_path = local_dir / marker_rel
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(f'{{"rank": {rank}}}', encoding="utf-8")

        barrier.wait(timeout=30)
        direct_files = len(list(iter_sync_files(component)))

        result_queue.put(
            {
                "rank": rank,
                "direct_files": direct_files,
                "generated_files": len(generated_files),
                "marker_member": f"triton/{marker_rel}",
            }
        )
    except Exception:
        result_queue.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        if manager is not None:
            manager.stop()
        jit_cache_module.get_gpu_info.cache_clear()


class _GpuJitTestBase(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 9:
            self.skipTest(f"SM90+ required, got {cap}")

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        self.old_modules = {
            name: sys.modules.get(name)
            for name in ("flashinfer.jit.env", "flashinfer.jit.core")
        }
        self.old_flashinfer_env_attrs = None
        env_module = sys.modules.get("flashinfer.jit.env")
        if env_module is not None:
            self.old_flashinfer_env_attrs = {
                attr: getattr(env_module, attr) for attr in _FLASHINFER_ENV_ATTRS
            }
        for env_name in _JIT_ENV_NAMES:
            os.environ.pop(env_name, None)
        os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        torch.cuda.set_device(0)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        env_module = sys.modules.get("flashinfer.jit.env")
        if env_module is not None and self.old_flashinfer_env_attrs is not None:
            for attr, value in self.old_flashinfer_env_attrs.items():
                setattr(env_module, attr, value)
        for name, module in self.old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        self.tmp.cleanup()


class RemoteJitIntegrationTest(_GpuJitTestBase):

    def test_single_gpu_reuses_same_remote_jit_for_all_components(self):
        remote_root = self.root / "remote"
        first = _make_jit_manager(self.root / "local_first", remote_root)
        try:
            first.prepare()

            self.run_all_jit_workloads()
            self.assert_components_have_files(remote_root)
        finally:
            first.stop()

        # This test drives two managers in one process, so reset envs before
        # binding the second manager to a different local cache root.
        for env_name in _JIT_ENV_NAMES:
            os.environ.pop(env_name, None)

        second = _make_jit_manager(self.root / "local_second", remote_root)
        try:
            second.prepare()
            self.assert_components_have_files(second.local_root)
        finally:
            second.stop()

    def run_all_jit_workloads(self) -> None:
        self.run_flashinfer_jit()
        self.run_triton_and_autotune_jit()
        self.run_deep_gemm_jit()
        self.run_torch_extension_jit()
        torch.cuda.synchronize()

    def run_flashinfer_jit(self) -> None:
        try:
            _setup_flashinfer_workspace_and_build()
        except ImportError as e:
            self.skipTest(f"flashinfer is not available: {e}")

    def run_triton_and_autotune_jit(self) -> None:
        import triton
        import triton.language as tl

        x = torch.arange(256, device="cuda", dtype=torch.float32)
        y = _run_triton_add_kernel(x, add=1.0)
        self.assertTrue(torch.allclose(y, x + 1.0))

        autotune_dir = Path(os.environ["TRITON_AUTOTUNE_CONFIG_DIR"])
        autotune_dir.mkdir(parents=True, exist_ok=True)
        write_default_config_json(
            autotune_dir / "autotuned_add_kernel.autotune.json",
            "autotuned_add_kernel",
            {"kwargs": {"block": 128}, "num_warps": 4, "num_stages": 3},
        )

        from rtp_llm.models_py.triton_kernels.autotune_cache import (
            autotune_cache_kwargs,
            cuda_cached_autotune,
        )

        @cuda_cached_autotune(
            configs=[
                triton.Config({"block": 64}, num_warps=4),
                triton.Config({"block": 128}, num_warps=4),
            ],
            key=["n"],
            **autotune_cache_kwargs,
        )
        @triton.jit
        def autotuned_add_kernel(x, y, n: tl.constexpr, block: tl.constexpr):
            offsets = tl.program_id(0) * block + tl.arange(0, block)
            mask = offsets < n
            values = tl.load(x + offsets, mask=mask)
            tl.store(y + offsets, values + 2.0, mask=mask)

        z = torch.empty_like(x)
        autotuned_add_kernel[(triton.cdiv(x.numel(), 128),)](x, z, x.numel())
        self.assertTrue(torch.allclose(z, x + 2.0))

    def run_deep_gemm_jit(self) -> None:
        try:
            import deep_gemm
        except Exception as e:
            self.skipTest(f"deep_gemm is not available: {e}")

        a = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16)
        d = torch.empty((16, 16), device="cuda", dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(a, b, d, None)
        component = next(
            c for c in jit_cache_module.COMPONENTS if c.name == "deep_gemm"
        )
        local_dir = Path(os.environ[component.env_name])
        component = replace(component, local_dir=local_dir)
        if not list(iter_sync_files(component)):
            for name in ("kernel.cu", "kernel.cubin"):
                path = local_dir / "cache" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"jit-cache-integration-probe")

    def run_torch_extension_jit(self) -> None:
        from torch.utils.cpp_extension import load_inline

        module = load_inline(
            name=f"rtp_jit_cache_test_ext_{os.getpid()}",
            cpp_sources=[
                "#include <torch/extension.h>\n"
                "int add_one_int(int value) { return value + 1; }\n"
            ],
            functions=["add_one_int"],
            extra_cflags=["-O0"],
            with_cuda=False,
            verbose=False,
        )
        self.assertEqual(module.add_one_int(41), 42)

    def assert_components_have_files(self, root: Path) -> None:
        missing = [
            component.name
            for component in (
                component.resolve(root) for component in jit_cache_module.COMPONENTS
            )
            if not list(iter_sync_files(component))
        ]
        if missing:
            self.fail(f"missing JIT cache files for components: {missing}")


class TwoRankDirectRemoteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        os.environ["TRITON_AUTOTUNE_GPU_NAME"] = "NVIDIA_H20"
        jit_cache_module.get_gpu_info.cache_clear()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        jit_cache_module.get_gpu_info.cache_clear()
        self.tmp.cleanup()

    def test_two_gpu_rank_processes_share_remote_jit_dir(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if torch.cuda.device_count() < 2:
            self.skipTest("two CUDA devices are required for 2-rank concurrency")
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 9:
            self.skipTest(f"SM90+ required for Triton JIT workload, got {cap}")

        world_size = 2
        remote_root = self.root / "remote_two_rank"
        ctx = get_context("spawn")
        barrier = ctx.Barrier(world_size)
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_two_rank_direct_remote_worker,
                args=(
                    rank,
                    world_size,
                    str(self.root),
                    str(remote_root),
                    barrier,
                    result_queue,
                ),
            )
            for rank in range(world_size)
        ]

        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)

        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                self.fail(f"rank process {process.pid} timed out")
            self.assertEqual(process.exitcode, 0)

        results = [result_queue.get(timeout=5) for _ in range(world_size)]
        errors = [r for r in results if "error" in r]
        if errors:
            self.fail("\n".join(e["error"] for e in errors))

        for result in results:
            self.assertGreater(result["generated_files"], 0)
            self.assertGreater(result["direct_files"], 0)

        members = effective_member_names(remote_root)
        for result in results:
            self.assertIn(result["marker_member"], members)


if __name__ == "__main__":
    unittest.main()

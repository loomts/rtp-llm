import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import torch

from rtp_llm.device.device_type import DeviceType
from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
    ASM_DECODE_HEAD_SIZE,
    AiterDecodeAttnOpTriton,
    AiterDecodeImplAsm,
    AiterDecodeImplNonAsm,
    AiterDecodeImplTritonLinear,
    AiterDecodeImplTritonVectorized,
    AiterPrefillImplAsm,
    AiterPrefillImplNonAsm,
    AiterPrefillImplPaged,
    FusedRopeKVCacheDecodeOpAsm,
    FusedRopeKVCacheDecodeOpNonAsm,
    FusedRopeKVCachePrefillOpAsm,
    FusedRopeKVCachePrefillOpNonAsm,
    UnsupportedVLayout,
    _v_layouts_coincide,
    _writes_linear_v,
    resolve_linear_v,
    view_triton_kv_cache,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType

ROCM = DeviceType.ROCm
BASE = KvCacheDataType.BASE
FP8 = KvCacheDataType.FP8
HEAD_256 = 256


class Flags(NamedTuple):
    aiter: bool
    asm: bool
    triton: bool


class ExpectedSelection(NamedTuple):
    prefill: type
    prefixed_prefill: type
    decode: type
    linear_v: bool


PREFILL_IMPLS = (AiterPrefillImplPaged, AiterPrefillImplAsm, AiterPrefillImplNonAsm)
# Must mirror the real registration order in attention/__init__.py.
DECODE_IMPLS = (
    AiterDecodeImplTritonVectorized,
    AiterDecodeImplTritonLinear,
    AiterDecodeImplAsm,
    AiterDecodeImplNonAsm,
)

L_NONASM = ExpectedSelection(
    AiterPrefillImplNonAsm,
    AiterPrefillImplNonAsm,
    AiterDecodeImplNonAsm,
    True,
)
L_TRITON = ExpectedSelection(
    AiterPrefillImplNonAsm,
    AiterPrefillImplNonAsm,
    AiterDecodeImplTritonLinear,
    True,
)
V_ASM = ExpectedSelection(
    AiterPrefillImplAsm,
    AiterPrefillImplPaged,
    AiterDecodeImplAsm,
    False,
)
V_TRITON_ASM_WRITER = ExpectedSelection(
    AiterPrefillImplAsm,
    AiterPrefillImplPaged,
    AiterDecodeImplTritonVectorized,
    False,
)
V_TRITON_NONASM_WRITER = ExpectedSelection(
    AiterPrefillImplNonAsm,
    AiterPrefillImplNonAsm,
    AiterDecodeImplTritonVectorized,
    False,
)

# Values follow HEAD_DTYPE_CASES; None means no self-consistent pair.
LAYOUT_MATRIX = {
    Flags(False, False, False): (None, None, None, None),
    Flags(False, False, True): (None, None, None, None),
    Flags(False, True, False): (V_ASM, V_ASM, None, None),
    Flags(False, True, True): (V_TRITON_ASM_WRITER,) * 4,
    # `use_aiter_pa` selects the paged Aiter stack, whose persistent V cache is
    # vectorized even when standalone ASM/Triton flags are off.
    Flags(True, False, False): (V_TRITON_ASM_WRITER,) * 4,
    Flags(True, False, True): (V_TRITON_ASM_WRITER,) * 4,
    Flags(True, True, False): (V_ASM, V_ASM, None, None),
    Flags(True, True, True): (V_TRITON_ASM_WRITER,) * 4,
}
HEAD_DTYPE_CASES = (
    (ASM_DECODE_HEAD_SIZE, BASE),
    (ASM_DECODE_HEAD_SIZE, FP8),
    (HEAD_256, BASE),
    (HEAD_256, FP8),
)
ONE_BLOCK = torch.tensor([[0]], dtype=torch.int32)
NO_BLOCK = torch.empty((0, 0), dtype=torch.int32)


def _fmha_config(flags):
    config = FMHAConfig()
    config.use_aiter_pa = flags.aiter
    config.use_asm_pa = flags.asm
    config.use_triton_pa = flags.triton
    return config


def _attn_configs(dtype, head_size, need_rope=True, page_size=32):
    config = AttentionConfigs()
    config.kv_cache_dtype = dtype
    config.size_per_head = head_size
    config.need_rope_kv_cache = need_rope
    config.kernel_tokens_per_block = page_size
    return config


def _select(
    attn_configs, fmha_config, is_prefill, has_prefix=False, block_id=ONE_BLOCK
):
    attn_inputs = SimpleNamespace(
        is_prefill=is_prefill,
        prefix_lengths=torch.tensor([4 if has_prefix else 0], dtype=torch.int32),
        kv_cache_kernel_block_id=block_id,
    )
    return attn_factory.get_fmha_impl(
        attn_configs, None, attn_inputs, fmha_config=fmha_config
    )


@contextmanager
def _factory_without_kernel_initialization():
    with ExitStack() as stack:
        enter = stack.enter_context
        for impl in PREFILL_IMPLS + DECODE_IMPLS:
            enter(patch.object(impl, "__init__", return_value=None))
        enter(patch.object(attn_factory, "PREFILL_MHA_IMPS", list(PREFILL_IMPLS)))
        enter(patch.object(attn_factory, "DECODE_MHA_IMPS", list(DECODE_IMPLS)))
        enter(patch.object(attn_factory, "get_device_type", return_value=ROCM))
        yield


def _matrix_cases():
    for flags, selections in LAYOUT_MATRIX.items():
        for (head, dtype), expected in zip(HEAD_DTYPE_CASES, selections):
            yield flags, head, dtype, expected


class TestVLayoutFactoryContract(unittest.TestCase):
    def test_complete_flag_dtype_head_size_matrix(self):
        with _factory_without_kernel_initialization():
            for flags, head, dtype, expected in _matrix_cases():
                with self.subTest(f=flags, head=head, dtype=dtype):
                    cfg = _fmha_config(flags)
                    attn = _attn_configs(dtype, head)
                    if expected is None:
                        # The resolver ignores attn_inputs, so prefix cannot change it.
                        for is_prefill in (True, False):
                            with self.assertRaises(UnsupportedVLayout):
                                _select(attn, cfg, is_prefill)
                        continue
                    self.assertIs(resolve_linear_v(attn, cfg), expected.linear_v)
                    decode = _select(attn, cfg, False)
                    self.assertIs(type(decode), expected.decode)
                    self.assertEqual(
                        _writes_linear_v(decode.WRITER, dtype), expected.linear_v
                    )
                    # Prefix may swap Asm for Paged; the explicit oracle pins both.
                    for prefix, expected_prefill in (
                        (False, expected.prefill),
                        (True, expected.prefixed_prefill),
                    ):
                        prefill = _select(attn, cfg, True, prefix)
                        self.assertIs(type(prefill), expected_prefill)
                        # Both phases write the same blocks, so both writers must agree.
                        self.assertEqual(
                            _writes_linear_v(prefill.WRITER, dtype), expected.linear_v
                        )

    def test_layout_check_is_skipped_without_a_persistent_cache(self):
        # One case per preflight exit; all must keep the pre-fix selection rather
        # than rejecting on layout grounds.
        for kwargs, need_rope in (
            (dict(block_id=None), True),
            (dict(block_id=NO_BLOCK), True),
            (dict(), False),
        ):
            with self.subTest(**kwargs, need_rope=need_rope):
                with _factory_without_kernel_initialization():
                    impl = _select(
                        _attn_configs(BASE, 64, need_rope=need_rope),
                        _fmha_config(Flags(False, True, False)),
                        is_prefill=True,
                        **kwargs,
                    )
                self.assertIs(type(impl), AiterPrefillImplAsm)

    def test_error_names_the_config_and_distinguishes_the_two_causes(self):
        cases = (
            # No backend at all, versus backends present but no common layout.
            (Flags(False, False, True), BASE, "backend is disabled", "use_asm_pa"),
            (Flags(False, False, False), FP8, "backend is disabled", "use_asm_pa"),
        )
        for flags, dtype, cause, hint in cases:
            with self.subTest(flags=flags, dtype=dtype):
                with _factory_without_kernel_initialization():
                    with self.assertRaises(UnsupportedVLayout) as ctx:
                        _select(
                            _attn_configs(dtype, HEAD_256),
                            _fmha_config(flags),
                            is_prefill=True,
                        )
                message = str(ctx.exception)
                self.assertIn(cause, message)
                self.assertIn(hint, message)
                self.assertIn(f"size_per_head={HEAD_256}", message)
                self.assertIn(f"kv_cache_dtype={dtype}", message)
                self.assertIn("kernel_tokens_per_block=32", message)

    def test_absent_fmha_config_treats_every_flag_as_enabled(self):
        # fmha_config=None disables nothing, so ASM priority wins.
        self.assertFalse(resolve_linear_v(_attn_configs(BASE, HEAD_256), None))


class TestCoincidentVLayouts(unittest.TestCase):
    def test_layouts_coincide_only_when_page_equals_vector_width(self):
        for page_size, element_size in ((8, 2), (16, 2), (16, 1), (32, 1)):
            with self.subTest(page_size=page_size, element_size=element_size):
                vector_width = 16 // element_size
                offsets_match = all(
                    dim * page_size + token
                    == (token // vector_width) * HEAD_256 * vector_width
                    + dim * vector_width
                    + token % vector_width
                    for token in range(page_size)
                    for dim in range(HEAD_256)
                )
                self.assertEqual(
                    _v_layouts_coincide(page_size, element_size), offsets_match
                )

    def test_fp8_page_16_rescues_only_layout_mismatches(self):
        cases = (
            (Flags(True, False, False), ASM_DECODE_HEAD_SIZE, AiterPrefillImplAsm),
            (Flags(True, False, False), HEAD_256, AiterPrefillImplAsm),
        )
        with _factory_without_kernel_initialization():
            for flags, head, expected_prefill in cases:
                with self.subTest(flags=flags, head=head):
                    attn = _attn_configs(FP8, head, page_size=16)
                    cfg = _fmha_config(flags)
                    self.assertIs(type(_select(attn, cfg, True)), expected_prefill)
                    if flags.aiter or flags.asm:
                        self.assertIs(
                            type(_select(attn, cfg, True, has_prefix=True)),
                            AiterPrefillImplPaged,
                        )
                    self.assertIs(
                        type(_select(attn, cfg, False)),
                        (
                            AiterDecodeImplTritonVectorized
                            if flags.aiter
                            else AiterDecodeImplNonAsm
                        ),
                    )

    def test_equivalent_layouts_keep_the_vectorized_pair(self):
        with _factory_without_kernel_initialization():
            impl = _select(
                _attn_configs(FP8, HEAD_256, page_size=16),
                _fmha_config(Flags(True, False, False)),
                is_prefill=True,
            )
        self.assertIs(type(impl), AiterPrefillImplAsm)


class TestWriterContracts(unittest.TestCase):
    def test_writer_layouts(self):
        cases = (
            (FusedRopeKVCacheDecodeOpNonAsm, True, True),
            (FusedRopeKVCachePrefillOpNonAsm, True, False),
            (FusedRopeKVCacheDecodeOpAsm, False, False),
            (FusedRopeKVCachePrefillOpAsm, False, False),
        )
        for writer, base_linear, fp8_linear in cases:
            with self.subTest(writer=writer.__name__):
                self.assertEqual(_writes_linear_v(writer, BASE), base_linear)
                self.assertEqual(_writes_linear_v(writer, FP8), fp8_linear)

    def test_rocm_registration_order(self):
        from rtp_llm.models_py.modules.factory.attention import (
            DECODE_MHA_IMPS,
            PREFILL_MHA_IMPS,
        )

        self.assertEqual(PREFILL_MHA_IMPS, list(PREFILL_IMPLS))
        self.assertEqual(DECODE_MHA_IMPS, list(DECODE_IMPLS))

    def test_reader_keeps_the_layout_it_is_given(self):
        # The impls derive this from their WRITER; the writers are pybind types, so
        # the derivation itself needs the ROCm numerical regression to pin.
        for linear_v in (True, False):
            with self.subTest(linear_v=linear_v):
                op = AiterDecodeAttnOpTriton(
                    _attn_configs(BASE, HEAD_256), linear_v=linear_v
                )
                self.assertEqual(op.linear_v, linear_v)


class TestWriterDeclarationGuards(unittest.TestCase):
    """The WRITER contract and the non-ROCm short circuit have no other coverage."""

    def test_every_registered_rocm_impl_declares_a_known_writer(self):
        from rtp_llm.models_py.modules.factory.attention import (
            DECODE_MHA_IMPS,
            PREFILL_MHA_IMPS,
        )

        for impl in PREFILL_MHA_IMPS + DECODE_MHA_IMPS:
            with self.subTest(impl=impl.__name__):
                writer = impl.WRITER
                self.assertIsNotNone(writer, f"{impl.__name__} declares no WRITER")
                for dtype in (BASE, FP8):
                    self.assertIsInstance(_writes_linear_v(writer, dtype), bool)

    def test_missing_writer_fails_closed(self):
        from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
            FMHAImplBase,
        )

        # Inherits the base-class default WRITER = None.
        class ImplWithoutWriter(FMHAImplBase):
            @classmethod
            def support(cls, attn_configs, attn_inputs):
                return True

        with _factory_without_kernel_initialization():
            with patch.object(attn_factory, "PREFILL_MHA_IMPS", [ImplWithoutWriter]):
                cases = (
                    (_attn_configs(BASE, HEAD_256), Flags(True, True, True)),
                    (
                        _attn_configs(FP8, HEAD_256, page_size=16),
                        Flags(True, False, False),
                    ),
                )
                for attn, flags in cases:
                    with self.subTest(flags=flags, dtype=attn.kv_cache_dtype):
                        with self.assertRaisesRegex(TypeError, "must declare a WRITER"):
                            _select(attn, _fmha_config(flags), is_prefill=True)

    def test_unknown_writer_fails_closed(self):
        class NotAKvWriter:
            pass

        with self.assertRaisesRegex(TypeError, "unknown ROCm KV-cache writer"):
            _writes_linear_v(NotAKvWriter, BASE)

    def test_non_rocm_device_skips_the_layout_contract(self):
        # A CUDA device must not consult resolve_linear_v at all, so a combination
        # that would raise on ROCm still selects normally here.
        with ExitStack() as stack:
            for impl in PREFILL_IMPLS:
                stack.enter_context(patch.object(impl, "__init__", return_value=None))
            stack.enter_context(
                patch.object(attn_factory, "PREFILL_MHA_IMPS", list(PREFILL_IMPLS))
            )
            stack.enter_context(
                patch.object(
                    attn_factory, "get_device_type", return_value=DeviceType.Cuda
                )
            )
            impl = _select(
                _attn_configs(FP8, HEAD_256),
                _fmha_config(Flags(True, False, False)),
                is_prefill=True,
            )
        # The Aiter flag enables the vectorized initial writer even when ASM is
        # disabled; non-ROCm devices simply skip the ROCm layout preflight.
        self.assertIs(type(impl), AiterPrefillImplAsm)


class TestTritonCacheViews(unittest.TestCase):
    # vs = 16 // element_size, so fp16 gives 8 and fp8 gives 16. Both must work, and
    # the vectorized V view groups tokens, so page < vs must not silently produce a
    # zero-length axis.
    def test_zero_copy_views_for_both_vector_widths(self):
        cases = (
            # dtype, page, vs, linear V shape, vectorized V shape
            (torch.float16, 16, 8, (2, 3, 128, 16), (2, 3, 2, 128, 8)),
            (torch.float8_e4m3fn, 16, 16, (2, 3, 128, 16), (2, 3, 1, 128, 16)),
        )
        for dtype, page, vs, linear_shape, vec_shape in cases:
            cache = torch.empty((2, 2, 3, page, 128), dtype=dtype)
            storage_ptr = cache.untyped_storage().data_ptr()
            for linear_v, value_shape in ((True, linear_shape), (False, vec_shape)):
                with self.subTest(dtype=dtype, linear_v=linear_v):
                    key, value = view_triton_kv_cache(cache, linear_v)
                    self.assertEqual(key.shape, (2, 3, 128 // vs, page, vs))
                    self.assertEqual(value.shape, value_shape)
                    self.assertEqual(key.untyped_storage().data_ptr(), storage_ptr)
                    self.assertEqual(value.untyped_storage().data_ptr(), storage_ptr)
                    # A view must never drop elements.
                    self.assertEqual(value.numel(), 2 * 3 * page * 128)

    def test_vectorized_view_rejects_page_smaller_than_the_vector(self):
        # page=8 with fp8 (vs=16) cannot be grouped; the view must fail loudly rather
        # than yield a 0-length axis that silently discards V.
        cache = torch.empty((2, 2, 3, 8, 128), dtype=torch.float8_e4m3fn)
        with self.assertRaises(RuntimeError):
            view_triton_kv_cache(cache, linear_v=False)


if __name__ == "__main__":
    unittest.main()

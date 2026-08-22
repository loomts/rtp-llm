import unittest
from types import SimpleNamespace

from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    _is_fmha_impl_disabled,
)


class AiterFactoryTest(unittest.TestCase):
    @staticmethod
    def _config(*, use_aiter_pa: bool, use_asm_pa: bool, use_triton_pa: bool):
        return SimpleNamespace(
            use_aiter_pa=use_aiter_pa,
            use_asm_pa=use_asm_pa,
            use_triton_pa=use_triton_pa,
        )

    def test_paged_prefill_is_enabled_by_aiter(self):
        config = self._config(use_aiter_pa=True, use_asm_pa=False, use_triton_pa=False)

        self.assertFalse(_is_fmha_impl_disabled("AiterPrefillImplPaged", config))

    def test_paged_prefill_is_not_enabled_by_asm_alone(self):
        config = self._config(use_aiter_pa=False, use_asm_pa=True, use_triton_pa=False)

        self.assertTrue(_is_fmha_impl_disabled("AiterPrefillImplPaged", config))

    def test_paged_prefill_is_not_enabled_by_triton_alone(self):
        config = self._config(use_aiter_pa=False, use_asm_pa=False, use_triton_pa=True)

        self.assertTrue(_is_fmha_impl_disabled("AiterPrefillImplPaged", config))

    def test_aiter_enables_vectorized_initial_prefill_writer(self):
        config = self._config(use_aiter_pa=True, use_asm_pa=False, use_triton_pa=False)

        self.assertFalse(_is_fmha_impl_disabled("AiterPrefillImplAsm", config))

    def test_triton_enables_vectorized_initial_prefill_writer(self):
        config = self._config(use_aiter_pa=False, use_asm_pa=False, use_triton_pa=True)

        self.assertFalse(_is_fmha_impl_disabled("AiterPrefillImplAsm", config))

    def test_aiter_without_asm_enables_vectorized_triton_decode(self):
        config = self._config(use_aiter_pa=True, use_asm_pa=False, use_triton_pa=False)

        self.assertFalse(_is_fmha_impl_disabled("AiterDecodeImplTriton", config))

    def test_aiter_with_asm_keeps_vectorized_asm_decode(self):
        config = self._config(use_aiter_pa=True, use_asm_pa=True, use_triton_pa=False)

        self.assertTrue(_is_fmha_impl_disabled("AiterDecodeImplTriton", config))
        self.assertFalse(_is_fmha_impl_disabled("AiterDecodeImplAsm", config))

    def test_v1_nonasm_fallbacks_are_disabled(self):
        config = self._config(use_aiter_pa=True, use_asm_pa=False, use_triton_pa=False)

        self.assertTrue(_is_fmha_impl_disabled("AiterPrefillImplNonAsm", config))
        self.assertTrue(_is_fmha_impl_disabled("AiterDecodeImplNonAsm", config))


if __name__ == "__main__":
    unittest.main()

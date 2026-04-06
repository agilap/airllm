import sys
import types

import pytest
import torch
from torch.testing import assert_close


try:
    from ..airllm.airllm_gemma4 import AirLLMGemma4, DynamicCacheWithShared
except ModuleNotFoundError as exc:
    if exc.name in {"airllm_gemma2", "air_llm.airllm.airllm_gemma2"}:
        gemma2_stub = types.ModuleType("air_llm.airllm.airllm_gemma2")

        class AirLLMGemma2:  # pragma: no cover - fallback for workspace snapshot
            pass

        gemma2_stub.AirLLMGemma2 = AirLLMGemma2
        sys.modules["air_llm.airllm.airllm_gemma2"] = gemma2_stub
        from ..airllm.airllm_gemma4 import AirLLMGemma4, DynamicCacheWithShared
    else:
        raise


class _MockConfig:
    sliding_window = 4


class _MockModel:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.config = _MockConfig()


def _make_airllm_gemma4_for_masks(dtype: torch.dtype = torch.float32) -> AirLLMGemma4:
    model = AirLLMGemma4.__new__(AirLLMGemma4)
    model.device = torch.device("cpu")
    model.model = _MockModel(dtype=dtype)
    return model


@pytest.mark.parametrize("seq_len", [1, 4, 8, 16])
def test_build_attention_masks(seq_len: int):
    dtype = torch.float32
    model = _make_airllm_gemma4_for_masks(dtype=dtype)
    input_ids = torch.zeros((1, seq_len), dtype=torch.long)

    masks = model._build_attention_masks(input_ids, dtype=dtype)

    sliding = masks["sliding"]
    full = masks["full"]
    finfo_min = torch.finfo(dtype).min

    if seq_len == 1:
        assert sliding.shape == (1, 1, 1, 1)
        assert full.shape == (1, 1, 1, 1)
        assert_close(sliding, torch.zeros((1, 1, 1, 1), dtype=dtype))
        assert_close(full, torch.zeros((1, 1, 1, 1), dtype=dtype))
        return

    assert sliding.shape == (1, 1, seq_len, seq_len)
    assert full.shape == (1, 1, seq_len, seq_len)

    q = torch.arange(seq_len)[:, None]
    k = torch.arange(seq_len)[None, :]

    causal_allowed = q >= k
    full_expected = torch.where(
        causal_allowed,
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(finfo_min, dtype=dtype),
    )
    assert_close(full[0, 0], full_expected)

    sliding_allowed = causal_allowed & ((q - k) <= model.model.config.sliding_window)
    sliding_expected = torch.where(
        sliding_allowed,
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(finfo_min, dtype=dtype),
    )
    assert_close(sliding[0, 0], sliding_expected)

    if seq_len == 8:
        assert_close(sliding[0, 0, 5, 0], torch.tensor(finfo_min, dtype=dtype))
        assert_close(sliding[0, 0, 5, 4], torch.tensor(0.0, dtype=dtype))


def test_dynamic_cache_with_shared_raises_before_set_shared():
    cache = DynamicCacheWithShared()
    with pytest.raises(RuntimeError, match="shared KV not populated"):
        cache.get_shared()


def test_dynamic_cache_with_shared_set_get_and_clear():
    cache = DynamicCacheWithShared()
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    cache.set_shared(k, v)
    got_k, got_v = cache.get_shared()
    assert got_k is k
    assert got_v is v

    cache.clear_shared()
    with pytest.raises(RuntimeError, match="shared KV not populated"):
        cache.get_shared()


def test_dynamic_cache_with_shared_update_does_not_raise():
    cache = DynamicCacheWithShared()
    key_states = torch.randn(1, 2, 3, 4)
    value_states = torch.randn(1, 2, 3, 4)

    try:
        cache.update(key_states, value_states, layer_idx=0)
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"update() raised unexpectedly: {exc}")

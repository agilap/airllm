import os

import pytest
import torch

from ..airllm.airllm_gemma4 import AirLLMGemma4, DynamicCacheWithShared


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Gemma 4 smoke test"),
    pytest.mark.skipif(os.environ.get("HF_TOKEN") is None, reason="HF_TOKEN is required"),
]


@pytest.fixture
def model():
    m = AirLLMGemma4(
        model_id="google/gemma-4-31B-it",
        device="cuda",
        compression="4bit",
    )
    yield m
    del m
    torch.cuda.empty_cache()


def _tokenize(model: AirLLMGemma4, text: str) -> torch.LongTensor:
    return model.tokenizer(text, return_tensors="pt").input_ids.to("cuda")


def _resolve_decoder_layers(model: AirLLMGemma4):
    decoder = getattr(model.model, "model", model.model)
    return decoder.layers


def test_basic_generate(model: AirLLMGemma4):
    input_ids = _tokenize(model, "The capital of France is")
    input_len = input_ids.shape[1]

    output = model.generate(input_ids, max_new_tokens=5, do_sample=False)

    assert output.shape == (1, input_len + 5)
    assert output.dtype == torch.long

    decoded = model.tokenizer.decode(output[0], skip_special_tokens=True)
    assert isinstance(decoded, str)
    assert decoded.strip() != ""


def test_mask_selection(model: AirLLMGemma4):
    layers = _resolve_decoder_layers(model)
    captured_masks = {}

    def hook_factory(idx: int):
        def _hook(_module, _args, kwargs):
            captured_masks[idx] = kwargs.get("attention_mask")

        return _hook

    hook0 = layers[0].register_forward_pre_hook(hook_factory(0), with_kwargs=True)
    hook1 = layers[1].register_forward_pre_hook(hook_factory(1), with_kwargs=True)
    try:
        input_ids = _tokenize(model, "Mask routing test")
        _ = model(input_ids)
    finally:
        hook0.remove()
        hook1.remove()

    assert 0 in captured_masks and captured_masks[0] is not None
    assert 1 in captured_masks and captured_masks[1] is not None

    layer_types = model.model.config.layer_types
    if layer_types[0] != layer_types[1]:
        assert not torch.equal(captured_masks[0], captured_masks[1])


def test_kv_shared_not_stale(model: AirLLMGemma4, monkeypatch: pytest.MonkeyPatch):
    original_set_shared = DynamicCacheWithShared.set_shared
    original_get_shared = DynamicCacheWithShared.get_shared

    state = {"set_calls": 0, "get_calls": 0, "errors": []}

    def _set_shared_wrapper(self, key_states, value_states):
        state["set_calls"] += 1
        return original_set_shared(self, key_states, value_states)

    def _get_shared_wrapper(self):
        state["get_calls"] += 1
        try:
            return original_get_shared(self)
        except RuntimeError as exc:
            state["errors"].append(str(exc))
            raise

    monkeypatch.setattr(DynamicCacheWithShared, "set_shared", _set_shared_wrapper)
    monkeypatch.setattr(DynamicCacheWithShared, "get_shared", _get_shared_wrapper)

    input_ids = _tokenize(model, "KV sharing smoke test")
    _ = model(input_ids)

    assert state["set_calls"] >= 1
    assert state["get_calls"] >= 1
    assert state["errors"] == []

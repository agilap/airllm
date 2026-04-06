import logging
import os
import importlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, Gemma4ForCausalLM, Gemma4ForConditionalGeneration
from transformers.cache_utils import DynamicCache

from .airllm_base import AirLLMBaseModel, init_empty_weights

try:
    AirLLMGemma2 = getattr(
        importlib.import_module(".airllm_gemma2", package=__package__),
        "AirLLMGemma2",
    )
except ModuleNotFoundError:
    AirLLMGemma2 = AirLLMBaseModel

logger = logging.getLogger(__name__)


class DynamicCacheWithShared(DynamicCache):
    def __init__(self):
        super().__init__()
        self._shared_key = None
        self._shared_value = None

    def set_shared(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self._shared_key = key_states
        self._shared_value = value_states

    def get_shared(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._shared_key is None or self._shared_value is None:
            raise RuntimeError(
                "shared KV not populated — call set_shared() before running shared layers"
            )
        return self._shared_key, self._shared_value

    def clear_shared(self):
        self._shared_key = None
        self._shared_value = None

    def update(self, *args, **kwargs):
        # explicit pass-through — do not remove.
        return super().update(*args, **kwargs)


class AirLLMGemma4(AirLLMGemma2):
    def __init__(
        self,
        model_local_path_or_repo_id=None,
        device="cuda:0",
        dtype=torch.float16,
        max_seq_len=512,
        layer_shards_saving_path=None,
        profiling_mode=False,
        compression=None,
        hf_token=None,
        prefetching=True,
        delete_original=False,
        model_id=None,
    ):
        resolved_model_id = model_local_path_or_repo_id or model_id
        if resolved_model_id is None:
            raise ValueError("Either model_local_path_or_repo_id or model_id must be provided")

        # Keep initialization flow in the parent implementation.
        self.model_id = resolved_model_id
        super(AirLLMGemma4, self).__init__(
            resolved_model_id,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
            layer_shards_saving_path=layer_shards_saving_path,
            profiling_mode=profiling_mode,
            compression=compression,
            hf_token=hf_token,
            prefetching=prefetching,
            delete_original=delete_original,
        )

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            "embed": "model.language_model.embed_tokens",  # embedding shard base
            "layer_prefix": "model.language_model.layers",  # dense transformer block prefix
            "norm": "model.language_model.norm",  # final normalization
            "lm_head": "model.language_model.lm_head",  # tied output head
        }
        return self.layer_names_dict

    def init_model(self):
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)

        # Gemma 4 checkpoints may wrap decoder config under text_config.
        if hasattr(config, "text_config"):
            logger.info("Using nested text_config")
            resolved_config = config.text_config
            with init_empty_weights():
                self.model = Gemma4ForConditionalGeneration(config)
        else:
            logger.info("Using config directly")
            resolved_config = config

            with init_empty_weights():
                self.model = Gemma4ForCausalLM(resolved_config)

        self.config = resolved_config

        # rotary_emb = sin/cos buffers only — tiny, never shard.
        if hasattr(self.model, "rotary_emb"):
            self.model.rotary_emb = self.model.rotary_emb.to(self.device)
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "rotary_emb"):
            self.model.language_model.rotary_emb = self.model.language_model.rotary_emb.to(self.device)
        self.model.eval()

    def _save_embed_shard(self, shard_path: str):
        model_state = self.model.state_dict()
        embed_state = {
            k: v
            for k, v in model_state.items()
            if k.startswith("model.embed_tokens.") or k.startswith("model.rotary_emb.")
        }
        # rotary_emb serialised here for completeness; always re-pinned on load.
        save_file(embed_state, str(Path(shard_path)))

    def _load_embed_shard(self, shard_path: str):
        embed_state = load_file(str(Path(shard_path)), device="cpu")

        embed_tokens_state = {
            k[len("model.embed_tokens.") :]: v
            for k, v in embed_state.items()
            if k.startswith("model.embed_tokens.")
        }
        rotary_state = {
            k[len("model.rotary_emb.") :]: v
            for k, v in embed_state.items()
            if k.startswith("model.rotary_emb.")
        }

        target_embed = self.model.embed_tokens if hasattr(self.model, "embed_tokens") else self.model.language_model.embed_tokens
        target_rotary = self.model.rotary_emb if hasattr(self.model, "rotary_emb") else self.model.language_model.rotary_emb

        target_embed.load_state_dict(embed_tokens_state, strict=False)
        target_rotary.load_state_dict(rotary_state, strict=False)
        # re-pin after every shard load — rotary must stay on device.
        if hasattr(self.model, "rotary_emb"):
            self.model.rotary_emb = self.model.rotary_emb.to(self.device)
        else:
            self.model.language_model.rotary_emb = self.model.language_model.rotary_emb.to(self.device)

    def _build_attention_masks(
        self,
        input_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        if seq_len == 1:
            decode_mask = torch.zeros((batch_size, 1, 1, 1), device=self.device, dtype=dtype)
            return {"sliding": decode_mask, "full": decode_mask.clone()}

        min_value = torch.finfo(dtype).min
        query_pos = torch.arange(seq_len, device=self.device)
        key_pos = torch.arange(seq_len, device=self.device)

        causal_allow = query_pos[:, None] >= key_pos[None, :]
        distance = query_pos[:, None] - key_pos[None, :]
        sliding_allow = causal_allow & (
            distance <= self.model.config.sliding_window
        )

        full_mask = torch.full(
            (batch_size, 1, seq_len, seq_len),
            min_value,
            device=self.device,
            dtype=dtype,
        )
        full_mask.masked_fill_(causal_allow.view(1, 1, seq_len, seq_len), 0.0)

        sliding_mask = torch.full(
            (batch_size, 1, seq_len, seq_len),
            min_value,
            device=self.device,
            dtype=dtype,
        )
        sliding_mask.masked_fill_(sliding_allow.view(1, 1, seq_len, seq_len), 0.0)

        return {"sliding": sliding_mask, "full": full_mask}

    def forward(
        self,
        input_ids: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        def _get_module(module_path: str):
            module = self.model
            for attr_name in module_path.split("."):
                module = getattr(module, attr_name)
            return module

        def _load_shard(layer_names):
            if isinstance(layer_names, str):
                layer_names = [layer_names]
            loaded = []
            for layer_name in layer_names:
                state_dict = self.load_layer_to_cpu(layer_name)
                self.move_layer_to_device(state_dict)
                loaded.append(_get_module(layer_name))
            return loaded

        def _unload_modules(modules):
            for module in modules:
                module.to("meta")

        def _extract_kv(layer_outputs):
            if not isinstance(layer_outputs, (tuple, list)):
                raise RuntimeError("could not extract shared KV tensors from layer output")

            for candidate in reversed(layer_outputs):
                if (
                    isinstance(candidate, (tuple, list))
                    and len(candidate) == 2
                    and torch.is_tensor(candidate[0])
                    and torch.is_tensor(candidate[1])
                ):
                    return candidate[0], candidate[1]

            raise RuntimeError("could not extract shared KV tensors from layer output")

        input_ids = input_ids.to(self.device)

        embed_names = ["model.language_model.embed_tokens", "model.language_model.rotary_emb"]
        layer_names = [
            f"model.language_model.layers.{i}" for i in range(self.model.config.num_hidden_layers)
        ]
        norm_names = ["model.language_model.norm", "model.language_model.lm_head"]

        embed_modules = _load_shard(embed_names)
        hidden_states = _get_module(embed_names[0])(input_ids)
        _unload_modules(embed_modules)

        # DUAL MASKS
        masks = self._build_attention_masks(input_ids, dtype=self.model.dtype)

        # KV SHARED SETUP
        cache = DynamicCacheWithShared()
        cache.clear_shared()
        num_layers = len(layer_names)
        n = getattr(self.model.config, "num_kv_shared_layers", 0)
        shared_source_index = num_layers - n - 1

        # LAYER LOOP
        for i, layer_name in enumerate(layer_names):
            layer_modules = _load_shard(layer_name)
            layer = layer_modules[0]

            mask = (
                masks["sliding"]
                if self.model.config.layer_types[i] == "sliding_attention"
                else masks["full"]
            )
            layer_outputs = layer(
                hidden_states,
                attention_mask=mask,
                past_key_value=cache,
                use_cache=True,
            )
            hidden_states = layer_outputs[0]

            if n > 0 and i == shared_source_index:
                k_states, v_states = _extract_kv(layer_outputs)
                cache.set_shared(k_states, v_states)

            _unload_modules(layer_modules)

        norm_modules = _load_shard(norm_names)
        hidden_states = _get_module(norm_names[0])(hidden_states)
        logits = _get_module(norm_names[1])(hidden_states)
        _unload_modules(norm_modules)

        return logits


__all__ = ["AirLLMGemma4", "DynamicCacheWithShared"]
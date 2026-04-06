import importlib
import logging
from transformers import AutoConfig
from sys import platform

from .airllm_gemma4 import AirLLMGemma4

logger = logging.getLogger(__name__)

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx

class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )
    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if 'hf_token' in kwargs:
            print(f"using hf_token")
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        if "Qwen2ForCausalLM" in config.architectures[0]:
            return "airllm", "AirLLMQWen2"
        elif "QWen" in config.architectures[0]:
            return "airllm", "AirLLMQWen"
        elif "Baichuan" in config.architectures[0]:
            return "airllm", "AirLLMBaichuan"
        elif "ChatGLM" in config.architectures[0]:
            return "airllm", "AirLLMChatGLM"
        elif "InternLM" in config.architectures[0]:
            return "airllm", "AirLLMInternLM"
        elif "Mistral" in config.architectures[0]:
            return "airllm", "AirLLMMistral"
        elif "Mixtral" in config.architectures[0]:
            return "airllm", "AirLLMMixtral"
        elif "Llama" in config.architectures[0]:
            return "airllm", "AirLLMLlama2"
        else:
            print(f"unknown artichitecture: {config.architectures[0]}, try to use Llama2...")
            return "airllm", "AirLLMLlama2"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):

        if is_on_mac_os:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, ** kwargs)

        if 'hf_token' in kwargs:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                token=kwargs['hf_token'],
            )
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        model_type = getattr(config, "model_type", None)
        if model_type in ("gemma4", "gemma4_text"):
            # gemma4 = Gemma4ForConditionalGeneration checkpoint (multimodal wrapper)
            # gemma4_text = Gemma4ForCausalLM checkpoint (pure text)
            # Both are handled identically — init_model() unwraps text_config internally.
            # Must precede Gemma 2 check: Gemma 4 text config inherits from Gemma 2
            # config, so a generic gemma prefix check would incorrectly match it.
            if model_type == "gemma4":
                logger.info("gemma4 checkpoint detected — AirLLM will use text decoder only.")
            return AirLLMGemma4(pretrained_model_name_or_path, *inputs, ** kwargs)

        module, cls = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, cls)
        return class_(pretrained_model_name_or_path, *inputs, ** kwargs)
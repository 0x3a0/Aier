from .providers import OpenAIModel
from .providers.base import LLMModel


_API_PROVIDERS_ = {
    "openai": OpenAIModel,
}

def get_model(
    format: str,
    model_name: str,
    api_key: str,
    base_url: str
) -> LLMModel:
    api_class = _API_PROVIDERS_.get(format, "")
    if not api_class:
        raise ValueError(f"不支持的格式: {format}")

    return api_class(model_name, api_key, base_url)
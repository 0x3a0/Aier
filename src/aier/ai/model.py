from .providers import LLMModel, OpenAIModel

_API_PROVIDERS_: dict[str, type[LLMModel]] = {
    "openai": OpenAIModel,
}


def get_model(
    format: str | None, model_name: str | None, api_key: str | None, base_url: str | None
) -> LLMModel:
    """创建对应 LLM API 的工厂函数"""
    if not format:
        raise ValueError("format参数不可为空")

    api_class: type[LLMModel] | None = _API_PROVIDERS_.get(format)
    if not api_class:
        raise ValueError(f"不支持的API格式: {format}")

    return api_class(model_name, api_key, base_url)

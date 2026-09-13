from collections.abc import Callable

from .providers import LLMModel, OpenAIModel

#: 各 provider 的构造函数签名。用 Callable 而非 type[LLMModel]，
#: 因为抽象基类本身不可实例化，type[LLMModel] 无法表达「可调用」。
type ProviderFactory = Callable[[str, str, str], LLMModel]

_API_PROVIDERS_: dict[str, ProviderFactory] = {
    "openai": OpenAIModel,
}


def get_model(
    format: str | None, model_name: str | None, api_key: str | None, base_url: str | None
) -> LLMModel:
    """创建对应 LLM API 的工厂函数

    `format` 沿用既有参数名以保持公开 API 稳定，故此处屏蔽内建名遮蔽告警。
    """
    if not format:
        raise ValueError("format参数不可为空")

    api_factory = _API_PROVIDERS_.get(format)
    if api_factory is None:
        raise ValueError(f"不支持的API格式: {format}")

    return api_factory(model_name or "", api_key or "", base_url or "")

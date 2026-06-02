from .base import BaseLLMModel
from ..types import AssistantMessageEvent


class OpenAIModel(BaseLLMModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str
    ) -> None:
        pass

    def stream_invoke(
        self,
        messages: list[Message],
    ) -> AssistantMessageEvent:
        """ 流式输出模型调用结果 """
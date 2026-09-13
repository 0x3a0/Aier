from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from ..tool import Tool
from ..types import AssistantMessageEvent, Context


class LLMModel(ABC):
    """所有 LLM provider 的抽象基类"""

    @abstractmethod
    def stream_invoke(self, context: Context, **kwargs: Any) -> Iterator[AssistantMessageEvent]:
        """流式输出

        `kwargs` 会原样透传给底层 SDK（temperature、extra_body 等），
        因此无法给出比 `Any` 更精确的类型。
        """

    @abstractmethod
    def _convert_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """将 Tool 转换为当前 provider 的 function-calling schema"""

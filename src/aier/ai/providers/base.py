from abc import ABC, abstractmethod
from typing import Iterator

from ..types import AssistantMessageEvent, Context
from ..tool import Tool


class LLMModel(ABC):
    """所有 LLM provider 的抽象基类"""

    @abstractmethod
    def stream_invoke(
        self,
        context: Context,
        **kwargs
    ) -> Iterator[AssistantMessageEvent]:
        """流式输出"""

    @abstractmethod
    def conver_tools(
        self,
        tools: list[Tool]
    ) -> list[dict]:
        """将 Tool 转换为当前 provider 的 function-calling schema"""
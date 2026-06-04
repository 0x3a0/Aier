from abc import ABC, abstractmethod

from ..tool import Tool
from ..types import (
    AssistantMessageEvent, Context
)


class LLMModel(ABC):
    """ LLMModel """

    @abstractmethod
    def stream_invoke(
        self,
        context: Context,
        **kwargs
    ) -> AssistantMessageEvent:
        """ 流式输出 """
        ...
    
    @abstractmethod
    def conver_tools(
        self,
        tools: list[Tool]
    ) -> list[dict]:
        """ 
        将 Tool 转换为标准的 function-calling schema

        Returns:
            list[dict]: 标准的 function-calling schema 列表
        """
        ...
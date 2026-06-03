from abc import ABC, abstractmethod
from typing import Union

from ..types import (
    AssistantMessageEvent, AssistantMessage, Context
)


class LLMModel(ABC):
    """ LLMModel """

    @abstractmethod
    def stream_invoke(
        self,
        context: Context,
    ) -> AssistantMessageEvent:
        """ 流式输出 """
        ...

    def complete(
        self,
        context: Context
    ) -> AssistantMessage:
        """ 同步输出 """
        pass
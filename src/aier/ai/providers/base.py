from abc import ABC, abstractmethod
from typing import Union

from ..types import (
    AssistantMessageEvent, AssistantMessage, Message,
    Context
)


class LLMModel(ABC):
    """ LLMModel """

    @abstractmethod
    def stream_invoke(
        self,
        context: Context,
    ) -> Union[AssistantMessageEvent, AssistantMessage]:
        """ 流式输出模型调用结果 """
        ...
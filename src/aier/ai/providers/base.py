from abc import ABC, abstractmethod

from ..types import AssistantMessageEvent, Message


class LLMModel(ABC):
    """ LLMModel """

    @abstractmethod
    def stream_invoke(
        self,
        context: list[Message]
    ) -> AssistantMessageEvent:
        """ 流式输出模型调用结果 """
        ...

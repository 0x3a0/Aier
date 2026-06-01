from abc import ABC, abstractmethod

from .types import AssistantMessageEvent, Message


class BaseModel(ABC):
    """ BaseModel """

    @abstractmethod
    def stream_invoke(
        self,
        messages: list[Message]
    ) -> AssistantMessageEvent:
        """ 流式输出模型调用结果 """
        ...

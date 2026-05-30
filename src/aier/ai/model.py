from abc import ABC, abstractmethod
from typing import Optional, Iterator

from .types import (
    Message,
)


class BaseModel(ABC):
    """ BaseModel """

    @abstractmethod
    def stream_invoke(
        self,
        messages: list[Message],
        *,
        tools: Optional[list] = None,
    ) -> Iterator[Message]:
        """ 流式输出模型调用结果 """
        ...

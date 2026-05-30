from abc import ABC, abstractmethod
from typing import Optional, Iterator

from .stream_types import (
    Message, StreamEvent,
)


class BaseModel(ABC):
    """ BaseModel """

    @abstractmethod
    def stream_invoke(
        self,
        messages: list[Message],
        *,
        tools: Optional[list] = None,
    ) -> Iterator[StreamEvent]:
        """ 流式输出模型调用结果 """
        ...
        # messages = [message.to_openai_message() for message in messages]

        # resp = self.openai_client.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages,
        #     tools=tools,
        #     stream=True,
        #     temperature=self.temperature,
        #     extra_body=self.extra_body,
        #     reasoning_effort=self.reasoning_effort,
        #     **self.kwargs
        # )

        # for chunk in resp:
        #     print(chunk)
        #     delta = chunk.choices[0].delta

        #     # reasoning content
        #     if delta.reasoning_content:
        #         yield LLMResponseChunk(reasoning_content=delta.reasoning_content, model=chunk.model)

        #     # content
        #     if delta.content:
        #         yield LLMResponseChunk(content=delta.content, model=chunk.model)

        #     # stop
        #     if chunk.choices[0].finish_reason == "stop":
        #         yield StreamEnd(
        #             finish_reason="stop",
        #             completion_tokens=chunk.usage.completion_tokens,
        #             prompt_tokens=chunk.usage.prompt_tokens,
        #             total_tokens=chunk.usage.total_tokens,
        #         )

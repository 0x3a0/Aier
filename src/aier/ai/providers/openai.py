from typing import Any, Iterator, Optional

from ..model import BaseModel
from ..types import LLMResponseChunk, Message, StreamEnd


class OpenAIModel(BaseModel):
    """ OpenAI 模型 """
    def __init__(
        self,
        model_name: str,
        openai_client: Any,
        temperature: float = 0.8,
        thinking: bool = True,
        reasoning_effort: str = "high",
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.openai_client = openai_client

        self.temperature = temperature
        # 默认开启思考模式，思考强度为 "high"
        self.extra_body = {"thinking": {"type": "enabled"}} if thinking else None
        self.reasoning_effort = reasoning_effort

        # 其他参数
        self.kwargs = kwargs

    def stream_invoke(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
    ) -> Iterator[Message]:
        """ 流式输出模型调用结果 """
        messages = [message.to_openai_message() for message in messages]

        resp = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            stream=True,
            temperature=self.temperature,
            extra_body=self.extra_body,
            reasoning_effort=self.reasoning_effort,
            **self.kwargs
        )

        for chunk in resp:
            print(chunk)
            delta = chunk.choices[0].delta

            # reasoning content
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                yield LLMResponseChunk(reasoning_content=reasoning_content, model=chunk.model)

            # content
            if delta.content:
                yield LLMResponseChunk(content=delta.content, model=chunk.model)

            # stop
            if chunk.choices[0].finish_reason == "stop":
                yield StreamEnd(
                    finish_reason="stop",
                    completion_tokens=chunk.usage.completion_tokens,
                    prompt_tokens=chunk.usage.prompt_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )

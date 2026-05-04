from typing import Optional, Any, Iterator

from openai import OpenAI

from .types import (
    Message, LLMResponseChunk, StreamEvent, StreamEnd,
    ToolCall,
)


class ChatModel():
    """ ChatModel """
    def __init__(
        self,
        *,
        model_name: str,
        openai_client: Optional[OpenAI] = None,
        temperature: float = 0.8,
        thinking: bool = True,
        reasoning_effort: str = "high",
        **kwargs: Any
    ):
        self.model_name = model_name
        self.openai_client = openai_client

        self.temperature = temperature
        # 默认开启思考模式，思考强度为 "high"
        if thinking:
            self.extra_body = {"thinking": {"type": "enabled"}}
        self.reasoning_effort = reasoning_effort

        # 其他参数
        self.kwargs = kwargs

    def stream_invoke(
        self,
        messages: list[Message],
        *,
        tools: Optional[list] = None,
    ) -> Iterator[StreamEvent]:
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

        # 用于组装 tool call 的 buffer
        tool_call_buffers: dict[str, dict[str, str]] = {}

        for chunk in resp:
            print(chunk)
            delta = chunk.choices[0].delta

            # reasoning content
            if delta.reasoning_content:
                yield LLMResponseChunk(reasoning_content=delta.reasoning_content, model=chunk.model)

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

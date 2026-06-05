from typing import Union, Iterator

from openai import OpenAI

from .base import LLMModel
from ..tool import Tool
from ..types import (
    TextContent, ThinkingContent,
    AssistantMessageEvent, StreamStartEvent, StreamEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    TextStartEvent, TextDeltaEvent, TextEndEvent,
    AssistantMessage, Message,
    Context, Usage
)


class OpenAIModel(LLMModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str
    ) -> None:
        self.model_name = model_name
        self.client = self._create_client(api_key, base_url)

    def _create_client(
        self,
        api_key: str,
        base_url: str
    ) -> OpenAI:
        return OpenAI(api_key=api_key, base_url=base_url)

    def _transform_messages(
        self,
        messages: list[Message]
    ) -> list[dict]:
        """ 将输入的上下文转换成标准的 OpenAI 格式 """
        transformed_messages = []
        for msg in messages:
            transformed_messages.append(msg.model_dump())
        return transformed_messages

    def conver_tools(
        self,
        tools: list[Tool]
    ) -> list[dict]:
        """ 将 Tool 转换为标准的 function-calling schema """
        return []

    def build_params(
        self,
        context: Context,
        kwargs: dict
    ) -> dict:
        """ 构建 OpenAI 的请求参数 """
        messages = []
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})
        transformed_messages = self._transform_messages(context.messages)
        messages.extend(transformed_messages)

        params = {
            "model": self.model_name,
            "messages": messages,
            "stream": True
        }

        if "stream" in kwargs:
            kwargs.pop("stream", None)

        if "tools" in kwargs:
            kwargs.pop("tools", None)

        params.update(**kwargs)
        return params

    def stream_invoke(
        self,
        context: Context,
        **kwargs
        ) -> Iterator[AssistantMessageEvent]:
        """ 流式输出 """
        llm_output: AssistantMessage = AssistantMessage(
            role="assistant",
            content=[],
            model=self.model_name,
            response_id="",
            usage=Usage(
                input=0,
                output=0,
                total_tokens=0
            ),
            finish_reason="stop",
            create_timestamp=0
        )

        params = self.build_params(context, kwargs)
        stream = self.client.chat.completions.create(**params)

        thinking_block: Optional[ThinkingContent] = None
        text_block: Optional[TextContent] = None

        yield StreamStartEvent(portion=llm_output)

        for chunk in stream:
            llm_output.response_id = chunk.id
            llm_output.create_timestamp = chunk.created

            delta = chunk.choices[0].delta
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                if thinking_block is None:
                    thinking_block = ThinkingContent(thinking="")
                    llm_output.content.append(thinking_block)
                    yield ThinkingStartEvent(portion=llm_output)

                thinking_block.thinking += reasoning_content
                yield ThinkingDeltaEvent(delta=reasoning_content, portion=llm_output)
            
            if reasoning_content is None and thinking_block is not None:
                yield ThinkingEndEvent(content=thinking_block.thinking ,portion=llm_output)
                thinking_block = None

            text_content = getattr(delta, "content", None)
            if text_content:
                if text_block is None:
                    text_block = TextContent(text="")
                    llm_output.content.append(text_block)
                    yield TextStartEvent(portion=llm_output)

                text_block.text += text_content
                yield TextDeltaEvent(delta=text_content, portion=llm_output)

            # 当 text_content 为空字符串，且 finish_reason="stop" 时，说明模型完成生成
            finish_signal = getattr(chunk.choices[0], "finish_reason", None)
            if finish_signal == "stop":
                if not text_content and text_block is not None:
                    yield TextEndEvent(content=text_block.text, portion=llm_output)
                    text_block = None
                
                if llm_output.finish_reason == finish_signal:
                    llm_usage = chunk.usage
                    llm_output.usage.input = llm_usage.prompt_tokens
                    llm_output.usage.output = llm_usage.completion_tokens
                    llm_output.usage.total_tokens = llm_usage.total_tokens
                    yield StreamEndEvent(finish_reason=finish_signal, portion=llm_output)

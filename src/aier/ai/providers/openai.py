from typing import Iterator

from openai import OpenAI

from .base import LLMModel
from ..types import (
    TextContent, ThinkingContent, ToolCall,
    AssistantMessageEvent, StreamStartEvent, StreamEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    TextStartEvent, TextDeltaEvent, TextEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    AssistantMessage, Message,
    Context, Usage, Tool
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

    def _convert_tools(
        self,
        tools: list[Tool]
    ) -> list[dict]:
        """ 将 Tool 转换为标准的 function-calling schema """
        return [t.parameters for t in tools]

    def _build_params(
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

        params["tools"] = self._convert_tools(context.tools)

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

        params = self._build_params(context, kwargs)
        stream = self.client.chat.completions.create(**params)

        thinking_block: Optional[ThinkingContent] = None
        text_block: Optional[TextContent] = None
        tool_call_block: Optional[ToolCall] = None

        yield StreamStartEvent(portion=llm_output)

        for chunk in stream:
            llm_output.response_id = chunk.id
            llm_output.create_timestamp = chunk.created

            delta = chunk.choices[0].delta

            # reasoning_content 字段是否存在
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                if thinking_block is None:
                    thinking_block = ThinkingContent(thinking="")
                    llm_output.content.append(thinking_block)
                    yield ThinkingStartEvent(portion=llm_output)

                thinking_block.thinking += reasoning_content
                yield ThinkingDeltaEvent(delta=reasoning_content, portion=llm_output)
            
            if reasoning_content is None and thinking_block is not None:
                yield ThinkingEndEvent(content=thinking_block.thinking, portion=llm_output)
                thinking_block = None

            # content 字段是否存在
            text_content = getattr(delta, "content", None)
            if text_content:
                if text_block is None:
                    text_block = TextContent(text="")
                    llm_output.content.append(text_block)
                    yield TextStartEvent(portion=llm_output)

                text_block.text += text_content
                yield TextDeltaEvent(delta=text_content, portion=llm_output)

            # tool_calls 字段是否存在
            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls:

                # tool_calls 存在时，表明 text content 内容已经生成完毕，此处应该返回 text_end 事件
                # ChatCompletionChunk(id='...', choices=[Choice(delta=ChoiceDelta(content='...', function_call=None, tool_calls=None, reasoning_content=None), finish_reason=None, index=0, logprobs=None)], usage=None)
                # ChatCompletionChunk(id='...', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, tool_calls=[ChoiceDeltaToolCall(index=0, id='...', function=ChoiceDeltaToolCallFunction(arguments='', name='...'), type='function')]), finish_reason=None, index=0, logprobs=None)], usage=None)
                if text_content is None and text_block is not None:
                    yield TextEndEvent(content=text_block.text, portion=llm_output)
                    text_block = None

                if tool_call_block is None:
                    tool_call_block = ToolCall(
                        id=tool_calls[0].id,
                        name=tool_calls[0].function.name,
                        arguments=""
                    )
                    llm_output.content.append(tool_call_block)
                    yield ToolCallStartEvent(portion=llm_output)

                tool_call_block.arguments += tool_calls[0].function.arguments
                yield ToolCallDeltaEvent(delta=tool_calls[0].function.arguments, portion=llm_output)

            # 处理 text content 结束的情况，分为两种：1. 模型完成生成 2. 模型调用工具
            finish_signal = getattr(chunk.choices[0], "finish_reason", None)

            # 模型完成生成时的情况
            if finish_signal == "stop":
                if not text_content and text_block is not None:
                    yield TextEndEvent(content=text_block.text, portion=llm_output)
                    text_block = None

            # 模型调用工具时的情况
            if finish_signal == "tool_calls":
                yield ToolCallEndEvent(tool_call=tool_call_block, portion=llm_output)
                tool_call_block = None
            
            # usage
            usage = getattr(chunk, "usage", None)
            if usage:
                llm_output.usage.input = usage.prompt_tokens
                llm_output.usage.output = usage.completion_tokens
                llm_output.usage.total_tokens = usage.total_tokens
                yield StreamEndEvent(finish_reason=finish_signal, portion=llm_output)
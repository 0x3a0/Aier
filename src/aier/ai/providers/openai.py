from collections.abc import Iterable, Iterator, Sequence
from typing import Any, cast, override

from openai import OpenAI

from ..tool import Tool
from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    FinishReason,
    Message,
    StreamEndEvent,
    StreamStartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from .base import LLMModel

# OpenAI SDK 的 chunk / delta 结构包含大量可选字段，且各供应商会加挂
# 自定义字段（如 DeepSeek 的 reasoning_content），故以不透明类型承载，
# 在取值处再做精确化处理。
Chunk = object
Delta = object
RequestParams = dict[str, Any]
# 单个 chunk 的处理结果：事件列表 + 更新后的三个累积块
ChunkResult = tuple[
    list[AssistantMessageEvent],
    ThinkingContent | None,
    TextContent | None,
    ToolCall | None,
]


def _field(source: object, name: str, default: Any = None) -> Any:
    """按键取值，缺失时回退到默认值。

    统一收口了 `getattr` 的三参形式，避免每个调用点都产生 Unknown 类型。
    """
    return getattr(source, name, default)


class OpenAIModel(LLMModel):
    model_name: str
    client: OpenAI

    def __init__(self, model_name: str, api_key: str, base_url: str) -> None:
        self.model_name = model_name
        self.client = self._create_client(api_key, base_url)

    def _create_client(self, api_key: str, base_url: str) -> OpenAI:
        return OpenAI(api_key=api_key, base_url=base_url)

    def _transform_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """将输入的上下文转换成标准的 OpenAI 格式"""
        transformed_messages: list[dict[str, Any]] = []
        for msg in messages:
            transformed_messages.append(dict(msg.model_dump()))
        return transformed_messages

    @override
    def _convert_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """将 Tool 转换为标准的 function-calling schema"""
        return [t.schema() for t in tools]

    def _build_params(self, context: Context, kwargs: dict[str, Any]) -> RequestParams:
        """构建 OpenAI 的请求参数"""
        messages: list[dict[str, Any]] = []
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})
        messages.extend(self._transform_messages(context.messages))

        params: RequestParams = {"model": self.model_name, "messages": messages, "stream": True}

        # Context.tools 默认为 None，未注册工具时不应把 tools 传给 API
        if context.tools:
            params["tools"] = self._convert_tools(context.tools)

        # stream / tools 由 provider 自行决定，忽略调用方传入的同名参数
        forwarded = {k: v for k, v in kwargs.items() if k not in ("stream", "tools")}
        params.update(forwarded)
        return params

    @override
    def stream_invoke(self, context: Context, **kwargs: Any) -> Iterator[AssistantMessageEvent]:
        """流式输出"""
        llm_output = AssistantMessage(
            role="assistant",
            content=[],
            model=self.model_name,
            response_id="",
            usage=Usage(input=0, output=0, total_tokens=0),
            finish_reason="stop",
            create_timestamp=0,
        )

        params = self._build_params(context, kwargs)
        stream = cast(Iterable[Chunk], self.client.chat.completions.create(**params))

        yield StreamStartEvent(portion=llm_output)

        thinking_block: ThinkingContent | None = None
        text_block: TextContent | None = None
        tool_call_block: ToolCall | None = None

        for chunk in stream:
            # usage 汇总 chunk 的 id / created 为 None，不应覆盖已有的响应信息
            chunk_id = _field(chunk, "id")
            if chunk_id is not None:
                llm_output.response_id = chunk_id

            chunk_created = _field(chunk, "created")
            if chunk_created is not None:
                llm_output.create_timestamp = chunk_created

            # OpenAI 在流末尾会追加 usage 汇总 chunk，其 choices 为空列表
            choices = _field(chunk, "choices", [])
            delta = _field(choices[0], "delta") if choices else None
            finish_reason = _field(choices[0], "finish_reason") if choices else None
            if finish_reason is not None:
                # 记住结束原因：末尾的 usage 汇总 chunk 不再携带 finish_reason
                llm_output.finish_reason = finish_reason

            events, thinking_block, text_block, tool_call_block = self._chunk_events(
                llm_output, delta, thinking_block, text_block, tool_call_block, finish_reason
            )
            yield from events

            usage = _field(chunk, "usage")
            if usage:
                llm_output.usage.input = _field(usage, "prompt_tokens", 0)
                llm_output.usage.output = _field(usage, "completion_tokens", 0)
                llm_output.usage.total_tokens = _field(usage, "total_tokens", 0)
                break

        # 收尾必须给出终态事件，避免调用方拿到的最后一个事件是 text_delta
        yield StreamEndEvent(finish_reason=llm_output.finish_reason, portion=llm_output)

    def _chunk_events(
        self,
        llm_output: AssistantMessage,
        delta: Delta | None,
        thinking_block: ThinkingContent | None,
        text_block: TextContent | None,
        tool_call_block: ToolCall | None,
        finish_reason: FinishReason | None,
    ) -> ChunkResult:
        """把一个 chunk 转成零到多个事件，并返回更新后的 block 状态"""
        events: list[AssistantMessageEvent] = []

        if delta is None:
            return events, thinking_block, text_block, tool_call_block

        # reasoning_content 字段是否存在
        reasoning_content = _field(delta, "reasoning_content")
        if reasoning_content:
            if thinking_block is None:
                thinking_block = ThinkingContent(thinking="")
                llm_output.content.append(thinking_block)
                events.append(ThinkingStartEvent(portion=llm_output))

            thinking_block.thinking += reasoning_content
            events.append(ThinkingDeltaEvent(delta=reasoning_content, portion=llm_output))

        if reasoning_content is None and thinking_block is not None:
            events.append(ThinkingEndEvent(content=thinking_block.thinking, portion=llm_output))
            thinking_block = None

        # content 字段是否存在
        text_content = _field(delta, "content")
        if text_content:
            if text_block is None:
                text_block = TextContent(text="")
                llm_output.content.append(text_block)
                events.append(TextStartEvent(portion=llm_output))

            text_block.text += text_content
            events.append(TextDeltaEvent(delta=text_content, portion=llm_output))

        # tool_calls 字段是否存在
        tool_calls = _field(delta, "tool_calls")
        if tool_calls:
            # tool_calls 存在时，表明 text content 内容已经生成完毕，此处应该返回 text_end 事件
            if text_content is None and text_block is not None:
                events.append(TextEndEvent(content=text_block.text, portion=llm_output))
                text_block = None

            first_call = tool_calls[0]
            call_function = _field(first_call, "function")
            if tool_call_block is None:
                tool_call_block = ToolCall(
                    id=_field(first_call, "id", ""),
                    name=_field(call_function, "name", ""),
                    arguments="",
                )
                llm_output.content.append(tool_call_block)
                events.append(ToolCallStartEvent(portion=llm_output))

            argument_delta = _field(call_function, "arguments", "")
            tool_call_block.arguments += argument_delta
            events.append(ToolCallDeltaEvent(delta=argument_delta, portion=llm_output))

        # 处理 text content 结束的情况，分为两种：1. 模型完成生成 2. 模型调用工具
        if finish_reason == "stop" and not text_content and text_block is not None:
            events.append(TextEndEvent(content=text_block.text, portion=llm_output))
            text_block = None

        if finish_reason == "tool_calls" and tool_call_block is not None:
            events.append(ToolCallEndEvent(tool_call=tool_call_block, portion=llm_output))
            tool_call_block = None

        return events, thinking_block, text_block, tool_call_block

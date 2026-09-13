from collections.abc import Iterable, Iterator, Sequence
from typing import Any, cast, override

from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
)

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

RequestParams = dict[str, Any]


class VendorChoiceDelta(ChoiceDelta):
    """OpenAI 兼容供应商会加挂自定义字段，这里补上已知的扩展。

    DeepSeek / Z.ai 等以 reasoning_content 返回推理内容，官方 SDK 的
    ChoiceDelta 并未声明该字段（运行时因 extra="allow" 会被保留）。
    """

    reasoning_content: str | None = None


def _as_vendor_delta(delta: ChoiceDelta) -> VendorChoiceDelta:
    return cast(VendorChoiceDelta, delta)


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
        stream = cast(Iterable[ChatCompletionChunk], self.client.chat.completions.create(**params))

        yield StreamStartEvent(portion=llm_output)

        thinking_block: ThinkingContent | None = None
        text_block: TextContent | None = None
        tool_call_block: ToolCall | None = None

        for chunk in stream:
            # usage 汇总 chunk 的 id / created 实测为 None，但 SDK 把它们声明为
            # 非可选（str / int）。运行时的真实取值优先于类型声明，故保留判断：
            # 直接赋值会把 response_id / create_timestamp 覆盖成 None。
            if chunk.id is not None:  # pyright: ignore[reportUnnecessaryComparison]
                llm_output.response_id = chunk.id

            if chunk.created is not None:  # pyright: ignore[reportUnnecessaryComparison]
                llm_output.create_timestamp = chunk.created

            # OpenAI 在流末尾会追加 usage 汇总 chunk，其 choices 为空列表
            choice: Choice | None = chunk.choices[0] if chunk.choices else None
            delta = choice.delta if choice is not None else None
            finish_reason = _parse_finish_reason(choice)
            if finish_reason is not None:
                # 记住结束原因：末尾的 usage 汇总 chunk 不再携带 finish_reason
                llm_output.finish_reason = finish_reason

            events, thinking_block, text_block, tool_call_block = self._chunk_events(
                llm_output, delta, thinking_block, text_block, tool_call_block, finish_reason
            )
            yield from events

            if chunk.usage is not None:
                llm_output.usage.input = chunk.usage.prompt_tokens
                llm_output.usage.output = chunk.usage.completion_tokens
                llm_output.usage.total_tokens = chunk.usage.total_tokens
                break

        # 收尾必须给出终态事件，避免调用方拿到的最后一个事件是 text_delta
        yield StreamEndEvent(finish_reason=llm_output.finish_reason, portion=llm_output)

    def _chunk_events(
        self,
        llm_output: AssistantMessage,
        delta: ChoiceDelta | None,
        thinking_block: ThinkingContent | None,
        text_block: TextContent | None,
        tool_call_block: ToolCall | None,
        finish_reason: FinishReason | None,
    ) -> tuple[
        list[AssistantMessageEvent],
        ThinkingContent | None,
        TextContent | None,
        ToolCall | None,
    ]:
        """把一个 chunk 转成零到多个事件，并返回更新后的 block 状态"""
        events: list[AssistantMessageEvent] = []

        if delta is None:
            return events, thinking_block, text_block, tool_call_block

        # reasoning_content 是供应商扩展字段，运行时可能不存在
        reasoning_content = _as_vendor_delta(delta).reasoning_content
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
        text_content = delta.content
        if text_content:
            if text_block is None:
                text_block = TextContent(text="")
                llm_output.content.append(text_block)
                events.append(TextStartEvent(portion=llm_output))

            text_block.text += text_content
            events.append(TextDeltaEvent(delta=text_content, portion=llm_output))

        # tool_calls 字段是否存在
        if delta.tool_calls:
            # tool_calls 存在时，表明 text content 内容已经生成完毕
            if text_content is None and text_block is not None:
                events.append(TextEndEvent(content=text_block.text, portion=llm_output))
                text_block = None

            first_call = delta.tool_calls[0]
            if tool_call_block is None:
                tool_call_block = ToolCall(
                    id=first_call.id or "",
                    name=_function_name(first_call),
                    arguments="",
                )
                llm_output.content.append(tool_call_block)
                events.append(ToolCallStartEvent(portion=llm_output))

            argument_delta = _function_arguments(first_call)
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


def _function_name(tool_call: ChoiceDeltaToolCall) -> str:
    if tool_call.function is None or tool_call.function.name is None:
        return ""
    return tool_call.function.name


def _function_arguments(tool_call: ChoiceDeltaToolCall) -> str:
    if tool_call.function is None or tool_call.function.arguments is None:
        return ""
    return tool_call.function.arguments


def _parse_finish_reason(choice: Choice | None) -> FinishReason | None:
    """把 SDK 的 finish_reason 收窄到项目自己的 FinishReason。

    SDK 声明为 Literal["stop", "length", "tool_calls", "content_filter",
    "function_call"]，其中 function_call 是已废弃的旧协议，不纳入支持范围。
    """
    reason = choice.finish_reason if choice is not None else None
    if reason in ("stop", "length", "tool_calls", "content_filter"):
        return reason
    return None

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from .tool import Tool

BaseModel.__str__ = BaseModel.__repr__

#: 模型结束生成的原因。OpenAI 兼容接口除 stop / tool_calls 外，
#: 还会在触达 token 上限或内容过滤时返回 length / content_filter。
FinishReason = Literal["stop", "tool_calls", "length", "content_filter"]


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: str


class Usage(BaseModel):
    input: int
    output: int
    total_tokens: int


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[
        TextContent | ThinkingContent | ToolCall
    ]  # LLM返回的信息会包含多种可能，例如：content、reasoning_content、tool_call 等
    provider: str | None = None
    model: str
    response_id: str
    usage: Usage
    finish_reason: FinishReason
    create_timestamp: int


class ToolResultMessage(BaseModel):
    role: Literal["tool"] = "tool"
    id: str
    name: str
    content: list[TextContent]


Message = UserMessage | AssistantMessage | ToolResultMessage


class Context(BaseModel):
    # 允许使用 Pydantic 外的类型
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    system_prompt: str | None = None
    messages: list[Message]
    tools: list[Tool] | None = None


class StreamStartEvent(BaseModel):
    type: Literal["stream_start"] = "stream_start"
    portion: AssistantMessage


class ThinkingStartEvent(BaseModel):
    type: Literal["thinking_start"] = "thinking_start"
    portion: AssistantMessage


class ThinkingDeltaEvent(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str
    portion: AssistantMessage


class ThinkingEndEvent(BaseModel):
    type: Literal["thinking_end"] = "thinking_end"
    content: str
    portion: AssistantMessage


class TextStartEvent(BaseModel):
    type: Literal["text_start"] = "text_start"
    portion: AssistantMessage


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    delta: str
    portion: AssistantMessage


class TextEndEvent(BaseModel):
    type: Literal["text_end"] = "text_end"
    content: str
    portion: AssistantMessage


class ToolCallStartEvent(BaseModel):
    type: Literal["tool_call_start"] = "tool_call_start"
    portion: AssistantMessage


class ToolCallDeltaEvent(BaseModel):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    delta: str
    portion: AssistantMessage


class ToolCallEndEvent(BaseModel):
    type: Literal["tool_call_end"] = "tool_call_end"
    tool_call: ToolCall
    portion: AssistantMessage


class StreamEndEvent(BaseModel):
    type: Literal["stream_end"] = "stream_end"
    finish_reason: FinishReason
    portion: AssistantMessage


AssistantMessageEvent = (
    StreamStartEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | StreamEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
)

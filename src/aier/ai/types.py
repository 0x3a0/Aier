from typing import Literal, Union, Optional
from pydantic import BaseModel
from .tool import Tool


BaseModel.__str__ = BaseModel.__repr__

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str

class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"

class Usage(BaseModel):
    input: int
    output: int
    total_tokens: int

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str

class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[Union[TextContent, ThinkingContent]]      # LLM返回的信息会包含多种可能，例如：content、reasoning_content、tool_call 等
    provider: Optional[str] = None
    model: str
    response_id: str
    usage: Usage
    finish_reason: Literal["stop"]
    create_timestamp: int
    
class ToolResultMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: TextContent

Message = UserMessage | AssistantMessage | ToolResultMessage

class Context(BaseModel):
    system_prompt: Optional[str] = None
    messages: list[Message]
    tools: Optional[list[Tool]] = None

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

class StreamEndEvent(BaseModel):
    type: Literal["stream_end"] = "stream_end"
    finish_reason: Literal["stop"]
    portion: AssistantMessage

AssistantMessageEvent = StreamStartEvent | ThinkingDeltaEvent | ThinkingEndEvent | TextStartEvent | TextDeltaEvent | TextEndEvent | StreamEndEvent

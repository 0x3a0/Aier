from pydantic import BaseModel
from typing import Literal


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str

class Usage(BaseModel):
    input: int
    output: int
    total_tokens: int

class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str | TextContent

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | TextContent

class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[TextContent]
    provider: str
    model: str
    response_id: str
    usage: Usage
    finish_reason: Literal["stop"]
    create_timestamp: int
    
class ToolResultMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: TextContent

Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage

class StartEvent(BaseModel):
    type: Literal["start_event"] = "start_event"
    portion: AssistantMessage

class ThinkingDeltaEvent(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str
    portion: AssistantMessage

class ThinkingEndEvent(BaseModel):
    type: Literal["thinking_end"] = "thinking_end"
    portion: AssistantMessage

AssistantMessageEvent = StartEvent | ThinkingDeltaEvent | ThinkingEndEvent


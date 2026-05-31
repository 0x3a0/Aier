from pydantic import BaseModel
from typing import Literal


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str

class SystemMessage(BaseModel):
    pass

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: TextContent

class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[TextContent]

class ToolResultMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: TextContent

class StreamStart(BaseModel):
    type: Literal["stream_start"] = "stream_start"
    message_id: str


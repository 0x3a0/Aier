from typing import Literal, Union
from pydantic import BaseModel


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

Message = UserMessage | AssistantMessage | ToolResultMessage

class Context(BaseModel):
    system_prompt: str = ""
    messages: list[Message]

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


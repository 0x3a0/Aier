from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMResponseChunk:
    tool_calls: Optional[list[ToolCall]] = None
    reasoning_content: Optional[str] = None
    content: Optional[str] = None
    model: str = ""


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    tool_calls: Optional[list[ToolCall]] = None
    content: str = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_openai_message(self) -> dict:
        """ Message to OpenAI message dict """
        if self.role == "tool":
            pass

        return {"role": self.role, "content": self.content}


@dataclass
class StreamEnd:
    finish_reason: str
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0


# 类型别名，方便类型标注
StreamEvent = LLMResponseChunk | StreamEnd

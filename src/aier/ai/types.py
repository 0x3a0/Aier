from dataclasses import dataclass
from typing import Literal


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str

    def to_openai_message(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass
class LLMResponseChunk:
    content: str = ""
    reasoning_content: str = ""
    model: str = ""


@dataclass
class StreamEnd:
    finish_reason: str
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0

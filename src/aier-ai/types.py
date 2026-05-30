from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Message:
    pass


@dataclass
class StreamEnd:
    finish_reason: str
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
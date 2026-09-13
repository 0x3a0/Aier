from collections.abc import Sequence
from typing import Any


class ShortTermMemory:
    """短期记忆"""

    max_messages: int
    messages: list[dict[str, Any]]

    def __init__(self, max_messages: int = 15) -> None:
        self.max_messages = max_messages
        self.messages = []

    def add(self, message: dict[str, Any]) -> None:
        """添加 message 到短期记忆"""
        # 先剔除最旧的一条再追加，保证长度不超过 max_messages
        if len(self.messages) >= self.max_messages:
            _ = self.messages.pop(0)

        self.messages.append(message)

    def all_messages(self) -> Sequence[dict[str, Any]]:
        """获取所有短期记忆"""
        return self.messages

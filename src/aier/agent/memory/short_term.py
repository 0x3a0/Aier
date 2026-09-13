class ShortTermMemory:
    """短期记忆"""

    def __init__(self, max_messages: int = 15) -> None:
        self.messages: list[dict[str, str] | None] = []
        self.max_messages = max_messages

    def add(self, message: dict[str, str]) -> None:
        """添加 message 到短期记忆"""
        if len(self.messages) == self.max_messages:
            self.messages.pop(0)

        self.messages.append(message)

    def all_messages(self) -> list[dict[str, str] | None]:
        """获取所有短期记忆"""
        return self.messages

from typing import Optional


class ShortTermMemory:
    """ 短期记忆 """
    def __init__(self, max_messages: int = 15) -> None:
        self.messages: list[Optional[dict[str, str]]] = []
        self.max_messages = max_messages

    def add(self, message: dict[str, str]) -> None:
        """ 添加 message 到短期记忆 """
        if len(self.messages) == self.max_messages:
            self.messages.pop(0)
        
        self.messages.append(message)

    def all_messages(self) -> list[Optional[dict[str, str]]]:
        """ 获取所有短期记忆 """
        return self.messages
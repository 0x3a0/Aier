from abc import ABC, abstractmethod


class Tool(ABC):
    @abstractmethod
    def execute(self, context: Context) -> str:
        """ 运行工具 """
        pass
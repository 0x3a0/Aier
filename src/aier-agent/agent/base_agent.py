from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def _execute_tool(self, tool_call_buffer: dict[str, str]) -> str:
        """ 执行工具调用 """
        pass
    
    @abstractmethod
    def run(self, input: str) -> str:
        """ 启动agent """
        pass
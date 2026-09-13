from abc import ABC, abstractmethod
from typing import Any

#: 工具参数的 JSON-Schema 描述，例如 {"city": {"type": "string"}}
ToolParameters = dict[str, Any]
#: 标准 function-calling schema
ToolSchema = dict[str, Any]


class Tool(ABC):
    name: str
    description: str
    parameters: ToolParameters

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """运行工具"""

    def schema(self) -> ToolSchema:
        """转换为标准 function-calling schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }

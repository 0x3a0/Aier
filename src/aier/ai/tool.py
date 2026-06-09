from abc import ABC, abstractmethod


class Tool(ABC):
    
    name: str
    description: str
    parameters: dict

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """ 运行工具 """
        ...

    def schema(self) -> dict:
        """ 转换为标准 function-calling schema """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys())
                }
            }
        }
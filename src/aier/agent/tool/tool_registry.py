from collections.abc import Callable
from inspect import Parameter
from typing import Any

#: 参数类型标注 → JSON-Schema 类型名
_JSON_SCHEMA_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolRegistry:
    """
    ToolRegistry 类
    用于注册和管理工具实例
    该类提供了一个 register 装饰器方法，用于注册工具实例
    """

    tools: list[dict[str, Any]]
    tool_funcs: dict[str, Callable[..., Any]]

    def __init__(self) -> None:
        self.tools = []
        self.tool_funcs = {}

    def _load_defaults(self) -> None:
        """加载默认工具"""

    def _parse_arg_property(self, arg: Parameter) -> dict[str, str]:
        """解析参数类型"""
        annotation = arg.annotation

        if annotation is Parameter.empty:
            raise NotImplementedError(f"参数 {arg.name} 缺少类型标注")

        schema_type = _JSON_SCHEMA_TYPES.get(annotation)
        if schema_type is None:
            raise NotImplementedError(f"暂不支持的参数类型: {annotation}")

        return {"type": schema_type}

    def register(
        self, *, description: str, parameters: dict[str, Any]
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        注册 tool 的装饰器方法
        生成符合模型调用格式的 Function Tool

        :param description: 工具函数的描述
        :param parameters: 工具函数的参数描述
        例如: {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "需要查询的城市名称"
                }
            },
            "required": ["city"]
        }
        """

        def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_schema: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": description,
                    "parameters": parameters,
                },
            }

            self.tools.append(tool_schema)
            self.tool_funcs[func.__name__] = func

            return func

        return wrapper

    def get_registered_tools(self) -> list[dict[str, Any]]:
        """获取已注册的工具列表"""
        return self.tools

    def get_registered_tool_funcs(self) -> dict[str, Callable[..., Any]]:
        """获取已注册的工具函数列表"""
        return self.tool_funcs

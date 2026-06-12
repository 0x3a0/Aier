from pydantic import BaseModel
from typing import Callable


class Tool(BaseModel):
    tool_schema: dict
    func: Callable

def build_tool(
    name: str,
    *,
    description: str,
    parameters: dict,
    func: Callable
) -> Tool:
    if not description:
        raise ValueError("description is required")
    if not parameters:
        raise ValueError("parameters is required")
    if not func:
        raise ValueError("func is required")
    if not isinstance(func, Callable):
        raise ValueError("func must be a callable")

    return Tool(
        tool_schema={
            "type": "function",
            "function": {
                "name": name or func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": list(parameters.keys())
                }
            }
        },
        func=func
    )
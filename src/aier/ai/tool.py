from pydantic import BaseModel
from typing import Callable, Optional


class Tool(BaseModel):
    tool_schema: dict
    func: Callable

def build_tool(
    name: str,
    *,
    description: str,
    parameters: dict,
    func: Optional[Callable]
) -> Tool:
    if not description:
        raise ValueError("description is required")
    if not parameters:
        raise ValueError("parameters is required")
    if not func:
        raise ValueError("func is required")

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
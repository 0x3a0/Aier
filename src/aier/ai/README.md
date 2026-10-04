# aier-ai

Aier mini-agent framework 的模型适配层，定义模型接口、消息、上下文、工具抽象与流式事件。目前实现 OpenAI Chat Completions 格式；Anthropic、Google 原生格式仍为后续计划。

`Context` 保存当前传入的消息，不会自动持久化对话或执行工具。Agent 循环、工具结果回填和记忆策略由 `aier.agent` 后续实现。参见[项目说明](../../../README.md)与[技术路线图](../../../docs/ROADMAP.md)。

## 快速入门

```python
from os import environ

from dotenv import load_dotenv

from aier.ai import Context, UserMessage, get_model

_ = load_dotenv()
model = get_model("openai", environ["DS_MODEL_NAME"], environ["DS_API_KEY"], environ["DS_BASE_URL"])
context = Context(system_prompt="你的名字是0xAI", messages=[UserMessage(content="你叫什么")])
for chunk in model.stream_invoke(context):
    print(chunk)
```

框架底层默认为流式输出，对话的过程中会不断发出 stream event 事件，可以在[事件类型](#事件类型)中查看所有的事件类型。

先在项目根目录运行 `uv sync --locked`，再用 `uv run --no-sync python` 运行保存后的示例。`python-dotenv` 是开发依赖。示例需要在本地环境或 `.env` 中设置 `DS_MODEL_NAME`、`DS_API_KEY`、`DS_BASE_URL`；模型名和地址按供应商实际接口填写，密钥不写入源文件。

## 更多的聊天会话参数

当前可用方法是 `stream_invoke`，可以透传底层 SDK 的参数；`complete` 尚未实现。参数支持取决于供应商与模型，下面的 `extra_body` 仅适用于支持该字段的接口：

```python
from os import environ

from dotenv import load_dotenv

from aier.ai import Context, UserMessage, get_model

_ = load_dotenv()
model = get_model("openai", environ["DS_MODEL_NAME"], environ["DS_API_KEY"], environ["DS_BASE_URL"])
context = Context(system_prompt="你的名字是0xAI", messages=[UserMessage(content="你叫什么")])
for event in model.stream_invoke(
    context, temperature=0.8, extra_body={"thinking": {"type": "disabled"}}
):
    print(event)
```

## 事件类型

| 事件名称 | 描述 | 事件标识 | 属性 |
| --- | --- | --- | --- |
| `StreamStartEvent` | 流式对话开始 | `stream_start` | `portion`: `AssistantMessage` |
| `ThinkingStartEvent` | 思考开始 | `thinking_start` | `portion`: `AssistantMessage` |
| `ThinkingDeltaEvent` | 思考内容增量 | `thinking_delta` | `delta`: `str`<br>`portion`: `AssistantMessage` |
| `ThinkingEndEvent` | 思考结束 | `thinking_end` | `content`: `str`<br>`portion`: `AssistantMessage` |
| `TextStartEvent` | 文本生成开始 | `text_start` | `portion`: `AssistantMessage` |
| `TextDeltaEvent` | 文本内容增量 | `text_delta` | `delta`: `str`<br>`portion`: `AssistantMessage` |
| `TextEndEvent` | 文本生成结束 | `text_end` | `content`: `str`<br>`portion`: `AssistantMessage` |
| `ToolCallStartEvent` | 工具调用开始 | `tool_call_start` | `portion`: `AssistantMessage` |
| `ToolCallDeltaEvent` | 工具参数增量 | `tool_call_delta` | `delta`: `str`<br>`portion`: `AssistantMessage` |
| `ToolCallEndEvent` | 工具调用生成结束 | `tool_call_end` | `tool_call`: `ToolCall`<br>`portion`: `AssistantMessage` |
| `StreamEndEvent` | 流式对话结束 | `stream_end` | `finish_reason`: `"stop"` / `"tool_calls"` / `"length"` / `"content_filter"`<br>`portion`: `AssistantMessage` |

工具调用事件表示模型生成了调用请求，并不代表工具已执行。`length` 和 `content_filter` 也不代表任务成功。当前工具调用流仅组装首个调用，多调用支持列入路线图。

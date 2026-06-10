# aier-ai

统一的 LLM API，具备工具注册和简单的上下文保存功能

## 快速入门

```python
from aier.ai import get_model, UserMessage, Context

load_dotenv()
model = get_model("openai", "deepseek-v4-flash", getenv("DS_API_KEY"), getenv("DS_BASE_URL"))
context = Context(
    system_prompt="你的名字是0xAI",
    messages=[
        UserMessage(content="你叫什么")
    ]
)
for chunk in model.stream_invoke(context):
    print(chunk)
```

框架底层默认为流式输出，对话的过程中会不断发出 stream event 事件，可以在[事件类型](#事件类型)中查看所有的事件类型。

## 更多的聊天会话参数

`stream_invoke`与`complete`方法可以传递更多的参数用于进行更精细的控制，参数的支持可能会因用于生成响应的模型而有所不同：

```python
from aier.ai import get_model, UserMessage, Context

load_dotenv()
model = get_model("openai", "deepseek-v4-flash", getenv("DS_API_KEY"), getenv("DS_BASE_URL"))
context = Context(
    system_prompt="你的名字是0xAI",
    messages=[
        UserMessage(content="你叫什么")
    ]
)
for event in model.stream_invoke(context, temperature=0.8, extra_body={"thinking": {"type": "disabled"}}):
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
| `StreamEndEvent` | 流式对话结束 | `stream_end` | `finish_reason`: `"stop"`<br>`portion`: `AssistantMessage` |

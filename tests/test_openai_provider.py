"""OpenAIModel.stream_invoke 的回归测试。

这里用假客户端跑通真实的流式代码路径，覆盖一批评测过程中暴露出来的运行期缺陷：

1. `stream_invoke` 体内使用 `Optional[...]` 注解，但模块顶部的 `Optional` 导入
   已被移除。该表达式位于生成器函数体内，只有真正消费流式响应时才会触发
   `NameError`。
2. `_build_params` 无条件遍历 `context.tools`，而 `Context.tools` 默认是
   `None`，导致不带工具的纯对话直接 `TypeError`。
3. 循环内无条件读取 `chunk.id` / `chunk.created` 并索引 `chunk.choices[0]`，
   但 OpenAI 在流末尾会追加一个 usage 汇总 chunk（`choices` 为空、`id` 为
   `None`），这会直接 `IndexError`。
4. `StreamEndEvent` 曾用 usage chunk 的 `finish_reason=None` 构造，与
   `FinishReason` 字面量冲突，导致每次流式输出都在末尾 ValidationError。

由于这些表达式都藏在延迟执行的生成器体内，只有真正驱动流才能发现。
"""

from types import SimpleNamespace
from typing import Any, override

import pytest

from aier.ai import Context, Tool, UserMessage
from aier.ai.providers.openai import OpenAIModel
from aier.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingEndEvent,
    ToolCallEndEvent,
)


class _WeatherTool(Tool):
    """用于验证 tools 字段下发的最小可用工具。"""

    name: str = "get_weather"
    description: str = "查询天气"
    # 测试替身只会被实例化一次，共享该 dict 不会互相污染
    parameters: dict[str, Any] = {"city": {"type": "string"}}  # noqa: RUF012

    @override
    def execute(self, **kwargs: Any) -> str:
        return "晴"


class _StubCompletions:
    """假的 chat.completions 资源，记录收到的请求参数。"""

    chunks: list[SimpleNamespace]
    params: dict[str, Any] | None

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self.chunks = chunks
        self.params = None

    def create(self, **params: Any) -> list[SimpleNamespace]:
        self.params = params
        return self.chunks


def _make_model(
    monkeypatch: pytest.MonkeyPatch, chunks: list[SimpleNamespace]
) -> tuple[OpenAIModel, _StubCompletions]:
    """构造 OpenAIModel，并把 OpenAI 客户端替换为返回固定 chunk 的假客户端。"""
    completions = _StubCompletions(chunks)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def _fake_create_client(_self: OpenAIModel, _api_key: str, _base_url: str) -> Any:
        return fake_client

    monkeypatch.setattr(OpenAIModel, "_create_client", _fake_create_client)

    model = OpenAIModel("test-model", "sk-test", "https://example.invalid")
    return model, completions


def _chunk(delta: SimpleNamespace, finish_reason: str | None = None) -> SimpleNamespace:
    """普通增量 chunk：delta 有内容，usage 为 None。"""
    return SimpleNamespace(
        id="chatcmpl-test",
        created=1700000000,
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=None,
    )


def _usage_chunk(prompt: int, completion: int) -> SimpleNamespace:
    """流末尾的 usage 汇总 chunk：choices 为空列表、id 与 created 为 None。

    这是 OpenAI 的真实行为（stream_options={"include_usage": True}），
    也是旧代码 `chunk.choices[0]` 会崩掉的输入。
    """
    return SimpleNamespace(
        id=None,
        created=None,
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


def _text(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(content=content, reasoning_content=None, tool_calls=None)


def _context(content: str = "你好") -> Context:
    return Context(messages=[UserMessage(content=content)])


def _events(model: OpenAIModel, context: Context, **kwargs: Any) -> list[AssistantMessageEvent]:
    return list(model.stream_invoke(context, **kwargs))


def _types(events: list[AssistantMessageEvent]) -> list[str]:
    return [event.type for event in events]


def _portions(events: list[AssistantMessageEvent]) -> list[AssistantMessage]:
    return [event.portion for event in events]


def _deltas(events: list[AssistantMessageEvent]) -> list[str]:
    """取出所有 TextDeltaEvent 的增量文本（按类型收窄后再访问字段）。"""
    return [event.delta for event in events if isinstance(event, TextDeltaEvent)]


def _text_ends(events: list[AssistantMessageEvent]) -> list[str]:
    return [event.content for event in events if isinstance(event, TextEndEvent)]


def _thinking_ends(events: list[AssistantMessageEvent]) -> list[str]:
    return [event.content for event in events if isinstance(event, ThinkingEndEvent)]


def test_usage_only_final_chunk_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """回归：末尾 usage chunk 的 choices 为空列表。"""
    model, _ = _make_model(monkeypatch, [_usage_chunk(3, 4)])

    events = _events(model, _context())

    assert _types(events) == ["stream_start", "stream_end"]
    assert _portions(events)[-1].usage.total_tokens == 7


def test_text_only_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """纯文本流：确保 Optional 注解所在的代码路径可以正常执行。"""
    chunks = [
        _chunk(_text("你")),
        _chunk(_text("好")),
        _chunk(_text(None), "stop"),
        _usage_chunk(8, 2),
    ]
    model, _ = _make_model(monkeypatch, chunks)

    events = _events(model, _context())

    assert _types(events)[0] == "stream_start"
    assert _types(events)[1:3] == ["text_start", "text_delta"]
    assert _deltas(events) == ["你", "好"]
    assert _text_ends(events) == ["你好"]

    last = events[-1]
    assert last.type == "stream_end"
    assert last.finish_reason == "stop"
    assert last.portion.response_id == "chatcmpl-test"
    assert last.portion.create_timestamp == 1700000000
    assert last.portion.usage.input == 8
    assert last.portion.usage.output == 2


def test_thinking_then_text_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """思考 + 文本：验证 thinking_block 与 text_block 的分支切换。"""
    chunks = [
        _chunk(SimpleNamespace(content=None, reasoning_content="思考", tool_calls=None)),
        _chunk(_text(None)),
        _chunk(_text("答案")),
        _chunk(_text(None), "stop"),
        _usage_chunk(10, 3),
    ]
    model, _ = _make_model(monkeypatch, chunks)

    events = _events(model, _context())

    assert [t for t in _types(events) if t.startswith("thinking")] == [
        "thinking_start",
        "thinking_delta",
        "thinking_end",
    ]
    assert _thinking_ends(events) == ["思考"]
    assert _text_ends(events) == ["答案"]


def test_tool_call_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """工具调用流：正是当初会抛 NameError 的分支，必须显式覆盖。"""
    first_tool_delta = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name="get_weather", arguments='{"city":'),
    )
    second_tool_delta = SimpleNamespace(
        index=0,
        id=None,
        function=SimpleNamespace(name=None, arguments='"北京"}'),
    )
    chunks = [
        _chunk(_text("查询中")),
        _chunk(
            SimpleNamespace(content=None, reasoning_content=None, tool_calls=[first_tool_delta])
        ),
        _chunk(
            SimpleNamespace(content=None, reasoning_content=None, tool_calls=[second_tool_delta])
        ),
        _chunk(_text(None), "tool_calls"),
        _usage_chunk(10, 5),
    ]
    model, _ = _make_model(monkeypatch, chunks)

    events = _events(model, _context("天气"))

    assert [t for t in _types(events) if t.startswith("tool_call")] == [
        "tool_call_start",
        "tool_call_delta",
        "tool_call_delta",
        "tool_call_end",
    ]
    end_event = next(e for e in events if isinstance(e, ToolCallEndEvent))
    assert end_event.tool_call.id == "call_1"
    assert end_event.tool_call.name == "get_weather"
    assert end_event.tool_call.arguments == '{"city":"北京"}'

    # 文本块应在工具调用开始时收尾
    assert _text_ends(events) == ["查询中"]

    last = events[-1]
    assert last.type == "stream_end"
    assert last.finish_reason == "tool_calls"
    assert last.portion.usage.total_tokens == 15


def test_truncated_stream_uses_length_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """触达 token 上限时 OpenAI 返回 finish_reason="length"。

    旧实现把 finish_reason 收窄为 stop / tool_calls，这里会 ValidationError。
    """
    chunks = [_chunk(_text("被截断"), "length"), _usage_chunk(5, 9)]
    model, _ = _make_model(monkeypatch, chunks)

    events = _events(model, _context())

    assert events[-1].type == "stream_end"
    assert events[-1].finish_reason == "length"


def test_stream_without_finish_chunk_still_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """即使流里没有 usage 汇总 chunk，也必须以终态事件收尾。"""
    model, _ = _make_model(monkeypatch, [_chunk(_text("你好"))])

    events = _events(model, _context())

    assert events[-1].type == "stream_end"


def test_build_params_overrides_stream_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """stream / tools 由 provider 自行决定，调用方传入的同名参数应被忽略。"""
    model, completions = _make_model(monkeypatch, [])
    context = Context(messages=[UserMessage(content="hi")])

    _ = _events(model, context, stream=False, tools=["hacked"], temperature=0.5)

    params = completions.params
    assert params is not None
    assert params["stream"] is True
    assert params["temperature"] == 0.5
    assert params["model"] == "test-model"
    # 未注册工具时不下发 tools 字段（历史缺陷：会对 None 迭代并抛 TypeError）
    assert "tools" not in params


def test_context_without_tools_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Context.tools 默认为 None，纯对话场景必须能正常工作。

    历史缺陷：`_build_params` 无条件执行 `self._convert_tools(context.tools)`，
    而 `Context.tools` 默认是 None，导致不带工具的调用直接 TypeError。
    """
    assert Context(messages=[UserMessage(content="hi")]).tools is None

    model, _ = _make_model(monkeypatch, [_chunk(_text("ok"), "stop"), _usage_chunk(1, 1)])

    events = _events(model, _context())

    assert events[-1].type == "stream_end"


def test_registered_tools_are_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """注册了工具时，tools 字段应正常下发。"""
    tool = _WeatherTool()
    model, completions = _make_model(monkeypatch, [])
    context = Context(messages=[UserMessage(content="hi")], tools=[tool])

    _ = _events(model, context)

    params = completions.params
    assert params is not None
    assert params["tools"] == [tool.schema()]

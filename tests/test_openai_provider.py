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

由于这些表达式都藏在延迟执行的生成器体内，只有真正驱动流才能发现。
"""

from types import SimpleNamespace
from typing import ClassVar

import pytest

from aier.ai import Context, Tool, UserMessage
from aier.ai.providers import openai as openai_provider


class _WeatherTool(Tool):
    """用于验证 tools 字段下发的最小可用工具。"""

    name: ClassVar[str] = "get_weather"
    description: ClassVar[str] = "查询天气"
    parameters: ClassVar[dict] = {"city": {"type": "string"}}

    def execute(self, **kwargs) -> str:
        return "晴"


class _StubCompletions:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks
        self.params: dict | None = None

    def create(self, **params) -> list[SimpleNamespace]:
        self.params = params
        return self._chunks


def _make_model(monkeypatch: pytest.MonkeyPatch, chunks: list[SimpleNamespace]):
    """构造 OpenAIModel，并把 OpenAI 客户端替换为返回固定 chunk 的假客户端。"""
    completions = _StubCompletions(chunks)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setattr(
        openai_provider.OpenAIModel, "_create_client", lambda self, k, b: fake_client
    )

    model = openai_provider.OpenAIModel("test-model", "sk-test", "https://example.invalid")
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


def test_usage_only_final_chunk_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """回归：末尾 usage chunk 的 choices 为空列表。"""
    model, _ = _make_model(monkeypatch, [_usage_chunk(3, 4)])

    events = list(model.stream_invoke(_context()))

    assert [e.type for e in events] == ["stream_start", "stream_end"]
    assert events[-1].portion.usage.total_tokens == 7


def test_text_only_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """纯文本流：确保 Optional 注解所在的代码路径可以正常执行。"""
    chunks = [
        _chunk(_text("你")),
        _chunk(_text("好")),
        _chunk(_text(None), "stop"),
        _usage_chunk(8, 2),
    ]
    model, _ = _make_model(monkeypatch, chunks)

    events = list(model.stream_invoke(_context()))

    assert events[0].type == "stream_start"
    assert [e.type for e in events[1:3]] == ["text_start", "text_delta"]
    assert [e.delta for e in events if e.type == "text_delta"] == ["你", "好"]
    assert [e.content for e in events if e.type == "text_end"] == ["你好"]
    assert events[-1].type == "stream_end"
    assert events[-1].finish_reason == "stop"
    assert events[-1].portion.response_id == "chatcmpl-test"
    assert events[-1].portion.create_timestamp == 1700000000
    assert events[-1].portion.usage.input == 8
    assert events[-1].portion.usage.output == 2


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

    events = list(model.stream_invoke(_context()))

    assert [e.type for e in events if e.type.startswith("thinking")] == [
        "thinking_start",
        "thinking_delta",
        "thinking_end",
    ]
    assert [e.content for e in events if e.type == "thinking_end"] == ["思考"]
    assert [e.content for e in events if e.type == "text_end"] == ["答案"]


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
    tool_chunk = SimpleNamespace(
        content=None, reasoning_content=None, tool_calls=[first_tool_delta]
    )
    tool_chunk_2 = SimpleNamespace(
        content=None, reasoning_content=None, tool_calls=[second_tool_delta]
    )
    chunks = [
        _chunk(_text("查询中")),
        _chunk(tool_chunk),
        _chunk(tool_chunk_2),
        _chunk(_text(None), "tool_calls"),
        _usage_chunk(10, 5),
    ]
    model, _ = _make_model(monkeypatch, chunks)

    events = list(model.stream_invoke(_context("天气")))

    assert [e.type for e in events if e.type.startswith("tool_call")] == [
        "tool_call_start",
        "tool_call_delta",
        "tool_call_delta",
        "tool_call_end",
    ]
    end_event = next(e for e in events if e.type == "tool_call_end")
    assert end_event.tool_call.id == "call_1"
    assert end_event.tool_call.name == "get_weather"
    assert end_event.tool_call.arguments == '{"city":"北京"}'

    # 文本块应在工具调用开始时收尾
    assert [e.content for e in events if e.type == "text_end"] == ["查询中"]

    assert events[-1].type == "stream_end"
    assert events[-1].finish_reason == "tool_calls"
    assert events[-1].portion.usage.total_tokens == 15


def test_build_params_overrides_stream_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """stream / tools 由 provider 自行决定，调用方传入的同名参数应被忽略。"""
    model, completions = _make_model(monkeypatch, [])
    context = Context(messages=[UserMessage(content="hi")])

    list(model.stream_invoke(context, stream=False, tools=["hacked"], temperature=0.5))

    assert completions.params is not None
    assert completions.params["stream"] is True
    assert completions.params["temperature"] == 0.5
    assert completions.params["model"] == "test-model"
    # 未注册工具时不下发 tools 字段（历史缺陷：会对 None 迭代并抛 TypeError）
    assert "tools" not in completions.params


def test_context_without_tools_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Context.tools 默认为 None，纯对话场景必须能正常工作。

    历史缺陷：`_build_params` 无条件执行 `self._convert_tools(context.tools)`，
    而 `Context.tools` 默认是 None，导致不带工具的调用直接 TypeError。
    """
    assert Context(messages=[UserMessage(content="hi")]).tools is None

    model, _ = _make_model(monkeypatch, [_chunk(_text("ok"), "stop"), _usage_chunk(1, 1)])

    events = list(model.stream_invoke(_context()))

    assert events[-1].type == "stream_end"


def test_registered_tools_are_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """注册了工具时，tools 字段应正常下发。"""
    tool = _WeatherTool()
    model, completions = _make_model(monkeypatch, [])
    context = Context(messages=[UserMessage(content="hi")], tools=[tool])

    list(model.stream_invoke(context))

    assert completions.params is not None
    assert completions.params["tools"] == [tool.schema()]

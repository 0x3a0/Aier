# Aier 路线图

- 状态：草案
- 目标版本：aier 0.1.0 → 0.5.x
- 更新日期：2026

## 1. 范围

Aier 是一个 Agent 运行时（harness），用于在 OpenAI 兼容的 LLM API 之上构建单 Agent 与多 Agent 应用。目标运行环境为 Python 3.13，并将推理输出（`reasoning_content`）作为响应流的一等组成部分处理。

Aier 不追求：

- 重新实现图编排（由 LangGraph 覆盖）；
- 在 OpenAI 线协议之外提供厂商专属适配器（DeepSeek、Z.ai、Qwen 均遵循该协议）；
- 替代 MCP，或定义与之竞争的工具格式。

## 2. 设计约束

以下决策后续改动必须遵守；任何与之冲突的提案都需要明确说明理由。

- D1. 响应流是唯一事实源。记忆、钩子、护栏、追踪、人在回路均为事件流的消费者，而非独立的旁路通道。
- D2. 每次运行产生一份只追加的事件日志。记忆层级与会话状态均由该日志派生，使运行可回放、可续跑。
- D3. 工具是 MCP 资源。Aier 是 MCP 客户端，不定义自己的工具 schema。
- D4. 供应商调用透传 `**kwargs`。高级用户无需 fork 框架即可触达底层客户端。
- D5. 推理输出（`reasoning_content` → `thinking_*` 事件）被保留，不丢弃、不并入正文。

## 3. 里程碑

### M0 — 收敛（0.1.x）

目标：移除两套并行且未完成的工具系统，修正类型与事件契约。

| # | 任务 | 涉及文件 | 交付物 |
|---|------|---------|--------|
| M0.1 | 补上缺失的 `Optional` 导入 | `ai/providers/openai.py` | 模块可正常导入 |
| M0.2 | 统一消息类型：在 `AssistantMessage.content` 联合中加入 `ToolResult`，对齐 `Usage` 与 `finish_reason` | `ai/types.py` | 单一类型契约 |
| M0.3 | 删除 `ToolRegistry` 与 `Tool.schema()` 路径，只保留一种工具抽象 | 删除 `agent/tool/tool_registry.py` | 无重复工具系统 |
| M0.4 | 为事件契约与消息序列化补充 pytest 测试 | `tests/` | CI 通过 |

退出标准：`import aier` 成功；pytest 全部通过；仅保留一种工具抽象。

### M1 — 工具层与 ReAct 循环（0.2.x）

目标：具备 MCP 工具支持的、可运行的 Agent 循环。

| # | 任务 | 涉及文件 | 交付物 |
|---|------|---------|--------|
| M1.1 | `MCPTool` 封装 MCP Python SDK（`list_tools`/`call_tool`），将结果映射为 `ToolCall`/`ToolResult` | `ai/tools/mcp.py` | 模型可调用 MCP 工具 |
| M1.2 | 新增 `ToolResult` 内容块与消息重建逻辑，使工具结果回传模型 | `ai/types.py`、`ai/providers/openai.py` | 工具结果可回传 |
| M1.3 | 实现 `ReactAgent`：流式调用 → 收集事件 → 执行 `tool_calls` → 回灌结果 → 循环，以 `max_iter` 为界 | `agent/agent/react_agent.py`（当前为空） | 首个可运行 Agent |
| M1.4 | 从类型注解与 docstring 自动推导 function-calling schema，取代手写的参数字典 | `ai/tools/function.py` | 声明式工具注册 |
| M1.5 | 编写可运行的 MCP 示例 | `examples/model_tool_call.py`（当前为空） | 可运行 demo |

退出标准：`examples/model_tool_call.py` 完成「用户提问 → 工具调用 → 返回结果」的端到端闭环。

### M2 — 事件日志、记忆、回放（0.3.x）

目标：使事件流可持久化，运行可回放。

| # | 任务 | 涉及文件 | 交付物 |
|---|------|---------|--------|
| M2.1 | 只追加的 JSONL 日志，含 `session_id` / `seq` / `timestamp` | `agent/runtime/event_log.py` | 事件日志 |
| M2.2 | 运行生命周期 + 发布/订阅分发器，提供 `EventListener` 接口 | `agent/runtime/runtime.py` | 事件总线 |
| M2.3 | 三层记忆：短期（窗口）、工作（结构化状态）、情景（日志回放） | `agent/memory/` | 分层记忆 |
| M2.4 | `resume(session_id)` 与 `rewind(seq)` 从日志重建 `Context` 与状态 | `agent/runtime/replay.py` | 可续跑 |
| M2.5 | 输出 OpenTelemetry 风格的 span，可接入 Langfuse/Phoenix | `agent/runtime/tracing.py` | 可观测 |

退出标准：一次已记录的运行可回放到相同结果；`rewind` 到中间点结果确定。

### M3 — 技能、钩子、护栏、人在回路（0.4.x）

目标：程序化知识与可插拔的横切行为。

| # | 任务 | 涉及文件 | 交付物 |
|---|------|---------|--------|
| M3.1 | 按需加载 `SKILL.md` 与资源并注入上下文 | `agent/skill/loader.py`（当前为空） | 技能 |
| M3.2 | 生命周期钩子（`pre_tool_call` / `post_tool_call` / `on_text_delta`），支持阻断 | `agent/runtime/hooks.py` | 扩展点 |
| M3.3 | 输入/输出校验器，作为可否决的事件订阅者 | `agent/runtime/guardrails.py` | 校验 |
| M3.4 | `pre_tool_call` 钩子挂起等待人工批准 | `agent/runtime/hooks.py` | 人在回路 |
| M3.5 | 首批内置技能（规划、检索、编码） | `skills/` | 可复用资产 |

退出标准：一个钩子示例与一个技能示例可运行；护栏能拦截非法工具调用。

### M4 — 子 Agent、互操作、评测（0.5.x）

目标：委托、跨框架互通、离线评测。

| # | 任务 | 涉及文件 | 交付物 |
|---|------|---------|--------|
| M4.1 | 具备独立运行时与上下文的子 Agent | `agent/agent/sub_agent.py` | 委托 |
| M4.2 | 将 Aier Agent 暴露为 A2A agent | `agent/interop/a2a.py` | 互通 |
| M4.3 | 基于 M2 事件日志的离线回放评测（评测集 + 断言） | `evals/` | 可复现评测 |
| M4.4 | 定位、对比与迁移文档 | `docs/` | 文档 |

退出标准：两个 Aier Agent 可相互交接；评测集可离线回归。

## 4. 优先级

- P0 — M0、M1。没有可运行的循环就没有框架。
- P1 — M2、M3。回放与技能/钩子以较低成本承载大部分差异。
- P2 — M4。依赖 M2/M3，待核心稳定后再排期。

## 5. 非目标

- 不做图编排。
- 不做 Anthropic/Google 的一手适配器。
- 不内置向量库；长期记忆先以文件系统/事件日志起步（参考 deepagents）。
- 不做「抽象之上再抽象」的层；保留 `**kwargs` 逃生舱。

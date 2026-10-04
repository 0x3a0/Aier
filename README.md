# Aier — mini-agent framework

Aier 是一个从零实现的轻量级 Agent 框架，用于学习 Agent 底层机制，并逐步构建可复用、可评测的 Agent 系统。核心学习与工程主线为 **Agent Systems / Infrastructure**，重点关注 **Tool Use、Memory、Evaluation 与长期任务执行**。

> 当前处于早期开发阶段：已具备模型适配、消息与流式事件、工具抽象及短期记忆组件；Agent 执行循环尚未实现。未来能力以路线图为准。

## 项目结构与当前能力

| 模块 | 职责 | 当前状态 |
| --- | --- | --- |
| [`aier.ai`](src/aier/ai/README.md) | 模型接口、消息、上下文、工具 schema、流式输出 | 已实现 OpenAI 格式适配；Anthropic、Google 原生格式尚未实现 |
| [`aier.agent`](src/aier/agent/) | Agent 循环、工具执行、记忆与运行状态 | 已有 `BaseAgent`、`ToolRegistry` 和滑动窗口短期记忆；尚未串成运行闭环 |
| [`tests`](tests/) | 离线回归验证 | 已有模型流式行为、代码检查与类型配置测试；尚无任务级 Agent 评测 |

`ReactAgent`、技能加载器及工具调用示例当前仍为空文件。长期记忆、运行轨迹持久化、任务恢复、MCP 与多 Agent 均为后续计划。

## 开发与运行

使用 Python 3.13 或更新版本，通过 uv 在项目 `.venv` 环境中管理依赖与运行：

```text
uv sync --locked
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync basedpyright
```

现有 [`examples/base_dialogue.py`](examples/base_dialogue.py) 演示模型流式对话。运行前在本地环境或 `.env` 中设置 `DS_API_KEY`、`DS_BASE_URL`，并将示例模型名调整为供应商实际提供的模型：

```text
uv run --no-sync python examples/base_dialogue.py
```

API 密钥、私有配置、运行轨迹与真实业务数据不纳入版本控制。`.env` 及其环境变体、私钥与常见凭据文件已加入忽略规则；新增文件提交前仍需检查内容。

## 发展方向

1. **最小执行闭环**：模型调用 → 工具执行 → 结果回填 → 继续执行或结束，支持步数上限与基本异常处理。
2. **应用验证**：通过文件分析、研究等单 Agent 应用检验框架复用能力。
3. **Trace 与 Evaluation**：记录运行轨迹，衡量成功率、工具调用正确性、步数、token 用量与耗时。
4. **Memory 与 Reliability**：比较上下文策略，探索摘要、检索、失败恢复与长任务检查点。
5. **进阶能力**：在实际需求和评测支持下增加规划、MCP、沙箱、技能、多 Agent；Agent Learning / RL 属于后期研究方向。

框架能力按实际任务逐步增加，不以模块数量作为完成标准。

## 文档

- [模块职责与边界](docs/MODULE_BOUNDARIES.md)：`aier.ai` 与 `aier.agent` 的功能归属、工具交互、上下文和记忆边界。
- [Agent 学习与研究路线整理](docs/AGENT_LEARNING_PATH.md)：原对话的主线、阶段目标、研究方向与项目集关系。
- [Aier 技术发展路线图](docs/ROADMAP.md)：当前基线、实现优先级与各阶段验收标准。
- [模型层使用说明](src/aier/ai/README.md)：当前可用 API 与流式事件。
- [类型检查配置说明](docs/type-checking.md)：代码质量约束与配置依据。

## 模型接口兼容性

当前实现使用 OpenAI Chat Completions 格式。项目此前通过该格式测试过 DeepSeek 与 Z.ai；供应商和具体模型的兼容性需按实际接口验证。尚未实现跨 OpenAI、Anthropic、Google 原生协议的完整统一适配。

## License

MIT

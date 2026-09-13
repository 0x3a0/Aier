# 类型检查配置说明

`pyrightconfig.json` 里每一项都有实测依据，改动前请先读这一页。

## 为什么必须配置（否则报错会多出约 20 倍）

项目采用 `src` 布局，若缺少以下两项，基于类型的分析会整体失效：

| 配置 | 缺了会怎样 |
|---|---|
| `extraPaths: ["src"]` | 第一方导入无法解析。实测 `providers/openai.py` 会被当成 `openai` 包本身，连锁产生「模块不能用作类型」「`chat` 不是该模块属性」等错误 |
| `venvPath` + `venv` | 第三方依赖（pydantic 等）解析失败，`BaseModel` 退化为 `Unknown`，所有继承它的类都被标记「基类类型未知」 |

这两项修复前，实测 252 条提示中约 95% 是配置造成的假象。

## 为什么用 all 模式

`all` 是 basedpyright 最严格的模式，比 pyright 官方的 `standard` 更严，额外开启：

- `reportUnnecessaryComparison` / `reportUnnecessaryCast` / `reportUnnecessaryTypeIgnoreComment`
- `reportImplicitOverride`、`reportUnannotatedClassAttribute`
- `reportAny`、`reportExplicitAny`
- `reportUnusedCallResult` 等

实际收益举例：切到 `all` 后立刻发现 `ChoiceDeltaToolCallFunction.name` 与
`.arguments` 在 SDK 中是 `str | None`，而实现按 `str` 处理——这是窄模式下
看不到的真实缺陷。

## 唯一的例外：关闭 reportAny / reportExplicitAny

理由是本框架的设计约束 **D4：供应商调用透传 `**kwargs`**。底层 SDK 的参数
（`temperature`、`extra_body`、供应商私有字段……）无法在本框架内穷举，
`Any` 是这里的正确类型而非疏漏。

实测关闭前这两条在 `all` 模式下产生 **93 条**提示，逐条核查后确认
**全部是风格提示、无一指向缺陷**；修复其中非 Any 类的 4 条真实缺陷后，
非 Any 类提示已归零。

**注意**：`reportUnknownVariableType`、`reportUnknownParameterType`、
`reportUnknownMemberType`、`reportUnknownArgumentType`、
`reportMissingTypeArgument` **不在例外之列**——它们才是发现「变量缺少类型
标注」的门禁，由 `tests/test_ruff_config.py` 中的测试看护，禁止被关闭。

## 与 Ruff 的分工

两个工具工作在不同抽象层，不可互相替代：

| | Ruff | basedpyright |
|---|---|---|
| 本质 | Linter + Formatter | 类型检查器 |
| 分析方式 | 语法树模式匹配 | 类型推断、跨文件符号解析 |
| 需要类型信息 | 不需要 | 必需 |

Ruff 的 `ANN` 规则只检查注解**是否存在**，不检查**是否正确**；因此
「`dict` 缺少类型参数」「注解与实现不符」这类问题只有类型检查器能发现。
反之，纯控制流问题（如缺少 `return`）Ruff 也能抓。

## 已知的类型声明与运行时不符

`openai` SDK 把 `ChatCompletionChunk.id` / `.created` 声明为非可选
（`str` / `int`），但流末尾的 usage 汇总 chunk 实测这两个字段为 `None`。
代码保留 `is not None` 判断并以 `pyright: ignore[reportUnnecessaryComparison]`
标注，同时由 `tests/test_openai_provider.py` 中的
`test_usage_chunk_does_not_erase_response_identity` 锁定该行为。

**运行时的真实取值优先于类型声明。**

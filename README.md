# Aier - 一个开箱即用的 Agent Framework

> 该框架是在学习 Agent 的过程中开发的，仅对一些常用的功能进行封装，很多功能并不完善。

Aier 包括以下两个重要的模块：

1. [aier-ai](https://github.com/0x3a0/Aier/tree/main/src/aier/ai)：统一的 LLM API（屏蔽 OpenAI、Anthropic API 的结构差异），具备工具注册和简单的上下文保存功能
2. [aier-agent](https://github.com/0x3a0/Aier/tree/main/src/aier/agent)：具备状态管理、长短期记忆、工具调用等功能的 Agent 运行环境

## 已测试的大模型供应商

- OpenAI API
    - 深度求索（Deepseek）
    - 智谱（Z.ai）
- 未来会补充 Anthropic、Google 等 API 格式

## License

MIT

# Aier - 一个开箱即用的 Agent Framework
> 该框架是在学习 Agent 的过程中开发的，仅对一些常用的功能进行封装，很多功能并不完善。

主要分为两个部分：
- [aier-ai](https://github.com/0x3a0/Aier/tree/main/src/aier/ai)：LLM API(OpenAI, Anthropic, Google...) 的抽象层，屏蔽不同大模型供应商之间的结构差异
- [aier-agent](https://github.com/0x3a0/Aier/tree/main/src/aier/agent)：具备状态管理、长短期记忆、工具调用等功能的 Agent 运行环境
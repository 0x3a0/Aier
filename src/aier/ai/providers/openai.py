from typing import Union

from openai import OpenAI

from .base import LLMModel
from ..types import (
    TextContent, ThinkingContent,
    AssistantMessageEvent, StartEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    AssistantMessage, Message,
    Context, Usage
)


class OpenAIModel(LLMModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str
    ) -> None:
        self.model_name = model_name
        self.client = self._create_client(api_key, base_url)

    def _create_client(
        self,
        api_key: str,
        base_url: str
    ) -> OpenAI:
        return OpenAI(api_key=api_key, base_url=base_url)

    def _transform_messages(
        self,
        messages: list[Message]
    ) -> list[dict]:
        """ 将输入的上下文转换成标准的 OpenAI 消息格式 """
        transformed_messages = []
        for msg in messages:
            transformed_messages.append(msg.model_dump())
        return transformed_messages

    def stream_invoke(
        self,
        context: Context,
    ) -> Union[AssistantMessageEvent, AssistantMessage]:
        """ 流式输出模型调用结果 """
        llm_output: AssistantMessage = AssistantMessage(
            role="assistant",
            content=[],
            model=self.model_name,
            response_id="",
            usage=Usage(
                input=0,
                output=0,
                total_tokens=0
            ),
            finish_reason="stop",
            create_timestamp=0
        )

        messages = []
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})
        transformed_messages = self._transform_messages(context.messages)
        messages.extend(transformed_messages)

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            temperature=0.8,
            extra_body={
                "thinking": {
                    "type": "enable"
                }
            }
        )

        yield StartEvent(portion=llm_output)

        thinking_block: Optional[ThinkingContent] = None
        text_block: Optional[TextContent] = None

        for chunk in stream:
            llm_output.response_id = chunk.id
            llm_output.create_timestamp = chunk.created

            delta = chunk.choices[0].delta
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                if thinking_block is None:
                    thinking_block = ThinkingContent(thinking="")
                    llm_output.content.append(thinking_block)
                    yield ThinkingStartEvent(portion=llm_output)

                thinking_block.thinking += reasoning_content
                yield ThinkingDeltaEvent(delta=reasoning_content, portion=llm_output)
            
            if reasoning_content is None and thinking_block is not None:
                yield ThinkingEndEvent(content=thinking_block.thinking ,portion=llm_output)
                thinking_block = None

            text_content = getattr(delta, "content", None)
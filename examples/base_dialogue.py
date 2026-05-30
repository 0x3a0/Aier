from os import getenv

from openai import OpenAI

from aier.ai.providers.openai import OpenAIModel
from aier.ai.types import Message


client = OpenAI(api_key=getenv("DEEPSEEK_API_KEY"), base_url=getenv("DEEPSEEK_BASE_URL"))

chat_model = OpenAIModel(
    model_name="deepseek-v4-flash",
    temperature=0.8,
    openai_client=client
)

for chunk in chat_model.stream_invoke([
    Message(role="user", content="你好, 你是谁?")
]):
    pass

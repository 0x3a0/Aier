from os import getenv

from aier.ai import get_model, UserMessage, Context

from dotenv import load_dotenv


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
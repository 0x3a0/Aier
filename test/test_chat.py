from os import getenv

from aier.ai import get_model, UserMessage, Context

from dotenv import load_dotenv


load_dotenv()

model = get_model("openai", "glm-4.7-flash", getenv("ZP_API_KEY"), getenv("ZP_BASE_URL"))
context = Context(
    system_prompt="你的名字是0xAI",
    messages=[
        UserMessage(content="你叫什么名字")
    ]
)
for event in model.stream_invoke(context):
    print(event)
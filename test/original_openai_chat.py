from os import getenv
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(api_key=getenv("ZP_API_KEY"), base_url=getenv("ZP_BASE_URL"))
stream = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[
        {"role": "system", "content": "你的名字是0x01"},
        {"role": "user", "content": "你好"}
    ],
    stream=True,
    temperature=0.8,
    extra_body={
        "thinking": {
            "type": "enable"
        }
    }
)
for stream_chunk in stream:
    print(stream_chunk)
from ..model import BaseModel


class OpenAIModel(BaseModel):
    """ OpenAI 模型 """
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.8,
        thinking: bool = True
    ) -> None:
        self.model_name = model_name

        self.temperature = temperature
        # 默认开启思考模式，思考强度为 "high"
        if thinking:
            self.extra_body = {"thinking": {"type": "enabled"}}
        self.reasoning_effort = reasoning_effort

        # 其他参数
        self.kwargs = kwargs
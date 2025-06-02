import openai
from model import call_api


class LLMAgent:
    def __init__(self, model):
        self.model = model

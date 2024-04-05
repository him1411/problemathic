# Load packages
from openai import OpenAI as OriginalOpenAI

from langchain_openai import OpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class Model:
    def __init__(self, use_langchain) -> None:
        self.use_langchain = use_langchain

    def get_openai_chat_model(self, **kwargs):
        if self.use_langchain:
            model = ChatOpenAI(**kwargs)
            return model
        else:
            client = OriginalOpenAI(api_key=kwargs["api_key"])
            return client

    def get_openai_model(self, **kwargs):
        if self.use_langchain:
            model = OpenAI(**kwargs)
            return model
        else:
            client = OriginalOpenAI(api_key=kwargs["api_key"])
            return client
    
    def get_google_gemini_chat_model(self, **kwargs):
        model = ChatGoogleGenerativeAI(**kwargs)
        return model
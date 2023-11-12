from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from app.service.langchain.models.chat_open_ai_with_token_count import ChatOpenAIWithTokenCount

def get_langchain_model():
    llm = ChatOpenAIWithTokenCount(temperature=0, verbose=True, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    return llm
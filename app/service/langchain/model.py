from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_langchain_model():
    llm = ChatOpenAI(temperature=0, verbose=True, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    return llm
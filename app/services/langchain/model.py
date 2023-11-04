from langchain.chat_models import ChatOpenAI

def get_langchain_model():
    llm = ChatOpenAI(temperature=0, verbose=True)
    return llm
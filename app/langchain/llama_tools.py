import threading
import os
from llama_index.vector_stores import PGVectorStore
from llama_index.extractors import BaseExtractor

class ConversationIdExtractor(BaseExtractor):
    pass

# Singleton pg vector store
class LlamaIndexPgVectorStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> PGVectorStore:
        if cls._instance is None:
            with cls._lock:
                if not cls._instance:
                    cls._instance = PGVectorStore.from_params(host=os.getenv('POSTGRES_DATABASE_URL'),
                                                              port=os.getenv('POSTGRES_DATABASE_PORT'),
                                                              database=os.getenv('POSTGRES_DATABASE_NAME'),
                                                              user=os.getenv('POSTGRES_DATABASE_USERNAME'),
                                                              password=os.getenv('POSTGRES_DATABASE_PASSWORD'),
                                                              hybrid_search=True)
        
        return cls._instance
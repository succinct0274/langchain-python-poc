from fastapi import APIRouter, Depends
from typing import Annotated, List, Union, Optional
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Header
from app.service.langchain.model import get_langchain_model
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain.utilities.serpapi import SerpAPIWrapper
from app.service.langchain.agents.multi_action_agent import MultiActionAgent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.document_loaders import UnstructuredAPIFileIOLoader
import pandas as pd
import os
from langchain.vectorstores.chroma import Chroma
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.storage.in_memory import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory, ConversationKGMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.schema import Document
import uuid
from langchain.memory.chat_message_histories import PostgresChatMessageHistory
from app.database.crud.conversation_history import create_conversation_history, get_conversation_historys_by_conversation_id
from app.database.schema.conversation_history import ConversationHistoryCreate
from app.database.base import SessionLocal, get_session_local
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks, Response
import logging
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from app.service.langchain.vectorstore.pgvector import PGVectorWithMetadata
from app.service.langchain.callbacks.postgres_callback_handler import PostgresCallbackHandler
from sse_starlette import EventSourceResponse
from app.service.langchain.callbacks.queue_callback_handler import QueueCallbackHandler
from queue import Queue, Empty
from threading import Thread
from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain.agents import initialize_agent
from app.service.langchain.callbacks.agent_queue_callback_handler import AgentQueueCallbackHandler
from app.service.langchain.parsers.output.output_parser import CustomConvoOutputParser

router = APIRouter(
    prefix='/langchains',
    tags=['langchains'],
)

def random_word(query: str) -> str:
    print("\nNow I'm doing this!")
    return "foo"

logger = logging.getLogger(__name__)

supported_docuemnt_types = set(['application/pdf', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'])

def _process_pdf_files(files: List[UploadFile]) -> List[Document]:
    documents = []
    pdf_files = [file for file in files if file.content_type == 'application/pdf']
    
    for pdf in pdf_files:
        from langchain.text_splitter import CharacterTextSplitter
    
        loader = UnstructuredAPIFileIOLoader(pdf.file, url=os.getenv('UNSTRUCTURED_API_URL'), metadata_filename=pdf.filename)
        docs = loader.load_and_split(text_splitter = CharacterTextSplitter(separator="\n",
                                                                           chunk_size=800,
                                                                           chunk_overlap=100,
                                                                           length_function=len)
                                                                           )
        for doc in docs:
            if 'doc_in' in doc.metadata:
                continue
            doc.metadata['doc_id'] = str(uuid.uuid4())
        documents.extend(docs)
    
    return documents

def get_xlsx_dataframes(files: List[UploadFile]):
    xlsx_files = [file for file in files if file.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']

    if not xlsx_files:
        return []

    dataframes = [pd.read_excel(excel.file) for excel in xlsx_files]
    return dataframes

@router.post('/conversate')
async def conversate(question: Annotated[str, Form()], 
                     background_tasks: BackgroundTasks,
                     response: Response,
                     files: Annotated[List[UploadFile], File()]=[], 
                     x_conversation_id: Annotated[Union[str, None], Header()]=None,
                     llm=Depends(get_langchain_model),
                     session: Session=Depends(get_session_local)):
    content_types = set([file.content_type for file in files])
    supported = content_types.issubset(supported_docuemnt_types)
    if not supported:
        raise HTTPException(status_code=400, detail='Unsupported document type')
    
    # Generate session id for embedding and chat history store
    if x_conversation_id is None:
        x_conversation_id = str(uuid.uuid4())

    # Process files uploaded into text
    documents = _process_pdf_files(files)

    # Make vector store asynchronous
    hf_embedding = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        encode_kwargs={'normalize_embeddings': False}
    )

    db = PGVectorWithMetadata(os.getenv('SQLALCHEMY_DATABASE_URL'), 
                              embedding_function=hf_embedding, 
                              distance_strategy=DistanceStrategy.EUCLIDEAN,
                              collection_metadata={'conversation_id': x_conversation_id})
    
    if documents:
        db.add_documents(documents)
    # db = Chroma(embedding_function=hf_embedding)
    # store = InMemoryStore()
    # retriever = ParentDocumentRetriever(
    #     vectorstore=db,
    #     docstore=store,
    #     child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    #     parent_splitter=RecursiveCharacterTextSplitter(chunk_size=500),
    # )

    # doc_ids = [doc.metadata['doc_id'] for doc in documents]
    # retriever.add_documents(documents, ids=None)

    # model_id = "Writer/palmyra-small"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # pipe = pipeline(
    #     "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    # )
    # llm = HuggingFacePipeline(pipeline=pipe)

    # Instantiate the summary llm and set the max length of output to 300
    summarization_model_name = 'pszemraj/led-large-book-summary'
    summarization_llm = HuggingFacePipeline(pipeline=pipeline('summarization', summarization_model_name, max_length=300))

    memory = ConversationSummaryBufferMemory(llm=summarization_llm, memory_key='chat_history', return_messages=True, verbose=True)

    # Load conversation history from the database if corresponding header provided
    if x_conversation_id is not None:
        chat_records = get_conversation_historys_by_conversation_id(session, x_conversation_id)
        for record in chat_records:
            memory.save_context({'input': record.human_message}, {'output': record.ai_message})


    queue = Queue()
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, verbose=True, streaming=True, callbacks=[QueueCallbackHandler(queue), PostgresCallbackHandler(session, x_conversation_id)]), 
        retriever=db.as_retriever(),
        condense_question_llm=ChatOpenAI(temperature=0, verbose=True, streaming=True),
        memory=memory,
        verbose=True,
    )

    df = get_xlsx_dataframes(files)
    def pandas_agent(input=""):
        pandas_agent_df = create_pandas_dataframe_agent(llm, df, number_of_head_rows=1, verbose=True) 
        return pandas_agent_df

    pandas_tool = Tool(
        name='Pandas Data frame tool',
        func=pandas_agent,
        description="Useful for when you need to answer questions about a Pandas Dataframe"
    )

    conversational_agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=[pandas_tool],
        llm=ChatOpenAI(temperature=0, verbose=True, streaming=True, callbacks=[
            AgentQueueCallbackHandler(queue),
            # QueueCallbackHandler(queue),
            PostgresCallbackHandler(session, x_conversation_id)]),
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={
            'output_parser': CustomConvoOutputParser()
        }
    )
    
    # Return conversation id (aka session id)
    response.headers['X-Conversation-Id'] = x_conversation_id

    def output_answer_token(queue: Queue):
        job_done = object()
        def task():
            result = conversational_agent({'input': question})
            # result = qa({'question': question})
            background_tasks.add_task(create_conversation_history, session, ConversationHistoryCreate(conversation_id=x_conversation_id, human_message=question, ai_message=result['output']))
            queue.put(job_done)
        
        thread = Thread(target=task)
        thread.start()

        while True:
            try:
                item = queue.get(True, timeout=1)
                if item is job_done:
                    break
                yield item
            except Empty:
                continue

    # Save current conversation message to the database
    return EventSourceResponse(output_answer_token(queue))


@router.post('/fake')
async def fake(question: Annotated[str, Form()], files: Annotated[list[UploadFile], File()], llm=Depends(get_langchain_model)):
    search = SerpAPIWrapper()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events",
        ),
        Tool(
            name="RandomWord",
            func=random_word,
            description="call this to get a random word.",
        ),
    ]

    agent = MultiActionAgent()

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    agent_executor.run("How many people live in canada as of 2023?")


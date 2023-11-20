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
from app.database.crud.conversation_history import create_conversation_history, find_conversation_historys_by_conversation_id, exists_conversation_historys_by_conversation_id
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
import pandas as pd
from langchain.agents import initialize_agent
from app.service.langchain.callbacks.agent_queue_callback_handler import AgentQueueCallbackHandler
from app.service.langchain.parsers.output.output_parser import CustomConvoOutputParser
from langchain.agents.load_tools import _LLM_TOOLS
from app.service.langchain.agents.panda_agent import create_pandas_dataframe_agent
from app.service.langchain.models.chat_open_ai_with_token_count import ChatOpenAIWithTokenCount
from bson import Binary
from langchain.schema.runnable import RunnableBranch
from app.mongodb.crud.document import create_document, find_document_by_conversation_id_and_filenames
from app.mongodb.schema.document import DocumentCreate
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from datetime import date
import time
import base64
import mimetypes

router = APIRouter(
    prefix='/langchains',
    tags=['langchains'],
)

logger = logging.getLogger(__name__)

DEFAULT_AI_GREETING_MESSAGE = 'Hi there, how can I help you?'
SUPPORTED_DOCUMENT_TYPES = set(['application/pdf', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'])

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

def load_document_to_vector_store(files: List[UploadFile], conversation_id: str):
    documents = _process_pdf_files(files)

    if len(documents) == 0:
        return
    
    # Make vector store asynchronous
    hf_embedding = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        encode_kwargs={'normalize_embeddings': False}
    )

    db = PGVectorWithMetadata(os.getenv('SQLALCHEMY_DATABASE_URL'), 
                              embedding_function=hf_embedding, 
                              distance_strategy=DistanceStrategy.EUCLIDEAN,
                              collection_metadata={'conversation_id': conversation_id})
    
    db.add_documents(documents)

@router.get('/conversate')
async def find_conversation(x_conversation_id: Annotated[str, Header()],
                            session: Session=Depends(get_session_local)):
    return find_conversation_historys_by_conversation_id(session, x_conversation_id)

@router.post('/upload')
async def upload(files: Annotated[List[UploadFile], File()],
                 x_conversation_id: Annotated[str, Header()],
                 background_tasks: BackgroundTasks):
    content_types = set([file.content_type for file in files])
    supported = content_types.issubset(SUPPORTED_DOCUMENT_TYPES)
    if not supported:
        raise HTTPException(status_code=400, detail='Unsupported document type')
    
    existed = await find_document_by_conversation_id_and_filenames(x_conversation_id, [f.filename for f in files])
    existed_filenames = set([persisted['filename'] for persisted in existed])
    docs_for_vector_store = []
    for file in files:
        if file.filename in existed_filenames:
            continue

        docs_for_vector_store.append(file)
        entity = DocumentCreate(content=Binary(file.file.read()), filename=file.filename, mime_type=file.content_type, conversation_id=x_conversation_id)
        file.file.seek(0)
        await create_document(entity)

    load_document_to_vector_store(docs_for_vector_store, x_conversation_id)
    # background_tasks.add_task(load_document_to_vector_store, docs_for_vector_store, x_conversation_id)

@router.post('/conversate')
async def conversate(question: Annotated[str, Form()], 
                     background_tasks: BackgroundTasks,
                     response: Response,
                     files: Annotated[List[UploadFile], File()]=[], 
                     x_conversation_id: Annotated[Union[str, None], Header()]=None,
                     llm=Depends(get_langchain_model),
                     session: Session=Depends(get_session_local)):
    content_types = set([file.content_type for file in files])
    supported = content_types.issubset(SUPPORTED_DOCUMENT_TYPES)
    if not supported:
        raise HTTPException(status_code=400, detail='Unsupported document type')
    
    # Generate session id for embedding and chat history store
    if x_conversation_id is None:
        x_conversation_id = str(uuid.uuid4())

    print(f'Question: {question}')

    # Process files uploaded into text
    file_detail = []
    for file in files:
        file_detail.append({'filename': file.filename, 'mime_type': file.content_type})
    await upload(files, x_conversation_id, background_tasks)

    # Make vector store asynchronous
    hf_embedding = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        encode_kwargs={'normalize_embeddings': False}
    )

    db = PGVectorWithMetadata(os.getenv('SQLALCHEMY_DATABASE_URL'), 
                              embedding_function=hf_embedding, 
                              distance_strategy=DistanceStrategy.EUCLIDEAN,
                              collection_metadata={'conversation_id': x_conversation_id})

    # Instantiate the summary llm and set the max length of output to 300
    summarization_model_name = 'pszemraj/led-large-book-summary'
    summarization_llm = HuggingFacePipeline(pipeline=pipeline('summarization', summarization_model_name, max_length=300))

    memory = ConversationSummaryBufferMemory(llm=summarization_llm, memory_key='chat_history', return_messages=True, verbose=True)

    # Load conversation history from the database if corresponding header provided
    if x_conversation_id is not None:
        chat_records = find_conversation_historys_by_conversation_id(session, x_conversation_id)
        for record in chat_records:
            memory.save_context({'input': record.human_message}, {'output': record.ai_message})


    queue = Queue()
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAIWithTokenCount(temperature=0, verbose=True, streaming=True, callbacks=[QueueCallbackHandler(queue), PostgresCallbackHandler(session, x_conversation_id)]), 
        retriever=db.as_retriever(),
        condense_question_llm=ChatOpenAIWithTokenCount(temperature=0, verbose=True, streaming=True),
        memory=memory,
        verbose=True,
    )

    df = get_xlsx_dataframes(files)

    chain = (
        PromptTemplate.from_template(
            """Given the user question below, classify it as either being about `DataFrame` or `RetrievalQA`.
                                        
            Do not respond with more than one word.

            <question>
            {question}
            </question>

            Classification:"""
        )
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )

    timestamp = str(int(time.time()))
    exported_chart_path = f'export/chart/{date.today()}/{timestamp}/{x_conversation_id}.png'
    def run_with_panda_agent(question: str):
        question += f"\n If you have plotted a chart, you don't need to show the chart but you have to save it as {exported_chart_path}. Also, you should create the directory if not existed."
        panda_agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, verbose=True), 
                                                        df[0] if len(df) == 1 else df,
                                                        verbose=True, 
                                                        agent_executor_kwargs={'handle_parsing_errors': True},
                                                        agent_type=AgentType.OPENAI_FUNCTIONS)
        
        return panda_agent.run({'input': question})

    branch = RunnableBranch(
        (lambda x: "dataframe" in x["topic"].lower(), lambda x: run_with_panda_agent(x['question'])),
        lambda x: qa(x['question'])['result']
    )

    full_chain = {"topic": chain, "question": lambda x: x["question"]} | branch
    answer = full_chain.invoke({'question': question})
    
    # Return conversation id (aka session id)
    response.headers['X-Conversation-Id'] = x_conversation_id

    res = { 'text': answer }

    # Convert the image into base64 as part of the response
    output_media = []
    if os.path.exists(exported_chart_path):
        with open(exported_chart_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            mime_type, _ = mimetypes.guess_type(exported_chart_path)
            output_media.append({
                'content': encoded_string,
                'content_type': mime_type
            })
            res['image'] = encoded_string

    # Save current conversation message to the database
    background_tasks.add_task(create_conversation_history, session, ConversationHistoryCreate(conversation_id=x_conversation_id, 
                                                                                              human_message=question, 
                                                                                              ai_message=answer, 
                                                                                              uploaded_file_detail=file_detail,
                                                                                              responded_media=output_media))
    return res

@router.post('/session/init')
async def init_session(session: Session=Depends(get_session_local)):
    sid = uuid.uuid4()

    while exists_conversation_historys_by_conversation_id(session, sid):
        sid = uuid.uuid4()

    create_conversation_history(session, ConversationHistoryCreate(conversation_id=sid,
                                                                   human_message='Hi',
                                                                   ai_message=DEFAULT_AI_GREETING_MESSAGE,
                                                                   greeting=True))
    
    return {
        'conversation_id': sid,
        'ai_message': DEFAULT_AI_GREETING_MESSAGE,
    }

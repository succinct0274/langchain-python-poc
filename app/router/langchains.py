from fastapi import APIRouter, Depends
from typing import Annotated, List, Union, Optional
from uuid import UUID
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Header, Path
from app.service.langchain.model import get_langchain_model
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.document_loaders import UnstructuredAPIFileIOLoader
import pandas as pd
import os
from langchain.embeddings import HuggingFaceEmbeddings
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
from app.mongodb.crud.document import create_document, acreate_document, find_document_by_conversation_id_and_filenames, find_document_by_conversation_id
from app.mongodb.schema.document import DocumentCreate
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from datetime import date
from langchain.memory import ConversationBufferMemory
import time
import base64
import mimetypes
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
import re
from langchain.document_loaders import PyPDFLoader
from io import BytesIO
import tempfile


router = APIRouter(
    prefix='/langchains',
    tags=['langchains'],
)

logger = logging.getLogger(__name__)

UUID_PATTERN = re.compile(r'^[\da-f]{8}-([\da-f]{4}-){3}[\da-f]{12}$', re.IGNORECASE)
DEFAULT_AI_GREETING_MESSAGE = 'Hi there, how can I help you?'
SUPPORTED_DOCUMENT_TYPES = set(['application/pdf', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'])

def _process_pdf_files(files: List[UploadFile], conversation_id: str) -> List[Document]:
    documents = []
    pdf_files = [file for file in files if file.content_type == 'application/pdf']
    
    for pdf in pdf_files:
        from langchain.text_splitter import CharacterTextSplitter

        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(pdf.file.read())
                pdf.file.seek(0)
                loader = PyPDFLoader(path)
                docs = loader.load_and_split(CharacterTextSplitter(separator="\n",
                                                                   chunk_size=800,
                                                                   chunk_overlap=100,
                                                                   length_function=len))
        finally:
            os.remove(path)

        # loader = UnstructuredAPIFileIOLoader(pdf.file, url=os.getenv('UNSTRUCTURED_API_URL'), metadata_filename=pdf.filename)
        # docs = loader.load_and_split(text_splitter = CharacterTextSplitter(separator="\n",
        #                                                                    chunk_size=800,
        #                                                                    chunk_overlap=100,
        #                                                                    length_function=len)
        #                                                                    )
        for doc in docs:
            if 'doc_in' in doc.metadata:
                continue
            doc.metadata['doc_id'] = str(uuid.uuid4())
            doc.metadata['conversation_id'] = conversation_id
        documents.extend(docs)
    
    return documents

def get_xlsx_dataframes(files: List[UploadFile]):
    xlsx_files = [file for file in files if file.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']

    if not xlsx_files:
        return []

    dataframes = [pd.read_excel(excel.file) for excel in xlsx_files]
    return dataframes

def load_document_to_vector_store(files: List[UploadFile], conversation_id: str):
    documents = _process_pdf_files(files, conversation_id)

    if len(documents) == 0:
        return
    
    # Make vector store asynchronous
    # hf_embedding = HuggingFaceEmbeddings(
    #     model_name='sentence-transformers/all-MiniLM-L6-v2',
    #     encode_kwargs={'normalize_embeddings': False}
    # )

    db = PGVector(os.getenv('SQLALCHEMY_DATABASE_URL'), 
                              embedding_function=OpenAIEmbeddings(), 
                              distance_strategy=DistanceStrategy.EUCLIDEAN,
                              collection_metadata={'conversation_id': conversation_id})
    
    db.add_documents(documents)

@router.get('/conversate')
async def find_conversation(x_conversation_id: Annotated[str, Header()],
                            session: Session=Depends(get_session_local)):
    return find_conversation_historys_by_conversation_id(session, x_conversation_id)

@router.post('/shared/upload')
def general_upload(files: Annotated[List[UploadFile], File()],
                   response: Response):
    shared_conversation_id = os.getenv('SHARED_KNOWLEDGE_BASE_UUID')
    upload(files, response, shared_conversation_id)

@router.get('/shared/files')
def find_shared_files():
    return find_files_by_conversation_id(os.getenv('SHARED_KNOWLEDGE_BASE_UUID'))

@router.get('/{conversation_id}/files')
def find_files_by_conversation_id(conversation_id: Annotated[UUID, Path(title="The conversation id for session")]):
    files = find_document_by_conversation_id(conversation_id)
    res = []
    for file in files:
        res.append({
            'filename': file.filename,
            'upload_date': int(datetime.timestamp(file.upload_date)) * 1000,
            'content_type': file.metadata['mime_type']
        })

    return res

@router.post('/upload')
def upload(files: Annotated[List[UploadFile], File()],
           response: Response,
           x_conversation_id: Annotated[str, Header()] = None):
    if x_conversation_id is None:
        x_conversation_id = str(uuid.uuid4())

    if x_conversation_id is not None and not bool(UUID_PATTERN.match(x_conversation_id)):
        raise HTTPException(status_code=422, detail="Invalid session id")

    content_types = set([file.content_type for file in files])
    supported = content_types.issubset(SUPPORTED_DOCUMENT_TYPES)
    if not supported:
        raise HTTPException(status_code=400, detail='Unsupported document type')
    
    existed = find_document_by_conversation_id_and_filenames(x_conversation_id, [f.filename for f in files])
    existed_filenames = set([persisted.filename for persisted in existed])
    docs_for_vector_store = []
    for file in files:
        if file.filename in existed_filenames:
            continue

        docs_for_vector_store.append(file)
        entity = DocumentCreate(content=Binary(file.file.read()), filename=file.filename, mime_type=file.content_type, conversation_id=x_conversation_id)
        file.file.seek(0)
        create_document(entity)

    load_document_to_vector_store(docs_for_vector_store, x_conversation_id)

    response.headers['X-Conversation-Id'] = x_conversation_id
    # background_tasks.add_task(load_document_to_vector_store, docs_for_vector_store, x_conversation_id)

@router.post('/conversate')
def conversate(question: Annotated[str, Form()], 
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

    upload(files, response, x_conversation_id)

    # Make vector store asynchronous
    # hf_embedding = HuggingFaceEmbeddings(
    #     model_name='sentence-transformers/all-MiniLM-L6-v2',
    #     encode_kwargs={'normalize_embeddings': False}
    # )

    db = PGVector(os.getenv('SQLALCHEMY_DATABASE_URL'), 
                              embedding_function=OpenAIEmbeddings(), 
                              distance_strategy=DistanceStrategy.EUCLIDEAN,
                              collection_metadata={'conversation_id': x_conversation_id})

    # Instantiate the summary llm and set the max length of output to 300
    # summarization_model_name = 'pszemraj/led-large-book-summary'
    # summarization_llm = HuggingFacePipeline(pipeline=pipeline('summarization', summarization_model_name, max_length=300))

    memory = ConversationSummaryBufferMemory(llm=llm, memory_key='chat_history', return_messages=True, verbose=True, output_key='answer')

    # Load conversation history from the database if corresponding header provided
    if x_conversation_id is not None:
        chat_records = find_conversation_historys_by_conversation_id(session, x_conversation_id)
        for record in chat_records:
            memory.save_context({'input': record.human_message}, {'output': record.ai_message})

    queue = Queue()
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAIWithTokenCount(temperature=0, verbose=True, streaming=True, callbacks=[QueueCallbackHandler(queue), PostgresCallbackHandler(session, x_conversation_id)]), 
        retriever=db.as_retriever(search_kwargs={
            'filter': { 'conversation_id': {"in": [os.getenv('SHARED_KNOWLEDGE_BASE_UUID'), x_conversation_id]} }
        }),
        condense_question_llm=ChatOpenAIWithTokenCount(temperature=0, verbose=True, streaming=True),
        memory=memory,
        verbose=True,
        return_source_documents=True,
    )

    df = get_xlsx_dataframes(files)

    chain = (
        PromptTemplate.from_template(
            """Given the user question below, classify it as either being about `Dataframe` or `RetrievalQA`.

            Choose `DataFrame` if the question is related to excel, csv or dataframe.
            Choose `RetrievalQA` if the question is about general stuff or relevant to document or pdf files.
                                        
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
    image_name = f'{x_conversation_id}.png'
    exported_chart_path = f'/export/chart/{date.today()}/{timestamp}'
    image_path = f'{exported_chart_path}/{image_name}'
    def run_with_panda_agent(question: str):
        question += f"""
        If you have plotted a chart, you must not show the chart but you must save it as {image_path}. 
        Create a directory with the path {exported_chart_path} before you save your image.
        Also, never include the path to image into your answer.
        """
        panda_agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, verbose=True), 
                                                        df[0] if len(df) == 1 else df,
                                                        verbose=True, 
                                                        agent_executor_kwargs={'handle_parsing_errors': True},
                                                        # return_intermediate_steps=True,
                                                        return_direct=True,
                                                        agent_type=AgentType.OPENAI_FUNCTIONS,
                                                        memory=ConversationBufferMemory())
        
        # Manually create a directory first
        Path(exported_chart_path).mkdir(parents=True, exist_ok=True)

        answer = panda_agent({'input': question})['output']

        # Add one more chain to rephrase answer
        prompt = ChatPromptTemplate.from_template("Rephrase the following output so that python code and image name does not appear in your response: {output}")
        chain = prompt | ChatOpenAIWithTokenCount(temperature=0, verbose=True)
        output = chain.invoke({"output": answer})
        return output.content
    branch = RunnableBranch(
        (lambda x: "dataframe" in x["topic"].lower(), lambda x: run_with_panda_agent(x['question'])),
        lambda x: qa(x['question'])['answer']
    )

    full_chain = {"topic": chain, "question": lambda x: x["question"]} | branch
    answer = full_chain.invoke({'question': question})

    print(answer)
    
    # Return conversation id (aka session id)
    response.headers['X-Conversation-Id'] = x_conversation_id

    res = { 'text': answer }

    # Convert the image into base64 as part of the response
    output_media = []
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            mime_type, _ = mimetypes.guess_type(image_path)
            output_media.append({
                'content': encoded_string,
                'content_type': mime_type
            })
            res['image'] = output_media

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

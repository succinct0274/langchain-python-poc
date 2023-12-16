from fastapi import APIRouter, Depends
from typing import Annotated, List, Union, Optional, Dict
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
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from app.mongodb.crud.document import update_document_status_by_ids, create_document, acreate_document, find_document_by_conversation_id_and_filenames, find_document_by_conversation_id
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
from langchain.document_loaders import PyMuPDFLoader
from io import BytesIO
import io
import tempfile

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
                binary_bytes = pdf.file.read()
                tmp.write(binary_bytes)
                tmp.flush()
                pdf.file.seek(0)
                loader = PyMuPDFLoader(path)
                # Attempt to parse file into docs without ocr
                docs = loader.load_and_split(CharacterTextSplitter(separator="\n",
                                                                   chunk_size=800,
                                                                   chunk_overlap=100,
                                                                   length_function=len))
                
                if len(docs) == 0:
                    loader = PyMuPDFLoader(path, extract_images=True)
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
            doc.metadata['conversation_id'] = conversation_id
            doc.metadata['filename'] = pdf.filename
            if 'doc_in' in doc.metadata:
                continue
            doc.metadata['doc_id'] = str(uuid.uuid4())
        documents.extend(docs)
    
    return documents

def initiate_conversation(db_session: Session):
    sid = uuid.uuid4()

    while exists_conversation_historys_by_conversation_id(db_session, sid):
        sid = uuid.uuid4()

    create_conversation_history(db_session, ConversationHistoryCreate(conversation_id=sid,
                                                                   human_message='Hi',
                                                                   ai_message=DEFAULT_AI_GREETING_MESSAGE,
                                                                   greeting=True))
    
    return {
        'conversation_id': sid,
        'ai_message': DEFAULT_AI_GREETING_MESSAGE,
    }    

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

def find_conversation_historys(db_session: Session, conversation_id: str):
    return find_conversation_historys_by_conversation_id(db_session, conversation_id)

def find_document(conversation_id: str, filenames: List[str]):
    return find_document_by_conversation_id_and_filenames(conversation_id, filenames)

def upload(files: List[UploadFile], conversation_id: str = None):
    filenames = [f.filename for f in files]
    existed = find_document(conversation_id, filenames)
    existed_filenames = set([persisted.filename for persisted in existed])
    res: List[Dict] = []
    for file in files:
        if file.filename in existed_filenames:
            continue

        entity = DocumentCreate(content=Binary(file.file.read()), filename=file.filename, mime_type=file.content_type, conversation_id=conversation_id)
        file.file.seek(0)
        grid_out = create_document(entity)
        res.append({
            'file_id': grid_out._id,
            'filename': file.filename,
            'file': file,
        })

    return res
    
def upload_and_load(files: List[UploadFile], conversation_id: str = None):
    
    uploaded_files = upload(files, conversation_id)
    load_document_to_vector_store([uploaded.file for uploaded in uploaded_files], conversation_id)

def conversate_with_llm(db_session: Session, 
                        question: str,
                        files: List[UploadFile],
                        metadata: List[Dict[str, any]],
                        conversation_id: str,
                        llm: ChatOpenAI,
                        background_tasks: BackgroundTasks | None,
                        instruction: str | None = None):
    logger.info(f"Question: {question}")
    

    if metadata is None:
        # Process files uploaded into text
        file_detail = []
        for file in files:
            file_detail.append({'filename': file.filename, 'mime_type': file.content_type})
        
        upload_and_load(files, conversation_id)
    else:
        # Changed status of all uploaded files in metadata
        update_document_status_by_ids([m['file_id'] for m in metadata], conversation_id)

    db = PGVector(os.getenv('SQLALCHEMY_DATABASE_URL'), 
                            embedding_function=OpenAIEmbeddings(), 
                            distance_strategy=DistanceStrategy.EUCLIDEAN,
                            collection_name=conversation_id,
                            collection_metadata={'conversation_id': conversation_id})

    memory = ConversationSummaryBufferMemory(llm=llm, memory_key='chat_history', return_messages=True, verbose=True, output_key='answer')

    # Load conversation history from the database if corresponding header provided
    if conversation_id is not None:
        chat_records = find_conversation_historys(db_session, conversation_id)
        for record in chat_records:
            memory.save_context({'input': record.human_message}, {'answer': record.ai_message})

    queue = Queue()

    additional_args = {}
    if instruction is not None:
        # https://stackoverflow.com/questions/76175046/how-to-add-prompt-to-langchain-conversationalretrievalchain-chat-over-docs-with
        messages = [
            SystemMessagePromptTemplate.from_template(instruction),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        additional_args['prompt'] = ChatPromptTemplate.from_messages(messages)

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAIWithTokenCount(temperature=0, verbose=True, streaming=True, callbacks=[QueueCallbackHandler(queue), PostgresCallbackHandler(db_session, conversation_id)]), 
        retriever=db.as_retriever(search_kwargs={
            'filter': { 'conversation_id': {"in": [os.getenv('SHARED_KNOWLEDGE_BASE_UUID'), conversation_id]} }
        }),
        condense_question_llm=ChatOpenAIWithTokenCount(temperature=0, verbose=True, streaming=True),
        memory=memory,
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs=additional_args
    )

    df = get_xlsx_dataframes(files)

    chain = (
        PromptTemplate.from_template(
            """Given the user question below, classify it as either `DataFrame` or `RetrievalQA`.

            - Choose `DataFrame` if the question involves:
            - Excel files, spreadsheets, or Excel-related operations.
            - CSV files, including reading, writing, or manipulating CSV data.
            - DataFrames in programming languages like Python (e.g., pandas DataFrame), including operations on these structures.

            - Choose `RetrievalQA` if the question is about:
            - General knowledge or information queries.
            - Documents or PDF files, including reading, processing, or extracting information from them.
            - Specific non-programming document-related queries.

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
    image_name = f'{conversation_id}.png'
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

    logger.info(f"Answer : {answer}")

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

    if background_tasks is not None:
        # Save current conversation message to the database
        background_tasks.add_task(create_conversation_history, db_session, ConversationHistoryCreate(conversation_id=conversation_id, 
                                                                                                     human_message=question, 
                                                                                                     ai_message=answer, 
                                                                                                     uploaded_file_detail=file_detail,
                                                                                                     responded_media=output_media))

    return res

def handle_websocket_request(body: Dict[str, any], 
                             llm: ChatOpenAIWithTokenCount,
                             background_tasks: BackgroundTasks,
                             x_conversation_id: Annotated[str, Header()],
                             db_session: Session=Depends(get_session_local)):
    question = body['question']
    files: List[UploadFile] = []
    if 'files' in body:
        file_payloads = body['files']
        # Convert file payload into UploadFile starlette objects
        for payload in file_payloads:
            base64_str: str = payload['base64']
            file_bytes: bytes = base64_str.encode('utf-8')
            file_bytes = base64.b64decode(file_bytes)
            content_type, _ = mimetypes.guess_type(payload['filename'])
            upload_file = UploadFile(io.BytesIO(file_bytes), size=len(file_bytes), filename=payload['filename'], headers={
                'content-type': content_type
            })
            files.append(upload_file)
            
    result = conversate_with_llm(db_session, question, files, x_conversation_id, llm, background_tasks)
    return result
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
from langchain.document_loaders import PyMuPDFLoader
from io import BytesIO
import tempfile
from app.langchain import service as langchain_service

router = APIRouter(
    prefix='/langchain',
    tags=['langchain'],
)

logger = logging.getLogger(__name__)

SHARED_CONVERSATION_ID = os.getenv('SHARED_KNOWLEDGE_BASE_UUID')
UUID_PATTERN = re.compile(r'^[\da-f]{8}-([\da-f]{4}-){3}[\da-f]{12}$', re.IGNORECASE)
DEFAULT_AI_GREETING_MESSAGE = 'Hi there, how can I help you?'
SUPPORTED_DOCUMENT_TYPES = set(['application/pdf', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'])

@router.get('/conversate')
async def find_conversation(x_conversation_id: Annotated[str, Header()],
                            session: Session=Depends(get_session_local)):
    return langchain_service.find_conversation_historys(session, x_conversation_id)

@router.post('/shared/upload')
def upload_shared_files(files: Annotated[List[UploadFile], File()],
                        response: Response):
    upload(files, response, SHARED_CONVERSATION_ID)

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
    
    # Upload files to mongodb
    langchain_service.upload_files(files, x_conversation_id)

    response.headers['X-Conversation-Id'] = x_conversation_id

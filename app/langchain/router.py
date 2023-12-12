from fastapi import APIRouter, Depends
from typing import Annotated, List, Union, Optional
from uuid import UUID
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Header, Path
from app.service.langchain.model import get_langchain_model
import pandas as pd
import os
import uuid
from app.database.base import SessionLocal, get_session_local
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks, Response
import logging
from pathlib import Path
import re
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

    result = langchain_service.conversate_with_llm(session, question, files, x_conversation_id, llm, background_tasks)
    return result
from fastapi import APIRouter, Depends, WebSocketDisconnect
from typing import Annotated, List, Union, Optional, Dict
from uuid import UUID
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Header, Path, WebSocket
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
import asyncio
from app.langchain import service as langchain_service
from app.service.langchain.models.chat_open_ai_with_token_count import ChatOpenAIWithTokenCount
from starlette.concurrency import run_in_threadpool
import datetime
from pydantic import BaseModel

router = APIRouter(
    prefix='/langchain',
    tags=['langchain'],
)

class ConversationRequest(BaseModel):
    question: str
    instruction: str | None = None
    metadata: Dict | None = None

logger = logging.getLogger(__name__)

SHARED_CONVERSATION_ID = os.getenv('SHARED_KNOWLEDGE_BASE_UUID')
UUID_PATTERN = re.compile(r'^[\da-f]{8}-([\da-f]{4}-){3}[\da-f]{12}$', re.IGNORECASE)
DEFAULT_AI_GREETING_MESSAGE = 'Hi there, how can I help you?'
SUPPORTED_DOCUMENT_TYPES = set(['application/pdf', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'])

@router.post('/session/init')
async def init_session(session: Session=Depends(get_session_local)):
    return langchain_service.initiate_conversation(session)

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
    langchain_service.upload_and_load(files, x_conversation_id)

    response.headers['X-Conversation-Id'] = x_conversation_id

@router.get('/{conversation_id}/files')
def find_files_by_conversation_id(conversation_id: Annotated[UUID, Path(title="The conversation id for session")]):
    files = langchain_service.find_document_by_conversation_id(conversation_id)
    res = []
    for file in files:
        res.append({
            'filename': file.filename,
            'upload_date': int(datetime.timestamp(file.upload_date)) * 1000,
            'content_type': file.metadata['mime_type']
        })

    return res

@router.post('/{conversation_id}/files')
def upload_files_by_conversation_id(conversation_id: Annotated[UUID, Path(title="The conversation id for session")],
                                    files: Annotated[List[UploadFile], File()],
                                    llm=Depends(get_langchain_model),
                                    session: Session=Depends(get_session_local)):
    content_types = set([file.content_type for file in files])
    supported = content_types.issubset(SUPPORTED_DOCUMENT_TYPES)
    if not supported:
        raise HTTPException(status_code=400, detail='Unsupported document type')
    
    # Upload files to mongodb
    uploaded = langchain_service.upload(files, str(conversation_id))
    res = []
    for record in uploaded:
        res.append({
            'file_id': record['file_id'],
            'filename': record['filename'],
            'status': 'uploaded'
        })
    return res

@router.post('/conversate')
def conversate(background_tasks: BackgroundTasks,
               response: Response,
               form_data: ConversationRequest, 
               x_conversation_id: Annotated[Union[str, None], Header()]=None,
               llm=Depends(get_langchain_model),
               session: Session=Depends(get_session_local)):
    instruction = form_data.instruction
    question = form_data.question
    metadata = form_data.metadata

    if 'attachment' in metadata:
        content_types = set([file['content_type'] for file in metadata['attachment']])
        supported = content_types.issubset(SUPPORTED_DOCUMENT_TYPES)
        if not supported:
                raise HTTPException(status_code=400, detail='Unsupported document type')
    
    # Generate session id for embedding and chat history store
    if x_conversation_id is None:
        x_conversation_id = str(uuid.uuid4())

    result = langchain_service.conversate_with_llm(session, question, [], metadata, x_conversation_id, llm, background_tasks, instruction=instruction)
    return result

@router.websocket('/ws')
async def ws_conversate(websocket: WebSocket, 
                        background_tasks: BackgroundTasks,
                        x_conversation_id: Annotated[Union[str, None], Header()]=None,
                        db_session: Session = Depends(get_session_local), 
                        llm: ChatOpenAIWithTokenCount=Depends(get_langchain_model)):
    await websocket.accept()

    try:
        while True:
            body = await websocket.receive_json()
            result = await run_in_threadpool(langchain_service.handle_websocket_request, body, llm, background_tasks, x_conversation_id, db_session)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info('Client disconnected')

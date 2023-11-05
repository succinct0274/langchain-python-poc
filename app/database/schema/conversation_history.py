from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from app.database.base import SessionLocal
from sqlalchemy import Column, Integer, String, TIMESTAMP
from uuid import uuid4, UUID

class ConversationHistoryBase(BaseModel):
    conversation_id: UUID | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None
    deleted_at: datetime | None = None
    deleted_by: str | None = None
    version: int = 1

class ConversationHistoryCreate(ConversationHistoryBase):
    id: int | None = None
    human_message: str
    ai_message: str
    existing_summary: str | None = None

    class Config:
        from_attributes = True
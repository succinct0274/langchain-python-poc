from sqlalchemy.orm import Session
from app.database.model.conversation_history import ConversationHistory
from app.database.schema.conversation_history import ConversationHistoryCreate
from uuid import UUID
from sqlalchemy import select, literal, exists, and_
from typing import List
import json

def create_conversation_history(db: Session, conversation_history: ConversationHistoryCreate):
    entity = ConversationHistory(**json.loads(conversation_history.json()))
    db.add(entity)
    db.commit()
    db.refresh(entity)
    return entity

def find_conversation_historys_by_conversation_id(db: Session, conversation_id: UUID) -> List[ConversationHistory]:
    return db.execute(select(ConversationHistory)
                      .filter(and_(ConversationHistory.conversation_id == conversation_id, ConversationHistory.greeting == False))
                      .order_by(ConversationHistory.created_at.asc())).scalars().all()

def exists_conversation_historys_by_conversation_id(db: Session, conversation_id: UUID) -> bool:
    subquery = select(ConversationHistory).where(ConversationHistory.conversation_id == conversation_id).limit(1)
    return db.execute(exists(subquery).select()).scalar()
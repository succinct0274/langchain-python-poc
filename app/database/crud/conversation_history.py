from sqlalchemy.orm import Session
from app.database.model.conversation_history import ConversationHistory
from app.database.schema.conversation_history import ConversationHistoryCreate
from uuid import UUID
from sqlalchemy import select
from typing import List

def create_conversation_history(db: Session, conversation_history: ConversationHistoryCreate):
    entity = ConversationHistory(**conversation_history.model_dump())
    db.add(entity)
    db.commit()
    db.refresh(entity)
    return entity

def get_conversation_historys_by_conversation_id(db: Session, conversation_id: UUID) -> List[ConversationHistory]:
    return db.execute(select(ConversationHistory)
                      .filter_by(conversation_id=conversation_id)
                      .order_by(ConversationHistory.created_at.asc())).scalars().all()
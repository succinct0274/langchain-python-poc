from sqlalchemy.orm import Session
from app.database.model.conversation_history import ConversationHistory
from app.database.schema.conversation_history import ConversationHistoryCreate

def create_conversation_history(db: Session, conversation_history: ConversationHistoryCreate):
    entity = ConversationHistory(**conversation_history.dict())
    db.add(entity)
    db.commit()
    db.refresh(entity)
    return entity
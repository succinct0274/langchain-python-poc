from app.database.base import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID

class ConversationHistory(Base):

    __tablename__ = 'conversation_history'

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), server_default='gen_random_uuid()')
    human_message = Column(String, nullable=False)
    ai_message = Column(String, nullable=False)
    existing_summary = Column(String)
    created_at = Column(TIMESTAMP, server_default='now()')
    created_by = Column(String(255))
    updated_at = Column(TIMESTAMP, server_default='now()')
    updated_by = Column(String(255))
    deleted_at = Column(TIMESTAMP)
    deleted_by = Column(String(255))
    version = Column(Integer)

    __mapper_args__ = { 'version_id_col': version }
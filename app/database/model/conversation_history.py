from app.database.base import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP, text, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB, TEXT
from sqlalchemy_json import mutable_json_type

class ConversationHistory(Base):

    __tablename__ = 'conversation_history'

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), server_default=text('gen_random_uuid()'))
    human_message = Column(String, nullable=False)
    ai_message = Column(String, nullable=False)
    uploaded_file_detail = Column(mutable_json_type(dbtype=JSONB, nested=True))
    responded_media = Column(mutable_json_type(dbtype=JSONB, nested=True), nullable=True)
    greeting = Column(Boolean, nullable=False, default=True)
    created_at = Column(TIMESTAMP, server_default=text('now()'))
    created_by = Column(String(255))
    updated_at = Column(TIMESTAMP, server_default=text('now()'))
    updated_by = Column(String(255))
    deleted_at = Column(TIMESTAMP)
    deleted_by = Column(String(255))
    version = Column(Integer)

    __mapper_args__ = { 'version_id_col': version }
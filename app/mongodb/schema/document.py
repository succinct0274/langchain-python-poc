from pydantic import BaseModel
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4, UUID

class DocumentSchema(BaseModel):
    filename: str = Field()
    mime_type: str = Field()
    content: bytes = Field()
    conversation_id: UUID = Field()
    created_at: datetime = datetime.now()
    created_by: str | None = None
    updated_at: datetime = datetime.now()
    updated_by: str | None = None


class DocumentCreate(BaseModel):
    filename: str
    mime_type: str
    conversation_id: str
    content: bytes
    created_at: datetime = datetime.now()
    created_by: str | None = None
    updated_at: datetime = datetime.now()
    updated_by: str | None = None
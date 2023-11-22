from app.mongodb.base import db, client
from pymongo.results import InsertOneResult
from app.mongodb.schema.document import DocumentCreate, DocumentSchema
from pymongo.collection import Collection
from typing import List
from pymongo.cursor import Cursor
from motor.motor_asyncio import AsyncIOMotorCursor, AsyncIOMotorGridFSBucket

document_collection: Collection = db.get_collection('user_document_collection')

async def create_document(doc: DocumentCreate) -> dict:
    gfs = AsyncIOMotorGridFSBucket(db)

    persisted = await gfs.upload_from_stream(filename=doc.filename, 
                                             source=doc.content, 
                                             metadata={'mime_type': doc.mime_type, 'conversation_id': doc.conversation_id})
    print(persisted)
    cursor = gfs.find({"_id": persisted}).limit(1)
    res = await cursor.to_list(None)
    return None if not res else res[0]

async def find_document_by_conversation_id(conversation_id: str):
    gfs = AsyncIOMotorGridFSBucket(db)
    cursor = gfs.find({'metadata.conversation_id': conversation_id})
    return await cursor.to_list(None)

async def exist_by_conversation_id_and_filename(conversation_id: str, filename: str):
    gfs = AsyncIOMotorGridFSBucket(db)
    cursor = gfs.find({'metadata.conversation_id': conversation_id, 'filename': filename})
    return await cursor.to_list(None) is not None

async def find_document_by_conversation_id_and_filenames(conversation_id: str, filenames: List[str]) -> List[dict]:
    gfs = AsyncIOMotorGridFSBucket(db)
    cursor = gfs.find({"metadata.conversation_id": conversation_id, "filename": {'$in': filenames}})
    return await cursor.to_list(None)
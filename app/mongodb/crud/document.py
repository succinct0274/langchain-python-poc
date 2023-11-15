from app.mongodb.base import db
from pymongo.results import InsertOneResult
from app.mongodb.schema.document import DocumentCreate, DocumentSchema
from pymongo.collection import Collection
from typing import List
from pymongo.cursor import Cursor
from motor.motor_asyncio import AsyncIOMotorCursor

document_collection: Collection = db.get_collection('user_document_collection')

async def create_document(doc: DocumentCreate) -> dict:
    persisted: InsertOneResult = await document_collection.insert_one(doc.__dict__)
    entity = document_collection.find_one({'_id': persisted.inserted_id})
    return entity

async def find_document_by_conversation_id(conversation_id: str):
    cursor: AsyncIOMotorCursor = document_collection.find({'conversation_id': conversation_id})
    return await cursor.to_list(None)

async def exist_by_conversation_id_and_filename(conversation_id: str, filename: str):
    return document_collection.find_one({'conversation_id': conversation_id, 'filename': filename}) is not None

async def find_document_by_conversation_id_and_filenames(conversation_id: str, filenames: List[str]) -> List[dict]:
    cursor: AsyncIOMotorCursor = document_collection.find({'conversation_id': conversation_id, 'filename': { '$in': filenames}})
    return await cursor.to_list(None)
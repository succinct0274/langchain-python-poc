from app.mongodb.base import async_db, async_client, fs, db
from pymongo.results import InsertOneResult
from app.mongodb.schema.document import DocumentCreate, DocumentSchema
from pymongo.collection import Collection
from typing import List
from pymongo.cursor import Cursor
from pymongo import MongoClient
from gridfs.grid_file import GridOut
from motor.motor_asyncio import AsyncIOMotorCursor, AsyncIOMotorGridFSBucket
from uuid import UUID
from bson.objectid import ObjectId

document_collection: Collection = async_db.get_collection('user_document_collection')

def update_document_status_by_ids(ids: List[str], status: str):
    collection = db.get_collection('fs.files')
    collection.update_many({'_id': { '$in': ids}}, {'$set': { 'status': 'success' }})

def find_binary_document_by_file_ids(file_ids: List[str]):
    binary_files = []
    for file_id in file_ids:
        grid_out = fs.find_one({'_id': ObjectId(file_id)})
        binary_files.append({
            'id': file_id,
            'filename': grid_out.filename,
            'content_type': grid_out.metadata['mime_type'],
            'data': grid_out.read()
        })
    
    return binary_files

def create_document(doc: DocumentCreate) -> GridOut:
    persisted = fs.put(doc.content, filename=doc.filename, metadata={'mime_type': doc.mime_type, 'conversation_id': doc.conversation_id})
    grid_out = fs.find_one({"_id": persisted})
    return grid_out

def find_document_by_conversation_id_and_filenames(conversation_id: str, filenames: List[str]) -> List[GridOut]:
    cursor = fs.find({"metadata.conversation_id": conversation_id, "filename": {'$in': filenames}})
    return list(cursor)

def find_document_by_conversation_id(conversation_id: UUID) -> List[GridOut]:
    cursor = fs.find({'metadata.conversation_id': str(conversation_id)})
    return list(cursor)

async def acreate_document(doc: DocumentCreate) -> dict:
    gfs = AsyncIOMotorGridFSBucket(async_db)

    persisted = await gfs.upload_from_stream(filename=doc.filename, 
                                             source=doc.content, 
                                             metadata={'mime_type': doc.mime_type, 'conversation_id': doc.conversation_id})
    print(persisted)
    cursor = gfs.find({"_id": persisted}).limit(1)
    res = await cursor.to_list(None)
    return None if not res else res[0]

async def afind_document_by_conversation_id(conversation_id: str):
    gfs = AsyncIOMotorGridFSBucket(async_db)
    cursor = gfs.find({'metadata.conversation_id': conversation_id})
    return await cursor.to_list(None)

async def aexist_by_conversation_id_and_filename(conversation_id: str, filename: str):
    gfs = AsyncIOMotorGridFSBucket(async_db)
    cursor = gfs.find({'metadata.conversation_id': conversation_id, 'filename': filename})
    return await cursor.to_list(None) is not None

async def afind_document_by_conversation_id_and_filenames(conversation_id: str, filenames: List[str]) -> List[dict]:
    gfs = AsyncIOMotorGridFSBucket(async_db)
    cursor = gfs.find({"metadata.conversation_id": conversation_id, "filename": {'$in': filenames}})
    return await cursor.to_list(None)
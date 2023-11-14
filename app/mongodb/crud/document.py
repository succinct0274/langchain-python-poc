from app.mongodb.base import db
from pymongo.results import InsertOneResult
from app.mongodb.schema.document import DocumentCreate, DocumentSchema

document_collection = db.get_collection('user_document_collection')

async def create_document(doc: DocumentCreate) -> dict:
    persisted: InsertOneResult = await document_collection.insert_one(doc.__dict__)
    entity = document_collection.find_one({'_id': persisted.inserted_id})
    return entity
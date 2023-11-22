from motor.motor_asyncio import AsyncIOMotorGridFSBucket, AsyncIOMotorClient
import os
import dns

client = AsyncIOMotorClient(os.getenv('MONGO_DATABASE_URL'))
db = client.get_database("chatbot")
# gfs = AsyncIOMotorGridFSBucket(db)
user_document_collection = db.get_collection('user_document_collection')


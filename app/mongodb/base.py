from motor.motor_asyncio import AsyncIOMotorGridFSBucket, AsyncIOMotorClient
import os
import dns
from pymongo import MongoClient
import gridfs

async_client = AsyncIOMotorClient(os.getenv('MONGO_DATABASE_URL'))
async_db = async_client.get_database("chatbot")
# gfs = AsyncIOMotorGridFSBucket(db)
user_document_collection = async_db.get_collection('user_document_collection')

db = MongoClient(os.getenv('MONGO_DATABASE_URL')).get_database('chatbot')
fs = gridfs.GridFS(db)
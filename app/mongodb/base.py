import motor.motor_asyncio
import os
import dns

client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv('MONGO_DATABASE_URL'))
db = client.get_database("chatbot")
user_document_collection = db.get_collection('user_document_collection')


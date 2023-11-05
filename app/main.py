from typing import Union
from fastapi import FastAPI
from dotenv import load_dotenv
from .router import langchains
import uvicorn
import os
from app.database.base import Base, SessionLocal, engine

load_dotenv()

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(langchains.router)

@app.get("/")
def read_root():
    print(os.environ['OPENAI_API_KEY'])
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
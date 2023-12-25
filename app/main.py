from typing import Union
from fastapi import FastAPI
from dotenv import load_dotenv
from .router import langchains
from .langchain.router import router as langchain_router
import uvicorn
import os
from app.database.base import Base, SessionLocal, engine
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import logging.config
import yaml

load_dotenv()

Base.metadata.create_all(bind=engine)

with open('log_conf.yaml', 'r') as f:
    confd = yaml.safe_load(f)

logging.config.dictConfig(confd)

import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from llama_index import download_loader
    download_loader("PyMuPDFReader", refresh_cache=False)
    yield

app = FastAPI(lifespan=lifespan)

origins = [
    'http://localhost',
    'http://localhost:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=['*'],
    allow_methods=["*"],
    expose_headers=["X-Conversation-Id"]
)
app.include_router(langchains.router)
app.include_router(langchain_router)

@app.get("/")
def read_root():
    print(os.environ['OPENAI_API_KEY'])
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
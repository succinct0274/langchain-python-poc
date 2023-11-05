from fastapi import APIRouter, Depends
from typing import Annotated, List, Union
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Header
from app.service.langchain.model import get_langchain_model
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain.utilities.serpapi import SerpAPIWrapper
from app.service.langchain.agents.fake_agent import FakeAgent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.document_loaders import UnstructuredAPIFileIOLoader
import pandas as pd
import os
from langchain.vectorstores.chroma import Chroma
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.storage.in_memory import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory, ConversationKGMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.schema import Document
import uuid
from langchain.memory.chat_message_histories import PostgresChatMessageHistory
from app.database.crud.conversation_history import create_conversation_history
from app.database.schema.conversation_history import ConversationHistoryCreate
from app.database.base import SessionLocal, get_session_local
from sqlalchemy.orm import Session

router = APIRouter(
    prefix='/langchains',
    tags=['langchains'],
)

def random_word(query: str) -> str:
    print("\nNow I'm doing this!")
    return "foo"

supported_docuemnt_types = set(['application/pdf', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'])

def _process_pdf_files(files: List[UploadFile]) -> List[Document]:
    documents = []
    pdf_files = [file for file in files if file.content_type == 'application/pdf']
    for pdf in pdf_files:
        loader = UnstructuredAPIFileIOLoader(pdf.file, url=os.getenv('UNSTRUCTURED_API_URL'), metadata_filename=pdf.filename)
        docs = loader.load_and_split()
        for doc in docs:
            if 'doc_in' in doc.metadata:
                continue
            doc.metadata['doc_id'] = str(uuid.uuid4())
        documents.extend(docs)
    
    return documents

@router.post('/')
async def create(db: Session=Depends(get_session_local)):
    value = uuid.uuid4()
    print(value)
    create_conversation_history(db, ConversationHistoryCreate(human_message='How are you?', ai_message='I am fine. Thank you.', conversation_id=value))

@router.post('/conversate')
async def conversate(question: Annotated[str, Form()], files: Annotated[List[UploadFile], File()], x_conversation_reference_number: Annotated[Union[str, None], Header()]=None,llm=Depends(get_langchain_model)):
    content_types = set([file.content_type for file in files])
    supported = content_types.issubset(supported_docuemnt_types)
    if not supported:
        raise HTTPException(status_code=400, detail='Unsupported document type')

    documents = _process_pdf_files(files)
    # Make vector store asynchronous
    hf_embedding = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        encode_kwargs={'normalize_embeddings': False}
    )
    db = Chroma.from_documents(documents, hf_embedding)
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=db,
        docstore=store,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=500),
    )

    doc_ids = [doc.metadata['doc_id'] for doc in documents]
    retriever.add_documents(documents, ids=None)

    sub_docs = db.similarity_search("Ketanji Brown Jackson")
    print(sub_docs[0].page_content)

    retrieved_docs = retriever.get_relevant_documents("Ketanji Brown Jackson")
    print(len(retrieved_docs[0].page_content))

    model_id = "Writer/palmyra-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    # llm = HuggingFacePipeline(pipeline=pipe)

    # Instantiate the summary llm and set the max length of output to 300
    summarization_model_name = 'pszemraj/led-large-book-summary'
    summarization_llm = HuggingFacePipeline(pipeline=pipeline('summarization', summarization_model_name, max_length=300))

    memory = ConversationSummaryBufferMemory(llm=summarization_llm, input_key='chat_history', return_messages=True, verbose=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever,
        # memory=ConversationKGMemory(llm=summarization_llm, memory_key='chat_history', return_messages=True),
        memory=memory,
        # memory = ConversationBufferMemory(memory_key='chat_history', output_key='answer', return_messages=True),
        verbose=True,
    )
    
    query = "What did the president say about Ketanji Brown Jackson"
    result = qa({"question": query})
    print(result)

    query = 'What happen to Justice Breyer'
    result = qa({'question': query})
    print(result)
    # Load documents into vector store
    # df = pd.DataFrame()
    # panda_agent = create_pandas_dataframe_agent(
    #     llm=llm,
    #     df=df,
    #     verbose=True,
    #     agent_type=AgentType.OPENAI_FUNCTIONS,
    # )

    # panda_agent.run('how many rows are there?')


@router.post('/fake')
async def fake(question: Annotated[str, Form()], files: Annotated[list[UploadFile], File()], llm=Depends(get_langchain_model)):
    search = SerpAPIWrapper()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events",
        ),
        Tool(
            name="RandomWord",
            func=random_word,
            description="call this to get a random word.",
        ),
    ]

    agent = FakeAgent()

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    agent_executor.run("How many people live in canada as of 2023?")


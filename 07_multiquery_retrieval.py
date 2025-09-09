from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
from typing import List


load_dotenv()

embedding_funciton = OpenAIEmbeddings(model = 'text-embedding-3-small')

db = Chroma(
    embedding_function=embedding_funciton,
    persist_directory='db/chroma_db',
    collection_metadata={'hnsw:space':"cosine"}
)

class QueryVariation(BaseModel):
    queries: List[str]
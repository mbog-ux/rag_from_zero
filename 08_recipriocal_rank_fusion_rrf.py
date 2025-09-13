from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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

original_query = 'How does Tesla make money?'

print(f'\n Original Query: {original_query}\n')

llm = ChatOpenAI(model = 'gpt-4o-mini')
llm_with_tools = llm.with_structured_output(QueryVariation)

prompt = f"""Generate 3 different variations of this query that whould help retreieve relevant couments.

Original query: {original_query}

Return 3 alternatives querie that rephrase of approch the same question from different angles

"""

resposne = llm_with_tools.invoke(prompt)
query_variations = resposne.queries

# print(query_variations)

for i, variation in enumerate(query_variations):
    print(f"{i+1}.{variation}")


retriever = db.as_retriever()

all_results=[]

for i, variation in enumerate(query_variations):
    print(f"Results for {i+1} query: {variation}")
    docs = retriever.invoke(variation)
    print(f'Found {len(docs)} documents')
    all_results.append(docs)
    for i,doc in enumerate(docs):
        print(f'Document {i}:')
        print(doc.page_content[:250],'...','\n')
    print('='*60)



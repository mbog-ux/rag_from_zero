from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = 'db/chroma_db'
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

db = Chroma(
    persist_directory=persistent_directory,embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
    )

retriever = db.as_retriever(search_kwags = {"k":3})
query     = 'Which islan does SpaceX lease for its launches in the Pacific?'

relevant_docs = retriever.invoke(query)

print(f'User query: {query}')
print(f'--- Context ---')

for i,doc in enumerate(relevant_docs):
    print(f"Document {i}:\n{doc.page_content}\n")
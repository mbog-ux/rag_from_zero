from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()

embedding_funciton = OpenAIEmbeddings(model = 'text-embedding-3-small')
db = Chroma(
    embedding_function=embedding_funciton,
    persist_directory='db/chroma_db',
    collection_metadata={'hwsw:space':"cosine"}
)




query = 'How much Microsoft paid to acquire Github?'
query = 'How to chunk the text?'

# query = 'Who founded Microsoft?'

print('=== Method 1. Similarity search (k=3) === ')
retriever = db.as_retriever(
    search_kwargs = {"k":3}
)
docs = retriever.invoke(query)
print(f'Found {len(docs)} documents')

for i,doc in enumerate(docs):
    print(f'Document {i}:')
    print(doc.page_content[:500],'...','\n')
print('='*60)
# print(docs)

print('=== Method 2. Similarity search (k=3) with scrore threshold === ')
retriever = db.as_retriever(
    search_type   = 'mmr',
    search_kwargs = {"k":3,
                     'score_threshold':0.3
                     }
)
docs = retriever.invoke(query)
print(f'Found {len(docs)} documents')

for i,doc in enumerate(docs):
    print(f'Document {i}:')
    print(doc.page_content[:250],'...','\n')
print('='*60)


print('=== Method 3. Maximum Marginal Relevance (MMR) === ')
retriever = db.as_retriever(
    search_type   = 'mmr',
    search_kwargs = {"k":3,
                     'fetch_k':10,
                     'lambda_mult':0.5
                     }
)
docs = retriever.invoke(query)
print(f'Found {len(docs)} documents')

for i,doc in enumerate(docs):
    print(f'Document {i}:')
    print(doc.page_content[:250],'...','\n')
print('='*60)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage

from dotenv import load_dotenv


load_dotenv()

persistent_directory = 'db/chroma_db'
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

db = Chroma(
    persist_directory=persistent_directory,embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
    )

retriever = db.as_retriever(search_kwags = {"k":3})
query     = 'Which island does SpaceX lease for its launches in the Pacific?'

relevant_docs = retriever.invoke(query)

print(f'User query: {query}')
print(f'--- Context ---')

for i,doc in enumerate(relevant_docs):
    print(f"Document {i}:\n{doc.page_content}\n")


combined_input = f"""Based on the following documents, please answer the question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a cleat, helpful answer unsing only the information from these documents. If you cammort fint the answer, tell you cannot find it. Say I dont know information to answer this question
"""

model = ChatOpenAI(model='gpt-4o-mini')

messages = [
    SystemMessage(content = "You are helpful assistant."),
    HumanMessage(content = combined_input),
]

result = model.invoke(messages)

print("\n--- Generated Response ---")
print("Content only:\n")
print(result.content)

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist")
    
    loader = DirectoryLoader(
        path = docs_path,
        glob = '*.txt',
        loader_cls = TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Add yout documents.")
    
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content len: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:500]} ...")
        print(f" MetaData: {doc.metadata}")
    return documents


def split_docuemnts(documents, chunk_size = 800, chunk_overlap = 0):
    text_spliter = CharacterTextSplitter(
        chunk_size = chunk_size,
    )

    chunks = text_spliter.split_documents(documents)
    return chunks

def create_vector_store(chunks,persist_directory = 'db/chroma_db'):
    print( "Creating embeddings and storing in ChromaDB...")
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("--- Finished creating vectore store ---")
    print("Vectore store created and saved to {persist_directory}")

    return vectorstore

def main():
    documents    = load_documents('docs')
    chunks       = split_docuemnts(documents=documents)
    vectorestore = create_vector_store(chunks)

if __name__ == "__main__":
    main()
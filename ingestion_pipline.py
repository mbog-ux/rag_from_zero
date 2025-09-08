import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
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
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content len: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:500]} ...")
        print(f" MetaData: {doc.metadata}")

def main():
    load_documents('docs')
    return None

if __name__ == "__main__":
    main()
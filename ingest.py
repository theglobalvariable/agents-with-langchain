## Tasks to perform
# - Load documents (TextLoader)
# - Split documents into chunks (TextSplitter)
# - Embed chunks (OpenAIEmbeddings)
# - Store embeddings in vector database (PineconeVectorStore)

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

MODEL = "qwen3-embedding:0.6b"

load_dotenv()


def ingest():
    print("Loading documents....")
    loader = TextLoader(
        "D:\\Babu\\Learning\\Udemy\\LangChain and LangGraph\\agents-with-langchain\\medium-article.txt",
        encoding="utf-8",
    )
    document = loader.load()
    print(f"Number of documents: {len(document)}")

    print("Splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)
    print(f"Number of chunks: {len(chunks)}")

    print("Ingesting....")
    embeddings = OllamaEmbeddings(model=MODEL)
    PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=os.getenv("INDEX_NAME")
    )
    print("Ingestion complete!")


if __name__ == "__main__":
    ingest()

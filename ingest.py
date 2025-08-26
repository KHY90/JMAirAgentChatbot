
import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from app.supabase_client import supabase
from dotenv import load_dotenv

load_dotenv()

def main():
    # Load the document
    loader = UnstructuredMarkdownLoader("documents/infomation.md")
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Store in Supabase
    vector_store = SupabaseVectorStore.from_documents(
        docs,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )

    print("Data ingestion complete.")

if __name__ == "__main__":
    main()

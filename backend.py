# Importing all required documents

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


# Step 1: Load PDF

file_path = os.path.join("documents", "tcs.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()

print("Total pages are:", len(documents))


# Step 2: Split documents into chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=600
)

my_chunks = splitter.split_documents(documents)

print("Total chunks are:", len(my_chunks))


# Step 3: Create embedding model

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2'
)


# Step 4: Create FAISS vector database

vector_store = FAISS.from_documents(
    documents=my_chunks,
    embedding=embeddings
)


# Step 5: Save FAISS database locally

vector_store.save_local("faiss_index")

print("Successfully saved PDF into FAISS database!")


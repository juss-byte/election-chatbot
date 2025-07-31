import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_and_index_documents(folder_path="docs", index_path="faiss_index"):
    print("Starting document loading...")

    docs = []
    for filename in os.listdir(folder_path):
        print(f"Found file: {filename}")
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}")
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())

    print(f"Loaded {len(docs)} documents. Splitting...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    print(f"Got {len(split_docs)} chunks. Embedding...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Saving FAISS vector store...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(index_path)

    print(f"DONE: Indexed {len(split_docs)} chunks and saved to {index_path}")

if __name__ == "__main__":
    load_and_index_documents()

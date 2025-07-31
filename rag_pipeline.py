from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

def get_rag_chain(index_path="faiss_index"):
    # Load embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(
    search_type="mmr",  # Max Marginal Relevance for diversity
    search_kwargs={
        "k": 5,          # Return top 5 relevant chunks
        "fetch_k": 10    # Search across top 10 before filtering for diversity
    }
)

    # Load huggingface model (flan-t5-large)
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        tokenizer="google/flan-t5-large",
        max_length=1024,
        device=-1,  # CPU
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt_template = PromptTemplate.from_template("""
    You are a knowledgeable and professional assistant focused on the Romanian 2025 elections.
    Always respond factually and objectively, based only on the provided context.
    Do not invent information or attempt humor.

    Answer in full sentences using clear, formal language.
    Begin every answer by saying: "Thank you for your question. Here is the information I found:" and end every answer with "If you have any further questions, feel free to ask."
    If the context does not include the answer, say you don't know.

    Context:
    {context}

    Question: {question}
    Answer:
    """)


    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    ),
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt_template}
)
    return qa_chain

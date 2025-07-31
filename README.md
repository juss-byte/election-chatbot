# Romanian 2025 Election Chatbot

A local chatbot that answers questions about the Romanian 2025 elections using RAG (Retrieval-Augmented Generation). Built with Streamlit, FAISS, HuggingFace embeddings, and Google's Flan-T5 model for natural language generation.

---

## Features

- Fast and accurate retrieval using FAISS
- Embeddings powered by `all-MiniLM-L6-v2` from HuggingFace
- Answer generation with `google/flan-t5-large`
- Preloaded with documents on Romanian 2025 election rules, candidates, and government structure
- Easy-to-use local Streamlit interface

---

## Launching

1. Clone the repository or download the ZIP
2. Install dependencies:

```bash
pip install -r recs.txt
```

3. Prepare the document index (FAISS)
This step processes all PDFs in the docs/ folder, splits them into chunks, generates embeddings, and saves a searchable FAISS index locally.

```bash
python load_documents.py
```

4. Run the chatbot

```bash
streamlit run app.py
```

5. Enjoy!

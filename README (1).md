# Document Q&A Assistant (RAG)

A Streamlit app where you upload a PDF and ask questions about it. Answers come from the document's own text, and the app shows the source passages (with page numbers) so you can verify them.

## How it works

1. **Ingestion:** the PDF is read page by page and split into overlapping chunks (800 characters, 150 overlap).
2. **Embedding:** each chunk is converted into a vector with the `all-MiniLM-L6-v2` Sentence-Transformers model.
3. **Indexing:** the vectors are stored in a FAISS index (cosine similarity).
4. **Retrieval:** the question is embedded the same way and the top 4 most similar chunks are retrieved.
5. **Generation:** the chunks and the question go into a prompt, and an LLM (Llama 3.1 via the Groq API) is told to answer only from that context.
6. **Sources:** the retrieved chunks are shown under the answer.

## Tech stack

Python, Streamlit, Sentence-Transformers, FAISS, PyPDF, Groq API

## Run it locally

```bash
pip install -r requirements.txt
```

Set your free Groq API key (get one at console.groq.com):

```bash
# Windows PowerShell
$env:GROQ_API_KEY="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"

# Mac / Linux
export GROQ_API_KEY=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
```

Then start the app:

```bash
streamlit run rag_app.py
```

## Experiments I ran

<!-- Fill these in with your real results after testing -->
- Chunk size 800 vs 300: ...
- Top-k 4 vs 1: ...
- Question not covered by the document: ...

## Limitations

- Works on text PDFs only (scanned image PDFs need OCR).
- Character-based chunking can split sentences; a sentence-aware splitter would be better.
- No conversation memory yet; each question is answered independently.
- Not evaluated with a formal retrieval or answer-quality metric.

## Possible improvements

- Add chat history so follow-up questions work
- Add a re-ranking step after retrieval
- Evaluate retrieval quality on a small set of question/answer pairs

"""
Minimal RAG app: upload a PDF, ask questions, get answers grounded in the document.

Flow:
  PDF -> text per page -> overlapping chunks -> embeddings -> FAISS index
  question -> embedding -> top-k similar chunks -> prompt -> LLM -> answer + sources
"""
import os

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---- Settings you must be able to explain in an interview ----
EMBED_MODEL = "all-MiniLM-L6-v2"   # turns text into 384-dimensional vectors
LLM_MODEL = "openai/gpt-oss-20b"  # check console.groq.com for current model names
CHUNK_SIZE = 800                    # characters per chunk
CHUNK_OVERLAP = 150                 # characters shared between neighbouring chunks
TOP_K = 4                           # number of chunks retrieved per question


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def read_pdf(file):
    """Return a list of (page_number, text)."""
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(pages):
    """Split each page into overlapping character chunks, remembering the page number."""
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for page_no, text in pages:
        for start in range(0, len(text), step):
            piece = text[start:start + CHUNK_SIZE].strip()
            if piece:
                chunks.append({"text": piece, "page": page_no})
    return chunks


def build_index(chunks, embedder):
    """Embed every chunk and store the vectors in a FAISS index."""
    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
    vectors = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product = cosine similarity on normalized vectors
    index.add(vectors)
    return index


def retrieve(question, index, chunks, embedder):
    """Embed the question and return the TOP_K most similar chunks with scores."""
    q = embedder.encode([question], normalize_embeddings=True)
    scores, ids = index.search(np.array(q, dtype="float32"), TOP_K)
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def generate_answer(question, retrieved):
    """Put the retrieved chunks into the prompt and ask the LLM to answer only from them."""
    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c, _ in retrieved)
    system = (
        "You answer questions using ONLY the provided context. "
        "If the answer is not in the context, say you could not find it in the document. "
        "Do not give medical advice beyond what the document states."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    return response.choices[0].message.content


# ------------------------- Streamlit UI -------------------------
st.title("Document Q&A Assistant (RAG)")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is not None:
    if st.session_state.get("doc_name") != uploaded.name:
        embedder = load_embedder()
        with st.spinner("Reading, chunking and indexing the document..."):
            chunks = chunk_text(read_pdf(uploaded))
            if not chunks:
                st.error("No text found in this PDF (it may be a scanned image).")
                st.stop()
            st.session_state.chunks = chunks
            st.session_state.index = build_index(chunks, embedder)
            st.session_state.doc_name = uploaded.name
        st.success(f"Indexed {len(chunks)} chunks.")

    question = st.text_input("Ask a question about the document")
    if question:
        embedder = load_embedder()
        retrieved = retrieve(question, st.session_state.index, st.session_state.chunks, embedder)
        with st.spinner("Generating answer..."):
            st.markdown("### Answer")
            st.write(generate_answer(question, retrieved))
        st.markdown("### Sources")
        for chunk, score in retrieved:
            with st.expander(f"Page {chunk['page']} (similarity {score:.2f})"):
                st.write(chunk["text"])

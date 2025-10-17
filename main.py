# main.py
import os
import glob
import pickle
from typing import List
from pydantic import BaseModel

# Document handling
from pypdf import PdfReader

# Vector store
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# LangChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------- CONFIG ----------------
DOCS_FOLDER = "redtec_docs"
FAISS_INDEX_PATH = "faiss_index.bin"
DOCS_META_PATH = "docs_meta.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("⚠️  No se detectó la variable OPENROUTER_API_KEY. Configúrala antes de ejecutar.")

# ---------------- FastAPI Setup ----------------
app = FastAPI(title="RAG Endpoint /ask - Prueba Técnica")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    top_k: int = 4

# ---------------- Document Loaders ----------------
def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(path: str) -> str:
    text = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    except Exception as e:
        print(f"Error leyendo PDF {path}: {e}")
    return "\n".join(text)

def load_documents(folder=DOCS_FOLDER) -> List[Document]:
    docs = []
    patterns = ["*.txt", "*.pdf", "*.md"]
    for pat in patterns:
        for fp in glob.glob(os.path.join(folder, pat)):
            if fp.lower().endswith(".txt") or fp.lower().endswith(".md"):
                txt = load_txt(fp)
            elif fp.lower().endswith(".pdf"):
                txt = load_pdf(fp)
            else:
                continue
            docs.append(Document(page_content=txt, metadata={"source": os.path.basename(fp)}))
            print(f" Cargado: {fp}, {len(txt)} caracteres")
    return docs

# ---------------- Text Splitter ----------------
def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    new_docs = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for i, c in enumerate(chunks):
            meta = dict(d.metadata)
            meta.update({"chunk": i})
            new_docs.append(Document(page_content=c, metadata=meta))
    print(f"🔹 Total chunks generados: {len(new_docs)}")
    return new_docs

# ---------------- FAISS Vector Store ----------------
class VectorStore:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadatas = []

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True))

    def create_index(self, docs: List[Document], rebuild=False):
        texts = [d.page_content for d in docs]
        embeddings = self.embed_texts(texts)
        dim = embeddings.shape[1]

        if os.path.exists(FAISS_INDEX_PATH) and not rebuild:
            print("Cargando índice FAISS existente...")
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(DOCS_META_PATH, "rb") as f:
                self.metadatas = pickle.load(f)
            return

        print("Creando nuevo índice FAISS")
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.metadatas = [d.metadata for d in docs]
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(DOCS_META_PATH, "wb") as f:
            pickle.dump(self.metadatas, f)
        print(" Índice FAISS guardado en disco.")

    def query(self, q: str, top_k=4):
        q_emb = self.embed_texts([q])
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.metadatas):
                results.append((self.metadatas[idx], idx))
        return results, D[0]

# ---------------- Global Instances ----------------
vector_store = None
documents_loaded = []

@app.on_event("startup")
def startup_event():
    global vector_store, documents_loaded
    print("Iniciando sistema RAG...")
    docs = load_documents(DOCS_FOLDER)
    if not docs:
        print(f"No se encontraron documentos en '{DOCS_FOLDER}'.")
    chunks = split_documents(docs)
    vector_store = VectorStore()
    vector_store.create_index(chunks, rebuild=False)
    documents_loaded = chunks
    print("VectorStore listo y documentos cargados.")

# ---------------- /ask Endpoint ----------------
@app.post("/ask")
def ask(req: AskRequest):
    if not vector_store or vector_store.index is None:
        raise HTTPException(status_code=500, detail="Vector store no inicializado.")

    q = req.question
    top_k = req.top_k
    print(f"Pregunta recibida: {q}")

    results, distances = vector_store.query(q, top_k=top_k)

    # Recuperar chunks
    hits = []
    for (meta, idx), dist in zip(results, distances):
        chunk_text = documents_loaded[idx].page_content if idx < len(documents_loaded) else ""
        hits.append({
            "source": meta.get("source"),
            "chunk": meta.get("chunk"),
            "score": float(dist),
            "text": chunk_text
        })

    # Concatenar contexto
    context = "\n\n---\n\n".join([h["text"] for h in hits])

    # Generar respuesta final usando OpenRouter GPT
    try:
        llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=[Document(page_content=context)], question=q)
    except Exception as e:
        print(f"Error en llamada a OpenAI/OpenRouter: {e}")
        answer = "Error al consultar el modelo. Revisa tu API key o conexión a internet."

    return {
        "question": q,
        "answer": answer,
        "retrieved_contexts": hits
    }

# ---------------- Main Entry ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

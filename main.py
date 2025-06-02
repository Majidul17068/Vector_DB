import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import uuid

from config import (
    PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME,
    PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP
)

# Initialize the model globally to avoid reloading
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # Using a reliable, public model

# 📘 1. Read PDF
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# ✂️ 2. Split into chunks
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# 🧠 3. Get embeddings using local model
def get_embedding(text):
    return model.encode(text).tolist()

# 🚀 4. Upload to Pinecone
def upload_to_pinecone(vectors):
    # Initialize Pinecone with new API
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Delete existing index if it exists
    if PINECONE_INDEX_NAME in pc.list_indexes().names():
        print(f"Deleting existing index '{PINECONE_INDEX_NAME}'...")
        pc.delete_index(PINECONE_INDEX_NAME)

    # Create new index with correct dimension
    print(f"Creating new index '{PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=len(vectors[0]['values']),  # This will be 384 for all-MiniLM-L6-v2
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_ENV
        )
    )

    # Get the index
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Upload vectors
    index.upsert(vectors=vectors)
    print(f"✅ Uploaded {len(vectors)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'.")

# 🧩 5. Orchestrator
def process_pdf():
    print("📄 Reading PDF...")
    text = extract_text(PDF_PATH)

    print("✂️ Splitting into chunks...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    print("🧠 Generating embeddings and preparing for upload...")
    vectors = []
    for i, chunk in enumerate(tqdm(chunks)):
        try:
            embedding = get_embedding(chunk)
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "text": chunk[:200],  # limit metadata text
                    "source": os.path.basename(PDF_PATH)
                }
            })
        except Exception as e:
            print(f"[Error in chunk {i}]: {e}")

    print("⬆️ Uploading to Pinecone...")
    upload_to_pinecone(vectors)

if __name__ == "__main__":
    process_pdf()

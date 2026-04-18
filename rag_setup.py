# rag_setup.py
# Loads the two documents, splits them into chunks, embeds them using a
# local sentence transformer model, and stores them in a ChromaDB vector database.
# This script only needs to be run once before the experiments begin.

import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Path to the documents folder relative to this file
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")

# Path where ChromaDB will store the vector database on disk
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Name of the ChromaDB collection that will store all document chunks
COLLECTION_NAME = "querypi_docs"

# Lightweight embedding model suitable for the Raspberry Pi
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of characters per chunk when splitting documents
# Kept small to stay within the Pi's memory constraints
CHUNK_SIZE = 500

# Number of characters of overlap between consecutive chunks
# Overlap ensures context is not lost at chunk boundaries
CHUNK_OVERLAP = 50

# Module-level cache for the embedding model and ChromaDB client
# These are loaded once on first use and reused across all retrieve_context calls
_embedding_model = None
_chroma_collection = None


def load_document(filepath):
    # Extract text from either a .txt or .pdf file
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    elif filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    return ""


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # Split a long string of text into smaller overlapping chunks
    # Each chunk is chunk_size characters long with overlap characters
    # shared with the previous chunk to preserve context at boundaries
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def setup_rag():
    # Load the embedding model locally — no internet connection required
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to ChromaDB using a persistent local client
    # Data is saved to disk so it persists between runs
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete the collection if it already exists to avoid duplicate entries
    # on repeated runs of this script
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    # Create a fresh collection to store the document chunks
    collection = client.create_collection(COLLECTION_NAME)

    # Process each supported document in the documents folder
    doc_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".txt") or f.endswith(".pdf")]

    if not doc_files:
        print("No .txt or .pdf files found in the documents folder.")
        return

    for doc_file in doc_files:
        filepath = os.path.join(DOCUMENTS_DIR, doc_file)
        print(f"Processing: {doc_file}")

        # Extract full text from the document
        text = load_document(filepath)

        # Split the text into overlapping chunks
        chunks = split_into_chunks(text)
        print(f"  Created {len(chunks)} chunks from {doc_file}")

        # Embed each chunk into a vector using the local embedding model
        print(f"  Embedding chunks...")
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        # Store each chunk, its embedding, and its metadata in ChromaDB
        # Each chunk is given a unique ID based on the filename and chunk index
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                ids=[f"{doc_file}_chunk_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": doc_file, "chunk_index": i}]
            )

        print(f"  Stored {len(chunks)} chunks for {doc_file}")

    print("RAG setup complete. ChromaDB is ready for experiments.")


def retrieve_context(question, n_results=3):
    # Retrieve the most relevant document chunks for a given question
    # Returns both the context string and the source metadata
    # so results can show which document the answer came from

    # Use the module-level cached model and collection
    # Load them on first call only — subsequent calls reuse the same objects
    global _embedding_model, _chroma_collection

    if _embedding_model is None:
        print("Loading embedding model (first call only)...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    if _chroma_collection is None:
        print("Connecting to ChromaDB (first call only)...")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = client.get_collection(COLLECTION_NAME)

    # Embed the question and find the most similar chunks
    question_embedding = _embedding_model.encode([question]).tolist()
    results = _chroma_collection.query(
        query_embeddings=question_embedding,
        n_results=n_results
    )

    # Extract the retrieved chunks and their source filenames
    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]

    # Combine chunks into a single context string for the prompt
    context = "\n\n".join(chunks)

    return context, chunks, sources


if __name__ == "__main__":
    # Run setup when this script is executed directly
    setup_rag()
# demo.py
# Demonstration script for QueryPi defence presentation.
# Runs a single question against TinyLlama under Condition A (no RAG)
# and Condition B (with RAG) to show the effect of local document retrieval.

import time
import os
import warnings
import logging
import psutil
import ollama

# Suppress warnings from sentence transformers and HuggingFace
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from rag_setup import retrieve_context

MODEL_NAME = "tinyllama:1.1b-chat-v1-q4_0"
QUESTION = "What was the Latin term for the government's official distribution of grain to the Roman populace?"
REFERENCE = "Annona"

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_query(prompt):
    mem_before = get_memory()
    start = time.time()
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_ctx": 2048}
    )
    latency = round(time.time() - start, 1)
    mem_after = get_memory()
    answer = response["message"]["content"].strip()
    return answer, latency, round(max(mem_before, mem_after), 1)

print("\n" + "="*55)
print("  QueryPi Demo")
print("  Question:", QUESTION)
print("  Expected Answer:", REFERENCE)
print("="*55)

print("\n CONDITION A: (No local documents) \n")
answer_a, lat_a, mem_a = run_query(QUESTION)
print("Answer:", answer_a)
print(f"Latency: {lat_a}s | Memory: {mem_a}MB")

print("\n CONDITION B: RAG pipeline (local documents) \n")
context, chunks, sources = retrieve_context(QUESTION)
augmented = (
    f"Use the following information to answer the question.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {QUESTION}"
)
answer_b, lat_b, mem_b = run_query(augmented)
print("Answer:", answer_b)
print(f"Latency: {lat_b}s | Memory: {mem_b}MB")
print(f"Sources used: {set(sources)}")

print("\n" + "="*55)
print("  Reference answer:", REFERENCE)
print("="*55 + "\n")

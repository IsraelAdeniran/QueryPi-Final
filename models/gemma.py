# gemma.py
# Runs all experiments for the Gemma 2B model.
# Tests the model under Condition A (no documents) and Condition B (RAG pipeline)
# and saves all responses and performance measurements to a JSON results file.

import json
import time
import os
import psutil
import ollama
import sys

# Add the project root to the path so questions.py can be imported from models/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from questions import CONDITION_A_QUESTIONS, CONDITION_B_QUESTIONS
from rag_setup import retrieve_context

# Ollama model tag for Gemma 2B Q4 quantised
MODEL_NAME = "gemma:2b-instruct-q4_0"

# Model display name used in results file
MODEL_LABEL = "Gemma 2B"

# Path to save results relative to the project root
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "gemma_results.json")

# Inference parameters — kept constant across all models and conditions
TEMPERATURE = 0.0
CONTEXT_WINDOW = 2048


def get_memory_usage():
    # Returns the current memory usage of this process in megabytes
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_query(prompt):
    # Sends a prompt to the Gemma model via Ollama and records
    # the response, latency in milliseconds, and peak memory usage

    # Record memory before the query
    memory_before = get_memory_usage()

    # Record start time before sending the prompt
    start_time = time.time()

    # Send the prompt to the model with fixed inference parameters
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": TEMPERATURE,
            "num_ctx": CONTEXT_WINDOW
        }
    )

    # Calculate latency in milliseconds
    latency_ms = (time.time() - start_time) * 1000

    # Record memory after the query and calculate peak usage
    memory_after = get_memory_usage()
    peak_memory_mb = max(memory_before, memory_after)

    # Extract the response text from the Ollama response object
    answer = response["message"]["content"].strip()

    return answer, latency_ms, peak_memory_mb


def run_condition_a():
    # Runs all Condition A questions — no documents provided to the model
    # The model responds using its internal pre-trained knowledge only
    print(f"\nRunning Condition A for {MODEL_LABEL}...")
    results = []

    for item in CONDITION_A_QUESTIONS:
        print(f"  {item['id']}: {item['question'][:60]}...")

        # Send the question directly to the model with no additional context
        answer, latency_ms, peak_memory_mb = run_query(item["question"])

        # Store the result with all relevant metadata
        results.append({
            "id": item["id"],
            "condition": "A",
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "model_answer": answer,
            "latency_ms": round(latency_ms, 2),
            "peak_memory_mb": round(peak_memory_mb, 2)
        })

        print(f"    Latency: {latency_ms:.0f}ms | Memory: {peak_memory_mb:.1f}MB")

    return results


def run_condition_b():
    # Runs all Condition B questions — model has access to retrieved document context
    # The question is augmented with relevant chunks from ChromaDB before being sent
    # Retrieved chunks and sources are saved to show RAG is working
    print(f"\nRunning Condition B for {MODEL_LABEL}...")
    results = []

    for item in CONDITION_B_QUESTIONS:
        print(f"  {item['id']}: {item['question'][:60]}...")

        # Retrieve the most relevant document chunks and their sources
        context, chunks, sources = retrieve_context(item["question"])

        # Build the augmented prompt combining the context and the question
        augmented_prompt = (
            f"Use the following information to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {item['question']}"
        )

        # Send the augmented prompt to the model
        answer, latency_ms, peak_memory_mb = run_query(augmented_prompt)

        # Store the result with retrieved chunks and sources included
        # This allows verification that RAG is working correctly
        results.append({
            "id": item["id"],
            "condition": "B",
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "model_answer": answer,
            "latency_ms": round(latency_ms, 2),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "retrieved_chunks": chunks,
            "retrieved_sources": sources
        })

        print(f"    Latency: {latency_ms:.0f}ms | Memory: {peak_memory_mb:.1f}MB")
        print(f"    Sources: {set(sources)}")

    return results


def save_results(results):
    # Save all results to a JSON file in the results folder
    # Creates the results folder if it does not already exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {
        "model": MODEL_LABEL,
        "ollama_tag": MODEL_NAME,
        "temperature": TEMPERATURE,
        "context_window": CONTEXT_WINDOW,
        "results": results
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    # Run Condition A then Condition B and save all results
    print(f"Starting experiments for {MODEL_LABEL}")

    condition_a_results = run_condition_a()
    condition_b_results = run_condition_b()

    all_results = condition_a_results + condition_b_results
    save_results(all_results)

    print(f"\nExperiments complete for {MODEL_LABEL}")
    print(f"Total questions answered: {len(all_results)}")
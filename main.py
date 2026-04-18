# main.py
# Entry point for the QueryPi experiment.
# Runs the full experiment pipeline in the correct order:
# 1. Sets up the RAG database
# 2. Runs each model as a completely separate process
# Running models as separate processes ensures each one is fully cleared
# from memory before the next one starts — important on the Raspberry Pi.

import sys
import os
import subprocess

# Add the project root to the path so rag_setup can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_setup import setup_rag

# Paths to each model file relative to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    {
        "label": "TinyLlama 1.1B",
        "path": os.path.join(BASE_DIR, "models", "tinyllama.py")
    },
    {
        "label": "Gemma 2B",
        "path": os.path.join(BASE_DIR, "models", "gemma.py")
    },
    {
        "label": "Qwen 2.5 3B",
        "path": os.path.join(BASE_DIR, "models", "qwen.py")
    },
]


def run_model(label, path):
    # Runs a single model file as a completely separate Python process
    # The current process waits until the model finishes before continuing
    # This ensures the model is fully cleared from memory before the next starts
    print(f"\n{'='*50}")
    print(f"Starting experiments for {label}")
    print(f"{'='*50}")

    result = subprocess.run(
        [sys.executable, path],
        check=True
    )

    if result.returncode == 0:
        print(f"Experiments complete for {label}")
    else:
        print(f"Something went wrong running {label} — check the output above")


if __name__ == "__main__":

    # Step 1 — build the ChromaDB vector database from the local documents
    # This only needs to run once but is included here for a clean full run
    print("Step 1: Setting up RAG database...")
    setup_rag()

    # Step 2 — run each model as a separate process one at a time
    print("\nStep 2: Running experiments...")
    for model in MODELS:
        run_model(model["label"], model["path"])

    # Step 3 — print final summary once all models are done
    print(f"\n{'='*50}")
    print("All experiments complete.")
    print("Results saved to the results/ folder:")
    print("  - results/tinyllama_results.json")
    print("  - results/gemma_results.json")
    print("  - results/qwen_results.json")
    print(f"{'='*50}")
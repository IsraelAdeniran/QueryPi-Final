# QueryPi

QueryPi is a final year research project developed at Technological University Dublin. It evaluates the performance of lightweight language models running fully offline on a Raspberry Pi 4, with and without a Retrieval-Augmented Generation (RAG) pipeline.

The goal is to explore whether small, quantised language models can serve as effective offline tutoring tools in resource-constrained environments, such as classrooms with no internet access.

---

## What it does

Three lightweight language models are tested across two conditions:

- **Condition A:** the model answers questions using only its built-in knowledge
- **Condition B:**  the model is given relevant context retrieved from local documents via a RAG pipeline

Performance is measured across five metrics: response latency, memory usage, answer accuracy, hallucination rate, and document grounding.

The three models tested are:
- TinyLlama 1.1B (q4_0)
- Gemma 2B (q4_0)
- Qwen 2.5 3B (q4_K_M)

All models are run in Q4 quantised format via [Ollama](https://ollama.com), entirely offline, on a Raspberry Pi 4 with 4GB RAM.

---

## Project structure

```
QueryPi-Final/
├── main.py              # Entry point - runs the full experiment pipeline
├── questions.py         # Loads and organises questions from questions.json
├── questions.json       # 30 questions across two topics and three categories
├── rag_setup.py         # Builds the ChromaDB vector database from local documents
├── requirements.txt     # Python dependencies
├── documents/
│   ├── astronomy.pdf        # Science document (solar system)
│   └── roman_empire.pdf     # History document (Roman Empire)
├── models/
│   ├── tinyllama.py     # Experiment runner for TinyLlama 1.1B
│   ├── gemma.py         # Experiment runner for Gemma 2B
│   └── qwen.py          # Experiment runner for Qwen 2.5 3B
└── results/             # Output folder - JSON results saved here after running
```

---

## Requirements

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running
- The three models pulled via Ollama (see installation steps below)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/IsraelAdeniran/QueryPi-Final.git
cd QueryPi-Final
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

On a Raspberry Pi, add the `--break-system-packages` flag if needed:

```bash
pip install -r requirements.txt --break-system-packages
```

**4. Install Ollama and pull the models**

Install Ollama from [https://ollama.com](https://ollama.com), then pull each model:

```bash
ollama pull tinyllama:1.1b-chat-v1-q4_0
ollama pull gemma:2b-instruct-q4_0
ollama pull qwen2.5:3b-instruct-q4_K_M
```

This must be done while connected to the internet. Once pulled, all models run fully offline.

---

## How to run the experiment

**Step 1. Build the RAG database**

This only needs to be run once. It loads the documents, splits them into chunks, embeds them, and stores them in a local ChromaDB vector database.

```bash
python3 rag_setup.py
```

**Step 2. Run the full experiment**

```bash
python3 main.py
```

This will run all three models in sequence, each as a separate process, under both Condition A and Condition B. Results are saved to the `results/` folder as JSON files:

```
results/
├── tinyllama_results.json
├── gemma_results.json
└── qwen_results.json
```

**Running a single model**

Each model file can also be run independently:

```bash
python3 models/tinyllama.py
python3 models/gemma.py
python3 models/qwen.py
```

---

## Acknowledgements

Developed by **Adefolajuwon Adeniran** as part of a final year project at the School of Computing, Technological University Dublin, 2026.

Supervisor: **Dr. Robert Smith**

Models used: [TinyLlama](https://github.com/jzhang38/TinyLlama), [Gemma](https://ai.google.dev/gemma), [Qwen 2.5](https://github.com/QwenLM/Qwen2.5)  
Inference engine: [Ollama](https://ollama.com)  
Vector database: [ChromaDB](https://www.trychroma.com)  
Embeddings: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

# Curriculum-Based-AI-Tutor
# 📚 Curriculum-Based AI Tutor — Class 8 NCERT Science

A Retrieval-Augmented Generation (RAG) tutoring system that answers student questions **strictly from the NCERT Class 8 Science textbook**, using hybrid BM25 + dense retrieval, cross-encoder reranking, and Groq-hosted LLMs.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![RAG](https://img.shields.io/badge/Architecture-Hybrid%20RAG-orange)
![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203-green)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

---

## Overview

This project builds an AI tutor for Class 8 NCERT Science that:

- Answers **only** from the textbook — refuses out-of-syllabus questions with a fixed, safe message
- Combines **dense semantic search (FAISS)** with **BM25 keyword search**, fused and reranked by a **cross-encoder**
- Supports **chapter-aware retrieval**, so answers can be scoped to a specific chapter for higher precision
- Is evaluated with **BLEU-4**, **ROUGE-L**, and **BERTScore** — with BERTScore treated as the primary signal, since it captures semantic correctness better than n-gram overlap metrics for open-ended explanatory answers

Built as part of a curriculum-based AI tutoring initiative at **NewtonAI Pvt. Ltd.**

---

## Architecture

```
NCERT PDF (228 pages)
        │
        ▼
 Text Extraction & Cleaning  (pypdf, regex cleanup)
        │
        ▼
 Chunking + Chapter Tagging  (RecursiveCharacterTextSplitter, keyword-based chapter tagger)
        │
        ├──────────────┐
        ▼              ▼
 Dense Embeddings   BM25 Index
 (MiniLM → FAISS)   (rank-bm25)
        │              │
        └──────┬───────┘
               ▼
     Hybrid Retrieval + Fusion
               │
               ▼
     Cross-Encoder Reranking  (ms-marco-MiniLM-L-6-v2)
               │
               ▼
   Optional Chapter Filter
               │
               ▼
     Groq LLM Generation  (llama-3.1-8b-instant / llama-3.3-70b-versatile)
               │
               ▼
        Student Answer
```

**Pipeline stages:**

1. **Ingestion & Cleaning** — Extracts raw text from the NCERT PDF and strips page numbers, timestamps, and chapter-heading artifacts.
2. **Chunking & Metadata Tagging** — Splits cleaned text into ~600-character chunks (80-char overlap) and auto-tags each with one of 13 NCERT chapter labels via keyword matching.
3. **Indexing** — Builds a FAISS inner-product index over MiniLM embeddings and a parallel BM25 index over the tokenized corpus.
4. **Hybrid Retrieval & Reranking** — Retrieves candidates from both indexes, deduplicates and fuses them, then reranks with a cross-encoder. An optional chapter filter can restrict results to one chapter.
5. **Generation** — Passes the top reranked chunks as context to a Groq-hosted Llama model under a system prompt that restricts answers to the supplied context and returns a fixed refusal for out-of-syllabus topics.

---

## Dataset

| | |
|---|---|
| Source | NCERT Class 8 Science textbook (PDF, 228 pages) |
| Extracted text | 421,139 characters (cleaned) |
| Total chunks | 896 (avg. 510 chars/chunk) |
| Chapters covered | 13 |
| Embedding model | `all-MiniLM-L6-v2` (384-dim), with `BAAI/bge-small-en-v1.5` tested for comparison |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

**Chunk distribution across chapters:**

| Chapter | Chunks |
|---|---|
| Ch01 – Crop Production and Management | 24 |
| Ch02 – Microorganisms: Friend and Foe | 54 |
| Ch03 – Synthetic Fibres and Plastics | 14 |
| Ch04 – Materials: Metals and Non-Metals | 36 |
| Ch05 – Coal and Petroleum | 5 |
| Ch06 – Combustion and Flame | 15 |
| Ch07 – Conservation of Plants and Animals | 30 |
| Ch08 – Cell: Structure and Functions | 60 |
| Ch09 – Reproduction in Animals | 16 |
| Ch11 – Force and Pressure | 120 |
| Ch12 – Friction | 4 |
| Ch13 – Sound | 6 |
| General (untagged / cross-chapter) | 512 |

> **Known limitation:** 57% of chunks fell into the untagged "General" bucket, and Chapters 10 (Adolescence) and 13 (Sound) have thin keyword coverage. Chapter-filtered retrieval is less reliable for these areas — see [Limitations](#limitations--future-work).

---

## Results

Evaluated on a 20-question test set (8 factual / 8 conceptual / 4 multi-hop), covering all 13 chapters:

| Metric | Overall | Factual | Conceptual | Multi-hop |
|---|---|---|---|---|
| BLEU-4 | 0.0439 | 0.0454 | 0.0395 | 0.0498 |
| ROUGE-L | 0.1794 | 0.1782 | 0.1817 | 0.1771 |
| **BERTScore F1** | **0.7817** | 0.7791 | 0.7913 | 0.7675 |

- BLEU-4 and ROUGE-L are low by design — they penalize paraphrasing, and the tutor generates fuller, differently-worded explanations rather than terse reference-matching text.
- **BERTScore F1 (0.78) is the primary quality signal** and indicates strong semantic alignment with reference answers.
- Mean end-to-end latency: **1.54s/query** (min 1.45s, max 1.63s).
- Total tokens across the 20-question eval run: 18,439 input / 3,528 output.

### Model comparison (5-question subset)

| LLM | Embedding | BLEU-4 | ROUGE-L | BERTScore F1 |
|---|---|---|---|---|
| llama-3.3-70b-versatile | MiniLM | 0.0459 | 0.1693 | **0.7990** |
| llama-3.1-8b-instant | MiniLM | 0.0478 | 0.1874 | 0.7800 |
| llama-3.1-8b-instant | BGE-small | 0.0231 | 0.1569 | 0.7698 |

The 70B model edges out 8B on BERTScore, but 8B + MiniLM is close behind at a fraction of the cost/latency — a reasonable default for interactive use, with 70B as an optional higher-quality mode. BGE-small underperformed MiniLM on every metric and isn't recommended.

### Error analysis

Five case studies were manually reviewed and classified by failure type:

| Type | Meaning |
|---|---|
| **R-FAIL** | Retrieval brought in the wrong chunks |
| **G-NOISE** | Generation added unsupported information |
| **F-TRIGGER** | Refusal fired incorrectly on an in-syllabus topic |

Key findings: the refusal mechanism correctly fires on genuinely out-of-syllabus questions (e.g. earthquakes — not covered in this textbook, retrieval score −5.41); occasional generation noise appears on complex multi-hop questions (e.g. human eye anatomy); and chapter filtering measurably improves retrieval precision when applied.

---

## Tech Stack

- **PDF parsing:** `pypdf`
- **Chunking:** `langchain-text-splitters`
- **Embeddings:** `sentence-transformers` (MiniLM, BGE)
- **Dense retrieval:** `faiss-cpu`
- **Sparse retrieval:** `rank-bm25`
- **Reranking:** `sentence-transformers` CrossEncoder
- **LLM inference:** `groq` (Llama 3.1 8B / Llama 3.3 70B)
- **Evaluation:** `nltk` (BLEU), `rouge-score` (ROUGE-L), `bert-score` (BERTScore)
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install pypdf langchain-text-splitters sentence-transformers faiss-cpu \
    groq nltk rouge-score pandas tqdm matplotlib rank-bm25 bert-score
```

### 2. Set your Groq API key

Get a free key at [console.groq.com](https://console.groq.com) (no credit card needed), then set it as an environment variable — **never hardcode it in the notebook or commit it to version control:**

```bash
export GROQ_API_KEY="your-key-here"        # macOS/Linux
setx GROQ_API_KEY "your-key-here"          # Windows
```

### 3. Add the textbook PDF

Place the NCERT Class 8 Science PDF in the project root as `Class 8 NCERT Science Book.pdf` (or update `PDF_PATH` in the notebook).

### 4. Run the notebook

Open `ai_tutor_class8.ipynb` and run all cells in order. This will:
- Build the chunked corpus (`class8_science.jsonl`)
- Build the FAISS index (`class8_science.faiss`)
- Run the hybrid retrieval + generation pipeline
- Run the full evaluation and produce `evaluation_chart.png` and `model_comparison_chart.png`

### Example usage

```python
result = ask_tutor("What is friction and why does it happen?")
print(result["answer"])

# Scope to a specific chapter
result = ask_tutor("What does a nucleus do?", chapter_filter="Ch08_Cell")
```

---

## Limitations & Future Work

- **Chapter tagging coverage** — keyword-based tagging leaves 57% of chunks unassigned to a specific chapter; a trained classifier would improve chapter-filtered retrieval.
- **Refusal robustness** — the out-of-syllabus refusal currently relies on the LLM following the system prompt rather than a hard retrieval-score cutoff; adding a numeric threshold would make it more reliable.
- **Evaluation scale** — current results are based on 20 (full) and 5 (comparison) questions; a 50–100 question set with more multi-hop coverage would give higher-confidence quality claims.
- **Deployment** — index-building currently happens on every run; caching the FAISS/BM25 indexes as build artifacts would reduce startup time for a deployed service.

---

## ⚠️ Security Note

If you're forking or adapting this notebook, double-check that no API key is hardcoded as a fallback default anywhere in the code (e.g. `os.environ.get('GROQ_API_KEY', '<key>')`). Always require the key via environment variable only, and add `.env` / credential files to `.gitignore`.

---

## License

Add your preferred license here (e.g. MIT).

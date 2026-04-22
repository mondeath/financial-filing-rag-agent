# Financial RAG Agent - V1 Specification

---

## 1. Project Overview

This project builds a **financial-domain RAG (Retrieval-Augmented Generation) system**.

### Goal

* Answer questions based on financial documents
* Reduce hallucination by grounding answers in retrieved evidence
* Provide citations and supporting text

> This is **NOT a chatbot**, but a **research assistant tool**.

---

## 2. V1 Scope (Strict)

### Included

* Document ingestion
* Text chunking
* Embedding + FAISS index
* Retrieval (top-k)
* LLM-based answer generation
* Evidence citation
* Simple evaluation

### Excluded

* Multi-agent system
* Tool routing
* Reranking
* Training models
* UI frontend
* Multi-turn memory
* MiniMind integration

---

## 3. System Architecture

### Pipeline

```
User Query  
→ Retrieval (vector search)  
→ Context Construction  
→ LLM Generation  
→ Answer + Evidence
```

### Key Principle

> The LLM must answer **ONLY based on retrieved documents**.

---

## 4. Project Structure

```
financial-rag-agent/
├── config.py
├── main.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── eval/
├── index/
├── src/
│   ├── data/
│   ├── retrieval/
│   ├── llm/
│   ├── pipeline/
│   └── eval/
```

---

## 5. Data Specification

### Raw Document Format (`finance_docs.jsonl`)

Each line:

```json
{
  "doc_id": "doc_0001",
  "title": "央行降准释放长期流动性",
  "source": "Sina Finance",
  "date": "2026-04-10",
  "category": "macro",
  "content": "全文文本..."
}
```

---

## 6. Chunking Strategy

### Parameters

* `chunk_size`: 500 characters
* `overlap`: 80 characters

### Output Format

```json
{
  "chunk_id": "doc_0001_chunk_0",
  "doc_id": "doc_0001",
  "title": "...",
  "source": "...",
  "date": "...",
  "chunk_index": 0,
  "text": "..."
}
```

---

## 7. Retrieval Design

### Steps

1. Embed all chunks
2. Store embeddings in FAISS
3. Embed query
4. Retrieve `top_k = 4` chunks

### Output

* List of retrieved chunks with metadata

---

## 8. Prompt Design

### System Instruction

The model must:

* Answer **ONLY using provided context**
* If insufficient info → say `"insufficient information"`
* Avoid hallucination
* Be concise and structured

### Input Format

```
[Question]
...

[Context]
chunk1
chunk2
chunk3
```

---

## 9. RAG Pipeline Logic

### Function: `answer_question(question)`

#### Steps

1. Retrieve top-k chunks
2. Concatenate into context
3. Pass to LLM
4. Generate answer
5. Attach citations

### Output Format

```
Answer:
...

Sources:
- doc_title_1
- doc_title_2

Evidence:
1. ...
2. ...
```

---

## 10. Evaluation Design

### Eval Dataset (`eval_cases.json`)

```json
[
  {
    "qid": "q1",
    "question": "...",
    "task_type": "qa"
  }
]
```

### Evaluation Criteria

* Retrieval relevance
* Groundedness (uses context)
* Hallucination (yes/no)
* Answer completeness

> Evaluation can be manual in V1.

---

## 11. CLI Commands

```bash
python main.py build-chunks
python main.py build-index
python main.py ask "your question"
python main.py eval
```

---

## 12. Key Design Principles

1. **Separation of concerns**
   Retrieval ≠ Generation

2. **Grounded generation**
   Always cite sources

3. **Simplicity first**
   V1 must run end-to-end

4. **Extensibility**

---

## 13. Future Extensions (V2 / V3)

* Tool-based Agent (summarize, compare)
* Structured JSON output
* Reranking
* MiniMind integration (model comparison)
* UI demo

---

## 14. Success Criteria

V1 is complete if:

* System answers questions using documents
* Returns supporting evidence
* Works on 10–20 test queries
* Minimal hallucination

---

## 15. Implementation Priority

1. Data loading + chunking
2. Embedding + FAISS
3. Retrieval
4. LLM generation
5. Pipeline orchestration
6. Evaluation

---

## 16. Notes for Coding Agent (IMPORTANT)

* Keep modules small and clean
* Avoid hardcoding parameters
* Use `config.py` for all configs
* Write reusable functions
* Log intermediate outputs (for debugging)

---

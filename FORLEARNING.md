# For Learning

This note is for building the project and learning RAG at the same time.

You mentioned that you already have some background in large-model inference and training, but you are new to RAG. That is actually a very good starting point. You already understand models; this project helps you learn how to make them useful and reliable on external knowledge.

## 1. What RAG Actually Is

RAG stands for Retrieval-Augmented Generation.

A simple way to understand it:

- the LLM is good at language generation
- the retriever is good at finding relevant evidence
- RAG combines them so the answer is grounded in documents instead of only model memory

In this project, the pipeline is:

```text
Question
-> retrieve relevant chunks
-> build context
-> ask the LLM to answer only from context
-> return answer + evidence
```

Why this matters:

- it reduces hallucination
- it makes answers more explainable
- it lets you update knowledge by changing documents instead of retraining a model

This is one of the most common production patterns around LLM systems, so it is a strong interview project direction.

## 2. What You Should Learn From This Project

If you treat this project as a learning path, there are four core ideas:

### A. Data is part of the model system

In classic model work, you may think in terms of:

- model weights
- tokenizer
- training data
- inference optimization

In RAG, you also need to think about:

- document quality
- metadata quality
- chunk quality
- retrieval quality

A weak retriever can make a strong LLM look bad.

### B. Retrieval and generation are different problems

This separation is very important.

- retrieval answers: "what evidence should we show the model?"
- generation answers: "how should the model use that evidence to respond?"

That is why the spec says:

> Retrieval != Generation

If you explain this clearly in an interview, it signals good system thinking.

### C. Chunking is not a boring preprocessing step

Chunking strongly affects retrieval quality.

If chunks are:

- too small: context is fragmented
- too large: retrieval becomes noisy
- no overlap: information can be cut at boundaries

That is why this V1 uses:

- `chunk_size = 500`
- `overlap = 80`

These are not sacred numbers; they are reasonable starting points for V1.

### D. RAG quality depends on the whole chain

A bad answer can come from many places:

- documents are missing
- chunking is poor
- embeddings are weak
- retrieval top-k misses the right evidence
- prompt is unclear
- model ignores instructions

Learning RAG means learning to debug the chain, not just the model.

## 2.5 Documents vs Eval Data

This distinction is very important, especially for a first RAG project.

Not all text data should be used the same way.

In this repo, you now have two different kinds of raw data:

- `finance_article.jsonl`: better suited as the document corpus
- `finance_QA.jsonl`: better suited as evaluation data

Why:

- article data looks like source material that the retriever should search over
- QA data already contains a question-answer structure, so it is better for checking whether the system can answer well

This is a common beginner mistake:

- putting QA pairs directly into the document store

That can make the system look stronger than it really is, because the retriever may find pre-written answers instead of real evidence documents.

The cleaner setup is:

- use articles as knowledge
- use QA as evaluation

That is much more defensible in an interview.

## 3. How To Read The Current Code

A good order is:

1. [main.py](/Users/ranxu/workspace/Rag-finance/main.py)
2. [config.py](/Users/ranxu/workspace/Rag-finance/config.py)
3. [src/data/schemas.py](/Users/ranxu/workspace/Rag-finance/src/data/schemas.py)
4. [src/data/loader.py](/Users/ranxu/workspace/Rag-finance/src/data/loader.py)
5. [src/data/chunking.py](/Users/ranxu/workspace/Rag-finance/src/data/chunking.py)
6. [src/retrieval/embeddings.py](/Users/ranxu/workspace/Rag-finance/src/retrieval/embeddings.py)
7. [src/retrieval/index.py](/Users/ranxu/workspace/Rag-finance/src/retrieval/index.py)
8. [src/retrieval/retriever.py](/Users/ranxu/workspace/Rag-finance/src/retrieval/retriever.py)
9. [scripts/prepare_demo_dataset.py](/Users/ranxu/workspace/Rag-finance/scripts/prepare_demo_dataset.py)
10. [scripts/prepare_eval_dataset.py](/Users/ranxu/workspace/Rag-finance/scripts/prepare_eval_dataset.py)

Why this order works:

- `main.py` shows the system entrypoints
- `config.py` shows the project assumptions
- `schemas.py` shows the core data shapes
- `loader.py` shows how raw data enters the system
- `chunking.py` shows how documents become retrieval units
- `embeddings.py` shows how text becomes vectors
- `index.py` shows how vectors become searchable
- `retriever.py` shows how the user question becomes top-k results
- `prepare_demo_dataset.py` shows how messy external raw data becomes a smaller demo corpus
- `prepare_eval_dataset.py` shows how QA-style data becomes a lightweight evaluation set

## 4. What Each Module Is Teaching You

### `src/data`

This module teaches:

- input validation
- schema design
- why clean metadata matters
- how raw documents become structured system inputs

Key learning question:

How do you make sure your downstream pipeline is stable even when raw data is messy?

### `src/data/chunking.py`

This module teaches:

- what a chunk is
- why overlap exists
- how document-level metadata is preserved at chunk level

Key learning question:

How do chunk boundaries affect retrieval quality?

Try this yourself later:

- test `chunk_size=300`
- test `chunk_size=800`
- test `overlap=0`
- compare retrieval results

### `src/retrieval/embeddings.py`

This module teaches:

- the abstraction of embeddings
- why we want a pluggable embedding interface
- how to keep a system runnable even before all dependencies are installed

Important note:

The current hashing embedding is a baseline for development and learning. It is not the final production-quality embedding method. It exists so the project is runnable now.

Interview-friendly explanation:

"I designed the embedding layer as an interface first, so I could start system development with a deterministic local baseline and later swap in a real embedding model without changing the upper pipeline."

### `src/retrieval/index.py`

This module teaches:

- vector indexing
- backend abstraction
- why FAISS is common in RAG systems
- how retrieval is often cosine-similarity-like matching over embeddings

Learning point:

RAG systems often benefit from clean interfaces around storage/index backends. That makes it easy to evolve from:

- local simple index
- to FAISS
- to Milvus / Weaviate / pgvector / Elasticsearch-like systems later

### `src/retrieval/retriever.py`

This module teaches:

- turning a user query into an embedding
- searching top-k chunks
- returning evidence candidates for the generation stage

This is the handoff point between search and reasoning.

### `scripts/prepare_demo_dataset.py`

This script teaches:

- dataset curation
- lightweight normalization from external raw data
- why a smaller but cleaner demo set is often better than a huge noisy corpus

Key learning question:

If the raw dataset is large but mixed-quality, how do you create a subset that is better for debugging and demos?

### `scripts/prepare_eval_dataset.py`

This script teaches:

- turning QA data into eval cases
- separating document data from evaluation data
- how to preserve a reference answer without pretending it is perfect ground truth

Key learning question:

How do you create an evaluation workflow that is useful even before you have a full automatic benchmark?

## 5. What To Learn Next

The next stage is where RAG becomes a full application instead of just a retrieval demo.

### A. Prompting for grounded answers

You want the model to behave like:

- use only provided context
- if context is insufficient, say so
- cite evidence
- stay concise

This is not just "prompt engineering" in the shallow sense. It is about aligning model behavior with the product contract.

### B. Context construction

The retriever returns multiple chunks. The pipeline must decide:

- how to order them
- how to format them
- how much metadata to include
- how to avoid context clutter

This step matters more than beginners usually expect.

### C. Evaluation

A RAG system is hard to improve if you only ask "does it feel good?"

You need at least lightweight evaluation around:

- did retrieval fetch the right evidence?
- did the answer actually use the evidence?
- did the model hallucinate beyond the evidence?

Even manual evaluation for 10-20 queries is already useful in V1.

In this repo, a practical V1 evaluation mindset is:

- keep a small `eval_cases.json`
- run representative questions manually
- compare system output against `reference_answer`
- inspect retrieved chunks before blaming the LLM

## 6. A Good Learning Sequence For You

Because you already know some model-side concepts, I would suggest this order:

1. Understand chunking and retrieval deeply.
2. Add a real embedding model.
3. Add LLM answer generation with a strict grounded prompt.
4. Add citations and evidence formatting.
5. Build a tiny eval set and review failures.
6. Iterate based on observed failure modes.

This sequence is good because it teaches you to debug from the bottom of the stack upward.

## 7. Suggested Hands-On Exercises

These are very worth doing yourself.

### Exercise 1: Create a tiny finance dataset

Write 5-10 financial documents in `data/raw/finance_docs.jsonl`.

Make sure they include different topics:

- macro
- equities
- policy
- banking
- market sentiment

Then test:

- does retrieval return the right chunk?
- when does it fail?

### Exercise 2: Stress test chunking

Use one long financial article and compare:

- `500/80`
- `300/50`
- `800/100`

Observe how answer quality changes after retrieval.

### Exercise 3: Swap in a real embedding model

Possible directions later:

- sentence-transformers
- bge series
- jina embeddings
- OpenAI embeddings

Your goal is not just to make it work. Your goal is to compare behavior against the hashing baseline.

This repository now includes a small comparison script:

```bash
python3 scripts/compare_embedding_backends.py --limit 5
```

It keeps the JPM corpus and eval slice fixed, runs the hashing baseline first, and then runs the configured remote embedding backend if it is available. That makes it easier to reason about whether a change in retrieval quality is really coming from the embedding layer.

### Exercise 4: Write 10 eval questions

For each question, inspect:

- retrieved chunks
- whether the answer is supported
- whether the answer misses an important point

This helps you learn failure analysis, which is one of the most valuable RAG skills.

### Exercise 5: Compare reference answers with retrieved evidence

Use `data/eval/eval_cases.json` and inspect:

- does the retrieved context actually support the reference answer?
- is the reference answer more abstract than the raw document evidence?
- does your system miss the right chunk, or retrieve it but answer poorly?

This exercise is especially useful because it teaches you not to treat every answer miss as "the model is bad." Very often, the retrieval side is the real issue.

## 8. How To Talk About This In Interviews

This project can be much stronger if you present it as a system design and iteration project, not just "I built a demo."

The current recommended positioning is:

> A RAG assistant for SEC 10-K annual report analysis, using JPMorgan Chase & Co.'s 2025 Form 10-K as the main demo corpus.

This is stronger than presenting it as a generic finance-news chatbot because the task boundary is clearer:

- the corpus is an authoritative company filing
- the user goal is financial research, not casual conversation
- citations can point back to filing sections and chunk titles
- eval questions can be written against concrete disclosed facts

The Sina Finance dataset is still useful, but it is better framed as secondary data for future multi-source expansion.

Good themes to highlight:

- "I separated retrieval and generation from the start."
- "I designed the data schema so metadata could flow through chunking into citations."
- "I used a runnable baseline first, then planned for better embeddings and FAISS."
- "I treated hallucination reduction as a system problem, not just a prompt problem."
- "I evaluated failures by checking retrieval quality before blaming the LLM."
- "I separated the article corpus and the QA eval set so the demo would remain honest."
- "I narrowed the main demo to a high-quality SEC filing corpus because focused retrieval is easier to evaluate and explain."
- "I added metadata-aware reranking using 10-K sections, semantic topics, and chunk quality labels."

If the interviewer asks what you would improve next, strong answers include:

- better embeddings
- reranking
- citation formatting
- eval benchmark
- hybrid retrieval
- domain-specific preprocessing for financial documents
- multi-filing comparison across companies or filing years

For JPM 10-K questions, good demo prompts include:

- What are JPMorgan Chase's reportable business segments?
- What are JPMorgan Chase's principal bank and non-bank subsidiaries?
- What regulatory risks does JPMorgan Chase disclose?
- What does the filing say about liquidity risk management?
- How does the filing describe cybersecurity or AI-related risks?

## 9. What Makes This A Good Interview Project

This project is a good interview project because it shows:

- practical LLM application ability
- system decomposition
- awareness of hallucination and grounding
- data pipeline thinking
- evaluation mindset

It also gives you something much more interesting to discuss than a simple chat UI.

## 10. What To Avoid

A few traps are common for first RAG projects:

- focusing on the model before validating retrieval
- using a fancy UI too early
- skipping eval and relying only on intuition
- mixing gold answers directly into the retrievable knowledge base
- not preserving metadata for citation
- stuffing too much text into context
- describing the system as "AI understands finance" instead of "the system retrieves grounded financial evidence"

That last point matters especially in interviews. Clear claims make you sound stronger, not weaker.

## 11. A Strong Project Mindset

A good way to think about this project:

- V1 proves the pipeline
- V2 improves retrieval quality
- V3 improves product quality and evaluation

That framing is realistic and mature.

You do not need to pretend this is a perfect production system. It is much better to show that you understand:

- what is already working
- what is intentionally simplified
- what you would improve next
- why those improvements matter

## 12. Recommended Next Move

The previous milestone was:

- add `llm`
- add `pipeline`
- define `answer_question(question)`
- enforce grounded prompting
- return `Answer + Sources + Evidence`

That milestone is now implemented.

The best next implementation step is:

- improve embeddings beyond the hashing baseline
- tighten evidence filtering so irrelevant top-k chunks do not leak into evidence
- add a JPM-specific eval report mode that highlights section/topic matches
- optionally add a second 10-K filing later for company-to-company comparison

That will make the project more stable as a live interview demo.

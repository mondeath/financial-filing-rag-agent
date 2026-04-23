# Embedding Comparison Report

This report compares embedding backends for the JPM 10-K RAG pipeline while keeping the corpus, reranking logic, and local grounded generator fixed.

## Setup

- corpus: `data/processed/jpm_2025_10k_chunks.jsonl`
- eval slice: first `5` cases from `data/eval/jpm_10k_eval_cases.json`
- answer generation: local grounded fallback
- retrieval logic: same metadata-aware reranking across all runs

This run should be read as a baseline experiment rather than a final benchmark. The goal is to isolate the effect of the embedding layer as cleanly as possible.

## Summary Table

| Backend | Status | Cases | Answered | Avg Lexical Overlap | Avg Sources | Avg Evidence | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hashing-baseline | ok | 5 | 5 | 0.588 | 1.00 | 1.00 | n/a |
| configured-remote-embedding | skipped | - | - | - | - | - | No remote embedding configuration found in environment. |

## Findings

- The hashing baseline is strong on direct factual lookups such as business segments and subsidiary structure.
- Performance drops on broader semantic questions such as competitive environment and risk-governance framing.
- The remote embedding run is intentionally marked as `skipped` rather than silently falling back, so this report stays honest about what was actually tested.

## Per-Case Snapshot

### jpm_q1

- `hashing-baseline`: overlap=`0.857`, top retrieved chunk=`Business segments`
- `configured-remote-embedding`: no result

### jpm_q2

- `hashing-baseline`: overlap=`0.900`, top retrieved chunk=`Subsidiaries and operating structure`
- `configured-remote-embedding`: no result

### jpm_q3

- `hashing-baseline`: overlap=`0.175`, top retrieved chunk=`Competitive environment`
- `configured-remote-embedding`: no result

### jpm_q4

- `hashing-baseline`: overlap=`0.316`, top retrieved chunk=`Risk governance`
- `configured-remote-embedding`: no result

### jpm_q5

- `hashing-baseline`: overlap=`0.692`, top retrieved chunk=`Liquidity risk management`
- `configured-remote-embedding`: no result

## Interpretation

- This comparison becomes more informative once a real embedding backend is configured and both runs succeed on the same eval slice.
- Lexical overlap is only a lightweight proxy. It should be interpreted together with retrieved chunk titles, groundedness, and failure cases.
- Even without the remote run, this report is useful because it documents the current hashing baseline before further retrieval upgrades.

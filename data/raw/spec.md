# SEC 10-K Chunking Specification

## Objective

Transform SEC 10-K filings into high-quality JSONL chunks for financial QA and RAG systems.

## 1. Scope

### 1.1 Included Sections

Only process the following sections:

- Item 1 — Business
- Item 1A — Risk Factors
- Item 7 — MD&A (Management’s Discussion and Analysis)

### 1.2 Excluded Content

Ignore the following content types:

- Exhibits (EX-xx)
- XML / XBRL files
- Images
- Website references
- SEC notices
- Cross-references (for example: see page 46)
- Table of contents
- Navigation fragments
- Empty sections

## 2. Output Format

Output must be in JSONL format.

- One line = one chunk
- Each chunk must be a valid JSON object

### 2.1 Required Schema

```json
{
  "id": "jpm_2025_10k_item7_0211",
  "company": "JPMorgan Chase & Co.",
  "doc_type": "10-K",
  "filing_date": "2026-02-13",
  "section": "Item 7 MD&A",
  "primary_topic": "financial_risk",
  "secondary_topic": "credit_risk",
  "chunk_type": "text",
  "quality": "high",
  "title": "Lending-related commitments and credit exposure",
  "text": "...",
  "source": "data/raw/jpm-20251231.htm"
}
```

### 2.2 Field Definitions

| Field | Definition |
| --- | --- |
| `id` | Unique chunk id |
| `company` | Company name |
| `doc_type` | Always `10-K` |
| `filing_date` | SEC filing date in `YYYY-MM-DD` |
| `section` | Original document section name |
| `primary_topic` | Top-level semantic topic |
| `secondary_topic` | More specific semantic topic |
| `chunk_type` | One of: `text`, `table_summary` |
| `quality` | One of: `high`, `medium`, `low` |
| `title` | Concise chunk title |
| `text` | Chunk content |
| `source` | Original raw file path |

## 3. Topic Taxonomy

### 3.1 Primary Topics

- `company_overview`
- `business_structure`
- `business_segment`
- `competition`
- `regulatory_risk`
- `financial_risk`
- `operational_risk`
- `performance_analysis`

### 3.2 Secondary Topics

#### Regulatory Risk

- `legal_risk`
- `supervision_and_compliance`
- `jurisdictional_regulation`
- `consumer_finance_regulation`
- `interchange_fee_regulation`
- `litigation_and_enforcement`
- `capital_regulation`

#### Financial Risk

- `credit_risk`
- `market_risk`
- `liquidity_risk`
- `funding_risk`
- `lending_commitments`
- `collateral_and_margin`

#### Operational Risk

- `model_risk`
- `model_governance`
- `technology_risk`
- `cyber_risk`

#### Performance Analysis

- `revenue_drivers`
- `expense_drivers`
- `balance_sheet_trends`
- `receivables`
- `segment_performance`

#### Business Structure

- `major_subsidiaries`
- `bank_subsidiaries`
- `international_structure`

#### Business Segment

- `segment_overview`
- `consumer_banking`
- `investment_banking`
- `asset_wealth_management`

## 4. Chunking Rules

### 4.1 Length Constraints

- Recommended length: 300–1200 characters
- If chunk length >1500, split it
- If chunk length <150, merge it or discard it

### 4.2 Core Principles

Each chunk must:

- Cover a single topic
- Be independently understandable
- Be able to answer a specific question
- Preserve factual meaning from the source text

### 4.3 Splitting Priority

Use the following splitting order:

1. Section headers
2. Subheadings
3. Bullet points
4. Paragraphs
5. Sentence-level splitting only as fallback

### 4.4 Light Normalization

Allowed:

- Reconstruct incomplete sentences caused by bad HTML extraction
- Split overly long compound sentences
- Remove boilerplate or obvious formatting noise

Not allowed:

- Add new facts
- Inject external knowledge
- Perform subjective summarization
- Rewrite the source into opinionated prose

## 5. Table Handling

### 5.1 Keep as `table_summary`

Convert the following tables into concise natural-language summaries:

- Segment revenue tables
- Capital summaries
- Liquidity summaries
- Risk exposure tables

Example:

```json
{
  "chunk_type": "table_summary",
  "text": "Revenue from CIB exceeds CCB, with the largest contribution coming from investment banking and markets-related activity."
}
```

### 5.2 Ignore

Discard the following tables:

- Legal tables
- Footnotes
- Entity lists
- Extremely granular accounting tables
- Tables with little retrieval value

## 6. Filtering Rules

Discard chunks if they are:

- Only titles
- Too short to be meaningful
- Not self-contained
- Pure references
- Formatting noise
- Navigation leftovers
- Empty or near-empty text

## 7. Quality Labels

### 7.1 High

Use `high` if the chunk is:

- About one clear topic
- Information-dense
- Fully self-contained
- Easy to interpret in isolation

### 7.2 Medium

Use `medium` if the chunk is:

- Mostly usable
- Somewhat dependent on nearby context
- Still retrieval-worthy

### 7.3 Low

Use `low` if the chunk is:

- Fragmented
- Low-value
- Hard to interpret
- Weak as standalone evidence

## 8. Retrieval and Reranking Strategy

### 8.1 Query Classification

```python
def classify_query(q):
    q = q.lower()

    if any(k in q for k in ["risk", "regulation", "compliance", "litigation"]):
        return {
            "sections": ["Item 1A Risk Factors", "Item 7 MD&A"],
            "topics": ["regulatory_risk", "financial_risk"]
        }

    if any(k in q for k in ["segment", "business", "subsidiary"]):
        return {
            "sections": ["Item 1 Business"],
            "topics": ["business_segment", "company_overview"]
        }

    if any(k in q for k in ["revenue", "expense", "balance", "cash flow", "liquidity"]):
        return {
            "sections": ["Item 7 MD&A"],
            "topics": ["performance_analysis"]
        }

    return {
        "sections": ["Item 7 MD&A", "Item 1 Business", "Item 1A Risk Factors"],
        "topics": []
    }
```

### 8.2 Scoring Formula

```text
final_score =
0.70 * embedding_score
+ 0.15 * section_bonus
+ 0.10 * topic_bonus
+ 0.05 * quality_bonus
```

### 8.3 Section Weighting

Risk Queries:

- Item 1A Risk Factors → +0.20
- Item 7 MD&A → +0.15
- Item 1 Business → +0.05

Business Queries:

- Item 1 Business → +0.20
- Item 7 MD&A → +0.10
- Item 1A Risk Factors → +0.05

### 8.4 Topic Weighting

- `primary_topic` match → +0.10
- `secondary_topic` match → +0.05

### 8.5 Quality Weighting

- `high` → +0.05
- `medium` → +0.02
- `low` → +0.00

## 9. Pipeline

The pipeline should follow this order:

1. HTML → raw text extraction
2. Section segmentation
3. Initial chunking based on document structure
4. Post-processing
   - filtering
   - merging
   - topic tagging
5. Quality scoring
6. JSONL output
7. Embedding and retrieval
8. Reranking

## 10. Key Principles

- `section` = document source location and must remain faithful to the original filing structure
- `topic` = semantic label for retrieval and reranking
- Retrieval should use embedding + reranking, not embedding alone
- The goal is not perfect chunking, but practically useful chunking

## 11. v1 Practical Goal

For v1, do not optimize for perfection.

Target:

- around 80% chunk correctness
- early retrieval validation
- fast iteration based on observed retrieval quality

## 12. Implementation Notes for Codex

When implementing this pipeline:

- keep section tied to the original filing section, not the inferred semantic topic
- allow `primary_topic` and `secondary_topic` to be assigned during post-processing
- prefer self-contained chunks over mechanically equal-sized chunks
- use structure first, sentence splitting last
- do not preserve useless formatting artifacts from HTML extraction
- prioritize retrieval usefulness over raw textual completeness

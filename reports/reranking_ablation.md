# Reranking Ablation Report

This report compares embedding-only retrieval against the V1.1 risk-aware metadata reranker.

## Summary

- cases: 12
- embedding_top1_section_hit_rate: 0.833
- reranked_top1_section_hit_rate: 1.000
- embedding_topk_section_hit_rate: 0.917
- reranked_topk_section_hit_rate: 1.000
- embedding_top1_topic_hit_rate: 0.667
- reranked_top1_topic_hit_rate: 1.000
- embedding_topk_topic_hit_rate: 0.917
- reranked_topk_topic_hit_rate: 1.000

## Per-Case Results

| Case | Query Type | Expected Sections | Expected Topics | Embedding Top-1 | Reranked Top-1 | Section Top-1 | Topic Top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| jpm_q1 | business_segment | Item 1 Business, Item 7 MD&A | business_segment, segment_overview | Firm business overview | Business segments | yes -> yes | yes -> yes |
| jpm_q2 | business_structure | Item 1 Business | business_structure, bank_subsidiaries, international_structure | Subsidiaries and operating structure | Subsidiaries and operating structure | yes -> yes | yes -> yes |
| jpm_q3 | business_competition | Item 1 Business | competition, competitive_environment | Competitive environment | Competitive environment | no -> yes | no -> yes |
| jpm_q4 | risk_regulatory | Item 1A Risk Factors, Item 1 Business | regulatory_risk, litigation_and_enforcement | Legal and regulatory risk | Legal and regulatory risk | yes -> yes | no -> yes |
| jpm_q5 | risk_liquidity | Item 7 MD&A | liquidity_risk | Liquidity risk management | Liquidity risk management | yes -> yes | no -> yes |
| jpm_q6 | risk_technology | Item 1A Risk Factors | cyber_risk, technology_risk, operational_risk | Cybersecurity risk | Cybersecurity risk | yes -> yes | yes -> yes |
| jpm_q7 | risk_governance | Item 7 MD&A | risk_governance, financial_risk | Risk governance | Risk governance | yes -> yes | yes -> yes |
| jpm_q8 | risk_governance | Item 7 MD&A | risk_governance, financial_risk | Risk governance | Risk governance | yes -> yes | yes -> yes |
| jpm_q9 | risk_market | Item 7 MD&A | market_risk | Market risk management | Market risk management | yes -> yes | yes -> yes |
| jpm_q10 | risk_capital | Item 7 MD&A, Item 1 Business | capital_regulation, regulatory_risk, financial_risk | Regulatory capital requirements | Financial holding company regulation | yes -> yes | yes -> yes |
| jpm_q11 | risk_credit | Item 7 MD&A | credit_risk | Market risk management | Credit risk management | no -> yes | no -> yes |
| jpm_q12 | risk_ai | Item 1A Risk Factors | cyber_risk, technology_risk, operational_risk | Cybersecurity risk | Cybersecurity risk | yes -> yes | yes -> yes |

## Interpretation

- Section hit rates show whether reranking moves candidates into the expected SEC filing area.
- Topic hit rates show whether reranking improves alignment with curated chunk labels.
- A top-k hit with a top-1 miss means the answer may still be recoverable, but ranking or generation can choose a weaker passage.

# Sample Eval Report

This file captures a small example evaluation run for the main JPMorgan 10-K demo corpus.

Command used:

```bash
python3 main.py eval --cases data/eval/jpm_10k_eval_cases.json --limit 3
```

Environment notes:

- corpus: `data/processed/jpm_2025_10k_chunks.jsonl`
- index backend: local simple backend
- embedding backend: local hashing baseline in the current environment
- answer generation: local grounded fallback unless a real LLM is configured

## Summary

- cases: `3`
- answered: `3`
- insufficient information: `0`
- avg lexical overlap vs reference: `0.644`
- avg sources per answer: `1.00`
- avg evidence items per answer: `1.00`

## Case 1

Question:

`What are JPMorgan Chase's reportable business segments?`

System answer:

`Business segments: For management reporting purposes, the Firm has three reportable business segments - Consumer & Community Banking ("CCB"), Commercial & Investment Bank ("CIB") and Asset & Wealth Management ("AWM") - with the remaining activities in Corporate. The Firm's consumer business segment is CCB, and the Firm's wholesale business segments are CIB and AWM.`

Reference answer:

`JPMorgan Chase reports three business segments for management reporting purposes: Consumer & Community Banking (CCB), Commercial & Investment Bank (CIB), and Asset & Wealth Management (AWM), with remaining activities reported in Corporate.`

Lexical overlap: `0.857`

Top retrieved chunk:

- score=`0.3436`, embedding=`0.4302`, section=`Item 1 Business`, primary_topic=`business_segment`, secondary_topic=`segment_overview`, quality=`high`, title=`Business segments`

Takeaway:

This is a strong retrieval-and-answer case. The top chunk matches the expected section and topic, and the generated answer is close to the reference.

## Case 2

Question:

`What are JPMorgan Chase's principal bank and non-bank subsidiaries?`

System answer:

`Subsidiaries and operating structure: JPMorganChase's principal bank subsidiary is JPMorgan Chase Bank, National Association ("JPMorgan Chase Bank, N.A."), a national banking association with U.S. branches in 48 states and Washington, D.C. JPMorganChase's principal non-bank subsidiary is J.P. Morgan Securities LLC ("J.P. Morgan Securities"), a U.S. broker-dealer. The bank and non-bank subsidiaries of JPMorganChase operate nationally as well as through overseas branches and subsidiaries, representative offices and subsidiary foreign banks. The Firm's principal operating subsidiaries outside the U.S. are J.P. Morgan Securities plc and J.P. Morgan SE ("JPMSE"), which are subsidiaries of JPMorgan Chase Bank, N.A. and are based in the United Kingdom ("U.K.") and Germany, respectively.`

Reference answer:

`The principal bank subsidiary is JPMorgan Chase Bank, National Association. The principal non-bank subsidiary is J.P. Morgan Securities LLC, a U.S. broker-dealer. The filing also identifies J.P. Morgan Securities plc and J.P. Morgan SE as principal operating subsidiaries outside the U.S.`

Lexical overlap: `0.900`

Top retrieved chunk:

- score=`0.4613`, embedding=`0.6233`, section=`Item 1 Business`, primary_topic=`business_structure`, secondary_topic=`international_structure`, quality=`high`, title=`Subsidiaries and operating structure`

Takeaway:

This is another strong case. The answer is grounded in the correct filing section and includes the same key entities as the reference answer.

## Case 3

Question:

`How does JPMorgan Chase describe its competitive environment?`

System answer:

`Competitive environment: Furthermore, both financial institutions and their non-banking competitors face the risk of disruption to payments processing and other products and services from the use of new technologies that may not require intermediation, such as tokenized securities or other products that leverage distributed ledger technology. New technologies have required and could require JPMorganChase to increase expenditures to modify its products to attract and retain clients and customers or to match products and services offered by its competitors, including technology companies. If JPMorganChase does not keep pace with rapidly changing technological advances, including the adoption of generative AI, it risks losing clients and market share to competitors, which could negatively impact revenues, operating costs and its competitive position. Competition could be intensified as the feasibility, capability and scalability of new technologies improves. In addition, new technologies (including generative AI) could be used by customers or bad actors in unexpected or disruptive ways, or could be breached or infiltrated by third parties, which could increase JPMorganChase's compliance expenses and reduce its income related to the offering of products and services through those technologies.`

Reference answer:

`JPMorgan Chase operates in highly competitive environments against banks, brokerage firms, investment banking companies, hedge funds, private equity firms, asset managers, credit card companies, fintech and internet-based competitors, among others. It competes on product and service quality, variety, execution, innovation, reputation, and price.`

Lexical overlap: `0.175`

Top retrieved chunk:

- score=`0.1983`, embedding=`0.2711`, section=`Item 1A Risk Factors`, primary_topic=`operational_risk`, secondary_topic=`technology_risk`, quality=`medium`, title=`Competitive environment`

Takeaway:

This is a useful failure case. The system retrieved a filing passage about technology-driven competitive risk rather than the broader competitive landscape summarized in the reference answer. For interview discussion, this example helps show why retrieval quality, chunk design, and evaluation matter.

## Notes

- This sample report is intentionally small and qualitative.
- The project includes a fuller eval command so results can be regenerated locally.
- If a real embedding API and real LLM are configured, answer quality may improve without changing the CLI surface.

# Embedding Comparison Report

This report compares the JPM 10-K RAG pipeline across embedding backends while keeping the corpus, reranking logic, and local grounded generator fixed.

## Summary Table

| Backend | Status | Cases | Answered | Avg Lexical Overlap | Avg Sources | Avg Evidence | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hashing-baseline | ok | 5 | 5 | 0.588 | 1.00 | 1.00 | n/a |
| local:5c38ec7c405ec4b44b94cc5a9bb96e735b38267a (dim=384) | ok | 5 | 5 | 0.518 | 1.00 | 1.00 | n/a |
| configured-remote-embedding | skipped | - | - | - | - | - | No remote embedding configuration found in environment. |

## Per-Case Snapshot

### jpm_q1

- `hashing-baseline`: overlap=0.857, top_title=Business segments, answer=Business segments: For management reporting purposes, the Firm has three reportable business segments - Consumer & Community Banking ("CCB"), Commercial & Investment Bank ("CIB") and Asset & Wealth Management ("AWM") - with the remaining activities in Corporate. The Firm's consumer business segment is CCB, and the Firm's wholesale business segments are CIB and AWM.
- `local:5c38ec7c405ec4b44b94cc5a9bb96e735b38267a (dim=384)`: overlap=0.857, top_title=Firm business overview, answer=Business segments: For management reporting purposes, the Firm has three reportable business segments - Consumer & Community Banking ("CCB"), Commercial & Investment Bank ("CIB") and Asset & Wealth Management ("AWM") - with the remaining activities in Corporate. The Firm's consumer business segment is CCB, and the Firm's wholesale business segments are CIB and AWM.
- `configured-remote-embedding`: no result

### jpm_q2

- `hashing-baseline`: overlap=0.900, top_title=Subsidiaries and operating structure, answer=Subsidiaries and operating structure: JPMorganChase's principal bank subsidiary is JPMorgan Chase Bank, National Association ("JPMorgan Chase Bank, N.A."), a national banking association with U.S. branches in 48 states and Washington, D.C. JPMorganChase's principal non-bank subsidiary is J.P. Morgan Securities LLC ("J.P. Morgan Securities"), a U.S. broker-dealer. The bank and non-bank subsidiaries of JPMorganChase operate nationally as well as through overseas branches and subsidiaries, representative offices and subsidiary foreign banks. The Firm's principal operating subsidiaries outside the U.S. are J.P. Morgan Securities plc and J.P. Morgan SE ("JPMSE"), which are subsidiaries of JPMorgan Chase Bank, N.A. and are based in the United Kingdom ("U.K.") and Germany, respectively.
- `local:5c38ec7c405ec4b44b94cc5a9bb96e735b38267a (dim=384)`: overlap=0.900, top_title=Subsidiaries and operating structure, answer=Subsidiaries and operating structure: JPMorganChase's principal bank subsidiary is JPMorgan Chase Bank, National Association ("JPMorgan Chase Bank, N.A."), a national banking association with U.S. branches in 48 states and Washington, D.C. JPMorganChase's principal non-bank subsidiary is J.P. Morgan Securities LLC ("J.P. Morgan Securities"), a U.S. broker-dealer. The bank and non-bank subsidiaries of JPMorganChase operate nationally as well as through overseas branches and subsidiaries, representative offices and subsidiary foreign banks. The Firm's principal operating subsidiaries outside the U.S. are J.P. Morgan Securities plc and J.P. Morgan SE ("JPMSE"), which are subsidiaries of JPMorgan Chase Bank, N.A. and are based in the United Kingdom ("U.K.") and Germany, respectively.
- `configured-remote-embedding`: no result

### jpm_q3

- `hashing-baseline`: overlap=0.175, top_title=Competitive environment, answer=Competitive environment: Furthermore, both financial institutions and their non-banking competitors face the risk of disruption to payments processing and other products and services from the use of new technologies that may not require intermediation, such as tokenized securities or other products that leverage distributed ledger technology. New technologies have required and could require JPMorganChase to increase expenditures to modify its products to attract and retain clients and customers or to match products and services offered by its competitors, including technology companies. If JPMorganChase does not keep pace with rapidly changing technological advances, including the adoption of generative AI, it risks losing clients and market share to competitors, which could negatively impact revenues, operating costs and its competitive position. Competition could be intensified as the feasibility, capability and scalability of new technologies improves. In addition, new technologies (including generative AI) could be used by customers or bad actors in unexpected or disruptive ways, or could be breached or infiltrated by third parties, which could increase JPMorganChase's compliance expenses and reduce its income related to the offering of products and services through those technologies.
- `local:5c38ec7c405ec4b44b94cc5a9bb96e735b38267a (dim=384)`: overlap=0.325, top_title=Competitive environment, answer=Competitive environment: JPMorganChase operates in a highly competitive environment in which it must constantly adapt to changes in financial regulation, technological advances and economic conditions. JPMorganChase expects that competition in the financial services industry will remain intense, with new competitors in the financial services industry continuing to emerge. For example, technological advances and the growth of e-commerce have made it possible for non-depository institutions to offer products and services that traditionally were banking products. These advances have also allowed financial institutions and other companies to provide electronic and internet-based financial solutions, including: • lending and other extensions of credit to consumers • payments processing • cryptocurrency, including stablecoins • tokenized securities, and • online automated algorithmic-based investment advice.
- `configured-remote-embedding`: no result

### jpm_q4

- `hashing-baseline`: overlap=0.316, top_title=Risk governance, answer=Risk governance: The Firm's risk governance framework involves understanding drivers of risks, types of risks and impacts of risks. Drivers of risks are factors that cause a risk to exist. Drivers of risks include the economic environment, regulatory or government policy, competitor or market evolution, business decisions, process or judgment error, deliberate wrongdoing, dysfunctional markets and natural disasters. Types of risks are categories by which risks manifest themselves. The Firm's risks are generally categorized in the following four risk types: • Strategic risk is the risk to earnings, capital, liquidity or reputation associated with poorly-designed or failed business plans or an inadequate response to changes in the operating environment. • Credit and investment risk is the risk associated with the default or change in credit profile of a client, counterparty or customer;
- `local:5c38ec7c405ec4b44b94cc5a9bb96e735b38267a (dim=384)`: overlap=0.316, top_title=Legal and regulatory risk, answer=Legal and regulatory risk: • certain clients and customers ceasing to do business with JPMorganChase, and encouraging others to do so • impairment of JPMorganChase's ability to attract new clients and customers, to expand its relationships with existing clients and customers, or to hire or retain employees, or • certain investors opting to divest from investments in securities of JPMorganChase. Failure to effectively manage potential conflicts of interest or to satisfy fiduciary obligations could result in litigation and enforcement actions and cause reputational harm. Managing potential conflicts of interest is highly complex for JPMorganChase due to its broad range of business activities which encompass a variety of transactions, obligations and interests with and among clients and customers. JPMorganChase could face litigation, enforcement actions and heightened regulatory scrutiny, and its reputation could be damaged, by the failure or perceived failure to: • adequately address or appropriately disclose actual or potential conflicts of interest, including those that may arise in connection with providing multiple products and services in, or having investments related to, the same transaction
- `configured-remote-embedding`: no result

### jpm_q5

- `hashing-baseline`: overlap=0.692, top_title=Liquidity risk management, answer=Liquidity risk management: The Firm's Contingency Funding Plan ("CFP") sets out the strategies for addressing and managing liquidity resource needs during a liquidity stress event and incorporates liquidity risk limits, indicators and risk appetite tolerances. The CFP also identifies the alternative contingent funding and liquidity resources available to the Firm and its legal entities in a period of stress.
- `local:5c38ec7c405ec4b44b94cc5a9bb96e735b38267a (dim=384)`: overlap=0.192, top_title=Liquidity risk management, answer=Liquidity risk management: • Liquidity risks, including the risk that JPMorganChase's ability to operate could be impaired by constrained liquidity;
- `configured-remote-embedding`: no result

## Interpretation

- This comparison is most useful when a stronger embedding backend such as local BGE or a remote embedding API is available on the same eval slice.
- The lexical overlap metric is lightweight and should be read together with retrieved chunk titles and answer groundedness.
- If stronger embedding backends are unavailable, the hashing baseline still provides a reproducible local reference point.

Eval Summary:
- cases: 12
- answered: 12
- insufficient_information: 0
- avg_lexical_overlap_vs_reference: 0.452
- avg_sources_per_answer: 1.17
- avg_evidence_items_per_answer: 1.17

Manual Evaluation Notes:
- Overall quality: acceptable V1 baseline. The pipeline runs end-to-end with BGE embeddings, retrieves filing-backed evidence, and avoids unsupported free-form claims.
- Lexical overlap: 0.452 is modest. This is partly because the local fallback generator copies long filing passages instead of producing concise reference-style summaries, so wording often diverges even when evidence is relevant.
- Retrieval quality: mixed. Business-structure and AI-risk questions retrieve strong evidence. Risk-governance, liquidity, and credit-risk questions retrieve broadly related sections but often miss the most directly explanatory chunk.
- Main bottleneck: generation is extractive and sentence-scoring based. It is grounded, but it tends to choose the highest-overlap passage rather than synthesize the best answer from multiple retrieved chunks.
- Recommended next step: keep this as the V1 baseline, then improve reranking/query classification for risk topics and optionally use an LLM generator for concise synthesis.

[jpm_q1] What are JPMorgan Chase's reportable business segments?
System Answer: Business segments: For management reporting purposes, the Firm has three reportable business segments - Consumer & Community Banking ("CCB"), Commercial & Investment Bank ("CIB") and Asset & Wealth Management ("AWM") - with the remaining activities in Corporate. The Firm's consumer business segment is CCB, and the Firm's wholesale business segments are CIB and AWM.
Reference Answer: JPMorgan Chase reports three business segments for management reporting purposes: Consumer & Community Banking (CCB), Commercial & Investment Bank (CIB), and Asset & Wealth Management (AWM), with remaining activities reported in Corporate.
Sources: Business segments (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: Business segments: For management reporting purposes, the Firm has three reportable business segments - Consumer & Community Banking ("CCB"), Commercial & Investment Bank ("CIB") and Asset & Wealth Management ("AWM") - with the remaining activities in Corporate. The Firm's consumer business segment is CCB, and the Firm's wholesale business segments are CIB and AWM.
Lexical Overlap: 0.857
Retrieved Chunks:
- 1. score=0.5879, embedding=0.8005, section=Item 7 MD&A, primary_topic=business_segment, secondary_topic=consumer_banking, quality=high, title=Firm business overview
- 2. score=0.5858, embedding=0.7761, section=Item 1 Business, primary_topic=company_overview, secondary_topic=company_profile, quality=high, title=Firm business overview
- 3. score=0.5746, embedding=0.7602, section=Item 1 Business, primary_topic=business_segment, secondary_topic=segment_overview, quality=high, title=Business segments
- 4. score=0.5693, embedding=0.7740, section=Item 7 MD&A, primary_topic=business_segment, secondary_topic=consumer_banking, quality=high, title=Subsidiaries and operating structure
Manual Review:
- retrieval_relevance: High. The correct business-segment chunk appears in the retrieved set, although it ranks third behind related overview chunks.
- groundedness: High. The answer is directly supported by the cited evidence.
- hallucination: No. The answer stays within the filing context.
- completeness: High. It includes CCB, CIB, AWM, and Corporate treatment.

[jpm_q2] What are JPMorgan Chase's principal bank and non-bank subsidiaries?
System Answer: Subsidiaries and operating structure: JPMorganChase's principal bank subsidiary is JPMorgan Chase Bank, National Association ("JPMorgan Chase Bank, N.A."), a national banking association with U.S. branches in 48 states and Washington, D.C. JPMorganChase's principal non-bank subsidiary is J.P. Morgan Securities LLC ("J.P. Morgan Securities"), a U.S. broker-dealer. The bank and non-bank subsidiaries of JPMorganChase operate nationally as well as through overseas branches and subsidiaries, representative offices and subsidiary foreign banks. The Firm's principal operating subsidiaries outside the U.S. are J.P. Morgan Securities plc and J.P. Morgan SE ("JPMSE"), which are subsidiaries of JPMorgan Chase Bank, N.A. and are based in the United Kingdom ("U.K.") and Germany, respectively.
Reference Answer: The principal bank subsidiary is JPMorgan Chase Bank, National Association. The principal non-bank subsidiary is J.P. Morgan Securities LLC, a U.S. broker-dealer. The filing also identifies J.P. Morgan Securities plc and J.P. Morgan SE as principal operating subsidiaries outside the U.S.
Sources: Subsidiaries and operating structure (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: Subsidiaries and operating structure: JPMorganChase's principal bank subsidiary is JPMorgan Chase Bank, National Association ("JPMorgan Chase Bank, N.A."), a national banking association with U.S. branches in 48 states and Washington, D.C. JPMorganChase's principal non-bank subsidiary is J.P. Morgan Securities LLC ("J.P. Morgan Securities"), a U.S. broker-dealer. The bank and non-bank subsidiaries of JPMorganChase operate nationally as well as through overseas branches and subsidiaries, representative offices and subsidiary foreign banks. The Firm's principal operating subsidiaries outside the U.S. are J.P. Morgan Securities plc and J.P. Morgan SE ("JPMSE"), which are subsidiaries of JPMorgan Chase Bank, N.A. and are based in the United Kingdom ("U.K.") and Germany, respectively.
Lexical Overlap: 0.900
Retrieved Chunks:
- 1. score=0.6393, embedding=0.8775, section=Item 1 Business, primary_topic=business_structure, secondary_topic=international_structure, quality=high, title=Subsidiaries and operating structure
- 2. score=0.6310, embedding=0.8657, section=Item 7 MD&A, primary_topic=business_segment, secondary_topic=consumer_banking, quality=high, title=Subsidiaries and operating structure
- 3. score=0.5642, embedding=0.7702, section=Item 1 Business, primary_topic=business_structure, secondary_topic=bank_subsidiaries, quality=high, title=Bank subsidiary regulation
- 4. score=0.5629, embedding=0.7685, section=Item 7 MD&A, primary_topic=business_segment, secondary_topic=consumer_banking, quality=high, title=Firm business overview
Manual Review:
- retrieval_relevance: High. The top retrieved chunk directly addresses principal bank and non-bank subsidiaries.
- groundedness: High. The answer is copied from the relevant source passage.
- hallucination: No. The named subsidiaries are present in the evidence.
- completeness: High. It covers JPMorgan Chase Bank, N.A., J.P. Morgan Securities LLC, and key non-U.S. operating subsidiaries.

[jpm_q3] How does JPMorgan Chase describe its competitive environment?
System Answer: Competitive environment: JPMorganChase operates in a highly competitive environment in which it must constantly adapt to changes in financial regulation, technological advances and economic conditions. JPMorganChase expects that competition in the financial services industry will remain intense, with new competitors in the financial services industry continuing to emerge. For example, technological advances and the growth of e-commerce have made it possible for non-depository institutions to offer products and services that traditionally were banking products. These advances have also allowed financial institutions and other companies to provide electronic and internet-based financial solutions, including: • lending and other extensions of credit to consumers • payments processing • cryptocurrency, including stablecoins • tokenized securities, and • online automated algorithmic-based investment advice.
Reference Answer: JPMorgan Chase operates in highly competitive environments against banks, brokerage firms, investment banking companies, hedge funds, private equity firms, asset managers, credit card companies, fintech and internet-based competitors, among others. It competes on product and service quality, variety, execution, innovation, reputation, and price.
Sources: Competitive environment (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: Competitive environment: JPMorganChase operates in a highly competitive environment in which it must constantly adapt to changes in financial regulation, technological advances and economic conditions. JPMorganChase expects that competition in the financial services industry will remain intense, with new competitors in the financial services industry continuing to emerge. For example, technological advances and the growth of e-commerce have made it possible for non-depository institutions to offer products and services that traditionally were banking products. These advances have also allowed financial institutions and other companies to provide electronic and internet-based financial solutions, including: • lending and other extensions of credit to consumers • payments processing • cryptocurrency, including stablecoins • tokenized securities, and • online automated algorithmic-based investment advice.
Lexical Overlap: 0.325
Retrieved Chunks:
- 1. score=0.6478, embedding=0.8668, section=Item 1 Business, primary_topic=competition, secondary_topic=competitive_environment, quality=medium, title=Competitive environment
- 2. score=0.6209, embedding=0.8727, section=Item 1A Risk Factors, primary_topic=financial_risk, secondary_topic=funding_risk, quality=high, title=Competitive environment
- 3. score=0.5744, embedding=0.7598, section=Item 1 Business, primary_topic=company_overview, secondary_topic=company_profile, quality=high, title=Firm business overview
- 4. score=0.5737, embedding=0.7802, section=Item 7 MD&A, primary_topic=business_segment, secondary_topic=consumer_banking, quality=high, title=Firm business overview
Manual Review:
- retrieval_relevance: Medium-high. The top result is the right Item 1 competition chunk, but the retrieved set also includes less central risk-factor chunks.
- groundedness: High. The answer is supported by the cited competitive environment passage.
- hallucination: No. The answer does not introduce unsupported competitors or claims.
- completeness: Medium. It captures technology-driven competition but misses some reference details about competitor categories and competition factors such as quality, execution, reputation, and price.

[jpm_q4] What regulatory risks does JPMorgan Chase disclose?
System Answer: Legal and regulatory risk: • certain clients and customers ceasing to do business with JPMorganChase, and encouraging others to do so • impairment of JPMorganChase's ability to attract new clients and customers, to expand its relationships with existing clients and customers, or to hire or retain employees, or • certain investors opting to divest from investments in securities of JPMorganChase. Failure to effectively manage potential conflicts of interest or to satisfy fiduciary obligations could result in litigation and enforcement actions and cause reputational harm. Managing potential conflicts of interest is highly complex for JPMorganChase due to its broad range of business activities which encompass a variety of transactions, obligations and interests with and among clients and customers. JPMorganChase could face litigation, enforcement actions and heightened regulatory scrutiny, and its reputation could be damaged, by the failure or perceived failure to: • adequately address or appropriately disclose actual or potential conflicts of interest, including those that may arise in connection with providing multiple products and services in, or having investments related to, the same transaction
Reference Answer: The filing states that JPMorgan Chase is subject to extensive regulation under U.S. federal and state laws and laws in non-U.S. jurisdictions. Regulatory challenges, litigation, supervision, compliance requirements, and changes in applicable rules may affect the scope, timing, or requirements of regulations applicable to the Firm.
Sources: Legal and regulatory risk (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: Legal and regulatory risk: • certain clients and customers ceasing to do business with JPMorganChase, and encouraging others to do so • impairment of JPMorganChase's ability to attract new clients and customers, to expand its relationships with existing clients and customers, or to hire or retain employees, or • certain investors opting to divest from investments in securities of JPMorganChase. Failure to effectively manage potential conflicts of interest or to satisfy fiduciary obligations could result in litigation and enforcement actions and cause reputational harm. Managing potential conflicts of interest is highly complex for JPMorganChase due to its broad range of business activities which encompass a variety of transactions, obligations and interests with and among clients and customers. JPMorganChase could face litigation, enforcement actions and heightened regulatory scrutiny, and its reputation could be damaged, by the failure or perceived failure to: • adequately address or appropriately disclose actual or potential conflicts of interest, including those that may arise in connection with providing multiple products and services in, or having investments related to, the same transaction
Lexical Overlap: 0.316
Retrieved Chunks:
- 1. score=0.6209, embedding=0.8285, section=Item 1A Risk Factors, primary_topic=regulatory_risk, secondary_topic=litigation_and_enforcement, quality=medium, title=Legal and regulatory risk
- 2. score=0.6177, embedding=0.8217, section=Item 1A Risk Factors, primary_topic=regulatory_risk, secondary_topic=litigation_and_enforcement, quality=high, title=Legal and regulatory risk
- 3. score=0.6172, embedding=0.8374, section=Item 1A Risk Factors, primary_topic=operational_risk, secondary_topic=technology_risk, quality=medium, title=Legal and regulatory risk
- 4. score=0.6154, embedding=0.8184, section=Item 1A Risk Factors, primary_topic=regulatory_risk, secondary_topic=litigation_and_enforcement, quality=high, title=Legal and regulatory risk
Manual Review:
- retrieval_relevance: Medium. The retrieved chunks are regulatory/legal risk chunks, but the selected answer is narrower than the reference and focuses on conflicts, litigation, and reputational effects.
- groundedness: High. The answer is supported by the cited legal and regulatory risk evidence.
- hallucination: No. The content is filing-grounded.
- completeness: Medium-low. It does not clearly summarize broad regulatory supervision, applicable laws, compliance obligations, and changing rules.

[jpm_q5] What does the filing say about liquidity risk management?
System Answer: Liquidity risk management: • Liquidity risks, including the risk that JPMorganChase's ability to operate could be impaired by constrained liquidity;
Reference Answer: The filing describes liquidity risk management through governance and planning mechanisms including the Contingency Funding Plan, liquidity risk limits, indicators, risk appetite tolerances, and alternative contingent funding and liquidity resources available during stress events.
Sources: Liquidity risk management (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: Liquidity risk management: • Liquidity risks, including the risk that JPMorganChase's ability to operate could be impaired by constrained liquidity;
Lexical Overlap: 0.192
Retrieved Chunks:
- 1. score=0.5911, embedding=0.7837, section=Item 1A Risk Factors, primary_topic=financial_risk, secondary_topic=liquidity_risk, quality=high, title=Liquidity risk management
- 2. score=0.5906, embedding=0.7936, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=liquidity_risk, quality=high, title=Liquidity risk management
- 3. score=0.5905, embedding=0.8078, section=Item 7 MD&A, primary_topic=performance_analysis, secondary_topic=balance_sheet_trends, quality=high, title=Liquidity risk management
- 4. score=0.5865, embedding=0.7772, section=Item 1A Risk Factors, primary_topic=financial_risk, secondary_topic=liquidity_risk, quality=high, title=Liquidity risk management
Manual Review:
- retrieval_relevance: Medium. The retrieved chunks are labeled liquidity risk management, but the top selected evidence is a generic liquidity-risk bullet rather than the stronger CFP/governance passage.
- groundedness: High. The short answer is supported by the evidence.
- hallucination: No. The answer stays inside the retrieved text.
- completeness: Low. It misses the Contingency Funding Plan, liquidity limits, indicators, risk appetite tolerances, and contingent funding resources.

[jpm_q6] How does JPMorgan Chase describe cybersecurity or technology risk?
System Answer: Cybersecurity risk: • increased volume and complexity of cyber attacks during periods of heightened geopolitical tensions • technological advances such as artificial intelligence ("AI") and quantum computing that may enable malicious actors to develop more advanced social engineering attacks, including targeted phishing attacks, and • technological advances which may counteract or nullify existing information security protections, including cryptographic protections, potentially exposing data. In addition, JPMorganChase could be required to make significant investments in technology in order to transition effectively to more robust security protections, including quantum-resistant encryption. Any such transition may not be completed before relevant threats become operational, and JPMorganChase's interconnectedness with third parties who may be slower to adopt such protections could further increase its vulnerability to data compromise. Furthermore, a third-party could misappropriate confidential information obtained by intercepting signals or communications from mobile devices used by JPMorganChase's employees.
Reference Answer: The filing discloses that cybersecurity and technology risks, including risks from advanced technologies such as AI, could disrupt operations, cause erroneous transactions, compromise data privacy, infringe intellectual property, harm clients and customers, or impair business decision-making.
Sources: Cybersecurity risk (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: Cybersecurity risk: • increased volume and complexity of cyber attacks during periods of heightened geopolitical tensions • technological advances such as artificial intelligence ("AI") and quantum computing that may enable malicious actors to develop more advanced social engineering attacks, including targeted phishing attacks, and • technological advances which may counteract or nullify existing information security protections, including cryptographic protections, potentially exposing data. In addition, JPMorganChase could be required to make significant investments in technology in order to transition effectively to more robust security protections, including quantum-resistant encryption. Any such transition may not be completed before relevant threats become operational, and JPMorganChase's interconnectedness with third parties who may be slower to adopt such protections could further increase its vulnerability to data compromise. Furthermore, a third-party could misappropriate confidential information obtained by intercepting signals or communications from mobile devices used by JPMorganChase's employees.
Lexical Overlap: 0.400
Retrieved Chunks:
- 1. score=0.6294, embedding=0.8528, section=Item 1A Risk Factors, primary_topic=operational_risk, secondary_topic=cyber_risk, quality=high, title=Cybersecurity risk
- 2. score=0.6228, embedding=0.8433, section=Item 1A Risk Factors, primary_topic=operational_risk, secondary_topic=cyber_risk, quality=high, title=Cybersecurity risk
- 3. score=0.6227, embedding=0.8432, section=Item 1A Risk Factors, primary_topic=operational_risk, secondary_topic=cyber_risk, quality=high, title=Cybersecurity risk
- 4. score=0.6216, embedding=0.8415, section=Item 1A Risk Factors, primary_topic=operational_risk, secondary_topic=cyber_risk, quality=high, title=Cybersecurity risk
Manual Review:
- retrieval_relevance: High. The retrieved chunks are consistently cybersecurity-risk chunks.
- groundedness: High. The answer is directly grounded in the cited evidence.
- hallucination: No. The technology and AI-related cyber-risk claims are supported.
- completeness: Medium. It covers cyber attacks and advanced technologies, but it is less complete than the reference on business disruption, erroneous transactions, privacy, IP, client harm, and decision-making impairment.

[jpm_q7] What does the filing say about risk governance?
System Answer: Risk governance: • Market risk is the risk associated with the effect of changes in market factors, such as interest and foreign exchange rates, equity and commodity prices, credit spreads or implied volatilities, on the value of assets and liabilities held for both the short and long term. • Operational risk is the risk of an adverse outcome resulting from inadequate or failed internal processes or systems;
Reference Answer: JPMorgan Chase describes a Firmwide risk governance framework involving risk drivers, risk types, and risk impacts. Independent Risk Management includes Risk Management and Compliance, and the Chief Risk Officer leads the function subject to Board Risk Committee approval.
Sources: Risk governance (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: Risk governance: • Market risk is the risk associated with the effect of changes in market factors, such as interest and foreign exchange rates, equity and commodity prices, credit spreads or implied volatilities, on the value of assets and liabilities held for both the short and long term. • Operational risk is the risk of an adverse outcome resulting from inadequate or failed internal processes or systems;
Lexical Overlap: 0.148
Retrieved Chunks:
- 1. score=0.5642, embedding=0.7559, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=liquidity_risk, quality=high, title=Risk governance
- 2. score=0.5557, embedding=0.7438, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=market_risk, quality=high, title=Risk governance
- 3. score=0.5500, embedding=0.7357, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=liquidity_risk, quality=high, title=Risk governance
- 4. score=0.5448, embedding=0.7283, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=market_risk, quality=high, title=Risk governance
Manual Review:
- retrieval_relevance: Medium-low. The chunks are titled Risk governance, but the selected passages emphasize risk categories rather than governance framework, Independent Risk Management, CRO, and Board oversight.
- groundedness: High. The answer is supported by the cited evidence.
- hallucination: No. There are no unsupported claims.
- completeness: Low. It does not answer the governance part of the question well; this is one of the clearest retrieval/generation misses.

[jpm_q8] What are the main categories of risk described in the filing's risk governance discussion?
System Answer: Risk governance: The Firm's risk governance framework involves understanding drivers of risks, types of risks and impacts of risks. Drivers of risks are factors that cause a risk to exist. Drivers of risks include the economic environment, regulatory or government policy, competitor or market evolution, business decisions, process or judgment error, deliberate wrongdoing, dysfunctional markets and natural disasters. Types of risks are categories by which risks manifest themselves. The Firm's risks are generally categorized in the following four risk types: • Strategic risk is the risk to earnings, capital, liquidity or reputation associated with poorly-designed or failed business plans or an inadequate response to changes in the operating environment. • Credit and investment risk is the risk associated with the default or change in credit profile of a client, counterparty or customer;
Reference Answer: The filing categorizes risks into broad risk types including strategic risk, credit and investment risk, liquidity risk, market risk, operational risk, compliance risk, conduct risk, legal risk, reputation risk, and model risk, depending on the relevant discussion.
Sources: Risk governance (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: Risk governance: The Firm's risk governance framework involves understanding drivers of risks, types of risks and impacts of risks. Drivers of risks are factors that cause a risk to exist. Drivers of risks include the economic environment, regulatory or government policy, competitor or market evolution, business decisions, process or judgment error, deliberate wrongdoing, dysfunctional markets and natural disasters. Types of risks are categories by which risks manifest themselves. The Firm's risks are generally categorized in the following four risk types: • Strategic risk is the risk to earnings, capital, liquidity or reputation associated with poorly-designed or failed business plans or an inadequate response to changes in the operating environment. • Credit and investment risk is the risk associated with the default or change in credit profile of a client, counterparty or customer;
Lexical Overlap: 0.440
Retrieved Chunks:
- 1. score=0.5978, embedding=0.8040, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=liquidity_risk, quality=high, title=Risk governance
- 2. score=0.5511, embedding=0.7373, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=market_risk, quality=high, title=Risk governance
- 3. score=0.5445, embedding=0.7278, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=liquidity_risk, quality=high, title=Risk governance
- 4. score=0.5422, embedding=0.7246, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=market_risk, quality=high, title=Risk governance
Manual Review:
- retrieval_relevance: Medium-high. The top chunk contains the risk-governance framing and starts listing risk types.
- groundedness: High. The answer is grounded in the cited passage.
- hallucination: No. The answer does not go beyond the evidence.
- completeness: Medium. It captures the framework and some risk categories, but truncates the full set of risk types expected by the reference.

[jpm_q9] How does the filing describe market risk management?
System Answer: Market Risk Management monitors market risks throughout the Firm and defines market risk policies and procedures. Market Risk Management seeks to measure risk, facilitate efficient risk/return decisions, reduce volatility in operating performance and provide transparency into the Firm's market risk profile for senior management, the Board of Directors and regulators. Market Risk Management is responsible for the following functions: • Maintaining a market risk policy framework • Independently measuring and monitoring LOB, Corporate, and Firmwide market risk • Defining, approving and monitoring limits • Performing stress testing and qualitative risk assessments
Reference Answer: Market risk management addresses the risk that changes in market conditions, such as interest rates, credit spreads, foreign exchange rates, commodity prices, or equity prices, could affect the value of positions and earnings.
Sources: Market risk management (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: Market Risk Management monitors market risks throughout the Firm and defines market risk policies and procedures. Market Risk Management seeks to measure risk, facilitate efficient risk/return decisions, reduce volatility in operating performance and provide transparency into the Firm's market risk profile for senior management, the Board of Directors and regulators. Market Risk Management is responsible for the following functions: • Maintaining a market risk policy framework • Independently measuring and monitoring LOB, Corporate, and Firmwide market risk • Defining, approving and monitoring limits • Performing stress testing and qualitative risk assessments
Lexical Overlap: 0.250
Retrieved Chunks:
- 1. score=0.5913, embedding=0.7947, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=market_risk, quality=high, title=Market risk management
- 2. score=0.5740, embedding=0.7593, section=Item 1A Risk Factors, primary_topic=financial_risk, secondary_topic=market_risk, quality=high, title=Market risk management
- 3. score=0.5727, embedding=0.7682, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=market_risk, quality=high, title=Market risk management
- 4. score=0.5719, embedding=0.7670, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=liquidity_risk, quality=high, title=Value-at-risk
Manual Review:
- retrieval_relevance: High. The top retrieved chunks are market-risk-management chunks from the correct MD&A area.
- groundedness: High. The answer is directly supported by the cited evidence.
- hallucination: No. The description is filing-backed.
- completeness: Medium. It describes the market-risk-management function well, but the reference emphasizes market-condition factors such as rates, spreads, FX, commodities, and equity prices.

[jpm_q10] What does the filing say about capital regulation or capital requirements?
System Answer: Financial holding company regulation: The Federal Reserve establishes capital requirements, including well-capitalized standards, for the Firm as a consolidated financial holding company. The Office of the Comptroller of the Currency ("OCC") establishes similar minimum capital requirements and standards for the Firm's principal IDI subsidiary, JPMorgan Chase Bank, N.A. The U.S. capital requirements generally follow the Capital Accord of the Basel Committee, as amended from time to time.
Reference Answer: The filing discusses capital risks and regulatory capital requirements, including the possibility that JPMorgan Chase's ability to distribute capital to shareholders or support business activities could be limited if it does not satisfy applicable regulatory capital requirements.
Sources: Financial holding company regulation (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: Financial holding company regulation: The Federal Reserve establishes capital requirements, including well-capitalized standards, for the Firm as a consolidated financial holding company. The Office of the Comptroller of the Currency ("OCC") establishes similar minimum capital requirements and standards for the Firm's principal IDI subsidiary, JPMorgan Chase Bank, N.A. The U.S. capital requirements generally follow the Capital Accord of the Basel Committee, as amended from time to time.
Lexical Overlap: 0.290
Retrieved Chunks:
- 1. score=0.5829, embedding=0.7828, section=Item 7 MD&A, primary_topic=regulatory_risk, secondary_topic=capital_regulation, quality=high, title=Financial holding company regulation
- 2. score=0.5816, embedding=0.7830, section=Item 7 MD&A, primary_topic=financial_risk, secondary_topic=market_risk, quality=medium, title=Regulatory capital requirements
- 3. score=0.5741, embedding=0.7702, section=Item 7 MD&A, primary_topic=regulatory_risk, secondary_topic=capital_regulation, quality=high, title=The Firm addresses these objectives through
- 4. score=0.5715, embedding=0.7664, section=Item 7 MD&A, primary_topic=regulatory_risk, secondary_topic=capital_regulation, quality=high, title=Regulatory capital requirements
Manual Review:
- retrieval_relevance: Medium-high. Retrieved chunks are capital-regulation related, although the top answer focuses on regulatory capital requirements rather than capital-risk consequences.
- groundedness: High. The answer is supported by the cited evidence.
- hallucination: No. The capital regulators and requirements are grounded.
- completeness: Medium. It identifies capital requirements and regulators but does not cover constraints on distributions or business support if requirements are not met.

[jpm_q11] How does JPMorgan Chase describe credit risk?
System Answer: Credit risk management: • reducing its access to capital markets • materially increasing its cost of issuing and servicing securities • triggering additional collateral or funding requirements, and • decreasing the number of investors and counterparties that are willing or permitted to do business with or lend to JPMorganChase. Any rating downgrade could also increase the credit spreads charged by market participants for taking credit risk on JPMorgan Chase & Co. and its subsidiaries. This could, in turn, adversely affect the value of debt and other obligations of JPMorgan Chase & Co. and its subsidiaries. Market risk: JPMorganChase's liquidity and cost of funding could be adversely affected by downgrades in its credit ratings. JPMorgan Chase & Co. and certain of its principal subsidiaries are rated by credit rating agencies, which evaluate general, firm-specific and industry-specific factors when determining credit ratings, including: • expected future profitability • risk management practices • legal expenses • regulatory developments • ratings differentials between bank holding companies and their bank and non-bank subsidiaries • assumptions about government support, and • economic and geopolitical developments. JPMorganChase has experienced credit ratings downgrades in the past, and there is no assurance that JPMorganChase's credit ratings will not be downgraded in the future. Furthermore, any such downgrade could occur at a time of broader market instability, limiting JPMorganChase's options for responding.
Reference Answer: Credit and investment risk is described as risk associated with default or changes in the credit profile of clients, counterparties, or customers, or loss of principal or reduced expected returns on investments, including consumer credit risk, wholesale credit risk, and investment portfolio risk.
Sources: Credit risk management (data/raw/jpm-20251231.htm, 2026-02-13); Market risk management (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: Credit risk management: • reducing its access to capital markets • materially increasing its cost of issuing and servicing securities • triggering additional collateral or funding requirements, and • decreasing the number of investors and counterparties that are willing or permitted to do business with or lend to JPMorganChase. Any rating downgrade could also increase the credit spreads charged by market participants for taking credit risk on JPMorgan Chase & Co. and its subsidiaries. This could, in turn, adversely affect the value of debt and other obligations of JPMorgan Chase & Co. and its subsidiaries. | Market risk: JPMorganChase's liquidity and cost of funding could be adversely affected by downgrades in its credit ratings. JPMorgan Chase & Co. and certain of its principal subsidiaries are rated by credit rating agencies, which evaluate general, firm-specific and industry-specific factors when determining credit ratings, including: • expected future profitability • risk management practices • legal expenses • regulatory developments • ratings differentials between bank holding companies and their bank and non-bank subsidiaries • assumptions about government support, and • economic and geopolitical developments. JPMorganChase has experienced credit ratings downgrades in the past, and there is no assurance that JPMorganChase's credit ratings will not be downgraded in the future. Furthermore, any such downgrade could occur at a time of broader market instability, limiting JPMorganChase's options for responding.
Lexical Overlap: 0.467
Retrieved Chunks:
- 1. score=0.6290, embedding=0.8378, section=Item 1A Risk Factors, primary_topic=financial_risk, secondary_topic=liquidity_risk, quality=high, title=Market risk management
- 2. score=0.6256, embedding=0.8330, section=Item 1A Risk Factors, primary_topic=financial_risk, secondary_topic=funding_risk, quality=high, title=Credit risk management
- 3. score=0.6197, embedding=0.8246, section=Item 1A Risk Factors, primary_topic=financial_risk, secondary_topic=market_risk, quality=high, title=Credit risk management
- 4. score=0.6095, embedding=0.8243, section=Item 1A Risk Factors, primary_topic=operational_risk, secondary_topic=model_risk, quality=high, title=Credit risk management
Manual Review:
- retrieval_relevance: Low-medium. Retrieval finds financial-risk and credit-labeled chunks, but the top evidence is about ratings/funding and market risk rather than the definition of credit and investment risk.
- groundedness: High. The generated answer is supported by retrieved evidence.
- hallucination: No. It remains grounded, but grounded in the wrong aspect of credit risk.
- completeness: Low. It misses the core definition involving default, credit-profile changes, consumer credit, wholesale credit, and investment portfolio risk.

[jpm_q12] What does the filing say about AI-related risks?
System Answer: People risk: It is also possible that JPMorganChase could miscalibrate its workforce planning and employee training efforts either because of over-reliance on AI or the failure to appropriately adopt AI. Over-reliance on AI could cause JPMorganChase to experience shortages in qualified staff due to reduced hiring or retention of employees, or could hinder the development or enhancement of important skills among its employees, including critical thinking, problem-solving, judgment, creativity and adaptability. On the other hand, any efficiencies or competitive advantages that AI may offer could be squandered if JPMorganChase fails to adopt AI in a timely and judicious manner and to make related adjustments to its workforce. Any of these factors could materially and adversely affect JPMorganChase's business and operations, results of operations, competitive position or reputation. The effects of climate change could adversely affect JPMorganChase's business and operations, both directly and as a result of impacts on its clients and customers. Cybersecurity risk: JPMorganChase's operations, results, and competitive standing could be adversely affected by the development of advanced technologies such as AI. The rapid development and deployment of advanced technologies, including generative and agentic AI systems, present a range of risks to JPMorganChase's businesses and operations, including: • AI system failures, inappropriate use of AI systems, lack of transparency in AI systems, or inaccurate or biased output from AI systems resulting from rapid deployment, insufficient testing, erroneous data, ineffective model design or insufficient controls, which could disrupt operations, cause erroneous transactions, compromise data privacy, infringe on intellectual property, harm clients and customers, or impair JPMorganChase's ability to make sound business decisions • increased exposure to cyber attacks, system manipulation, or data loss if AI systems, particularly agentic systems, are not designed and implemented with appropriate safeguards to prevent systems from accessing sensitive data sources or system resources and taking actions
Reference Answer: The filing states that rapid development and deployment of advanced technologies, including generative and agentic AI systems, may create risks such as system failures, inappropriate use, lack of transparency, inaccurate or biased outputs, cyber exposure, data loss, or operational disruption.
Sources: People risk (data/raw/jpm-20251231.htm, 2026-02-13); Cybersecurity risk (data/raw/jpm-20251231.htm, 2026-02-13)
Evidence: People risk: It is also possible that JPMorganChase could miscalibrate its workforce planning and employee training efforts either because of over-reliance on AI or the failure to appropriately adopt AI. Over-reliance on AI could cause JPMorganChase to experience shortages in qualified staff due to reduced hiring or retention of employees, or could hinder the development or enhancement of important skills among its employees, including critical thinking, problem-solving, judgment, creativity and adaptability. On the other hand, any efficiencies or competitive advantages that AI may offer could be squandered if JPMorganChase fails to adopt AI in a timely and judicious manner and to make related adjustments to its workforce. Any of these factors could materially and adversely affect JPMorganChase's business and operations, results of operations, competitive position or reputation. The effects of climate change could adversely affect JPMorganChase's business and operations, both directly and as a result of impacts on its clients and customers. | Cybersecurity risk: JPMorganChase's operations, results, and competitive standing could be adversely affected by the development of advanced technologies such as AI. The rapid development and deployment of advanced technologies, including generative and agentic AI systems, present a range of risks to JPMorganChase's businesses and operations, including: • AI system failures, inappropriate use of AI systems, lack of transparency in AI systems, or inaccurate or biased output from AI systems resulting from rapid deployment, insufficient testing, erroneous data, ineffective model design or insufficient controls, which could disrupt operations, cause erroneous transactions, compromise data privacy, infringe on intellectual property, harm clients and customers, or impair JPMorganChase's ability to make sound business decisions • increased exposure to cyber attacks, system manipulation, or data loss if AI systems, particularly agentic systems, are not designed and implemented with appropriate safeguards to prevent systems from accessing sensitive data sources or system resources and taking actions
Lexical Overlap: 0.838
Retrieved Chunks:
- 1. score=0.5594, embedding=0.7526, section=Item 1A Risk Factors, primary_topic=operational_risk, secondary_topic=cyber_risk, quality=high, title=Cybersecurity risk
- 2. score=0.5365, embedding=0.7200, section=Item 1A Risk Factors, primary_topic=operational_risk, secondary_topic=cyber_risk, quality=high, title=Cybersecurity risk
- 3. score=0.5258, embedding=0.7047, section=Item 1A Risk Factors, primary_topic=operational_risk, secondary_topic=technology_risk, quality=high, title=People risk
- 4. score=0.5257, embedding=0.6903, section=Item 1A Risk Factors, primary_topic=regulatory_risk, secondary_topic=litigation_and_enforcement, quality=high, title=Government policy risk
Manual Review:
- retrieval_relevance: High. The retrieved chunks include the cybersecurity-risk passage with explicit AI-risk language and a related people-risk passage.
- groundedness: High. The answer is well supported by the cited evidence.
- hallucination: No. The AI risks listed are present in the filing evidence.
- completeness: High. It covers generative/agentic AI, system failures, inappropriate use, transparency, biased output, cyber exposure, data loss, and operational impact.

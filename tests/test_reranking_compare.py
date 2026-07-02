import unittest

from scripts.compare_reranking import CaseComparison, build_report


class RerankingCompareTests(unittest.TestCase):
    def test_build_report_includes_ablation_metrics(self) -> None:
        report = build_report(
            [
                CaseComparison(
                    qid="q1",
                    question="What is liquidity risk?",
                    query_type="risk_liquidity",
                    expected_sections=["Item 7 MD&A"],
                    expected_topics=["liquidity_risk"],
                    embedding_top1="Liquidity risk",
                    reranked_top1="Liquidity risk management",
                    embedding_top1_hit=False,
                    reranked_top1_hit=True,
                    embedding_topk_hit=True,
                    reranked_topk_hit=True,
                    embedding_topic_top1_hit=False,
                    reranked_topic_top1_hit=True,
                    embedding_topic_topk_hit=True,
                    reranked_topic_topk_hit=True,
                )
            ]
        )

        self.assertIn("# Reranking Ablation Report", report)
        self.assertIn("embedding_top1_section_hit_rate: 0.000", report)
        self.assertIn("reranked_top1_section_hit_rate: 1.000", report)
        self.assertIn("risk_liquidity", report)
        self.assertIn("no -> yes", report)


if __name__ == "__main__":
    unittest.main()

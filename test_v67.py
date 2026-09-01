from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest

import agent_core
import factor_analysis


class FundamentalQualityV67Tests(unittest.TestCase):
    def test_four_quality_factors_use_declared_formulas(self):
        fields = agent_core.calculate_quality_factor_fields(
            operating_income=20,
            pretax_income=18,
            tax_provision=3.6,
            total_debt=30,
            equity=100,
            cash=10,
            prior_total_debt=25,
            prior_equity=90,
            prior_cash=10,
            operating_cashflow=25,
            capital_expenditure=-8,
            revenue=100,
            gross_profit=45,
            prior_revenue=90,
            prior_gross_profit=36,
            prior_operating_income=14,
        )
        self.assertAlmostEqual(fields["投入资本回报率ROIC"], 16 / 112.5)
        self.assertAlmostEqual(fields["自由现金流FCF"], 17)
        self.assertAlmostEqual(fields["自由现金流率"], 0.17)
        self.assertAlmostEqual(fields["毛利率趋势"], 0.05)
        self.assertAlmostEqual(fields["营业利润率趋势"], 0.20 - 14 / 90)

    def test_four_factor_groups_have_bounded_transparent_scores(self):
        strong = {
            "投入资本回报率ROIC": 0.14,
            "自由现金流FCF": 17,
            "自由现金流率": 0.17,
            "毛利率趋势": 0.02,
            "营业利润率趋势": 0.01,
            "估值历史分位": 0.10,
        }
        weak = {
            "投入资本回报率ROIC": -0.02,
            "自由现金流FCF": -5,
            "自由现金流率": -0.05,
            "毛利率趋势": -0.04,
            "营业利润率趋势": -0.01,
            "估值历史分位": 0.95,
        }
        strong_score, positives, _ = agent_core._score_fundamentals(strong)
        weak_score, _, risks = agent_core._score_fundamentals(weak)
        self.assertEqual(strong_score, 67.0)
        self.assertEqual(weak_score, 28.0)
        self.assertIn("自由现金流为正且现金创造能力较强", positives)
        self.assertIn("自由现金流为负", risks)

    def test_valuation_percentile_requires_enough_positive_samples(self):
        percentile, count = agent_core.valuation_history_percentile(15, [10, 20], minimum_samples=3)
        self.assertIsNone(percentile)
        self.assertEqual(count, 2)
        percentile, count = agent_core.valuation_history_percentile(15, [10, 20, 30, -2, 900])
        self.assertAlmostEqual(percentile, 1 / 3)
        self.assertEqual(count, 3)

    def test_sec_series_excludes_filings_after_historical_date(self):
        facts = {
            "Revenue": {
                "units": {
                    "USD": [
                        {"form": "10-K", "fp": "FY", "end": "2021-12-31", "filed": "2022-02-01", "val": 100},
                        {"form": "10-K", "fp": "FY", "end": "2022-12-31", "filed": "2023-02-01", "val": 120},
                    ]
                }
            }
        }
        series = agent_core._sec_fact_series(
            facts,
            ("Revenue",),
            ("USD",),
            as_of=date(2022, 6, 1),
        )
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["val"], 100)

    def test_catalog_contains_95_rules_and_all_new_factors(self):
        rows = factor_analysis.factor_catalog()
        self.assertEqual(len(rows), 95)
        names = {row["因子"] for row in rows}
        self.assertTrue(
            {
                "投入资本回报率ROIC",
                "自由现金流FCF",
                "毛利率／营业利润率趋势",
                "估值历史分位",
            }.issubset(names)
        )

    def test_formal_app_passes_the_same_price_history_to_fundamentals(self):
        source = Path(__file__).with_name("app.py").read_text(encoding="utf-8")
        self.assertIn("price_history: pd.DataFrame", source)
        self.assertGreaterEqual(source.count("bundle.stock,"), 2)


if __name__ == "__main__":
    unittest.main()

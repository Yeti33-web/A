from __future__ import annotations

from datetime import datetime, timezone
import unittest

from add_position_analysis import evaluate_add_position
from news_analysis import _enrich_items, assess_news


class NewsAnalysisTests(unittest.TestCase):
    def test_positive_news_raises_observation_score_with_cap(self) -> None:
        payload = {
            "items": [
                {
                    "title": "公司业绩预增并宣布回购",
                    "source": "公司公告",
                    "published_at": "2026-08-19T01:00:00+00:00",
                    "sentiment": "偏正面",
                    "sentiment_score": 1.0,
                    "relevance_score": 0.98,
                    "recency_weight": 1.0,
                    "source_weight": 1.2,
                },
                {
                    "title": "公司获得重大订单",
                    "source": "财经媒体",
                    "published_at": "2026-08-18T01:00:00+00:00",
                    "sentiment": "偏正面",
                    "sentiment_score": 1.0,
                    "relevance_score": 0.92,
                    "recency_weight": 0.85,
                    "source_weight": 1.0,
                },
            ],
            "fetched_at": "2026-08-19T02:00:00+00:00",
            "window_days": 7,
        }
        result = assess_news(payload, 60)
        self.assertTrue(result["usable_for_score"])
        self.assertEqual(result["score_adjustment"], 8)
        self.assertEqual(result["combined_score"], 68)
        self.assertEqual(result["direction"], "偏正面")

    def test_no_news_never_changes_existing_score(self) -> None:
        result = assess_news({"items": [], "window_days": 30}, 53)
        self.assertFalse(result["available"])
        self.assertEqual(result["score_adjustment"], 0)
        self.assertEqual(result["combined_score"], 53)

    def test_duplicate_titles_are_removed(self) -> None:
        now = datetime(2026, 8, 19, tzinfo=timezone.utc)
        raw = [
            {
                "title": "贵州茅台发布半年报 - 财经媒体",
                "summary": "贵州茅台业绩增长",
                "source": "财经媒体",
                "published_at": "2026-08-18T01:00:00+00:00",
                "url": "https://example.com/1",
                "provider": "测试源",
            },
            {
                "title": "贵州茅台发布半年报 | 另一媒体",
                "summary": "贵州茅台业绩增长",
                "source": "另一媒体",
                "published_at": "2026-08-18T02:00:00+00:00",
                "url": "https://example.com/2",
                "provider": "测试源",
            },
        ]
        items = _enrich_items(
            raw,
            market="A股",
            code="600519",
            name="贵州茅台股份有限公司",
            now=now,
            max_days=30,
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["relevance"], "高")

    def test_add_position_uses_news_adjusted_timing_without_changing_base(self) -> None:
        assessment = evaluate_add_position(
            session={"market": "A股", "code": "600519", "name": "贵州茅台", "principal_rmb": 5000},
            transaction={"principal_rmb": 1000, "input_method": "amount"},
            analysis={
                "suitability": {"fit": "适配", "fit_reason": "风险等级覆盖"},
                "selected_horizon": {
                    "score": 55,
                    "name": "2—4周",
                    "direction_available": True,
                    "signal_validation": {"status": "通过"},
                },
                "news_analysis": {
                    "available": True,
                    "usable_for_score": True,
                    "combined_score": 47,
                    "score_adjustment": -8,
                    "direction": "偏负面",
                    "items": [],
                },
                "position": {"upper_amount": 20000},
                "data_confidence": 80,
            },
            sell_signals={"status": "继续持有", "signals": []},
            profile={"investable_assets": 100000},
            portfolio={"rows": [], "total_principal_rmb": 5000, "position_count": 1},
            holding_snapshot={"current_rmb": 5000, "shares": None, "value_source": "本金代理"},
            market_data={
                "latest_price_native": 1500,
                "price_unit": "人民币元",
                "latest_market_date": "2026-08-18",
                "provider": "测试行情",
                "history_complete": True,
                "assessed_at": "2026-08-19T02:00:00",
            },
        )
        self.assertEqual(assessment["base_timing_score"], 55)
        self.assertEqual(assessment["timing_score"], 47)
        self.assertEqual(assessment["news_analysis"]["score_adjustment"], -8)
        self.assertIn("满足条件", assessment["conclusion"])


if __name__ == "__main__":
    unittest.main()

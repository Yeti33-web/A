from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import agent_core
import factor_analysis


def synthetic_market(rows: int = 1_250) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260831)
    dates = pd.bdate_range("2021-01-04", periods=rows)
    market_returns = rng.normal(0.00020, 0.008, rows)
    stock_returns = 0.00030 + 0.75 * market_returns + rng.normal(0.0, 0.010, rows)
    stock_close = 50.0 * np.exp(np.cumsum(stock_returns))
    benchmark_close = 100.0 * np.exp(np.cumsum(market_returns))
    volume = rng.lognormal(14.0, 0.25, rows)
    volume[-20:] *= 2.5
    stock = pd.DataFrame(
        {
            "日期": dates,
            "开盘": stock_close * 0.998,
            "最高": stock_close * 1.010,
            "最低": stock_close * 0.990,
            "收盘": stock_close,
            "成交量": volume,
        }
    )
    benchmark = pd.DataFrame(
        {
            "日期": dates,
            "开盘": benchmark_close * 0.999,
            "最高": benchmark_close * 1.006,
            "最低": benchmark_close * 0.994,
            "收盘": benchmark_close,
            "成交量": volume * 3,
        }
    )
    return stock, benchmark


class FactorCalibrationV66Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.stock, self.benchmark = synthetic_market()
        self.metrics = agent_core.calculate_quant_metrics(self.stock, self.benchmark)
        self.neutral = agent_core.EvidenceSnapshot(True, "synthetic", score=50.0)

    def test_trend_is_one_block_not_two_independent_scores(self):
        close = self.metrics["close"]
        volume = self.metrics["volume"]
        config = next(item for item in agent_core.HORIZONS if item["days"] == 20)
        frame = agent_core._technical_timing_frame(
            close,
            volume,
            self.metrics["benchmark_close"],
            config,
        )
        self.assertLessEqual(abs(float(frame["trend_points"].dropna().iloc[-1])), 10.0)

    def test_volume_and_analogues_no_longer_add_points(self):
        analogue = {
            "horizons": [
                {
                    "days": 20,
                    "available": True,
                    "sample_count": 30,
                    "confidence_score": 90,
                    "positive_ratio": 0.80,
                    "median_return": 0.10,
                }
            ]
        }
        results = agent_core.score_horizons(
            self.metrics,
            self.neutral,
            self.neutral,
            analogue,
            None,
        )
        item = next(result for result in results if result["days"] == 20)
        self.assertEqual(item["factor_contributions"]["volume"], 0.0)
        self.assertEqual(item["factor_contributions"]["historical_analog"], 0.0)
        self.assertEqual(item["analog_adjustment"], 0.0)
        self.assertFalse(item["analog_used"])

    def test_horizon_choice_does_not_chase_the_highest_current_score(self):
        scores = []
        for name in ["2—5个交易日", "2—4周", "1—3个月", "3—12个月", "1—3年"]:
            scores.append(
                {
                    "name": name,
                    "available": True,
                    "score": 99 if name == "2—5个交易日" else 50,
                    "signal_confidence": 60,
                    "direction_available": True,
                }
            )
        profile = {
            "earliest_need": "没有明确时间",
            "goal": "长期增值",
            "monitor_time": "30—60分钟",
            "stop_loss": "有明确且能执行的规则",
            "experience": "3年以上",
        }
        selected, _ = agent_core.choose_horizon(scores, profile)
        self.assertEqual(selected["name"], "1—3年")

    def test_failed_validation_blocks_a_buy_direction(self):
        selected = {
            "score": 90,
            "direction_available": False,
            "signal_validation": {"status": "未通过"},
        }
        suitability = {"fit": "适配", "fit_reason": "风险等级覆盖"}
        conclusion, _ = agent_core.build_final_conclusion(
            suitability,
            selected,
            {"upper_pct": 0.20},
        )
        self.assertEqual(conclusion, "个人条件可讨论，但方向证据不足")

    def test_limited_validation_is_not_actionable(self):
        scores = agent_core.score_horizons(
            self.metrics,
            self.neutral,
            self.neutral,
            None,
            None,
            "A股个股",
        )
        for item in scores:
            if (item.get("signal_validation") or {}).get("status") == "有限通过":
                self.assertFalse(item["direction_available"])

    def test_short_reversal_is_bounded_and_only_used_for_20_days(self):
        close = self.metrics["close"]
        volume = self.metrics["volume"]
        frame_20 = agent_core._technical_timing_frame(
            close,
            volume,
            self.metrics["benchmark_close"],
            next(item for item in agent_core.HORIZONS if item["days"] == 20),
        )
        frame_60 = agent_core._technical_timing_frame(
            close,
            volume,
            self.metrics["benchmark_close"],
            next(item for item in agent_core.HORIZONS if item["days"] == 60),
        )
        self.assertGreaterEqual(float(frame_20["short_reversal_adjustment"].min()), -5.0)
        self.assertLessEqual(float(frame_20["short_reversal_adjustment"].max()), 3.0)
        self.assertTrue((frame_60["short_reversal_adjustment"] == 0.0).all())

    def test_local_calibration_can_only_keep_reduce_or_block(self):
        close = self.metrics["close"]
        volume = self.metrics["volume"]
        frame = agent_core._technical_timing_frame(
            close,
            volume,
            self.metrics["benchmark_close"],
            next(item for item in agent_core.HORIZONS if item["days"] == 20),
        )
        validation = agent_core._validate_timing_signal(frame, 20)
        self.assertIn(validation["reliability_multiplier"], {0.0, 0.5, 1.0})
        self.assertIn(validation["status"], {"通过", "有限通过", "未通过"})
        self.assertNotIn("反转", validation["reason"])

    def test_catalog_documents_all_95_rules(self):
        rows = factor_analysis.factor_catalog()
        self.assertEqual(len(rows), 95)
        volume_row = next(
            item
            for item in rows
            if item["模块"] == "持有期时点评分" and item["因子"] == "成交量比"
        )
        self.assertEqual(volume_row["当前权重／分值"], "0分；仅作量价背景")
        factor_names = {item["因子"] for item in rows}
        self.assertTrue(
            {
                "20日短期均值回归",
                "52周高点位置",
                "10日量价确认",
                "波动率历史分位",
                "当前分数局部自校准",
                "跨股票样本外认证",
            }.issubset(factor_names)
        )

    def test_no_horizon_is_certified_after_final_sealed_audit(self):
        self.assertFalse(agent_core.direction_certification("A股个股", 5)["certified"])
        self.assertFalse(agent_core.direction_certification("美股个股", 5)["certified"])
        self.assertFalse(agent_core.direction_certification("A股个股", 20)["certified"])
        self.assertFalse(agent_core.direction_certification("港股个股", 5)["certified"])


if __name__ == "__main__":
    unittest.main()

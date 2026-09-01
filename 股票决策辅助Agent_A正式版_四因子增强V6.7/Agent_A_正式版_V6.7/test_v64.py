from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import pandas as pd


@dataclass
class FakePriceBundle:
    stock: pd.DataFrame
    benchmark: pd.DataFrame
    code: str = "TEST"
    name: str = "测试股票"
    provider: str = "synthetic"
    benchmark_name: str = "测试基准"
    asset_type: str = "个股"
    price_unit: str = "人民币元"


@dataclass
class FakeEvidenceSnapshot:
    available: bool
    provider: str = "synthetic"
    fields: dict | None = None
    score: float | None = None
    positives: list | None = None
    risks: list | None = None
    notes: list | None = None


def fake_safe_float(value):
    try:
        if value is None or value == "" or pd.isna(value):
            return None
        parsed = float(str(value).replace(",", "").replace("%", ""))
        return parsed if np.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def load_factor_module():
    core_path = Path(__file__).with_name("agent_core.py")
    core_spec = importlib.util.spec_from_file_location("agent_core", core_path)
    fake_agent_core = importlib.util.module_from_spec(core_spec)
    assert core_spec and core_spec.loader
    original = sys.modules.get("agent_core")
    sys.modules["agent_core"] = fake_agent_core
    try:
        core_spec.loader.exec_module(fake_agent_core)
        path = Path(__file__).with_name("factor_analysis.py")
        spec = importlib.util.spec_from_file_location("factor_analysis_under_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        if original is None:
            sys.modules.pop("agent_core", None)
        else:
            sys.modules["agent_core"] = original


FACTOR = load_factor_module()


def synthetic_inputs():
    rng = np.random.default_rng(20260825)
    rows = 1_100
    dates = pd.bdate_range("2021-01-01", periods=rows)
    market_returns = rng.normal(0.00025, 0.010, rows)
    stock_returns = 0.00015 + 0.85 * market_returns + rng.normal(0.0, 0.009, rows)
    benchmark_close = 100 * np.exp(np.cumsum(market_returns))
    stock_close = 40 * np.exp(np.cumsum(stock_returns))
    volume = rng.lognormal(14.0, 0.25, rows)
    stock = pd.DataFrame({"日期": dates, "收盘": stock_close, "成交量": volume})
    benchmark = pd.DataFrame({"日期": dates, "收盘": benchmark_close, "成交量": volume * 2})
    close = stock.set_index("日期")["收盘"]
    bench = benchmark.set_index("日期")["收盘"]
    selected = {
        "name": "2—4周",
        "days": 20,
        "fast": 20,
        "slow": 60,
        "score": 63,
        "stock_return": float(close.iloc[-1] / close.iloc[-21] - 1),
        "benchmark_return": float(bench.iloc[-1] / bench.iloc[-21] - 1),
        "analog_adjustment": 1.5,
        "analog_status": "+1.5分（严格同股样本）",
    }
    metrics = {
        "close": close,
        "benchmark_close": bench,
        "volume_ratio": float(volume[-20:].mean() / volume[-60:].mean()),
        "annual_volatility": 0.285,
        "max_drawdown": -0.365,
        "downside_volatility": 0.192,
        "beta": 0.985,
    }
    analysis = {
        "selected_horizon": selected,
        "metrics": metrics,
        "fundamental": FakeEvidenceSnapshot(True, score=58.0),
        "macro": FakeEvidenceSnapshot(True, score=54.0),
        "analog_forecast": {
            "backtest": {
                "available": True,
                "cases": 45,
                "direction_accuracy": 0.555,
                "momentum_accuracy": 0.530,
            }
        },
        "news_analysis": {
            "direction": "中性／存在分歧",
            "score_adjustment": 0,
        },
    }
    profile = {
        "fund_source": "闲置自有资金",
        "emergency_reserve": "6个月以上",
        "earliest_need": "没有明确时间",
        "loss_response": "继续按原计划持有",
        "max_loss": "10%—20%",
        "goal": "长期增值",
        "income_stability": "稳定",
        "experience": "3年以上",
    }
    return FakePriceBundle(stock, benchmark), analysis, profile


class FactorAnalysisV64Tests(unittest.TestCase):
    def test_full_catalog_is_complete_and_auditable(self):
        rows = FACTOR.factor_catalog()
        self.assertEqual(len(rows), 95)
        expected_columns = {"模块", "因子", "计算公式／规则", "数据要求", "方向", "当前权重／分值", "缺失数据处理"}
        self.assertTrue(all(set(row) == expected_columns for row in rows))
        modules = {row["模块"] for row in rows}
        self.assertTrue({"用户风险分", "股票风险分", "相似周期距离", "最新资讯修正", "卖出信号", "加仓条件分"}.issubset(modules))

    def test_current_contributions_are_reconstructed(self):
        bundle, analysis, profile = synthetic_inputs()
        result = FACTOR.build_factor_analysis(bundle, analysis, profile)
        self.assertAlmostEqual(sum(item["本次贡献"] for item in result["investor_contributions"]), 85.0)
        self.assertEqual(len(result["stock_risk_contributions"]), 4)
        timing_names = {item["因子"] for item in result["timing_contributions"]}
        self.assertIn("量化评分合计", timing_names)
        self.assertIn("最新公开资讯", timing_names)

    def test_walk_forward_validation_returns_governed_recommendations(self):
        bundle, analysis, profile = synthetic_inputs()
        validation = FACTOR.walk_forward_factor_validation(bundle, analysis)
        self.assertTrue(validation["available"])
        self.assertEqual(validation["forward_days"], 20)
        self.assertGreaterEqual(len(validation["rows"]), 6)
        self.assertTrue(all(item["建议"] in {"保留", "降低权重", "候选删除"} for item in validation["rows"]))
        self.assertTrue(all(item["有效验证时点"] >= 0 for item in validation["rows"]))
        self.assertIn("随后20个交易日收益", validation["method"])

    def test_app_contains_both_fresh_and_saved_factor_tabs(self):
        source = Path(__file__).with_name("app.py").read_text(encoding="utf-8")
        self.assertIn("from factor_analysis import build_factor_analysis", source)
        self.assertGreaterEqual(source.count('"因子解释与验证"'), 4)
        self.assertIn('analysis["factor_analysis"] = build_factor_analysis', source)
        self.assertIn("def render_factor_analysis", source)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import agent_core
from agent_core import EvidenceSnapshot
from questionnaire import QUESTIONS, answers_complete, answers_to_profile, compose_analysis_profile
from session_memory import (
    append_note,
    delete_session,
    portfolio_from_sessions,
    set_invested_principal,
    upsert_analysis_session,
)


def frame(start: str, periods: int) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=periods)
    close = pd.Series(100 * np.cumprod(np.full(periods, 1.0005)))
    return pd.DataFrame(
        {
            "日期": dates,
            "开盘": close * 0.998,
            "最高": close * 1.01,
            "最低": close * 0.99,
            "收盘": close,
            "成交量": np.full(periods, 1_000_000),
        }
    )


def default_answers() -> dict[str, str]:
    choices = [
        "20万—50万元",
        "闲置自有资金",
        "6个月以上",
        "3年内",
        "稳定",
        "10%—20%",
        "先复核原因再决定",
        "长期增值",
        "1—3年",
        "每月1—3次",
        "15—30分钟",
        "10%—30%",
        "有明确规则并能执行",
        "只能接受较小波动",
    ]
    return {question["key"]: value for question, value in zip(QUESTIONS, choices)}


def test_questionnaire_model() -> None:
    answers = default_answers()
    assert len(QUESTIONS) == 14
    assert answers_complete(answers)
    profile = answers_to_profile(answers)
    assert profile["asset_band"] == "20万—50万元"
    assert profile["investable_assets"] == 350_000
    composed = compose_analysis_profile(profile, 50_000, "否")
    assert composed["planned_amount"] == 50_000
    assert composed["leverage"] == "否"


def test_five_year_fetch_window_and_short_history() -> None:
    stock = frame("2026-07-01", 20)
    benchmark = frame("2021-08-11", 1000)
    calls: list[tuple[str, str]] = []

    def fake_stock(code: str, start_text: str, end_text: str):
        calls.append((start_text, end_text))
        return stock, "Test Inc.", "mock"

    with patch.object(agent_core, "fetch_us_security", side_effect=fake_stock), patch.object(
        agent_core, "fetch_us_benchmark", return_value=benchmark
    ):
        bundle = agent_core.fetch_price_bundle("美股", "AAPL")
    expected_start = (pd.Timestamp(date.today()) - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    assert calls[0][0] == expected_start
    assert bundle.requested_start == expected_start
    assert bundle.history_complete is False
    assert any("数据不足，无法准确判断" in item for item in bundle.warnings)


def test_hk_code_window_and_exchange_rate() -> None:
    assert agent_core.normalize_hk_code("700") == "00700"
    assert agent_core.normalize_hk_code("0700.HK") == "00700"
    assert agent_core.normalize_hk_code("09988") == "09988"
    assert agent_core.hk_yahoo_ticker("00700") == "0700.HK"
    assert agent_core.hk_yahoo_ticker("09988") == "9988.HK"

    stock = frame("2021-08-12", 1000)
    benchmark = frame("2021-08-12", 1000)
    calls: list[tuple[str, str]] = []

    def fake_hk_stock(code: str, start_text: str, end_text: str):
        calls.append((start_text, end_text))
        return stock, "Tencent Holdings", "mock"

    with patch.object(agent_core, "fetch_hk_security", side_effect=fake_hk_stock), patch.object(
        agent_core, "fetch_hk_benchmark", return_value=benchmark
    ):
        bundle = agent_core.fetch_price_bundle("港股", "0700.HK")
    expected_start = (pd.Timestamp(date.today()) - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    assert calls[0][0] == expected_start
    assert bundle.code == "00700"
    assert bundle.price_unit == "港元"
    assert bundle.benchmark_name == "恒生指数（HSI）"

    fx_frame = frame("2026-07-01", 20)
    fx_frame["收盘"] = 0.86
    with patch.object(
        agent_core, "fetch_yahoo_chart_history", return_value=(fx_frame, "HKD/CNY")
    ), patch.object(agent_core, "fetch_yfinance_history", return_value=pd.DataFrame()), patch.object(
        agent_core, "_fetch_fred_latest_value", side_effect=RuntimeError("mock")
    ):
        fx = agent_core.fetch_hkd_cny_rate()
    assert np.isclose(fx["rate"], 0.86)


def test_hk_akshare_history_channel() -> None:
    expected = frame("2021-08-12", 1000)

    class FakeAkshare:
        @staticmethod
        def stock_hk_hist(**kwargs):
            assert kwargs["symbol"] == "00700"
            assert kwargs["period"] == "daily"
            assert kwargs["adjust"] == "qfq"
            return expected

        @staticmethod
        def stock_hk_daily(**kwargs):
            return pd.DataFrame()

    with patch.object(agent_core, "ak", FakeAkshare()), patch.object(
        agent_core, "fetch_yahoo_chart_history", return_value=(pd.DataFrame(), "")
    ), patch.object(agent_core, "fetch_yfinance_history", return_value=pd.DataFrame()):
        data, name, provider = agent_core.fetch_hk_security("700", "2021-08-11", "2026-08-11")
    assert len(data) == 1000
    assert name == "00700"
    assert "港股" in provider


def test_short_history_returns_evidence_insufficient() -> None:
    stock = frame("2026-07-01", 20)
    benchmark = frame("2021-08-11", 1000)
    bundle = agent_core.PriceBundle(
        stock,
        benchmark,
        "AAPL",
        "Test Inc.",
        "mock",
        "SPY",
        "美股个股",
        "美元",
        history_complete=False,
        coverage_ratio=0.03,
    )
    profile = compose_analysis_profile(answers_to_profile(default_answers()), 50_000, "否")
    neutral = EvidenceSnapshot(False, "mock")
    analysis = agent_core.analyze_all(bundle, profile, neutral, neutral)
    assert analysis["data_confidence"] <= 30
    assert analysis["selected_horizon"] is None
    assert analysis["conclusion"].startswith("证据不足")


def test_holding_calculations() -> None:
    a_holding = agent_core.calculate_holding_values("A股", 1000, 37.65, 38.42)
    assert a_holding["current_rmb"] == 37_650
    assert a_holding["cost_total_rmb"] == 38_420
    assert round(a_holding["profit_rmb"], 2) == -770.00
    assert round(a_holding["return_rate"], 6) == round(37.65 / 38.42 - 1, 6)

    us_holding = agent_core.calculate_holding_values("美股", 10, 200, 180, 7.20)
    assert us_holding["current_native"] == 2_000
    assert us_holding["current_rmb"] == 14_400
    assert us_holding["cost_total_rmb"] == 12_960
    assert us_holding["profit_rmb"] == 1_440

    hk_holding = agent_core.calculate_holding_values("港股", 100, 470.4, 450, hkd_cny_rate=0.86)
    assert round(hk_holding["current_rmb"], 6) == 40_454.4
    assert round(hk_holding["cost_total_rmb"], 6) == 38_700
    assert hk_holding["native_currency"] == "港元"
    assert hk_holding["fx_pair"] == "港元兑人民币"

    amount_holding = agent_core.calculate_amount_holding_values(50_000, 45_000)
    assert amount_holding["current_rmb"] == 50_000
    assert amount_holding["profit_rmb"] == 5_000
    assert round(amount_holding["return_rate"], 6) == round(50_000 / 45_000 - 1, 6)


def test_existing_position_sell_signal_ladder() -> None:
    periods = 320
    dates = pd.bdate_range("2025-05-01", periods=periods)

    def price_frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "日期": dates,
                "开盘": values * 0.998,
                "最高": values * 1.01,
                "最低": values * 0.99,
                "收盘": values,
                "成交量": np.full(periods, 1_000_000),
            }
        )

    benchmark = price_frame(np.linspace(100, 125, periods))
    neutral_analog = {
        "horizons": [
            {
                "available": True,
                "days": 20,
                "confidence_score": 70,
                "positive_ratio": 0.50,
                "median_return": 0.0,
            }
        ]
    }

    def scenario(stock: pd.DataFrame, cost: float, fundamental: EvidenceSnapshot) -> dict:
        bundle = agent_core.PriceBundle(
            stock,
            benchmark,
            "600000",
            "测试股票",
            "mock",
            "沪深300",
            "A股个股",
            "人民币元",
            history_complete=True,
            coverage_ratio=1.0,
        )
        metrics = agent_core.calculate_quant_metrics(stock, benchmark)
        analysis = {
            "metrics": metrics,
            "selected_horizon": {
                "name": "2—4周",
                "days": 20,
                "fast": 20,
                "slow": 60,
                "review": "每周复核",
            },
            "fundamental": fundamental,
            "analog_forecast": neutral_analog,
        }
        holding = agent_core.calculate_holding_values(
            "A股", 1_000, float(stock["收盘"].iloc[-1]), cost
        )
        profile = {"max_loss": "10%—20%"}
        return agent_core.analyze_sell_signals(bundle, analysis, profile, holding)

    healthy = price_frame(np.linspace(100, 150, periods))
    neutral_fundamental = EvidenceSnapshot(True, "mock", score=55, risks=[])
    healthy_result = scenario(healthy, 110, neutral_fundamental)
    assert healthy_result["status"] == "继续持有"
    assert healthy_result["hard_count"] == 0
    assert healthy_result["auxiliary_count"] == 0

    weak_fundamental = EvidenceSnapshot(
        True,
        "mock",
        score=30,
        risks=["净利润同比下降", "营业收入同比下降"],
    )
    warning_result = scenario(healthy, 110, weak_fundamental)
    assert warning_result["hard_count"] == 0
    assert warning_result["auxiliary_count"] == 1
    assert warning_result["status"] == "警戒观察"

    weakening_values = np.concatenate(
        [np.linspace(100, 145, periods - 70), np.linspace(145, 118, 70)]
    )
    weakening = price_frame(weakening_values)
    reduce_result = scenario(weakening, 100, weak_fundamental)
    assert reduce_result["hard_count"] >= 1
    assert reduce_result["auxiliary_count"] >= 1
    assert reduce_result["status"] == "考虑分批减仓"

    exit_result = scenario(weakening, 150, neutral_fundamental)
    assert exit_result["hard_count"] == 2
    assert exit_result["status"] == "退出复核"
    assert exit_result["cost_protection_price"] is not None
    assert any(item["name"] == "趋势破位" for item in exit_result["signals"])

    no_position = agent_core.analyze_sell_signals(
        agent_core.PriceBundle(
            healthy,
            benchmark,
            "600000",
            "测试股票",
            "mock",
            "沪深300",
            "A股个股",
            "人民币元",
        ),
        {},
        {},
        None,
    )
    assert not no_position["available"]


def test_historical_analog_forecast_and_confidence_gate() -> None:
    periods = 1300
    random = np.random.default_rng(42)
    points = np.arange(periods)
    stock_returns = (
        0.00035
        + 0.0025 * np.sin(2 * np.pi * points / 90)
        + 0.0015 * np.sin(2 * np.pi * points / 240)
        + random.normal(0, 0.009 + 0.004 * (np.sin(2 * np.pi * points / 180) > 0), periods)
    )
    benchmark_returns = 0.00025 + 0.0012 * np.sin(2 * np.pi * points / 120) + random.normal(0, 0.007, periods)
    dates = pd.bdate_range("2021-08-12", periods=periods)

    def market_frame(returns: np.ndarray, base_volume: float) -> pd.DataFrame:
        close = 100 * np.cumprod(1 + returns)
        return pd.DataFrame(
            {
                "日期": dates,
                "开盘": close * 0.998,
                "最高": close * 1.01,
                "最低": close * 0.99,
                "收盘": close,
                "成交量": base_volume * (1 + 0.25 * np.sin(2 * np.pi * points / 60)),
            }
        )

    stock = market_frame(stock_returns, 1_000_000)
    benchmark = market_frame(benchmark_returns, 2_000_000)
    forecast = agent_core.analyze_historical_analogs(stock, benchmark, history_complete=True)
    assert forecast["available"]
    assert forecast["eligible_candidate_count"] < forecast["candidate_count"]
    assert any(item["available"] and item["sample_count"] >= 10 for item in forecast["horizons"])
    assert forecast["backtest"]["available"]
    assert forecast["backtest"]["cases"] > 0

    short_forecast = agent_core.analyze_historical_analogs(stock.head(250), benchmark.head(250), history_complete=False)
    assert not short_forecast["available"]
    assert short_forecast["confidence_label"] == "样本不足"


def test_low_confidence_analog_does_not_change_score() -> None:
    stock = frame("2021-08-12", 1300)
    benchmark = frame("2021-08-12", 1300)
    metrics = agent_core.calculate_quant_metrics(stock, benchmark)
    neutral = EvidenceSnapshot(False, "mock")
    common = {
        "days": 20,
        "available": True,
        "sample_count": 20,
        "positive_ratio": 0.90,
        "median_return": 0.12,
        "q10_return": -0.05,
    }
    low = {"horizons": [{**common, "confidence_score": 10}]}
    high = {"horizons": [{**common, "confidence_score": 80}]}
    low_result = next(item for item in agent_core.score_horizons(metrics, neutral, neutral, low) if item["days"] == 20)
    high_result = next(item for item in agent_core.score_horizons(metrics, neutral, neutral, high) if item["days"] == 20)
    assert not low_result["analog_used"]
    assert low_result["analog_adjustment"] == 0
    # V6.5中，相似周期不再进入正式方向评分：即使样本置信度高，也只用于展示。
    assert not high_result["analog_used"]
    assert high_result["analog_adjustment"] == 0


def test_adaptive_same_stock_samples_are_used_with_lower_confidence() -> None:
    stock = frame("2021-01-04", 1300)
    benchmark = frame("2021-01-04", 1300)

    def controlled_similarities(candidate_features, current_features, weights):
        return pd.Series(65.0, index=candidate_features.index, dtype="float64")

    unavailable_backtest = {
        "available": False,
        "cases": 0,
        "note": "测试中关闭回测，只检查自适应选样。",
    }
    with patch.object(agent_core, "_similarity_scores", side_effect=controlled_similarities), patch.object(
        agent_core, "_walk_forward_analog_backtest", return_value=unavailable_backtest
    ):
        forecast = agent_core.analyze_historical_analogs(stock, benchmark, history_complete=True)

    available = [item for item in forecast["horizons"] if item["available"]]
    assert available
    assert all(item["strict_sample_count"] == 0 for item in available)
    assert all(item["selection_mode"] == "自适应同股样本" for item in available)
    assert all(item["sample_count"] >= agent_core.ANALOG_MIN_SAMPLES for item in available)
    assert all(item["selection_threshold"] == agent_core.ANALOG_FALLBACK_MIN_SIMILARITY for item in available)
    assert all("下调可信度" in item["reason"] for item in available)


def test_market_fallback_is_small_and_reasons_are_explicit() -> None:
    stock = frame("2021-01-04", 1300)
    benchmark = frame("2021-01-04", 1300)
    metrics = agent_core.calculate_quant_metrics(stock, benchmark)
    neutral = EvidenceSnapshot(False, "mock")
    stock_forecast = {
        "horizons": [
            {
                "days": 20,
                "available": False,
                "sample_count": 7,
                "confidence_score": 0,
                "reason": "严格样本4/10个；放宽后仍只有7/10个。",
            }
        ]
    }
    market_forecast = {
        "source_label": "测试市场基准",
        "horizons": [
            {
                "days": 20,
                "available": True,
                "sample_count": 16,
                "confidence_score": 80,
                "positive_ratio": 0.85,
                "median_return": 0.10,
                "q10_return": -0.04,
            }
        ],
    }
    results = agent_core.score_horizons(
        metrics,
        neutral,
        neutral,
        stock_forecast,
        market_forecast,
    )
    twenty_day = next(item for item in results if item["days"] == 20)
    # V6.5将个股／市场相似周期都降为展示证据，不再改动方向分。
    assert not twenty_day["analog_used"]
    assert twenty_day["analog_source"] == "仅展示"
    assert twenty_day["analog_adjustment"] == 0
    assert "仅展示、不计分" in twenty_day["analog_status"]

    low_confidence_stock = {
        "horizons": [
            {
                "days": 20,
                "available": True,
                "sample_count": 13,
                "confidence_score": 31,
                "selection_mode": "自适应同股样本",
                "positive_ratio": 0.55,
                "median_return": 0.01,
                "q10_return": -0.08,
                "reason": "使用自适应同股样本，并下调可信度。",
            }
        ]
    }
    low_confidence_result = next(
        item
        for item in agent_core.score_horizons(
            metrics,
            neutral,
            neutral,
            low_confidence_stock,
            market_forecast,
        )
        if item["days"] == 20
    )
    assert "仅展示、不计分" in low_confidence_result["analog_status"]

    one_day = next(item for item in results if item["days"] == 1)
    assert "分钟" in one_day["analog_status"]
    long_horizon = next(item for item in results if item["days"] == 250)
    assert "五年窗口不足" in long_horizon["analog_status"]


def click_button(app, label: str):
    matches = [button for button in app.button if button.label == label]
    if not matches:
        raise AssertionError(f"未找到按钮：{label}；当前按钮：{[item.label for item in app.button]}")
    return matches[0].click().run(timeout=15)


def test_streamlit_questionnaire_and_profile_center() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    assert not app.exception
    if not app.button and any(
        "Streamlit Secrets" in str(getattr(item, "value", ""))
        for item in app.markdown
    ):
        print("SKIP: 本地未配置Supabase Secrets，整页交互测试跳过。")
        return
    assert not app.text_input
    assert all(button.label not in {"登录", "创建账号"} for button in app.button)
    answers = default_answers()
    for question in QUESTIONS:
        app = click_button(app, answers[question["key"]])
        assert not app.exception
    app = click_button(app, "确认提交并生成风险等级")
    assert not app.exception
    assert app.session_state["saved_profile"]["asset_band"] == "20万—50万元"
    assert app.session_state["profile_record"]["version"] == 1
    market_radio = next(item for item in app.radio if item.label == "市场")
    assert "港股" in market_radio.options
    holding_state = next(item for item in app.radio if item.label == "目前是否已经持有？")
    app = holding_state.set_value("已经持有").run(timeout=15)
    assert not app.exception
    assert any(item.label == "持仓信息填写方式" for item in app.radio)
    assert any(item.label == "持股数量（股）" for item in app.number_input)
    holding_method = next(item for item in app.radio if item.label == "持仓信息填写方式")
    app = holding_method.set_value("按持仓金额填写").run(timeout=15)
    assert not app.exception
    assert any(item.label == "当前持仓市值（折合人民币元）" for item in app.number_input)
    assert any(item.label == "累计投入成本（人民币元，可填0表示未知）" for item in app.number_input)
    app = click_button(app, "个人中心")
    assert not app.exception
    assert any(button.label == "更改个人信息" for button in app.button)


def test_stock_conversation_memory_and_delete_as_sold() -> None:
    analysis = {
        "conclusion": "可小仓观察",
        "conclusion_reason": "测试记录",
        "investor_level": "C3",
        "stock_risk_level": "R4",
        "suitability": {"fit": "有限适配"},
        "selected_horizon": {"name": "2—4周"},
        "data_confidence": 91,
    }
    sessions, added = upsert_analysis_session(
        {},
        event_id="analysis-1",
        market="A股",
        code="600519",
        name="贵州茅台",
        analysis=analysis,
        holding_state="已经持有",
        holding_snapshot={"cost_total_rmb": 10_000},
    )
    assert added
    sessions, added = upsert_analysis_session(
        sessions,
        event_id="analysis-1",
        market="A股",
        code="600519",
        name="贵州茅台",
        analysis=analysis,
        holding_state="已经持有",
        holding_snapshot={"cost_total_rmb": 10_000},
    )
    assert not added
    assert len(sessions["A股:600519"]["messages"]) == 2

    sessions, _ = upsert_analysis_session(
        sessions,
        event_id="analysis-2",
        market="美股",
        code="AAPL",
        name="Apple Inc.",
        analysis=analysis,
        holding_state="尚未持有",
        holding_snapshot=None,
    )
    assert sessions["美股:AAPL"]["principal_rmb"] == 0
    sessions = set_invested_principal(sessions, "美股:AAPL", 30_000)
    sessions = append_note(sessions, "美股:AAPL", "实际成交后登记")
    portfolio = portfolio_from_sessions(sessions)
    weights = {item["key"]: item["weight"] for item in portfolio["rows"]}
    assert portfolio["total_principal_rmb"] == 40_000
    assert weights["A股:600519"] == 0.25
    assert weights["美股:AAPL"] == 0.75

    sessions = delete_session(sessions, "美股:AAPL")
    portfolio = portfolio_from_sessions(sessions)
    assert portfolio["total_principal_rmb"] == 10_000
    assert portfolio["position_count"] == 1
    assert portfolio["rows"][0]["key"] == "A股:600519"


def main() -> None:
    tests = [
        test_questionnaire_model,
        test_five_year_fetch_window_and_short_history,
        test_hk_code_window_and_exchange_rate,
        test_hk_akshare_history_channel,
        test_short_history_returns_evidence_insufficient,
        test_holding_calculations,
        test_existing_position_sell_signal_ladder,
        test_historical_analog_forecast_and_confidence_gate,
        test_low_confidence_analog_does_not_change_score,
        test_adaptive_same_stock_samples_are_used_with_lower_confidence,
        test_market_fallback_is_small_and_reasons_are_explicit,
        test_stock_conversation_memory_and_delete_as_sold,
        test_streamlit_questionnaire_and_profile_center,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("简化部署版关键逻辑测试全部通过。")


if __name__ == "__main__":
    main()

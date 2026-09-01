from __future__ import annotations

import json
from pathlib import Path

from add_position_analysis import (
    build_add_position_messages,
    build_current_holding_snapshot,
    evaluate_add_position,
)


def base_inputs() -> dict:
    session = {
        "key": "A股:600000",
        "market": "A股",
        "code": "600000",
        "name": "测试股票",
        "principal_rmb": 10_000.0,
        "total_shares": 100.0,
        "shares_complete": True,
    }
    holding = build_current_holding_snapshot(
        session=session,
        latest_price_native=120.0,
        fx_rate=1.0,
        price_unit="人民币元",
    )
    return {
        "session": session,
        "transaction": {
            "input_method": "amount",
            "principal_rmb": 5_000.0,
            "shares": None,
            "price_native": None,
            "fx_rate": None,
        },
        "analysis": {
            "suitability": {"fit": "适配", "fit_reason": "风险等级覆盖"},
            "selected_horizon": {
                "score": 70,
                "name": "2—4周",
                "direction_available": True,
                "signal_validation": {"status": "通过"},
            },
            "position": {"upper_amount": 30_000.0},
            "data_confidence": 80,
        },
        "sell_signals": {
            "status": "继续持有",
            "signals": [{"key": "trend_break", "state": "未触发"}],
        },
        "profile": {"investable_assets": 200_000.0},
        "portfolio": {
            "rows": [],
            "total_principal_rmb": 40_000.0,
            "position_count": 2,
        },
        "holding_snapshot": holding,
        "market_data": {
            "latest_price_native": 120.0,
            "price_unit": "人民币元",
            "latest_market_date": "2026-08-18",
            "provider": "test",
            "history_complete": True,
            "assessed_at": "2026-08-18T12:00:00",
        },
    }


def test_holding_snapshot_uses_known_shares() -> None:
    inputs = base_inputs()
    holding = inputs["holding_snapshot"]
    assert holding["current_rmb"] == 12_000.0
    assert round(float(holding["return_rate"]), 12) == 0.2
    assert holding["cost_price"] == 100.0


def test_positive_assessment_does_not_mutate_position() -> None:
    inputs = base_inputs()
    original_principal = inputs["session"]["principal_rmb"]
    result = evaluate_add_position(**inputs)
    assert result["conclusion"] == "可在风险预算内小额分批加仓"
    assert result["post_principal_rmb"] == 15_000.0
    assert result["remaining_upper_amount_rmb"] == 18_000.0
    assert inputs["session"]["principal_rmb"] == original_principal


def test_over_budget_is_blocked() -> None:
    inputs = base_inputs()
    inputs["analysis"]["position"]["upper_amount"] = 13_000.0
    result = evaluate_add_position(**inputs)
    assert result["conclusion"] == "当前暂不适合加仓"
    assert any("超过当前剩余风险预算" in item for item in result["hard_reasons"])


def test_active_sell_signal_is_blocked() -> None:
    inputs = base_inputs()
    inputs["sell_signals"]["status"] = "考虑分批减仓"
    result = evaluate_add_position(**inputs)
    assert result["conclusion"] == "当前暂不适合加仓"
    assert any("卖出模块状态" in item for item in result["hard_reasons"])


def test_assessment_message_is_json_serialisable() -> None:
    inputs = base_inputs()
    result = evaluate_add_position(**inputs)
    messages = build_add_position_messages(
        transaction=inputs["transaction"],
        assessment=result,
    )
    encoded = json.dumps(messages, ensure_ascii=False)
    assert "add_position_assessment" in encoded
    assert "没有记为真实成交" in encoded


def test_streamlit_page_keeps_original_modes_and_adds_new_mode() -> None:
    source = Path("app.py").read_text(encoding="utf-8")
    assert 'modes = ["完整分析"]' in source
    assert 'modes.append("加仓适配分析")' in source
    assert 'modes.extend(["首次买入／加仓", "会话记录"])' in source
    assert "render_position_entry(session)" in source
    assert "render_saved_analysis(session)" in source


if __name__ == "__main__":
    tests = [
        test_holding_snapshot_uses_known_shares,
        test_positive_assessment_does_not_mutate_position,
        test_over_budget_is_blocked,
        test_active_sell_signal_is_blocked,
        test_assessment_message_is_json_serialisable,
        test_streamlit_page_keeps_original_modes_and_adds_new_mode,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

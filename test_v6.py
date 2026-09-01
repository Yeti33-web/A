from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# The verification container does not ship requests; production installs it from requirements.txt.
if "requests" not in sys.modules:
    sys.modules["requests"] = SimpleNamespace(Response=object, RequestException=Exception)

from agent_core import EvidenceSnapshot, PriceBundle
from cloud_store import CloudConfig, CloudStore
from session_memory import (
    build_position_messages,
    calculate_position_transaction,
    portfolio_from_sessions,
)
from snapshot_codec import build_analysis_snapshot, restore_analysis_snapshot


BASELINE_AGENT_HASH = "46d70d5757a5c33d9739d63080ace91e64ab09499ab937e165abc5820945519c"
BASELINE_QUESTIONNAIRE_HASH = "bd1de754bf15802c362b890e93eafaf1311e4852fdc3796ca3343ab33ee2f8c2"


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v67_fundamental_quality_business_core() -> None:
    assert sha256(ROOT / "agent_core.py") == BASELINE_AGENT_HASH
    assert sha256(ROOT / "questionnaire.py") == BASELINE_QUESTIONNAIRE_HASH


def test_transaction_calculation() -> None:
    amount = calculate_position_transaction(
        market="A股",
        input_method="按实际支付人民币总额",
        amount_rmb=12_345.678,
    )
    assert amount["principal_rmb"] == 12_345.678
    assert amount["shares"] is None

    shares = calculate_position_transaction(
        market="美股",
        input_method="按股数和成交单价",
        shares=10.5,
        price_native=200.25,
        fx_rate=7.2,
        fees_rmb=15.5,
    )
    expected = 10.5 * 200.25 * 7.2 + 15.5
    assert np.isclose(shares["principal_rmb"], expected)
    assert shares["currency"] == "美元"
    messages = build_position_messages(
        transaction_type="add",
        transaction=shares,
        trade_date="2026-08-17",
        note="测试加仓",
    )
    assert len(messages) == 2
    assert "加仓" in messages[0]["content"]
    assert "组合本金" in messages[1]["content"]


def test_portfolio_recalculation() -> None:
    sessions = {
        "A股:600519": {"market": "A股", "code": "600519", "name": "甲", "principal_rmb": 10_000},
        "美股:AAPL": {"market": "美股", "code": "AAPL", "name": "乙", "principal_rmb": 30_000},
        "港股:00700": {"market": "港股", "code": "00700", "name": "丙", "principal_rmb": 0},
    }
    result = portfolio_from_sessions(sessions)
    weights = {item["key"]: item["weight"] for item in result["rows"]}
    assert result["total_principal_rmb"] == 40_000
    assert weights["A股:600519"] == 0.25
    assert weights["美股:AAPL"] == 0.75
    del sessions["美股:AAPL"]
    assert portfolio_from_sessions(sessions)["total_principal_rmb"] == 10_000


def test_snapshot_roundtrip() -> None:
    dates = pd.bdate_range("2026-01-01", periods=4)
    frame = pd.DataFrame(
        {
            "日期": dates,
            "开盘": [10.0, 10.1, 10.2, 10.3],
            "最高": [10.2, 10.3, 10.4, 10.5],
            "最低": [9.8, 9.9, 10.0, 10.1],
            "收盘": [10.1, 10.2, 10.3, 10.4],
            "成交量": [1000, 1100, 1200, 1300],
        }
    )
    bundle = PriceBundle(frame, frame.copy(), "600000", "测试股票", "mock", "沪深300", "A股个股", "人民币元")
    analysis = {
        "conclusion": "测试结论",
        "fundamental": EvidenceSnapshot(True, "mock", {"市盈率": 12.3}, 60),
        "metrics": {"drawdown": pd.Series([0.0, -0.1], index=dates[:2]), "latest_price": np.float64(10.4)},
    }
    payload = build_analysis_snapshot(
        bundle=bundle,
        analysis=analysis,
        profile={"planned_amount": 10_000.0},
        holding_state="尚未持有",
        holding_method="按持股数量填写",
        holding_snapshot=None,
    )
    restored = restore_analysis_snapshot(payload)
    assert isinstance(restored["bundle"], PriceBundle)
    pd.testing.assert_frame_equal(restored["bundle"].stock, frame, check_freq=False)
    assert isinstance(restored["analysis"]["fundamental"], EvidenceSnapshot)
    assert isinstance(restored["analysis"]["metrics"]["drawdown"], pd.Series)
    assert restored["analysis"]["conclusion"] == "测试结论"


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows
        self.filters: list[tuple[str, object]] = []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def limit(self, *_args):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def execute(self):
        filtered = [
            row for row in self.rows
            if all(row.get(key) == value for key, value in self.filters)
        ]
        return FakeResponse(filtered)


class FakeClient:
    def __init__(self, tables):
        self.tables = tables
        self.last_query = None

    def table(self, name):
        self.last_query = FakeQuery(self.tables.get(name, []))
        return self.last_query


class FakeAuth:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def sign_up(self, payload):
        self.calls.append(("sign_up", dict(payload)))
        return SimpleNamespace(
            user=SimpleNamespace(id="user-otp", email=payload["email"]),
            session=None,
        )

    def verify_otp(self, payload):
        self.calls.append(("verify_otp", dict(payload)))
        return SimpleNamespace(
            user=SimpleNamespace(id="user-otp", email=payload["email"]),
            session=SimpleNamespace(access_token="access", refresh_token="refresh"),
        )

    def resend(self, payload):
        self.calls.append(("resend", dict(payload)))
        return SimpleNamespace()


class FakeAuthClient:
    def __init__(self):
        self.auth = FakeAuth()


def test_user_scoped_cloud_read() -> None:
    rows = [
        {
            "id": "s1",
            "user_id": "user-a",
            "market": "A股",
            "stock_code": "600000",
            "stock_name": "甲",
            "principal_rmb": 1000,
        },
        {
            "id": "s2",
            "user_id": "user-b",
            "market": "美股",
            "stock_code": "AAPL",
            "stock_name": "乙",
            "principal_rmb": 2000,
        },
    ]
    fake = FakeClient({"stock_sessions": rows})
    store = CloudStore(CloudConfig("https://example.supabase.co", "public-key"), client=fake)
    sessions = store.load_sessions("user-a")
    assert list(sessions) == ["A股:600000"]
    assert ("user_id", "user-a") in fake.last_query.filters


def test_email_signup_otp_flow() -> None:
    fake = FakeAuthClient()
    store = CloudStore(CloudConfig("https://example.supabase.co", "public-key"), client=fake)
    pending = store.sign_up("person@example.com", "password123")
    assert pending["email_confirmation_required"] is True
    store.resend_signup_otp("person@example.com")
    verified = store.verify_signup_otp("person@example.com", "123456")
    assert verified["user_id"] == "user-otp"
    assert verified["access_token"] == "access"
    assert fake.auth.calls == [
        ("sign_up", {"email": "person@example.com", "password": "password123"}),
        ("resend", {"type": "signup", "email": "person@example.com"}),
        ("verify_otp", {"email": "person@example.com", "token": "123456", "type": "email"}),
    ]


def test_sql_security_and_app_wiring() -> None:
    sql = (ROOT / "一键建表_V6.0.sql").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert sql.count("enable row level security") == 5
    assert "to authenticated" in sql
    assert "auth.uid()" in sql
    assert "SUPABASE_SERVICE_ROLE_KEY" not in app
    assert "SUPABASE_PUBLISHABLE_KEY" in app
    assert "save_snapshot" in app and "restore_analysis_snapshot" in app
    assert "永久保存本次首次买入／加仓" in app
    assert "发送注册验证码" in app
    assert "verify_signup_otp" in app
    assert "resend_signup_otp" in app
    email_template = (ROOT / "Supabase注册验证码邮件模板.html").read_text(encoding="utf-8")
    assert "{{ .Token }}" in email_template
    assert "{{ .ConfirmationURL }}" not in email_template
    assert "supabase>=2.0,<3.0" in requirements


def main() -> None:
    tests = [
        test_v65_calibrated_business_core,
        test_transaction_calculation,
        test_portfolio_recalculation,
        test_snapshot_roundtrip,
        test_user_scoped_cloud_read,
        test_email_signup_otp_flow,
        test_sql_security_and_app_wiring,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("V6.1 邮箱验证码注册与原永久保存逻辑测试全部通过。")


if __name__ == "__main__":
    main()

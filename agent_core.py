from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from io import StringIO
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import baostock as bs
except ImportError:
    bs = None

try:
    import yfinance as yf
except ImportError:
    yf = None


HISTORY_YEARS = 5
MODEL_VERSION = "V6.6.1"
MODEL_RULESET_ID = "fused-point-in-time-v1.1"

# Cross-stock certification is retained as a confidence label, not as a switch
# that clears every stock's current-state score. A stock may form an actionable
# direction only when its own walk-forward validation passes. Before broader
# certification, confidence and position size remain capped.
DIRECTION_CERTIFICATION: dict[str, set[int]] = {
    "A股个股": set(),
    "美股个股": set(),
    "港股个股": set(),
    "场内基金": set(),
}


def direction_certification(asset_type: str | None, days: int) -> dict[str, Any]:
    scope = str(asset_type or "未指定证券类型")
    certified = int(days) in DIRECTION_CERTIFICATION.get(scope, set())
    if certified:
        reason = "该市场的本周期已通过多股票、跨阶段和独立留出样本检查。"
    elif int(days) == 5:
        reason = "该周期尚未在第二批独立股票样本中稳定复现，只允许受限的单股判断。"
    elif int(days) in {20, 60}:
        reason = "该周期尚未在跨股票或跨市场留出样本中稳定复现，只允许受限的单股判断。"
    elif int(days) in {120, 250}:
        reason = "五年窗口内的独立长期样本不足，只允许受限的单股判断。"
    else:
        reason = "该证券类型尚未完成独立样本认证，只允许受限的单股判断。"
    return {
        "status": "已认证" if certified else "未认证",
        "certified": certified,
        "asset_type": scope,
        "days": int(days),
        "reason": reason,
    }
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 StockResearchAgent/5.7"}
SEC_HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "IndividualInvestorResearchAgent/5.7 educational-use")}


HORIZONS: list[dict[str, Any]] = [
    {
        "name": "1个交易日",
        "days": 1,
        "fast": 5,
        "slow": 20,
        "minimum_rows": 60,
        "intraday_required": True,
        "review": "下一个交易日前复核",
    },
    {
        "name": "2—5个交易日",
        "days": 5,
        "fast": 5,
        "slow": 20,
        "minimum_rows": 120,
        "intraday_required": False,
        "review": "每个交易日复核",
    },
    {
        "name": "2—4周",
        "days": 20,
        "fast": 20,
        "slow": 60,
        "minimum_rows": 120,
        "intraday_required": False,
        "review": "每周复核",
    },
    {
        "name": "1—3个月",
        "days": 60,
        "fast": 20,
        "slow": 120,
        "minimum_rows": 250,
        "intraday_required": False,
        "review": "每月或重大公告后复核",
    },
    {
        "name": "3—12个月",
        "days": 120,
        "fast": 60,
        "slow": 250,
        "minimum_rows": 500,
        "intraday_required": False,
        "review": "每季度及财报后复核",
    },
    {
        "name": "1—3年",
        "days": 250,
        "fast": 120,
        "slow": 250,
        "minimum_rows": 750,
        "intraday_required": False,
        "review": "每季度及投资逻辑变化时复核",
    },
]


ANALOG_HORIZONS: list[dict[str, Any]] = [
    {"name": "2—5个交易日", "days": 5, "minimum_gap": 20},
    {"name": "2—4周", "days": 20, "minimum_gap": 20},
    {"name": "1—3个月", "days": 60, "minimum_gap": 40},
    {"name": "3—12个月", "days": 120, "minimum_gap": 60},
]

ANALOG_MIN_SAMPLES = 10
ANALOG_MIN_SIMILARITY = 72.0
ANALOG_FALLBACK_MIN_SIMILARITY = 60.0
# V6.6 keeps historical analogues as an explanatory scenario tool.  The
# earlier +/-8 point production adjustment was too large relative to its
# unstable walk-forward results, so it no longer changes the buy-side score.
ANALOG_SCORE_MIN_CONFIDENCE = 60
ANALOG_MARKET_MIN_CONFIDENCE = 70
ANALOG_SCORE_ENABLED = False
ANALOG_FEATURE_WEIGHTS: dict[str, float] = {
    "return_5": 0.05,
    "return_20": 0.10,
    "return_60": 0.09,
    "return_120": 0.04,
    "volatility_20": 0.12,
    "volatility_60": 0.08,
    "downside_volatility_60": 0.06,
    "drawdown_60": 0.08,
    "drawdown_120": 0.06,
    "price_ma20": 0.06,
    "price_ma60": 0.06,
    "volume_ratio": 0.04,
    "relative_return_20": 0.06,
    "relative_return_60": 0.04,
    "benchmark_return_20": 0.03,
    "benchmark_volatility_20": 0.03,
}


GOAL_PRIORS = {
    "保值为主": {"2—5个交易日": -8, "2—4周": -3, "1—3个月": 2, "3—12个月": 9, "1—3年": 8},
    "股息／稳健收益": {"2—5个交易日": -8, "2—4周": -4, "1—3个月": 2, "3—12个月": 9, "1—3年": 13},
    "长期增值": {"2—5个交易日": -9, "2—4周": -4, "1—3个月": 2, "3—12个月": 10, "1—3年": 14},
    "波段操作": {"2—5个交易日": 3, "2—4周": 12, "1—3个月": 13, "3—12个月": 2, "1—3年": -6},
    "短线交易": {"2—5个交易日": 14, "2—4周": 8, "1—3个月": -1, "3—12个月": -8, "1—3年": -12},
}


@dataclass
class PriceBundle:
    stock: pd.DataFrame
    benchmark: pd.DataFrame
    code: str
    name: str
    provider: str
    benchmark_name: str
    asset_type: str
    price_unit: str
    warnings: list[str] = field(default_factory=list)
    requested_start: str = ""
    requested_end: str = ""
    history_complete: bool = True
    coverage_ratio: float = 1.0


@dataclass
class EvidenceSnapshot:
    available: bool
    provider: str
    fields: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    positives: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "" or pd.isna(value):
            return None
        result = float(str(value).replace(",", "").replace("%", ""))
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def ratio_fraction(value: Any) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    if abs(number) > 5:
        return number / 100
    return number


def calculate_holding_values(
    market: str,
    shares: float,
    latest_price: float,
    cost_price: float | None = None,
    usd_cny_rate: float | None = None,
    hkd_cny_rate: float | None = None,
) -> dict[str, Any]:
    share_count = float(shares)
    price = float(latest_price)
    if share_count <= 0:
        raise ValueError("持股数量必须大于0。")
    if price <= 0:
        raise ValueError("最新公开价格无效，暂时无法计算持仓市值。")
    if market not in {"A股", "美股", "港股"}:
        raise ValueError("暂不支持该市场的持仓换算。")

    current_native = share_count * price
    rate = 1.0
    native_currency = "人民币元"
    fx_pair = None
    if market == "美股":
        rate = float(usd_cny_rate or 0.0)
        if rate <= 0:
            raise ValueError("美元兑人民币汇率暂不可用，请改用“按持仓金额填写”。")
        native_currency = "美元"
        fx_pair = "美元兑人民币"
    elif market == "港股":
        rate = float(hkd_cny_rate or 0.0)
        if rate <= 0:
            raise ValueError("港元兑人民币汇率暂不可用，请改用“按持仓金额填写”。")
        native_currency = "港元"
        fx_pair = "港元兑人民币"
    current_rmb = current_native * rate

    parsed_cost = float(cost_price or 0.0)
    cost_total_native = share_count * parsed_cost if parsed_cost > 0 else None
    profit_native = current_native - cost_total_native if cost_total_native is not None else None
    return_rate = price / parsed_cost - 1 if parsed_cost > 0 else None
    return {
        "method": "按持股数量填写",
        "shares": share_count,
        "latest_price": price,
        "native_currency": native_currency,
        "current_native": current_native,
        "current_rmb": current_rmb,
        "cost_price": parsed_cost if parsed_cost > 0 else None,
        "cost_total_native": cost_total_native,
        "cost_total_rmb": cost_total_native * rate if cost_total_native is not None else None,
        "profit_native": profit_native,
        "profit_rmb": profit_native * rate if profit_native is not None else None,
        "return_rate": return_rate,
        "usd_cny_rate": rate if market == "美股" else None,
        "hkd_cny_rate": rate if market == "港股" else None,
        "fx_rate": rate if market in {"美股", "港股"} else None,
        "fx_pair": fx_pair,
    }


def calculate_amount_holding_values(
    current_rmb: float,
    total_cost_rmb: float | None = None,
) -> dict[str, Any]:
    current_value = float(current_rmb)
    if current_value <= 0:
        raise ValueError("当前持仓市值必须大于0。")

    parsed_cost = float(total_cost_rmb or 0.0)
    profit_rmb = current_value - parsed_cost if parsed_cost > 0 else None
    return_rate = current_value / parsed_cost - 1 if parsed_cost > 0 else None
    return {
        "method": "按持仓金额填写",
        "shares": None,
        "latest_price": None,
        "native_currency": "人民币元",
        "current_native": current_value,
        "current_rmb": current_value,
        "cost_price": None,
        "cost_total_native": parsed_cost if parsed_cost > 0 else None,
        "cost_total_rmb": parsed_cost if parsed_cost > 0 else None,
        "profit_native": profit_rmb,
        "profit_rmb": profit_rmb,
        "return_rate": return_rate,
        "usd_cny_rate": None,
    }


def normalize_a_code(raw_code: str) -> str:
    code = raw_code.strip().upper()
    for suffix in (".SZ", ".SH", ".SS", ".BJ"):
        code = code.replace(suffix, "")
    if not (len(code) == 6 and code.isdigit()):
        raise ValueError("A股代码应为6位数字，例如000001、600519、300750或北交所代码。")
    return code


def normalize_us_code(raw_code: str) -> str:
    code = raw_code.strip().upper()
    if not code or len(code) > 15 or not all(char.isalnum() or char in ".-" for char in code):
        raise ValueError("请输入有效美股代码，例如AAPL、MSFT、NVDA或BRK-B。")
    return code


def normalize_hk_code(raw_code: str) -> str:
    code = raw_code.strip().upper().replace(" ", "")
    if code.endswith(".HK"):
        code = code[:-3]
    if not code or not code.isdigit() or len(code) > 5 or int(code) <= 0:
        raise ValueError("港股代码应为1至5位数字，例如00700、09988；也可以输入700或0700.HK。")
    return code.zfill(5)


def hk_yahoo_ticker(code: str) -> str:
    normalized = normalize_hk_code(code)
    numeric = str(int(normalized))
    yahoo_code = numeric.zfill(4) if len(numeric) < 4 else numeric
    return f"{yahoo_code}.HK"


def is_exchange_traded_fund_code(code: str) -> bool:
    return code.startswith(("1", "5"))


def a_share_exchange(code: str) -> str:
    if code.startswith(("4", "8", "92")):
        return "bj"
    if code[0] in {"5", "6", "9"}:
        return "sh"
    return "sz"


def a_share_baostock_code(code: str) -> str:
    return f"{a_share_exchange(code)}.{code}"


def a_share_yahoo_ticker(code: str) -> str:
    suffix = ".BJ" if a_share_exchange(code) == "bj" else ".SS" if a_share_exchange(code) == "sh" else ".SZ"
    return f"{code}{suffix}"


def clean_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘", "成交量"])
    result = data.copy()
    result["日期"] = pd.to_datetime(result["日期"], errors="coerce").dt.tz_localize(None)
    for column in ["开盘", "最高", "最低", "收盘", "成交量"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["日期", "开盘", "最高", "最低", "收盘"])
    result = result[(result["收盘"] > 0) & (result["最高"] >= result["最低"])]
    result = result[result["日期"] <= pd.Timestamp(date.today()) + pd.Timedelta(days=1)]
    result["成交量"] = result["成交量"].fillna(0).clip(lower=0)
    return result.sort_values("日期").drop_duplicates("日期", keep="last").reset_index(drop=True)


def standardize_chinese_ohlcv(raw: pd.DataFrame, volume_in_lots: bool = False) -> pd.DataFrame:
    required = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"行情接口返回数据缺少字段：{', '.join(missing)}")
    data = raw[required].copy()
    if volume_in_lots:
        data["成交量"] = pd.to_numeric(data["成交量"], errors="coerce") * 100
    return clean_ohlcv(data)


def standardize_english_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return clean_ohlcv(pd.DataFrame())
    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index() if not isinstance(data.index, pd.RangeIndex) else data
    lower_map = {str(column).lower(): column for column in data.columns}
    date_column = lower_map.get("date") or lower_map.get("datetime")
    mapping = {
        date_column: "日期",
        lower_map.get("open"): "开盘",
        lower_map.get("high"): "最高",
        lower_map.get("low"): "最低",
        lower_map.get("close"): "收盘",
        lower_map.get("volume"): "成交量",
    }
    if any(key is None for key in mapping):
        raise ValueError("英文行情接口返回字段不完整。")
    return clean_ohlcv(data[list(mapping)].rename(columns=mapping))


def filter_dates(data: pd.DataFrame, start_text: str, end_text: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_text)
    end_ts = pd.Timestamp(end_text)
    return data[(data["日期"] >= start_ts) & (data["日期"] <= end_ts)].copy().reset_index(drop=True)


def _baostock_result_to_frame(result: Any) -> pd.DataFrame:
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields) if rows else pd.DataFrame(columns=result.fields)


def fetch_baostock_history(symbol: str, start_text: str, end_text: str, adjustflag: str = "2") -> pd.DataFrame:
    if bs is None:
        return pd.DataFrame()
    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"BaoStock登录失败：{login_result.error_msg}")
    try:
        result = bs.query_history_k_data_plus(
            symbol,
            "date,open,high,low,close,volume",
            start_date=start_text,
            end_date=end_text,
            frequency="d",
            adjustflag=adjustflag,
        )
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock查询失败：{result.error_msg}")
        raw = _baostock_result_to_frame(result).rename(
            columns={"date": "日期", "open": "开盘", "high": "最高", "low": "最低", "close": "收盘", "volume": "成交量"}
        )
        return clean_ohlcv(raw)
    finally:
        bs.logout()


def fetch_yfinance_history(symbol: str, start_text: str, end_text: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    end_exclusive = (pd.Timestamp(end_text) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        symbol,
        start=start_text,
        end=end_exclusive,
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=15,
    )
    return standardize_english_ohlcv(raw)


def _get_with_retry(url: str, *, params: dict[str, Any] | None = None, timeout: int = 20) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=timeout)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_error = RuntimeError(f"HTTP {response.status_code}")
                if attempt < 2:
                    time.sleep(0.6 * (2**attempt))
                    continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.6 * (2**attempt))
    raise RuntimeError(f"公开行情接口连续重试后仍失败：{last_error}")


def fetch_yahoo_chart_history(symbol: str, start_text: str, end_text: str) -> tuple[pd.DataFrame, str]:
    period1 = int(pd.Timestamp(start_text, tz="UTC").timestamp())
    period2 = int((pd.Timestamp(end_text, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    response = _get_with_retry(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        },
        timeout=20,
    )
    payload = response.json().get("chart", {})
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    results = payload.get("result") or []
    if not results:
        return pd.DataFrame(), symbol
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjusted_list = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or quotes.get("close") or []
    if not timestamps or not quotes.get("close"):
        return pd.DataFrame(), symbol
    raw_close = pd.Series(quotes.get("close"), dtype="float64")
    adjusted_close = pd.Series(adjusted_list, dtype="float64")
    factor = adjusted_close / raw_close.replace(0, np.nan)
    frame = pd.DataFrame(
        {
            "日期": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
            "开盘": pd.Series(quotes.get("open"), dtype="float64") * factor,
            "最高": pd.Series(quotes.get("high"), dtype="float64") * factor,
            "最低": pd.Series(quotes.get("low"), dtype="float64") * factor,
            "收盘": adjusted_close,
            "成交量": pd.Series(quotes.get("volume"), dtype="float64"),
        }
    )
    meta = result.get("meta", {})
    name = str(meta.get("longName") or meta.get("shortName") or symbol)
    return clean_ohlcv(frame), name


def fetch_stooq_history(symbol: str, start_text: str, end_text: str) -> pd.DataFrame:
    normalized = normalize_us_code(symbol).lower().replace("-", ".")
    response = _get_with_retry(
        "https://stooq.com/q/d/l/",
        params={
            "s": f"{normalized}.us",
            "d1": start_text.replace("-", ""),
            "d2": end_text.replace("-", ""),
            "i": "d",
        },
        timeout=20,
    )
    if not response.text.strip() or response.text.lstrip().lower().startswith("no data"):
        return pd.DataFrame()
    raw = pd.read_csv(StringIO(response.text))
    return standardize_english_ohlcv(raw)


def fetch_usd_cny_rate() -> dict[str, Any]:
    end_text = date.today().strftime("%Y-%m-%d")
    start_text = (pd.Timestamp(end_text) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    errors: list[str] = []
    candidates: list[dict[str, Any]] = []

    try:
        data, _ = fetch_yahoo_chart_history("CNY=X", start_text, end_text)
        if not data.empty:
            latest = data.iloc[-1]
            candidates.append(
                {
                    "rate": float(latest["收盘"]),
                    "date": pd.Timestamp(latest["日期"]),
                    "provider": "Yahoo Finance美元兑人民币公开日线（CNY=X）",
                }
            )
    except Exception as exc:
        errors.append(f"Yahoo图表接口：{exc}")

    try:
        data = fetch_yfinance_history("CNY=X", start_text, end_text)
        if not data.empty:
            latest = data.iloc[-1]
            candidates.append(
                {
                    "rate": float(latest["收盘"]),
                    "date": pd.Timestamp(latest["日期"]),
                    "provider": "yfinance美元兑人民币备用日线（CNY=X）",
                }
            )
    except Exception as exc:
        errors.append(f"yfinance：{exc}")

    try:
        response = _get_with_retry(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": "DEXCHUS"},
            timeout=20,
        )
        raw = pd.read_csv(StringIO(response.text))
        date_column = next((column for column in raw.columns if "date" in str(column).lower()), None)
        value_column = next((column for column in raw.columns if str(column).upper() == "DEXCHUS"), None)
        if date_column and value_column:
            values = raw[[date_column, value_column]].copy()
            values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
            values[date_column] = pd.to_datetime(values[date_column], errors="coerce")
            values = values.dropna()
            if not values.empty:
                latest = values.iloc[-1]
                candidates.append(
                    {
                        "rate": float(latest[value_column]),
                        "date": pd.Timestamp(latest[date_column]),
                        "provider": "美国FRED美元兑人民币参考汇率（DEXCHUS）",
                    }
                )
    except Exception as exc:
        errors.append(f"FRED：{exc}")

    valid = [item for item in candidates if 4.0 <= float(item["rate"]) <= 12.0]
    if not valid:
        detail = "；".join(errors) if errors else "公开接口没有返回有效汇率"
        raise RuntimeError(f"美元兑人民币汇率获取失败。{detail}")
    return max(valid, key=lambda item: pd.Timestamp(item["date"]))


def _fetch_fred_latest_value(series_id: str) -> tuple[float, pd.Timestamp]:
    response = _get_with_retry(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id},
        timeout=20,
    )
    raw = pd.read_csv(StringIO(response.text))
    date_column = next((column for column in raw.columns if "date" in str(column).lower()), None)
    value_column = next((column for column in raw.columns if str(column).upper() == series_id.upper()), None)
    if not date_column or not value_column:
        raise RuntimeError(f"FRED没有返回{series_id}所需字段。")
    values = raw[[date_column, value_column]].copy()
    values[date_column] = pd.to_datetime(values[date_column], errors="coerce")
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    values = values.dropna().sort_values(date_column)
    if values.empty:
        raise RuntimeError(f"FRED没有返回{series_id}有效数值。")
    latest = values.iloc[-1]
    return float(latest[value_column]), pd.Timestamp(latest[date_column])


def fetch_hkd_cny_rate() -> dict[str, Any]:
    end_text = date.today().strftime("%Y-%m-%d")
    start_text = (pd.Timestamp(end_text) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    errors: list[str] = []
    candidates: list[dict[str, Any]] = []

    try:
        data, _ = fetch_yahoo_chart_history("HKDCNY=X", start_text, end_text)
        if not data.empty:
            latest = data.iloc[-1]
            candidates.append(
                {
                    "rate": float(latest["收盘"]),
                    "date": pd.Timestamp(latest["日期"]),
                    "provider": "Yahoo Finance港元兑人民币公开日线（HKDCNY=X）",
                }
            )
    except Exception as exc:
        errors.append(f"Yahoo图表接口：{exc}")

    try:
        data = fetch_yfinance_history("HKDCNY=X", start_text, end_text)
        if not data.empty:
            latest = data.iloc[-1]
            candidates.append(
                {
                    "rate": float(latest["收盘"]),
                    "date": pd.Timestamp(latest["日期"]),
                    "provider": "yfinance港元兑人民币备用日线（HKDCNY=X）",
                }
            )
    except Exception as exc:
        errors.append(f"yfinance：{exc}")

    try:
        cny_per_usd, cny_date = _fetch_fred_latest_value("DEXCHUS")
        hkd_per_usd, hkd_date = _fetch_fred_latest_value("DEXHKUS")
        if hkd_per_usd > 0:
            candidates.append(
                {
                    "rate": cny_per_usd / hkd_per_usd,
                    "date": min(cny_date, hkd_date),
                    "provider": "美国FRED交叉汇率（DEXCHUS÷DEXHKUS）",
                }
            )
    except Exception as exc:
        errors.append(f"FRED交叉汇率：{exc}")

    valid = [item for item in candidates if 0.5 <= float(item["rate"]) <= 1.5]
    if not valid:
        detail = "；".join(errors) if errors else "公开接口没有返回有效汇率"
        raise RuntimeError(f"港元兑人民币汇率获取失败。{detail}")
    return max(valid, key=lambda item: pd.Timestamp(item["date"]))


def fetch_akshare_us_history(symbol: str, start_text: str, end_text: str) -> tuple[pd.DataFrame, str]:
    if ak is None:
        return pd.DataFrame(), ""
    last_error: Exception | None = None
    candidates = [symbol] if symbol[:3].isdigit() and "." in symbol else [f"105.{symbol}", f"106.{symbol}", f"107.{symbol}"]
    for internal_code in candidates:
        try:
            raw = ak.stock_us_hist(
                symbol=internal_code,
                period="daily",
                start_date=start_text.replace("-", ""),
                end_date=end_text.replace("-", ""),
                adjust="qfq",
            )
            data = standardize_chinese_ohlcv(raw)
            if not data.empty:
                return data, internal_code
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.DataFrame(), ""


def _candidate_quality(item: tuple[pd.DataFrame, str, str]) -> tuple[int, int]:
    data = item[0]
    if data.empty:
        return (-10_000, 0)
    lag = max(0, (pd.Timestamp(date.today()) - data["日期"].max()).days)
    return (len(data) - lag * 5, len(data))


def fetch_us_security(symbol: str, start_text: str, end_text: str) -> tuple[pd.DataFrame, str, str]:
    code = normalize_us_code(symbol)
    candidates: list[tuple[pd.DataFrame, str, str]] = []
    errors: list[str] = []
    try:
        data, name = fetch_yahoo_chart_history(code, start_text, end_text)
        if not data.empty:
            candidates.append((data, name, "Yahoo Finance图表公开接口（复权日线）"))
            if len(data) >= 500 and (pd.Timestamp(date.today()) - data["日期"].max()).days <= 10:
                return candidates[-1]
    except Exception as exc:
        errors.append(f"Yahoo图表接口：{exc}")
    for source_name, loader in [
        ("yfinance备用日线（自动复权）", lambda: fetch_yfinance_history(code, start_text, end_text)),
        ("Stooq独立备用日线", lambda: fetch_stooq_history(code, start_text, end_text)),
    ]:
        try:
            data = loader()
            if not data.empty:
                candidates.append((data, code, source_name))
        except Exception as exc:
            errors.append(f"{source_name.split('公开')[0]}：{exc}")
    try:
        data, internal_code = fetch_akshare_us_history(code, start_text, end_text)
        if not data.empty:
            candidates.append((data, code, f"AKShare／东方财富美股日线（{internal_code}）"))
    except Exception as exc:
        errors.append(f"AKShare备用：{exc}")
    if candidates:
        return max(candidates, key=_candidate_quality)
    detail = "；".join(errors) if errors else "相关组件未安装或接口无数据"
    raise RuntimeError(f"美股数据通道均未返回有效数据。{detail}")


def fetch_hk_security(code: str, start_text: str, end_text: str) -> tuple[pd.DataFrame, str, str]:
    normalized = normalize_hk_code(code)
    yahoo_symbol = hk_yahoo_ticker(normalized)
    candidates: list[tuple[pd.DataFrame, str, str]] = []
    errors: list[str] = []
    resolved_name = normalized

    if ak is not None:
        try:
            raw = ak.stock_hk_hist(
                symbol=normalized,
                period="daily",
                start_date=start_text.replace("-", ""),
                end_date=end_text.replace("-", ""),
                adjust="qfq",
            )
            data = standardize_chinese_ohlcv(raw)
            if not data.empty:
                candidates.append((data, normalized, "AKShare／东方财富港股公开行情（前复权日线）"))
        except Exception as exc:
            errors.append(f"AKShare／东方财富：{exc}")

    try:
        data, yahoo_name = fetch_yahoo_chart_history(yahoo_symbol, start_text, end_text)
        if not data.empty:
            resolved_name = yahoo_name or normalized
            candidates.append((data, resolved_name, f"Yahoo Finance港股图表公开接口（{yahoo_symbol}，复权日线）"))
    except Exception as exc:
        errors.append(f"Yahoo图表接口：{exc}")

    try:
        data = fetch_yfinance_history(yahoo_symbol, start_text, end_text)
        if not data.empty:
            candidates.append((data, resolved_name, f"yfinance港股备用行情（{yahoo_symbol}，自动复权）"))
    except Exception as exc:
        errors.append(f"yfinance备用：{exc}")

    if ak is not None:
        try:
            raw = ak.stock_hk_daily(symbol=normalized, adjust="qfq")
            data = filter_dates(standardize_english_ohlcv(raw), start_text, end_text)
            if not data.empty:
                candidates.append((data, resolved_name, "AKShare／新浪港股备用行情（前复权日线）"))
        except Exception as exc:
            errors.append(f"AKShare／新浪：{exc}")

    if candidates:
        best = max(candidates, key=_candidate_quality)
        return best[0], resolved_name if resolved_name != normalized else best[1], best[2]
    detail = "；".join(errors) if errors else "相关组件未安装或接口无数据"
    raise RuntimeError(f"港股数据通道均未返回有效数据。{detail}")


def _fetch_sina_fund(code: str, start_text: str, end_text: str) -> pd.DataFrame:
    if ak is None:
        return pd.DataFrame()
    exchange = "sh" if code.startswith("5") else "sz"
    raw = ak.fund_etf_hist_sina(symbol=f"{exchange}{code}")
    return filter_dates(standardize_english_ohlcv(raw), start_text, end_text)


def fetch_a_security(code: str, start_text: str, end_text: str) -> tuple[pd.DataFrame, str, str]:
    normalized = normalize_a_code(code)
    candidates: list[tuple[pd.DataFrame, str, str]] = []
    errors: list[str] = []
    if is_exchange_traded_fund_code(normalized):
        try:
            data = _fetch_sina_fund(normalized, start_text, end_text)
            if not data.empty:
                candidates.append((data, normalized, "AKShare／新浪场内基金日线（未复权）"))
        except Exception as exc:
            errors.append(f"新浪基金：{exc}")
    try:
        bao_code = a_share_baostock_code(normalized)
        data = fetch_baostock_history(bao_code, start_text, end_text, adjustflag="2")
        if not data.empty:
            candidates.append((data, normalized, f"BaoStock公开行情（{bao_code}，前复权日线）"))
    except Exception as exc:
        errors.append(f"BaoStock：{exc}")
    if ak is not None:
        try:
            if is_exchange_traded_fund_code(normalized):
                raw = ak.fund_etf_hist_em(
                    symbol=normalized,
                    period="daily",
                    start_date=start_text.replace("-", ""),
                    end_date=end_text.replace("-", ""),
                    adjust="qfq",
                )
            else:
                raw = ak.stock_zh_a_hist(
                    symbol=normalized,
                    period="daily",
                    start_date=start_text.replace("-", ""),
                    end_date=end_text.replace("-", ""),
                    adjust="qfq",
                    timeout=15,
                )
            data = standardize_chinese_ohlcv(raw, volume_in_lots=True)
            if not data.empty:
                candidates.append((data, normalized, "AKShare／东方财富公开行情（前复权日线）"))
        except Exception as exc:
            errors.append(f"AKShare／东方财富：{exc}")
    try:
        yahoo_symbol = a_share_yahoo_ticker(normalized)
        data, yahoo_name = fetch_yahoo_chart_history(yahoo_symbol, start_text, end_text)
        if not data.empty:
            candidates.append((data, yahoo_name or normalized, f"Yahoo Finance图表备用行情（{yahoo_symbol}）"))
            if len(data) >= 500 and (pd.Timestamp(date.today()) - data["日期"].max()).days <= 10:
                return candidates[-1]
    except Exception as exc:
        errors.append(f"Yahoo图表备用：{exc}")
    try:
        yahoo_symbol = a_share_yahoo_ticker(normalized)
        data = fetch_yfinance_history(yahoo_symbol, start_text, end_text)
        if not data.empty:
            candidates.append((data, normalized, f"yfinance备用行情（{yahoo_symbol}）"))
    except Exception as exc:
        errors.append(f"yfinance备用：{exc}")
    if candidates:
        return max(candidates, key=_candidate_quality)
    detail = "；".join(errors) if errors else "相关组件未安装或接口无数据"
    raise RuntimeError(f"A股数据通道均未返回有效数据。{detail}")


def fetch_a_benchmark(start_text: str, end_text: str) -> pd.DataFrame:
    errors: list[str] = []
    try:
        data = fetch_baostock_history("sh.000300", start_text, end_text, adjustflag="3")
        if not data.empty:
            return data
    except Exception as exc:
        errors.append(str(exc))
    if ak is not None:
        try:
            raw = ak.index_zh_a_hist(
                symbol="000300",
                period="daily",
                start_date=start_text.replace("-", ""),
                end_date=end_text.replace("-", ""),
            )
            data = standardize_chinese_ohlcv(raw)
            if not data.empty:
                return data
        except Exception as exc:
            errors.append(str(exc))
    try:
        data, _ = fetch_yahoo_chart_history("000300.SS", start_text, end_text)
        if not data.empty:
            return data
    except Exception as exc:
        errors.append(str(exc))
    try:
        data = fetch_yfinance_history("000300.SS", start_text, end_text)
        if not data.empty:
            return data
    except Exception as exc:
        errors.append(str(exc))
    raise RuntimeError("沪深300基准获取失败。" + "；".join(errors))


def fetch_us_benchmark(start_text: str, end_text: str) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    try:
        data, _ = fetch_yahoo_chart_history("SPY", start_text, end_text)
        if not data.empty:
            return data
    except Exception:
        pass
    for loader in (
        lambda: fetch_yfinance_history("SPY", start_text, end_text),
        lambda: fetch_stooq_history("SPY", start_text, end_text),
    ):
        try:
            data = loader()
            if not data.empty:
                candidates.append(data)
        except Exception:
            pass
    if not candidates:
        try:
            data, _ = fetch_akshare_us_history("SPY", start_text, end_text)
            if not data.empty:
                candidates.append(data)
        except Exception:
            pass
    if not candidates:
        raise RuntimeError("标普500代理基准SPY获取失败。")
    return max(candidates, key=len)


def fetch_hk_benchmark(start_text: str, end_text: str) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    errors: list[str] = []
    try:
        data, _ = fetch_yahoo_chart_history("^HSI", start_text, end_text)
        if not data.empty:
            candidates.append(data)
    except Exception as exc:
        errors.append(f"Yahoo恒生指数：{exc}")
    try:
        data = fetch_yfinance_history("^HSI", start_text, end_text)
        if not data.empty:
            candidates.append(data)
    except Exception as exc:
        errors.append(f"yfinance恒生指数：{exc}")
    if ak is not None:
        try:
            raw = ak.stock_hk_index_daily_sina(symbol="HSI")
            data = filter_dates(standardize_english_ohlcv(raw), start_text, end_text)
            if not data.empty:
                candidates.append(data)
        except Exception as exc:
            errors.append(f"AKShare／新浪恒生指数：{exc}")
    if not candidates:
        raise RuntimeError("恒生指数基准获取失败。" + "；".join(errors))
    return max(candidates, key=lambda data: _candidate_quality((data, "", "")))


def _bao_first_row(result: Any) -> dict[str, Any]:
    if result.error_code != "0":
        return {}
    frame = _baostock_result_to_frame(result)
    if frame.empty:
        return {}
    return frame.iloc[-1].to_dict()


def _recent_report_quarters(count: int = 10) -> list[tuple[int, int]]:
    today = date.today()
    year = today.year
    quarter = max(1, (today.month - 1) // 3)
    result: list[tuple[int, int]] = []
    for _ in range(count):
        result.append((year, quarter))
        quarter -= 1
        if quarter == 0:
            year -= 1
            quarter = 4
    return result


def _score_fundamentals(fields: dict[str, Any]) -> tuple[float | None, list[str], list[str]]:
    measurable = 0
    score = 50.0
    positives: list[str] = []
    risks: list[str] = []

    roe = ratio_fraction(fields.get("净资产收益率"))
    if roe is not None:
        measurable += 1
        if roe >= 0.15:
            score += 10
            positives.append("净资产收益率相对较高")
        elif roe >= 0.08:
            score += 5
        elif roe < 0:
            score -= 12
            risks.append("净资产收益率为负")

    margin = ratio_fraction(fields.get("净利率"))
    if margin is not None:
        measurable += 1
        if margin >= 0.15:
            score += 8
            positives.append("净利率相对较高")
        elif margin >= 0.05:
            score += 3
        elif margin < 0:
            score -= 10
            risks.append("净利率为负")

    profit_growth = ratio_fraction(fields.get("净利润同比"))
    if profit_growth is not None:
        measurable += 1
        if profit_growth >= 0.15:
            score += 8
            positives.append("净利润同比增长")
        elif profit_growth < 0:
            score -= 9
            risks.append("净利润同比下降")

    revenue_growth = ratio_fraction(fields.get("营收同比"))
    if revenue_growth is not None:
        measurable += 1
        if revenue_growth >= 0.10:
            score += 6
            positives.append("营业收入保持增长")
        elif revenue_growth < 0:
            score -= 6
            risks.append("营业收入同比下降")

    debt_ratio = ratio_fraction(fields.get("资产负债率"))
    if debt_ratio is not None:
        measurable += 1
        if debt_ratio >= 0.75:
            score -= 10
            risks.append("资产负债率较高，需结合行业判断")
        elif debt_ratio >= 0.55:
            score -= 4
        elif debt_ratio <= 0.35:
            score += 4

    cash_quality = ratio_fraction(fields.get("经营现金流／净利润"))
    if cash_quality is not None:
        measurable += 1
        if cash_quality >= 1:
            score += 7
            positives.append("经营现金流对利润覆盖较好")
        elif cash_quality < 0:
            score -= 9
            risks.append("经营现金流与净利润方向不一致")

    pe = safe_float(fields.get("市盈率TTM"))
    if pe is not None:
        measurable += 1
        if pe <= 0:
            score -= 7
            risks.append("市盈率不可用或公司处于亏损状态")
        elif pe > 80:
            score -= 6
            risks.append("市盈率较高，估值容错空间可能较低")
        elif pe < 20:
            score += 3

    if measurable < 2:
        return None, positives, risks
    return float(np.clip(score, 0, 100)), positives, risks


def fetch_a_fundamentals(code: str, last_price: float, asset_type: str) -> EvidenceSnapshot:
    if asset_type != "A股个股":
        return EvidenceSnapshot(False, "不适用", notes=["场内基金不使用单一上市公司的财务报表评分。"])
    if bs is None:
        return EvidenceSnapshot(False, "BaoStock未安装", notes=["未取得财务数据，不参与评分。"])
    bao_code = a_share_baostock_code(code)
    login_result = bs.login()
    if login_result.error_code != "0":
        return EvidenceSnapshot(False, "BaoStock", notes=[f"财务接口登录失败：{login_result.error_msg}"])
    fields: dict[str, Any] = {}
    notes: list[str] = []
    name = code
    industry = "未取得"
    try:
        basic = _bao_first_row(bs.query_stock_basic(code=bao_code))
        name = str(basic.get("code_name") or code)
        industry_row = _bao_first_row(bs.query_stock_industry(code=bao_code))
        industry = str(industry_row.get("industry") or "未取得")
        latest_period = ""
        for year, quarter in _recent_report_quarters():
            profit = _bao_first_row(bs.query_profit_data(code=bao_code, year=year, quarter=quarter))
            if not profit:
                continue
            growth = _bao_first_row(bs.query_growth_data(code=bao_code, year=year, quarter=quarter))
            balance = _bao_first_row(bs.query_balance_data(code=bao_code, year=year, quarter=quarter))
            cashflow = _bao_first_row(bs.query_cash_flow_data(code=bao_code, year=year, quarter=quarter))
            latest_period = str(profit.get("statDate") or f"{year}Q{quarter}")
            previous_profit = _bao_first_row(bs.query_profit_data(code=bao_code, year=year - 1, quarter=quarter))
            current_revenue = safe_float(profit.get("MBRevenue"))
            previous_revenue = safe_float(previous_profit.get("MBRevenue"))
            revenue_growth = (
                current_revenue / previous_revenue - 1
                if current_revenue is not None and previous_revenue not in {None, 0}
                else None
            )
            fields.update(
                {
                    "公司名称": name,
                    "行业": industry,
                    "报告期": latest_period,
                    "净资产收益率": safe_float(profit.get("roeAvg")),
                    "净利率": safe_float(profit.get("npMargin")),
                    "净利润同比": safe_float(growth.get("YOYNI")),
                    "营收同比": revenue_growth,
                    "资产负债率": safe_float(balance.get("liabilityToAsset")),
                    "经营现金流／净利润": safe_float(cashflow.get("CFOToNP")),
                    "每股收益TTM": safe_float(profit.get("epsTTM")),
                }
            )
            break
        eps = safe_float(fields.get("每股收益TTM"))
        fields["市盈率TTM"] = last_price / eps if eps is not None and eps != 0 else None
        if not latest_period:
            notes.append("公开接口没有返回近期财务报告。")
    except Exception as exc:
        notes.append(f"部分财务数据获取失败：{exc}")
    finally:
        bs.logout()
    score, positives, risks = _score_fundamentals(fields)
    return EvidenceSnapshot(score is not None, "BaoStock公开财务数据", fields, score, positives, risks, notes)


def _sec_fact_series(
    facts: dict[str, Any],
    tags: tuple[str, ...],
    unit_names: tuple[str, ...],
    as_of: date | None = None,
) -> list[dict[str, Any]]:
    for tag in tags:
        fact = facts.get(tag)
        if not fact:
            continue
        units = fact.get("units", {})
        for unit_name in unit_names:
            observations = units.get(unit_name)
            if observations:
                annual = [
                    item
                    for item in observations
                    if item.get("form") in {"10-K", "20-F", "40-F"}
                    and item.get("fp") == "FY"
                    and (
                        as_of is None
                        or (
                            item.get("filed")
                            and pd.Timestamp(str(item.get("filed"))).date() <= as_of
                        )
                    )
                ]
                annual.sort(key=lambda item: (str(item.get("end", "")), str(item.get("filed", ""))))
                unique_by_end: dict[str, dict[str, Any]] = {}
                for item in annual:
                    unique_by_end[str(item.get("end", ""))] = item
                return list(unique_by_end.values())
    return []


def _last_value(series: list[dict[str, Any]]) -> float | None:
    return safe_float(series[-1].get("val")) if series else None


def fetch_sec_fundamentals(
    symbol: str,
    last_price: float | None,
    price_history: pd.DataFrame | None = None,
    as_of: date | None = None,
) -> EvidenceSnapshot:
    ticker_response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=20)
    ticker_response.raise_for_status()
    ticker_items = ticker_response.json().values()
    matched = next((item for item in ticker_items if str(item.get("ticker", "")).upper() == symbol.upper()), None)
    if not matched:
        return EvidenceSnapshot(False, "美国SEC", notes=["SEC代码表未找到该证券，可能是ETF或非美国申报主体。"])
    cik = f"{int(matched['cik_str']):010d}"
    facts_response = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=SEC_HEADERS, timeout=30)
    facts_response.raise_for_status()
    payload = facts_response.json()
    facts = payload.get("facts", {}).get("us-gaap", {})
    revenue_series = _sec_fact_series(facts, ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"), ("USD",), as_of=as_of)
    income_series = _sec_fact_series(facts, ("NetIncomeLoss", "ProfitLoss"), ("USD",), as_of=as_of)
    equity_series = _sec_fact_series(facts, ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), ("USD",), as_of=as_of)
    assets_series = _sec_fact_series(facts, ("Assets",), ("USD",), as_of=as_of)
    liabilities_series = _sec_fact_series(facts, ("Liabilities",), ("USD",), as_of=as_of)
    cashflow_series = _sec_fact_series(facts, ("NetCashProvidedByUsedInOperatingActivities",), ("USD",), as_of=as_of)
    eps_series = _sec_fact_series(facts, ("EarningsPerShareDiluted", "EarningsPerShareBasic"), ("USD/shares", "USD / shares"), as_of=as_of)
    revenue = _last_value(revenue_series)
    prior_revenue = safe_float(revenue_series[-2].get("val")) if len(revenue_series) >= 2 else None
    income = _last_value(income_series)
    prior_income = safe_float(income_series[-2].get("val")) if len(income_series) >= 2 else None
    equity = _last_value(equity_series)
    assets = _last_value(assets_series)
    liabilities = _last_value(liabilities_series)
    cashflow = _last_value(cashflow_series)
    eps = _last_value(eps_series)
    report_item = revenue_series[-1] if revenue_series else income_series[-1] if income_series else None
    report_end = report_item.get("end") if report_item else "最近年度"
    fields = {
        "公司名称": payload.get("entityName") or matched.get("title") or symbol,
        "行业": "请结合SEC申报行业另行核对",
        "报告期": report_end,
        "披露日期": report_item.get("filed") if report_item else None,
        "净资产收益率": income / equity if income is not None and equity not in {None, 0} else None,
        "净利率": income / revenue if income is not None and revenue not in {None, 0} else None,
        "净利润同比": income / prior_income - 1 if income is not None and prior_income not in {None, 0} else None,
        "营收同比": revenue / prior_revenue - 1 if revenue is not None and prior_revenue not in {None, 0} else None,
        "资产负债率": liabilities / assets if liabilities is not None and assets not in {None, 0} else None,
        "经营现金流／净利润": cashflow / income if cashflow is not None and income not in {None, 0} else None,
        "市盈率TTM": last_price / eps if last_price is not None and eps not in {None, 0} else None,
    }
    score, positives, risks = _score_fundamentals(fields)
    notes = [
        "SEC公司事实数据采用最近可得年度申报口径，市盈率为最新价格除以年度每股收益的简化值。",
        "历史测试时只使用实际披露日期不晚于T的SEC事实；缺失数据不回填。",
    ]
    return EvidenceSnapshot(score is not None, "美国SEC Companyfacts公开申报数据", fields, score, positives, risks, notes)


def fetch_us_fundamentals(symbol: str, last_price: float | None = None) -> EvidenceSnapshot:
    sec_notes: list[str] = []
    try:
        sec_result = fetch_sec_fundamentals(symbol, last_price)
        if sec_result.available:
            return sec_result
        sec_notes.extend(sec_result.notes)
    except Exception as exc:
        sec_notes.append(f"SEC财务接口暂不可用：{exc}")
    if yf is None:
        return EvidenceSnapshot(False, "美国SEC／yfinance", notes=sec_notes + ["未取得财务数据，不参与评分。"])
    fields: dict[str, Any] = {}
    notes: list[str] = []
    try:
        info = yf.Ticker(symbol).get_info()
        fields = {
            "公司名称": info.get("longName") or info.get("shortName") or symbol,
            "行业": info.get("industry") or info.get("sector") or "未取得",
            "报告期": "Yahoo Finance最近可得口径",
            "净资产收益率": safe_float(info.get("returnOnEquity")),
            "净利率": safe_float(info.get("profitMargins")),
            "净利润同比": safe_float(info.get("earningsGrowth")),
            "营收同比": safe_float(info.get("revenueGrowth")),
            "资产负债率": None,
            "经营现金流／净利润": None,
            "市盈率TTM": safe_float(info.get("trailingPE")),
            "市净率": safe_float(info.get("priceToBook")),
            "总市值": safe_float(info.get("marketCap")),
        }
        debt_to_equity = safe_float(info.get("debtToEquity"))
        if debt_to_equity is not None:
            fields["债务／权益"] = debt_to_equity / 100 if abs(debt_to_equity) > 5 else debt_to_equity
    except Exception as exc:
        notes.append(f"Yahoo财务接口暂不可用：{exc}")
    score, positives, risks = _score_fundamentals(fields)
    return EvidenceSnapshot(score is not None, "Yahoo Finance公开公司资料", fields, score, positives, risks, sec_notes + notes)


def fetch_hk_fundamentals(code: str, last_price: float | None = None) -> EvidenceSnapshot:
    normalized = normalize_hk_code(code)
    symbol = hk_yahoo_ticker(normalized)
    if yf is None:
        return EvidenceSnapshot(False, "yfinance未安装", notes=["未取得港股公司资料，不参与基本面评分。"])
    fields: dict[str, Any] = {}
    notes: list[str] = []
    try:
        info = yf.Ticker(symbol).get_info()
        fields = {
            "公司名称": info.get("longName") or info.get("shortName") or normalized,
            "行业": info.get("industry") or info.get("sector") or "未取得",
            "报告期": "Yahoo Finance最近可得口径",
            "净资产收益率": safe_float(info.get("returnOnEquity")),
            "净利率": safe_float(info.get("profitMargins")),
            "净利润同比": safe_float(info.get("earningsGrowth")),
            "营收同比": safe_float(info.get("revenueGrowth")),
            "资产负债率": None,
            "经营现金流／净利润": None,
            "市盈率TTM": safe_float(info.get("trailingPE")),
            "市净率": safe_float(info.get("priceToBook")),
            "总市值": safe_float(info.get("marketCap")),
        }
        debt_to_equity = safe_float(info.get("debtToEquity"))
        if debt_to_equity is not None:
            fields["债务／权益"] = debt_to_equity / 100 if abs(debt_to_equity) > 5 else debt_to_equity
    except Exception as exc:
        notes.append(f"Yahoo港股公司资料暂不可用：{exc}")
    score, positives, risks = _score_fundamentals(fields)
    return EvidenceSnapshot(score is not None, "Yahoo Finance港股公开公司资料", fields, score, positives, risks, notes)


def derive_market_regime(benchmark: pd.DataFrame) -> dict[str, Any]:
    if benchmark is None or benchmark.empty:
        return {
            "市场状态": "基准数据暂不可用",
            "市场分": 50,
            "近60日收益": np.nan,
            "近250日收益": np.nan,
            "近60日年化波动": np.nan,
        }
    close = benchmark.set_index("日期")["收盘"].sort_index()
    returns = close.pct_change().dropna()
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else close.mean()
    ma250 = close.rolling(250).mean().iloc[-1] if len(close) >= 250 else close.mean()
    r60 = close.iloc[-1] / close.iloc[-min(61, len(close))] - 1 if len(close) > 1 else 0.0
    r250 = close.iloc[-1] / close.iloc[-min(251, len(close))] - 1 if len(close) > 1 else 0.0
    volatility = returns.tail(min(60, len(returns))).std() * np.sqrt(252) if not returns.empty else np.nan
    if close.iloc[-1] >= ma60 and close.iloc[-1] >= ma250 and r60 > 0:
        regime = "偏强"
        score = 60
    elif close.iloc[-1] < ma60 and close.iloc[-1] < ma250 and r60 < 0:
        regime = "偏弱"
        score = 38
    else:
        regime = "震荡／方向不明"
        score = 50
    if np.isfinite(volatility) and volatility > 0.30:
        score -= 5
    return {
        "市场状态": regime,
        "市场分": int(np.clip(score, 0, 100)),
        "近60日收益": r60,
        "近250日收益": r250,
        "近60日年化波动": volatility,
    }


def _find_column(frame: pd.DataFrame, patterns: tuple[str, ...]) -> str | None:
    for column in frame.columns:
        text = str(column).lower()
        if any(pattern.lower() in text for pattern in patterns):
            return str(column)
    return None


def fetch_macro_snapshot(market: str, benchmark: pd.DataFrame) -> EvidenceSnapshot:
    regime = derive_market_regime(benchmark)
    fields: dict[str, Any] = dict(regime)
    notes: list[str] = []
    score = float(regime["市场分"])
    if benchmark is None or benchmark.empty:
        notes.append("市场基准暂未取得，宏观环境仅按中性处理，不影响个股行情继续分析。")
    if market == "A股" and ak is not None:
        try:
            raw = ak.macro_china_lpr()
            date_col = _find_column(raw, ("trade_date", "date", "日期"))
            rate_col = _find_column(raw, ("lpr1y", "1年", "1y"))
            if date_col and rate_col:
                temp = raw[[date_col, rate_col]].copy()
                temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
                temp[rate_col] = pd.to_numeric(temp[rate_col], errors="coerce")
                temp = temp.dropna().sort_values(date_col)
                if not temp.empty:
                    latest = float(temp[rate_col].iloc[-1])
                    previous = float(temp[rate_col].iloc[max(0, len(temp) - 7)])
                    direction = "下降" if latest < previous else "上升" if latest > previous else "持平"
                    fields.update({"利率指标": "1年期LPR", "最新利率": latest, "近半年方向": direction})
                    score += 3 if direction == "下降" else -3 if direction == "上升" else 0
        except Exception as exc:
            notes.append(f"LPR数据暂不可用：{exc}")
    elif market == "美股":
        try:
            response = requests.get(
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
                headers=HTTP_HEADERS,
                timeout=15,
            )
            response.raise_for_status()
            temp = pd.read_csv(StringIO(response.text))
            value_col = _find_column(temp, ("dgs10",))
            if value_col:
                values = pd.to_numeric(temp[value_col], errors="coerce").dropna()
                if not values.empty:
                    latest = float(values.iloc[-1])
                    previous = float(values.iloc[max(0, len(values) - 60)])
                    direction = "下降" if latest < previous - 0.10 else "上升" if latest > previous + 0.10 else "持平"
                    fields.update({"利率指标": "美国10年期国债收益率", "最新利率": latest, "近阶段方向": direction})
                    score += 3 if direction == "下降" else -3 if direction == "上升" else 0
        except Exception as exc:
            notes.append(f"美国利率数据暂不可用：{exc}")
    elif market == "港股":
        notes.append("港股宏观环境以恒生指数的趋势、收益与波动状态为主要修正依据。")
    score = float(np.clip(score, 0, 100))
    return EvidenceSnapshot(True, "市场基准与公开宏观数据", fields, score, notes=notes)


def fetch_price_bundle(market: str, raw_code: str) -> PriceBundle:
    end_text = date.today().strftime("%Y-%m-%d")
    start_text = (pd.Timestamp(end_text) - pd.DateOffset(years=HISTORY_YEARS)).strftime("%Y-%m-%d")
    warnings: list[str] = []
    if market == "A股":
        code = normalize_a_code(raw_code)
        stock, name, provider = fetch_a_security(code, start_text, end_text)
        try:
            benchmark = fetch_a_benchmark(start_text, end_text)
        except Exception as exc:
            benchmark = pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘", "成交量"])
            warnings.append(f"沪深300基准暂不可用：{exc}")
        asset_type = "场内基金" if is_exchange_traded_fund_code(code) else "A股个股"
        bundle = PriceBundle(stock, benchmark, code, name, provider, "沪深300", asset_type, "人民币元", warnings)
    elif market == "美股":
        code = normalize_us_code(raw_code)
        stock, name, provider = fetch_us_security(code, start_text, end_text)
        try:
            benchmark = fetch_us_benchmark(start_text, end_text)
        except Exception as exc:
            benchmark = pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘", "成交量"])
            warnings.append(f"标普500代理基准SPY暂不可用：{exc}")
        bundle = PriceBundle(stock, benchmark, code, name, provider, "标普500代理ETF（SPY）", "美股个股", "美元", warnings)
    elif market == "港股":
        code = normalize_hk_code(raw_code)
        stock, name, provider = fetch_hk_security(code, start_text, end_text)
        try:
            benchmark = fetch_hk_benchmark(start_text, end_text)
        except Exception as exc:
            benchmark = pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘", "成交量"])
            warnings.append(f"恒生指数基准暂不可用：{exc}")
        bundle = PriceBundle(stock, benchmark, code, name, provider, "恒生指数（HSI）", "港股个股", "港元", warnings)
    else:
        raise ValueError("市场仅支持A股、美股或港股。")

    first_date = pd.Timestamp(bundle.stock["日期"].min())
    requested_start = pd.Timestamp(start_text)
    expected_span = max(1, (pd.Timestamp(end_text) - requested_start).days)
    actual_span = max(0, (pd.Timestamp(end_text) - first_date).days)
    bundle.requested_start = start_text
    bundle.requested_end = end_text
    bundle.coverage_ratio = float(np.clip(actual_span / expected_span, 0, 1))
    bundle.history_complete = bool(first_date <= requested_start + pd.Timedelta(days=45))
    if not bundle.history_complete:
        bundle.warnings.append("该股票可得历史不足五年。数据不足，无法准确判断，结果仅作低置信度参考。")
    return bundle


def calculate_quant_metrics(stock: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, Any]:
    close = stock.set_index("日期")["收盘"].sort_index()
    volume = stock.set_index("日期")["成交量"].sort_index()
    returns = close.pct_change().dropna()
    drawdown = close / close.cummax() - 1
    recent_returns = returns.tail(min(252, len(returns)))
    if benchmark is None or benchmark.empty:
        benchmark_close = pd.Series(dtype="float64", name="收盘")
        benchmark_returns = pd.Series(dtype="float64")
    else:
        benchmark_close = benchmark.set_index("日期")["收盘"].sort_index()
        benchmark_returns = benchmark_close.pct_change().dropna()
    aligned = pd.concat([returns.rename("stock"), benchmark_returns.rename("benchmark")], axis=1).dropna().tail(756)
    beta = np.nan
    correlation = np.nan
    if len(aligned) >= 60 and aligned["benchmark"].var() > 0:
        beta = aligned["stock"].cov(aligned["benchmark"]) / aligned["benchmark"].var()
        correlation = aligned["stock"].corr(aligned["benchmark"])
    volatility = recent_returns.std() * np.sqrt(252) if not recent_returns.empty else np.nan
    downside = recent_returns[recent_returns < 0].std() * np.sqrt(252) if (recent_returns < 0).any() else 0.0
    value_at_risk = recent_returns.quantile(0.05) if not recent_returns.empty else np.nan
    abnormal_days = int((returns.abs() > 0.70).sum())
    latest_lag = max(0, (pd.Timestamp(date.today()) - close.index.max()).days)
    volume20 = volume.tail(20).mean()
    volume60 = volume.tail(60).mean()
    volume_ratio = volume20 / volume60 if volume60 > 0 else np.nan
    return {
        "close": close,
        "volume": volume,
        "returns": returns,
        "drawdown": drawdown,
        "latest_price": float(close.iloc[-1]),
        "first_date": close.index.min(),
        "last_date": close.index.max(),
        "rows": len(close),
        "annual_volatility": float(volatility) if np.isfinite(volatility) else np.nan,
        "downside_volatility": float(downside) if np.isfinite(downside) else np.nan,
        "max_drawdown": float(drawdown.min()),
        "var95_daily": float(value_at_risk) if np.isfinite(value_at_risk) else np.nan,
        "beta": float(beta) if np.isfinite(beta) else np.nan,
        "correlation": float(correlation) if np.isfinite(correlation) else np.nan,
        "volume_ratio": float(volume_ratio) if np.isfinite(volume_ratio) else np.nan,
        "abnormal_days": abnormal_days,
        "latest_lag": latest_lag,
        "benchmark_close": benchmark_close,
    }


def _rolling_downside_volatility(returns: pd.Series, window: int) -> pd.Series:
    def calculate(values: np.ndarray) -> float:
        downside = values[np.isfinite(values) & (values < 0)]
        if len(downside) < 5:
            return np.nan
        return float(np.std(downside, ddof=1) * np.sqrt(252))

    return returns.rolling(window).apply(calculate, raw=True)


def build_analog_feature_frame(stock: pd.DataFrame, benchmark: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.Series]:
    prices = stock.copy()
    prices["日期"] = pd.to_datetime(prices["日期"], errors="coerce")
    prices = prices.dropna(subset=["日期", "收盘"]).drop_duplicates("日期").sort_values("日期")
    close = pd.to_numeric(prices.set_index("日期")["收盘"], errors="coerce").dropna()
    volume = pd.to_numeric(prices.set_index("日期")["成交量"], errors="coerce").reindex(close.index)
    returns = close.pct_change()
    features = pd.DataFrame(index=close.index)

    for days in (5, 20, 60, 120):
        features[f"return_{days}"] = close.pct_change(days)
    features["volatility_20"] = returns.rolling(20).std() * np.sqrt(252)
    features["volatility_60"] = returns.rolling(60).std() * np.sqrt(252)
    features["downside_volatility_60"] = _rolling_downside_volatility(returns, 60)
    features["drawdown_60"] = close / close.rolling(60).max() - 1
    features["drawdown_120"] = close / close.rolling(120).max() - 1
    features["price_ma20"] = close / close.rolling(20).mean() - 1
    features["price_ma60"] = close / close.rolling(60).mean() - 1
    average_volume_20 = volume.rolling(20).mean()
    average_volume_60 = volume.rolling(60).mean()
    features["volume_ratio"] = average_volume_20 / average_volume_60

    benchmark_close = pd.Series(dtype="float64")
    if benchmark is not None and not benchmark.empty:
        benchmark_prices = benchmark.copy()
        benchmark_prices["日期"] = pd.to_datetime(benchmark_prices["日期"], errors="coerce")
        benchmark_prices = benchmark_prices.dropna(subset=["日期", "收盘"]).drop_duplicates("日期").sort_values("日期")
        benchmark_close = pd.to_numeric(benchmark_prices.set_index("日期")["收盘"], errors="coerce").dropna()
        benchmark_close = benchmark_close.reindex(close.index).ffill()
        benchmark_returns = benchmark_close.pct_change(fill_method=None)
        features["benchmark_return_20"] = benchmark_close.pct_change(20, fill_method=None)
        features["benchmark_return_60"] = benchmark_close.pct_change(60, fill_method=None)
        features["benchmark_volatility_20"] = benchmark_returns.rolling(20).std() * np.sqrt(252)
        features["relative_return_20"] = features["return_20"] - features["benchmark_return_20"]
        features["relative_return_60"] = features["return_60"] - features["benchmark_return_60"]

    features = features.replace([np.inf, -np.inf], np.nan)
    return features, close


def _similarity_scores(
    candidate_features: pd.DataFrame,
    current_features: pd.Series,
    weights: dict[str, float],
) -> pd.Series:
    if candidate_features.empty:
        return pd.Series(dtype="float64")
    columns = [
        column
        for column in weights
        if column in candidate_features.columns
        and column in current_features.index
        and np.isfinite(current_features[column])
    ]
    if len(columns) < 8:
        return pd.Series(dtype="float64")
    candidates = candidate_features[columns].dropna()
    if candidates.empty:
        return pd.Series(dtype="float64")
    first_quartile = candidates.quantile(0.25)
    third_quartile = candidates.quantile(0.75)
    scale = third_quartile - first_quartile
    fallback = candidates.std().replace(0, np.nan)
    scale = scale.where(scale.abs() > 1e-8, fallback).replace(0, np.nan).fillna(1.0)
    differences = ((candidates - current_features[columns]) / scale).clip(-8, 8)
    normalized_weights = pd.Series({column: weights[column] for column in columns}, dtype="float64")
    normalized_weights = normalized_weights / normalized_weights.sum()
    distance = np.sqrt(differences.pow(2).mul(normalized_weights, axis=1).sum(axis=1))
    absolute_similarity = 100 * np.exp(-0.35 * distance)
    relative_similarity = absolute_similarity.rank(method="average", pct=True) * 100
    return absolute_similarity * 0.70 + relative_similarity * 0.30


def _select_spaced_matches(
    similarities: pd.Series,
    date_positions: dict[pd.Timestamp, int],
    minimum_gap: int,
    limit: int,
    minimum_similarity: float,
) -> list[tuple[pd.Timestamp, int, float]]:
    selected: list[tuple[pd.Timestamp, int, float]] = []
    for timestamp, similarity in similarities.sort_values(ascending=False).items():
        if float(similarity) < minimum_similarity:
            continue
        position = date_positions.get(pd.Timestamp(timestamp))
        if position is None:
            continue
        if any(abs(position - existing_position) < minimum_gap for _, existing_position, _ in selected):
            continue
        selected.append((pd.Timestamp(timestamp), position, float(similarity)))
        if len(selected) >= limit:
            break
    return selected


def _analog_state_summary(features: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    current = features.iloc[-1]
    trend = "震荡"
    if current.get("return_20", np.nan) > 0 and current.get("price_ma60", np.nan) > 0:
        trend = "偏强上行"
    elif current.get("return_20", np.nan) < 0 and current.get("price_ma60", np.nan) < 0:
        trend = "偏弱下行"

    volatility_series = features["volatility_20"].dropna()
    volatility_percentile = float((volatility_series <= current.get("volatility_20", np.nan)).mean()) if not volatility_series.empty else np.nan
    if not np.isfinite(volatility_percentile):
        volatility_state = "波动数据不足"
    elif volatility_percentile >= 0.75:
        volatility_state = "高波动"
    elif volatility_percentile <= 0.25:
        volatility_state = "低波动"
    else:
        volatility_state = "常态波动"

    drawdown = current.get("drawdown_120", np.nan)
    if not np.isfinite(drawdown):
        drawdown_state = "回撤数据不足"
    elif drawdown <= -0.20:
        drawdown_state = "深度回撤"
    elif drawdown <= -0.08:
        drawdown_state = "中等回撤"
    else:
        drawdown_state = "接近阶段高位"

    benchmark_return = current.get("benchmark_return_20", np.nan)
    if not np.isfinite(benchmark_return):
        market_state = "市场基准数据不足"
    elif benchmark_return >= 0.03:
        market_state = "市场环境偏强"
    elif benchmark_return <= -0.03:
        market_state = "市场环境偏弱"
    else:
        market_state = "市场环境震荡"

    state = {
        "trend": trend,
        "volatility": volatility_state,
        "drawdown": drawdown_state,
        "market": market_state,
        "summary": f"{trend} · {volatility_state} · {drawdown_state} · {market_state}",
    }
    display_features = {
        "近5日收益": current.get("return_5"),
        "近20日收益": current.get("return_20"),
        "近60日收益": current.get("return_60"),
        "20日年化波动": current.get("volatility_20"),
        "60日年化波动": current.get("volatility_60"),
        "距120日高点": current.get("drawdown_120"),
        "20日／60日成交量比": current.get("volume_ratio"),
        "近20日相对基准": current.get("relative_return_20"),
        "基准近20日收益": current.get("benchmark_return_20"),
    }
    return state, display_features


def _walk_forward_analog_backtest(
    features: pd.DataFrame,
    close: pd.Series,
    weights: dict[str, float],
    horizon: int = 20,
) -> dict[str, Any]:
    date_positions = {pd.Timestamp(timestamp): position for position, timestamp in enumerate(close.index)}
    cases: list[dict[str, Any]] = []
    start_position = max(360, int(len(close) * 0.35))
    for evaluation_position in range(start_position, len(close) - horizon, 20):
        evaluation_date = pd.Timestamp(close.index[evaluation_position])
        if evaluation_date not in features.index:
            continue
        current = features.loc[evaluation_date]
        last_training_position = evaluation_position - horizon
        training_dates = close.index[120:last_training_position]
        training = features.reindex(training_dates)
        similarities = _similarity_scores(training, current, weights)
        selected = _select_spaced_matches(similarities, date_positions, 20, 12, 35.0)
        outcomes: list[float] = []
        for _, position, _ in selected:
            if position + horizon >= evaluation_position:
                continue
            outcomes.append(float(close.iloc[position + horizon] / close.iloc[position] - 1))
        if len(outcomes) < 6:
            continue
        prediction = float(np.median(outcomes))
        actual = float(close.iloc[evaluation_position + horizon] / close.iloc[evaluation_position] - 1)
        momentum = float(close.iloc[evaluation_position] / close.iloc[evaluation_position - horizon] - 1)
        cases.append(
            {
                "date": evaluation_date,
                "prediction": prediction,
                "actual": actual,
                "correct": (prediction >= 0) == (actual >= 0),
                "momentum_correct": (momentum >= 0) == (actual >= 0),
                "absolute_error": abs(prediction - actual),
            }
        )
    if not cases:
        return {"available": False, "cases": 0, "note": "可回测时点不足。"}
    return {
        "available": True,
        "cases": len(cases),
        "direction_accuracy": float(np.mean([item["correct"] for item in cases])),
        "momentum_accuracy": float(np.mean([item["momentum_correct"] for item in cases])),
        "median_absolute_error": float(np.median([item["absolute_error"] for item in cases])),
        "note": "采用滚动时点验证；每次只使用该时点之前已经可见的数据。",
    }


def _unavailable_analog_horizon(config: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **config,
        "available": False,
        "sample_count": 0,
        "strict_sample_count": 0,
        "adaptive_sample_count": 0,
        "selection_mode": "未形成样本",
        "selection_threshold": ANALOG_MIN_SIMILARITY,
        "positive_ratio": None,
        "median_return": None,
        "q10_return": None,
        "q25_return": None,
        "q75_return": None,
        "median_worst_loss": None,
        "average_similarity": None,
        "confidence_score": 0,
        "direction": "样本不足，不形成预测",
        "reason": reason,
        "outcomes": [],
        "paths": [],
    }


def analyze_historical_analogs(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    history_complete: bool = True,
    source_label: str = "目标股票",
) -> dict[str, Any]:
    features, close = build_analog_feature_frame(stock, benchmark)
    notes = [
        "相似周期仅表示历史状态接近，不代表未来一定重复。",
        "上涨占比是历史样本频率，不是经过校准的真实上涨概率。",
    ]
    if len(close) < 360 or features.empty:
        reason = f"当前只有{len(close)}个交易日；至少需要约360个交易日才能建立状态特征与后续检验。"
        return {
            "available": False,
            "source_label": source_label,
            "candidate_count": 0,
            "best_similarity": None,
            "confidence_score": 0,
            "confidence_label": "样本不足",
            "state": {"summary": "历史数据不足，无法识别稳定状态"},
            "current_features": {},
            "horizons": [_unavailable_analog_horizon(config, reason) for config in ANALOG_HORIZONS],
            "matches": [],
            "backtest": {"available": False, "cases": 0, "note": "至少需要约360个交易日。"},
            "notes": notes + [reason],
        }

    state, current_display_features = _analog_state_summary(features)
    current_date = pd.Timestamp(close.index[-1])
    current_features = features.loc[current_date]
    maximum_forward = max(item["days"] for item in ANALOG_HORIZONS)
    date_positions = {pd.Timestamp(timestamp): position for position, timestamp in enumerate(close.index)}
    latest_candidate_position = len(close) - maximum_forward - 1
    candidate_dates = close.index[120 : max(120, latest_candidate_position + 1)]
    candidate_features = features.reindex(candidate_dates)
    similarities = _similarity_scores(candidate_features, current_features, ANALOG_FEATURE_WEIGHTS)
    eligible_similarities = similarities[similarities >= ANALOG_MIN_SIMILARITY]

    global_matches = _select_spaced_matches(
        similarities,
        date_positions,
        minimum_gap=20,
        limit=10,
        minimum_similarity=ANALOG_MIN_SIMILARITY,
    )
    global_match_mode = "严格同股样本"
    if len(global_matches) < 6:
        adaptive_global_matches = _select_spaced_matches(
            similarities,
            date_positions,
            minimum_gap=20,
            limit=10,
            minimum_similarity=ANALOG_FALLBACK_MIN_SIMILARITY,
        )
        if len(adaptive_global_matches) > len(global_matches):
            global_matches = adaptive_global_matches
            global_match_mode = "自适应同股样本"
    match_rows: list[dict[str, Any]] = []
    for timestamp, position, similarity in global_matches:
        row: dict[str, Any] = {
            "start_date": pd.Timestamp(close.index[max(0, position - 59)]),
            "anchor_date": timestamp,
            "similarity": similarity,
        }
        for config in ANALOG_HORIZONS:
            days = config["days"]
            row[f"return_{days}"] = float(close.iloc[position + days] / close.iloc[position] - 1)
        match_rows.append(row)

    horizon_results: list[dict[str, Any]] = []
    for config in ANALOG_HORIZONS:
        days = int(config["days"])
        strict_selected = _select_spaced_matches(
            similarities,
            date_positions,
            minimum_gap=int(config["minimum_gap"]),
            limit=20,
            minimum_similarity=ANALOG_MIN_SIMILARITY,
        )
        selected = strict_selected
        fallback_used = False
        selection_threshold = ANALOG_MIN_SIMILARITY
        if len(strict_selected) < ANALOG_MIN_SAMPLES:
            adaptive_selected = _select_spaced_matches(
                similarities,
                date_positions,
                minimum_gap=int(config["minimum_gap"]),
                limit=20,
                minimum_similarity=ANALOG_FALLBACK_MIN_SIMILARITY,
            )
            if len(adaptive_selected) > len(strict_selected):
                selected = adaptive_selected
                fallback_used = True
                selection_threshold = ANALOG_FALLBACK_MIN_SIMILARITY
        outcomes: list[float] = []
        worst_losses: list[float] = []
        paths: list[dict[str, Any]] = []
        similarities_used: list[float] = []
        for timestamp, position, similarity in selected:
            if position + days >= len(close):
                continue
            entry = float(close.iloc[position])
            future = close.iloc[position + 1 : position + days + 1]
            outcome = float(future.iloc[-1] / entry - 1)
            worst_loss = float(future.min() / entry - 1)
            outcomes.append(outcome)
            worst_losses.append(worst_loss)
            similarities_used.append(similarity)
            paths.append(
                {
                    "anchor_date": timestamp,
                    "similarity": similarity,
                    "values": [0.0] + [float(value / entry - 1) for value in future.tolist()],
                }
            )
        sample_count = len(outcomes)
        strict_sample_count = len(strict_selected)
        available = sample_count >= ANALOG_MIN_SAMPLES
        if available:
            positive_ratio = float(np.mean(np.asarray(outcomes) > 0))
            median_return = float(np.median(outcomes))
            if positive_ratio >= 0.65 and median_return > 0:
                direction = "历史情景偏正面"
            elif positive_ratio <= 0.35 and median_return < 0:
                direction = "历史情景偏负面"
            else:
                direction = "历史情景分歧／中性"
            average_similarity = float(np.mean(similarities_used))
            sample_component = min(sample_count / 15, 1.0) * 45
            quality_component = float(
                np.clip(
                    (average_similarity - ANALOG_FALLBACK_MIN_SIMILARITY) / 25,
                    0,
                    1,
                )
                * 30
            )
            horizon_confidence = int(
                np.clip(
                    sample_component
                    + quality_component
                    + (10 if history_complete else 0)
                    + (10 if not fallback_used else -12),
                    0,
                    95,
                )
            )
            if not history_complete:
                horizon_confidence = max(0, horizon_confidence - 20)
            if fallback_used:
                reason = (
                    f"严格样本{strict_sample_count}/{ANALOG_MIN_SAMPLES}个；"
                    f"使用相似度不低于{ANALOG_FALLBACK_MIN_SIMILARITY:.0f}分的"
                    f"自适应同股样本{sample_count}个，并下调可信度。"
                )
                selection_mode = "自适应同股样本"
            else:
                reason = f"严格同股样本{sample_count}个，达到最低样本要求。"
                selection_mode = "严格同股样本"
            horizon_results.append(
                {
                    **config,
                    "available": True,
                    "sample_count": sample_count,
                    "strict_sample_count": strict_sample_count,
                    "adaptive_sample_count": sample_count,
                    "selection_mode": selection_mode,
                    "selection_threshold": selection_threshold,
                    "positive_ratio": positive_ratio,
                    "median_return": median_return,
                    "q10_return": float(np.quantile(outcomes, 0.10)),
                    "q25_return": float(np.quantile(outcomes, 0.25)),
                    "q75_return": float(np.quantile(outcomes, 0.75)),
                    "median_worst_loss": float(np.median(worst_losses)),
                    "average_similarity": average_similarity,
                    "confidence_score": horizon_confidence,
                    "direction": direction,
                    "reason": reason,
                    "outcomes": outcomes,
                    "paths": paths,
                }
            )
        else:
            best_similarity = float(similarities.max()) if not similarities.empty else None
            reason = (
                f"严格样本{strict_sample_count}/{ANALOG_MIN_SAMPLES}个；"
                f"放宽至相似度{ANALOG_FALLBACK_MIN_SIMILARITY:.0f}分后仍只有"
                f"{sample_count}/{ANALOG_MIN_SAMPLES}个"
                + (f"；最高相似度{best_similarity:.3f}分。" if best_similarity is not None else "；没有可比较候选。")
            )
            horizon_results.append(
                {
                    **config,
                    "available": False,
                    "sample_count": sample_count,
                    "strict_sample_count": strict_sample_count,
                    "adaptive_sample_count": sample_count,
                    "selection_mode": "样本不足",
                    "selection_threshold": selection_threshold,
                    "positive_ratio": None,
                    "median_return": None,
                    "q10_return": None,
                    "q25_return": None,
                    "q75_return": None,
                    "median_worst_loss": None,
                    "average_similarity": float(np.mean(similarities_used)) if similarities_used else None,
                    "confidence_score": 0,
                    "direction": "样本不足，不形成预测",
                    "reason": reason,
                    "outcomes": [],
                    "paths": [],
                }
            )

    backtest = _walk_forward_analog_backtest(features, close, ANALOG_FEATURE_WEIGHTS, horizon=20)
    available_horizons = [item for item in horizon_results if item["available"]]
    if available_horizons:
        if backtest.get("available") and backtest.get("cases", 0) >= 10:
            accuracy = float(backtest["direction_accuracy"])
            backtest_factor = float(np.clip((accuracy - 0.35) / 0.30, 0.25, 1.0))
            multiplier = 0.45 + 0.55 * backtest_factor
            for item in available_horizons:
                adjusted = float(item["confidence_score"]) * multiplier
                if accuracy + 0.03 < float(backtest.get("momentum_accuracy", accuracy)):
                    adjusted -= 5
                item["confidence_score"] = int(np.clip(round(adjusted), 0, 95))
        else:
            for item in available_horizons:
                item["confidence_score"] = min(int(item["confidence_score"]), 65)
        confidence_score = float(np.mean([item["confidence_score"] for item in available_horizons]))
        confidence_score = int(np.clip(round(confidence_score), 0, 100))
        confidence_label = "较高" if confidence_score >= 75 else "中等" if confidence_score >= 55 else "较低"
    else:
        confidence_score = 0
        confidence_label = "样本不足"
    if not history_complete:
        notes.append("该标的上市不足五年或数据覆盖不完整，已下调相似周期可信度。")
    if not available_horizons:
        notes.append("各期限均未同时达到最低样本数与可信度要求；具体原因见逐期限表格，Agent不会强行形成方向预测。")

    best_similarity = float(similarities.max()) if not similarities.empty else None
    return {
        "available": bool(available_horizons),
        "source_label": source_label,
        "candidate_count": int(len(similarities)),
        "eligible_candidate_count": int(len(eligible_similarities)),
        "best_similarity": best_similarity,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "state": state,
        "current_features": current_display_features,
        "horizons": horizon_results,
        "matches": match_rows,
        "matches_selection_mode": global_match_mode,
        "backtest": backtest,
        "notes": notes,
    }


def score_stock_risk(metrics: dict[str, Any]) -> tuple[int, int, str, list[str]]:
    volatility = metrics["annual_volatility"]
    max_drawdown = abs(metrics["max_drawdown"])
    downside = metrics["downside_volatility"]
    beta = 1.0 if pd.isna(metrics["beta"]) else max(0.0, metrics["beta"])
    vol_part = np.clip((volatility - 0.12) / 0.58 * 35, 0, 35) if np.isfinite(volatility) else 20
    drawdown_part = np.clip((max_drawdown - 0.12) / 0.68 * 35, 0, 35)
    downside_part = np.clip((downside - 0.08) / 0.52 * 20, 0, 20) if np.isfinite(downside) else 10
    beta_part = np.clip((beta - 0.6) / 1.8 * 10, 0, 10)
    score = int(round(np.clip(vol_part + drawdown_part + downside_part + beta_part, 0, 100)))
    if score <= 18:
        level, level_number = "R2（中低风险）", 2
    elif score <= 38:
        level, level_number = "R3（中等风险）", 3
    elif score <= 63:
        level, level_number = "R4（中高风险）", 4
    else:
        level, level_number = "R5（高风险）", 5
    reasons = [f"近一年年化波动率约{volatility:.3%}" if np.isfinite(volatility) else "波动率数据不足", f"近五年或上市以来最大回撤约{metrics['max_drawdown']:.3%}"]
    if beta > 1.2:
        reasons.append("相对市场的Beta较高")
    if metrics["abnormal_days"]:
        reasons.append("历史数据中存在极端单日变动，需核对复权或公司事件")
    return score, level_number, level, reasons


def score_investor(profile: dict[str, Any]) -> tuple[int, int, str, str, list[str]]:
    source_points = {"闲置自有资金": 15, "未来有明确用途的资金": 8, "应急资金": 0, "借款／融资资金": 0}[profile["fund_source"]]
    reserve_points = {"不足3个月": 0, "3—6个月": 8, "6个月以上": 15}[profile["emergency_reserve"]]
    need_points = {"1周内": 0, "1个月内": 3, "3个月内": 7, "1年内": 12, "3年内": 15, "没有明确时间": 15}[profile["earliest_need"]]
    loss_points = {"立即全部卖出": 1, "大部分减仓": 4, "先复核原因再决定": 8, "继续按原计划持有": 11, "在条件允许时分批增加": 13}[profile["loss_response"]]
    max_loss_points = {"不超过5%": 0, "5%—10%": 4, "10%—20%": 8, "20%—30%": 12, "超过30%": 15}[profile.get("max_loss", "10%—20%")]
    goal_points = {"保值为主": 0, "股息／稳健收益": 3, "长期增值": 6, "波段操作": 8, "短线交易": 10}[profile["goal"]]
    stability_points = {"不稳定": 2, "较稳定": 7, "稳定": 10}[profile["income_stability"]]
    knowledge_points = {"没有经验": 0, "不足1年": 2, "1—3年": 4, "3年以上": 5}[profile["experience"]]
    score = int(np.clip(source_points + reserve_points + need_points + loss_points + max_loss_points + goal_points + stability_points + knowledge_points, 0, 100))
    if score <= 25:
        level_number, level = 1, "C1（保守型）"
    elif score <= 45:
        level_number, level = 2, "C2（谨慎型）"
    elif score <= 65:
        level_number, level = 3, "C3（平衡型）"
    elif score <= 89:
        level_number, level = 4, "C4（积极型）"
    else:
        level_number, level = 5, "C5（激进型）"

    if profile["experience"] in {"没有经验", "不足1年"}:
        style = "基础型"
    elif profile["goal"] == "短线交易" or profile["trade_frequency"] == "几乎每天":
        style = "活跃交易型"
    elif profile["goal"] in {"股息／稳健收益", "长期增值"}:
        style = "长期配置型"
    else:
        style = "进阶研究型"

    flags: list[str] = []
    if profile["fund_source"] in {"应急资金", "借款／融资资金"}:
        flags.append("资金来源不适合承担单只股票波动")
    if profile["emergency_reserve"] == "不足3个月":
        flags.append("应急储备不足3个月")
    if profile.get("existing_concentration") in {"30%—50%", "超过50%"}:
        flags.append("现有投资的单只股票集中度较高")
    if profile.get("leverage") == "是":
        flags.append("计划使用杠杆会放大损失")
    return score, level_number, level, style, flags


def _return_over(close: pd.Series, days: int) -> float:
    if len(close) <= days:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


def _analog_adjustment_value(
    evidence: dict[str, Any],
    days: int,
    maximum_adjustment: float,
) -> float:
    positive_ratio = float(evidence["positive_ratio"])
    median_return = float(evidence["median_return"])
    expected_scale = 0.015 * np.sqrt(max(float(days), 5.0) / 5.0)
    raw_adjustment = (positive_ratio - 0.50) * 20
    raw_adjustment += float(np.clip(median_return / expected_scale, -1, 1) * 4)
    q10_return = safe_float(evidence.get("q10_return"))
    if q10_return is not None and q10_return < -0.15:
        raw_adjustment -= 1.5
    confidence_weight = 0.45 + 0.55 * float(evidence["confidence_score"]) / 100
    return float(
        np.clip(
            raw_adjustment * confidence_weight,
            -maximum_adjustment,
            maximum_adjustment,
        )
    )


def _timing_rank_ic(signal: pd.Series, outcome: pd.Series) -> float | None:
    pair = pd.concat(
        [signal.rename("signal"), outcome.rename("outcome")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 8 or pair["signal"].nunique() < 2 or pair["outcome"].nunique() < 2:
        return None
    value = pair["signal"].rank(method="average").corr(
        pair["outcome"].rank(method="average")
    )
    return float(value) if value is not None and np.isfinite(value) else None


def _technical_timing_frame(
    close: pd.Series,
    volume: pd.Series,
    benchmark_close: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build the V6.6 directional factors at every visible historical date.

    Trend is one combined block so price/MA and MA/MA cannot be counted twice.
    Momentum and relative strength are scaled by the stock's recent volatility,
    which makes a five-percent move comparable across low- and high-volatility
    securities.  Volume is retained as context but deliberately scores zero:
    higher volume alone does not say whether price should rise or fall.  The
    only added directional rule is a 20-day mean-reversion adjustment already
    present in the second candidate version.  It was retained only for the
    20-day horizon where the pre-2023 development sample improved; it is not
    applied to 5, 60, 120 or 250 days.
    """

    days = int(config["days"])
    fast = int(config["fast"])
    slow = int(config["slow"])
    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(slow, min_periods=slow).mean()
    valid_ma = fast_ma.notna() & slow_ma.notna()
    price_above_fast = close >= fast_ma
    fast_above_slow = fast_ma >= slow_ma
    trend_points = pd.Series(0.0, index=close.index)
    trend_points.loc[valid_ma & price_above_fast & fast_above_slow] = 10.0
    trend_points.loc[valid_ma & ~price_above_fast & ~fast_above_slow] = -10.0
    trend_points.loc[~valid_ma] = np.nan

    stock_return = close.pct_change(days, fill_method=None)
    daily_volatility = close.pct_change(fill_method=None).rolling(
        60,
        min_periods=40,
    ).std()
    expected_move = (daily_volatility * np.sqrt(days)).replace(0, np.nan)
    momentum_signal = (stock_return / (2.0 * expected_move)).clip(-1, 1)
    momentum_points = momentum_signal * 8.0

    benchmark_available = isinstance(benchmark_close, pd.Series) and not benchmark_close.empty
    if benchmark_available:
        benchmark_aligned = benchmark_close.reindex(close.index).ffill()
        benchmark_return = benchmark_aligned.pct_change(days, fill_method=None)
        excess_return = stock_return - benchmark_return
        relative_signal = (excess_return / (2.0 * expected_move)).clip(-1, 1)
        relative_points = relative_signal * 7.0
        future_benchmark_return = benchmark_aligned.shift(-days) / benchmark_aligned - 1
    else:
        benchmark_return = pd.Series(np.nan, index=close.index)
        excess_return = pd.Series(np.nan, index=close.index)
        relative_signal = pd.Series(np.nan, index=close.index)
        relative_points = pd.Series(0.0, index=close.index)
        future_benchmark_return = pd.Series(np.nan, index=close.index)

    volume_ratio = volume.rolling(20, min_periods=20).mean() / volume.rolling(
        60,
        min_periods=60,
    ).mean()
    volume_context = np.sign(stock_return) * np.log(volume_ratio.clip(lower=0.20, upper=5.0))

    ret10 = close.pct_change(10, fill_method=None)
    ret10_mean = ret10.rolling(250, min_periods=120).mean()
    ret10_std = ret10.rolling(250, min_periods=120).std(ddof=1)
    ret10_zscore = (ret10 - ret10_mean) / ret10_std.replace(0, np.nan)
    short_reversal_adjustment = pd.Series(0.0, index=close.index)
    if days == 20:
        short_reversal_adjustment = np.where(
            ret10_zscore >= 1.0,
            -np.clip((ret10_zscore - 0.8) * 3.0, 0.0, 5.0),
            np.where(
                ret10_zscore <= -1.0,
                np.clip((0.8 - ret10_zscore) * 1.8, 0.0, 3.0),
                0.0,
            ),
        )
        short_reversal_adjustment = pd.Series(short_reversal_adjustment, index=close.index).fillna(0.0)

    high_252 = close.rolling(252, min_periods=252).max()
    position_52_week = close / high_252.replace(0, np.nan) - 1.0
    avg10 = volume.rolling(10, min_periods=10).mean()
    avg30 = volume.rolling(30, min_periods=30).mean()
    short_volume_ratio = avg10 / avg30.replace(0, np.nan)
    price_volume_confirmation = pd.Series("中性", index=close.index, dtype="object")
    price_volume_confirmation.loc[(ret10 > 0.02) & (short_volume_ratio > 1.15)] = "放量上涨"
    price_volume_confirmation.loc[(ret10 > 0.02) & (short_volume_ratio < 0.95)] = "缩量上涨"
    price_volume_confirmation.loc[(ret10 < -0.02) & (short_volume_ratio > 1.15)] = "放量下跌"
    annual_volatility_20 = close.pct_change(fill_method=None).rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    volatility_percentile = annual_volatility_20.rolling(250, min_periods=120).rank(pct=True)

    raw_score = 50.0 + trend_points + momentum_points + relative_points + short_reversal_adjustment
    future_return = close.shift(-days) / close - 1
    future_target = future_return - future_benchmark_return if benchmark_available else future_return

    return pd.DataFrame(
        {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "price_above_fast": price_above_fast,
            "fast_above_slow": fast_above_slow,
            "stock_return": stock_return,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "volume_ratio": volume_ratio,
            "volume_context": volume_context,
            "trend_points": trend_points,
            "momentum_points": momentum_points,
            "relative_points": relative_points,
            "short_reversal_adjustment": short_reversal_adjustment,
            "ret10_zscore": ret10_zscore,
            "position_52_week": position_52_week,
            "short_volume_ratio": short_volume_ratio,
            "price_volume_confirmation": price_volume_confirmation,
            "volatility_percentile": volatility_percentile,
            "raw_score": raw_score,
            "future_return": future_return,
            "future_target": future_target,
        },
        index=close.index,
    )


def _validate_timing_signal(frame: pd.DataFrame, days: int) -> dict[str, Any]:
    """Validate the fixed technical score using outcomes already known by T.

    This is a gate, not an optimiser: it may keep, halve or suppress a signal,
    but it never searches for a more flattering weight or reverses a failed
    factor after seeing the answer.
    """

    usable = frame[["raw_score", "future_target"]].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    stride = max(5, int(round(days / 4)))
    usable = usable.iloc[::stride].copy()
    samples = int(len(usable))
    minimum_samples = 40 if days <= 20 else 30 if days <= 60 else 24 if days <= 120 else 16
    current_score = float(frame["raw_score"].dropna().iloc[-1]) if frame["raw_score"].notna().any() else 50.0
    current_direction = int(np.sign(current_score - 50.0))
    signal = usable["raw_score"] - 50.0
    outcome = usable["future_target"]
    ic = _timing_rank_ic(signal, outcome)
    midpoint = samples // 2
    first_ic = _timing_rank_ic(signal.iloc[:midpoint], outcome.iloc[:midpoint]) if midpoint >= 8 else None
    second_ic = _timing_rank_ic(signal.iloc[midpoint:], outcome.iloc[midpoint:]) if samples - midpoint >= 8 else None

    active = usable.loc[signal.abs() >= 4.0].copy()
    hit_rate = None
    if len(active) >= 12:
        active_signal = active["raw_score"] - 50.0
        hit_rate = float(((active_signal > 0) == (active["future_target"] > 0)).mean())

    spread = None
    if samples >= 16 and signal.nunique() >= 5:
        lower = signal.quantile(0.25)
        upper = signal.quantile(0.75)
        low_outcomes = outcome[signal <= lower]
        high_outcomes = outcome[signal >= upper]
        if not low_outcomes.empty and not high_outcomes.empty:
            spread = float(high_outcomes.median() - low_outcomes.median())

    local_band_count = 0
    local_direction_hit_rate = None
    local_signed_median = None
    if samples >= 12 and current_direction != 0:
        local_band_count = min(samples, max(12, int(np.ceil(samples * 0.25))))
        nearest = (usable["raw_score"] - current_score).abs().nsmallest(local_band_count).index
        local_outcomes = usable.loc[nearest, "future_target"]
        local_direction_hit_rate = float(
            ((local_outcomes > 0) if current_direction > 0 else (local_outcomes < 0)).mean()
        )
        local_signed_median = float(local_outcomes.median() * current_direction)

    local_supports = bool(
        local_direction_hit_rate is not None
        and local_direction_hit_rate >= 0.52
        and local_signed_median is not None
        and local_signed_median > 0
    )
    local_strongly_opposes = bool(
        local_direction_hit_rate is not None
        and local_direction_hit_rate < 0.45
        and local_signed_median is not None
        and local_signed_median < 0
    )

    global_passed = bool(
        samples >= minimum_samples
        and ic is not None
        and ic >= 0.03
        and first_ic is not None
        and first_ic > 0
        and second_ic is not None
        and second_ic > 0
        and spread is not None
        and spread > 0
        and hit_rate is not None
        and hit_rate >= 0.52
    )
    global_limited = bool(
        not global_passed
        and samples >= minimum_samples
        and ic is not None
        and ic > 0
        and second_ic is not None
        and second_ic >= 0
        and spread is not None
        and spread > 0
        and hit_rate is not None
        and hit_rate >= 0.50
    )
    passed = bool(global_passed and local_supports)
    limited = bool(
        not passed
        and (global_passed or global_limited)
        and not local_strongly_opposes
    )

    if passed:
        status = "通过"
        reliability_multiplier = 1.0
    elif limited:
        status = "有限通过"
        reliability_multiplier = 0.5
    else:
        status = "未通过"
        reliability_multiplier = 0.0

    sample_points = min(samples / max(minimum_samples, 1), 1.0) * 25
    ic_points = float(np.clip((ic or 0.0) / 0.10, 0, 1) * 25)
    stability_points = 20 if first_ic is not None and second_ic is not None and first_ic > 0 and second_ic > 0 else 8 if second_ic is not None and second_ic >= 0 else 0
    spread_points = 15 if spread is not None and spread > 0 else 0
    hit_points = 15 if hit_rate is not None and hit_rate >= 0.55 else 8 if hit_rate is not None and hit_rate >= 0.52 else 3 if hit_rate is not None and hit_rate >= 0.50 else 0
    local_points = 10 if local_supports else 4 if not local_strongly_opposes else 0
    confidence = int(round(np.clip(sample_points + ic_points + stability_points + spread_points + hit_points + local_points, 0, 90)))
    if status == "未通过":
        confidence = min(confidence, 39)
    elif status == "有限通过":
        confidence = min(max(confidence, 45), 59)
    else:
        confidence = max(confidence, 60)

    if samples < minimum_samples:
        reason = f"只有{samples}个历史验证时点，低于最低要求{minimum_samples}个。"
    elif status == "通过":
        reason = "整体历史验证及当前相近分数区间均支持该方向。"
    elif status == "有限通过":
        reason = "整体历史方向初步为正，且当前相近分数区间未明显反对；技术分只保留一半影响。"
    else:
        reason = "整体历史验证或当前相近分数区间至少一项未达到要求。"

    return {
        "status": status,
        "confidence_score": confidence,
        "reliability_multiplier": reliability_multiplier,
        "samples": samples,
        "minimum_samples": minimum_samples,
        "stride": stride,
        "rank_ic": ic,
        "first_half_ic": first_ic,
        "second_half_ic": second_ic,
        "hit_rate": hit_rate,
        "high_low_spread": spread,
        "current_raw_score": current_score,
        "local_band_count": local_band_count,
        "local_direction_hit_rate": local_direction_hit_rate,
        "local_signed_median_return": local_signed_median,
        "local_strongly_opposes": local_strongly_opposes,
        "reason": reason,
    }


def score_horizons(
    metrics: dict[str, Any],
    fundamental: EvidenceSnapshot,
    macro: EvidenceSnapshot,
    analog_forecast: dict[str, Any] | None = None,
    market_analog_forecast: dict[str, Any] | None = None,
    asset_type: str | None = None,
) -> list[dict[str, Any]]:
    close: pd.Series = metrics["close"]
    volume: pd.Series = metrics["volume"]
    benchmark_close: pd.Series = metrics["benchmark_close"]
    results: list[dict[str, Any]] = []
    for config in HORIZONS:
        if config["intraday_required"]:
            results.append(
                {
                    **config,
                    "available": False,
                    "score": None,
                    "label": "需要分钟级／实时数据",
                    "analog_adjustment": 0.0,
                    "analog_evidence": None,
                    "analog_used": False,
                    "analog_source": "未使用",
                    "analog_status": "缺少分钟／实时数据，不做次日相似修正",
                    "signal_confidence": 0,
                    "direction_available": False,
                    "signal_validation": {
                        "status": "不可验证",
                        "reason": "缺少分钟级、实时盘口和交易成本数据。",
                    },
                    "reasons": ["当前免费接口只提供日线，不能可靠判断下一交易日涨跌。"],
                }
            )
            continue
        if len(close) < config["minimum_rows"]:
            results.append(
                {
                    **config,
                    "available": False,
                    "score": None,
                    "label": "历史数据不足",
                    "analog_adjustment": 0.0,
                    "analog_evidence": None,
                    "analog_used": False,
                    "analog_source": "未使用",
                    "analog_status": f"行情仅{len(close)}日，最低需要{config['minimum_rows']}日",
                    "signal_confidence": 0,
                    "direction_available": False,
                    "signal_validation": {
                        "status": "样本不足",
                        "reason": f"行情仅{len(close)}日，最低需要{config['minimum_rows']}日。",
                    },
                    "reasons": [f"需要至少{config['minimum_rows']}个交易日，当前只有{len(close)}个。"],
                }
            )
            continue

        factor_frame = _technical_timing_frame(close, volume, benchmark_close, config)
        current = factor_frame.iloc[-1]
        if not np.isfinite(float(current["raw_score"])):
            results.append(
                {
                    **config,
                    "available": False,
                    "score": None,
                    "label": "技术指标不足",
                    "analog_adjustment": 0.0,
                    "analog_evidence": None,
                    "analog_used": False,
                    "analog_source": "未使用",
                    "analog_status": "技术指标尚未形成完整窗口",
                    "signal_confidence": 0,
                    "direction_available": False,
                    "signal_validation": {
                        "status": "样本不足",
                        "reason": "当前均线或波动率窗口尚不完整。",
                    },
                    "reasons": ["当前均线或波动率窗口尚不完整。"],
                }
            )
            continue

        validation = _validate_timing_signal(factor_frame, int(config["days"]))
        certification = direction_certification(asset_type, int(config["days"]))
        validation["within_security_status"] = validation.get("status")
        validation["within_security_reason"] = validation.get("reason")
        if not certification["certified"]:
            if validation.get("status") == "通过":
                validation["confidence_score"] = min(int(validation.get("confidence_score") or 0), 69)
            elif validation.get("status") == "有限通过":
                validation["confidence_score"] = min(int(validation.get("confidence_score") or 0), 54)
            else:
                validation["confidence_score"] = min(int(validation.get("confidence_score") or 0), 39)
            validation["certification_note"] = (
                f"{certification['reason']} 当前状态分仍然展示；"
                "只有本股票自身历史验证通过时才能形成受限方向。"
            )
        validation["cross_security_certification"] = certification
        reliability_multiplier = float(validation["reliability_multiplier"])
        raw_technical_score = float(current["raw_score"])
        # The score below describes the observable state at this date. Historical
        # validation determines whether that state is actionable; it must not turn
        # every factor into zero merely because broader certification is unfinished.
        trend_points = float(current["trend_points"])
        momentum_points = float(current["momentum_points"])
        relative_points = float(current["relative_points"])
        short_reversal_points = float(current["short_reversal_adjustment"])
        score = 50.0 + trend_points + momentum_points + relative_points + short_reversal_points
        stock_return = safe_float(current["stock_return"])
        benchmark_return = safe_float(current["benchmark_return"])
        volume_ratio = safe_float(current["volume_ratio"])
        ret10_zscore = safe_float(current["ret10_zscore"])
        position_52_week = safe_float(current["position_52_week"])
        short_volume_ratio = safe_float(current["short_volume_ratio"])
        price_volume_confirmation = str(current["price_volume_confirmation"])
        volatility_percentile = safe_float(current["volatility_percentile"])
        reasons: list[str] = []

        raw_trend_points = float(current["trend_points"])
        if raw_trend_points > 0:
            reasons.append(
                f"现价位于{config['fast']}日均线上方，且快线位于{config['slow']}日均线上方"
            )
        elif raw_trend_points < 0:
            reasons.append(
                f"现价位于{config['fast']}日均线下方，且快线位于{config['slow']}日均线下方"
            )
        else:
            reasons.append("价格与快慢均线信号不一致，趋势暂未确认")

        if stock_return is not None:
            reasons.append(f"近{config['days']}个交易日收益{stock_return:.3%}")
        if stock_return is not None and benchmark_return is not None:
            excess = stock_return - benchmark_return
            reasons.append("同期跑赢市场基准" if excess > 0 else "同期弱于市场基准")
        if volume_ratio is not None and volume_ratio >= 1.20:
            direction_text = "上涨" if (stock_return or 0.0) > 0 else "下跌" if (stock_return or 0.0) < 0 else "横盘"
            reasons.append(f"近期成交量放大且价格{direction_text}；成交量仅作确认，不再单独加分")
        elif volume_ratio is not None and volume_ratio <= 0.75:
            reasons.append("近期成交量明显收缩；仅作流动性提示，不再单独扣分")
        if float(current["short_reversal_adjustment"]) < 0:
            reasons.append(
                f"近10日涨幅处于自身历史极端区间"
                f"{f'（z={ret10_zscore:.2f}）' if ret10_zscore is not None else ''}，"
                "20日均值回归修正下调当前分数"
            )
        elif float(current["short_reversal_adjustment"]) > 0:
            reasons.append(
                f"近10日涨幅处于自身历史偏低区间"
                f"{f'（z={ret10_zscore:.2f}）' if ret10_zscore is not None else ''}，"
                "20日均值回归修正小幅上调当前分数"
            )
        if position_52_week is not None:
            reasons.append(f"现价距近52周高点{position_52_week:.3%}；该项仅展示，不重复计入趋势分")
        if price_volume_confirmation != "中性":
            reasons.append(f"近期量价状态为“{price_volume_confirmation}”；仅作背景确认")
        if volatility_percentile is not None and volatility_percentile >= 0.85:
            reasons.append("当前20日波动率处于自身近一年高位；用于风险提示，不改写方向")

        macro_weight = 0.08 if config["days"] <= 20 else 0.10
        macro_points = 0.0
        if macro.score is not None:
            macro_points = float(np.clip((macro.score - 50) * macro_weight, -2.0, 2.0))
            score += macro_points
        fundamental_weight = (
            0.0
            if config["days"] <= 20
            else 0.06
            if config["days"] <= 60
            else 0.10
            if config["days"] <= 120
            else 0.12
        )
        fundamental_points = 0.0
        if fundamental.score is not None:
            fundamental_points = float(
                np.clip((fundamental.score - 50) * fundamental_weight, -4.0, 4.0)
            )
            score += fundamental_points
            if fundamental_weight > 0:
                reasons.append(
                    "基本面评分提供有限支持"
                    if fundamental.score >= 55
                    else "基本面评分未形成明显支持"
                )

        analog_adjustment = 0.0
        stock_analog_result = next(
            (
                item
                for item in (analog_forecast or {}).get("horizons", [])
                if int(item.get("days", -1)) == int(config["days"])
            ),
            None,
        )
        market_analog_result = next(
            (
                item
                for item in (market_analog_forecast or {}).get("horizons", [])
                if int(item.get("days", -1)) == int(config["days"])
            ),
            None,
        )
        stock_analog_evidence = stock_analog_result if stock_analog_result and stock_analog_result.get("available") else None
        market_analog_evidence = market_analog_result if market_analog_result and market_analog_result.get("available") else None
        analog_evidence = stock_analog_evidence or market_analog_evidence
        analog_used = False
        analog_source = "仅展示"
        if analog_evidence:
            sample_count = int(analog_evidence.get("sample_count", 0))
            confidence_value = int(analog_evidence.get("confidence_score", 0))
            analog_status = (
                f"V6.6仅展示、不计分（{sample_count}个样本，可信度{confidence_value}/100）"
            )
        elif stock_analog_result:
            analog_status = f"仅展示、不计分：{stock_analog_result.get('reason') or '同股有效样本不足'}"
        elif int(config["days"]) >= 250:
            analog_status = "近五年窗口不足以可靠检验250日后续表现；不计分"
        else:
            analog_status = "当前期限没有可用的相似周期样本；不计分"

        signal_confidence = int(validation["confidence_score"])
        # "有限通过" remains visible as a research score, but it is not
        # strong enough to create a buy/sell direction or position budget.
        direction_available = bool(
            validation["status"] == "通过"
            and signal_confidence >= 60
        )
        score_int = int(round(np.clip(score, 0, 100)))
        if score_int >= 65:
            label = "当前状态偏强"
        elif score_int >= 55:
            label = "当前状态中性偏强"
        elif score_int >= 45:
            label = "当前状态中性"
        elif score_int >= 35:
            label = "当前状态偏弱"
        else:
            label = "当前状态明显偏弱"
        reasons.append(
            f"本期限历史验证：{validation['status']}，可信度{signal_confidence}/100；"
            f"{validation['reason']}"
        )
        if validation.get("certification_note"):
            reasons.append(str(validation["certification_note"]))
        stress_returns = close.pct_change(config["days"]).dropna()
        stress_loss = abs(float(stress_returns.quantile(0.05))) if not stress_returns.empty else np.nan
        historical_worst = float(stress_returns.min()) if not stress_returns.empty else np.nan
        results.append(
            {
                **config,
                "available": True,
                "score": score_int,
                "label": label,
                "stock_return": stock_return,
                "benchmark_return": benchmark_return,
                "stress_loss": stress_loss,
                "historical_worst": historical_worst,
                "raw_technical_score": raw_technical_score,
                "technical_reliability_multiplier": reliability_multiplier,
                "score_type": "固定权重当前状态观察分",
                "signal_confidence": signal_confidence,
                "direction_available": direction_available,
                "signal_validation": validation,
                "cross_security_certification": certification,
                "factor_contributions": {
                    "trend": trend_points,
                    "momentum": momentum_points,
                    "relative_strength": relative_points,
                    "short_reversal": short_reversal_points,
                    "volume": 0.0,
                    "macro": macro_points,
                    "fundamental": fundamental_points,
                    "historical_analog": 0.0,
                },
                "context_factors": {
                    "ret10_zscore": ret10_zscore,
                    "position_52_week": position_52_week,
                    "short_volume_ratio": short_volume_ratio,
                    "price_volume_confirmation": price_volume_confirmation,
                    "volatility_percentile": volatility_percentile,
                },
                "analog_adjustment": analog_adjustment,
                "analog_evidence": analog_evidence,
                "stock_analog_result": stock_analog_result,
                "market_analog_result": market_analog_result,
                "analog_used": analog_used,
                "analog_source": analog_source,
                "analog_status": analog_status,
                "reasons": reasons,
            }
        )
    return results


def _maximum_horizon_rank(earliest_need: str) -> int:
    return {"1周内": 0, "1个月内": 1, "3个月内": 2, "1年内": 3, "3年内": 4, "没有明确时间": 4}[earliest_need]


def choose_horizon(horizon_scores: list[dict[str, Any]], profile: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    available = [item for item in horizon_scores if item["available"]]
    if not available:
        return None, ["所有可用持有周期的数据都不足。"]
    max_rank = _maximum_horizon_rank(profile["earliest_need"])
    ranked_names = ["2—5个交易日", "2—4周", "1—3个月", "3—12个月", "1—3年"]
    allowed_names = set(ranked_names[: max_rank + 1])
    candidates = [item for item in available if item["name"] in allowed_names]
    notes: list[str] = []
    if not candidates:
        candidates = available[:1]
        notes.append("资金使用时间与可用行情周期存在冲突，只能提供最短可分析周期。")
    priors = GOAL_PRIORS[profile["goal"]]
    for item in candidates:
        operational = 0
        if item["name"] in {"2—5个交易日", "2—4周"}:
            if profile["monitor_time"] == "不足15分钟":
                operational -= 9
            if profile.get("stop_loss") in {"没有明确规则", "有规则但经常改变"}:
                operational -= 7
            if profile["experience"] in {"没有经验", "不足1年"}:
                operational -= 6
        signal_confidence = int(item.get("signal_confidence") or 0)
        confidence_adjustment = float(np.clip((signal_confidence - 50) * 0.12, -6, 6))
        if item.get("direction_available"):
            confidence_adjustment += 4
        # V6.4 chose the largest current score, which could cherry-pick the
        # most flattering horizon. V6.6 chooses the horizon from the user's
        # objective, liquidity and ability to execute, then evaluates the
        # direction inside that fixed horizon.
        item["selection_score"] = priors.get(item["name"], 0) + operational + confidence_adjustment
    selected = max(candidates, key=lambda item: item["selection_score"])
    if profile["goal"] == "短线交易" and selected["name"] != "2—5个交易日":
        notes.append("短线意向与看盘条件、纪律或市场信号不匹配，因此Agent没有选择最短周期。")
    if profile["earliest_need"] in {"1周内", "1个月内"}:
        notes.append("资金近期可能使用，任何股票持有周期都存在被迫在不利价格退出的风险。")
    if not selected.get("direction_available"):
        notes.append("该持有期由用户目标和资金期限确定，但历史验证未通过，因此不形成买入方向判断。")
    else:
        notes.append("持有期先按用户目标、资金期限和执行条件确定，没有从多个周期中挑选最高分。")
    return selected, notes


def calculate_data_confidence(
    metrics: dict[str, Any],
    fundamental: EvidenceSnapshot,
    macro: EvidenceSnapshot,
    asset_type: str,
) -> tuple[int, list[str]]:
    score = 0
    notes: list[str] = []
    rows = metrics["rows"]
    score += 40 if rows >= 750 else 34 if rows >= 500 else 26 if rows >= 250 else 16 if rows >= 120 else 8
    if metrics["latest_lag"] <= 5:
        score += 20
    elif metrics["latest_lag"] <= 12:
        score += 12
        notes.append("最新行情可能存在延迟。")
    else:
        score += 3
        notes.append("最新行情日期较旧。")
    score += 15 if np.isfinite(metrics["beta"]) else 6
    if asset_type == "场内基金":
        score += 12
        notes.append("基金未使用单一公司财务评分，需另行核对跟踪标的和费率。")
    elif fundamental.available:
        score += 18
    else:
        score += 5
        notes.append("财务或估值数据未取得，结论主要基于行情。")
    score += 7 if macro.available else 2
    if metrics["abnormal_days"]:
        score -= min(10, metrics["abnormal_days"] * 2)
        notes.append("检测到极端日收益，可能包含公司事件或复权差异。")
    if not metrics.get("history_complete", True):
        rows = int(metrics.get("rows", 0))
        cap = 30 if rows < 60 else 45 if rows < 250 else 65 if rows < 750 else 78
        score = min(score, cap)
        notes.append("该股票可得历史不足五年。数据不足，无法准确判断。")
    if not metrics.get("benchmark_available", True):
        score = min(score, 72)
        notes.append("市场基准暂不可用，Beta和相对强弱未参与判断。")
    return int(np.clip(score, 0, 100)), notes


def assess_suitability(
    profile: dict[str, Any],
    investor_level_number: int,
    stock_risk_number: int,
    selected_horizon: dict[str, Any] | None,
    data_confidence: int,
) -> dict[str, Any]:
    hard_reasons: list[str] = []
    if profile["fund_source"] == "借款／融资资金":
        hard_reasons.append("不使用借款或融资资金承担单只股票风险")
    if profile["fund_source"] == "应急资金":
        hard_reasons.append("应急资金应优先保持安全性和流动性")
    if profile["emergency_reserve"] == "不足3个月" and profile["planned_amount"] / profile["investable_assets"] > 0.10:
        hard_reasons.append("应急储备不足且计划投入比例较高")
    if data_confidence < 35 or selected_horizon is None:
        return {"fit": "证据不足", "fit_reason": "公开数据不足以形成可靠判断", "hard_reasons": hard_reasons}
    if hard_reasons:
        return {"fit": "不适配", "fit_reason": hard_reasons[0], "hard_reasons": hard_reasons}
    gap = investor_level_number - stock_risk_number
    if gap <= -2:
        fit, reason = "不适配", "用户风险承受能力与该股票风险等级差距较大"
    elif gap == -1:
        fit, reason = "有限适配", "股票风险比用户风险等级高一级，只适合严格限制风险预算"
    else:
        fit, reason = "适配", "用户风险承受能力覆盖该股票的模型风险等级"
    return {"fit": fit, "fit_reason": reason, "hard_reasons": hard_reasons}


def position_budget(
    profile: dict[str, Any],
    investor_level_number: int,
    stock_risk_number: int,
    suitability_result: dict[str, Any],
    selected_horizon: dict[str, Any] | None,
) -> dict[str, Any]:
    if selected_horizon is None or suitability_result["fit"] in {"不适配", "证据不足"}:
        return {"lower_pct": 0.0, "upper_pct": 0.0, "lower_amount": 0.0, "upper_amount": 0.0, "stress_loss": None}
    if not selected_horizon.get("direction_available"):
        return {
            "lower_pct": 0.0,
            "upper_pct": 0.0,
            "lower_amount": 0.0,
            "upper_amount": 0.0,
            "stress_loss": selected_horizon.get("stress_loss"),
        }
    risk_budget = {1: 0.0025, 2: 0.005, 3: 0.010, 4: 0.020, 5: 0.030}[investor_level_number]
    cap = {1: 0.03, 2: 0.05, 3: 0.10, 4: 0.15, 5: 0.20}[investor_level_number]
    stress_loss = max(float(selected_horizon.get("stress_loss") or 0.0), 0.08)
    upper = min(risk_budget / stress_loss, cap)
    if stock_risk_number == 5:
        upper *= 0.75
    if suitability_result["fit"] == "有限适配":
        upper *= 0.50
    concentration = profile["existing_concentration"]
    if concentration in {"30%—50%", "超过50%"}:
        upper *= 0.60
    timing_score = selected_horizon["score"]
    if timing_score < 45:
        upper = 0.0
    elif timing_score < 60:
        upper *= 0.50
    certification = selected_horizon.get("cross_security_certification") or {}
    certification_cap_applied = not bool(certification.get("certified"))
    if certification_cap_applied:
        upper = min(upper, 0.03)
    lower = upper * 0.40 if upper > 0 else 0.0
    assets = float(profile["investable_assets"])
    return {
        "lower_pct": lower,
        "upper_pct": upper,
        "lower_amount": lower * assets,
        "upper_amount": upper * assets,
        "stress_loss": stress_loss,
        "certification_cap_applied": certification_cap_applied,
    }


def build_final_conclusion(
    suitability_result: dict[str, Any],
    selected_horizon: dict[str, Any] | None,
    position: dict[str, Any],
) -> tuple[str, str]:
    fit = suitability_result["fit"]
    if fit == "证据不足" or selected_horizon is None:
        return "证据不足，暂不形成判断", "公开数据完整度不足，需要更可靠的数据源。"
    if fit == "不适配":
        return "不适合该用户", suitability_result["fit_reason"]
    if not selected_horizon.get("direction_available"):
        validation = selected_horizon.get("signal_validation") or {}
        score = int(selected_horizon.get("score") or 50)
        if score < 45:
            conclusion = "当前时点偏弱，暂缓"
        elif score >= 60:
            conclusion = "当前状态偏强，但仅建议观察"
        else:
            conclusion = "当前信号中性，继续观察"
        return conclusion, (
            f"当前状态观察分为{score}/100；"
            f"所选持有期的单股历史验证为{validation.get('status', '未通过')}，"
            "因此可以说明当前强弱，但不把它包装成已验证的买入信号，仓位上限为0。"
        )
    if selected_horizon["score"] < 45:
        return "个人条件可讨论，但当前时点暂缓", "股票与用户并非绝对不匹配，但当前多周期信号偏弱。"
    if selected_horizon["score"] < 60 or position["upper_pct"] <= 0.02:
        return "可以小仓观察", "当前证据尚不支持较高风险预算，重点观察后续信号。"
    if fit == "有限适配":
        return "条件适配，可进一步研究", "风险等级略高于用户等级，需要严格限制仓位并按期复核。"
    return "在风险预算内相对适配", "个人条件与股票风险基本匹配，但仍需满足仓位和复核条件。"


def _personal_loss_cap(max_loss: str | None) -> float:
    """Translate the questionnaire band into a conservative position-level ceiling."""
    return {
        "不超过5%": 0.05,
        "5%—10%": 0.10,
        "10%—20%": 0.20,
        "20%—30%": 0.30,
        "超过30%": 0.30,
    }.get(str(max_loss or ""), 0.15)


def analyze_sell_signals(
    bundle: PriceBundle,
    analysis: dict[str, Any],
    profile: dict[str, Any],
    holding_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate transparent end-of-day exit signals for an existing position.

    This is deliberately separate from ``analyze_all`` so adding position
    management cannot change the existing buy-side score, suitability,
    horizon or position-budget calculations.
    """
    if not holding_snapshot:
        return {
            "available": False,
            "status": "未持有",
            "summary": "只有选择“已经持有”并填写持仓后，才会计算卖出信号。",
            "signals": [],
            "limitations": [],
        }

    prices = bundle.stock.copy()
    prices["日期"] = pd.to_datetime(prices["日期"], errors="coerce")
    prices["收盘"] = pd.to_numeric(prices["收盘"], errors="coerce")
    prices["成交量"] = pd.to_numeric(prices.get("成交量"), errors="coerce")
    prices = prices.dropna(subset=["日期", "收盘"]).drop_duplicates("日期").sort_values("日期")
    close = prices.set_index("日期")["收盘"]
    volume = prices.set_index("日期")["成交量"]
    if len(close) < 20:
        return {
            "available": False,
            "status": "证据不足",
            "summary": "可得日线不足20个交易日，无法形成可复核的卖出信号。",
            "signals": [],
            "limitations": ["股票历史数据过短。"],
        }

    selected = analysis.get("selected_horizon") or {}
    selected_days = int(selected.get("days") or 20)
    fast_window = int(selected.get("fast") or 20)
    slow_window = int(selected.get("slow") or 60)
    fast_window = max(5, min(fast_window, len(close)))
    slow_window = max(fast_window, min(slow_window, len(close)))
    latest_price = float(close.iloc[-1])
    latest_date = pd.Timestamp(close.index[-1])

    fast_series = close.rolling(fast_window, min_periods=fast_window).mean()
    slow_series = close.rolling(slow_window, min_periods=slow_window).mean()
    fast_ma = float(fast_series.iloc[-1])
    slow_ma = float(slow_series.iloc[-1])
    comparable = pd.concat(
        [close.rename("close"), fast_series.rename("fast")], axis=1
    ).dropna()
    consecutive_below_fast = bool(
        len(comparable) >= 2
        and (comparable["close"].tail(2) < comparable["fast"].tail(2)).all()
    )
    fast_below_slow = bool(np.isfinite(slow_ma) and fast_ma < slow_ma)
    trend_triggered = consecutive_below_fast and fast_below_slow
    trend_warning = bool(latest_price < fast_ma or fast_below_slow)

    annual_volatility = safe_float(analysis.get("metrics", {}).get("annual_volatility"))
    if annual_volatility is None or annual_volatility <= 0:
        annual_volatility = 0.25
    risk_span_days = max(5, min(selected_days, 120))
    normal_horizon_move = annual_volatility * float(np.sqrt(risk_span_days / 252))
    personal_cap = _personal_loss_cap(profile.get("max_loss"))
    dynamic_loss_limit = float(
        min(personal_cap, np.clip(normal_horizon_move * 0.85, 0.04, 0.20))
    )

    current_return = safe_float(holding_snapshot.get("return_rate"))
    cost_price = safe_float(holding_snapshot.get("cost_price"))
    loss_triggered = bool(
        current_return is not None and current_return <= -dynamic_loss_limit
    )
    cost_protection_price = (
        float(cost_price * (1 - dynamic_loss_limit)) if cost_price is not None and cost_price > 0 else None
    )

    peak_window = min(max(slow_window, 60), 252, len(close))
    recent_peak = float(close.tail(peak_window).max())
    peak_drawdown = float(latest_price / recent_peak - 1) if recent_peak > 0 else None
    trailing_limit = float(
        np.clip(annual_volatility * np.sqrt(20 / 252) * 1.25, 0.06, 0.18)
    )
    trailing_protection_price = float(recent_peak * (1 - trailing_limit))
    trailing_activated = bool(
        current_return is not None and current_return >= max(0.08, trailing_limit)
    )
    trailing_triggered = bool(
        trailing_activated
        and peak_drawdown is not None
        and peak_drawdown <= -trailing_limit
    )

    relative_window = 20 if selected_days <= 20 else 60
    benchmark_close = analysis.get("metrics", {}).get("benchmark_close")
    relative_excess = None
    relative_threshold = -float(
        np.clip(annual_volatility * np.sqrt(relative_window / 252) * 0.50, 0.03, 0.10)
    )
    if isinstance(benchmark_close, pd.Series) and not benchmark_close.empty:
        aligned = pd.concat(
            [close.rename("stock"), benchmark_close.rename("benchmark")], axis=1
        ).dropna()
        if len(aligned) > relative_window:
            stock_return = float(aligned["stock"].iloc[-1] / aligned["stock"].iloc[-1 - relative_window] - 1)
            benchmark_return = float(
                aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[-1 - relative_window] - 1
            )
            relative_excess = stock_return - benchmark_return
    relative_triggered = bool(
        relative_excess is not None and relative_excess <= relative_threshold
    )

    fundamental: EvidenceSnapshot = analysis.get("fundamental") or EvidenceSnapshot(False, "未取得")
    fundamental_score = safe_float(fundamental.score)
    deterioration_terms = (
        "净资产收益率为负",
        "净利率为负",
        "净利润同比下降",
        "营业收入同比下降",
        "经营现金流与净利润方向不一致",
    )
    deterioration_risks = [
        item for item in fundamental.risks if any(term in item for term in deterioration_terms)
    ]
    fundamental_triggered = bool(
        fundamental_score is not None
        and ((fundamental_score <= 35 and len(fundamental.risks) >= 1) or len(deterioration_risks) >= 2)
    )

    analog_target_days = 5 if selected_days <= 5 else 20 if selected_days <= 20 else 60 if selected_days <= 60 else 120
    analog_item = next(
        (
            item
            for item in (analysis.get("analog_forecast") or {}).get("horizons", [])
            if item.get("available") and int(item.get("days", -1)) == analog_target_days
        ),
        None,
    )
    analog_usable = bool(
        analog_item and int(analog_item.get("confidence_score", 0)) >= ANALOG_SCORE_MIN_CONFIDENCE
    )
    analog_triggered = bool(
        analog_usable
        and float(analog_item.get("positive_ratio")) <= 0.35
        and float(analog_item.get("median_return")) < 0
    )

    volume20 = float(volume.tail(20).mean()) if volume.tail(20).notna().any() else np.nan
    latest_volume = safe_float(volume.iloc[-1])
    down_day = bool(len(close) >= 2 and close.iloc[-1] < close.iloc[-2])
    volume_confirmation = bool(
        latest_volume is not None
        and np.isfinite(volume20)
        and volume20 > 0
        and latest_volume >= volume20 * 1.20
        and down_day
    )

    signals: list[dict[str, Any]] = [
        {
            "key": "personal_loss",
            "name": "个人亏损边界",
            "level": "核心",
            "state": "触发" if loss_triggered else "数据不足" if current_return is None else "未触发",
            "triggered": loss_triggered,
            "current": current_return,
            "threshold": -dynamic_loss_limit,
            "current_text": "成本未知" if current_return is None else f"当前持仓收益率 {current_return:.3%}",
            "threshold_text": f"亏损达到 {-dynamic_loss_limit:.3%}",
            "detail": "结合问卷最大可承受损失、所选周期和股票正常波动计算，不对所有股票使用同一个固定比例。",
        },
        {
            "key": "trend_break",
            "name": "趋势破位",
            "level": "核心",
            "state": "触发" if trend_triggered else "警戒" if trend_warning else "未触发",
            "triggered": trend_triggered,
            "current": latest_price,
            "threshold": fast_ma,
            "current_text": f"现价 {latest_price:.3f}；MA{fast_window} {fast_ma:.3f}；MA{slow_window} {slow_ma:.3f}",
            "threshold_text": f"连续2个交易日低于MA{fast_window}，且MA{fast_window}低于MA{slow_window}",
            "detail": "只有价格和均线结构同时确认才算核心触发，降低单日假跌破影响。"
            + (" 最新下跌日同时放量。" if volume_confirmation else ""),
        },
        {
            "key": "profit_trailing",
            "name": "盈利回撤保护",
            "level": "辅助",
            "state": "触发" if trailing_triggered else "尚未启用" if current_return is None or not trailing_activated else "未触发",
            "triggered": trailing_triggered,
            "current": peak_drawdown,
            "threshold": -trailing_limit,
            "current_text": f"近{peak_window}日高点回撤 {peak_drawdown:.3%}" if peak_drawdown is not None else "数据不足",
            "threshold_text": f"盈利达到启用条件后，高点回撤达到 {-trailing_limit:.3%}",
            "detail": "未填写首次买入日期，因此使用与所选周期匹配的近期高点，不冒充实际持有期最高点。",
        },
        {
            "key": "relative_weakness",
            "name": "相对市场弱势",
            "level": "辅助",
            "state": "触发" if relative_triggered else "数据不足" if relative_excess is None else "未触发",
            "triggered": relative_triggered,
            "current": relative_excess,
            "threshold": relative_threshold,
            "current_text": f"近{relative_window}日相对基准 {relative_excess:.3%}" if relative_excess is not None else "基准数据不足",
            "threshold_text": f"相对基准弱于 {relative_threshold:.3%}",
            "detail": f"市场基准为{bundle.benchmark_name}；阈值随股票波动率调整。",
        },
        {
            "key": "fundamental_risk",
            "name": "基本面恶化信号",
            "level": "辅助",
            "state": "触发" if fundamental_triggered else "数据不足" if fundamental_score is None else "未触发",
            "triggered": fundamental_triggered,
            "current": fundamental_score,
            "threshold": 35.0,
            "current_text": f"基本面评分 {fundamental_score:.3f}/100" if fundamental_score is not None else "财务数据不足",
            "threshold_text": "评分≤35且存在风险，或至少2项经营指标恶化",
            "detail": "；".join(deterioration_risks[:3]) if deterioration_risks else "未发现足够的经营恶化证据。",
        },
        {
            "key": "analog_negative",
            "name": "相似周期转弱",
            "level": "辅助",
            "state": "触发" if analog_triggered else "可信度不足" if not analog_usable else "未触发",
            "triggered": analog_triggered,
            "current": float(analog_item.get("median_return")) if analog_usable else None,
            "threshold": 0.0,
            "current_text": (
                f"上涨样本占比 {float(analog_item['positive_ratio']):.3%}；中位收益 {float(analog_item['median_return']):.3%}"
                if analog_usable
                else "没有达到最低可信度的相似样本"
            ),
            "threshold_text": "可信度达标、上涨样本占比≤35%，且收益中位数为负",
            "detail": "只作为辅助信号，绝不单独决定卖出。",
        },
    ]

    hard_count = sum(1 for item in signals if item["level"] == "核心" and item["triggered"])
    auxiliary_count = sum(1 for item in signals if item["level"] == "辅助" and item["triggered"])
    if hard_count >= 2:
        status = "退出复核"
        summary = "个人亏损边界与趋势破位同时触发，应优先复核退出或明显降低风险敞口。"
    elif hard_count >= 1 and auxiliary_count >= 1:
        status = "考虑分批减仓"
        summary = "核心风险与辅助风险同时出现，可结合流动性和交易成本制定分批降低仓位方案。"
    elif hard_count >= 1 or auxiliary_count >= 1:
        status = "警戒观察"
        summary = "已经出现风险信号，暂不宜仅因短期反弹而忽略原定退出纪律。"
    else:
        status = "继续持有"
        summary = "当前未触发设定的核心或辅助卖出条件，继续按所选周期复核。"

    reference_conditions: list[str] = []
    if cost_protection_price is not None:
        reference_conditions.append(
            f"成本风险参考：收盘价降至或低于 {cost_protection_price:.3f} {bundle.price_unit}。"
        )
    else:
        reference_conditions.append("未取得每股成本价，因此不显示可能误导的成本保护价格。")
    reference_conditions.append(
        f"趋势确认参考：连续2个交易日收盘低于MA{fast_window}（当前 {fast_ma:.3f}），"
        f"同时MA{fast_window}低于MA{slow_window}。"
    )
    if trailing_activated:
        reference_conditions.append(
            f"盈利回撤参考：近期高点保护价约 {trailing_protection_price:.3f} {bundle.price_unit}。"
        )
    else:
        reference_conditions.append("盈利回撤保护尚未启用；需要先达到与股票波动相匹配的浮盈。")

    limitations = [
        "本模块使用最近公开日线，只适合收盘后复核，不提供盘中实时卖出提醒。",
        "网页关闭后不会在后台自动监控，重新打开或刷新后才会获取最新公开数据。",
        "基本面数据采用最近公开口径；未取得数据的项目不会被当成利空。",
        "参考触发价不是自动止损订单，也不保证实际成交价格。",
    ]
    if not bundle.history_complete:
        limitations.append("该股票可得历史不足五年，卖出信号的统计背景置信度已降低。")

    return {
        "available": True,
        "status": status,
        "summary": summary,
        "hard_count": hard_count,
        "auxiliary_count": auxiliary_count,
        "signals": signals,
        "latest_price": latest_price,
        "latest_date": latest_date,
        "selected_horizon": selected.get("name") or "数据不足",
        "next_review": selected.get("review") or "下一交易日收盘后复核",
        "fast_window": fast_window,
        "slow_window": slow_window,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
        "current_return": current_return,
        "recent_peak": recent_peak,
        "peak_window": peak_window,
        "peak_drawdown": peak_drawdown,
        "personal_loss_limit": dynamic_loss_limit,
        "cost_protection_price": cost_protection_price,
        "trailing_limit": trailing_limit,
        "trailing_protection_price": trailing_protection_price if trailing_activated else None,
        "reference_conditions": reference_conditions,
        "limitations": limitations,
    }


def analyze_all(
    bundle: PriceBundle,
    profile: dict[str, Any],
    fundamental: EvidenceSnapshot,
    macro: EvidenceSnapshot,
) -> dict[str, Any]:
    metrics = calculate_quant_metrics(bundle.stock, bundle.benchmark)
    metrics["history_complete"] = bundle.history_complete
    metrics["coverage_ratio"] = bundle.coverage_ratio
    metrics["benchmark_available"] = bundle.benchmark is not None and not bundle.benchmark.empty
    investor_score, investor_number, investor_level, style, profile_flags = score_investor(profile)
    stock_risk_score, stock_risk_number, stock_risk_level, risk_reasons = score_stock_risk(metrics)
    analog_forecast = analyze_historical_analogs(
        bundle.stock,
        bundle.benchmark,
        history_complete=bundle.history_complete,
        source_label="目标股票（结合市场状态）",
    )
    analog_forecast["notes"].append(
        "本版使用目标股票自身历史，并把市场基准状态纳入相似度；未取得稳定行业指数时，不用无关股票冒充行业样本。"
    )
    if bundle.benchmark is not None and not bundle.benchmark.empty:
        market_analog_forecast = analyze_historical_analogs(
            bundle.benchmark,
            None,
            history_complete=len(bundle.benchmark) >= 1000,
            source_label=bundle.benchmark_name,
        )
    else:
        market_analog_forecast = {
            "available": False,
            "source_label": bundle.benchmark_name,
            "confidence_label": "基准数据不足",
            "horizons": [],
            "matches": [],
            "notes": ["市场基准未返回足够数据，无法单独检索市场相似周期。"],
        }
    horizon_scores = score_horizons(
        metrics,
        fundamental,
        macro,
        analog_forecast,
        market_analog_forecast,
        bundle.asset_type,
    )
    selected_horizon, horizon_notes = choose_horizon(horizon_scores, profile)
    confidence, confidence_notes = calculate_data_confidence(metrics, fundamental, macro, bundle.asset_type)
    if not analog_forecast["available"]:
        confidence_notes.append("相似周期样本不足，相关情景预测未参与结论。")
    suitability_result = assess_suitability(profile, investor_number, stock_risk_number, selected_horizon, confidence)
    position = position_budget(profile, investor_number, stock_risk_number, suitability_result, selected_horizon)
    conclusion, conclusion_reason = build_final_conclusion(suitability_result, selected_horizon, position)
    selected_analog = None
    if selected_horizon:
        selected_analog = next(
            (
                item
                for item in analog_forecast.get("horizons", [])
                if item.get("available")
                and int(item.get("confidence_score", 0)) >= ANALOG_SCORE_MIN_CONFIDENCE
                and int(item.get("days", -1)) == int(selected_horizon["days"])
            ),
            None,
        )
    if selected_analog:
        conclusion_reason += (
            f" 相似历史状态后{selected_analog['days']}个交易日上涨样本占比"
            f"{selected_analog['positive_ratio']:.3%}，中位收益{selected_analog['median_return']:.3%}；"
            "该频率仅作情景展示，不参与V6.6评分，也不等于确定概率。"
        )
    market_signal = selected_horizon.get("label") if selected_horizon else "证据不足"
    signal_confidence = int(selected_horizon.get("signal_confidence") or 0) if selected_horizon else 0
    return {
        "metrics": metrics,
        "investor_score": investor_score,
        "investor_number": investor_number,
        "investor_level": investor_level,
        "style": style,
        "profile_flags": profile_flags,
        "stock_risk_score": stock_risk_score,
        "stock_risk_number": stock_risk_number,
        "stock_risk_level": stock_risk_level,
        "risk_reasons": risk_reasons,
        "horizon_scores": horizon_scores,
        "selected_horizon": selected_horizon,
        "market_signal": market_signal,
        "signal_confidence": signal_confidence,
        "horizon_notes": horizon_notes,
        "data_confidence": confidence,
        "confidence_notes": confidence_notes,
        "suitability": suitability_result,
        "position": position,
        "conclusion": conclusion,
        "conclusion_reason": conclusion_reason,
        "fundamental": fundamental,
        "macro": macro,
        "analog_forecast": analog_forecast,
        "market_analog_forecast": market_analog_forecast,
    }

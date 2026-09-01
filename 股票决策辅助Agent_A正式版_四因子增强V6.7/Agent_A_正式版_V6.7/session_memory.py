from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Mapping


def session_key(market: str, code: str) -> str:
    """Return the stable identifier used by the stock conversation store."""
    return f"{str(market).strip()}:{str(code).strip().upper()}"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def market_currency(market: str) -> str:
    return {"A股": "人民币元", "美股": "美元", "港股": "港元"}.get(str(market), "人民币元")


def calculate_position_transaction(
    *,
    market: str,
    input_method: str,
    amount_rmb: float = 0.0,
    shares: float = 0.0,
    price_native: float = 0.0,
    fx_rate: float = 1.0,
    fees_rmb: float = 0.0,
) -> dict[str, Any]:
    """Validate one recorded purchase and return its RMB principal impact."""
    cleaned_method = str(input_method).strip()
    fees = float(fees_rmb)
    if fees < 0:
        raise ValueError("交易费用不能小于0。")
    if cleaned_method == "按实际支付人民币总额":
        principal = float(amount_rmb)
        if principal <= 0:
            raise ValueError("实际支付人民币总额必须大于0。")
        return {
            "input_method": "amount",
            "principal_rmb": principal,
            "shares": None,
            "price_native": None,
            "fx_rate": None,
            "fees_rmb": 0.0,
            "currency": market_currency(market),
        }
    if cleaned_method != "按股数和成交单价":
        raise ValueError("请选择一种有效的买入记录方式。")
    parsed_shares = float(shares)
    parsed_price = float(price_native)
    parsed_fx = 1.0 if market == "A股" else float(fx_rate)
    if parsed_shares <= 0:
        raise ValueError("买入股数必须大于0。")
    if parsed_price <= 0:
        raise ValueError("成交单价必须大于0。")
    if parsed_fx <= 0:
        raise ValueError("折算汇率必须大于0。")
    principal = parsed_shares * parsed_price * parsed_fx + fees
    return {
        "input_method": "shares",
        "principal_rmb": principal,
        "shares": parsed_shares,
        "price_native": parsed_price,
        "fx_rate": parsed_fx,
        "fees_rmb": fees,
        "currency": market_currency(market),
    }


def build_position_messages(
    *,
    transaction_type: str,
    transaction: Mapping[str, Any],
    trade_date: str,
    note: str = "",
) -> list[dict[str, Any]]:
    timestamp = now_text()
    action = "首次买入" if transaction_type == "initial" else "加仓"
    principal = float(transaction.get("principal_rmb") or 0.0)
    if transaction.get("input_method") == "shares":
        detail = (
            f"{float(transaction.get('shares') or 0.0):,.3f}股 × "
            f"{float(transaction.get('price_native') or 0.0):,.3f}{transaction.get('currency', '')}/股"
        )
        if float(transaction.get("fx_rate") or 1.0) != 1.0:
            detail += f" × 汇率{float(transaction.get('fx_rate')):,.3f}"
        if float(transaction.get("fees_rmb") or 0.0) > 0:
            detail += f" + 费用{float(transaction.get('fees_rmb')):,.3f}元"
    else:
        detail = f"实际支付人民币{principal:,.3f}元"
    user_content = f"记录{trade_date}{action}：{detail}。"
    if str(note).strip():
        user_content += f" 备注：{str(note).strip()}"
    assistant_content = (
        f"已记录{action}，本次计入投入本金{principal:,.3f}元；"
        "组合本金和持仓占比已根据所有保留的有效股票会话重新计算。"
    )
    return [
        {"role": "user", "content": user_content, "created_at": timestamp, "kind": "position_record"},
        {"role": "assistant", "content": assistant_content, "created_at": timestamp, "kind": "position_confirmation"},
    ]


def known_invested_principal(holding_snapshot: Mapping[str, Any] | None) -> float:
    """Only a known historical cost is treated as invested principal."""
    if not holding_snapshot:
        return 0.0
    try:
        value = float(holding_snapshot.get("cost_total_rmb") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(value, 0.0)


def build_analysis_messages(
    *,
    market: str,
    code: str,
    name: str,
    analysis: Mapping[str, Any],
    holding_state: str,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    """Create compact, serialisable chat messages from an existing analysis result."""
    timestamp = created_at or now_text()
    selected = analysis.get("selected_horizon") or {}
    suitability = analysis.get("suitability") or {}
    news = analysis.get("news_analysis") or {}
    user_content = f"请分析{market} {code}（{name}），当前状态：{holding_state}。"
    assistant_content = (
        f"结论：{analysis.get('conclusion', '数据不足')}。"
        f"个人适配：{suitability.get('fit', '数据不足')}；"
        f"股票风险：{analysis.get('stock_risk_level', '数据不足')}；"
        f"建议复核／持有周期：{selected.get('name', '数据不足')}；"
        f"数据完整度：{float(analysis.get('data_confidence') or 0.0):.3f}%；"
        f"最新资讯倾向：{news.get('direction', '未取得有效资讯')}；"
        f"资讯修正：{int(news.get('score_adjustment') or 0):+d}分。"
    )
    return [
        {"role": "user", "content": user_content, "created_at": timestamp, "kind": "analysis_request"},
        {
            "role": "assistant",
            "content": assistant_content,
            "created_at": timestamp,
            "kind": "analysis_summary",
        },
    ]


def upsert_analysis_session(
    sessions: Mapping[str, Any] | None,
    *,
    event_id: str,
    market: str,
    code: str,
    name: str,
    analysis: Mapping[str, Any],
    holding_state: str,
    holding_snapshot: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    """Append one analysis to its stock conversation, once per analysis event."""
    updated = deepcopy(dict(sessions or {}))
    key = session_key(market, code)
    timestamp = now_text()
    record = updated.get(key)
    created = record is None
    if record is None:
        known_shares = None
        if holding_snapshot and holding_snapshot.get("method") == "按持股数量填写":
            try:
                parsed_shares = float(holding_snapshot.get("shares") or 0.0)
                known_shares = parsed_shares if parsed_shares > 0 else None
            except (TypeError, ValueError):
                known_shares = None
        record = {
            "key": key,
            "market": market,
            "code": code,
            "name": name,
            "created_at": timestamp,
            "updated_at": timestamp,
            "principal_rmb": known_invested_principal(holding_snapshot),
            "principal_source": "已知持仓成本" if known_invested_principal(holding_snapshot) > 0 else "尚未记录实际投入",
            "total_shares": known_shares,
            "shares_complete": known_shares is not None,
            "messages": [],
            "recorded_event_ids": [],
        }
    event_ids = list(record.get("recorded_event_ids") or [])
    if event_id in event_ids:
        return updated, False

    record["market"] = market
    record["code"] = code
    record["name"] = name
    record["updated_at"] = timestamp
    record["messages"] = list(record.get("messages") or []) + build_analysis_messages(
        market=market,
        code=code,
        name=name,
        analysis=analysis,
        holding_state=holding_state,
        created_at=timestamp,
    )
    record["recorded_event_ids"] = event_ids + [event_id]
    record["latest_summary"] = {
        "conclusion": analysis.get("conclusion", "数据不足"),
        "conclusion_reason": analysis.get("conclusion_reason", ""),
        "investor_level": analysis.get("investor_level", "数据不足"),
        "stock_risk_level": analysis.get("stock_risk_level", "数据不足"),
        "suitability": (analysis.get("suitability") or {}).get("fit", "数据不足"),
        "selected_horizon": (analysis.get("selected_horizon") or {}).get("name", "数据不足"),
        "data_confidence": float(analysis.get("data_confidence") or 0.0),
        "holding_state": holding_state,
        "recorded_at": timestamp,
    }
    if created:
        known_principal = known_invested_principal(holding_snapshot)
        record["principal_rmb"] = known_principal
        record["principal_source"] = "已知持仓成本" if known_principal > 0 else "尚未记录实际投入"
    elif (
        float(record.get("principal_rmb") or 0.0) <= 0
        and record.get("principal_source") != "用户在会话中确认"
        and known_invested_principal(holding_snapshot) > 0
    ):
        record["principal_rmb"] = known_invested_principal(holding_snapshot)
        record["principal_source"] = "已知持仓成本"
    updated[key] = record
    return updated, True


def set_invested_principal(
    sessions: Mapping[str, Any] | None,
    key: str,
    amount: float,
) -> dict[str, Any]:
    """Update the actual principal recorded in one retained stock conversation."""
    parsed = float(amount)
    if parsed < 0:
        raise ValueError("投入本金不能小于0。")
    updated = deepcopy(dict(sessions or {}))
    if key not in updated:
        raise KeyError("股票会话不存在或已经删除。")
    updated[key]["principal_rmb"] = parsed
    updated[key]["principal_source"] = "用户在会话中确认"
    updated[key]["updated_at"] = now_text()
    return updated


def append_note(
    sessions: Mapping[str, Any] | None,
    key: str,
    content: str,
) -> dict[str, Any]:
    cleaned = str(content).strip()
    if not cleaned:
        return deepcopy(dict(sessions or {}))
    updated = deepcopy(dict(sessions or {}))
    if key not in updated:
        raise KeyError("股票会话不存在或已经删除。")
    timestamp = now_text()
    updated[key]["messages"] = list(updated[key].get("messages") or []) + [
        {"role": "user", "content": cleaned, "created_at": timestamp, "kind": "note"}
    ]
    updated[key]["updated_at"] = timestamp
    return updated


def delete_session(sessions: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    """Deleting a conversation is the business event 'fully sold'."""
    updated = deepcopy(dict(sessions or {}))
    updated.pop(key, None)
    return updated


def portfolio_from_sessions(sessions: Mapping[str, Any] | None) -> dict[str, Any]:
    """Calculate portfolio weights from retained conversation principals only."""
    rows: list[dict[str, Any]] = []
    for key, record in dict(sessions or {}).items():
        try:
            principal = max(float(record.get("principal_rmb") or 0.0), 0.0)
        except (TypeError, ValueError):
            principal = 0.0
        if principal <= 0:
            continue
        rows.append(
            {
                "key": key,
                "market": record.get("market", "—"),
                "code": record.get("code", "—"),
                "name": record.get("name", "—"),
                "principal_rmb": principal,
                "updated_at": record.get("updated_at", ""),
            }
        )
    rows.sort(key=lambda item: (-item["principal_rmb"], item["key"]))
    total = sum(item["principal_rmb"] for item in rows)
    for item in rows:
        item["weight"] = item["principal_rmb"] / total if total > 0 else 0.0
    return {
        "rows": rows,
        "total_principal_rmb": total,
        "position_count": len(rows),
        "conversation_count": len(dict(sessions or {})),
    }

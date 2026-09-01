from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping


ASSESSMENT_VERSION = 1


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def build_current_holding_snapshot(
    *,
    session: Mapping[str, Any],
    latest_price_native: float,
    fx_rate: float,
    price_unit: str,
) -> dict[str, Any]:
    """Build a conservative current holding view from retained session data.

    A session always knows invested RMB principal.  Market value is exact only
    when the retained share count is complete; otherwise principal is used as a
    clearly-labelled exposure proxy rather than inventing a share count.
    """
    principal = max(_as_float(session.get("principal_rmb")), 0.0)
    shares_value = session.get("total_shares")
    shares_complete = bool(session.get("shares_complete")) and shares_value is not None
    shares = max(_as_float(shares_value), 0.0) if shares_complete else None
    parsed_price = max(_as_float(latest_price_native), 0.0)
    parsed_fx = max(_as_float(fx_rate, 1.0), 0.0)
    if parsed_fx <= 0:
        parsed_fx = 1.0

    if shares is not None and shares > 0 and parsed_price > 0:
        current_rmb = shares * parsed_price * parsed_fx
        return_rate = current_rmb / principal - 1 if principal > 0 else None
        cost_price_native = principal / shares / parsed_fx if principal > 0 else None
        value_source = "已记录股数 × 最新公开价格 × 当前参考汇率"
    else:
        current_rmb = principal
        return_rate = None
        cost_price_native = None
        value_source = "股数不完整，暂以累计投入本金作为风险敞口代理"

    return {
        "principal_rmb": principal,
        "shares": shares,
        "shares_complete": shares is not None and shares > 0,
        "latest_price_native": parsed_price,
        "fx_rate": parsed_fx,
        "current_rmb": current_rmb,
        "return_rate": return_rate,
        "cost_price": cost_price_native,
        "cost_total_rmb": principal,
        "current_price": parsed_price,
        "price_unit": price_unit,
        "value_source": value_source,
    }


def evaluate_add_position(
    *,
    session: Mapping[str, Any],
    transaction: Mapping[str, Any],
    analysis: Mapping[str, Any],
    sell_signals: Mapping[str, Any] | None,
    profile: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    holding_snapshot: Mapping[str, Any],
    market_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a planned add without changing the retained position.

    The function deliberately consumes the existing analysis outputs instead
    of modifying their buy-side scoring rules.  It layers current holding,
    portfolio concentration and sell-signal checks on top.
    """
    planned_principal = max(_as_float(transaction.get("principal_rmb")), 0.0)
    current_principal = max(_as_float(session.get("principal_rmb")), 0.0)
    current_value = max(_as_float(holding_snapshot.get("current_rmb")), 0.0)
    investable_assets = max(_as_float(profile.get("investable_assets")), 0.0)
    post_value = current_value + planned_principal

    rows = list(portfolio.get("rows") or [])
    portfolio_total_before = max(_as_float(portfolio.get("total_principal_rmb")), 0.0)
    portfolio_total_after = portfolio_total_before + planned_principal
    post_session_principal = current_principal + planned_principal
    current_weight = current_principal / portfolio_total_before if portfolio_total_before > 0 else 0.0
    post_weight = post_session_principal / portfolio_total_after if portfolio_total_after > 0 else 0.0

    current_shares = holding_snapshot.get("shares")
    planned_shares = transaction.get("shares")
    post_shares = None
    post_average_cost_rmb = None
    if current_shares is not None and planned_shares is not None:
        parsed_current_shares = max(_as_float(current_shares), 0.0)
        parsed_planned_shares = max(_as_float(planned_shares), 0.0)
        if parsed_current_shares + parsed_planned_shares > 0:
            post_shares = parsed_current_shares + parsed_planned_shares
            post_average_cost_rmb = post_session_principal / post_shares

    suitability = dict(analysis.get("suitability") or {})
    selected = dict(analysis.get("selected_horizon") or {})
    news_analysis = dict(analysis.get("news_analysis") or {})
    position = dict(analysis.get("position") or {})
    fit = str(suitability.get("fit") or "证据不足")
    direction_available = bool(selected.get("direction_available"))
    base_timing_score = selected.get("score")
    timing_score_value = _as_float(base_timing_score, 0.0) if base_timing_score is not None else None
    if direction_available and news_analysis.get("usable_for_score") and news_analysis.get("combined_score") is not None:
        timing_score_value = _as_float(news_analysis.get("combined_score"), timing_score_value or 0.0)
    data_confidence = _as_float(analysis.get("data_confidence"), 0.0)
    upper_amount = max(_as_float(position.get("upper_amount")), 0.0)
    remaining_upper = max(upper_amount - current_value, 0.0)
    sell = dict(sell_signals or {})
    sell_status = str(sell.get("status") or "证据不足")

    hard_reasons: list[str] = []
    conditional_reasons: list[str] = []
    support_factors: list[str] = []
    trigger_conditions: list[str] = []
    stop_add_signals: list[str] = []

    if current_principal <= 0:
        hard_reasons.append("该会话尚未记录实际持仓本金，不能把计划加仓与现有仓位合并判断。")
    if planned_principal <= 0:
        hard_reasons.append("计划加仓本金必须大于0。")
    if fit in {"不适配", "证据不足"}:
        hard_reasons.append(f"当前个人适配结果为“{fit}”：{suitability.get('fit_reason', '证据不足')}。")
    elif fit == "有限适配":
        conditional_reasons.append("股票风险比用户风险等级高一级，只能在更严格的风险预算内评估。")
    else:
        support_factors.append("用户风险承受能力覆盖该股票的模型风险等级。")

    if not direction_available:
        validation = dict(selected.get("signal_validation") or {})
        hard_reasons.append(
            f"所选持有期的历史验证{validation.get('status', '未通过')}，不能把当前评分解释为可靠的加仓信号。"
        )
        trigger_conditions.append("等待该持有期历史验证通过，或取得更可靠的数据后重新分析。")
    elif timing_score_value is None:
        hard_reasons.append("当前没有可用的持有周期评分，无法评价加仓时点。")
    elif timing_score_value < 45:
        hard_reasons.append(f"当前时点评分仅{timing_score_value:.0f}/100，尚未达到观察分界。")
        trigger_conditions.append("等待所选周期评分恢复到45分以上，并重新检查趋势。")
    elif timing_score_value < 60:
        conditional_reasons.append(f"当前时点评分为{timing_score_value:.0f}/100，尚未达到中性偏积极分界。")
        trigger_conditions.append("等待所选周期评分达到60分以上，或支持因素明显增强后重新评估。")
    else:
        support_factors.append(f"当前所选周期评分为{timing_score_value:.0f}/100，未处于偏弱区间。")

    news_adjustment = int(news_analysis.get("score_adjustment") or 0)
    if news_analysis.get("usable_for_score"):
        if news_adjustment <= -4:
            conditional_reasons.append(
                f"近期有效公开资讯整体偏负面，对原有时点评分作{news_adjustment:+d}分修正。"
            )
            trigger_conditions.append("等待重大负面事件得到澄清或被新的正式披露替代后重新分析。")
            stop_add_signals.append("新的高相关负面公告或监管、业绩风险资讯出现。")
        elif news_adjustment >= 4:
            support_factors.append(
                f"近期有效公开资讯整体偏正面，对原有时点评分作{news_adjustment:+d}分有限修正。"
            )
        else:
            support_factors.append("近期公开资讯整体中性或影响有限，未明显改变原有量化判断。")
    elif news_analysis.get("available"):
        conditional_reasons.append("已检索到参考资讯，但有效样本或可信度不足，未用于修正加仓评分。")

    if data_confidence < 35:
        hard_reasons.append(f"数据完整度仅{data_confidence:.3f}%，不足以形成可靠的加仓判断。")
    elif data_confidence < 60:
        conditional_reasons.append(f"数据完整度为{data_confidence:.3f}%，结论可信度需要下调。")
    else:
        support_factors.append(f"数据完整度为{data_confidence:.3f}%，达到基础分析要求。")

    if upper_amount <= 0:
        hard_reasons.append("模型当前给出的单股风险预算上限为0，暂不应新增风险敞口。")
    elif planned_principal > remaining_upper + 0.005:
        hard_reasons.append(
            f"计划加仓{planned_principal:,.3f}元超过当前剩余风险预算{remaining_upper:,.3f}元。"
        )
        trigger_conditions.append(f"如后续条件改善，计划新增金额仍应控制在{remaining_upper:,.3f}元以内。")
        stop_add_signals.append("计划加仓后超过个人单股风险预算上限。")
    else:
        support_factors.append("本次计划金额未超过当前模型风险预算的剩余额度。")

    if sell_status in {"退出复核", "考虑分批减仓"}:
        hard_reasons.append(f"现有持仓卖出模块状态为“{sell_status}”，加仓方向与当前风险信号冲突。")
        trigger_conditions.append("等待核心卖出信号解除，再重新进行加仓分析。")
        stop_add_signals.append(f"卖出信号状态达到“{sell_status}”。")
    elif sell_status == "警戒观察":
        conditional_reasons.append("现有持仓正处于警戒观察状态，不宜在风险信号未解除时直接加仓。")
        trigger_conditions.append("等待警戒信号解除，并在下一个复核周期重新分析。")
    elif sell_status == "继续持有":
        support_factors.append("当前没有触发设定的核心或辅助卖出条件。")
    else:
        conditional_reasons.append("卖出信号证据不足，需要先确认现有持仓风险。")

    trend_signal = next(
        (item for item in list(sell.get("signals") or []) if item.get("key") == "trend_break"),
        None,
    )
    if trend_signal and str(trend_signal.get("state")) in {"触发", "警戒"}:
        conditional_reasons.append(f"趋势信号当前为“{trend_signal.get('state')}”。")
        trigger_conditions.append(str(trend_signal.get("threshold_text") or "等待趋势结构重新确认。"))
        stop_add_signals.append(str(trend_signal.get("current_text") or "趋势结构转弱。"))

    position_count = int(portfolio.get("position_count") or 0)
    if position_count >= 2 and post_weight > 0.50:
        conditional_reasons.append(f"加仓后该股票将占会话组合{post_weight:.3%}，集中度较高。")
        trigger_conditions.append("优先检查其他股票持仓与单股集中度，再决定是否增加本标的风险敞口。")
    elif position_count >= 2:
        support_factors.append(f"加仓后该股票约占会话组合{post_weight:.3%}。")

    history_complete = bool(market_data.get("history_complete"))
    if not history_complete:
        conditional_reasons.append("该股票可得历史不足五年，加仓评估只能作为低置信度参考。")

    if hard_reasons:
        conclusion = "当前暂不适合加仓"
        reason = hard_reasons[0]
    elif conditional_reasons:
        conclusion = "满足条件后再考虑加仓"
        reason = conditional_reasons[0]
    else:
        conclusion = "可在风险预算内小额分批加仓"
        reason = "个人适配、当前时点、卖出信号和仓位预算暂未出现硬性冲突。"

    condition_score = timing_score_value if timing_score_value is not None else 0.0
    condition_score += 8 if fit == "适配" else -8 if fit == "有限适配" else -30
    condition_score += 8 if sell_status == "继续持有" else -12 if sell_status == "警戒观察" else -30
    condition_score += 8 if planned_principal <= remaining_upper and remaining_upper > 0 else -25
    condition_score += (data_confidence - 50) * 0.12
    condition_score -= min(len(hard_reasons) * 8, 24)
    condition_score = int(round(_clamp(condition_score, 0.0, 100.0)))

    if not trigger_conditions and conclusion.startswith("可在"):
        trigger_conditions.extend(
            [
                "如实际成交前价格、趋势或卖出信号发生变化，应重新运行本页分析。",
                "分批执行时，每次实际成交后先更新持仓，再评估下一次加仓。",
            ]
        )
    if not stop_add_signals:
        stop_add_signals.extend(
            [
                "计划金额导致加仓后风险敞口超过模型仓位上限。",
                "卖出模块出现核心风险信号或趋势破位。",
                "个人资金用途、应急储备或风险承受能力发生变化。",
            ]
        )

    timestamp = str(market_data.get("assessed_at") or datetime.now().isoformat(timespec="seconds"))
    return {
        "version": ASSESSMENT_VERSION,
        "assessed_at": timestamp,
        "market": str(session.get("market") or ""),
        "code": str(session.get("code") or ""),
        "name": str(session.get("name") or session.get("code") or ""),
        "conclusion": conclusion,
        "reason": reason,
        "condition_score": condition_score,
        "planned_principal_rmb": planned_principal,
        "input_method": str(transaction.get("input_method") or "amount"),
        "planned_shares": transaction.get("shares"),
        "planned_price_native": transaction.get("price_native"),
        "planned_fx_rate": transaction.get("fx_rate"),
        "current_principal_rmb": current_principal,
        "current_market_value_rmb": current_value,
        "current_value_source": str(holding_snapshot.get("value_source") or ""),
        "current_shares": holding_snapshot.get("shares"),
        "current_average_cost_rmb": (
            current_principal / _as_float(holding_snapshot.get("shares"))
            if _as_float(holding_snapshot.get("shares")) > 0
            else None
        ),
        "post_principal_rmb": post_session_principal,
        "post_market_exposure_rmb": post_value,
        "post_shares": post_shares,
        "post_average_cost_rmb": post_average_cost_rmb,
        "investable_assets_rmb": investable_assets,
        "current_asset_pct": current_value / investable_assets if investable_assets > 0 else 0.0,
        "post_asset_pct": post_value / investable_assets if investable_assets > 0 else 0.0,
        "current_portfolio_weight": current_weight,
        "post_portfolio_weight": post_weight,
        "model_upper_amount_rmb": upper_amount,
        "remaining_upper_amount_rmb": remaining_upper,
        "timing_score": timing_score_value,
        "base_timing_score": _as_float(base_timing_score) if base_timing_score is not None else None,
        "news_analysis": news_analysis,
        "selected_horizon": str(selected.get("name") or "数据不足"),
        "data_confidence": data_confidence,
        "suitability": fit,
        "sell_status": sell_status,
        "hard_reasons": hard_reasons,
        "conditional_reasons": conditional_reasons,
        "support_factors": support_factors,
        "trigger_conditions": list(dict.fromkeys(trigger_conditions)),
        "stop_add_signals": list(dict.fromkeys(stop_add_signals)),
        "latest_price_native": _as_float(market_data.get("latest_price_native")),
        "price_unit": str(market_data.get("price_unit") or ""),
        "latest_market_date": str(market_data.get("latest_market_date") or ""),
        "provider": str(market_data.get("provider") or ""),
        "history_complete": history_complete,
        "note": "本次仅进行加仓适配测算，没有改变实际持仓；真实成交后仍需另行登记。",
    }


def build_add_position_messages(
    *,
    transaction: Mapping[str, Any],
    assessment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    timestamp = str(assessment.get("assessed_at") or datetime.now().isoformat(timespec="seconds"))
    principal = _as_float(transaction.get("principal_rmb"))
    if transaction.get("input_method") == "shares":
        plan_text = (
            f"计划买入{_as_float(transaction.get('shares')):,.3f}股，"
            f"预计成交价{_as_float(transaction.get('price_native')):,.3f}，"
            f"折合计划本金{principal:,.3f}元"
        )
    else:
        plan_text = f"计划新增人民币{principal:,.3f}元"
    news = dict(assessment.get("news_analysis") or {})
    news_text = (
        f"最新资讯倾向：{news.get('direction', '未取得有效资讯')}，"
        f"资讯修正{int(news.get('score_adjustment') or 0):+d}分；"
    )
    user_message = {
        "role": "user",
        "content": f"请判断本股票现在是否适合加仓：{plan_text}。",
        "created_at": timestamp,
        "kind": "add_position_request",
    }
    assistant_message = {
        "role": "assistant",
        "content": (
            f"加仓适配结论：{assessment.get('conclusion', '证据不足')}。"
            f"加仓条件分：{int(assessment.get('condition_score') or 0)}/100；"
            f"当前剩余风险预算参考上限：{_as_float(assessment.get('remaining_upper_amount_rmb')):,.3f}元；"
            f"加仓后风险敞口约占可投资金融资产{_as_float(assessment.get('post_asset_pct')):.3%}。"
            f"{news_text}"
            "本次只保存分析，没有记为真实成交。"
        ),
        "created_at": timestamp,
        "kind": "add_position_assessment",
        "data": dict(assessment),
    }
    return [user_message, assistant_message]

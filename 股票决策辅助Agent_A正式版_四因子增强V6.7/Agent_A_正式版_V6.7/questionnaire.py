from __future__ import annotations

from typing import Any


QUESTIONNAIRE_VERSION = 1


QUESTIONS: list[dict[str, Any]] = [
    {
        "key": "asset_band",
        "title": "你目前可用于证券投资的金融资产大约处于哪个范围？",
        "hint": "只保存区间，不要求填写精确资产金额。",
        "options": ["5万元以下", "5万—20万元", "20万—50万元", "50万—100万元", "100万—300万元", "300万元以上"],
    },
    {
        "key": "fund_source",
        "title": "你用于股票投资的资金主要来自哪里？",
        "hint": "借款、应急资金或近期必须使用的资金不适合承担单只股票波动。",
        "options": ["闲置自有资金", "未来有明确用途的资金", "应急资金", "借款／融资资金"],
    },
    {
        "key": "emergency_reserve",
        "title": "你目前预留的生活应急资金可以覆盖多长时间？",
        "hint": "应急储备越充足，越不容易因为临时用款被迫卖出。",
        "options": ["不足3个月", "3—6个月", "6个月以上"],
    },
    {
        "key": "earliest_need",
        "title": "用于投资的资金最早什么时候可能需要取回？",
        "hint": "这会限制 Agent 可以选择的持有周期。",
        "options": ["1周内", "1个月内", "3个月内", "1年内", "3年内", "没有明确时间"],
    },
    {
        "key": "income_stability",
        "title": "你目前的收入稳定程度如何？",
        "hint": "请选择最符合当前实际情况的一项。",
        "options": ["不稳定", "较稳定", "稳定"],
    },
    {
        "key": "max_loss",
        "title": "在不影响正常生活的前提下，你最多能承受多大投资账面亏损？",
        "hint": "这里询问的是整个投资计划可以承受的损失，不是期望收益。",
        "options": ["不超过5%", "5%—10%", "10%—20%", "20%—30%", "超过30%"],
    },
    {
        "key": "loss_response",
        "title": "假设10万元下跌到8万元，你更可能怎么处理？",
        "hint": "没有标准答案，请按照真实行为选择。",
        "options": ["立即全部卖出", "大部分减仓", "先复核原因再决定", "继续按原计划持有", "在条件允许时分批增加"],
    },
    {
        "key": "goal",
        "title": "你的主要投资目标是什么？",
        "hint": "Agent 会结合目标选择更合适的分析周期。",
        "options": ["保值为主", "股息／稳健收益", "长期增值", "波段操作", "短线交易"],
    },
    {
        "key": "experience",
        "title": "你的股票投资经验有多长？",
        "hint": "经验与风险承受能力分开评价，经验丰富不等于能承受更大亏损。",
        "options": ["没有经验", "不足1年", "1—3年", "3年以上"],
    },
    {
        "key": "trade_frequency",
        "title": "你通常多久进行一次股票交易？",
        "hint": "请选择最接近实际习惯的一项。",
        "options": ["几乎不交易", "每月1—3次", "每周1—3次", "几乎每天"],
    },
    {
        "key": "monitor_time",
        "title": "你每天可以用于查看行情和公告的时间有多少？",
        "hint": "看盘时间不足时，短线策略的执行风险会明显上升。",
        "options": ["不足15分钟", "15—30分钟", "30—60分钟", "1小时以上"],
    },
    {
        "key": "existing_concentration",
        "title": "你现有投资中，仓位最高的一只股票约占多少？",
        "hint": "该问题用于识别集中度风险。",
        "options": ["目前没有股票持仓", "不足10%", "10%—30%", "30%—50%", "超过50%"],
    },
    {
        "key": "stop_loss",
        "title": "你是否有明确并能执行的止损、退出或复核规则？",
        "hint": "长期持有也需要在基本面恶化时重新评估。",
        "options": ["有明确规则并能执行", "有规则但经常改变", "没有明确规则"],
    },
    {
        "key": "fx_acceptance",
        "title": "如果投资美股，你能否接受人民币兑美元波动影响最终收益？",
        "hint": "即使暂时只投资A股，也请按照未来可能投资美股时的真实接受程度选择。",
        "options": ["能够接受", "只能接受较小波动", "不能接受", "不确定"],
    },
]


ASSET_BANDS: dict[str, dict[str, float | None]] = {
    "5万元以下": {"estimate": 30_000.0, "upper": 50_000.0},
    "5万—20万元": {"estimate": 125_000.0, "upper": 200_000.0},
    "20万—50万元": {"estimate": 350_000.0, "upper": 500_000.0},
    "50万—100万元": {"estimate": 750_000.0, "upper": 1_000_000.0},
    "100万—300万元": {"estimate": 2_000_000.0, "upper": 3_000_000.0},
    "300万元以上": {"estimate": 5_000_000.0, "upper": None},
}


def question_keys() -> list[str]:
    return [str(item["key"]) for item in QUESTIONS]


def answers_complete(answers: dict[str, Any] | None) -> bool:
    if not isinstance(answers, dict):
        return False
    for question in QUESTIONS:
        key = str(question["key"])
        if answers.get(key) not in question["options"]:
            return False
    return True


def first_unanswered_index(answers: dict[str, Any] | None) -> int:
    values = answers if isinstance(answers, dict) else {}
    for index, question in enumerate(QUESTIONS):
        if values.get(question["key"]) not in question["options"]:
            return index
    return len(QUESTIONS)


def answers_to_profile(answers: dict[str, Any]) -> dict[str, Any]:
    if not answers_complete(answers):
        raise ValueError("风险问卷尚未全部完成。")
    asset = ASSET_BANDS[str(answers["asset_band"])]
    return {
        "questionnaire_version": QUESTIONNAIRE_VERSION,
        "asset_band": answers["asset_band"],
        "investable_assets": float(asset["estimate"] or 0.0),
        "asset_upper": asset["upper"],
        "fund_source": answers["fund_source"],
        "emergency_reserve": answers["emergency_reserve"],
        "earliest_need": answers["earliest_need"],
        "income_stability": answers["income_stability"],
        "max_loss": answers["max_loss"],
        "loss_response": answers["loss_response"],
        "goal": answers["goal"],
        "experience": answers["experience"],
        "trade_frequency": answers["trade_frequency"],
        "monitor_time": answers["monitor_time"],
        "existing_concentration": answers["existing_concentration"],
        "stop_loss": answers["stop_loss"],
        "fx_acceptance": answers["fx_acceptance"],
        "fundamental_action": "会重新评估" if answers["stop_loss"] == "有明确规则并能执行" else "不确定如何判断",
    }


def compose_analysis_profile(
    saved_profile: dict[str, Any],
    planned_amount: float,
    leverage: str,
) -> dict[str, Any]:
    profile = dict(saved_profile)
    profile["planned_amount"] = float(planned_amount)
    profile["leverage"] = leverage
    return profile


def public_profile_rows(profile: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("可投资金融资产", str(profile.get("asset_band", "未填写"))),
        ("资金来源", str(profile.get("fund_source", "未填写"))),
        ("应急储备", str(profile.get("emergency_reserve", "未填写"))),
        ("最早用款时间", str(profile.get("earliest_need", "未填写"))),
        ("收入稳定性", str(profile.get("income_stability", "未填写"))),
        ("最大可承受损失", str(profile.get("max_loss", "未填写"))),
        ("投资目标", str(profile.get("goal", "未填写"))),
        ("投资经验", str(profile.get("experience", "未填写"))),
        ("交易频率", str(profile.get("trade_frequency", "未填写"))),
        ("每日查看时间", str(profile.get("monitor_time", "未填写"))),
        ("最高单股集中度", str(profile.get("existing_concentration", "未填写"))),
        ("退出／复核纪律", str(profile.get("stop_loss", "未填写"))),
        ("美股汇率波动", str(profile.get("fx_acceptance", "未填写"))),
    ]

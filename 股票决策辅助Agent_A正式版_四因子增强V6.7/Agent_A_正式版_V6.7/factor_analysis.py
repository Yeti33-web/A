from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

import numpy as np
import pandas as pd

from agent_core import (
    EvidenceSnapshot,
    PriceBundle,
    _technical_timing_frame,
    _validate_timing_signal,
    quality_factor_contributions,
    safe_float,
)


def _catalog_row(
    module: str,
    factor: str,
    formula: str,
    data_requirement: str,
    direction: str,
    weight: str,
    missing_rule: str,
) -> dict[str, str]:
    return {
        "模块": module,
        "因子": factor,
        "计算公式／规则": formula,
        "数据要求": data_requirement,
        "方向": direction,
        "当前权重／分值": weight,
        "缺失数据处理": missing_rule,
    }


def factor_catalog() -> list[dict[str, str]]:
    """Return the auditable factor dictionary used by V6.7.

    The dictionary deliberately separates predictive factors from suitability,
    confidence and risk-control rules. A rule can affect the final decision even
    when it is not intended to predict the next return.
    """

    rows: list[dict[str, str]] = []

    rows.extend(
        [
            _catalog_row("用户风险分", "资金来源", "按选项映射为15／8／0／0分", "风险问卷单选", "闲置自有资金提高承受分；应急或融资资金触发限制", "0—15分", "问卷未完成则不能进入分析"),
            _catalog_row("用户风险分", "应急储备", "不足3个月=0；3—6个月=8；6个月以上=15", "风险问卷单选", "储备越充足，风险承受分越高", "0—15分", "问卷未完成则不能进入分析"),
            _catalog_row("用户风险分", "最早用款时间", "1周内／1个月内／3个月内／1年内／3年内／无明确时间=0／3／7／12／15／15", "风险问卷单选", "资金可使用时间越长，风险承受分越高", "0—15分；同时限制可选持有周期", "问卷未完成则不能进入分析"),
            _catalog_row("用户风险分", "亏损后的实际反应", "立即卖出／大部减仓／复核／持有／条件允许时增加=1／4／8／11／13", "风险问卷单选", "能按计划复核并承受波动时分值较高", "1—13分", "问卷未完成则不能进入分析"),
            _catalog_row("用户风险分", "最大可承受损失", "≤5%／5%—10%／10%—20%／20%—30%／>30%=0／4／8／12／15", "风险问卷单选", "承受区间越高，风险分越高；同时约束持仓亏损边界", "0—15分", "缺失时旧快照按10%—20%处理"),
            _catalog_row("用户风险分", "投资目标", "保值／稳健收益／长期增值／波段／短线=0／3／6／8／10", "风险问卷单选", "收益目标越进取，风险分越高", "0—10分；另有持有期先验", "问卷未完成则不能进入分析"),
            _catalog_row("用户风险分", "收入稳定性", "不稳定／较稳定／稳定=2／7／10", "风险问卷单选", "收入越稳定，风险承受分越高", "2—10分", "问卷未完成则不能进入分析"),
            _catalog_row("用户风险分", "投资经验", "无经验／不足1年／1—3年／3年以上=0／2／4／5", "风险问卷单选", "经验越长，知识分越高；不等同于风险承受力", "0—5分；短期周期可再减6分", "问卷未完成则不能进入分析"),
            _catalog_row("周期选择约束", "交易频率", "几乎每天或短线目标时标记为活跃交易型", "风险问卷单选", "只影响用户类型，不直接增加风险分", "0分", "缺失时无法完成问卷"),
            _catalog_row("周期选择约束", "每日看盘时间", "短周期且不足15分钟时，周期选择分-9", "风险问卷单选", "看盘时间不足降低短期周期可执行性", "0或-9分", "缺失时无法完成问卷"),
            _catalog_row("周期选择约束", "退出纪律", "短周期且规则缺失或经常改变时，周期选择分-7", "风险问卷单选", "纪律不足降低短期周期可执行性", "0或-7分", "缺失时无法完成问卷"),
            _catalog_row("适配与仓位", "可投资资产区间", "用区间代表值作为仓位金额分母", "风险问卷单选", "不预测涨跌；决定金额上限和集中度", "规则变量", "问卷未完成则不能进入分析"),
            _catalog_row("适配与仓位", "现有最高单股集中度", "30%—50%或>50%时单股上限×0.60", "风险问卷单选", "集中度越高，新增仓位上限越低", "乘数1.00或0.60", "缺失时无法完成问卷"),
            _catalog_row("适配与仓位", "计划投入／现有风险敞口", "计划金额÷可投资资产；已持仓时合并现有市值", "用户本次填写", "占比过高触发安全限制或集中度提示", "规则变量", "金额≤0不允许提交；成本缺失时不编造收益率"),
            _catalog_row("适配与仓位", "杠杆使用", "选择“是”时加入风险警示", "本次分析单选", "杠杆提高风险", "当前为警示，不直接加减风险分", "缺失默认不使用杠杆"),
            _catalog_row("问卷保留项", "外汇波动接受度", "当前只记录并展示，尚未进入数值评分", "风险问卷单选", "后续可用于美股／港股汇率风险适配", "0分（待深化）", "缺失时无法完成问卷"),
        ]
    )

    rows.extend(
        [
            _catalog_row("股票风险分", "近一年年化波动率", "clip((σ252-12%)/58%×35,0,35)", "至少约120日收益；优先252日", "越高表示股票风险越高", "35%", "缺失时使用20分中性偏高替代"),
            _catalog_row("股票风险分", "近五年或上市以来最大回撤", "clip((abs(min(P/历史峰值-1))-12%)/68%×35,0,35)", "全部可得日线收盘价", "回撤越深，风险越高", "35%", "行情不足时按已有全部数据并降低完整度"),
            _catalog_row("股票风险分", "下行波动率", "负收益样本标准差×√252；再映射为0—20分", "近一年日收益中的负收益", "越高表示下行风险越高", "20%", "无负收益记0；无法计算时使用10分"),
            _catalog_row("股票风险分", "Beta", "Cov(r股票,r基准)/Var(r基准)；映射为0—10分", "股票与基准至少60个对齐交易日", "Beta越高，系统性风险分越高", "10%", "缺失时Beta按1.0并降低数据完整度"),
        ]
    )

    rows.extend(
        [
            _catalog_row("基本面评分", "净资产收益率ROE", "≥15%:+10；8%—15%:+5；<0:-12", "最近可得财报", "越高越正面；负值为风险", "+10至-12分", "缺失则本项0分"),
            _catalog_row("基本面评分", "净利率", "≥15%:+8；5%—15%:+3；<0:-10", "最近可得利润和收入", "越高越正面；负值为风险", "+8至-10分", "缺失则本项0分"),
            _catalog_row("基本面评分", "净利润同比", "≥15%:+8；<0:-9", "连续可比报告期净利润", "增长为正面，下降为负面", "+8至-9分", "缺失则本项0分"),
            _catalog_row("基本面评分", "营业收入同比", "≥10%:+6；<0:-6", "连续可比报告期收入", "增长为正面，下降为负面", "+6至-6分", "缺失则本项0分"),
            _catalog_row("基本面评分", "资产负债率", "≥75%:-10；55%—75%:-4；≤35%:+4", "资产和负债", "在未做行业中性化前，过高偏负面", "+4至-10分", "缺失则本项0分"),
            _catalog_row("基本面评分", "经营现金流／净利润", "≥1:+7；<0:-9", "经营现金流和净利润", "现金覆盖越好越正面", "+7至-9分", "缺失则本项0分"),
            _catalog_row("基本面评分", "市盈率TTM", "≤0:-7；>80:-6；0—20:+3", "最新价格和每股收益／公开估值", "极高或亏损偏负面；较低仅小幅加分", "+3至-7分", "缺失则本项0分；可测指标少于2项时基本面不参与总分"),
            _catalog_row("基本面评分", "投入资本回报率ROIC", "NOPAT÷平均投入资本；≥15%:+6；8%—15%:+3；0—4%:-3；<0:-6", "营业利润、所得税、债务、权益、现金，优先连续两期", "衡量公司使用债权和股权资本创造经营回报的效率", "+6至-6分", "任一关键字段缺失则本项0分，不使用行业均值代填"),
            _catalog_row("基本面评分", "自由现金流FCF", "经营现金流-资本开支；FCF为正:+2；且FCF／营收≥10%:+5；FCF<0:-6", "经营现金流、资本开支、营业收入", "持续为正说明经营产生的现金在维持投资后仍有剩余", "+5至-6分", "经营现金流或资本开支缺失则本项0分"),
            _catalog_row("基本面评分", "毛利率／营业利润率趋势", "本期利润率-上期利润率；两项改善:+5；单项改善:+2；任一下降≥3个百分点:-5；两项小幅下降:-4", "连续两期收入、毛利和营业利润", "改善表示盈利质量增强，下降表示成本或竞争压力上升", "+5至-5分（两项合并为一组，避免重复计分）", "仅一项可计算时按该项；两项均缺失则0分"),
            _catalog_row("基本面评分", "估值历史分位", "当前正市盈率在本公司历史正市盈率样本中的百分位；≤20%:+4；≤40%:+2；≥75%:-3；≥90%:-5", "当前市盈率及同公司历史市盈率；A股至少60个日样本，美国SEC至少3个披露时点样本", "低分位仅表示相对自身较便宜，高分位表示估值容错较低", "+4至-5分", "样本不足、当前亏损或估值不可比时本项0分"),
        ]
    )

    rows.extend(
        [
            _catalog_row("宏观与市场", "基准趋势状态", "现价同时高于MA60和MA250且60日收益>0记60分；同时低于且收益<0记38分；其余50分", "沪深300／SPY／恒生指数日线", "市场越强，对个股时点越正面", "乘0.08或0.10，最终最多±2分", "基准缺失时按50分且降低数据完整度"),
            _catalog_row("宏观与市场", "基准60日年化波动", ">30%时市场分-5", "基准近60日日收益", "高波动偏负面", "-5分后再进入宏观权重", "缺失则不扣分"),
            _catalog_row("宏观与市场", "公开利率方向", "A股LPR／美股10年国债收益率下降:+3；上升:-3；持平0", "最近公开利率序列", "利率下降小幅正面，上升小幅负面", "+3／0／-3后进入宏观权重", "接口失败时本项0分并备注"),
        ]
    )

    rows.extend(
        [
            _catalog_row("持有期时点评分", "基础分", "每个可分析持有期从50分开始", "满足该周期最低历史行数", "中性起点", "50分", "历史行数不足则该周期不评分"),
            _catalog_row("持有期时点评分", "价格相对快速均线", "与快慢均线合并：两个信号同为正时趋势块+10，同为负时-10，否则0", "按周期设置的fast窗口", "站上均线是趋势确认的一部分", "不再独立±8；与下一项共用±10分", "窗口不足则该周期不可用"),
            _catalog_row("持有期时点评分", "快慢均线结构", "与价格位置合并：两个信号同为正时趋势块+10，同为负时-10，否则0", "按周期设置的fast／slow窗口", "多头结构是趋势确认的一部分", "不再独立±8；与上一项共用±10分", "窗口不足则该周期不可用"),
            _catalog_row("持有期时点评分", "同周期动量", "clip(Rh÷(2×近60日波动×√h),-1,1)×8", "当前与h日前收盘价、近60日日收益波动", "正收益为正面，但需按股票正常波动调整", "原始最多±8分；再乘历史验证系数1／0.5／0", "收益或波动不可得时该周期不可用"),
            _catalog_row("持有期时点评分", "相对基准收益", "clip((R股票,h-R基准,h)÷(2×正常波动),-1,1)×7", "股票和对应基准同周期价格", "跑赢基准为正面", "原始最多±7分；再乘历史验证系数1／0.5／0", "基准缺失时0分并降低完整度"),
            _catalog_row("持有期时点评分", "20日短期均值回归", "近10日收益相对自身近250日分布的z分数；z≥1扣分，z≤-1小幅加分", "至少120个交易日；只用于20日持有期", "极端追高偏负面，短期超跌小幅正面", "-5至+3分，再乘历史验证系数；其他周期0分", "样本不足或非20日周期时0分"),
            _catalog_row("持有期时点评分", "成交量比", "V20/V60仅结合价格方向解释，不直接计分", "近20日与60日平均成交量", "成交量本身没有固定涨跌方向", "0分；仅作量价背景", "成交量缺失时不展示确认信息"),
            _catalog_row("持有期时点评分", "52周高点位置", "P/rolling_max(P,252)-1", "至少252个交易日", "只表示所处位置，不重复代替趋势分", "0分；仅展示", "数据不足时不展示"),
            _catalog_row("持有期时点评分", "10日量价确认", "近10日涨跌与V10/V30组合为放量上涨、缩量上涨、放量下跌或中性", "至少30个交易日价格和成交量", "只作解释背景，因样本外表现不稳定不计分", "0分；仅展示", "价格或成交量缺失时不展示"),
            _catalog_row("持有期时点评分", "波动率历史分位", "20日年化波动率在自身近250日中的分位", "至少120个交易日", "高分位提示不确定性高，不直接预测涨跌", "0分；进入风险解释", "数据不足时不展示"),
            _catalog_row("持有期时点评分", "当前分数局部自校准", "在T日以前已知结果中，选取与当前原始分最接近的25%历史时点，计算同向命中率与符号调整后中位收益", "至少12个已完成的独立历史结果", "只能维持、减半或阻断方向信号，绝不反转失败信号", "通过×1并可形成方向；有限通过×0.5但只展示；未通过×0", "样本不足时不允许据此增强方向"),
            _catalog_row("持有期时点评分", "跨股票样本外认证", "同一周期须在开发股票、较晚时段和至少两批不同股票留出样本中保持正向后才允许输出方向", "多股票、跨阶段且与调参样本分离的历史检验", "只控制能否给方向，不把测试答案写回当次分数", "已认证才可输出方向；未认证×0", "最终封闭检验尚无周期稳定通过；当前全部只展示评分"),
            _catalog_row("持有期时点评分", "宏观修正", "clip((宏观分-50)×0.08或0.10,-2,2)", "宏观评分", "宏观分高于50为正面", "最多±2分", "宏观分缺失时0分"),
            _catalog_row("持有期时点评分", "基本面修正", "短期0；60／120／250日分别乘0.06／0.10／0.12并截断", "基本面评分", "基本面分高于50为正面；仅用于中长期", "最多±4分", "基本面不可用时0分"),
            _catalog_row("持有期时点评分", "历史相似周期修正", "继续展示相似样本分布，但不进入生产评分", "可得相似历史样本", "只作情景说明，不代表未来方向", "0分", "样本不足时明确显示，不强行预测"),
            _catalog_row("持有期时点评分", "最新资讯修正", "Σ(情绪×时效×相关度×来源权重)/Σ权重×8，并截断到±8", "至少2条有效公开资讯且可信度≥35", "正面资讯加分，负面资讯减分", "最多±8分；不修改原量化分", "无资讯或可信度不足时0分"),
        ]
    )

    rows.extend(
        [
            _catalog_row("最新资讯修正", "标题与摘要情绪", "(正面词命中数-负面词命中数)/总命中数", "公开资讯标题与摘要；中英文透明词典", "正值偏正面，负值偏负面", "进入资讯加权平均；自身范围-1至1", "没有情绪词时记0，不臆测方向"),
            _catalog_row("最新资讯修正", "个股相关度", "公司全称命中0.98；简称0.92；代码0.84—0.86；基础0.55", "标题、摘要、市场、代码和公司名", "相关度越高，资讯权重越高", "作为乘数0.55—0.98", "相关度<0.50的资讯剔除"),
            _catalog_row("最新资讯修正", "资讯时效", "≤1日=1.00；≤3日=0.85；≤7日=0.65；更久=0.35", "可解析的公开发布时间", "越新权重越高", "作为乘数0.35—1.00", "日期缺失时按本次检索窗口最旧档0.35"),
            _catalog_row("最新资讯修正", "来源权重", "交易所、监管、公司公告等高权威来源=1.20；其他公开来源=1.00", "来源名称", "高权威来源权重更高", "作为乘数1.00或1.20", "来源缺失按公开资讯1.00"),
            _catalog_row("最新资讯修正", "资讯可信度闸门", "min(90,15+min(条数×7,42)+min(来源数×5,20)+min(有日期条数×2,13))", "有效资讯数、独立来源数、可解析日期数", "越高表示资讯证据更完整", "至少2条且可信度≥35才允许修正", "未通过闸门时资讯修正固定为0"),
        ]
    )

    analog_factors = [
        ("5日收益", "P/P[-5]-1", "5%"),
        ("20日收益", "P/P[-20]-1", "10%"),
        ("60日收益", "P/P[-60]-1", "9%"),
        ("120日收益", "P/P[-120]-1", "4%"),
        ("20日年化波动", "std(r,20)×√252", "12%"),
        ("60日年化波动", "std(r,60)×√252", "8%"),
        ("60日下行波动", "std(r<0,60)×√252", "6%"),
        ("60日回撤位置", "P/rolling_max(P,60)-1", "8%"),
        ("120日回撤位置", "P/rolling_max(P,120)-1", "6%"),
        ("价格相对MA20", "P/MA20-1", "6%"),
        ("价格相对MA60", "P/MA60-1", "6%"),
        ("20日／60日成交量比", "V20/V60", "4%"),
        ("20日相对基准", "R股票,20-R基准,20", "6%"),
        ("60日相对基准", "R股票,60-R基准,60", "4%"),
        ("基准20日收益", "R基准,20", "3%"),
        ("基准20日年化波动", "std(r基准,20)×√252", "3%"),
    ]
    for factor, formula, weight in analog_factors:
        rows.append(
            _catalog_row(
                "相似周期距离",
                factor,
                "先按历史四分位距标准化；加权欧氏距离转相似度。" + formula,
                "当前时点与历史候选时点均可计算",
                "不是单调涨跌方向；数值越接近当前状态越相似",
                weight,
                "当前可用特征少于8项则不检索；其余可用权重重新归一，候选行缺失即剔除",
            )
        )

    rows.extend(
        [
            _catalog_row("数据完整度", "历史行数", "≥750／500／250／120／其他=40／34／26／16／8分", "股票日线", "数据越长，可信度越高", "最高40分", "上市不足五年时再设置总分上限"),
            _catalog_row("数据完整度", "行情新鲜度", "滞后≤5日:+20；≤12日:+12；更久:+3", "最后交易日期", "越新越好", "最高20分", "日期较旧时明确提示"),
            _catalog_row("数据完整度", "Beta可得性", "可得:+15；不可得:+6", "股票与基准对齐收益", "可得时证据更完整", "15或6分", "不可得不伪造，仅降低完整度"),
            _catalog_row("数据完整度", "基本面可得性", "个股可得:+18；不可得:+5；基金:+12", "公开财务／证券类型", "可得时证据更完整", "最高18分", "缺失时行情分析继续"),
            _catalog_row("数据完整度", "宏观可得性", "可得:+7；不可得:+2", "基准和公开宏观数据", "可得时证据更完整", "7或2分", "缺失时按中性"),
            _catalog_row("数据完整度", "极端日收益", "绝对日收益>70%的天数×2扣分，最多-10", "全部日收益", "异常越多，可信度越低", "0至-10分", "只提示复权／公司事件核对，不直接判涨跌"),
        ]
    )

    rows.extend(
        [
            _catalog_row("适配与仓位", "用户等级与股票等级差", "C等级-R等级≤-2不适配；=-1有限适配；其余适配", "用户风险等级、股票风险等级", "用户承受力高于股票风险时更适配", "门槛规则", "任一等级不可得时证据不足"),
            _catalog_row("适配与仓位", "单股风险预算", "C1—C5风险预算=0.25%／0.50%／1%／2%／3%", "用户等级", "等级越高允许承担的资产损失预算越高", "风险预算率", "不适配或证据不足时上限为0"),
            _catalog_row("适配与仓位", "历史压力损失", "max(同周期历史收益5%分位绝对值,8%)", "所选周期历史收益", "压力损失越大，可持仓比例越低", "仓位分母", "无周期时仓位为0"),
            _catalog_row("适配与仓位", "等级仓位上限", "C1—C5=3%／5%／10%／15%／20%；R5再×0.75", "用户等级和股票等级", "风险越高上限越低", "硬上限", "不适配或证据不足时上限为0"),
            _catalog_row("适配与仓位", "时点与有限适配乘数", "历史验证未通过或时点<45时上限=0；45—59上限×0.5；有限适配再×0.5", "所选周期时点评分、历史验证和适配结果", "信号不可靠、偏弱或适配有限时降低仓位", "0／0.5／1.0乘数", "方向证据不足或时点评分缺失时上限为0"),
        ]
    )

    rows.extend(
        [
            _catalog_row("卖出信号", "个人亏损边界", "min(问卷亏损上限, clip(年化波动×√(周期/252)×0.85,4%,20%))", "持仓成本、收益率、所选周期、波动率", "亏损超过边界为核心风险", "核心信号", "成本缺失时显示数据不足，不触发"),
            _catalog_row("卖出信号", "趋势破位", "连续2日收盘<MAfast且MAfast<MAslow", "至少20日日线", "触发为负面", "核心信号", "数据不足时不形成卖出判断"),
            _catalog_row("卖出信号", "盈利回撤保护", "浮盈达到启用条件后，从近期高点回撤达到动态阈值", "成本收益、近期高点、波动率", "触发为负面", "辅助信号", "成本缺失或未达到浮盈条件时不启用"),
            _catalog_row("卖出信号", "相对市场弱势", "R股票-R基准≤-clip(波动×√(周期/252)×0.5,3%,10%)", "股票和基准日线", "显著跑输为负面", "辅助信号", "基准缺失时数据不足"),
            _catalog_row("卖出信号", "基本面恶化", "基本面分≤35且有风险，或至少2项经营指标恶化", "最近公开财务", "触发为负面", "辅助信号", "财务缺失时不触发"),
            _catalog_row("卖出信号", "相似周期转弱", "可信度达标且上涨样本占比≤35%且收益中位数<0", "相似周期结果", "触发为负面", "辅助信号，不单独决定卖出", "可信度不足时不触发"),
        ]
    )

    rows.extend(
        [
            _catalog_row("加仓条件分", "原时点评分（含可用资讯）", "先要求持有期历史验证通过；再以所选周期原分为起点，资讯仅作有限修正", "当前完整分析和最新公开资讯", "验证通过后，分数越高越支持加仓", "起点0—100；<45硬限制，45—59条件限制", "无可用方向或周期时不允许形成加仓结论"),
            _catalog_row("加仓条件分", "个人适配", "适配:+8；有限适配:-8；不适配／证据不足:-30", "用户等级、股票等级和安全限制", "适配提高条件分", "+8／-8／-30分", "缺失按证据不足-30并形成硬限制"),
            _catalog_row("加仓条件分", "现有卖出状态", "继续持有:+8；警戒观察:-12；退出复核／分批减仓:-30", "卖出信号模块", "风险状态越严重越不支持加仓", "+8／-12／-30分", "证据不足不加分并加入条件限制"),
            _catalog_row("加仓条件分", "剩余风险预算", "计划加仓≤模型上限-当前市值且剩余额度>0:+8；否则-25", "当前持仓市值、模型仓位上限和计划金额", "不超预算为正面，超预算为硬限制", "+8或-25分", "任一金额不可得时不允许通过"),
            _catalog_row("加仓条件分", "数据完整度", "(数据完整度-50)×0.12", "主分析数据完整度0—100", "完整度越高小幅加分", "约-6至+6分", "<35形成硬限制；35—59降低可信度"),
            _catalog_row("加仓条件分", "硬限制数量", "-min(硬限制条数×8,24)", "适配、时点、预算、卖出状态等硬规则", "硬限制越多，条件分越低", "0至-24分；最终截断0—100", "无硬限制时0分"),
            _catalog_row("加仓条件分", "会话组合集中度", "已有≥2只持仓且加仓后本股占会话组合>50%时提示集中", "永久会话中有效投入本金", "集中度高为负面约束", "当前为条件限制，不直接进入数值分", "其他会话本金缺失时只按有效记录计算并提示口径"),
        ]
    )
    return rows


INVESTOR_POINT_MAPS: list[tuple[str, str, dict[str, int], int]] = [
    ("fund_source", "资金来源", {"闲置自有资金": 15, "未来有明确用途的资金": 8, "应急资金": 0, "借款／融资资金": 0}, 15),
    ("emergency_reserve", "应急储备", {"不足3个月": 0, "3—6个月": 8, "6个月以上": 15}, 15),
    ("earliest_need", "最早用款时间", {"1周内": 0, "1个月内": 3, "3个月内": 7, "1年内": 12, "3年内": 15, "没有明确时间": 15}, 15),
    ("loss_response", "亏损后的实际反应", {"立即全部卖出": 1, "大部分减仓": 4, "先复核原因再决定": 8, "继续按原计划持有": 11, "在条件允许时分批增加": 13}, 13),
    ("max_loss", "最大可承受损失", {"不超过5%": 0, "5%—10%": 4, "10%—20%": 8, "20%—30%": 12, "超过30%": 15}, 15),
    ("goal", "投资目标", {"保值为主": 0, "股息／稳健收益": 3, "长期增值": 6, "波段操作": 8, "短线交易": 10}, 10),
    ("income_stability", "收入稳定性", {"不稳定": 2, "较稳定": 7, "稳定": 10}, 10),
    ("experience", "投资经验", {"没有经验": 0, "不足1年": 2, "1—3年": 4, "3年以上": 5}, 5),
]


def investor_contribution_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, name, mapping, maximum in INVESTOR_POINT_MAPS:
        value = str(profile.get(key) or "未取得")
        default_value = "10%—20%" if key == "max_loss" else None
        points = mapping.get(value, mapping.get(default_value, 0) if default_value else 0)
        rows.append(
            {
                "模块": "用户风险分",
                "因子": name,
                "当前值": value,
                "本次贡献": float(points),
                "分值范围": f"0—{maximum}分" if name != "亏损后的实际反应" else "1—13分",
                "说明": "按问卷固定映射计分",
            }
        )
    return rows


def stock_risk_contribution_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    volatility = safe_float(metrics.get("annual_volatility"))
    max_drawdown_value = safe_float(metrics.get("max_drawdown"))
    downside = safe_float(metrics.get("downside_volatility"))
    beta_value = safe_float(metrics.get("beta"))
    beta_used = 1.0 if beta_value is None else max(0.0, beta_value)

    vol_part = np.clip((volatility - 0.12) / 0.58 * 35, 0, 35) if volatility is not None else 20.0
    drawdown_abs = abs(max_drawdown_value or 0.0)
    drawdown_part = np.clip((drawdown_abs - 0.12) / 0.68 * 35, 0, 35)
    downside_part = np.clip((downside - 0.08) / 0.52 * 20, 0, 20) if downside is not None else 10.0
    beta_part = np.clip((beta_used - 0.6) / 1.8 * 10, 0, 10)

    values = [
        ("近一年年化波动率", volatility, vol_part, "0—35分", "越高风险贡献越大"),
        ("最大回撤", max_drawdown_value, drawdown_part, "0—35分", "回撤越深风险贡献越大"),
        ("下行波动率", downside, downside_part, "0—20分", "负收益波动越高风险贡献越大"),
        ("Beta", beta_value, beta_part, "0—10分", "缺失时按Beta=1.0参与并降低完整度"),
    ]
    return [
        {
            "模块": "股票风险分",
            "因子": name,
            "当前值": value,
            "本次贡献": float(points),
            "分值范围": score_range,
            "说明": note,
        }
        for name, value, points, score_range, note in values
    ]


def timing_contribution_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = dict(analysis.get("selected_horizon") or {})
    metrics = dict(analysis.get("metrics") or {})
    if not selected:
        return []
    close = metrics.get("close")
    benchmark_close = metrics.get("benchmark_close")
    if not isinstance(close, pd.Series) or close.empty:
        return []

    fast = int(selected.get("fast") or 20)
    slow = int(selected.get("slow") or 60)
    days = int(selected.get("days") or 20)
    fast_ma = float(close.rolling(fast).mean().iloc[-1])
    slow_ma = float(close.rolling(slow).mean().iloc[-1])
    latest = float(close.iloc[-1])
    stock_return = safe_float(selected.get("stock_return"))
    benchmark_return = safe_float(selected.get("benchmark_return"))
    volume_ratio = safe_float(metrics.get("volume_ratio"))
    contributions = dict(selected.get("factor_contributions") or {})
    context_factors = dict(selected.get("context_factors") or {})
    reliability = float(selected.get("technical_reliability_multiplier") or 0.0)
    validation = dict(selected.get("signal_validation") or {})

    rows: list[dict[str, Any]] = [
        {"模块": "当前时点评分", "因子": "基础分", "当前值": "中性起点", "本次贡献": 50.0, "分值范围": "固定50分", "说明": selected.get("name", "所选周期")},
        {
            "模块": "当前时点评分",
            "因子": "趋势结构（合并去重）",
            "当前值": f"价格/MA{fast}={(latest / fast_ma - 1):.3%}；MA{fast}/MA{slow}={(fast_ma / slow_ma - 1):.3%}",
            "本次贡献": float(contributions.get("trend") or 0.0),
            "分值范围": "原始±10分，再乘历史验证系数",
            "说明": f"两个均线信号合并为一个趋势块，避免重复计分；验证系数{reliability:.2f}",
        },
    ]

    rows.append(
        {
            "模块": "当前时点评分",
            "因子": f"近{days}日波动调整动量",
            "当前值": stock_return,
            "本次贡献": float(contributions.get("momentum") or 0.0),
            "分值范围": "原始最多±8分，再乘历史验证系数",
            "说明": f"收益先除以该股票近期正常波动，再截断；验证系数{reliability:.2f}",
        }
    )

    excess = stock_return - benchmark_return if stock_return is not None and benchmark_return is not None else None
    rows.extend(
        [
            {
                "模块": "当前时点评分",
                "因子": "波动调整相对强弱",
                "当前值": excess,
                "本次贡献": float(contributions.get("relative_strength") or 0.0),
                "分值范围": "原始最多±7分，再乘历史验证系数",
                "说明": f"相对基准收益按个股正常波动标准化；验证系数{reliability:.2f}",
            },
            {
                "模块": "当前时点评分",
                "因子": "20日短期均值回归",
                "当前值": context_factors.get("ret10_zscore"),
                "本次贡献": float(contributions.get("short_reversal") or 0.0),
                "分值范围": "20日周期-5至+3分，其他周期0分；再乘历史验证系数",
                "说明": "仅保留开发样本与未见A股样本方向一致的20日均值回归修正",
            },
            {
                "模块": "量价背景",
                "因子": "20日／60日成交量比",
                "当前值": volume_ratio,
                "本次贡献": 0.0,
                "分值范围": "0分",
                "说明": "成交量没有固定涨跌方向，V6.7仅作价格信号确认，不再独立加减分",
            },
            {
                "模块": "状态背景",
                "因子": "52周高点位置",
                "当前值": context_factors.get("position_52_week"),
                "本次贡献": 0.0,
                "分值范围": "0分",
                "说明": "与趋势因子高度重复且未见样本不稳定，只展示",
            },
            {
                "模块": "量价背景",
                "因子": "10日量价确认",
                "当前值": context_factors.get("price_volume_confirmation"),
                "本次贡献": 0.0,
                "分值范围": "0分",
                "说明": "样本外表现不稳定，只用于说明量价背景",
            },
            {
                "模块": "风险背景",
                "因子": "20日波动率历史分位",
                "当前值": context_factors.get("volatility_percentile"),
                "本次贡献": 0.0,
                "分值范围": "0分",
                "说明": "用于提示当前波动是否处于自身高位，不伪装成方向因子",
            },
            {
                "模块": "当前时点评分",
                "因子": "宏观与市场修正",
                "当前值": safe_float(getattr(analysis.get("macro"), "score", None)),
                "本次贡献": float(contributions.get("macro") or 0.0),
                "分值范围": "最多±2分",
                "说明": "只作小幅环境修正",
            },
            {
                "模块": "当前时点评分",
                "因子": "基本面修正",
                "当前值": safe_float(getattr(analysis.get("fundamental"), "score", None)),
                "本次贡献": float(contributions.get("fundamental") or 0.0),
                "分值范围": "短期0分；中长期最多±4分",
                "说明": "缺少行业横向标准化，因此降低权重并限制上限",
            },
            {
                "模块": "历史情景",
                "因子": "历史相似周期",
                "当前值": selected.get("analog_status") or "未使用",
                "本次贡献": 0.0,
                "分值范围": "0分",
                "说明": "V6.7保留情景展示，但不进入生产方向分",
            },
            {
                "模块": "可靠性闸门",
                "因子": "历史验证结果",
                "当前值": f"{validation.get('status', '未验证')}；可信度{selected.get('signal_confidence', 0)}/100",
                "本次贡献": 0.0,
                "分值范围": "技术贡献乘1.0／0.5／0",
                "说明": (
                    str(validation.get("reason") or "历史验证不足时不形成方向判断")
                    + (
                        f"；当前相近分数{int(validation.get('local_band_count') or 0)}个，"
                        f"同向命中{float(validation.get('local_direction_hit_rate')):.3%}"
                        if validation.get("local_direction_hit_rate") is not None
                        else "；当前相近分数样本不足"
                    )
                ),
            },
        ]
    )

    raw_total = float(sum(float(item["本次贡献"]) for item in rows))
    displayed_score = safe_float(selected.get("score"))
    rows.append({"模块": "当前时点评分", "因子": "量化评分合计", "当前值": raw_total, "本次贡献": float((displayed_score if displayed_score is not None else np.clip(raw_total, 0, 100))), "分值范围": "0—100分", "说明": "对上述贡献求和、四舍五入并截断；此行为汇总，不重复计入图表"})

    news = dict(analysis.get("news_analysis") or {})
    rows.append({"模块": "资讯辅助修正", "因子": "最新公开资讯", "当前值": news.get("direction", "无有效资讯"), "本次贡献": float(safe_float(news.get("score_adjustment")) or 0.0), "分值范围": "最多±8分", "说明": "仅形成资讯结合后观察分，不改写原量化评分"})
    return rows


VALIDATION_FACTOR_META: dict[str, dict[str, str]] = {
    "trend_block": {"因子": "合并趋势结构", "当前权重": "原始±10分，受验证闸门控制"},
    "momentum": {"因子": "波动调整动量", "当前权重": "原始最多±8分，受验证闸门控制"},
    "relative_strength": {"因子": "波动调整相对强弱", "当前权重": "原始最多±7分，受验证闸门控制"},
    "short_reversal": {"因子": "20日短期均值回归", "当前权重": "20日期-5至+3分；其他期限0分"},
    "volume_context": {"因子": "量价背景", "当前权重": "0分，仅展示"},
}


def _rank_ic(x: pd.Series, y: pd.Series) -> float | None:
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 8 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return None
    value = pair["x"].rank(method="average").corr(pair["y"].rank(method="average"))
    return float(value) if value is not None and isfinite(float(value)) else None


def _validation_decision(
    samples: int,
    ic: float | None,
    hit_rate: float | None,
    spread: float | None,
    first_half_ic: float | None,
    second_half_ic: float | None,
) -> tuple[str, str]:
    if samples < 30 or ic is None or hit_rate is None or spread is None:
        return "降低权重", "有效独立时点不足30个；证据不足时不应维持完整预测权重。"
    stable_negative = (
        first_half_ic is not None
        and second_half_ic is not None
        and first_half_ic < 0
        and second_half_ic < 0
    )
    stable_positive = (
        first_half_ic is not None
        and second_half_ic is not None
        and first_half_ic > 0
        and second_half_ic > 0
    )
    if ic <= -0.05 and hit_rate < 0.47 and spread < 0 and stable_negative:
        return "候选删除", "方向长期与设计相反，且前后半段均为负；需在多股票样本复核后才可真正删除。"
    if ic < 0.03 or hit_rate < 0.50 or spread <= 0 or not stable_positive:
        return "降低权重", "预测力较弱、方向不稳定或分组收益未拉开；建议只作辅助信号。"
    return "保留", "IC、方向命中和分组收益均为正，且前后半段方向一致。"


def walk_forward_factor_validation(
    bundle: PriceBundle,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    selected = dict(analysis.get("selected_horizon") or {})
    if not selected:
        return {
            "available": False,
            "rows": [],
            "summary": "没有可用持有周期，无法设置对应的未来收益验证期限。",
            "method": "",
        }

    stock = bundle.stock.copy()
    stock["日期"] = pd.to_datetime(stock["日期"], errors="coerce")
    stock["收盘"] = pd.to_numeric(stock["收盘"], errors="coerce")
    stock["成交量"] = pd.to_numeric(stock.get("成交量"), errors="coerce")
    stock = stock.dropna(subset=["日期", "收盘"]).drop_duplicates("日期").sort_values("日期")
    close = stock.set_index("日期")["收盘"]
    volume = stock.set_index("日期")["成交量"]
    days = int(selected.get("days") or 20)

    benchmark = bundle.benchmark.copy() if bundle.benchmark is not None else pd.DataFrame()
    if not benchmark.empty:
        benchmark["日期"] = pd.to_datetime(benchmark["日期"], errors="coerce")
        benchmark["收盘"] = pd.to_numeric(benchmark["收盘"], errors="coerce")
        benchmark = benchmark.dropna(subset=["日期", "收盘"]).drop_duplicates("日期").sort_values("日期")
        benchmark_close = benchmark.set_index("日期")["收盘"]
    else:
        benchmark_close = pd.Series(dtype="float64")

    technical = _technical_timing_frame(close, volume, benchmark_close, selected)
    features = pd.DataFrame(index=technical.index)
    features["trend_block"] = technical["trend_points"] / 10.0
    features["momentum"] = technical["momentum_points"] / 8.0
    features["relative_strength"] = technical["relative_points"] / 7.0
    features["short_reversal"] = technical["short_reversal_adjustment"] / 5.0
    features["volume_context"] = technical["volume_context"]
    target = technical["future_target"]
    stride = max(5, int(round(days / 4)))
    eligible_index = technical[["raw_score", "future_target"]].dropna().iloc[::stride].index
    validation_frame = features.reindex(eligible_index)
    target = target.reindex(eligible_index)

    rows: list[dict[str, Any]] = []
    for key, meta in VALIDATION_FACTOR_META.items():
        pair = pd.concat([validation_frame[key].rename("signal"), target.rename("future")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        samples = len(pair)
        ic = _rank_ic(pair["signal"], pair["future"]) if samples else None
        non_neutral = pair[pair["signal"].abs() > 1e-10]
        hit_rate = (
            float(((non_neutral["signal"] > 0) == (non_neutral["future"] > 0)).mean())
            if len(non_neutral) >= 8
            else None
        )
        spread = None
        if samples >= 10 and pair["signal"].nunique() >= 5:
            lower = pair["signal"].quantile(0.20)
            upper = pair["signal"].quantile(0.80)
            bottom = pair.loc[pair["signal"] <= lower, "future"]
            top = pair.loc[pair["signal"] >= upper, "future"]
            if not bottom.empty and not top.empty:
                spread = float(top.median() - bottom.median())
        midpoint = samples // 2
        first_ic = _rank_ic(pair["signal"].iloc[:midpoint], pair["future"].iloc[:midpoint]) if midpoint >= 8 else None
        second_ic = _rank_ic(pair["signal"].iloc[midpoint:], pair["future"].iloc[midpoint:]) if samples - midpoint >= 8 else None
        decision, reason = _validation_decision(samples, ic, hit_rate, spread, first_ic, second_ic)
        if key == "volume_context":
            decision = "降低权重"
            reason = "成交量本身没有固定涨跌方向；V6.7生产权重固定为0，只保留量价背景展示。"
        if key == "short_reversal" and days != 20:
            decision = "降低权重"
            reason = "开发样本只支持20日期限；当前期限生产权重为0。"
        rows.append(
            {
                "key": key,
                "因子": meta["因子"],
                "当前权重": meta["当前权重"],
                "有效验证时点": samples,
                "秩相关IC": ic,
                "方向命中率": hit_rate,
                "高低组收益差": spread,
                "前半段IC": first_ic,
                "后半段IC": second_ic,
                "建议": decision,
                "依据": reason,
            }
        )

    # Redundant signals are not deleted automatically. The weaker member of a
    # highly correlated pair is only down-weighted pending broader validation.
    correlation_frame = validation_frame[list(VALIDATION_FACTOR_META)].rank(method="average").corr()
    row_map = {item["key"]: item for item in rows}
    for i, left in enumerate(correlation_frame.columns):
        for right in correlation_frame.columns[i + 1 :]:
            correlation = safe_float(correlation_frame.loc[left, right])
            if correlation is None or abs(correlation) < 0.85:
                continue
            left_ic = abs(safe_float(row_map[left].get("秩相关IC")) or 0.0)
            right_ic = abs(safe_float(row_map[right].get("秩相关IC")) or 0.0)
            weaker, stronger = (left, right) if left_ic <= right_ic else (right, left)
            if row_map[weaker]["建议"] == "保留":
                row_map[weaker]["建议"] = "降低权重"
                row_map[weaker]["依据"] = (
                    f"与“{row_map[stronger]['因子']}”的历史秩相关为{correlation:.3f}，"
                    "信息重复度较高；保留解释作用但建议降低权重。"
                )

    production_gate = _validate_timing_signal(technical, days)
    gate_status = str(production_gate.get("status") or "未通过")
    rows.append(
        {
            "key": "production_validation_gate",
            "因子": "组合信号验证闸门",
            "当前权重": "通过×1；有限通过×0.5；未通过×0",
            "有效验证时点": int(production_gate.get("samples") or 0),
            "秩相关IC": production_gate.get("rank_ic"),
            "方向命中率": production_gate.get("hit_rate"),
            "高低组收益差": production_gate.get("high_low_spread"),
            "前半段IC": production_gate.get("first_half_ic"),
            "后半段IC": production_gate.get("second_half_ic"),
            "建议": "保留" if gate_status == "通过" else "降低权重" if gate_status == "有限通过" else "候选删除",
            "依据": str(production_gate.get("reason") or "历史验证不足时不形成方向判断。"),
        }
    )
    local_hit = safe_float(production_gate.get("local_direction_hit_rate"))
    local_median = safe_float(production_gate.get("local_signed_median_return"))
    rows.append(
        {
            "key": "local_score_calibration",
            "因子": "当前分数局部自校准",
            "当前权重": "只可降低或阻断方向，不反转信号",
            "有效验证时点": int(production_gate.get("local_band_count") or 0),
            "秩相关IC": None,
            "方向命中率": local_hit,
            "高低组收益差": local_median,
            "前半段IC": None,
            "后半段IC": None,
            "建议": "候选删除" if production_gate.get("local_strongly_opposes") else "保留" if local_hit is not None and local_hit >= 0.52 and local_median is not None and local_median > 0 else "降低权重",
            "依据": (
                f"与当前原始分最接近的历史样本中，同向命中率{local_hit:.3%}，"
                f"符号调整后中位收益{local_median:.3%}。"
                if local_hit is not None and local_median is not None
                else "当前相近分数的已完成历史样本不足，不允许据此增强信号。"
            ),
        }
    )

    analog_backtest = dict((analysis.get("analog_forecast") or {}).get("backtest") or {})
    analog_cases = int(analog_backtest.get("cases") or 0)
    if analog_backtest.get("available"):
        accuracy = safe_float(analog_backtest.get("direction_accuracy"))
        rows.append(
            {
                "key": "historical_analog",
                "因子": "历史相似周期",
                "当前权重": "0分，仅作情景展示",
                "有效验证时点": analog_cases,
                "秩相关IC": None,
                "方向命中率": accuracy,
                "高低组收益差": None,
                "前半段IC": None,
                "后半段IC": None,
                "建议": "降低权重",
                "依据": "单只股票的相似样本少且稳定性不足；V6.7不计入生产评分，继续作为情景说明。",
            }
        )
    else:
        rows.append(
            {
                "key": "historical_analog",
                "因子": "历史相似周期",
                "当前权重": "0分，仅作情景展示",
                "有效验证时点": analog_cases,
                "秩相关IC": None,
                "方向命中率": None,
                "高低组收益差": None,
                "前半段IC": None,
                "后半段IC": None,
                "建议": "降低权重",
                "依据": str(analog_backtest.get("note") or "可回测时点不足；不允许相似周期影响评分。"),
            }
        )

    counts = {label: sum(1 for item in rows if item["建议"] == label) for label in ("保留", "降低权重", "候选删除")}
    summary = (
        f"本股票在“{selected.get('name')}”期限下：保留{counts['保留']}项，"
        f"降低权重{counts['降低权重']}项，候选删除{counts['候选删除']}项。"
    )
    return {
        "available": True,
        "selected_horizon": selected.get("name"),
        "forward_days": days,
        "stride": stride,
        "rows": rows,
        "counts": counts,
        "summary": summary,
        "method": (
            f"滚动验证以每个历史时点可见数据计算因子，用随后{days}个交易日收益作为结果；"
            f"每隔{stride}个交易日取一个时点以降低样本重叠。IC为因子排序与后续收益排序的相关系数。"
        ),
        "limitations": [
            "这是单只股票、单一当前期限的时间序列检验，不等于跨市场通用有效性。",
            "基本面、宏观和资讯缺少完整的历史时点快照，未用今天的数据回填过去，因此不参与本次历史保留／删除裁决。",
            "“候选删除”不会自动改写生产评分；需要在多股票、不同市场和样本外区间复核。",
        ],
    }


def build_factor_analysis(
    bundle: PriceBundle,
    analysis: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    fundamental = analysis.get("fundamental")
    fundamental_fields = getattr(fundamental, "fields", {}) if fundamental is not None else {}
    return {
        "catalog": factor_catalog(),
        "investor_contributions": investor_contribution_rows(profile),
        "stock_risk_contributions": stock_risk_contribution_rows(analysis.get("metrics") or {}),
        "timing_contributions": timing_contribution_rows(analysis),
        "fundamental_quality_contributions": quality_factor_contributions(dict(fundamental_fields or {}))["rows"],
        "historical_validation": walk_forward_factor_validation(bundle, analysis),
        "policy_note": (
            "风险承受、适配和仓位规则属于安全约束，不以短期收益预测结果决定删除；"
            "历史验证主要评估能够用日线无前视还原的量价因子。V6.7固定采用通过=1、有限通过=0.5、"
            "未通过=0的技术贡献闸门，并增加当前相近分数校准；相似周期和成交量不再独立计分。"
        ),
    }

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import json
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


NEWS_VERSION = 1
DEFAULT_WINDOW_DAYS = 7
EXTENDED_WINDOW_DAYS = 30
MIN_RECENT_ITEMS = 3
MAX_DISPLAY_ITEMS = 12
MAX_SCORE_ADJUSTMENT = 8

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
)

POSITIVE_WORDS = {
    "预增", "增长", "扭亏", "盈利", "超预期", "上调", "增持", "回购", "中标", "订单",
    "获批", "突破", "合作", "扩产", "分红", "创新高", "利好", "改善", "升级", "签约",
    "beat", "beats", "growth", "profit", "upgrade", "buyback", "approved", "approval",
    "contract", "partnership", "record high", "outperform", "raises guidance", "dividend",
}

NEGATIVE_WORDS = {
    "预亏", "亏损", "下滑", "下调", "减持", "立案", "调查", "处罚", "问询", "诉讼", "违约",
    "退市", "终止", "暴跌", "跌停", "风险", "裁员", "召回", "利空", "被查", "爆雷", "债务",
    "miss", "misses", "loss", "downgrade", "investigation", "probe", "lawsuit", "recall",
    "layoff", "default", "warning", "cuts guidance", "fraud", "delisting", "penalty",
}

CATEGORY_RULES = (
    ("监管与合规", ("监管", "证监", "立案", "调查", "处罚", "问询", "合规", "sec ", "probe", "investigation", "penalty")),
    ("业绩与财报", ("业绩", "财报", "营收", "净利润", "预增", "预亏", "盈利", "亏损", "earnings", "revenue", "profit", "guidance")),
    ("股东与资本运作", ("回购", "增持", "减持", "分红", "并购", "重组", "定增", "股权", "buyback", "dividend", "merger", "acquisition")),
    ("经营与重大合同", ("中标", "订单", "合同", "签约", "合作", "获批", "产品", "扩产", "contract", "partnership", "approval", "product")),
    ("诉讼与风险事件", ("诉讼", "违约", "召回", "裁员", "债务", "退市", "lawsuit", "default", "recall", "layoff", "delisting")),
    ("行业与宏观", ("行业", "政策", "利率", "通胀", "关税", "产业", "sector", "policy", "rates", "inflation", "tariff")),
    ("机构观点", ("评级", "目标价", "研报", "分析师", "上调", "下调", "rating", "analyst", "target price", "upgrade", "downgrade")),
)

HIGH_AUTHORITY_SOURCES = (
    "证券交易所", "上交所", "深交所", "北交所", "港交所", "证监会", "公司公告",
    "sec", "nasdaq", "nyse", "hkex", "sse", "szse",
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _strip_html(value: Any) -> str:
    text = unescape(str(value or ""))
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(value: Any, limit: int = 220) -> str:
    text = _strip_html(value)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _iso_text(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _short_company_name(name: str, code: str) -> str:
    text = re.sub(r"\s+", " ", str(name or "")).strip()
    suffixes = (
        "股份有限公司", "集团有限公司", "有限公司", "控股有限公司", "Company Limited",
        "Corporation", "Corp.", "Corp", "Incorporated", "Inc.", "Inc", "Limited", "Ltd.", "Ltd",
    )
    for suffix in suffixes:
        if text.lower().endswith(suffix.lower()) and len(text) > len(suffix) + 1:
            text = text[: -len(suffix)].strip(" -—,，")
            break
    return text or str(code)


def _yahoo_symbol(market: str, code: str) -> str:
    cleaned = str(code or "").strip().upper()
    if market == "美股":
        return cleaned
    digits = re.sub(r"\D", "", cleaned)
    if market == "港股":
        return f"{digits.zfill(4)}.HK"
    if not digits:
        return cleaned
    suffix = ".SS" if digits.startswith(("5", "6", "9")) else ".BJ" if digits.startswith(("4", "8")) else ".SZ"
    return f"{digits.zfill(6)}{suffix}"


def _search_queries(market: str, code: str, name: str, days: int) -> list[tuple[str, str, str]]:
    short_name = _short_company_name(name, code)
    if market == "美股":
        queries = [
            (f'"{short_name}" {code} stock when:{days}d', "en-US", "US:en"),
            (f'"{short_name}" {code} 股票 when:{days}d', "zh-CN", "CN:zh-Hans"),
        ]
    else:
        queries = [
            (f'"{short_name}" 股票 when:{days}d', "zh-CN", "CN:zh-Hans"),
            (f'"{short_name}" {re.sub(r"\D", "", code)} when:{days}d', "zh-CN", "CN:zh-Hans"),
        ]
    deduped: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for query in queries:
        if query[0] not in seen:
            deduped.append(query)
            seen.add(query[0])
    return deduped


def _request_rss(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=10) as response:
        return response.read()


def _parse_rss(content: bytes, provider: str) -> list[dict[str, Any]]:
    root = ET.fromstring(content)
    records: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        source_node = item.find("source")
        source = _strip_html(source_node.text if source_node is not None else "")
        title = _strip_html(item.findtext("title"))
        link = _strip_html(item.findtext("link"))
        description = _truncate(item.findtext("description"))
        published = _parse_datetime(item.findtext("pubDate"))
        if not title or not link:
            continue
        records.append(
            {
                "title": title,
                "summary": description,
                "source": source or provider,
                "published_at": _iso_text(published),
                "url": link,
                "provider": provider,
            }
        )
    return records


def _fetch_google_news(market: str, code: str, name: str, days: int) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    def fetch_one(settings: tuple[str, str, str]) -> tuple[list[dict[str, Any]], str | None]:
        query, language, edition = settings
        country = edition.split(":", 1)[0]
        url = "https://news.google.com/rss/search?" + urlencode(
            {"q": query, "hl": language, "gl": country, "ceid": edition}
        )
        try:
            return _parse_rss(_request_rss(url), "Google News RSS"), None
        except Exception as exc:
            return [], f"Google News RSS暂不可用：{type(exc).__name__}"

    queries = _search_queries(market, code, name, days)
    with ThreadPoolExecutor(max_workers=min(len(queries), 2) or 1) as executor:
        for fetched, warning in executor.map(fetch_one, queries):
            records.extend(fetched)
            if warning:
                warnings.append(warning)
    return records, list(dict.fromkeys(warnings))


def _fetch_bing_news(market: str, code: str, name: str, days: int) -> tuple[list[dict[str, Any]], list[str]]:
    short_name = _short_company_name(name, code)
    query = f'"{short_name}" {code} 股票' if market != "美股" else f'"{short_name}" {code} stock'
    url = "https://www.bing.com/news/search?" + urlencode(
        {"q": query, "format": "rss", "qft": f'interval="{days}"'}
    )
    try:
        return _parse_rss(_request_rss(url), "Bing News RSS"), []
    except Exception as exc:
        return [], [f"Bing News RSS暂不可用：{type(exc).__name__}"]


def _fetch_yahoo_news(market: str, code: str) -> tuple[list[dict[str, Any]], list[str]]:
    symbol = _yahoo_symbol(market, code)
    url = "https://query1.finance.yahoo.com/v1/finance/search?" + urlencode(
        {
            "q": symbol,
            "quotesCount": 1,
            "newsCount": 20,
            "enableFuzzyQuery": "false",
        }
    )
    try:
        payload = json.loads(_request_rss(url).decode("utf-8", errors="replace"))
        raw_items = payload.get("news") or []
    except Exception as exc:
        return [], [f"Yahoo Finance资讯暂不可用：{type(exc).__name__}"]

    records: list[dict[str, Any]] = []
    for raw in raw_items or []:
        raw_mapping = dict(raw) if isinstance(raw, Mapping) else {}
        content = raw_mapping.get("content")
        data = dict(content) if isinstance(content, Mapping) else raw_mapping
        provider_data = data.get("provider") if isinstance(data.get("provider"), Mapping) else {}
        canonical = data.get("canonicalUrl") if isinstance(data.get("canonicalUrl"), Mapping) else {}
        click_through = data.get("clickThroughUrl") if isinstance(data.get("clickThroughUrl"), Mapping) else {}
        title = _strip_html(data.get("title"))
        link = str(
            canonical.get("url")
            or click_through.get("url")
            or data.get("link")
            or raw_mapping.get("link")
            or ""
        ).strip()
        if not title or not link:
            continue
        published = _parse_datetime(
            data.get("pubDate") or data.get("displayTime") or data.get("providerPublishTime") or raw_mapping.get("providerPublishTime")
        )
        records.append(
            {
                "title": title,
                "summary": _truncate(data.get("summary") or data.get("description")),
                "source": _strip_html(provider_data.get("displayName") or data.get("publisher") or raw_mapping.get("publisher") or "Yahoo Finance"),
                "published_at": _iso_text(published),
                "url": link,
                "provider": "Yahoo Finance资讯",
            }
        )
    return records, []


def _normalised_title(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s*[-–—|]\s*[^-–—|]{1,35}$", "", text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _category(text: str) -> str:
    lowered = text.lower()
    for label, words in CATEGORY_RULES:
        if any(word in lowered for word in words):
            return label
    return "公司动态"


def _sentiment(text: str) -> tuple[str, float, list[str]]:
    lowered = text.lower()
    positive_hits = sorted({word for word in POSITIVE_WORDS if word in lowered})
    negative_hits = sorted({word for word in NEGATIVE_WORDS if word in lowered})
    total = len(positive_hits) + len(negative_hits)
    if total == 0:
        return "中性／不确定", 0.0, []
    score = (len(positive_hits) - len(negative_hits)) / total
    if score >= 0.25:
        label = "偏正面"
    elif score <= -0.25:
        label = "偏负面"
    else:
        label = "中性／不确定"
    return label, float(score), (positive_hits + negative_hits)[:6]


def _relevance(text: str, market: str, code: str, name: str) -> tuple[str, float]:
    lowered = text.lower()
    clean_code = re.sub(r"\W", "", str(code)).lower()
    full_name = str(name or "").lower().strip()
    short_name = _short_company_name(name, code).lower()
    score = 0.55
    if full_name and len(full_name) >= 3 and full_name in lowered:
        score = max(score, 0.98)
    if short_name and len(short_name) >= 2 and short_name in lowered:
        score = max(score, 0.92)
    if clean_code and len(clean_code) >= 4 and clean_code in re.sub(r"\W", "", lowered):
        score = max(score, 0.86)
    if market == "美股" and clean_code and re.search(rf"(?<![a-z]){re.escape(clean_code)}(?![a-z])", lowered):
        score = max(score, 0.84)
    label = "高" if score >= 0.85 else "中" if score >= 0.65 else "低"
    return label, score


def _source_weight(source: str) -> float:
    lowered = str(source or "").lower()
    return 1.20 if any(item in lowered for item in HIGH_AUTHORITY_SOURCES) else 1.0


def _enrich_items(
    raw_items: Iterable[Mapping[str, Any]],
    *,
    market: str,
    code: str,
    name: str,
    now: datetime,
    max_days: int,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    seen: set[str] = set()
    cutoff = now - timedelta(days=max_days + 1)
    for raw in raw_items:
        title = _strip_html(raw.get("title"))
        if not title:
            continue
        dedupe_key = _normalised_title(title)
        if len(dedupe_key) < 6 or dedupe_key in seen:
            continue
        published = _parse_datetime(raw.get("published_at"))
        if published is not None and published < cutoff:
            continue
        text = f"{title} {_strip_html(raw.get('summary'))}"
        relevance_label, relevance_score = _relevance(text, market, code, name)
        if relevance_score < 0.50:
            continue
        sentiment_label, sentiment_score, matched_terms = _sentiment(text)
        age_days = max((now - published).total_seconds() / 86400, 0.0) if published else float(max_days)
        recency_weight = 1.0 if age_days <= 1 else 0.85 if age_days <= 3 else 0.65 if age_days <= 7 else 0.35
        source = _strip_html(raw.get("source")) or str(raw.get("provider") or "公开资讯")
        enriched.append(
            {
                "title": title,
                "summary": _truncate(raw.get("summary")),
                "source": source,
                "published_at": _iso_text(published),
                "url": str(raw.get("url") or "").strip(),
                "provider": str(raw.get("provider") or "公开资讯"),
                "category": _category(text),
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "matched_terms": matched_terms,
                "relevance": relevance_label,
                "relevance_score": relevance_score,
                "age_days": age_days,
                "recency_weight": recency_weight,
                "source_weight": _source_weight(source),
            }
        )
        seen.add(dedupe_key)
    enriched.sort(key=lambda item: item.get("published_at") or "", reverse=True)
    return enriched[:MAX_DISPLAY_ITEMS]


def fetch_stock_news(market: str, code: str, name: str) -> dict[str, Any]:
    """Fetch best-effort public news without requiring a user's app credentials.

    A provider failure is returned as a warning and never blocks the existing
    price/fundamental analysis.
    """
    now = _now_utc()
    warnings: list[str] = []
    raw_items: list[dict[str, Any]] = []
    providers_attempted = ["Google News RSS", "Yahoo Finance资讯"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        google_future = executor.submit(_fetch_google_news, market, code, name, EXTENDED_WINDOW_DAYS)
        yahoo_future = executor.submit(_fetch_yahoo_news, market, code)
        google_items, google_warnings = google_future.result()
        yahoo_items, yahoo_warnings = yahoo_future.result()
    raw_items.extend(google_items)
    raw_items.extend(yahoo_items)
    warnings.extend(google_warnings)
    warnings.extend(yahoo_warnings)

    enriched = _enrich_items(
        raw_items,
        market=market,
        code=code,
        name=name,
        now=now,
        max_days=EXTENDED_WINDOW_DAYS,
    )
    recent = [item for item in enriched if float(item.get("age_days") or 999.0) <= DEFAULT_WINDOW_DAYS]
    window_days = DEFAULT_WINDOW_DAYS
    if len(recent) >= MIN_RECENT_ITEMS:
        selected = recent[:MAX_DISPLAY_ITEMS]
    else:
        selected = enriched[:MAX_DISPLAY_ITEMS]
        window_days = EXTENDED_WINDOW_DAYS

    if len(selected) < MIN_RECENT_ITEMS:
        providers_attempted.append("Bing News RSS")
        bing_items, bing_warnings = _fetch_bing_news(market, code, name, EXTENDED_WINDOW_DAYS)
        warnings.extend(bing_warnings)
        selected = _enrich_items(
            [*raw_items, *bing_items],
            market=market,
            code=code,
            name=name,
            now=now,
            max_days=EXTENDED_WINDOW_DAYS,
        )[:MAX_DISPLAY_ITEMS]
        window_days = EXTENDED_WINDOW_DAYS

    used = list(dict.fromkeys(str(item.get("provider") or "") for item in selected if item.get("provider")))
    return {
        "version": NEWS_VERSION,
        "market": market,
        "code": code,
        "name": name,
        "fetched_at": _iso_text(now),
        "window_days": window_days,
        "providers_attempted": providers_attempted,
        "providers_used": used,
        "items": selected,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _score_label(score: int) -> str:
    return "条件较积极" if score >= 70 else "中性偏积极" if score >= 60 else "中性观察" if score >= 45 else "偏弱／暂缓"


def assess_news(news_payload: Mapping[str, Any], base_score: Any) -> dict[str, Any]:
    items = [dict(item) for item in news_payload.get("items") or [] if isinstance(item, Mapping)]
    try:
        parsed_base = int(round(float(base_score)))
    except (TypeError, ValueError):
        parsed_base = None

    weighted_sum = 0.0
    total_weight = 0.0
    for item in items:
        weight = (
            float(item.get("recency_weight") or 0.0)
            * float(item.get("relevance_score") or 0.0)
            * float(item.get("source_weight") or 1.0)
        )
        weighted_sum += float(item.get("sentiment_score") or 0.0) * weight
        total_weight += weight
    net_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    source_count = len({str(item.get("source") or "") for item in items if item.get("source")})
    dated_count = sum(bool(item.get("published_at")) for item in items)
    confidence_score = min(90, 15 + min(len(items) * 7, 42) + min(source_count * 5, 20) + min(dated_count * 2, 13))
    if len(items) < 2:
        confidence_score = min(confidence_score, 30)
    confidence_label = "较高" if confidence_score >= 70 else "中等" if confidence_score >= 45 else "较低"

    usable = len(items) >= 2 and confidence_score >= 35
    adjustment = int(round(max(-MAX_SCORE_ADJUSTMENT, min(net_score * MAX_SCORE_ADJUSTMENT, MAX_SCORE_ADJUSTMENT)))) if usable else 0
    combined_score = max(0, min((parsed_base if parsed_base is not None else 50) + adjustment, 100)) if parsed_base is not None else None
    if net_score >= 0.18:
        direction = "偏正面"
    elif net_score <= -0.18:
        direction = "偏负面"
    else:
        direction = "中性／存在分歧"

    counts = {
        "positive": sum(item.get("sentiment") == "偏正面" for item in items),
        "negative": sum(item.get("sentiment") == "偏负面" for item in items),
        "neutral": sum(item.get("sentiment") == "中性／不确定" for item in items),
    }
    if not items:
        summary = "未检索到足够的可核验公开资讯，资讯因子不参与本次判断。"
    elif not usable:
        summary = "已展示检索到的参考资讯，但有效样本不足，资讯因子未修改原有量化评分。"
    elif adjustment == 0:
        summary = "近期资讯整体中性或正负影响相互抵消，原有量化评分保持不变。"
    else:
        summary = f"近期资讯整体{direction}，对原有量化时点评分作{adjustment:+d}分的有限修正。"

    return {
        **dict(news_payload),
        "available": bool(items),
        "usable_for_score": usable,
        "valid_count": len(items),
        "source_count": source_count,
        "counts": counts,
        "direction": direction,
        "net_sentiment_score": net_score,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "score_adjustment": adjustment,
        "maximum_adjustment": MAX_SCORE_ADJUSTMENT,
        "base_score": parsed_base,
        "base_label": _score_label(parsed_base) if parsed_base is not None else "无基础评分",
        "combined_score": combined_score,
        "combined_label": _score_label(combined_score) if combined_score is not None else "无法形成综合评分",
        "summary": summary,
        "method_note": "标题和公开摘要采用透明关键词、相关度、来源和时效权重分析；资讯最多修正±8分，不是涨跌概率。",
    }

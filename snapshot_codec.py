from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any, Mapping

import numpy as np
import pandas as pd

from agent_core import EvidenceSnapshot, PriceBundle


SCHEMA_VERSION = 1


def _clean_float(value: float) -> float | None:
    parsed = float(value)
    return parsed if isfinite(parsed) else None


def encode_value(value: Any) -> Any:
    """Convert the model result into JSON-safe values without losing table types."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return _clean_float(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return {"__type__": "timestamp", "value": value.isoformat()}
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"__type__": "date", "value": value.isoformat()}
    if value is pd.NA:
        return None
    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "dataframe",
            "columns": [encode_value(item) for item in value.columns.tolist()],
            "index": [encode_value(item) for item in value.index.tolist()],
            "index_name": encode_value(value.index.name),
            "data": [[encode_value(item) for item in row] for row in value.itertuples(index=False, name=None)],
        }
    if isinstance(value, pd.Series):
        return {
            "__type__": "series",
            "name": encode_value(value.name),
            "index": [encode_value(item) for item in value.index.tolist()],
            "index_name": encode_value(value.index.name),
            "data": [encode_value(item) for item in value.tolist()],
        }
    if isinstance(value, np.ndarray):
        return {"__type__": "ndarray", "data": encode_value(value.tolist())}
    if isinstance(value, PriceBundle):
        return {
            "__type__": "PriceBundle",
            "fields": {item.name: encode_value(getattr(value, item.name)) for item in fields(value)},
        }
    if isinstance(value, EvidenceSnapshot):
        return {
            "__type__": "EvidenceSnapshot",
            "fields": {item.name: encode_value(getattr(value, item.name)) for item in fields(value)},
        }
    if is_dataclass(value):
        return {
            "__type__": value.__class__.__name__,
            "fields": {item.name: encode_value(getattr(value, item.name)) for item in fields(value)},
        }
    if isinstance(value, Mapping):
        return {str(key): encode_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "data": [encode_value(item) for item in value]}
    if isinstance(value, (list, set)):
        return [encode_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__type__")
    if not kind:
        return {key: decode_value(item) for key, item in value.items()}
    if kind in {"timestamp", "datetime"}:
        return pd.Timestamp(value["value"])
    if kind == "date":
        return date.fromisoformat(value["value"])
    if kind == "dataframe":
        frame = pd.DataFrame(
            [[decode_value(item) for item in row] for row in value.get("data", [])],
            columns=[decode_value(item) for item in value.get("columns", [])],
        )
        index = [decode_value(item) for item in value.get("index", [])]
        if len(index) == len(frame):
            frame.index = pd.Index(index, name=decode_value(value.get("index_name")))
        return frame
    if kind == "series":
        data = [decode_value(item) for item in value.get("data", [])]
        index = [decode_value(item) for item in value.get("index", [])]
        return pd.Series(
            data,
            index=pd.Index(index, name=decode_value(value.get("index_name"))),
            name=decode_value(value.get("name")),
        )
    if kind == "ndarray":
        return np.asarray(decode_value(value.get("data", [])))
    if kind == "tuple":
        return tuple(decode_value(item) for item in value.get("data", []))
    decoded_fields = {key: decode_value(item) for key, item in value.get("fields", {}).items()}
    if kind == "PriceBundle":
        return PriceBundle(**decoded_fields)
    if kind == "EvidenceSnapshot":
        return EvidenceSnapshot(**decoded_fields)
    return decoded_fields


def build_analysis_snapshot(
    *,
    bundle: PriceBundle,
    analysis: Mapping[str, Any],
    profile: Mapping[str, Any],
    holding_state: str,
    holding_method: str,
    holding_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle": encode_value(bundle),
        "analysis": encode_value(dict(analysis)),
        "profile": encode_value(dict(profile)),
        "holding_state": str(holding_state),
        "holding_method": str(holding_method),
        "holding_snapshot": encode_value(dict(holding_snapshot)) if holding_snapshot else None,
    }


def restore_analysis_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    if int(payload.get("schema_version") or 0) != SCHEMA_VERSION:
        raise ValueError("该历史分析使用了不受支持的数据版本，请重新分析该股票。")
    bundle = decode_value(payload.get("bundle"))
    analysis = decode_value(payload.get("analysis"))
    profile = decode_value(payload.get("profile"))
    if not isinstance(bundle, PriceBundle) or not isinstance(analysis, dict) or not isinstance(profile, dict):
        raise ValueError("历史分析快照不完整，请重新分析该股票。")
    return {
        "bundle": bundle,
        "analysis": analysis,
        "profile": profile,
        "holding_state": str(payload.get("holding_state") or "尚未持有"),
        "holding_method": str(payload.get("holding_method") or "按持股数量填写"),
        "holding_snapshot": decode_value(payload.get("holding_snapshot")),
    }

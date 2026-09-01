from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from session_memory import session_key

try:
    from supabase import Client, create_client
except ImportError:  # Allows the setup page to explain a missing dependency.
    Client = Any  # type: ignore[assignment,misc]
    create_client = None


class CloudStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class CloudConfig:
    url: str
    publishable_key: str

    @classmethod
    def from_mapping(cls, secrets: Mapping[str, Any]) -> "CloudConfig":
        section = secrets.get("supabase") if hasattr(secrets, "get") else None
        nested = section if isinstance(section, Mapping) else {}
        url = str(secrets.get("SUPABASE_URL") or nested.get("url") or "").strip()
        key = str(
            secrets.get("SUPABASE_PUBLISHABLE_KEY")
            or secrets.get("SUPABASE_ANON_KEY")
            or nested.get("publishable_key")
            or nested.get("anon_key")
            or ""
        ).strip()
        if not url or not key:
            raise CloudStoreError("尚未配置 SUPABASE_URL 和 SUPABASE_PUBLISHABLE_KEY。")
        if not url.startswith("https://"):
            raise CloudStoreError("SUPABASE_URL 格式不正确，应以 https:// 开头。")
        return cls(url=url, publishable_key=key)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _response_data(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, Mapping):
        data = response.get("data")
    if not data:
        return []
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, Mapping):
        return [dict(data)]
    return []


def _auth_payload(response: Any) -> dict[str, Any]:
    user = getattr(response, "user", None)
    session = getattr(response, "session", None)
    if user is None and isinstance(response, Mapping):
        user = response.get("user")
        session = response.get("session")
    user_id = getattr(user, "id", None) if user is not None else None
    email = getattr(user, "email", None) if user is not None else None
    if isinstance(user, Mapping):
        user_id = user.get("id")
        email = user.get("email")
    access_token = getattr(session, "access_token", None) if session is not None else None
    refresh_token = getattr(session, "refresh_token", None) if session is not None else None
    if isinstance(session, Mapping):
        access_token = session.get("access_token")
        refresh_token = session.get("refresh_token")
    return {
        "user_id": str(user_id or ""),
        "email": str(email or ""),
        "access_token": str(access_token or ""),
        "refresh_token": str(refresh_token or ""),
        "email_confirmation_required": bool(user_id and not access_token),
    }


class CloudStore:
    """Small, user-scoped Supabase data layer used by the Streamlit app."""

    def __init__(self, config: CloudConfig, client: Client | None = None):
        if client is None:
            if create_client is None:
                raise CloudStoreError("尚未安装 supabase，请重新部署并确认 requirements.txt 已上传。")
            client = create_client(config.url, config.publishable_key)
        self.client = client

    def sign_up(self, email: str, password: str) -> dict[str, Any]:
        return _auth_payload(self.client.auth.sign_up({"email": email, "password": password}))

    def verify_signup_otp(self, email: str, token: str) -> dict[str, Any]:
        """Verify the email code sent by Supabase and return the new session."""
        return _auth_payload(
            self.client.auth.verify_otp(
                {
                    "email": email,
                    "token": token,
                    "type": "email",
                }
            )
        )

    def resend_signup_otp(self, email: str) -> None:
        """Resend the confirmation code for an existing unverified signup."""
        self.client.auth.resend(
            {
                "type": "signup",
                "email": email,
            }
        )

    def sign_in(self, email: str, password: str) -> dict[str, Any]:
        return _auth_payload(self.client.auth.sign_in_with_password({"email": email, "password": password}))

    def restore_auth_session(self, access_token: str, refresh_token: str) -> dict[str, Any]:
        return _auth_payload(self.client.auth.set_session(access_token, refresh_token))

    def sign_out(self) -> None:
        self.client.auth.sign_out()

    def load_profile(self, user_id: str) -> dict[str, Any] | None:
        response = (
            self.client.table("risk_profiles")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = _response_data(response)
        return rows[0] if rows else None

    def save_profile(self, user_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "profile_data": dict(record.get("profile_data") or {}),
            "risk_score": int(record.get("risk_score") or 0),
            "risk_level": str(record.get("risk_level") or ""),
            "version": int(record.get("version") or 1),
            "completed_at": record.get("completed_at") or _utc_now(),
            "updated_at": _utc_now(),
        }
        response = self.client.table("risk_profiles").upsert(payload, on_conflict="user_id").execute()
        rows = _response_data(response)
        return rows[0] if rows else payload

    def load_draft(self, user_id: str) -> dict[str, Any] | None:
        response = (
            self.client.table("risk_drafts")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = _response_data(response)
        return rows[0] if rows else None

    def save_draft(self, user_id: str, answers: Mapping[str, Any], current_index: int) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "answers": dict(answers),
            "current_index": int(current_index),
            "updated_at": _utc_now(),
        }
        response = self.client.table("risk_drafts").upsert(payload, on_conflict="user_id").execute()
        rows = _response_data(response)
        return rows[0] if rows else payload

    def delete_draft(self, user_id: str) -> None:
        self.client.table("risk_drafts").delete().eq("user_id", user_id).execute()

    @staticmethod
    def row_to_session(row: Mapping[str, Any]) -> dict[str, Any]:
        market = str(row.get("market") or "")
        code = str(row.get("stock_code") or row.get("code") or "")
        return {
            "key": session_key(market, code),
            "db_id": str(row.get("id") or ""),
            "market": market,
            "code": code,
            "name": str(row.get("stock_name") or row.get("name") or code),
            "created_at": row.get("created_at") or "",
            "updated_at": row.get("updated_at") or "",
            "principal_rmb": float(row.get("principal_rmb") or 0.0),
            "principal_source": str(row.get("principal_source") or "尚未记录实际投入"),
            "total_shares": float(row["total_shares"]) if row.get("total_shares") is not None else None,
            "shares_complete": bool(row.get("shares_complete")),
            "messages": list(row.get("messages") or []),
            "recorded_event_ids": list(row.get("recorded_event_ids") or []),
            "latest_summary": dict(row.get("latest_summary") or {}),
        }

    def load_sessions(self, user_id: str) -> dict[str, dict[str, Any]]:
        response = (
            self.client.table("stock_sessions")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        rows = _response_data(response)
        return {item["key"]: item for item in (self.row_to_session(row) for row in rows)}

    def get_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        response = (
            self.client.table("stock_sessions")
            .select("*")
            .eq("user_id", user_id)
            .eq("id", session_id)
            .limit(1)
            .execute()
        )
        rows = _response_data(response)
        return self.row_to_session(rows[0]) if rows else None

    def upsert_stock_session(
        self,
        user_id: str,
        record: Mapping[str, Any],
        *,
        include_position: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "user_id": user_id,
            "market": str(record.get("market") or ""),
            "stock_code": str(record.get("code") or "").upper(),
            "stock_name": str(record.get("name") or record.get("code") or ""),
            "messages": list(record.get("messages") or []),
            "recorded_event_ids": list(record.get("recorded_event_ids") or []),
            "latest_summary": dict(record.get("latest_summary") or {}),
            "updated_at": _utc_now(),
        }
        if include_position:
            payload.update(
                {
                    "principal_rmb": float(record.get("principal_rmb") or 0.0),
                    "principal_source": str(record.get("principal_source") or "尚未记录实际投入"),
                    "total_shares": record.get("total_shares"),
                    "shares_complete": bool(record.get("shares_complete")),
                }
            )
        response = (
            self.client.table("stock_sessions")
            .upsert(payload, on_conflict="user_id,market,stock_code")
            .execute()
        )
        rows = _response_data(response)
        if rows:
            return self.row_to_session(rows[0])
        sessions = self.load_sessions(user_id)
        key = session_key(payload["market"], payload["stock_code"])
        if key not in sessions:
            raise CloudStoreError("股票会话保存后未能重新读取。")
        return sessions[key]

    def update_stock_session(self, user_id: str, session_id: str, **fields_to_update: Any) -> dict[str, Any]:
        allowed = {
            "stock_name",
            "principal_rmb",
            "principal_source",
            "total_shares",
            "shares_complete",
            "messages",
            "recorded_event_ids",
            "latest_summary",
        }
        payload = {key: value for key, value in fields_to_update.items() if key in allowed}
        payload["updated_at"] = _utc_now()
        response = (
            self.client.table("stock_sessions")
            .update(payload)
            .eq("user_id", user_id)
            .eq("id", session_id)
            .execute()
        )
        rows = _response_data(response)
        if rows:
            return self.row_to_session(rows[0])
        refreshed = self.get_session(user_id, session_id)
        if refreshed is None:
            raise CloudStoreError("股票会话不存在或无权访问。")
        return refreshed

    def delete_stock_session(self, user_id: str, session_id: str) -> None:
        (
            self.client.table("stock_sessions")
            .delete()
            .eq("user_id", user_id)
            .eq("id", session_id)
            .execute()
        )

    def save_snapshot(
        self,
        user_id: str,
        session_id: str,
        event_id: str,
        summary: Mapping[str, Any],
        snapshot_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "stock_session_id": session_id,
            "event_id": event_id,
            "summary": dict(summary),
            "snapshot_data": dict(snapshot_data),
        }
        response = (
            self.client.table("analysis_snapshots")
            .upsert(payload, on_conflict="user_id,event_id")
            .execute()
        )
        rows = _response_data(response)
        return rows[0] if rows else payload

    def list_snapshots(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        response = (
            self.client.table("analysis_snapshots")
            .select("id,event_id,summary,created_at")
            .eq("user_id", user_id)
            .eq("stock_session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return _response_data(response)

    def load_snapshot(self, user_id: str, snapshot_id: str) -> dict[str, Any] | None:
        response = (
            self.client.table("analysis_snapshots")
            .select("*")
            .eq("user_id", user_id)
            .eq("id", snapshot_id)
            .limit(1)
            .execute()
        )
        rows = _response_data(response)
        return rows[0] if rows else None

    def insert_transaction(self, user_id: str, session_id: str, transaction: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "stock_session_id": session_id,
            "transaction_type": str(transaction.get("transaction_type") or "add"),
            "input_method": str(transaction.get("input_method") or "amount"),
            "trade_date": str(transaction.get("trade_date")),
            "shares": transaction.get("shares"),
            "price_native": transaction.get("price_native"),
            "currency": str(transaction.get("currency") or "人民币元"),
            "fx_rate": transaction.get("fx_rate"),
            "fees_rmb": float(transaction.get("fees_rmb") or 0.0),
            "principal_rmb": float(transaction.get("principal_rmb") or 0.0),
            "note": str(transaction.get("note") or ""),
        }
        response = self.client.table("position_transactions").insert(payload).execute()
        rows = _response_data(response)
        return rows[0] if rows else payload

    def list_transactions(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        response = (
            self.client.table("position_transactions")
            .select("*")
            .eq("user_id", user_id)
            .eq("stock_session_id", session_id)
            .order("trade_date", desc=True)
            .order("created_at", desc=True)
            .execute()
        )
        return _response_data(response)

    def delete_transaction(self, user_id: str, transaction_id: str) -> None:
        (
            self.client.table("position_transactions")
            .delete()
            .eq("user_id", user_id)
            .eq("id", transaction_id)
            .execute()
        )

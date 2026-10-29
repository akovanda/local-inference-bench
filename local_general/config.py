from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from urllib.parse import urlsplit


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _read_secret(env_name: str, file_env_name: str, *, required: bool) -> str | None:
    value = os.getenv(env_name, "").strip()
    if not value:
        file_name = os.getenv(file_env_name, "").strip()
        if file_name:
            try:
                value = Path(file_name).read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise ValueError(f"could not read {file_env_name}") from exc
    if required and not value:
        raise ValueError(f"set {env_name} or {file_env_name}")
    return value or None


@dataclass(frozen=True, slots=True)
class Settings:
    api_token: str = field(repr=False)
    upstream_base_url: str = "http://127.0.0.1:18082/v1"
    upstream_model: str = "lfm2.5-24b-a2b-qad-q4_0"
    upstream_api_token: str | None = field(default=None, repr=False)
    canonical_model: str = "local-general"
    enable_legacy_gpt_oss_alias: bool = False
    context_window: int = 49_152
    default_max_tokens: int = 8_192
    max_output_tokens: int = 8_192
    max_active: int = 1
    max_pending: int = 16
    queue_timeout_seconds: float = 120.0
    upstream_connect_timeout_seconds: float = 5.0
    upstream_read_timeout_seconds: float = 900.0
    max_request_bytes: int = 8 * 1024 * 1024
    max_sse_event_bytes: int = 1024 * 1024
    docs_html: Path | None = None
    bind_host: str = "127.0.0.1"
    bind_port: int = 11_441

    def __post_init__(self) -> None:
        if not self.api_token.strip():
            raise ValueError("api_token must not be empty")
        split = urlsplit(self.upstream_base_url)
        if (
            split.scheme not in {"http", "https"}
            or not split.hostname
            or split.username
            or split.query
            or split.fragment
        ):
            raise ValueError(
                "upstream_base_url must be an HTTP(S) URL without user info, query, or fragment"
            )
        if not self.upstream_model.strip() or not self.canonical_model.strip():
            raise ValueError("model names must not be empty")
        for name in (
            "context_window",
            "default_max_tokens",
            "max_output_tokens",
            "max_active",
            "max_pending",
            "max_request_bytes",
            "max_sse_event_bytes",
            "bind_port",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.default_max_tokens > self.max_output_tokens:
            raise ValueError("default_max_tokens must not exceed max_output_tokens")
        if self.max_output_tokens > self.context_window:
            raise ValueError("max_output_tokens must not exceed context_window")
        for name in (
            "queue_timeout_seconds",
            "upstream_connect_timeout_seconds",
            "upstream_read_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.docs_html is not None and not self.docs_html.is_file():
            raise ValueError("docs_html must name a readable file")
        if not 1 <= self.bind_port <= 65_535:
            raise ValueError("bind_port must be between 1 and 65535")

    @classmethod
    def from_env(cls) -> "Settings":
        docs_value = os.getenv("LOCAL_GENERAL_DOCS_HTML", "").strip()
        return cls(
            api_token=_read_secret(
                "LOCAL_GENERAL_API_TOKEN",
                "LOCAL_GENERAL_API_TOKEN_FILE",
                required=True,
            )
            or "",
            upstream_base_url=os.getenv(
                "LOCAL_GENERAL_UPSTREAM_BASE_URL", "http://127.0.0.1:18082/v1"
            ).rstrip("/"),
            upstream_model=os.getenv(
                "LOCAL_GENERAL_UPSTREAM_MODEL", "lfm2.5-24b-a2b-qad-q4_0"
            ),
            upstream_api_token=_read_secret(
                "LOCAL_GENERAL_UPSTREAM_API_TOKEN",
                "LOCAL_GENERAL_UPSTREAM_API_TOKEN_FILE",
                required=False,
            ),
            canonical_model=os.getenv("LOCAL_GENERAL_CANONICAL_MODEL", "local-general"),
            enable_legacy_gpt_oss_alias=_env_bool(
                "LOCAL_GENERAL_ENABLE_LEGACY_GPT_OSS_ALIAS", False
            ),
            context_window=int(os.getenv("LOCAL_GENERAL_CONTEXT_WINDOW", "49152")),
            default_max_tokens=int(
                os.getenv("LOCAL_GENERAL_DEFAULT_MAX_TOKENS", "8192")
            ),
            max_output_tokens=int(os.getenv("LOCAL_GENERAL_MAX_OUTPUT_TOKENS", "8192")),
            max_active=int(os.getenv("LOCAL_GENERAL_MAX_ACTIVE", "1")),
            max_pending=int(os.getenv("LOCAL_GENERAL_MAX_PENDING", "16")),
            queue_timeout_seconds=float(
                os.getenv("LOCAL_GENERAL_QUEUE_TIMEOUT_SECONDS", "120")
            ),
            upstream_connect_timeout_seconds=float(
                os.getenv("LOCAL_GENERAL_UPSTREAM_CONNECT_TIMEOUT_SECONDS", "5")
            ),
            upstream_read_timeout_seconds=float(
                os.getenv("LOCAL_GENERAL_UPSTREAM_READ_TIMEOUT_SECONDS", "900")
            ),
            max_request_bytes=int(
                os.getenv("LOCAL_GENERAL_MAX_REQUEST_BYTES", str(8 * 1024 * 1024))
            ),
            max_sse_event_bytes=int(
                os.getenv("LOCAL_GENERAL_MAX_SSE_EVENT_BYTES", str(1024 * 1024))
            ),
            docs_html=Path(docs_value).expanduser() if docs_value else None,
            bind_host=os.getenv("LOCAL_GENERAL_BIND_HOST", "127.0.0.1"),
            bind_port=int(os.getenv("LOCAL_GENERAL_BIND_PORT", "11441")),
        )

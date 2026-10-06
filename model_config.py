"""Cấu hình model offline/API; không gọi mạng khi đọc cấu hình."""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


class ModelConfigurationError(ValueError):
    """Lỗi cấu hình phải sửa trước khi chạy, không tự chuyển sang mock."""


def redact_secrets(value, secrets=()):
    # NOTE API 1: lọc cả lỗi SDK và trace trước khi in/xuất JSON. Không đưa
    # model repr hay toàn bộ biến môi trường vào báo cáo để tránh lộ API key.
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"\b(?:sk-[A-Za-z0-9_-]+|AIza[A-Za-z0-9_-]+)", "[REDACTED]", value)
        return value
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if str(k).lower() in {"api_key", "authorization", "x-api-key"}
                    else redact_secrets(v, secrets)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(v, secrets) for v in value]
    return value


def classify_model_error(exc: Exception) -> str:
    """Phân nhóm lỗi model; lỗi nghiệp vụ mock được ghi riêng trong tool trace."""
    # Một số adapter bọc lỗi HTTP trong ModelError: xem cả cause có cấu trúc,
    # không dò chuỗi message chứa thông tin xác thực để đoán mã lỗi.
    causes, current = [], exc
    while current is not None and current not in causes and len(causes) < 8:
        causes.append(current)
        current = current.__cause__
    name = " ".join(type(error).__name__.lower() for error in causes)
    status = next((getattr(error, "status_code", None) or getattr(error, "code", None)
                   for error in causes if getattr(error, "status_code", None) or getattr(error, "code", None)), None)
    if status in {401, 403} or any(s in name for s in ("authentication", "permissiondenied", "unauthenticated")):
        return "API_AUTH_ERROR"
    # NOTE API 12: HTTP 429 có cả lỗi hết credit/quota và lỗi gọi quá nhanh.
    # Đọc code/type có cấu trúc của SDK (kể cả adapter bọc lỗi), không dò message.
    # Retry không khôi phục credit; cần phân biệt để log/báo cáo đúng nguyên nhân.
    quota_codes = {"credit_balance_exhausted", "insufficient_quota", "organization_spend_limit_exceeded",
                   "project_spend_limit_exceeded", "organization_usage_limit_exceeded"}
    for error in causes:
        body = getattr(error, "body", None)
        detail = body.get("error", body) if isinstance(body, dict) else {}
        detail = detail if isinstance(detail, dict) else {}
        codes = [getattr(error, "code", None), getattr(error, "type", None), detail.get("code"), detail.get("type")]
        if any(isinstance(code, str) and code in quota_codes for code in codes):
            return "API_QUOTA_EXHAUSTED"
    if status == 429 or any(s in name for s in ("ratelimit", "resourceexhausted")):
        return "API_RATE_LIMIT"
    if "timeout" in name or "deadlineexceeded" in name:
        return "API_TIMEOUT"
    if any(s in name for s in ("connection", "connecterror", "network")):
        return "API_CONNECTION_ERROR"
    if isinstance(exc, NotImplementedError):
        return "MODEL_CAPABILITY_ERROR"
    if any(s in name for s in ("validation", "outputparser", "structuredoutput")):
        return "MODEL_SCHEMA_ERROR"
    if status in {400, 404, 422} or any(s in name for s in ("badrequest", "invalidrequest", "modelnotfound")):
        return "API_REQUEST_ERROR"
    if isinstance(status, int) and status >= 500:
        return "API_SERVICE_ERROR"
    if any(s in name for s in ("modelcalllimit", "modellimit", "graphrecursion", "stopagent")):
        return "AGENT_LIMIT"
    return "MODEL_EXECUTION_ERROR"


@dataclass(frozen=True)
class ModelConfig:
    mode: str = "offline"
    provider: str = "mock"
    model: str = "offline-policy-v1"
    api_key: str = field(default="", repr=False, compare=False)
    base_url: str | None = None
    plan_method: str = "function_calling"
    timeout_seconds: float = 30.0
    max_retries: int = 0
    max_output_tokens: int = 1024
    temperature: float | None = None

    def __post_init__(self):
        if self.mode not in {"offline", "api"}:
            raise ModelConfigurationError("Mode phải là offline hoặc api.")
        if self.mode == "offline":
            return
        if self.provider not in {"openai", "openai-compatible", "google-genai"}:
            raise ModelConfigurationError("Provider phải là openai, openai-compatible hoặc google-genai.")
        if not self.model.strip():
            raise ModelConfigurationError("Thiếu LLM_MODEL; điền tên model API mà tài khoản của bạn được dùng.")
        if not self.api_key.strip() or self.api_key.lower().startswith(("your_", "replace_", "dien_")):
            raise ModelConfigurationError("Thiếu API key thật. Điền LLM_API_KEY hoặc biến key của provider trong .env.")
        if self.plan_method not in {"function_calling", "json_schema"}:
            raise ModelConfigurationError("LLM_PLAN_METHOD phải là function_calling hoặc json_schema.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ModelConfigurationError("LLM_TIMEOUT_SECONDS phải là số hữu hạn > 0.")
        if not 0 <= self.max_retries <= 2 or self.max_output_tokens <= 0:
            raise ModelConfigurationError("LLM_MAX_RETRIES phải từ 0 đến 2; LLM_MAX_OUTPUT_TOKENS phải > 0.")
        if self.temperature is not None and (not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2):
            raise ModelConfigurationError("LLM_TEMPERATURE phải nằm trong [0, 2], hoặc để trống theo mặc định model.")
        if self.provider == "openai-compatible" and not self.base_url:
            raise ModelConfigurationError("Provider openai-compatible yêu cầu LLM_BASE_URL, thường có hậu tố /v1.")
        if self.provider == "google-genai" and self.base_url:
            raise ModelConfigurationError("LLM_BASE_URL chỉ dùng cho provider OpenAI-compatible.")
        if self.base_url:
            url = urlsplit(self.base_url)
            if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                    or url.query or url.fragment):
                raise ModelConfigurationError("LLM_BASE_URL phải là URL HTTP(S), không chứa key, user/password hoặc query.")
            if self.provider == "openai" and url.hostname != "api.openai.com":
                raise ModelConfigurationError("Endpoint tùy chỉnh: chọn provider openai-compatible.")

    def public_settings(self) -> dict:
        # NOTE API 2: chỉ xuất các trường được chọn, tuyệt đối không asdict(config).
        return {"mode": self.mode, "provider": self.provider, "model": self.model,
                "base_url": self.base_url, "plan_method": self.plan_method,
                "timeout_seconds": self.timeout_seconds, "max_retries": self.max_retries,
                "max_output_tokens": self.max_output_tokens, "temperature": self.temperature}

    def redact(self, value):
        return redact_secrets(value, (self.api_key,))


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--mode", choices=["offline", "api"], default="offline")
    parser.add_argument("--provider", choices=["openai", "openai-compatible", "google-genai"])
    parser.add_argument("--model", help="Tên model chính xác do provider cung cấp")
    parser.add_argument("--base-url", help="Endpoint tương thích OpenAI, ví dụ https://host/v1")
    parser.add_argument("--plan-method", choices=["function_calling", "json_schema"])
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    parser.add_argument("--timeout", type=float, help="Giới hạn thời gian mỗi request API (giây)")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--check-config", action="store_true", help="Kiểm tra cấu hình và khởi tạo SDK, không gọi API")


def load_model_config(args=None, *, mode="offline", env_file=None) -> ModelConfig:
    selected_mode = getattr(args, "mode", mode)
    # NOTE API 3: default offline dù .env có key; không đọc key/nhập SDK ở nhánh này.
    if selected_mode == "offline":
        return ModelConfig()
    path = Path(env_file or getattr(args, "env_file", Path(__file__).with_name(".env")))
    file_values = {}
    if path.is_file():
        try:
            from dotenv import dotenv_values
        except ImportError as exc:
            raise ModelConfigurationError("Cài python-dotenv bằng requirements.txt để đọc .env.") from exc
        file_values = dotenv_values(path, encoding="utf-8-sig", interpolate=False)
    values = {**file_values, **os.environ}  # môi trường terminal ưu tiên hơn file

    def value(arg_name, env_name, default=None):
        override = getattr(args, arg_name, None)
        return override if override is not None else (values.get(env_name) or default)

    provider = value("provider", "LLM_PROVIDER", "openai")
    key_name = "OPENAI_API_KEY" if provider in {"openai", "openai-compatible"} else "GOOGLE_API_KEY"
    key = values.get("LLM_API_KEY") or values.get(key_name) or (values.get("GEMINI_API_KEY") if provider == "google-genai" else "")
    try:
        temperature = values.get("LLM_TEMPERATURE")
        return ModelConfig(mode="api", provider=provider, model=value("model", "LLM_MODEL", ""),
                           api_key=key or "", base_url=value("base_url", "LLM_BASE_URL"),
                           plan_method=value("plan_method", "LLM_PLAN_METHOD", "function_calling"),
                           timeout_seconds=float(value("timeout", "LLM_TIMEOUT_SECONDS", "30")),
                           max_retries=int(values.get("LLM_MAX_RETRIES") or "0"),
                           max_output_tokens=int(value("max_output_tokens", "LLM_MAX_OUTPUT_TOKENS", "1024")),
                           temperature=float(temperature) if temperature else None)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ModelConfigurationError):
            raise
        raise ModelConfigurationError("Timeout, retries, token hoặc temperature trong cấu hình không đúng kiểu số.") from exc


def create_model(config: ModelConfig):
    """Khởi tạo SDK không gọi API. Model phải hỗ trợ tools và structured output."""
    if config.mode == "offline":
        return None
    try:
        if config.provider in {"openai", "openai-compatible"}:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=config.model, api_key=config.api_key,
                              base_url=config.base_url or "https://api.openai.com/v1",
                              timeout=config.timeout_seconds, max_retries=config.max_retries,
                              max_tokens=config.max_output_tokens, temperature=config.temperature,
                              # Giữ cơ chế proxy HTTP(S) của httpx nếu máy có cấu hình proxy.
                              use_responses_api=False, http_socket_options=())
        from langchain_google_genai import ChatGoogleGenerativeAI

        # NOTE API 13: Gemini chỉ đề xuất tool calls; LangChain và Harness mới
        # thực thi tool. Tắt AFC của Google SDK để không có bộ thực thi thứ hai.
        # Adapter dùng điểm tạo request của langchain-google-genai==4.4.0
        # (đã ghim trong requirements-api.txt), áp dụng cả planner và ReAct.
        class HarnessGoogleGenerativeAI(ChatGoogleGenerativeAI):
            def _prepare_request(self, *args, **kwargs):
                kwargs["automatic_function_calling"] = {"disable": True}
                return super()._prepare_request(*args, **kwargs)

        options = {"model": config.model, "google_api_key": config.api_key, "vertexai": False,
                   # Google SDK nhận tổng số attempts, OpenAI nhận số retry.
                   # +1 giữ ý nghĩa LLM_MAX_RETRIES thống nhất giữa hai provider.
                   "timeout": config.timeout_seconds, "max_retries": config.max_retries + 1,
                   "max_output_tokens": config.max_output_tokens, "temperature": config.temperature}
        return HarnessGoogleGenerativeAI(**options)
    except ImportError as exc:
        raise ModelConfigurationError("Thiếu SDK provider. Chạy python -m pip install -r requirements-api.txt.") from exc
    except Exception as exc:
        raise ModelConfigurationError(config.redact(f"Không khởi tạo được SDK ({type(exc).__name__}): {exc}")) from exc


def execution_options(config: ModelConfig | None = None, llm=None) -> dict:
    selected = config or ModelConfig()
    return {"fake_mode": selected.mode == "offline", "llm": llm, "model_config": selected}

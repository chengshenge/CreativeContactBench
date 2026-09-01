"""Backend-neutral model adapters with one request and zero retries."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import openai
from openai import OpenAI


OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class ModelResult:
    raw_response: str
    latency_seconds: float
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)


class AdapterExecutionError(RuntimeError):
    """A provider error carrying sanitized audit metadata."""

    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, image_path: Path, config: dict[str, Any]) -> ModelResult:
        """Return the exact text from one multimodal model request."""


def _data_url(image_path: Path) -> tuple[str, str]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported image MIME type: {mime_type}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}", mime_type


def _required_key(config: dict[str, Any]) -> tuple[str, str]:
    variable = config.get("api_key_env")
    if not isinstance(variable, str) or not variable:
        raise ValueError("endpoint.api_key_env must name an environment variable")
    value = os.environ.get(variable)
    if not value:
        raise AdapterExecutionError(
            f"Required environment variable is not set: {variable}",
            metadata={"error_category": "authentication", "model_requests": 0},
        )
    return variable, value


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


class OpenAIResponsesAdapter(ModelAdapter):
    def __init__(self, client_factory: Callable[..., Any] = OpenAI) -> None:
        self.client_factory = client_factory

    def generate(self, prompt: str, image_path: Path, config: dict[str, Any]) -> ModelResult:
        key_variable, api_key = _required_key(config)
        base_url = str(config.get("base_url", OFFICIAL_OPENAI_BASE_URL)).rstrip("/")
        if base_url != OFFICIAL_OPENAI_BASE_URL:
            raise ValueError(f"openai_responses permits only {OFFICIAL_OPENAI_BASE_URL}")
        if config.get("store", False) is not False:
            raise ValueError("Standard evaluation requires generation.store: false")
        image_url, mime_type = _data_url(image_path)
        model = str(config.get("model") or config["model_name"])
        request: dict[str, Any] = {
            "model": model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }],
            "max_output_tokens": int(config.get("max_output_tokens", 4000)),
            "store": False,
        }
        if config.get("reasoning_mode") or config.get("reasoning_effort"):
            request["reasoning"] = {
                "mode": str(config.get("reasoning_mode", "standard")),
                "effort": str(config.get("reasoning_effort", "low")),
            }
        request_metadata = {
            "provider": "openai",
            "endpoint": "responses",
            "api_key_env": key_variable,
            "image_mime_type": mime_type,
            "model_requests": 1,
            "automatic_retries": 0,
            "structured_outputs": False,
            "tools": False,
            "store": False,
        }
        client = self.client_factory(
            api_key=api_key,
            base_url=base_url,
            timeout=float(config.get("timeout", 180)),
            max_retries=0,
        )
        started = time.perf_counter()
        try:
            response = client.responses.create(**request)
        except openai.OpenAIError as exc:
            category = "api_error"
            if isinstance(exc, openai.AuthenticationError):
                category = "authentication"
            elif isinstance(exc, (openai.PermissionDeniedError, openai.NotFoundError)):
                category = "model_access"
            elif isinstance(exc, openai.RateLimitError):
                category = "rate_limit"
            elif isinstance(exc, openai.APITimeoutError):
                category = "timeout"
            elif isinstance(exc, openai.APIConnectionError):
                category = "connection"
            raise AdapterExecutionError(
                "OpenAI Responses request failed",
                metadata={
                    "provider": "openai",
                    "error_category": category,
                    "http_status": getattr(exc, "status_code", None),
                    "request_id": getattr(exc, "request_id", None),
                    "model_requests": 1,
                },
            ) from exc
        latency = time.perf_counter() - started
        raw = response.output_text
        if not isinstance(raw, str):
            raise AdapterExecutionError(
                "OpenAI response output_text is not a string",
                metadata={
                    "provider": "openai",
                    "error_category": "invalid_provider_response",
                    "request_id": getattr(response, "_request_id", None),
                    "model_requests": 1,
                },
            )
        usage = _field(response, "usage")
        input_details = _field(usage, "input_tokens_details")
        output_details = _field(usage, "output_tokens_details")
        return ModelResult(
            raw_response=raw,
            latency_seconds=latency,
            backend_metadata={
                "provider": "openai",
                "endpoint": "responses",
                "request_id": getattr(response, "_request_id", None),
                "status": _field(response, "status"),
                "input_tokens": _field(usage, "input_tokens"),
                "cached_input_tokens": _field(input_details, "cached_tokens"),
                "output_tokens": _field(usage, "output_tokens"),
                "reasoning_tokens": _field(output_details, "reasoning_tokens"),
                "total_tokens": _field(usage, "total_tokens"),
            },
            model_metadata={"requested_model": model, "returned_model": _field(response, "model")},
            request_metadata=request_metadata,
        )


class OpenAICompatibleAdapter(ModelAdapter):
    def generate(self, prompt: str, image_path: Path, config: dict[str, Any]) -> ModelResult:
        key_variable, api_key = _required_key(config)
        if config.get("store", False) is not False:
            raise ValueError("Standard evaluation requires generation.store: false")
        image_url, mime_type = _data_url(image_path)
        base_url = str(config["base_url"]).rstrip("/")
        model = str(config.get("model") or config["model_name"])
        body: dict[str, Any] = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            "temperature": config.get("temperature", 0),
            "max_tokens": int(config.get("max_tokens", 1600)),
        }
        for name in ("top_p", "top_k", "repetition_penalty", "presence_penalty", "seed"):
            if name in config:
                body[name] = config[name]
        request_metadata = {
            "provider": "openai-compatible",
            "endpoint": "chat/completions",
            "api_key_env": key_variable,
            "image_mime_type": mime_type,
            "model_requests": 1,
            "automatic_retries": 0,
            "structured_outputs": False,
            "store": False,
        }
        started = time.perf_counter()
        try:
            with httpx.Client(
                timeout=float(config.get("timeout", 300)),
                trust_env=False,
            ) as client:
                response = client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                payload = response.json()
            raw = payload["choices"][0]["message"]["content"]
            if not isinstance(raw, str):
                raise ValueError("Response content is not a string")
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            raise AdapterExecutionError(
                "OpenAI-compatible request failed",
                metadata={
                    "provider": "openai-compatible",
                    "error_category": "endpoint_error",
                    "model_requests": 1,
                },
            ) from exc
        usage = payload.get("usage") or {}
        return ModelResult(
            raw_response=raw,
            latency_seconds=time.perf_counter() - started,
            backend_metadata={
                "provider": "openai-compatible",
                "endpoint": "chat/completions",
                "http_status": response.status_code,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            model_metadata={"requested_model": model, "returned_model": payload.get("model")},
            request_metadata=request_metadata,
        )


class MockAdapter(ModelAdapter):
    def generate(self, prompt: str, image_path: Path, config: dict[str, Any]) -> ModelResult:
        started = time.perf_counter()
        digest = hashlib.sha256(
            prompt.encode("utf-8") + image_path.read_bytes() + config["model_name"].encode("utf-8")
        ).digest()
        ratings = {
            option: {
                "expected_task_effectiveness": digest[index * 3] % 5 + 1,
                "embodied_feasibility": digest[index * 3 + 1] % 5 + 1,
                "functional_creativity": digest[index * 3 + 2] % 5 + 1,
            }
            for index, option in enumerate("ABCD")
        }
        ranked = sorted("ABCD", key=lambda option: (-sum(ratings[option].values()), option))
        raw = json.dumps(
            {
                "ratings": ratings,
                "overall_ranking": [[option] for option in ranked],
                "confidence": digest[12] % 5 + 1,
            },
            separators=(",", ":"),
        )
        return ModelResult(
            raw_response=raw,
            latency_seconds=time.perf_counter() - started,
            backend_metadata={"provider": "mock"},
            model_metadata={"requested_model": config["model_name"], "returned_model": config["model_name"]},
            request_metadata={"model_requests": 0, "automatic_retries": 0},
        )


def adapter_for(name: str) -> ModelAdapter:
    adapters: dict[str, type[ModelAdapter]] = {
        "mock": MockAdapter,
        "openai_responses": OpenAIResponsesAdapter,
        "openai_compatible": OpenAICompatibleAdapter,
    }
    try:
        return adapters[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown backend: {name}") from exc

"""Mailbox and HTTP message transports for construction agents."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import mimetypes
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """A transport or model reply failed; preserve the original reply for review."""

    def __init__(self, message: str, *, raw_response: Any = None):
        super().__init__(message)
        self.raw_response = raw_response


class ResponsePending(ProviderError):
    """The request is persisted and can be resumed with the same request object."""

    def __init__(self, request_path: Path, response_path: Path):
        super().__init__(f"Waiting for agent response: {response_path}")
        self.request_path = request_path
        self.response_path = response_path


@dataclass(frozen=True)
class AgentRequest:
    request_id: str
    model: str
    messages: list[dict]
    response_schema: dict
    response_mode: str = "json_schema"
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        # Mailbox readers can run from another working directory.
        for message in payload["messages"]:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part.get("type") == "image_file":
                        part["path"] = str(Path(part["path"]).expanduser().resolve())
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> AgentRequest:
        return cls(**payload)


@dataclass(frozen=True)
class AgentResponse:
    request_id: str
    action: dict
    raw_response: dict
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class AgentProvider(Protocol):
    def complete(self, request: AgentRequest) -> AgentResponse:
        ...


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Publish a whole JSON file, including responses written by mailbox workers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(serialized)
            output.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProviderError(f"Invalid JSON in {path}: {error}", raw_response=text) from error
    if not isinstance(payload, dict):
        raise ProviderError(f"Expected a JSON object in {path}", raw_response=payload)
    return payload


class FileMailboxProvider:
    """One runner publishes requests; an external agent atomically publishes replies."""

    def __init__(self, mailbox_dir: str | Path, *, timeout_seconds: float = 0,
                 poll_seconds: float = 0.25):
        if timeout_seconds < 0 or poll_seconds <= 0:
            raise ValueError("timeout_seconds must be nonnegative and poll_seconds positive")
        self.mailbox_dir = Path(mailbox_dir).resolve()
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds

    def complete(self, request: AgentRequest) -> AgentResponse:
        if not request.request_id or Path(request.request_id).name != request.request_id:
            raise ValueError("request_id must be a nonempty filename component")
        request_path = self.mailbox_dir / "requests" / f"{request.request_id}.json"
        response_path = self.mailbox_dir / "responses" / f"{request.request_id}.json"
        payload = request.to_dict()
        if request_path.exists():
            if _read_json(request_path) != payload:
                raise ProviderError(f"request_id {request.request_id!r} already has a different request")
        elif response_path.exists():
            raise ProviderError(f"Response exists without its original request: {response_path}")
        else:
            atomic_write_json(request_path, payload)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if response_path.exists():
                raw = _read_json(response_path)
                if raw.get("request_id") != request.request_id:
                    raise ProviderError("Mailbox response request_id does not match", raw_response=raw)
                if raw.get("error") is not None:
                    raise ProviderError(f"External agent failed: {raw['error']}", raw_response=raw)
                if not isinstance(raw.get("action"), dict):
                    raise ProviderError("Mailbox response needs an action object", raw_response=raw)
                return AgentResponse(request.request_id, raw["action"], raw,
                                     {"transport": "mailbox", "response_path": str(response_path)})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResponsePending(request_path, response_path)
            time.sleep(min(self.poll_seconds, remaining))


def _http_messages(messages: list[dict]) -> list[dict]:
    messages = deepcopy(messages)
    for message in messages:
        if not isinstance(message.get("content"), list):
            continue
        for index, part in enumerate(message["content"]):
            if part.get("type") != "image_file":
                continue
            path = Path(part["path"])
            mime, _ = mimetypes.guess_type(path.name)
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                raise ProviderError(f"Unsupported local image format: {path}")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            message["content"][index] = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}",
                              "detail": part.get("detail", "auto")},
            }
    return messages


def _chat_payload(request: AgentRequest) -> dict:
    reserved = {"model", "messages", "response_format", "tools", "tool_choice",
                "parallel_tool_calls", "stream", "n"}
    if reserved.intersection(request.parameters):
        raise ValueError("parameters cannot replace model, messages, or the response contract")
    payload = {"model": request.model,
               "messages": _http_messages(request.to_dict()["messages"]),
               "stream": False, **request.parameters}
    if request.response_mode == "json_schema":
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "agent_action", "strict": True, "schema": request.response_schema}}
    elif request.response_mode == "tool_call":
        payload["tools"] = [{"type": "function", "function": {
            "name": "submit_action", "description": "Return the next map agent action.",
            "strict": True, "parameters": request.response_schema}}]
        payload["tool_choice"] = {"type": "function", "function": {"name": "submit_action"}}
        payload["parallel_tool_calls"] = False
    else:
        raise ValueError(f"Unknown response_mode: {request.response_mode}")
    return payload


def _chat_action(request: AgentRequest, raw: dict) -> dict:
    try:
        if raw.get("error"):
            raise ValueError(f"Provider error: {raw['error']}")
        choices = raw["choices"]
        if len(choices) != 1:
            raise ValueError("Expected exactly one completion choice")
        choice = choices[0]
        message = choice["message"]
        if message.get("refusal"):
            raise ValueError(f"Model declined the request: {message['refusal']}")
        if request.response_mode == "tool_call":
            if choice.get("finish_reason") != "tool_calls":
                raise ValueError(f"Expected a completed tool call; finish_reason={choice.get('finish_reason')!r}")
            calls = message.get("tool_calls", [])
            if len(calls) != 1 or calls[0]["function"]["name"] != "submit_action":
                raise ValueError("Expected one submit_action tool call")
            content = calls[0]["function"]["arguments"]
        else:
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"Incomplete structured reply; finish_reason={choice.get('finish_reason')!r}")
            content = message["content"]
        action = json.loads(content)
        if not isinstance(action, dict):
            raise ValueError("The action must be a JSON object")
        return action
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ProviderError(f"Invalid agent response: {error}", raw_response=raw) from error


class OpenAICompatibleProvider:
    """Nonstreaming chat-completions HTTP transport with caller-selected endpoint/model."""

    def __init__(self, base_url: str, *, api_key: str | None = None,
                 timeout_seconds: float = 120):
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def complete(self, request: AgentRequest) -> AgentResponse:
        payload = _chat_payload(request)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        wire_request = Request(self.url, data=json.dumps(payload, allow_nan=False).encode("utf-8"),
                               headers=headers, method="POST")
        try:
            with urlopen(wire_request, timeout=self.timeout_seconds) as reply:
                response_text = reply.read().decode("utf-8")
                metadata = {"transport": "chat_completions", "status": reply.status,
                            "request_id": reply.headers.get("x-request-id")}
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                raw_error = json.loads(body)
            except json.JSONDecodeError:
                raw_error = body
            raise ProviderError(f"Chat completions HTTP {error.code}: {body}",
                                raw_response=raw_error) from error
        except (URLError, TimeoutError) as error:
            raise ProviderError(f"Chat completions transport failed: {error}") from error
        try:
            raw = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise ProviderError("Chat completions returned invalid JSON", raw_response=response_text) from error
        if not isinstance(raw, dict):
            raise ProviderError("Chat completions response must be an object", raw_response=raw)
        action = _chat_action(request, raw)
        return AgentResponse(request.request_id, action, raw, metadata)

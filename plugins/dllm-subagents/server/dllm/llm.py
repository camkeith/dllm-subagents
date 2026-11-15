from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .config import Profile

Transport = Callable[[str, dict, bytes, float], Tuple[int, dict, bytes]]


class LLMError(Exception):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Completion:
    content: Optional[str]
    tool_calls: List[ToolCall]
    input_tokens: int
    output_tokens: int
    finish_reason: Optional[str]
    message: dict = field(default_factory=dict)


def urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> Tuple[int, dict, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        try:
            data = exc.read()
        except Exception:
            data = b""
        return exc.code, dict(exc.headers.items()) if exc.headers else {}, data


def _header(headers: dict, name: str) -> Optional[str]:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


class ChatClient:
    def __init__(
        self,
        profile: Profile,
        api_key: str,
        *,
        model: Optional[str] = None,
        transport: Transport = urllib_transport,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        request_timeout: float = 120.0,
    ):
        self.profile = profile
        self._api_key = api_key
        self.model = model or profile.model
        self._transport = transport
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._request_timeout = request_timeout

    def __repr__(self) -> str:
        return "ChatClient(profile=%r, model=%r)" % (self.profile.name, self.model)

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    def _body(self, messages: list, tools: list) -> bytes:
        p = self.profile
        body: dict = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            p.max_tokens_field: p.max_output_tokens,
        }
        body.update(p.extra_body)
        if p.temperature is not None:
            body["temperature"] = p.temperature
        return json.dumps(body).encode("utf-8")

    def _error_message(self, raw: bytes) -> str:
        text = raw.decode("utf-8", "replace")
        try:
            message = json.loads(text)["error"]["message"]
            if isinstance(message, str) and message:
                text = message
        except (ValueError, KeyError, TypeError):
            pass
        return self._scrub(text)[:500]

    def complete(self, messages: list, tools: list, deadline: Optional[float] = None) -> Completion:
        url = self.profile.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"}
        body = self._body(messages, tools)
        attempt = 0
        while True:
            retry_after: Optional[str] = None
            try:
                timeout = self._request_timeout
                if deadline is not None:
                    timeout = max(1.0, min(timeout, deadline - self._clock()))
                status, resp_headers, raw = self._transport(url, headers, body, timeout)
            except (OSError, ValueError) as exc:  # URLError and timeouts are OSErrors
                status, raw = None, b""
                failure = "Network error contacting provider: %s" % self._scrub(str(exc))
            else:
                if status == 200:
                    return self._parse(raw)
                retry_after = _header(resp_headers, "Retry-After")
                failure = "Provider returned HTTP %d: %s" % (status, self._error_message(raw))
                if status != 429 and status < 500:
                    raise LLMError(failure, status)
            if attempt >= self._max_retries:
                raise LLMError(failure, status)
            delay = min(2 ** attempt, 20)
            if retry_after is not None:
                try:
                    delay = max(0.0, float(retry_after))
                except ValueError:
                    pass
            if deadline is not None and self._clock() + delay >= deadline:
                raise LLMError("Stopped retrying: time limit reached (%s)" % failure, status)
            self._sleep(delay)
            attempt += 1

    def _parse(self, raw: bytes) -> Completion:
        try:
            data = json.loads(raw.decode("utf-8"))
            choice = data["choices"][0]
            msg = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError("Provider returned an unreadable response: %s" % self._scrub(
                raw.decode("utf-8", "replace")[:500]))
        usage = data.get("usage") or {}
        calls: List[ToolCall] = []
        for item in msg.get("tool_calls") or []:
            fn = item.get("function") or {}
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args)
            calls.append(ToolCall(item.get("id") or "call_" + uuid.uuid4().hex[:12],
                                  fn.get("name", ""), args))
        content = msg.get("content")
        message: dict = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in calls
            ]
        return Completion(
            content=content,
            tool_calls=calls,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=choice.get("finish_reason"),
            message=message,
        )

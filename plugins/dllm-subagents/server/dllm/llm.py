from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .config import Profile

Transport = Callable[[str, dict, bytes, float], Tuple[int, dict, bytes]]
MAX_RETRY_DELAY = 60


class LLMError(Exception):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class DeadlineReached(LLMError):
    """The run's time limit left no time to retry the model call."""


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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Hand 3xx responses back as errors instead of following them: urllib would resend the
    Authorization header (the key) to the new address."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)  # use for every request that carries a key


def urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> Tuple[int, dict, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        try:
            data = exc.read()
        except Exception:
            data = b""
        finally:
            exc.close()
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
        if not self._api_key:
            return text
        for form in (self._api_key, repr(self._api_key)[1:-1], json.dumps(self._api_key)[1:-1]):
            if form:
                text = text.replace(form, "***")
        return text

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

    def _redirect_message(self, status: int, headers: dict) -> str:
        location = _header(headers, "Location") or ""
        try:  # a relative Location stays on the base_url's host
            host = (urllib.parse.urlsplit(location).hostname
                    or urllib.parse.urlsplit(self.profile.base_url).hostname)
        except ValueError:
            host = None
        return self._scrub(
            "Provider redirected the request (HTTP %d) to host %s; redirects are not followed, so "
            "the API key is not sent there. Set base_url to the final address." % (status, host or "unknown")
        )

    def complete(
        self,
        messages: list,
        tools: list,
        deadline: Optional[float] = None,
        cancel: Optional[threading.Event] = None,
    ) -> Completion:
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
            except http.client.InvalidURL as exc:  # retrying cannot fix the configured URL
                raise LLMError("Invalid provider URL: %s" % self._scrub(str(exc)))
            except ValueError:
                # e.g. an invalid header value; never echo the text, it may hold the key
                raise LLMError("Request could not be sent (invalid request data, check the API key format)")
            except (OSError, http.client.HTTPException) as exc:  # URLError, timeouts, cut-off bodies
                status, raw = None, b""
                failure = "Network error contacting provider: %s" % self._scrub(str(exc))
            else:
                if status == 200:
                    return self._parse(raw)
                if 300 <= status < 400:
                    raise LLMError(self._redirect_message(status, resp_headers), status)
                retry_after = _header(resp_headers, "Retry-After")
                failure = "Provider returned HTTP %d: %s" % (status, self._error_message(raw))
                if status != 429 and status < 500:
                    raise LLMError(failure, status)
            if deadline is not None and self._clock() >= deadline:  # e.g. a timeout clamped to it
                raise DeadlineReached("Stopped: time limit reached (%s)" % failure, status)
            if attempt >= self._max_retries:
                raise LLMError(failure, status)
            delay = min(2 ** attempt, 20)
            if retry_after is not None:
                try:
                    delay = max(0.0, float(retry_after))
                except ValueError:
                    pass
            delay = min(delay, MAX_RETRY_DELAY)
            if deadline is not None and self._clock() + delay >= deadline:
                raise DeadlineReached("Stopped retrying: time limit reached (%s)" % failure, status)
            if cancel is None:
                self._sleep(delay)
            elif cancel.wait(delay):
                raise LLMError("cancelled")
            attempt += 1

    def _parse(self, raw: bytes) -> Completion:
        try:
            data = json.loads(raw.decode("utf-8"))
            choice = data["choices"][0]
            msg = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError("Provider returned an unreadable response: %s" % self._scrub(
                raw.decode("utf-8", "replace"))[:500])
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

from __future__ import annotations

import json
import unittest

from dllm.config import Profile
from dllm.llm import ChatClient, LLMError

KEY = "sk-super-secret-key"


def ok(message, usage=None):
    body = {"choices": [{"message": message, "finish_reason": "stop"}],
            "usage": usage or {"prompt_tokens": 3, "completion_tokens": 4}}
    return (200, {}, json.dumps(body).encode())


def err(status, message=None, headers=None):
    body = json.dumps({"error": {"message": message}}).encode() if message else b"oops"
    return (status, headers or {}, body)


class Fake:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append((url, headers, json.loads(body), timeout))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make(fake, profile=None, **kw):
    profile = profile or Profile("p", "http://h/v1/", "K", "m")
    sleeps = []
    client = ChatClient(profile, KEY, transport=fake, sleep=sleeps.append, **kw)
    return client, sleeps


TOOLS = [{"type": "function", "function": {"name": "f"}}]
MSGS = [{"role": "user", "content": "hi"}]


class ChatClientTests(unittest.TestCase):
    def test_request_body_shape(self):
        fake = Fake(ok({"role": "assistant", "content": "x"}))
        p = Profile("p", "http://h/v1/", "K", "m", extra_body={"reasoning_effort": "low"}, temperature=0.5)
        client, _ = make(fake, p)
        client.complete(MSGS, TOOLS)
        url, headers, body, _ = fake.requests[0]
        self.assertEqual(url, "http://h/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer " + KEY)
        self.assertEqual(body, {"model": "m", "messages": MSGS, "tools": TOOLS, "tool_choice": "auto",
                                "max_completion_tokens": 8192, "reasoning_effort": "low", "temperature": 0.5})

    def test_model_override(self):
        fake = Fake(ok({"role": "assistant", "content": "x"}))
        client, _ = make(fake, model="other")
        client.complete(MSGS, TOOLS)
        self.assertEqual(fake.requests[0][2]["model"], "other")

    def test_parses_tool_calls_and_usage(self):
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read", "arguments": '{"p": 1}'}},
            {"id": "b", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
        ]}
        client, _ = make(Fake(ok(msg)))
        c = client.complete(MSGS, TOOLS)
        self.assertEqual([(t.id, t.name, t.arguments) for t in c.tool_calls],
                         [("a", "read", '{"p": 1}'), ("b", "bash", "{}")])
        self.assertEqual((c.input_tokens, c.output_tokens, c.finish_reason), (3, 4, "stop"))

    def test_arguments_object_is_serialized(self):
        msg = {"role": "assistant", "tool_calls": [
            {"id": "a", "function": {"name": "read", "arguments": {"p": 1}}}]}
        client, _ = make(Fake(ok(msg)))
        c = client.complete(MSGS, TOOLS)
        self.assertEqual(c.tool_calls[0].arguments, json.dumps({"p": 1}))
        self.assertEqual(c.message["tool_calls"][0]["function"]["arguments"], json.dumps({"p": 1}))

    def test_missing_tool_call_id_gets_generated(self):
        msg = {"role": "assistant", "tool_calls": [
            {"function": {"name": "a", "arguments": "{}"}}, {"function": {"name": "b", "arguments": "{}"}}]}
        client, _ = make(Fake(ok(msg)))
        c = client.complete(MSGS, TOOLS)
        ids = [t.id for t in c.tool_calls]
        self.assertTrue(all(ids))
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual([t["id"] for t in c.message["tool_calls"]], ids)

    def test_sanitized_message_drops_extra_fields(self):
        msg = {"role": "assistant", "content": "hello", "reasoning": "secret thoughts"}
        client, _ = make(Fake(ok(msg)))
        c = client.complete(MSGS, TOOLS)
        self.assertEqual(c.message, {"role": "assistant", "content": "hello"})
        msg = {"role": "assistant", "content": None, "reasoning": "r", "tool_calls": [
            {"id": "a", "type": "function", "index": 0, "function": {"name": "n", "arguments": "{}"}}]}
        client, _ = make(Fake(ok(msg)))
        c = client.complete(MSGS, TOOLS)
        self.assertEqual(c.message, {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "n", "arguments": "{}"}}]})

    def test_retries_429_honoring_retry_after(self):
        fake = Fake(err(429, "slow", {"Retry-After": "2"}), ok({"role": "assistant", "content": "x"}))
        client, sleeps = make(fake)
        c = client.complete(MSGS, TOOLS)
        self.assertEqual(c.content, "x")
        self.assertEqual(sleeps, [2])
        self.assertEqual(len(fake.requests), 2)

    def test_retries_5xx_with_backoff(self):
        fake = Fake(err(503), err(503), ok({"role": "assistant", "content": "x"}))
        client, sleeps = make(fake)
        client.complete(MSGS, TOOLS)
        self.assertEqual(len(fake.requests), 3)
        self.assertEqual(len(sleeps), 2)
        self.assertLess(sleeps[0], sleeps[1])

    def test_retries_transport_exception(self):
        fake = Fake(OSError("boom"), ok({"role": "assistant", "content": "x"}))
        client, sleeps = make(fake)
        self.assertEqual(client.complete(MSGS, TOOLS).content, "x")
        self.assertEqual(len(sleeps), 1)

    def test_gives_up_after_max_retries(self):
        fake = Fake(*[err(500)] * 5)
        client, _ = make(fake, max_retries=4)
        with self.assertRaises(LLMError) as cm:
            client.complete(MSGS, TOOLS)
        self.assertEqual(cm.exception.status, 500)
        self.assertEqual(len(fake.requests), 5)

    def test_4xx_raises_with_provider_message(self):
        fake = Fake(err(400, "bad tools"))
        client, sleeps = make(fake)
        with self.assertRaises(LLMError) as cm:
            client.complete(MSGS, TOOLS)
        self.assertIn("bad tools", str(cm.exception))
        self.assertEqual(cm.exception.status, 400)
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(sleeps, [])

    def test_deadline_stops_retries(self):
        fake = Fake(err(500), ok({"role": "assistant", "content": "x"}))
        profile = Profile("p", "http://h/v1", "K", "m")
        sleeps = []
        client = ChatClient(profile, KEY, transport=fake, sleep=sleeps.append, clock=lambda: 100.0)
        with self.assertRaises(LLMError) as cm:
            client.complete(MSGS, TOOLS, deadline=100.5)
        self.assertIn("time limit", str(cm.exception))
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(sleeps, [])

    def assert_clean(self, exc):
        for text in (str(exc), repr(exc)):
            self.assertNotIn(KEY, text)
            self.assertNotIn(repr(KEY)[1:-1], text)

    def test_valueerror_with_key_is_not_retried_or_leaked(self):
        for text in ("Invalid header value Bearer " + KEY, "Invalid header value %r" % ("Bearer " + KEY + "\n",)):
            fake = Fake(ValueError(text), ok({"role": "assistant", "content": "x"}))
            client, sleeps = make(fake)
            with self.assertRaises(LLMError) as cm:
                client.complete(MSGS, TOOLS)
            self.assert_clean(cm.exception)
            self.assertEqual(len(fake.requests), 1)
            self.assertEqual(sleeps, [])

    def test_oserror_text_with_key_is_scrubbed(self):
        for text in ("failed " + KEY, "failed %r" % (KEY + "\n",)):
            client, _ = make(Fake(*[OSError(text)] * 5), max_retries=4)
            with self.assertRaises(LLMError) as cm:
                client.complete(MSGS, TOOLS)
            self.assertIn("failed", str(cm.exception))
            self.assert_clean(cm.exception)

    def test_provider_body_echoing_key_is_scrubbed(self):
        for fake in (Fake(err(400, "bad key " + KEY)), Fake(*[(500, {}, ("raw " + KEY).encode())] * 5)):
            client, _ = make(fake)
            with self.assertRaises(LLMError) as cm:
                client.complete(MSGS, TOOLS)
            self.assertIn("***", str(cm.exception))
            self.assert_clean(cm.exception)

    def test_unreadable_response_redacted_before_cut(self):
        raw = ("<html>" + "x" * 480 + KEY + "</html>").encode()
        client, _ = make(Fake((200, {}, raw)))
        with self.assertRaises(LLMError) as cm:
            client.complete(MSGS, TOOLS)
        for i in range(len(KEY) - 7):
            self.assertNotIn(KEY[i:i + 8], str(cm.exception))

    def test_client_repr_has_no_key(self):
        client, _ = make(Fake())
        self.assertNotIn(KEY, repr(client))
        self.assertNotIn(KEY, str(client))


if __name__ == "__main__":
    unittest.main()

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest

from genai_pyo3 import ChatMessage, ChatRequest, Client, GenaiError


class _ResponseHandler(BaseHTTPRequestHandler):
    status = 503
    body = b'{"error":{"type":"overloaded_error","message":"try later"}}'
    content_type = "application/json"
    retry_after = "2"
    truncate_body = False

    def do_POST(self):
        self.send_response(self.status)
        self.send_header("content-type", self.content_type)
        self.send_header("retry-after", self.retry_after)
        if self.truncate_body:
            self.send_header("content-length", str(len(self.body) + 100))
        else:
            self.send_header("content-length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)
        self.wfile.flush()
        if self.truncate_body:
            self.close_connection = True

    def log_message(self, _format, *_args):
        pass


class _LocalResponseServer:
    def __init__(self, handler):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}/v1/"

    def __exit__(self, _exc_type, _exc, _traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _client(base_url):
    return Client.with_api_key_and_base_url("anthropic", "test-key", base_url)


def _request():
    return ChatRequest(messages=[ChatMessage("user", "hello")], store=False)


async def _achat(base_url):
    return await _client(base_url).achat("test-model", _request())


async def _achat_via_stream(base_url):
    return await _client(base_url).achat_via_stream("test-model", _request())


class GenaiErrorTests(unittest.TestCase):
    def test_nonstream_http_error_exposes_status_headers_and_body(self):
        with _LocalResponseServer(_ResponseHandler) as base_url:
            with self.assertRaises(GenaiError) as raised:
                asyncio.run(_achat(base_url))

        error = raised.exception
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(error.kind, "http_error")
        self.assertEqual(error.status_code, 503)
        self.assertTrue(error.retryable)
        self.assertEqual(error.headers["retry-after"], "2")
        self.assertIn("overloaded_error", error.body)

    def test_stream_http_error_exposes_status_and_is_retryable(self):
        with _LocalResponseServer(_ResponseHandler) as base_url:
            with self.assertRaises(GenaiError) as raised:
                asyncio.run(_achat_via_stream(base_url))

        error = raised.exception
        self.assertEqual(error.kind, "http_error")
        self.assertEqual(error.status_code, 503)
        self.assertTrue(error.retryable)
        self.assertIn("overloaded_error", error.body)

    def test_broken_success_stream_is_retryable_without_an_http_status(self):
        class TruncatedStreamHandler(_ResponseHandler):
            status = 200
            body = b"event: message_start\ndata: {}\n\n"
            content_type = "text/event-stream"
            truncate_body = True

        with _LocalResponseServer(TruncatedStreamHandler) as base_url:
            with self.assertRaises(GenaiError) as raised:
                asyncio.run(_achat_via_stream(base_url))

        error = raised.exception
        self.assertIn(error.kind, {"response_body", "response_decode", "web_stream"})
        self.assertIsNone(error.status_code)
        self.assertTrue(error.retryable)


if __name__ == "__main__":
    unittest.main()

"""The real ClaudeLLM client against a local mock of the Messages API.

Checks what is actually sent over HTTP (model, beta header, fallbacks, caching,
tool schemas, history) and that genuine SDK response objects flow through the
graph. No network access or API key needed.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.agent.graph import LidarAgent
from src.agent.llm import ClaudeLLM

RESPONSES = [
    {  # 1st call: Claude asks for a tool
        "content": [
            {"type": "text", "text": "Let me check the coverage."},
            {"type": "tool_use", "id": "toolu_01", "name": "find_lidar_tiles", "input": {}},
        ],
        "stop_reason": "tool_use",
    },
    {  # 2nd call: final answer
        "content": [{"type": "text", "text": "The data covers 4 tiles."}],
        "stop_reason": "end_turn",
    },
]


@pytest.fixture
def mock_api():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
            reply = {"id": f"msg_{len(requests)}", "type": "message", "role": "assistant", "model": body["model"],
                     "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 10},
                     **RESPONSES[len(requests) - 1]}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", requests
    server.shutdown()


def test_request_shape_and_round_trip(mock_api, monkeypatch):
    base_url, requests = mock_api
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    monkeypatch.delenv("LIDAR_AGENT_MODEL", raising=False)

    result = LidarAgent(llm=ClaudeLLM()).ask("Which area does the data cover?")

    assert result.answer == "The data covers 4 tiles."
    first, second = requests[0], requests[1]
    assert first["path"].startswith("/v1/messages")
    assert first["headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01"

    body = first["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "medium"}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert {t["name"] for t in body["tools"]} >= {"load_point_cloud", "classify_lidar_points", "create_map"}

    # The second request replays the assistant turn unchanged, then the tool result.
    history = second["body"]["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[1]["content"][1]["type"] == "tool_use"
    tool_result = history[2]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_01" and not tool_result["is_error"]
    assert len(json.loads(tool_result["content"])["tiles"]) == 4

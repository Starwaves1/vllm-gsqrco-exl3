"""A minimal OpenAI-style vLLM server for the torture harness's CPU tests: valid answers for every
request kind the generator sends; tokens are characters (ord + 1000).
  python torture_mock.py PORT [TIER_NAMESPACE_DIR]     serve until SIGINT (a stand-in serve script)"""

import http.server
import json
import os
import sys
import threading
import time
from pathlib import Path


class Mock:
    """Minimal OpenAI-style vLLM server: valid answers for every request kind the generator sends.
    Tokens are characters (ord + 1000)."""

    def __init__(self, port: int, tier: str | None = None):
        mock = self
        self.start_time = 1.0

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def reply(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self.reply(200, {"data": [{"id": "mock", "max_model_len": 200000}]})
                if self.path == "/health":
                    return self.reply(200, b"", "text/plain")
                self.reply(200, (f"vllm:num_requests_running 0\nvllm:prefix_cache_hits_total 5\nvllm:prefix_cache_queries_total 10\n"
                                 f"process_start_time_seconds {mock.start_time}\n"
                                 'vllm:cache_config_info{block_size="16",engine="0",num_gpu_blocks="10000"} 1.0\n').encode(), "text/plain")

            def do_POST(self):
                q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    return self.reply(200, {"tokens": [ord(c) + 1000 for c in q["prompt"]], "count": len(q["prompt"])})
                if self.path == "/detokenize":
                    return self.reply(200, {"prompt": "".join(chr(t - 1000) for t in q["tokens"])})
                assert self.path.startswith("/v1/") and q["priority"] >= 100000
                try:
                    self.answer(q)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def answer(self, q):
                chat = self.path.endswith("chat/completions")
                mt = q["max_tokens"]
                ct = max(min(mt, 64), min(q.get("min_tokens", 0), mt))
                fin = "length" if ct == mt else "stop"
                text, tools = "word " * ct, []
                if chat and q.get("tools") and q.get("tool_choice") != "none":
                    tools = [{"id": "c0", "type": "function", "function": {"name": "run_shell", "arguments": '{"command": "ls"}'}}]
                    text, fin = "", "tool_calls" if fin == "stop" else fin
                if q.get("response_format"):
                    text, fin = '{"city": "Copenhagen", "population": 660000}', "stop"
                think = (q.get("chat_template_kwargs") or {}).get("enable_thinking")
                pt = len(q["prompt"]) if isinstance(q.get("prompt"), list) else 20
                usage = {"prompt_tokens": pt, "completion_tokens": ct * q.get("n", 1)}
                if not q.get("stream"):
                    msg = {"content": text or None, "reasoning": "hmm" if think else None, "tool_calls": tools}
                    ch = {"index": 0, "finish_reason": fin, **({"message": msg} if chat else {"text": text})}
                    if q.get("logprobs"):
                        ch["logprobs"] = {"content": [{"logprob": -0.5, "top_logprobs": [{"logprob": -0.5}]}] * ct} if chat \
                            else {"token_logprobs": [None] + [-0.5] * ct}
                    if q.get("prompt_logprobs"):
                        ch["prompt_logprobs"] = [None] + [{"1": {"logprob": -1.0}}] * (pt - 1)
                    return self.reply(200, {"choices": [dict(ch, index=i) for i in range(q.get("n", 1))], "usage": usage})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                ev = lambda d: self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n") or self.wfile.flush()  # noqa: E731
                if chat:
                    ev({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
                for piece in [text] if q.get("response_format") else ["word "] * (ct if text else 0):
                    time.sleep(0.005)
                    ev({"choices": [{"index": 0, **({"delta": {"content": piece}} if chat else {"text": piece})}]})
                for t in tools:
                    ev({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, **t}]}}]})
                ev({"choices": [{"index": 0, "delta": {}, "finish_reason": fin}]})
                ev({"choices": [], "usage": usage})
                self.wfile.write(b"data: [DONE]\n\n")

        if tier:  # a fake KV fs-tier namespace this server "writes"
            os.makedirs(tier, exist_ok=True)
            Path(tier, "blocks").write_text(str(time.time()))
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


if __name__ == "__main__":
    m = Mock(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        m.srv.shutdown()

"""Deterministic stand-in for the vLLM OpenAI server (tests only).

    python3 mock_upstream.py PORT

Scenario = first keyword found in the request body:
  SPLIT3 OPENREASON MIDSENT REPEAT FFFD BLANK NANLP SLOW (default: clean).
Streams use one HTTP chunk per SSE event like uvicorn, except SPLIT3, whose
content event is cut into two chunks inside the 3-byte character U+95AE.
GET /mock/stats returns what the mock received (sha256 of each raw request)
and how many clients disconnected during a SLOW request.
"""
import asyncio
import hashlib
import json
import sys

STATE = {"requests": [], "aborted": 0, "active": 0, "n": 0}

SCENARIOS = {
    "OPENREASON": ("The user asks about the capital. Let me think about whether", ""),
    "MIDSENT": ("Easy.", "The capital of France is located in the northern part of the"),
    "REPEAT": ("ok", "The question asks: During whose reignThe question asks about whose "
                     "reign the treaty was signed."),
    "FFFD": ("ok", "The answer is �1994."),
    "BLANK": ("", "\n\n\n"),
    "SPLIT3": ("ok", "It was founded in 199閮1994 by the city council of Lyon."),
    "SLOW": ("Slow thought.", "Slow but complete answer."),
    "NANLP": ("ok", "Fine answer."),
}
DEFAULT = ("Simple question.", "Paris is the capital of France.")


def pick(body):
    for k in SCENARIOS:
        if k.encode() in body:
            return k, SCENARIOS[k]
    return "DEFAULT", DEFAULT


def sse(obj):
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


def chunk(b):
    return f"{len(b):x}\r\n".encode() + b + b"\r\n"


def metrics():
    n = STATE["n"]
    return (
        "# HELP vllm:num_requests_running x\n"
        f'vllm:num_requests_running{{engine="0",model_name="m"}} {STATE["active"]}.0\n'
        f'vllm:num_requests_waiting{{engine="0",model_name="m"}} 0.0\n'
        f'vllm:kv_cache_usage_perc{{engine="0",model_name="m"}} 0.25\n'
        f'vllm:prompt_tokens_total{{engine="0",model_name="m"}} {10.0 * n}\n'
        f'vllm:generation_tokens_total{{engine="0",model_name="m"}} {20.0 * n}\n'
        f'vllm:spec_decode_num_drafts_total{{engine="0",model_name="m"}} {4.0 * n}\n'
        f'vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="m"}} {20.0 * n}\n'
        f'vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="m"}} {10.0 * n}\n'
        f'vllm:num_preemptions_total{{engine="0",model_name="m"}} 0.0\n'
        f'vllm:request_success_total{{engine="0",finished_reason="stop",model_name="m"}} {n}.0\n'
        "process_start_time_seconds 1700000000.0\n"
    ).encode()


def completion(path, req, scen, kw):
    reasoning, content = scen
    chat = path == "/v1/chat/completions"
    stream = bool(req.get("stream"))
    usage = {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16}
    lp = None
    if kw == "NANLP":
        lp = {"content": [{"token": "Fine", "logprob": -9999.0, "top_logprobs": []},
                          {"token": " answer", "logprob": None, "top_logprobs": []}]}
    if not stream:
        if chat:
            msg = {"role": "assistant", "content": content or None, "reasoning": reasoning or None}
            ch = {"index": 0, "message": msg, "logprobs": lp, "finish_reason": "stop",
                  "stop_reason": None}
            obj = {"id": "chatcmpl-mock", "object": "chat.completion", "created": 1,
                   "model": "m", "choices": [ch], "usage": usage}
        else:
            ch = {"index": 0, "text": (f"{reasoning}</think>\n\n{content}" if reasoning else content),
                  "logprobs": None, "finish_reason": "stop", "stop_reason": None}
            obj = {"id": "cmpl-mock", "object": "text_completion", "created": 1, "model": "m",
                   "choices": [ch], "usage": usage}
        return [json.dumps(obj, ensure_ascii=False).encode()], False
    events = []
    base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    if chat:
        events.append(sse(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""},
                                                "logprobs": None, "finish_reason": None}])))
        if reasoning:
            for i in range(0, len(reasoning), 12):
                events.append(sse(dict(base, choices=[{"index": 0, "delta": {"reasoning": reasoning[i:i + 12]},
                                                        "logprobs": None, "finish_reason": None}])))
        if content:
            events.append(sse(dict(base, choices=[{"index": 0, "delta": {"content": content},
                                                    "logprobs": lp, "finish_reason": None}])))
        events.append(sse(dict(base, choices=[{"index": 0, "delta": {}, "logprobs": None,
                                                "finish_reason": "stop", "stop_reason": None}])))
    else:
        for piece in (reasoning + "</think>\n\n" if reasoning else "", content):
            if piece:
                events.append(sse({"id": "cmpl-mock", "object": "text_completion", "created": 1,
                                   "model": "m", "choices": [{"index": 0, "text": piece, "logprobs": None,
                                                              "finish_reason": None}]}))
        events.append(sse({"id": "cmpl-mock", "object": "text_completion", "created": 1, "model": "m",
                           "choices": [{"index": 0, "text": "", "logprobs": None,
                                        "finish_reason": "stop", "stop_reason": None}]}))
    if (req.get("stream_options") or {}).get("include_usage"):
        events.append(sse(dict(base, choices=[], usage=usage)))
    events.append(b"data: [DONE]\n\n")
    if kw == "SPLIT3":
        out = []
        for e in events:
            k = e.find("閮".encode())
            if k >= 0:
                out += [e[:k + 1], e[k + 1:]]  # boundary after the lead byte
            else:
                out.append(e)
        events = out
    return events, True


async def handle(r, w):
    try:
        while True:
            try:
                head = await r.readuntil(b"\r\n\r\n")
            except asyncio.IncompleteReadError:
                return
            lines = head.decode("latin-1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            hdr = {k.strip().lower(): v.strip() for k, v in
                   (ln.split(":", 1) for ln in lines[1:] if ":" in ln)}
            if hdr.get("expect", "").lower() == "100-continue":
                w.write(b"HTTP/1.1 100 Continue\r\n\r\n")
                await w.drain()
            body = b""
            if "content-length" in hdr:
                body = await r.readexactly(int(hdr["content-length"]))
            STATE["requests"].append(hashlib.sha256(head + body).hexdigest())
            close = "close" in hdr.get("connection", "").lower()
            path = target.split("?")[0]
            keep = b"Connection: close\r\n" if close else b""
            if path in ("/v1/chat/completions", "/v1/completions") and method == "POST":
                req = json.loads(body)
                kw, scen = pick(body)
                STATE["active"] += 1
                STATE["n"] += 1
                try:
                    if kw == "SLOW":
                        for _ in range(30):
                            await asyncio.sleep(0.1)
                            if r.at_eof():
                                STATE["aborted"] += 1
                                return
                    parts, stream = completion(path, req, scen, kw)
                    if stream:
                        w.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream; charset=utf-8\r\n"
                                + keep + b"transfer-encoding: chunked\r\n\r\n")
                        for p in parts:
                            w.write(chunk(p))
                            await w.drain()
                            await asyncio.sleep(0.005)
                        w.write(b"0\r\n\r\n")
                    else:
                        b = parts[0]
                        w.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n" + keep +
                                f"content-length: {len(b)}\r\n\r\n".encode() + b)
                    await w.drain()
                finally:
                    STATE["active"] -= 1
            else:
                if path == "/metrics":
                    b, ct = metrics(), b"text/plain; version=0.0.4"
                elif path == "/v1/models":
                    b, ct = json.dumps({"object": "list", "data": [{"id": "m", "object": "model"}]}).encode(), \
                        b"application/json"
                elif path == "/mock/stats":
                    b, ct = json.dumps(STATE).encode(), b"application/json"
                elif path == "/health":
                    b, ct = b"", b"text/plain"
                else:
                    w.write(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n" + keep + b"\r\n")
                    await w.drain()
                    if close:
                        return
                    continue
                w.write(b"HTTP/1.1 200 OK\r\ncontent-type: " + ct + b"\r\n" + keep +
                        f"content-length: {len(b)}\r\n\r\n".encode() + b)
                await w.drain()
            if close:
                return
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        w.close()


async def main(port):
    srv = await asyncio.start_server(handle, "127.0.0.1", port)
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1])))

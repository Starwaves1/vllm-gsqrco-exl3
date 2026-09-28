"""Phase 1 smoke: plain chat, reasoning split, tool call (qwen3_coder parser) against serve-gsq.sh."""
import json, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18090"
KEY = "gsq-local-test"


def post(body):
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    out["_secs"] = round(time.time() - t, 2)
    return out


res = {}
m = [{"role": "user", "content": "In two sentences, explain why the sky is blue."}]
r = post({"model": "qwen3.8-27b", "messages": m, "max_tokens": 1024, "temperature": 0.6, "top_p": 0.95})
msg = r["choices"][0]["message"]
res["chat"] = {"content": msg.get("content"), "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "")[:1500],
               "finish": r["choices"][0]["finish_reason"], "usage": r["usage"], "secs": r["_secs"]}

tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   "required": ["city"]}}}]
m = [{"role": "user", "content": "What's the weather in Paris right now, in celsius? Use the tool."}]
r = post({"model": "qwen3.8-27b", "messages": m, "tools": tools, "tool_choice": "auto", "max_tokens": 2048, "temperature": 0.6, "top_p": 0.95})
msg = r["choices"][0]["message"]
res["tool"] = {"tool_calls": msg.get("tool_calls"), "content": msg.get("content"),
               "reasoning_len": len(msg.get("reasoning_content") or msg.get("reasoning") or ""),
               "finish": r["choices"][0]["finish_reason"], "usage": r["usage"], "secs": r["_secs"]}
print(json.dumps(res, indent=2))
tc = res["tool"]["tool_calls"] or []
ok = bool(res["chat"]["content"]) and bool(res["chat"]["reasoning"]) and tc and tc[0]["function"]["name"] == "get_weather" \
    and json.loads(tc[0]["function"]["arguments"]).get("city", "").lower().startswith("paris")
print("SMOKE_OK" if ok else "SMOKE_CHECK_FAILED")

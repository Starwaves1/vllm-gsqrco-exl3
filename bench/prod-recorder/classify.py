"""Parse recorded OpenAI-API responses and flag generation corruption.

Stdlib only. Everything here is pure (no I/O) so the recorder, the report
(--reclassify) and the tests share one definition of every flag.

Flags (each a dict {"field", "span": [a, b], "detail"} when set):
  early_stop_open_reasoning  EOS while the <think> block is still open
  early_stop_mid_sentence    EOS (not a stop string), fewer than max_tokens,
                             last line is a sentence (>= 4 words, not a list
                             item, heading, table row or code) ending in a
                             letter or comma
  repeat_fragment            a >= 12-char wordy fragment repeated back-to-back
                             or with <= 40 chars between the two copies
                             (code fences and table rows are ignored)
  foreign_script             CJK/Cyrillic/Arabic/... letters inside an
                             otherwise Latin-script response, unless the
                             prompt itself uses that script or it is allowed
  replacement_char           U+FFFD in the decoded text (prompt has none)
  blank_only                 every text field empty or whitespace, no tool call
  nonfinite_logprobs         returned logprobs contain NaN/inf/null or the
                             -9999 clamp (only with --save-logprobs)
"""
import json
import math
import re
import unicodedata

FLAG_NAMES = (
    "early_stop_open_reasoning",
    "early_stop_mid_sentence",
    "repeat_fragment",
    "foreign_script",
    "replacement_char",
    "blank_only",
    "nonfinite_logprobs",
)

REPEAT_MIN = 12
REPEAT_GAP = 40
DEFAULT_ALLOW = frozenset({"greek"})

# Code-point ranges of non-Latin scripts. Punctuation/symbol blocks (CJK
# punctuation U+3000-303F, fullwidth forms) are deliberately not listed.
_SCRIPT_RANGES = (
    (0x0370, 0x03FF, "greek"), (0x1F00, 0x1FFF, "greek"),
    (0x0400, 0x052F, "cyrillic"), (0x1C80, 0x1C8F, "cyrillic"),
    (0x2DE0, 0x2DFF, "cyrillic"), (0xA640, 0xA69F, "cyrillic"),
    (0x0530, 0x058F, "armenian"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"), (0x0750, 0x077F, "arabic"),
    (0x08A0, 0x08FF, "arabic"), (0xFB50, 0xFDFF, "arabic"),
    (0xFE70, 0xFEFF, "arabic"),
    (0x0700, 0x074F, "syriac"),
    (0x0780, 0x07BF, "thaana"),
    (0x0900, 0x0DFF, "indic"),
    (0x0E00, 0x0E7F, "thai"),
    (0x0E80, 0x0FFF, "lao_tibetan"),
    (0x1000, 0x109F, "myanmar"),
    (0x10A0, 0x10FF, "georgian"),
    (0x1100, 0x11FF, "hangul"), (0x3130, 0x318F, "hangul"),
    (0xA960, 0xA97F, "hangul"), (0xAC00, 0xD7AF, "hangul"),
    (0xFFA0, 0xFFDC, "hangul"),
    (0x1200, 0x139F, "ethiopic"),
    (0x1780, 0x17FF, "khmer"),
    (0x1800, 0x18AF, "mongolian"),
    (0x2E80, 0x2FDF, "han"), (0x3400, 0x4DBF, "han"),
    (0x4E00, 0x9FFF, "han"), (0xF900, 0xFAFF, "han"),
    (0x20000, 0x3FFFF, "han"),
    (0x3040, 0x30FF, "kana"), (0x31F0, 0x31FF, "kana"),
    (0xFF66, 0xFF9F, "kana"),
    (0x3100, 0x312F, "bopomofo"), (0x31A0, 0x31BF, "bopomofo"),
)


def script_of(ch):
    """'latin', a script name from _SCRIPT_RANGES, or None (not a letter)."""
    cp = ord(ch)
    if cp < 0x80:
        return "latin" if ch.isalpha() else None
    if cp < 0x250 or 0x1E00 <= cp <= 0x1EFF:
        return "latin" if ch.isalpha() else None
    for lo, hi, name in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            cat = unicodedata.category(ch)
            return name if cat[0] in "LM" else None
    return None


def scripts_in(text):
    """Set of non-Latin script names present in text."""
    out = set()
    for ch in text:
        if ord(ch) >= 0x370:
            s = script_of(ch)
            if s and s != "latin":
                out.add(s)
    return out


# ---------------------------------------------------------------- requests

_SECRET_KEYS = {"api_key", "apikey", "api-key", "authorization", "x-api-key",
                "openai_api_key", "secret", "password"}


def sanitize(obj):
    """Copy of a request body without API-key-like fields (any depth)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()
                if str(k).lower() not in _SECRET_KEYS}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def _content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return ""


def prompt_text(body):
    """All prompt-side text of a chat or completions request body."""
    if not isinstance(body, dict):
        return ""
    if isinstance(body.get("messages"), list):
        return "\n".join(_content_text(m.get("content")) for m in body["messages"]
                         if isinstance(m, dict))
    p = body.get("prompt")
    if isinstance(p, str):
        return p
    if isinstance(p, list):
        return "\n".join(x for x in p if isinstance(x, str))
    return ""


def request_info(body, path):
    """Sampling/conditions tags of a parsed request body (never raises)."""
    if not isinstance(body, dict):
        return {"type": "other"}
    kwargs = body.get("chat_template_kwargs") or {}
    so = body.get("stream_options") or {}
    mt = body.get("max_tokens")
    if mt is None:
        mt = body.get("max_completion_tokens")
    info = {
        "type": "chat" if path.startswith("/v1/chat/") else
                "completions" if path.startswith("/v1/completions") else "other",
        "model": body.get("model"),
        "stream": bool(body.get("stream")),
        "include_usage": bool(isinstance(so, dict) and so.get("include_usage")),
        "max_tokens": mt if isinstance(mt, int) else None,
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "top_k": body.get("top_k"),
        "min_p": body.get("min_p"),
        "seed": body.get("seed"),
        "n": body.get("n") or 1,
        "logprobs": body.get("logprobs") not in (None, False),
        "top_logprobs": body.get("top_logprobs"),
        "prompt_logprobs": body.get("prompt_logprobs") is not None,
        "echo": bool(body.get("echo")),
        "priority": body.get("priority"),
        "has_stop": bool(body.get("stop")),
        "stop_token_ids": body.get("stop_token_ids") or [],
        "enable_thinking": kwargs.get("enable_thinking") if isinstance(kwargs, dict) else None,
        "tools": bool(body.get("tools")),
    }
    return info


def prompt_info(body):
    """Facts about the prompt the classifier needs (kept so --reclassify
    works even when the stored prompt was truncated)."""
    text = prompt_text(body)
    tail = text[text.rfind("<|im_start|>assistant"):] if "<|im_start|>assistant" in text else text[-4000:]
    open_think = isinstance(body, dict) and isinstance(body.get("prompt"), (str, list)) \
        and tail.count("<think>") > tail.count("</think>")
    return {"chars": len(text), "scripts": sorted(scripts_in(text)),
            "has_fffd": "�" in text, "open_think": bool(open_think)}


# --------------------------------------------------------------- responses

_LP_KEYS = {"logprobs", "prompt_logprobs", "top_logprobs", "token_logprobs"}
_CLAMP = -9999.0


def _scan_lp(obj, acc, in_lp=False, key=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _scan_lp(v, acc, in_lp or k in _LP_KEYS, k)
    elif isinstance(obj, list):
        for v in obj:
            _scan_lp(v, acc, in_lp, key)
    elif in_lp:
        if isinstance(obj, float) or (isinstance(obj, int) and not isinstance(obj, bool)):
            if key in ("rank", "bytes", "text_offset", "token_id", "index"):
                return
            acc["values"] += 1
            if isinstance(obj, float) and not math.isfinite(obj):
                acc["nonfinite"] += 1
            elif obj <= _CLAMP:
                acc["clamped"] += 1
        elif obj is None and key == "logprob":
            acc["values"] += 1
            acc["nonfinite"] += 1


def _choice(choices, idx):
    return choices.setdefault(idx, {"content": "", "reasoning": "", "text": "",
                                    "finish_reason": None, "stop_reason": None,
                                    "tool_calls": 0})


def _absorb(choices, ch, delta_key):
    idx = ch.get("index", 0)
    c = _choice(choices, idx)
    d = ch.get(delta_key) if delta_key else None
    if isinstance(d, dict):
        if isinstance(d.get("content"), str):
            c["content"] += d["content"]
        r = d.get("reasoning")
        if r is None:
            r = d.get("reasoning_content")
        if isinstance(r, str):
            c["reasoning"] += r
        if d.get("tool_calls"):
            c["tool_calls"] += len(d["tool_calls"])
    if isinstance(ch.get("text"), str):
        c["text"] += ch["text"]
    if ch.get("finish_reason") is not None:
        c["finish_reason"] = ch["finish_reason"]
    if ch.get("stop_reason") is not None:
        c["stop_reason"] = ch["stop_reason"]


def parse_response(body, content_type="", scan_logprobs=False):
    """Reassemble choices from an SSE stream or a JSON body.

    Returns {"kind", "choices": [..], "usage", "error", "events",
             "bad_events", "utf8_valid", "lp": {...} | None}.
    """
    out = {"kind": "other", "choices": [], "usage": None, "error": None,
           "events": 0, "bad_events": 0, "utf8_valid": True, "lp": None}
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        out["utf8_valid"] = False
        text = body.decode("utf-8", "replace")
    lp = {"values": 0, "nonfinite": 0, "clamped": 0} if scan_logprobs else None
    choices = {}
    if "text/event-stream" in (content_type or "") or text.startswith("data:"):
        out["kind"] = "sse"
        for event in re.split(r"\r?\n\r?\n", text):
            data = "\n".join(line[5:].lstrip(" ") for line in event.splitlines()
                             if line.startswith("data:"))
            if not data or data.strip() == "[DONE]":
                continue
            out["events"] += 1
            try:
                obj = json.loads(data)
            except ValueError:
                out["bad_events"] += 1
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("error"):
                out["error"] = obj["error"]
            if obj.get("usage"):
                out["usage"] = obj["usage"]
            for ch in obj.get("choices") or []:
                if isinstance(ch, dict):
                    _absorb(choices, ch, "delta")
            if lp is not None:
                _scan_lp(obj, lp)
    else:
        try:
            obj = json.loads(text) if text.strip() else None
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            out["kind"] = "json"
            out["usage"] = obj.get("usage")
            if obj.get("error") or obj.get("object") == "error":
                out["error"] = obj.get("error") or obj.get("message")
            for ch in obj.get("choices") or []:
                if isinstance(ch, dict):
                    _absorb(choices, ch, "message")
            if lp is not None:
                _scan_lp(obj, lp)
    out["choices"] = [choices[k] for k in sorted(choices)]
    out["lp"] = lp
    return out


def utf8_split_boundaries(body, chunk_sizes):
    """How many chunk boundaries fall inside a multi-byte UTF-8 sequence.

    A boundary at offset b splits a character exactly when body[b] is a
    continuation byte (10xxxxxx)."""
    n, off = 0, 0
    for size in chunk_sizes[:-1]:
        off += size
        if 0 < off < len(body) and (body[off] & 0xC0) == 0x80:
            n += 1
    return n


def sse_split_boundaries(body, chunk_sizes):
    """How many chunk boundaries fall inside an SSE event (not after \\n\\n)."""
    n, off = 0, 0
    for size in chunk_sizes[:-1]:
        off += size
        if 0 < off < len(body) and body[max(0, off - 2):off] != b"\n\n" \
                and body[max(0, off - 4):off] != b"\r\n\r\n":
            n += 1
    return n


# ----------------------------------------------------------------- flags

_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
_TABLE = re.compile(r"^[ \t]*\|.*$", re.M)


def _mask(text):
    """Blank out code fences and table rows, keeping offsets."""
    def blank(m):
        return "\x00" * len(m.group(0))
    return _TABLE.sub(blank, _FENCE.sub(blank, text))


def find_repeat(text, k=REPEAT_MIN, gap=REPEAT_GAP):
    """(first, second, length) of the first wordy fragment of >= k chars that
    reappears with at most `gap` chars between the copies, else None."""
    s = _mask(text)
    last = {}
    for j in range(len(s) - k + 1):
        g = s[j:j + k]
        p = last.get(g)
        if p is not None and k <= j - p <= k + gap and "\n" not in g \
                and "\x00" not in g and sum(c.isalpha() for c in g) >= 8:
            L = k
            while j + L < len(s) and p + L < j and s[p + L] == s[j + L]:
                L += 1
            return p, j, L
        last[g] = j
    return None


_LIST_START = re.compile(r"^\s*([-*+>#|]|\d+[.)]|[A-Za-z][.)]\s)")


def ends_mid_sentence(text):
    """True when the final line reads like an unfinished sentence."""
    t = text.rstrip()
    if not t or t.count("```") % 2 == 1:
        return False
    last_char = t[-1]
    if not (last_char == "," or unicodedata.category(last_char).startswith("L")):
        return False
    line = t.splitlines()[-1]
    if _LIST_START.match(line):
        return False
    return len(line.split()) >= 4


def foreign_runs(text, allow, prompt_scripts):
    """Spans of disallowed non-Latin letters in an otherwise Latin text."""
    latin = foreign = 0
    runs = []
    for i, ch in enumerate(text):
        s = script_of(ch)
        if s is None:
            continue
        if s == "latin":
            latin += 1
        elif s in allow or s in prompt_scripts:
            continue
        else:
            foreign += 1
            if runs and runs[-1][1] >= i - 1:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1, s])
    if not runs or latin < 20 or latin < 4 * foreign:
        return []
    return runs


def _flag(field, span=None, detail=None):
    return {"field": field, "span": span, "detail": detail}


def stop_kind(choice, rinfo):
    """'eos', 'stop_string', 'stop_token' (client-requested id) or None."""
    if choice.get("finish_reason") != "stop":
        return None
    sr = choice.get("stop_reason")
    if isinstance(sr, str):
        return "stop_string"
    if isinstance(sr, int) and sr in (rinfo.get("stop_token_ids") or []):
        return "stop_token"
    return "eos"


def classify_choice(choice, rinfo, pinfo, usage_tokens=None, allow=DEFAULT_ALLOW):
    flags = {}
    content, reasoning, text = choice["content"], choice["reasoning"], choice["text"]
    fields = [("content", content), ("reasoning", reasoning), ("text", text)]
    tool = choice.get("tool_calls", 0) > 0
    sk = stop_kind(choice, rinfo)
    eos = sk == "eos"
    mt = rinfo.get("max_tokens")
    under_max = usage_tokens is None or mt is None or usage_tokens < mt

    if not tool and choice.get("finish_reason") and all(not v.strip() for _, v in fields):
        flags["blank_only"] = _flag("all", None, repr((content or reasoning or text)[:40]))

    # The visible answer: content (chat) or text after </think> (completions).
    answer_field, answer = "content", content
    if rinfo.get("type") == "completions" or (not content and text):
        answer_field, answer = "text", text
        if pinfo.get("open_think"):
            answer = text.split("</think>", 1)[1] if "</think>" in text else ""

    if eos and under_max and not tool and "blank_only" not in flags:
        open_r = False
        if reasoning.strip() and not content.strip():
            open_r = True
        elif "<think>" in content and content.count("<think>") > content.count("</think>"):
            open_r = True
        elif pinfo.get("open_think") and "</think>" not in text and text.strip():
            open_r = True
        if open_r:
            src = reasoning or content or text
            tail = len(src)
            flags["early_stop_open_reasoning"] = _flag(
                "reasoning" if reasoning.strip() else answer_field,
                [max(0, tail - 1), tail], repr(src[-80:]))
        elif ends_mid_sentence(answer):
            n = len(answer)
            flags["early_stop_mid_sentence"] = _flag(
                answer_field, [max(0, n - 1), n], repr(answer.rstrip()[-80:]))

    for name, value in fields:
        if not value:
            continue
        if "repeat_fragment" not in flags:
            r = find_repeat(value)
            if r:
                p, j, L = r
                flags["repeat_fragment"] = _flag(name, [p, j + L], repr(value[j:j + L]))
        if "foreign_script" not in flags:
            runs = foreign_runs(value, allow, set(pinfo.get("scripts") or ()))
            if runs:
                flags["foreign_script"] = _flag(
                    name, runs[0][:2],
                    {"scripts": sorted({r[2] for r in runs}), "runs": len(runs),
                     "chars": value[runs[0][0]:runs[0][1]]})
        if "replacement_char" not in flags and "�" in value and not pinfo.get("has_fffd"):
            i = value.index("�")
            flags["replacement_char"] = _flag(name, [i, i + 1], value.count("�"))
    return flags


def classify(parsed, rinfo, pinfo, allow=DEFAULT_ALLOW):
    """Flags of a whole response: union over choices, keyed by flag name;
    each value also carries the choice index."""
    usage = parsed.get("usage") or {}
    single = len(parsed["choices"]) == 1
    ctoks = usage.get("completion_tokens") if single else None
    flags = {}
    for i, ch in enumerate(parsed["choices"]):
        for name, f in classify_choice(ch, rinfo, pinfo, ctoks, allow).items():
            if name not in flags:
                f["choice"] = i
                flags[name] = f
    lp = parsed.get("lp")
    if lp and (lp["nonfinite"] or lp["clamped"]):
        flags["nonfinite_logprobs"] = _flag("logprobs", None, dict(lp))
    return flags

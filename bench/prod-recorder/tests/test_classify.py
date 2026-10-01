"""Classifier unit tests on synthetic texts (positives and negatives)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import classify as C  # noqa: E402

CHAT = {"type": "chat", "max_tokens": 4096, "stop_token_ids": []}
CMPL = {"type": "completions", "max_tokens": 4096, "stop_token_ids": []}
EN = {"scripts": [], "has_fffd": False, "open_think": False}


def choice(content="", reasoning="", text="", finish="stop", stop_reason=None, tools=0):
    return {"content": content, "reasoning": reasoning, "text": text, "finish_reason": finish,
            "stop_reason": stop_reason, "tool_calls": tools}


def flags(ch, rinfo=CHAT, pinfo=EN, tokens=None):
    return set(C.classify_choice(ch, rinfo, pinfo, tokens))


GOOD = choice("Paris is the capital of France. It has been since the 10th century.",
              "The user asks for the capital of France. That is Paris.")


class Flags(unittest.TestCase):
    def test_clean_answer(self):
        self.assertEqual(flags(GOOD), set())

    def test_open_reasoning(self):
        ch = choice("", "The question asks: During whose reign was the")
        self.assertEqual(flags(ch), {"early_stop_open_reasoning"})
        # same text but the model hit max_tokens -> not an early stop
        self.assertEqual(flags(choice("", "The question asks about", finish="length")), set())
        # a client stop string ended it -> not ours to flag
        self.assertEqual(flags(choice("", "reasoning text here", stop_reason="###")), set())
        # unclosed <think> in raw content (no reasoning parser)
        self.assertIn("early_stop_open_reasoning", flags(choice("<think>\nLet me see whether")))
        # raw completion after a prompt that opened <think>
        p = dict(EN, open_think=True)
        self.assertIn("early_stop_open_reasoning", flags(choice(text="Hmm, the user wants"), CMPL, p))
        self.assertEqual(flags(choice(text="Easy.</think>\n\nIt is Paris."), CMPL, p), set())

    def test_mid_sentence(self):
        ch = choice("The capital of France is located in the northern part of the")
        self.assertEqual(flags(ch), {"early_stop_mid_sentence"})
        self.assertIn("early_stop_mid_sentence", flags(choice("We compared three options, and the best one,")))
        # completion_tokens == max_tokens -> not early
        self.assertEqual(flags(ch, tokens=4096), set())
        self.assertEqual(flags(ch, tokens=12), {"early_stop_mid_sentence"})
        for ok in ("B", "Answer: B", "**Final answer: 1994**", "The answer is 42",
                   "- use a cache for repeated lookups", "1. Install the package first",
                   "## Summary of the main findings", "| a | b | c | d |",
                   "```python\nprint('hello world how are you')\n```", "Done!", "It is Paris.",
                   "See the table below:", "He said \"it works fine now\""):
            self.assertEqual(flags(choice(ok)), set(), ok)
        self.assertEqual(flags(choice("The capital of France is located in the", finish="length")), set())

    def test_repeat_fragment(self):
        bad = "The question asks: During whose reignThe question asks about whose reign the treaty was signed."
        self.assertEqual(flags(choice(bad + "")), {"repeat_fragment"})
        f = C.classify_choice(choice(bad), CHAT, EN)["repeat_fragment"]
        a, b = f["span"]
        self.assertTrue(bad[a:].startswith("The question asks"))
        self.assertIn("The question asks", f["detail"])
        self.assertIn("repeat_fragment", flags(choice("It was built in the city ofIt was built in the city of Rome.")))
        # same fragment far apart is fine
        far = "The treaty was signed in Paris. " + "x" * 0 + "Many details followed, then later on. " \
              "Historians agree the treaty was signed in Paris."
        self.assertNotIn("repeat_fragment", flags(choice(far)))
        # code fences, tables and separator rows are ignored
        code = "Here:\n```python\nresult = compute(alpha)\nresult = compute(beta)\n```\nThat is all."
        self.assertNotIn("repeat_fragment", flags(choice(code)))
        table = "| Model name here | Score |\n|---|---|\n| Model name here | 1 |\n\nDone."
        self.assertNotIn("repeat_fragment", flags(choice(table)))
        self.assertNotIn("repeat_fragment", flags(choice("=" * 60 + "\nAll good here.")))
        self.assertNotIn("repeat_fragment", flags(choice("Step one: open it.\nStep two: close it.")))

    def test_foreign_script(self):
        self.assertEqual(flags(choice("The war ended in 199閮1994 after long negotiations.")),
                         {"foreign_script"})
        self.assertIn("foreign_script", flags(choice("The result is very хорошо good overall.")))
        self.assertIn("foreign_script", flags(choice("Then we compute the value من for the answer.")))
        # CJK answer to a CJK prompt: legit
        cjk = dict(EN, scripts=["han"])
        self.assertEqual(flags(choice("巴黎是法国的首都。"), CHAT, cjk), set())
        self.assertEqual(flags(choice("In Chinese, Paris is 巴黎 and it is the capital."), CHAT, cjk), set())
        # mostly non-Latin answer (e.g. a translation request) is not "inside English"
        self.assertNotIn("foreign_script", flags(choice("巴黎是法国的首都。 Paris.")))
        # Greek math and fullwidth/CJK punctuation are allowed by default
        self.assertEqual(flags(choice("The angle θ equals π/2 radians, so α + β = π.")), set())
        self.assertEqual(flags(choice("Paris is the capital of France， obviously。 Yes it is.")), set())
        # configurable allowlist
        ru = choice("The result is very хорошо good overall.")
        self.assertEqual(set(C.classify_choice(ru, CHAT, EN, None, allow={"cyrillic"})), set())

    def test_replacement_char(self):
        self.assertEqual(flags(choice("The answer is �1994.")), {"replacement_char"})
        self.assertEqual(flags(choice("", "reason � here.", finish="length")), {"replacement_char"})
        self.assertEqual(flags(choice("The answer is �1994."), CHAT, dict(EN, has_fffd=True)), set())

    def test_blank_only(self):
        self.assertEqual(flags(choice("\n\n\n")), {"blank_only"})
        self.assertEqual(flags(choice("")), {"blank_only"})
        self.assertEqual(flags(choice(text="\n \n", finish="length"), CMPL), {"blank_only"})
        self.assertEqual(flags(choice("", tools=1, finish="tool_calls")), set())
        self.assertEqual(flags(choice("", finish=None)), set())  # never finished (aborted)

    def test_stop_token_from_client(self):
        r = dict(CHAT, stop_token_ids=[42])
        self.assertEqual(flags(choice("The capital of France is in the", stop_reason=42), r), set())
        # an int stop_reason the client did not ask for is EOS-like (<|im_end|>)
        self.assertIn("early_stop_mid_sentence",
                      flags(choice("The capital of France is in the", stop_reason=151645)))


class Parsing(unittest.TestCase):
    def sse(self, *objs):
        return b"".join(b"data: " + json.dumps(o, ensure_ascii=False).encode() + b"\n\n" for o in objs) \
            + b"data: [DONE]\n\n"

    def test_stream_chat_with_reasoning(self):
        body = self.sse(
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"index": 0, "delta": {"reasoning": "Think "}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": "more."}}]},
            {"choices": [{"index": 0, "delta": {"content": "Paris 閮."}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "stop_reason": None}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5}})
        p = C.parse_response(body, "text/event-stream")
        self.assertEqual(p["kind"], "sse")
        (ch,) = p["choices"]
        self.assertEqual(ch["reasoning"], "Think more.")
        self.assertEqual(ch["content"], "Paris 閮.")
        self.assertEqual(ch["finish_reason"], "stop")
        self.assertEqual(p["usage"]["completion_tokens"], 5)

    def test_json_completion_and_logprobs(self):
        obj = {"choices": [{"index": 0, "text": "hi", "finish_reason": "length",
                            "logprobs": {"token_logprobs": [None, -0.5, float("nan")],
                                         "top_logprobs": [None, {"a": -9999.0}]}}],
               "prompt_logprobs": [None, {"5": {"logprob": None, "rank": 1}}]}
        raw = json.dumps(obj).encode()
        p = C.parse_response(raw, "application/json", scan_logprobs=True)
        self.assertEqual(p["choices"][0]["text"], "hi")
        self.assertEqual(p["lp"]["nonfinite"], 2)  # NaN + null "logprob"
        self.assertEqual(p["lp"]["clamped"], 1)
        f = C.classify(p, CMPL, EN)
        self.assertIn("nonfinite_logprobs", f)
        self.assertIsNone(C.parse_response(raw, "application/json")["lp"])

    def test_utf8_split_counter(self):
        body = "ab閮cd".encode()  # 61 62 e9 96 ae 63 64
        self.assertEqual(C.utf8_split_boundaries(body, [3, 4]), 1)
        self.assertEqual(C.utf8_split_boundaries(body, [4, 3]), 1)
        self.assertEqual(C.utf8_split_boundaries(body, [2, 5]), 0)
        self.assertEqual(C.utf8_split_boundaries(body, [5, 2]), 0)
        self.assertEqual(C.utf8_split_boundaries(body, [7]), 0)

    def test_sanitize_and_prompt_info(self):
        body = {"model": "m", "api_key": "sk-1", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "你好"}]}],
            "extra_body": {"Authorization": "Bearer x", "max_tokens": 5}}
        s = C.sanitize(body)
        self.assertNotIn("api_key", s)
        self.assertEqual(s["extra_body"], {"max_tokens": 5})
        self.assertEqual(C.prompt_info(body)["scripts"], ["han"])
        raw = {"prompt": "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"}
        self.assertTrue(C.prompt_info(raw)["open_think"])
        closed = {"prompt": "<|im_start|>assistant\n<think>\n\n</think>\n\n"}
        self.assertFalse(C.prompt_info(closed)["open_think"])


if __name__ == "__main__":
    unittest.main()

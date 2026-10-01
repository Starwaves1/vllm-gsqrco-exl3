# prod-recorder

A recording reverse proxy for the production vLLM on ms4. Clients pointed at `:18082` get relayed byte-for-byte to `:18081`, and every generation gets recorded and checked for the GSQ-RCO corruption signatures. Stdlib-only Python 3.12+, no venv. Canonical source: gsq-vllm branch `prod-recorder`, `bench/prod-recorder/`. Refresh an installed copy with `./sync-from-repo.sh`.

```bash
./recorder start --save-logprobs [--bind 192.168.1.5] [--label NAME]   # proxy :18082 -> :18081 plus the 1 Hz /metrics sampler
./recorder status                                                      # counts, flags so far, detected server config, disk use
./recorder report --since 1h          # data/REPORT.md + report.json;  ./recorder stop  ends it (in-flight requests finish first)
```

The proxy and the sampler run in one process (one pid file, `data/recorder.pid`), because each response is tagged with the sampler's view of its own lifetime. `python3 sampler.py` also runs on its own and records metrics only.

## Pointing clients at it

`OPENAI_BASE_URL=http://127.0.0.1:18082/v1` (or `http://192.168.1.5:18082/v1` after `start --bind 192.168.1.5`). Nothing on production changes: the vision proxy on 18080, vLLM on 18081, and their units and configs stay as they are. Only traffic sent to 18082 is recorded. Image requests should keep using 18080, because 18082 talks to vLLM directly and skips the vision sidecar.

Pass-through guarantees:
- The request head and body are forwarded unchanged. That includes `priority`, every sampling field and every header; the recorder never adds or removes a field.
- The response is relayed with its exact bytes and HTTP chunk framing, so SSE events are never re-chunked. `Expect: 100-continue` and keep-alive work.
- Each request gets its own upstream connection. If the client disconnects, the recorder closes that connection and vLLM aborts the request, the same as a direct connection.
- Recording never fails a request. Errors are counted (`record_errors` in `status`) and that request's record is dropped.
- vLLM sees every proxied client as 127.0.0.1.

## What it records (`data/`)

- `requests-<YYYYmmdd-HH>.jsonl`: one line per request, filed under the hour it finished. Each line has:
  - the sanitized request body (API keys dropped; strings over 8 KB are cut once the body passes 256 KB)
  - timing and status
  - the raw response bytes reassembled from the stream (capped at 1 MB stored)
  - the HTTP chunk sizes, plus how many chunk boundaries split a UTF-8 multi-byte character (`utf8_split`) or an SSE event (`sse_split`)
  - finish and stop reasons, usage, and the flags
  - `cond`: the conditions during generation:
    - running count min/max/mean
    - the k in use, from the MTP schedule in the server's argv (`k_mode`, `k_set`, `k_mixed`)
    - acceptance over the window
    - preemptions, waiting and KV peaks
    - whether another prompt_logprobs or echo request was in flight
  - `req_info`: temperature, top_p, top_k, max_tokens, stream, prompt_logprobs, echo, priority
  - `up`: the vLLM config found by scanning /proc: `async`, `spec` (mtp/off), `k_cfg` (schedule or fixed k), `pid`, `start_ts`, `label`. A prod restart is picked up automatically.
- `metrics-<hour>.jsonl`: one line per second with the dashboard counters (running, waiting, KV usage, prompt/generated tokens, prefix cache, drafts/accepted, preemptions, finished by reason) and the proxied in-flight requests by type.
- `events.jsonl`: recorder start/stop and each server config seen. `status.json` is refreshed every 5 s. `recorder.log` holds the log.
- Hour files are gzipped after the hour has passed. `report` reads both forms.

## Flags (`classify.py`; `report --reclassify` re-runs them on stored data)

| flag | rule |
|---|---|
| `early_stop_open_reasoning` | EOS (not a client stop string) while `<think>` is open: reasoning but no content, or an unclosed `<think>` |
| `early_stop_mid_sentence` | EOS before max_tokens and the last line is a sentence of 4+ words (not a list item, heading, table row or code) that ends in a letter or comma |
| `repeat_fragment` | a fragment of 12+ characters (at least 8 letters) repeated back-to-back or within 40 characters, outside code fences and tables |
| `foreign_script` | CJK, Cyrillic, Arabic or other non-Latin letters inside an otherwise Latin response. Scripts that appear in the prompt are exempt; `--allow-scripts` defaults to `greek`. |
| `replacement_char` | U+FFFD in the decoded text. The bytes are reassembled before decoding, so a U+FFFD here came from the server, not from a chunk split. |
| `blank_only` | the response finished and every text field is empty or whitespace (no tool call) |
| `nonfinite_logprobs` | with `--save-logprobs`: NaN, inf or null logprob, or the -9999 clamp, in logprobs the client asked for |

`--save-logprobs` never adds `logprobs` to a request. When a client asked for logprobs, prompt_logprobs or echo, the recorder keeps the whole payload and counts NaN, inf and -9999 values in it. A sampled token at -9999 means it had zero probability. That is the signature of NaN logits breaking rejection sampling.

## Report

`report.py` gives the corruption rate with 95% Wilson intervals in this order:
1. overall and per flag
2. by temperature (T=0 vs T>0; an unset temperature defaults to 1.0 here, so it counts as T>0)
3. by k in use and by running count, each split by T=0 and T>0
4. by prompt length, prompt_logprobs in flight, echo in flight, streaming, endpoint, preemption, server config, label and hour

It also lists the 20 newest flagged examples, with the flagged span marked ⟦like this⟧, and the UTF-8 split statistic. `--compare A B` compares two windows using Newcombe 95% intervals for the difference. A window is `FROM..TO` (for example `11:00..12:30` or `-2h..now`) or a tag such as `spec=off`, `k_cfg=3` or `label=x`. Join several with commas.

## Disk

Measured on the mock: 2.4 KB per small request. Rough sizes in production:
- non-streamed generation: about 4 B per output token plus the prompt (≤256 KB). A judge call is 10–20 KB.
- streamed generation: about 250 B per SSE event, so 0.2–0.5 MB for 2k tokens before gzip (gzip gives about 3–20x).
- metrics: about 1.5 MB/h.

3,000 judge generations per hour comes to roughly 50 MB/h non-streamed and up to 1 GB/h streamed before the hourly gzip.

## Experiment plan

Production already runs with `--no-async-scheduling`, so async on/off is not the experiment. The leading suspect is NaN/inf in draft or target logits, which turns rejection sampling into random tokens at T>0, and gives odd argmax/EOS at T=0. Each window needs a prod restart, and those are Garrett's call. Record each for at least 1 h of judge traffic (aim for ≥1,500 generations per window: at that size a drop from 2% to 0.7% or lower is significant):

1. As-is, with the MTP schedule: `./recorder start --save-logprobs`, then point the judges at :18082.
2. Speculative decoding off. The recorder keeps running and tags the restart itself (`spec=off`).
3. k fixed at 3 (`num_speculative_tokens: 3`, no per-batch schedule; tag `k_cfg=3`).

```bash
./recorder report --compare spec=mtp spec=off
./recorder report --compare 'k_cfg=[[1,4,5],[5,8,3]]' k_cfg=3     # use the k_cfg shown by `status`
```

The T=0 and T>0 rows of each comparison are the important ones.

## Tests

`./recorder test` takes about 80 s on CPU. It starts a mock upstream in a subprocess and covers:
- byte-identical streaming and requests
- a UTF-8 split detected and left unaltered
- keep-alive, 100-continue, and client abort propagation
- a recording failure that does not fail the client, and a 502 when the upstream is down
- the classifier on synthetic positives and negatives, including legitimate CJK
- the report and `--compare` on synthetic data
- a `--minutes 1` smoke under load

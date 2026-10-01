# torture-harness

Torture soak for any OpenAI-compatible vLLM server: Python 3.12+ standard library only, polite by default (priority 100000, ≤16 requests and ≤256k tokens in flight). Full doc: `TORTURE.md`. Source: gsq-vllm branch `torture`, `bench/torture/`; refresh with `./sync-from-repo.sh`.

```bash
./torture --plan --minutes 20                                                    # print the schedule only
./torture --base-url http://127.0.0.1:18081/v1 --minutes 20 --priority 100000    # 20-minute smoke
./torture --base-url http://127.0.0.1:18081/v1 --hours 12 --priority 100000      # ms4: 12 h against production
./torture serve --cmd ./serve.sh --port 18090 --hours 12                         # start a server, torture it, stop it
./torture switch --a ./serve-a.sh --b ./serve-b.sh --port 18090 --rounds 3       # model switches + leftover checks
```

API key: `--api-key` or `VLLM_API_KEY`. Server-side checks (GPU MiB, RSS, death, fault grep): add `--server-pid PID --server-log FILE`. Results: `torture-runs/<ts>-<mode>/REPORT.md`. The exit code is the verdict.

# Sending results back

Each run writes one folder, `kit/results/<prefix>-<YYYY-MM-DD>/`. It holds no secrets: hostname, GPU and driver
facts, versions, timings, and the generated text of the corruption check. Look through `summary.md` first if you
want to.

Pick one way:

1. **Pull request (preferred).** Fork https://github.com/Starwaves1/vllm-gsqrco-exl3, commit the folder on a
   branch, and open a PR against `validation-kit` (or `main` once the kit is merged) titled
   `kit results: <card>`. Paste `summary.md` into the description.
2. **GitHub issue.** Zip the folder (`cd kit/results && zip -r <folder>.zip <folder>`), open an issue titled
   `kit results: <card>`, attach the zip, and paste `summary.md` into the body.
3. **Discord.** Send the zip to the person who asked you to run the kit, with one line: the card, and anything
   unusual (a different power limit, other jobs on the machine, a failed step).

If a step failed, include the `kit/.work/build-*.log` files the summary mentions. They are outside the results
folder.

Thank you. Every card adds a row to the tested-hardware table. Without it, that card is listed as untested.

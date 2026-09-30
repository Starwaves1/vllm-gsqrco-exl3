# vllm-exl3-plugin

Out-of-tree vLLM quantization plugin for exllamav3 EXL3 (trellis) checkpoints. In progress: CPU-only
so far, the kernels have not run. Design, build, tests and the GPU plan: `../EXL3.md`.

Kernels: exllamav3 v1.5.3 (d3739fd, MIT), vendored unmodified in `vllm_exl3_plugin/csrc/exl3/`
(`VENDORED.md`). Build with `VLLM_EXL3_BUILD=1`; without it the package is pure Python.

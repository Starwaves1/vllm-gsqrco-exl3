#!/bin/bash
# EXL3 phase 1, part A (box prep, CPU/disk/network only; never touches the GPU: every Python
# step runs with CUDA_VISIBLE_DEVICES=""). Idempotent; one subcommand per stage, in order:
#   00-prep.sh venv     /workspace/venv-main = env/prod-main-freeze.txt; packages the 0.27.1
#                       venv already has at the same version are hard-linked (venv-seed.py),
#                       the rest come from PyPI / the FlashInfer cu130 index
#   00-prep.sh overlay  Starwaves1/vllm qwen38/main overlay (pins.sh) onto the wheel, verified
#   00-prep.sh plugins  CUDA 13.0 toolchain; GGUF plugin (_C_gguf, VLLM_GGUF_BUILD_LCPP=1) and
#                       EXL3 plugin (_C_exl3), both editable into venv-main, sm86, MAX_JOBS<=8
#   00-prep.sh ref      exllamav3 d3739fd in /workspace/ref/exllamav3, /workspace/venv-exl3ref
#                       (the parity reference: same torch 2.13.0 cu130, exllamav3_ext built for
#                       sm86, EXL3_INT8_GEMV forced to 0 at interpreter start as the shim does)
#   00-prep.sh model    erlidev/Swift-1.5-Qwen3.8-27B-EXL3 @ SC_3.50bpw_H4_V6 (pins.sh EXL3_REV)
#                       into $EXL3_MODEL, parallel ranges, every file checked against the Hub;
#                       `model --alt`: turboderp/Qwen3.8-27B-exl3 @ 3.50bpw (the A/B)
#   00-prep.sh check    import checks for both venvs, ops listed, tests collected (no GPU)
#   00-prep.sh postsoak after the 24 h soak job has finished: delete the soak's leftover KV
#                       state (fs tier /workspace/kvtier, 28 GB; /dev/shm CPU-tier mmap), which
#                       is what makes room for the model. Refuses while any gpuq job or vLLM runs,
#                       and without CONFIRM_DELETE_SOAK_KV=1 (Garrett's go)
# Logs: /workspace/logs/exl3/00-prep-<stage>.log (the caller redirects).
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/pins.sh"
WT=${WT:-/workspace/wt-exl3}
V=${GSQ_VENV_MAIN:-/workspace/venv-main}
R=${EXL3_REF_VENV:-/workspace/venv-exl3ref}
M=${EXL3_MODEL:-/workspace/models/Swift-1.5-Qwen3.8-27B-exl3-SC_3.50bpw_H4_V6}
if [ "${2:-}" = --alt ]; then
  EXL3_REPO=$EXL3_ALT_REPO EXL3_REV=$EXL3_ALT_REV M=/workspace/models/Qwen3.8-27B-exl3-3.50bpw
fi
OLD=/workspace/gsq-vllm/.venv                     # 0.27.1 venv (read-only here: hard-link source)
L=/workspace/logs/exl3; mkdir -p "$L"
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/uv-cache} CUDA_VISIBLE_DEVICES=""
export MAX_JOBS=${MAX_JOBS:-8}
PY312=/.uv/python_install/cpython-$PY_VERSION-linux-x86_64-gnu/bin/python3.12
[ -x "$PY312" ] || PY312=$PY_VERSION
MIN_FREE_GB=${MIN_FREE_GB:-5}                     # never fill the disk under the running soak
die() { echo "prep: $*" >&2; exit 1; }
free_gb() { df --output=avail -B1G /workspace | tail -1 | tr -d ' '; }
need_free() { [ "$(free_gb)" -ge $(($1 + MIN_FREE_GB)) ] || die "needs $1 GB + $MIN_FREE_GB GB margin, /workspace has $(free_gb) GB free"; }
STAGE=${1:-}; t0=$(date +%s); trap 'echo "[$STAGE] $(( $(date +%s) - t0 )) s, /workspace free $(free_gb) GB"' EXIT

case ${1:-} in
venv)
  need_free 4
  [ -x "$V/bin/python" ] || uv venv -q --python "$PY312" "$V"
  python3 "$S/venv-seed.py" "$OLD" "$V" "$WT/env/prod-main-freeze.txt"
  uv pip install --python "$V/bin/python" --link-mode=hardlink --no-deps \
    --extra-index-url "$FLASHINFER_INDEX" --index-strategy unsafe-best-match -r "$WT/env/prod-main-freeze.txt"
  uv pip freeze --python "$V/bin/python" > "$L/venv-main-freeze.txt"
  diff "$WT/env/prod-main-freeze.txt" <(grep -vE '^(gguf @ |gguf==|-e file://)' "$L/venv-main-freeze.txt") \
    && echo "venv-main = env/prod-main-freeze.txt ($(wc -l < "$WT/env/prod-main-freeze.txt") pins)" \
    || die "venv-main differs from env/prod-main-freeze.txt"
  du -sh "$V" ;;
overlay)
  src=/workspace/ref/vllm-src-main
  if [ ! -d "$src/.git" ]; then git init -q "$src"; git -C "$src" remote add origin "$VLLM_FORK_URL"; fi
  git -C "$src" cat-file -e "$VLLM_OVERLAY^{commit}" 2>/dev/null && git -C "$src" cat-file -e "$VLLM_BASE^{commit}" 2>/dev/null \
    || git -C "$src" fetch -q --filter=blob:none --depth=64 origin "$VLLM_OVERLAY"
  "$S/overlay-vllm.sh" "$src" "$V/lib/python3.12/site-packages"
  "$V/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())' ;;
plugins)
  need_free 2
  cd "$WT"
  [ "$(git -C /workspace/llama.cpp rev-parse HEAD)" = "$LLAMA_COMMIT" ] || die "/workspace/llama.cpp is not $LLAMA_TAG"
  # same pinned CUDA 13.0 toolchain as /workspace/gsq-vllm's: hard-linked copy, then the (idempotent) setup
  if [ ! -x build/cu130/nvidia/cu13/bin/nvcc ] && [ -x /workspace/gsq-vllm/build/cu130/nvidia/cu13/bin/nvcc ]; then
    mkdir -p build; cp -al /workspace/gsq-vllm/build/cu130 build/cu130
  fi
  PY=$V/bin/python tools/setup-cuda-toolchain.sh
  uv pip install --python "$V/bin/python" --no-deps --link-mode=copy /workspace/llama.cpp/gguf-py
  t=$(date +%s)
  GSQ_VENV=$V VLLM_GGUF_BUILD_LCPP=1 tools/build-plugin.sh > "$L/build-gguf.log" 2>&1 || { tail -30 "$L/build-gguf.log"; die "GGUF plugin build failed"; }
  echo "_C_gguf built in $(( $(date +%s) - t )) s"; t=$(date +%s)
  GSQ_VENV=$V GSQ_PLUGIN=plugin-exl3 tools/build-plugin.sh > "$L/build-exl3.log" 2>&1 || { tail -30 "$L/build-exl3.log"; die "EXL3 plugin build failed"; }
  echo "_C_exl3 built in $(( $(date +%s) - t )) s"
  ls -la plugin/vllm_gguf_plugin/_C_gguf*.so plugin-exl3/vllm_exl3_plugin/_C_exl3*.so
  (cd plugin-exl3/vllm_exl3_plugin/csrc/exl3 && sed -n '/^```$/,/^```$/p' VENDORED.md | grep -E '^[0-9a-f]{64} ' | sha256sum -c --quiet) \
    && echo "vendored exllamav3 sha256: OK" || die "vendored exllamav3 sources differ from VENDORED.md"
  (cd plugin/vllm_gguf_plugin/csrc/lcpp && sed -n '/^```$/,/^```$/p' VENDORED.md | grep -E '^[0-9a-f]{64} ' | sha256sum -c --quiet) \
    && echo "vendored llama.cpp sha256: OK" || die "vendored llama.cpp sources differ from VENDORED.md"
  uv pip freeze --python "$V/bin/python" > "$L/venv-main-freeze-plugins.txt"
  norm() { sed 's#@ file://.*#@ file#; s#^-e file://.*#-e file#' "$1"; }
  diff <(norm "$WT/env/gsq-main-freeze.txt") <(grep -v 'plugin-exl3' "$L/venv-main-freeze-plugins.txt" | norm /dev/stdin) \
    && echo "venv-main = env/gsq-main-freeze.txt + vllm-exl3-plugin (editable)" || echo "WARNING: freeze differs from env/gsq-main-freeze.txt (above)" ;;
ref)
  need_free 2
  E=/workspace/ref/exllamav3
  if [ ! -d "$E/.git" ]; then git clone -q "$EXL3_URL" "$E"; fi
  git -C "$E" cat-file -e "$EXL3_COMMIT^{commit}" 2>/dev/null || git -C "$E" fetch -q origin
  git -C "$E" checkout -q --detach "$EXL3_COMMIT"
  [ -z "$(git -C "$E" status --porcelain --untracked-files=no)" ] || die "$E has local changes"
  [ -x "$R/bin/python" ] || uv venv -q --python "$PY312" "$R"
  # exllamav3's requirements.txt + build deps, resolved against production's pins so torch
  # 2.13.0 (cu130) and its nvidia libraries are the same files as venv-main's (hard-linked)
  grep -v '^vllm @' "$WT/env/prod-main-freeze.txt" > /workspace/ref/exl3ref-constraints.txt
  uv pip compile -q --python "$R/bin/python" --no-header -c /workspace/ref/exl3ref-constraints.txt \
    "$E/requirements.txt" <(printf 'setuptools\nwheel\npip\n') -o /workspace/ref/exl3ref-lock.txt
  python3 "$S/venv-seed.py" "$V" "$R" /workspace/ref/exl3ref-lock.txt
  uv pip install --python "$R/bin/python" --link-mode=hardlink --no-deps -r /workspace/ref/exl3ref-lock.txt
  T=$WT/build/cu130/nvidia/cu13
  [ -x "$T/bin/nvcc" ] || die "run 'plugins' first (CUDA 13.0 toolchain in $WT/build/cu130)"
  inc=$("$R/bin/python" -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13", "include"))')
  if ! (cd / && "$R/bin/python" -c 'import torch, exllamav3_ext') 2>/dev/null; then
    t=$(date +%s)
    ( export CUDA_HOME=$T TORCH_CUDA_ARCH_LIST=8.6 CUDA_INC_PATH=$inc PATH=$T/bin:$R/bin:$PATH
      cd "$E" && "$R/bin/python" -m pip install --no-build-isolation --no-deps --no-cache-dir -v . ) > "$L/build-exllamav3.log" 2>&1 \
      || { tail -30 "$L/build-exllamav3.log"; die "exllamav3 build failed"; }
    echo "exllamav3_ext built in $(( $(date +%s) - t )) s"
    grep -q "arch=compute_86,code=sm_86" "$L/build-exllamav3.log" || die "no sm_86 gencode in the exllamav3 build log"
    rm -rf "$E/build"
  fi
  # exllamav3 d3739fd turns its int8-activation GEMV on when EXL3_INT8_GEMV is unset; the plugin's
  # shim forces it off (setenv(.., "0", 1) at library load), so the reference does the same
  echo 'import os; os.environ["EXL3_INT8_GEMV"] = "0"  # gsq-vllm EXL3 parity reference (00-prep.sh ref)' \
    > "$R/lib/python3.12/site-packages/zz_exl3ref_int8_gemv_off.pth"
  (cd / && "$R/bin/python" -c '
import os, importlib.util, torch, exllamav3, exllamav3_ext
from exllamav3.version import __version__ as v
assert os.environ["EXL3_INT8_GEMV"] == "0"
assert v == "1.5.3", v
print("exllamav3", v, "ext", importlib.util.find_spec("exllamav3_ext").origin)
print("torch", torch.__version__, "cuda", torch.version.cuda, "EXL3_INT8_GEMV", os.environ["EXL3_INT8_GEMV"])')
  uv pip freeze --python "$R/bin/python" > "$L/venv-exl3ref-freeze.txt"
  du -sh "$R" ;;
model)
  size=$(curl -sSfL "https://huggingface.co/api/models/$EXL3_REPO/tree/$EXL3_REV?recursive=true" \
    | python3 -c 'import json,sys; print(sum((f.get("lfs") or {}).get("size", f["size"]) for f in json.load(sys.stdin) if f["type"] == "file"))')
  have=$(du -sb "$M" 2>/dev/null | cut -f1 || echo 0); have=${have:-0}
  need_free $(( (size - have) / 1073741824 + 1 ))
  DL_REV=$EXL3_REV DL_CHUNKS=${DL_CHUNKS:-16} "$S/dl.sh" repo "$EXL3_REPO" "$M"
  { echo "$EXL3_REPO @ $EXL3_REV -> $M"
    (cd "$M" && for f in $(find . -type f ! -name '*.ok' | sort); do printf '%s\t%s\t%s\n' "${f#./}" "$(stat -c %s "$f")" "$(cat "$f.ok")"; done)
    echo "total $(du -sb --exclude='*.ok' "$M" | cut -f1) bytes"; } | tee "$L/model-files-$(basename "$M").txt" ;;
check)
  cd "$WT"
  "$V/bin/python" - <<'PY'
import torch, vllm
import vllm_gguf_plugin._C_gguf  # noqa: F401
import vllm_exl3_plugin._C_exl3  # noqa: F401
ops = sorted(n for n in torch._C._dispatch_get_all_op_names() if n.startswith(("_C_gguf::", "_C_exl3::")))
g = [o for o in ops if o.startswith("_C_gguf::")]; e = [o for o in ops if o.startswith("_C_exl3::")]
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda initialized:", torch.cuda.is_initialized())
print(len(g), "_C_gguf ops:", " ".join(o.split("::")[1] for o in g))
print(len(e), "_C_exl3 ops:", " ".join(o.split("::")[1] for o in e))
assert len(g) == 15 and "lcpp_mul_mat_q" in " ".join(g), "expected 15 _C_gguf ops incl. Route L"
assert {"exl3_gemm", "exl3_dequant", "exl3_had_r_128", "exl3_hgemm", "exl3_warmup"} <= {o.split("::")[1] for o in e}
import ctypes; libc = ctypes.CDLL(None); libc.getenv.restype = ctypes.c_char_p
print("EXL3_INT8_GEMV after _C_exl3 load (C getenv; the shim setenv()s it):", libc.getenv(b"EXL3_INT8_GEMV"))
assert libc.getenv(b"EXL3_INT8_GEMV") == b"0"
from importlib.metadata import entry_points
print("vllm.general_plugins:", sorted(ep.name for ep in entry_points(group="vllm.general_plugins")))
PY
  GSQ_VENV=$V tools/pytest -p no_gpu tests/gpu --collect-only -q 2>&1 | tail -3
  GSQ_VENV=$V tools/pytest -p no_gpu tests/gpu --collect-only -q -k exl3 2>&1 | tail -1 ;;
postsoak)
  [ "${CONFIRM_DELETE_SOAK_KV:-0}" = 1 ] || die "deletes the soak's KV state: needs CONFIRM_DELETE_SOAK_KV=1"
  # (run as a gpuq job, /workspace/gpuq/running names this job, so look for the soak by name)
  ! grep -q final-soak-24h /workspace/gpuq/running 2>/dev/null || die "the soak job is still running: $(cat /workspace/gpuq/running)"
  ! pgrep -f "VLLM::EngineCore|bin/vllm serve|soak_load.py|bench/soak.sh" > /dev/null || die "a vLLM server or soak process is running"
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || die "the GPU has compute processes"
  ls /workspace/gpuq/out/*final-soak-24h.status > /dev/null 2>&1 || die "the soak job has no exit status yet"
  SOAK_RUN=${SOAK_RUN:-/workspace/runs/20260930-100520-soak-gsq}   # results: never touched here
  [ -s "$SOAK_RUN/report.json" ] || die "$SOAK_RUN/report.json missing: the soak did not finish its report"
  echo "soak job exit status: $(cat /workspace/gpuq/out/*final-soak-24h.status); report: $SOAK_RUN/report.json"
  du -sh /workspace/kvtier 2>/dev/null || true; ls -la /dev/shm/vllm_offload_*.mmap 2>/dev/null || true
  rm -rf /workspace/kvtier; rm -f /dev/shm/vllm_offload_*.mmap ;;
*) sed -n '2,24p' "$0"; exit 2 ;;
esac

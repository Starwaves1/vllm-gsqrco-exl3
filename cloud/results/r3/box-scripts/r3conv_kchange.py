"""R3 incident: does causal_conv1d_update stay correct when the speculative query length changes
between steps? CPU only (Triton interpreter), seconds. Each argument is a copy of
vllm/model_executor/layers/mamba/ops/causal_conv1d.py (e.g. upstream main, the overlay, a patched one).

A request verifies L = K + 1 tokens per step; num_accepted (1..L) from the previous step selects
the conv window. With num_speculative_tokens_per_batch_size, K can drop between steps (5 -> 3 when
the batch grows from 4 to 5), so num_accepted can exceed this step's L. Reference: a plain causal
conv over the accepted token sequence.
  CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python r3conv_kchange.py FILE [FILE...]
Exit code 1 if any plan is wrong for any file."""
import importlib.util
import os
import sys

import torch

W, D, S = 4, 16, 5           # conv width (Qwen3.5 GDN), channels, num_speculative_tokens (config max)
STATE = W - 1 + S            # conv state width, as MambaStateShapeCalculator.gated_delta_net_state_shape
PLANS = {                    # (query length L, num_accepted after the step) per step
    "fixed K=3": [(4, 4), (4, 4), (4, 2), (4, 4)],
    "K 5->3, accept 6 then L=4": [(6, 6), (4, 3), (4, 4), (4, 1)],
    "K 5->3, accept 4 then L=4": [(6, 4), (4, 4), (4, 2)],
    "K 3->2, accept 4 then L=3": [(4, 4), (3, 3), (3, 1)],
    "K 3->5, accept 4 then L=6": [(4, 4), (6, 6), (6, 2)],
}


def load(path):
    src = open(path).read().split("\nif current_platform.is_cpu():")[0]  # keep the Triton path on CPU
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(os.path.basename(path), loader=None))
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def run(mod, plan, w, b, hist, xs):
    def ref(seq, t):
        y = (seq[t - W + 1:t + 1].T * w).sum(-1) + b
        return y * torch.sigmoid(y)

    st = torch.zeros(3, D, STATE)
    st[1, :, :W - 1] = hist.T
    seq, num_acc, pos, errs = list(hist), 1, 0, []
    for L, acc in plan:
        x = xs[pos:pos + L]
        out = mod.causal_conv1d_update(
            x.clone(), st, w, b, "silu",
            conv_state_indices=torch.tensor([1], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([num_acc], dtype=torch.int32),
            query_start_loc=torch.tensor([0, L], dtype=torch.int32),
            max_query_len=S + 1)  # spec_state_indices_tensor.size(-1), as qwen_gdn_linear_attn.py passes it
        full = torch.stack(seq + list(x))
        want = torch.stack([ref(full, len(seq) + i) for i in range(L)])
        errs.append(float((out - want).abs().max()))
        seq += list(x[:acc]); pos += acc; num_acc = acc
    return errs


def main():
    torch.manual_seed(0)
    w, b, hist, xs = torch.randn(D, W), torch.randn(D), torch.randn(W - 1, D), torch.randn(40, D)
    bad = 0
    for path in sys.argv[1:]:
        mod = load(os.path.abspath(path))
        for name, plan in PLANS.items():
            errs = run(mod, plan, w, b, hist, xs)
            ok = all(e < 1e-4 for e in errs)
            bad += not ok
            print(f"{path}: {name:28s} {'ok ' if ok else 'BAD'} max err per step {['%.2g' % e for e in errs]}")
        for n in (0, S + 2):  # counts outside 1..S+1 must still be rejected (zero output, state untouched)
            st = torch.zeros(3, D, STATE)
            out = mod.causal_conv1d_update(
                xs[:4].clone(), st, w, b, "silu", conv_state_indices=torch.tensor([1], dtype=torch.int32),
                num_accepted_tokens=torch.tensor([n], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 4], dtype=torch.int32), max_query_len=S + 1)
            print(f"{path}: num_accepted={n}: output zeroed {bool((out == 0).all())}, state untouched {bool((st == 0).all())}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

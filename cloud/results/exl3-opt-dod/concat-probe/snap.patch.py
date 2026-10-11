# applies the snapshot instrumentation to a debug worktree's plugin (never committed to the plugin)
import pathlib, sys
root = pathlib.Path(sys.argv[1]) / "plugin-exl3/vllm_exl3_plugin"
p = root / "plugin.py"; s = p.read_text()
if "EXL3DBG" not in s:
    s += '''

import os as _os
if _os.environ.get("EXL3_DBG_SNAP"):
    import torch as _t
    _t.cuda.memory._record_memory_history(max_entries=500000, stacks="python")
'''
    p.write_text(s)
p = root / "quantization/draft_head.py"; s = p.read_text()
anchor = "        w = layer.weight.data\n"
if "EXL3DBG" not in s:
    s = s.replace(anchor, anchor + '''        import os, pickle
        if os.environ.get("EXL3_DBG_SNAP"):
            print("EXL3DBG draft head entry: allocated %.2f GiB reserved %.2f GiB" % (torch.cuda.memory_allocated() / 2**30,
                  torch.cuda.memory_reserved() / 2**30), flush=True)
            with open(os.environ["EXL3_DBG_SNAP"], "wb") as f:
                pickle.dump(torch.cuda.memory._snapshot(), f)
            raise SystemExit("EXL3DBG snapshot taken")
''', 1)
    p.write_text(s)
print("patched", root)

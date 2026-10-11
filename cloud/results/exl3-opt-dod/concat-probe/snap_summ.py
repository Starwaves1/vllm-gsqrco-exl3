import pickle, sys, collections
s = pickle.load(open(sys.argv[1], "rb"))
pools = collections.Counter(); sites = collections.Counter(); tot = 0
for seg in s["segments"]:
    for b in seg["blocks"]:
        if b["state"] != "active_allocated":
            continue
        tot += b["size"]; pools[str(seg.get("segment_pool_id"))] += b["size"]
        fr = [f for f in b.get("frames", []) if "site-packages/torch" not in f["filename"]]
        key = " < ".join(f"{f['filename'].split('/')[-1]}:{f['line']}:{f['name']}" for f in fr[:4]) or "(no frames)"
        sites[key] += b["size"]
G = 2**30
print(f"active {tot/G:.2f} GiB")
for k, v in pools.most_common(): print(f"  pool {k}: {v/G:.2f} GiB")
for k, v in sites.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 15): print(f"{v/G:7.3f} GiB  {k}")

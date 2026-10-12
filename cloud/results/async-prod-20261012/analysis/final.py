from an import *
import datetime, bisect, calendar
def ts(l):
    d=datetime.datetime.strptime(l.split(',')[0],'%Y/%m/%d %H:%M:%S.%f'); return d.timestamp()
def smi(d):
    out=[]
    for r in load(d,'smi'):
        p=[x.strip() for x in r['l'].split(',')]
        try: out.append((ts(r['l']),float(p[5]),float(p[1]),float(p[2])))
        except Exception: pass
    out.sort(); return out
def attach(ws,sm):
    T=[x[0] for x in sm]
    for w in ws:
        i=bisect.bisect_left(T,w['t']); j=bisect.bisect_left(T,w['t']+w['dt'])
        seg=sm[i:j]
        w['util']=sum(x[1] for x in seg)/len(seg) if seg else None
        w['pw']=sum(x[2] for x in seg)/len(seg) if seg else None
        w['smclk']=sum(x[3] for x in seg)/len(seg) if seg else None
def clean(ws): return [w for w in ws if w['minrun']==w['maxrun'] and w['minrun']>0]
bs=series(load(B)); ws=windows(bs); attach(ws,smi(B))
as_=series(load(A)); wa=windows(as_); attach(wa,smi(A))
cut=datetime.datetime(2026,10,12,1,27,57).timestamp()
ws=[w for w in ws if w['t']+w['dt']<cut]
t0=datetime.datetime(2026,10,12,1,53,43).timestamp()
wa=[w for w in wa if w['t']>=t0]
nowin=lambda lo,hi:[w for w in ws if lo<=w['t']<hi]
H=lambda h,m=0:datetime.datetime(2026,10,11,0,0).timestamp()+0  # unused
near=[w for w in ws if w['t']>=cut-3*3600]
sets={'before-all':ws,'before-last3h':near,'after(async)':wa}
def line(name,fn,ws_):
    return summ([fn(w) for w in ws_ if fn(w) is not None])
fast=lambda w: w['itl'] is not None and w['itl']<0.1
out=[]
P=out.append
P('## Per-step time (ms/engine step = 1000*dt/steps, decode-only windows, "fast mode": step<100ms)')
for r in (1,2):
  for b in ('<30','30-60','>60'):
    P(f'\n### running={r} kv={b}')
    for n,S in sets.items():
        xs=[w for w in clean(S) if w['minrun']==r and kvbin(w['kv'])==b and decode_only(w)]
        xf=[w for w in xs if 1000/w['sps']<100]
        P(f"{n:15s} step_ms {summ([1000/w['sps'] for w in xf])} | slow-mode frac {1-len(xf)/max(1,len(xs)):.2f} of {len(xs)} | tok/s/req {summ([w['tps']/r for w in xf])} | tok/draft {summ([w['tok_per_draft'] for w in xf if w['tok_per_draft']])} | util% {summ([w['util'] for w in xf if w['util'] is not None])} | W {summ([w['pw'] for w in xf if w['pw'] is not None])}")
P('\n## All windows (incl. prefill), constant running count')
for r in (1,2):
  for b in ('<30','30-60','>60'):
    P(f'\n### running={r} kv={b}')
    for n,S in sets.items():
        xs=[w for w in clean(S) if w['minrun']==r and kvbin(w['kv'])==b]
        P(f"{n:15s} tok/s/req {summ([w['tps']/r for w in xs])} | steps/s {summ([w['sps'] for w in xs])} | util% {summ([w['util'] for w in xs if w['util'] is not None])} | prefill-window frac {1-sum(decode_only(w) for w in xs)/max(1,len(xs)):.2f}")
P('\n## Overall (all running>0 windows)')
for n,S in sets.items():
    xs=[w for w in S if w['run']>0]
    d=[w for w in xs if decode_only(w)]
    P(f"{n:15s} windows {len(xs)} decodeonly {len(d)} | accept rate {summ([w['accrate'] for w in xs if w['accrate'] is not None])} | util% all {summ([w['util'] for w in xs if w['util'] is not None])} decode {summ([w['util'] for w in d if w['util'] is not None])} | prefill-window share {1-len(d)/max(1,len(xs)):.2f} | kv {summ([w['kv'] for w in xs])}")
    # token-weighted acceptance
    a=sum(w['acc'] for w in xs); dr=sum(w['dr'] for w in xs); P(f"    pooled accept {a/dr:.3f}  tokens/draft {(a+sum(w['drafts'] for w in xs))/sum(w['drafts'] for w in xs):.3f}")
print('\n'.join(out))
print('\n## SUPP: decode-only windows, bin by rounded mean running (not necessarily constant)')
for r in (1,2):
  for b in ('30-60','>60'):
    print(f'\n### run~{r} kv={b}')
    for n,S in sets.items():
        xs=[w for w in S if round(w['run'])==r and kvbin(w['kv'])==b and decode_only(w)]
        xf=[w for w in xs if 1000/w['sps']<100]
        print(f"{n:15s} step_ms {summ([1000/w['sps'] for w in xf])} | all-decode step_ms {summ([1000/w['sps'] for w in xs])} | util {summ([w['util'] for w in xs if w['util'] is not None])}")

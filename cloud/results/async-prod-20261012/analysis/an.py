from lib import *
import statistics as st, sys, datetime
W=int(os.environ.get("W","10"))
def windows(s):
    out=[]; i=0
    while i<len(s):
        j=i
        while j<len(s) and s[j][0]-s[i][0]<W: j+=1
        if j>=len(s): break
        a,b=s[i],s[j]; dt=b[0]-a[0]
        if dt>W*1.5: i=j; continue
        # mean running/kv over the window
        seg=s[i:j+1]
        run=sum(x[1][0] for x in seg)/len(seg); kv=sum(x[1][1] for x in seg)/len(seg)
        minrun=min(x[1][0] for x in seg); maxrun=max(x[1][0] for x in seg)
        d=lambda n: b[1][n]-a[1][n]
        if min(d(2),d(3),d(5))<0: i=j; continue  # counter reset
        steps=d(5)
        if steps<5: i=j; continue
        w=dict(maxrun=maxrun,t=a[0],dt=dt,run=run,minrun=minrun,kv=kv,tps=d(2)/dt,sps=steps/dt,
          itl=(d(4)/d(3)) if d(3)>0 else None, tokstep=d(6)/steps,
          small=d(7)/steps, drafts=d(10),acc=d(8),dr=d(9),
          prefill=d(11) if b[1][11] is not None and a[1][11] is not None else None)
        w['accrate']=w['acc']/w['dr'] if w['dr']>0 else None
        w['tok_per_draft']=(w['acc']+w['drafts'])/w['drafts'] if w['drafts']>0 else None
        out.append(w); i=j
    return out
def q(x,p):
    x=sorted(x); k=(len(x)-1)*p; f=int(k); c=min(f+1,len(x)-1); return x[f]+(x[c]-x[f])*(k-f)
def summ(x):
    if not x: return 'n=0'
    return f'{q(x,.5):.4g} [{q(x,.25):.4g}-{q(x,.75):.4g}] n={len(x)}'
def kvbin(kv): return '<30' if kv<.3 else ('30-60' if kv<.6 else '>60')
def decode_only(w): return w['small']>=0.99 and (w['prefill'] or 0)==0

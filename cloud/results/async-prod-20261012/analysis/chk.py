from an import *
import datetime
s=series(load(A)); 
first=next((x[0] for x in s if x[1][0]>0),None)
f=lambda t: datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S')
print('samples',len(s),'last',f(s[-1][0]),'first running>0:',f(first) if first else None, 'frac busy since:', (sum(x[1][0]>0 for x in s if x[0]>=first)/max(1,sum(1 for x in s if x[0]>=first))) if first else None)

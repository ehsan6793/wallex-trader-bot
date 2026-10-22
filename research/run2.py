from lab import *
import itertools, io, contextlib
rows=[]
for tf,bh in (("15min",4),("1h",1),("5min",12)):
    fr=load(tf)
    for drop,look,tp,sl,hold in itertools.product((-0.025,-0.03,-0.035,-0.04),(1,2,3),(0.015,0.02,0.03),(0.03,0.04),(4,8)):
        allt=[]
        for s,df in fr.items():
            sig=np.log(df.close).diff(max(1,int(look*bh)))<=drop
            allt+=[(t,r) for t,r in simulate(df,sig,tp=tp,sl=sl,max_bars=int(hold*bh))]
        if len(allt)<8: continue
        t=pd.DataFrame(allt,columns=["t","r"]); end=max(df.index.max() for df in fr.values())
        w=[t[(t.t>end-pd.Timedelta(days=a))&(t.t<=end-pd.Timedelta(days=b))].r.sum()*100 for a,b in ((21,14),(14,7),(7,0))]
        rows.append(dict(tf=tf,drop=drop,look=look,tp=tp,sl=sl,hold=hold,n=len(t),avg=t.r.mean()*100,win=(t.r>0).mean(),w1=w[0],w2=w[1],w3=w[2],minw=min(w),pos=sum(x>0 for x in w)))
d=pd.DataFrame(rows)
print(d[d.pos==3].sort_values(["minw","avg"],ascending=False).head(25).round(2).to_string(index=False))
print("configs:",len(d)," all-3-weeks-positive:",(d.pos==3).sum())

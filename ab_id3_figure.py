import numpy as np, pandas as pd, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ab_id3_experiment import load, equal_freq_bins, supervised_bins, info_gain, CONT, CAT
from decision_trees import ID3Classifier, C45Classifier, CARTClassifier

df0 = load(); y_all = df0["income"].to_numpy(np.int64)
rng = np.random.default_rng(42); idx = rng.permutation(len(df0))[:20000]
df = df0.iloc[idx].reset_index(drop=True); y = y_all[idx]
cut = int(len(df)*0.7); tr = np.arange(cut); te = np.arange(cut, len(df))

def id3_view(mode):
    cols=[]
    for c in CONT:
        if mode=="baseline": cols.append(equal_freq_bins(df[c],5))
        else:
            mb = 2 if mode=="ab2" else 5
            vals=df[c].astype(float).to_numpy(); trv=df[c].iloc[tr].astype(float).to_numpy()
            cand=np.unique(np.percentile(trv,np.linspace(10,90,9))); chosen=[]
            for _ in range(mb-1):
                best=None
                for t in cand:
                    if t in chosen: continue
                    lab=np.digitize(trv,np.sort(np.array(chosen+[t]))); ig=info_gain(lab,y[tr],2)
                    if best is None or ig>best[0]: best=(ig,t)
                if best is None or best[0]<=0: break
                chosen.append(best[1])
            edges=np.sort(np.array(chosen)) if chosen else np.array([np.median(trv)])
            cols.append(np.digitize(vals,edges).astype(str))
    for c in CAT: cols.append(df[c].astype(str).to_numpy())
    return np.column_stack(cols).astype(object), ["categorical"]*(len(CONT)+len(CAT))

res={}
for name,mode in [("ID3\n(baseline)","baseline"),("AB-ID3\n(2 bins)","ab2"),("AB-ID3\n(5 bins)","ab5")]:
    X,ty=id3_view(mode); t0=time.perf_counter()
    m=ID3Classifier(max_depth=15,min_samples_split=10).fit(X[tr],y[tr],ty); tt=time.perf_counter()-t0
    res[name]=(tt,(m.predict(X[te])==y[te]).mean())
# refs
cn=[df[c].astype(float).to_numpy().astype(object) for c in CONT]; cc=[df[c].astype(str).to_numpy() for c in CAT]
Xc=np.column_stack(cn+cc).astype(object); tc=["numeric"]*len(CONT)+["categorical"]*len(CAT)
t0=time.perf_counter(); mc=C45Classifier(max_depth=15,min_samples_split=10).fit(Xc[tr],y[tr],tc)
res["C4.5"]=(time.perf_counter()-t0,(mc.predict(Xc[te])==y[te]).mean())
def le(c):
    v={x:i for i,x in enumerate(sorted(set(c)))}; return np.array([v[x] for x in c],dtype=float)
ca=[df[c].astype(float).to_numpy() for c in CONT]+[le(df[c].astype(str)) for c in CAT]
Xa=np.column_stack(ca).astype(object); ta=["numeric"]*(len(CONT)+len(CAT))
t0=time.perf_counter(); mca=CARTClassifier(max_depth=15,min_samples_split=10).fit(Xa[tr],y[tr],ta)
res["CART"]=(time.perf_counter()-t0,(mca.predict(Xa[te])==y[te]).mean())

names=list(res); accs=[res[n][1] for n in names]; times=[res[n][0] for n in names]
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.6))
colors=["#888","#2a9d8f","#9cc5bd","#e76f51","#e9a23b"]
b1=a1.bar(names,accs,color=colors); a1.set_ylim(0.78,0.85); a1.set_ylabel("Test accuracy")
a1.set_title("Accuracy: AB-ID3 recovers ID3's lost accuracy"); a1.grid(axis="y",alpha=0.3)
for b,v in zip(b1,accs): a1.text(b.get_x()+b.get_width()/2,v+0.001,f"{v:.3f}",ha="center",fontsize=9)
b2=a2.bar(names,times,color=colors); a2.set_ylabel("Training time (s)")
a2.set_title("Speed: AB-ID3 keeps ID3's efficiency"); a2.grid(axis="y",alpha=0.3)
for b,v in zip(b2,times): a2.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontsize=9)
plt.tight_layout(); plt.savefig("ab_id3_comparison.png",dpi=150)
print("saved"); [print(n, f"{res[n][0]:.3f}s", f"{res[n][1]:.3f}") for n in names]
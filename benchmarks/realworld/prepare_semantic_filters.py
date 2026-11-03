#!/usr/bin/env python3
"""Generate embedding-correlated synthetic eligibility labels.

These are synthetic semantic clusters, NOT source business metadata.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--clusters",type=int,default=128)
    p.add_argument("--seed",type=int,default=3311)
    p.add_argument("--batch-size",type=int,default=4096)
    p.add_argument("--output",type=Path,default=None)
    a=p.parse_args(); d=a.dataset_dir
    m=json.loads((d/"manifest.json").read_text(encoding="utf-8"))
    n,nq,dim=int(m["corpus_rows"]),int(m["query_rows"]),int(m["dimension"])
    x=np.memmap(d/"corpus.f32",dtype="<f4",mode="r",shape=(n,dim))
    q=np.memmap(d/"queries.f32",dtype="<f4",mode="r",shape=(nq,dim))
    if not 2 <= a.clusters <= n: raise SystemExit("clusters must be in [2, corpus_rows]")
    km=MiniBatchKMeans(n_clusters=a.clusters,random_state=a.seed,batch_size=a.batch_size,n_init=3)
    km.fit(x)
    labels=km.predict(x).astype("<u4")
    qlabels=km.predict(q).astype("<u4")
    counts=np.bincount(labels,minlength=a.clusters)
    centers=km.cluster_centers_.astype(np.float32)
    out=a.output or d/"semantic_filters"; out.mkdir(parents=True,exist_ok=True)
    cp=out/"corpus_cluster_ids.u32"; qp=out/"query_cluster_ids.u32"
    labels.tofile(cp); qlabels.tofile(qp)
    fp=out/"query_cluster_filters.jsonl"; nlines=0
    with fp.open("w",encoding="utf-8",newline="\n") as f:
        for qi in range(nq):
            delta=centers-q[qi]
            d2=np.einsum("ij,ij->i",delta,delta)
            nearest=np.argsort(d2,kind="stable")
            for target in (0.01,0.10,1.00):
                chosen=[]; total=0; goal=max(1,int(round(n*target)))
                for cid in nearest:
                    chosen.append(int(cid)); total+=int(counts[cid])
                    if total>=goal: break
                rec={"query_row":qi,"target_selectivity":target,
                     "query_cluster":int(qlabels[qi]),"eligible_cluster_ids":chosen,
                     "eligible_count":total,"actual_selectivity":total/n}
                f.write(json.dumps(rec,separators=(",",":"))+"\n"); nlines+=1
    meta={"kind":"synthetic_embedding_correlated_filter_labels",
          "not_real_business_metadata":True,
          "method":"MiniBatchKMeans fit on corpus only; held-out queries assigned to frozen centroids",
          "clusters":a.clusters,"seed":a.seed,"corpus_rows":n,"query_rows":nq,"dimension":dim,
          "filter_records":nlines,
          "files":{p.name:{"sha256":sha256(p),"bytes":p.stat().st_size} for p in (cp,qp,fp)},
          "cluster_size_min":int(counts.min()),"cluster_size_max":int(counts.max())}
    (out/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2))
if __name__=="__main__": main()

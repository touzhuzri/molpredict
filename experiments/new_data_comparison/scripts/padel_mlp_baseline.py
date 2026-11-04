"""补充基线：24 项 PaDEL 的简单 MLP（不是 A–E 主矩阵的一部分）。

用于判断“B 组 Transformer token 基线偏弱”是否只是架构不适合表格特征。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn

ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT/'src'))
from molpredict.v3_data import NumericStandardizer, load_cache_arrays, load_rows, load_split
from molpredict.v2 import best_threshold, binary_metrics

torch.set_num_threads(4)
arrays=load_cache_arrays(ROOT/'experiments/v2/cache',ROOT/'experiments/v3/cache',[r['inchikey'] for r in load_rows()[2]])
_,labels,rows=load_rows()
X=arrays['descriptors']
out=ROOT/'experiments/new_data_comparison/ablations/padel_mlp'; out.mkdir(parents=True,exist_ok=True)

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    split=load_split(seed); st=NumericStandardizer.fit(X,split['train'])
    Z=st.transform(X).astype(np.float32); y=labels.astype(np.float32)
    def t(idx): return torch.tensor(Z[idx]), torch.tensor(y[idx])
    tr,va,te=[t(split[k]) for k in ('train','val','test')]
    model=nn.Sequential(nn.Linear(24,64),nn.GELU(),nn.Dropout(.3),nn.Linear(64,32),nn.GELU(),nn.Dropout(.3),nn.Linear(32,1))
    pos=float(y[split['train']].sum()); neg=len(split['train'])-pos
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos]))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=(-1,None,0,0)
    for ep in range(1,201):
        model.train(); opt.zero_grad(); loss=crit(model(tr[0]).squeeze(-1),tr[1]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp=torch.sigmoid(model(va[0]).squeeze(-1)).numpy()
        auc=roc_auc_score(va[1].numpy().astype(int),vp)
        if auc>best[0]: best=(auc,{k:v.clone() for k,v in model.state_dict().items()},ep,0)
        else:
            best=(best[0],best[1],best[2],best[3]+1)
            if best[3]>=20: break
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        vp=torch.sigmoid(model(va[0]).squeeze(-1)).numpy(); tp=torch.sigmoid(model(te[0]).squeeze(-1)).numpy()
    thr=best_threshold(va[1].numpy().astype(int),vp)
    m=binary_metrics(te[1].numpy().astype(int),tp,thr)
    return {'seed':seed,'val_auc':float(best[0]),'best_epoch':int(best[2]),'test_auc':m['roc_auc'],'test_pr_auc':m['pr_auc'],'test_f1':m['f1'],'test_mcc':m['mcc'],'threshold':float(thr)}

res=[run(s) for s in (42,43,44)]
(out/'metrics.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
print(json.dumps(res,indent=2))

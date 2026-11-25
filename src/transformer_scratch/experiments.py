"""Controlled CPU forward-pass experiments."""
import json
import platform
from pathlib import Path
import numpy as np
from .core import attention,causal_mask,padding_mask,sinusoidal,MultiHead,TransformerBlock
from .reference import ref_attention,ref_multi,ref_block

SEEDS=(11,22,33)
TITLES=['Scaled dot-product attention','Multi-head attention','Transformer block','Causal prefix invariance','Padding and fully masked rows','Positional information','Large-logit stability','Quadratic attention storage']


def run():
    import torch
    torch.set_num_threads(1)
    rows=[]
    def add(i,seed,c,metric,a,b,ref):
        rows.append(dict(experiment=f'EXP-{i:03}',seed=seed,configuration=c,metric=metric,ours=float(a),reference=float(b),reference_kind=ref))
    def err(a,b): return np.max(np.abs(a-b))
    for seed in SEEDS:
        r=np.random.default_rng(seed); q,k,v=r.normal(size=(3,1,2,6,4)); x=r.normal(size=(1,6,8))
        for mode,m in [('unmasked',None),('causal',causal_mask(6))]:
            add(1,seed,mode,'max absolute output discrepancy',err(attention(q,k,v,m)[0],ref_attention(q,k,v,m)),0,'PyTorch agreement ideal')
        for h in (1,2,4):
            m=MultiHead(8,h,r); a,w=m.forward(x); b,z=ref_multi(m,x); add(2,seed,f'heads={h}','max absolute output/weight discrepancy',max(err(a,b),err(w,z)),0,'PyTorch agreement ideal')
        block=TransformerBlock(8,2,16,r)
        for mode,m in [('unmasked',None),('causal',causal_mask(6))]:
            add(3,seed,mode,'max absolute output discrepancy',err(block.forward(x,m)[0],ref_block(block,x,m)),0,'PyTorch agreement ideal')
        changed=x.copy(); changed[:,3:]+=r.normal(0,3,size=(1,3,8))
        for mode,m in [('unmasked',None),('causal',causal_mask(6))]:
            add(4,seed,mode,'max prefix output change',err(block.forward(x,m)[0][:,:3],block.forward(changed,m)[0][:,:3]),0,'unchanged prefix ideal')
        q,k,v=r.normal(size=(3,2,2,4,4)); m=padding_mask([[True,False,True,False],[False]*4]); a,w=attention(q,k,v,m)
        add(5,seed,'padded + fully masked','max absolute output discrepancy',err(a,ref_attention(q,k,v,m)),0,'PyTorch agreement ideal')
        add(5,seed,'masked keys','maximum masked weight',np.max(np.where(m,0,w)),0,'zero masked mass ideal')
        x=r.normal(size=(1,5,8)); p=np.array([3,0,4,1,2]); pos=sinusoidal(5,8)
        for name,a,b in [('without positions',x,x[:,p]),('with positions',x+pos,x[:,p]+pos)]:
            add(6,seed,name,'max permutation discrepancy',err(block.forward(a)[0][:,p],block.forward(b)[0]),0,'permutation equivariance ideal')
        q,k,v=r.normal(size=(3,1,2,6,4))
        for scale in (1,10,100,1000):
            add(7,seed,f'scale={scale}','max absolute output discrepancy',err(attention(q*scale,k*scale,v)[0],ref_attention(q*scale,k*scale,v)),0,'PyTorch agreement ideal')
        for t in (8,16,32,64):
            q,k,v=r.normal(size=(3,1,2,t,4)); _,w=attention(q,k,v)
            add(8,seed,f'T={t}','attention weight elements',w.size,2*t*t,'2 × T² analytic count')
    return rows


def compare(a,b):
    assert len(a)==len(b)==63
    for x,y in zip(a,b):
        assert x.keys()==y.keys()
        for k,v in x.items():
            if isinstance(v,float): np.testing.assert_allclose(v,y[k],atol=1e-9,rtol=0)
            else: assert v==y[k]


def main():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'AIR-005'})
    a,b=run(),run(); compare(a,b); summary=[]
    Path('results').mkdir(exist_ok=True); Path('figures').mkdir(exist_ok=True)
    for exp,title in zip(sorted({r['experiment'] for r in a}),TITLES):
        rows=[r for r in a if r['experiment']==exp]; group=[]
        for c in dict.fromkeys(r['configuration'] for r in rows):
            rr=[r for r in rows if r['configuration']==c]; item={k:rr[0][k] for k in ('experiment','configuration','metric','reference_kind')}; item['n']=len(rr)
            for k in ('ours','reference'):
                vals=[r[k] for r in rr]; item[k+'_mean']=float(np.mean(vals)); item[k+'_sd']=float(np.std(vals,ddof=1))
            group.append(item)
        summary.extend(group); fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(group))
        for i,k in enumerate(('ours','reference')): ax.bar(x+(i-.5)*.36,[r[k+'_mean'] for r in group],.36,yerr=[r[k+'_sd'] for r in group],label='measured' if i==0 else 'reference / ideal',capsize=3)
        ax.set_xticks(x,[r['configuration'] for r in group],rotation=15,ha='right'); ax.set_ylabel('Discrepancy / weight (see table)' if exp=='EXP-005' else group[0]['metric']); ax.set_title(exp+' — '+title+' (3 seeds)'); ax.legend(); fig.tight_layout()
        for ext in ('svg','png'): fig.savefig('figures/'+exp+'.'+ext)
        plt.close(fig)
    payload=dict(project='AIR-005',seeds=SEEDS,environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,platform=platform.system()),reproducibility=dict(runs=2,atol=1e-9,rtol=0,status='pass'),records=a,summary=summary)
    Path('results/benchmarks.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n'); print(len(a),'records;',len(summary),'configurations; two runs matched')

if __name__=='__main__': main()

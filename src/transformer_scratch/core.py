"""Educational float64 forward-only transformer components, B x T x D."""
import numpy as np


def real(x):
    if np.iscomplexobj(x): raise ValueError('Real arrays only')
    a=np.asarray(x,dtype=np.float64)
    if not a.size or not np.isfinite(a).all(): raise ValueError('Nonempty finite array required')
    return a


def posint(n):
    if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<1: raise ValueError('Positive integer required')
    return int(n)


def embedding(ids,table):
    t=real(table); i=np.asarray(ids)
    if t.ndim!=2 or i.ndim!=2 or not i.size or i.dtype.kind not in 'iu' or (i<0).any() or (i>=len(t)).any(): raise ValueError('Invalid token IDs or table')
    return t[i]


def sinusoidal(length,dim):
    length,dim=posint(length),posint(dim)
    angles=np.arange(length)[:,None]*np.exp(-np.log(10000.)*np.arange(0,dim,2)/dim)
    out=np.zeros((length,dim)); out[:,0::2]=np.sin(angles); out[:,1::2]=np.cos(angles[:,:dim//2]); return out


def causal_mask(length):
    length=posint(length)
    return np.tril(np.ones((length,length),dtype=bool))


def padding_mask(valid):
    """Boolean B x T valid-key mask -> B x 1 x 1 x T. Does not suppress queries."""
    v=np.asarray(valid)
    if v.ndim!=2 or not v.size or v.dtype.kind!='b': raise ValueError('Boolean B x T mask required')
    return v[:,None,None,:]


def attention(q,k,v,allowed=None):
    """B x H x T x D. True means allowed; fully masked rows return zeros."""
    q,k,v=map(real,(q,k,v))
    if any(a.ndim!=4 for a in (q,k,v)) or q.shape[:2]!=k.shape[:2] or k.shape[:2]!=v.shape[:2] or q.shape[-1]!=k.shape[-1] or k.shape[2]!=v.shape[2]: raise ValueError('Incompatible Q/K/V shapes')
    with np.errstate(over='raise',invalid='raise'):
        scores=(q/np.sqrt(q.shape[-1]))@k.swapaxes(-2,-1)
    if not np.isfinite(scores).all(): raise FloatingPointError('Nonfinite scores')
    if allowed is None: m=np.ones(scores.shape,dtype=bool)
    else:
        m=np.asarray(allowed)
        if m.dtype.kind!='b' or m.ndim not in (2,4): raise ValueError('Expected boolean T x S or B x H x T x S mask')
        try: m=np.broadcast_to(m,scores.shape)
        except ValueError as e: raise ValueError('Incompatible mask') from e
    masked=np.where(m,scores,-np.inf); active=m.any(axis=-1,keepdims=True)
    maximum=np.where(active,masked.max(axis=-1,keepdims=True),0.)
    with np.errstate(over='ignore',under='ignore'):
        weights=np.exp(masked-maximum)
    denom=weights.sum(axis=-1,keepdims=True); weights=np.divide(weights,denom,out=np.zeros_like(weights),where=denom>0)
    with np.errstate(over='raise',invalid='raise'):
        out=weights@v
    if not np.isfinite(out).all(): raise FloatingPointError('Nonfinite attention output')
    return out,weights


def layer_norm(x,gamma,beta,eps=1e-5):
    x,gamma,beta=map(real,(x,gamma,beta))
    if x.ndim<2 or gamma.shape!=(x.shape[-1],) or beta.shape!=gamma.shape or not np.isfinite(eps) or eps<=0: raise ValueError('Invalid layer norm parameters')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        c=x-x.mean(axis=-1,keepdims=True); out=c/np.sqrt(np.mean(c*c,axis=-1,keepdims=True)+eps)*gamma+beta
    if not np.isfinite(out).all(): raise FloatingPointError('Nonfinite layer norm')
    return out


class MultiHead:
    def __init__(self,dim,heads,rng):
        self.dim,self.heads=posint(dim),posint(heads)
        if dim%heads: raise ValueError('dim must divide by heads')
        self.wq,self.wk,self.wv,self.wo=[rng.normal(0,1/np.sqrt(dim),(dim,dim)) for _ in range(4)]

    def forward(self,x,allowed=None):
        x=real(x)
        if x.ndim!=3 or x.shape[-1]!=self.dim: raise ValueError('Expected B x T x D')
        b,t,d=x.shape
        def split(w): return (x@w).reshape(b,t,self.heads,d//self.heads).transpose(0,2,1,3)
        z,a=attention(split(self.wq),split(self.wk),split(self.wv),allowed)
        with np.errstate(over='raise',invalid='raise'):
            out=z.transpose(0,2,1,3).reshape(b,t,d)@self.wo
        if not np.isfinite(out).all(): raise FloatingPointError('Nonfinite projection')
        return out,a


class TransformerBlock:
    """Post-norm self-attention + residual + ReLU FFN + residual. No dropout."""
    def __init__(self,dim,heads,hidden,rng):
        dim,hidden=posint(dim),posint(hidden)
        self.attn=MultiHead(dim,heads,rng)
        self.w1=rng.normal(0,1/np.sqrt(dim),(dim,hidden)); self.b1=np.zeros(hidden)
        self.w2=rng.normal(0,1/np.sqrt(hidden),(hidden,dim)); self.b2=np.zeros(dim)
        self.g1=np.ones(dim); self.bnorm1=np.zeros(dim); self.g2=np.ones(dim); self.bnorm2=np.zeros(dim)
    def forward(self,x,allowed=None):
        x=real(x); a,weights=self.attn.forward(x,allowed)
        h=layer_norm(x+a,self.g1,self.bnorm1)
        with np.errstate(over='raise',invalid='raise'):
            f=np.maximum(h@self.w1+self.b1,0)@self.w2+self.b2
        return layer_norm(h+f,self.g2,self.bnorm2),weights

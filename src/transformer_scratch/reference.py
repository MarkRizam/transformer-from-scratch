"""Independent PyTorch reference paths; never used by core implementation."""
import torch
import numpy as np


def tensor(x): return torch.tensor(x,dtype=torch.float64)


def ref_attention(q,k,v,mask=None):
    return torch.nn.functional.scaled_dot_product_attention(tensor(q),tensor(k),tensor(v),attn_mask=None if mask is None else torch.tensor(mask,dtype=torch.bool),dropout_p=0.).numpy()


def ref_multi(layer,x,mask=None):
    # PyTorch MultiheadAttention accepts True = blocked, unlike NumPy API.
    d,h=layer.dim,layer.heads
    m=torch.nn.MultiheadAttention(d,h,bias=False,batch_first=True,dropout=0.,dtype=torch.float64)
    with torch.no_grad():
        m.in_proj_weight.copy_(tensor(np.concatenate([layer.wq.T,layer.wk.T,layer.wv.T])))
        m.out_proj.weight.copy_(tensor(layer.wo.T))
        a=tensor(x); y,w=m(a,a,a,attn_mask=None if mask is None else torch.tensor(~mask),average_attn_weights=False)
    return y.numpy(),w.numpy()


def ref_block(block,x,mask=None):
    d,h=block.attn.dim,block.attn.heads
    m=torch.nn.TransformerEncoderLayer(d,h,dim_feedforward=block.w1.shape[1],dropout=0.,activation='relu',batch_first=True,norm_first=False,dtype=torch.float64)
    with torch.no_grad():
        m.self_attn.in_proj_weight.copy_(tensor(np.concatenate([block.attn.wq.T,block.attn.wk.T,block.attn.wv.T])))
        m.self_attn.in_proj_bias.zero_(); m.self_attn.out_proj.weight.copy_(tensor(block.attn.wo.T)); m.self_attn.out_proj.bias.zero_()
        for layer,w,b in [(m.linear1,block.w1,block.b1),(m.linear2,block.w2,block.b2)]: layer.weight.copy_(tensor(w.T)); layer.bias.copy_(tensor(b))
        for layer,g,b in [(m.norm1,block.g1,block.bnorm1),(m.norm2,block.g2,block.bnorm2)]: layer.weight.copy_(tensor(g)); layer.bias.copy_(tensor(b))
        return m(tensor(x),src_mask=None if mask is None else torch.tensor(~mask)).numpy()

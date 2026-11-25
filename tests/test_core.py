import numpy as np
import pytest
import torch
from transformer_scratch import *
from transformer_scratch.reference import ref_attention,ref_multi,ref_block,tensor

torch.set_num_threads(1)
@pytest.mark.parametrize('seed',[11,22,33])
@pytest.mark.parametrize('masked',[False,True])
def test_attention(seed,masked):
    r=np.random.default_rng(seed); q,k,v=r.normal(size=(3,2,2,5,4)); m=causal_mask(5) if masked else None
    out,w=attention(q,k,v,m); np.testing.assert_allclose(out,ref_attention(q,k,v,m),atol=1e-12); np.testing.assert_allclose(w.sum(-1),1,atol=1e-12)
    if masked: assert np.all(w[...,~m]==0)
@pytest.mark.parametrize('seed',[11,22,33])
def test_multi(seed):
    r=np.random.default_rng(seed); m=MultiHead(8,2,r); x=r.normal(size=(2,5,8)); a,b=m.forward(x,causal_mask(5)); c,d=ref_multi(m,x,causal_mask(5)); np.testing.assert_allclose(a,c,atol=1e-12); np.testing.assert_allclose(b,d,atol=1e-12)
@pytest.mark.parametrize('seed',[11,22,33])
@pytest.mark.parametrize('masked',[False,True])
def test_block(seed,masked):
    r=np.random.default_rng(seed); m=TransformerBlock(8,2,16,r); x=r.normal(size=(2,5,8)); mask=causal_mask(5) if masked else None
    np.testing.assert_allclose(m.forward(x,mask)[0],ref_block(m,x,mask),atol=1e-12)
@pytest.mark.parametrize('dim',[1,3,8])
def test_positions(dim):
    a=sinusoidal(7,dim); assert a.shape==(7,dim); np.testing.assert_array_equal(a[0,::2],0); np.testing.assert_array_equal(a[0,1::2],1)

def test_padding_and_empty_rows():
    q=np.ones((2,2,3,4)); mask=padding_mask([[True,False,True],[False,False,False]]); a,w=attention(q,q,q,mask)
    assert np.all(w[0,...,1]==0); np.testing.assert_array_equal(a[1],0); np.testing.assert_array_equal(w[1],0); np.testing.assert_allclose(a,ref_attention(q,q,q,mask),atol=1e-12)

def test_causal_prefix():
    r=np.random.default_rng(11); x=r.normal(size=(1,6,8)); m=TransformerBlock(8,2,16,r); a=m.forward(x,causal_mask(6))[0]; x[:,3:]+=10; b=m.forward(x,causal_mask(6))[0]; np.testing.assert_allclose(a[:,:3],b[:,:3],atol=1e-12)

def test_embedding():
    t=np.arange(24).reshape(6,4); i=np.array([[0,5,2]]); np.testing.assert_array_equal(embedding(i,t),torch.nn.functional.embedding(torch.tensor(i),tensor(t)).numpy())

def test_layer_norm():
    r=np.random.default_rng(11); x=r.normal(size=(2,3,8)); g=r.normal(size=8); b=r.normal(size=8); np.testing.assert_allclose(layer_norm(x,g,b),torch.nn.functional.layer_norm(tensor(x),(8,),tensor(g),tensor(b)).numpy(),atol=1e-12)

def test_large_logits():
    q=np.array([[[[1000.,0.],[0.,1000.]]]]); a,w=attention(q,q,q); np.testing.assert_allclose(w.sum(-1),1); assert np.isfinite(a).all()
@pytest.mark.parametrize('f',[lambda:embedding([[6]],np.ones((6,2))),lambda:embedding([[.5]],np.ones((6,2))),lambda:sinusoidal(0,4),lambda:sinusoidal(3,True),lambda:padding_mask([[1,0]]),lambda:MultiHead(7,2,np.random.default_rng(1)),lambda:attention(np.ones((1,2)),np.ones((1,2)),np.ones((1,2))),lambda:attention(np.ones((1,1,2,2)),np.ones((1,1,2,2)),np.ones((1,1,2,2)),np.ones((2,2))),lambda:layer_norm([[1,2]],[1],[0]),lambda:layer_norm([[1,2]],[1,1],[0,0],0),lambda:embedding([[0]],np.array([[1j]]))])
def test_invalid(f):
    with pytest.raises(ValueError): f()

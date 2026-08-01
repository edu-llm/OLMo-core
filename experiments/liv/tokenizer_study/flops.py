D=1024;L=16;NH=16;NKV=8;HD=64;FF=4608;K=3
def flops_tok(V,d=D,ff=FF,k=K,natt=6,nliv=10,nkv=NKV,T=4096,gate_saving=0):
    # per-token fwd+bwd ~ 6 * params_used, plus attention score term 6*2*T*d_head*heads... use 2*.. fwd
    liv=nliv*(4*d*d+k*d)-gate_saving
    attn=natt*(d*NH*HD+2*d*nkv*HD+NH*HD*d)
    mlp=L*3*d*ff
    dense=6*(liv+attn+mlp+V*d)          # 6ND-style incl LM head
    score=natt*6*2*T*NH*HD              # QK^T + AV, fwd+bwd
    return dense+score
def nls(r,d=D,n=10): return n*(2*d*d-4*r*d)
arms={'L0':dict(),'A16-P':dict(ff=4820,natt=16,nliv=0),'F-r128':dict(gate_saving=nls(128)),
      'A-fewer3':dict(natt=3,nliv=13),'Q-mqa':dict(nkv=1),'N-narrow':dict(d=976,ff=4668)}
for V in (65536,50304):
    print(f"\n=== V={V} : FLOPs/token relative to L0 ===")
    for T in (4096,32768):
        base=flops_tok(V,T=T)
        row=f" T={T:>6}: "
        for n,kw in arms.items():
            kw2=dict(kw)
            if V==50304 and n=='N-narrow': kw2['ff']=4652
            if V==50304 and n=='A16-P': kw2['ff']=4820
            row+=f"{n} {flops_tok(V,T=T,**kw2)/base:.3f}x  "
        print(row)
    # attention-score share of 6ND
    for T in (4096,16384,32768):
        b=flops_tok(V,T=T); s=6*2*T*NH*HD*6
        sa=6*2*T*NH*HD*16
        print(f"   T={T:>6}: score share L0 {100*s/b:.1f}%   A16-P {100*sa/flops_tok(V,T=T,ff=4820,natt=16,nliv=0):.1f}%")

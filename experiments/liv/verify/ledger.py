def ledger(d,L,V,ff,nliv,ngqa,k,hq,hkv,hd):
    emb=V*d; mlp=L*3*d*ff; liv=nliv*(4*d*d+k*d)
    gqa_per=d*(hq*hd)+2*d*(hkv*hd)+(hq*hd)*d
    gqa=ngqa*gqa_per
    norms=L*2*d+d+ngqa*2*hd
    return dict(emb=emb,mlp=mlp,liv=liv,gqa=gqa,gqa_per=gqa_per,norms=norms,
                tot=emb+mlp+liv+gqa+norms)

print('===== d=2048 / LFM2-1.2B =====')
b=ledger(2048,16,65536,8192,10,6,3,32,8,64)
for k_,v in b.items(): print('  ',k_,v)
print('  target 1170340608 match:', b['tot']==1170340608)
T=b['tot']
print('  emb %.2f%%  liv %.2f%%  gqa %.2f%%  mlp %.2f%%'%(100*b['emb']/T,100*b['liv']/T,100*b['gqa']/T,100*b['mlp']/T))
print('  gqa per-layer %d = %.3f d^2'%(b['gqa_per'],b['gqa_per']/2048**2))
NEb=T-b['emb']
print('  NONEMB denom(2048): liv %.2f%%  mlp %.2f%%'%(100*b['liv']/NEb,100*b['mlp']/NEb))
print()
print('  rank sweep at d=2048 (10 LIV layers, gates 2d^2 -> 4dr):')
d=2048;n=10
for r in [32,64,128,256,512,1024]:
    saved=n*(2*d*d-4*d*r)
    print('   r=%4d  gate params %12d  saved %12d  = %.3f%% of model'%(r,n*4*d*r,saved,100*saved/T))
print()
print('===== d=1024 / frozen L0 =====')
a=ledger(1024,16,65536,4608,10,6,3,16,8,64)
for k_,v in a.items(): print('  ',k_,v)
print('  target 354483968 match:', a['tot']==354483968)
T2=a['tot']; NE=T2-a['emb']
print('  FULL denom  : emb %.2f%%  liv %.2f%%  gqa %.2f%%  mlp %.2f%%'%(100*a['emb']/T2,100*a['liv']/T2,100*a['gqa']/T2,100*a['mlp']/T2))
print('  NONEMB denom: liv %.2f%%  gqa %.2f%%  mlp %.2f%%  (nonemb=%d)'%(100*a['liv']/NE,100*a['gqa']/NE,100*a['mlp']/NE,NE))
print()
d=1024;n=10
print('  rank sweep at d=1024:')
for r in [32,64,128,256,512]:
    saved=n*(2*d*d-4*d*r)
    print('   r=%4d gate %10d saved %10d  full%%=%.3f  nonemb%%=%.3f  new_total=%d'%(r,n*4*d*r,saved,100*saved/T2,100*saved/NE,T2-saved))
print()
print('  GQA at 2.5d^2 vs 3d^2, d=1024:')
print('   2.5d^2 per-layer =', 2.5*1024**2, ' x6 =', 6*2.5*1024**2)
print('   3.0d^2 per-layer =', 3*1024**2, ' x6 =', 6*3*1024**2)
print('   shortfall total  =', 6*3*1024**2 - int(6*2.5*1024**2), ' per-layer', 3*1024**2-int(2.5*1024**2))

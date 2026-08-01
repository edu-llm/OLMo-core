arms = [
 ('dense',              41943040, 56.223999708890915, 695),
 ('lowrank_fused r=128',10485760, 60.83200126886368,  161),
 ('lowrank_sep r=128',  10485760, 76.48000121116638,  128),
 ('grouped g=4',        10485760, 47.600001096725464, 205),
 ('grouped g=2',        20971520, 47.68000170588493,  None),
 ('lowrank_fused r=512',41943040, 90.01599997282028,  None),
]
print('%-22s %12s %10s %10s %10s %8s'%('arm','bytes','us','GB/s(1e9)','GiB/s(2^30)','doc'))
for n,b,t,doc in arms:
    s=t*1e-6
    gb=b/s/1e9; gib=b/s/2**30
    print('%-22s %12d %10.3f %10.1f %11.1f %8s'%(n,b,t,gb,gib,doc))
print()
print('code line 87 recompute: mb/median*1e6/1024  where mb = bytes/2**20')
for n,b,t,doc in arms:
    mb=b/2**20
    v=mb/t*1e6/1024
    print('  %-22s -> %8.2f   (rounds to %d)  == GiB/s? %s'%(n,v,round(v), abs(v-(b/(t*1e-6)/2**30))<1e-6))
print()
peak=864.0
print('L40S HBM spec peak = %.0f GB/s (1e9)'%peak)
print('  dense GB/s / peak  = %.1f%%'%(100*(41943040/(56.223999708890915e-6)/1e9)/peak))
print('  dense GiB/s / peak = %.1f%%  <- the doc\'s 80%% (unit-mismatched)'%(100*(41943040/(56.223999708890915e-6)/2**30)/peak))
print('  fused GB/s / peak  = %.1f%%'%(100*(10485760/(60.83200126886368e-6)/1e9)/peak))
print('  fused GiB/s / peak = %.1f%%  <- the doc\'s 19%%'%(100*(10485760/(60.83200126886368e-6)/2**30)/peak))
print()
print('peak in GiB/s = %.1f'%(864e9/2**30))
print('  dense GiB/s / peak_GiB = %.1f%%  (self-consistent unit comparison)'%(100*(41943040/(56.223999708890915e-6)/2**30)/(864e9/2**30)))
print()
print('grouped g=2 vs g=4: bytes %dx, time ratio %.5f -> delta %.2f%%'%(20971520//10485760, 47.68000170588493/47.600001096725464, 100*(47.68000170588493-47.600001096725464)/47.600001096725464))

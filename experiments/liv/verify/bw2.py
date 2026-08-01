# Claim (e) part 2: is the '86.4% of peak' correction meaningful, given the L2 confound?
L2 = 100663296  # measured, torch.cuda.get_device_properties on L40S, job 1671407
print('L40S L2 = %d B = %.1f MiB'%(L2, L2/2**20))
for mib in (10,20,40):
    print('  arm working set %2d MiB -> %.1f%% of L2  (fits: %s)'%(mib, 100*mib*2**20/L2, mib*2**20 < L2))
print()
# in-cache vs out-of-cache dense rate, from p1_scaled_results.json (job 1671420)
rows = [(40,56.32000043988228,40.0),(320,728.0640006065369,320.0),(960,2180.0639629364014,960.0)]
print('dense achieved rate by working set (from p1_scaled_results.json):')
for ws,us,mib in rows:
    B = mib*2**20
    gb = B/(us*1e-6)/1e9
    gib= B/(us*1e-6)/2**30
    print('  %4d MiB  %9.2f us  ->  %7.1f GB/s (%6.1f GiB/s)   = %5.1f%% of 864 GB/s HBM peak'%(ws,us,gb,gib,100*gb/864))
print()
print('So the two candidate headline numbers:')
print('  (A) units-fixed, in-cache 40 MiB run : 746 GB/s = 86.3%% of HBM peak  <- measures L2')
print('  (B) out-of-cache 960 MiB run          : %.0f GB/s = %.1f%% of HBM peak  <- measures HBM'%(960*2**20/(2180.0639629364014e-6)/1e9, 100*(960*2**20/(2180.0639629364014e-6)/1e9)/864))
print()
# implied L2 rate sanity: per-kernel fixed cost
print('per-kernel fixed cost check (dense, 40 MiB, 20 kernels):')
print('  %.2f us/kernel total; 2 MiB/kernel at a nominal 4 TB/s L2 = %.2f us of data movement'%(56.224/20, (2*2**20)/4e12*1e6))
print('grouped g2 vs g4 (identical 20 kernels, 2x bytes): 47.680 vs 47.600 us -> %.2f%% '%(100*(47.68000170588493-47.600001096725464)/47.600001096725464))

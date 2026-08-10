# Final local mixer and block throughput

Snapshot: 2026-08-09, current comparison tree.

## Method

Hardware is an RTX 5050 Laptop GPU (sm120, 8 GiB). Every measurement uses
`B=2, T=4096, D=1024`, bfloat16, `torch.compile`, 20 warmup iterations and 50
CUDA-event measurements. This matches the rank microbatch and sequence geometry
of the comparison. The production run compiles each transformer block
independently.

Absolute laptop timings are unstable: frozen GDN2 code has varied by 37% across
sessions. Therefore every target below was instantiated alongside a GDN2 control
in the **same process**, then target and control steps were alternated. The primary
quantity is `time / GDN2`; raw milliseconds and tokens/s are retained as receipts,
not as cross-process comparisons. Benchmark harness:
`/tmp/olmo-paired-bench.py`.

The identical-control checks resolved the harness at approximately 0.1%:

- mixer GDN2 vs GDN2: time ratio 1.000856;
- high-width block GDN2 vs GDN2: time ratio 0.998904.

## Final-model throughput proxy

This is the useful result. It composes every arm's exact solved block inventory:
12 recurrent blocks plus the four shared attention blocks at `(3, 7, 11, 15)`.
The `/32` FFN-width distribution is included rather than approximating every
recurrent layer with layer 0.

The raw proxy sums block forward+backward latency and converts the rank's 8,192
tokens to tokens/s. It excludes embeddings, LM head, FSDP communication and the
optimizer. Those costs are shared or parameter-proportional across parameter-
matched arms, so they should compress differences rather than widen them.

| Arm | Block inventory used | Proxy ms | Proxy tok/s | Throughput / GDN2 |
| --- | --- | ---: | ---: | ---: |
| Mamba-3 b=3 | 7 high + 5 low | 735.0 | 11,146 | **1.165x** |
| native PD-SSM | 6 high + 6 low | 764.2 | 10,719 | **1.120x** |
| xLSTM | 7 m-high + 3 m-low + 1 s-high + 1 s-low | 828.1 | 9,892 | **1.034x** |
| Mamba-3 SISO PD | 11 high + 1 low | 802.2 | 10,211 | **1.067x** |
| KDA | 3 high + 9 low | 840.2 | 9,750 | **1.019x** |
| GDN2 | 1 high + 11 low | 856.0 | 9,570 | **1.000x** |
| KDA + gated conv | 2 high + 10 low | 861.0 | 9,514 | 0.994x |
| KDA Householder R=2 | 8 high + 4 low | 1,439.6 | 5,690 | **0.595x** |

The absolute proxy uses 54.849 ms as the high-width GDN2 anchor, the midpoint of
the same-process identical-control run. Ratios do not depend on that anchor.

Six arms are within 4% of GDN2. Mixer-only measurements are a biased proxy here:
parameter matching respends mixer parameters into FFN width, and dense FFN GEMMs
run much more efficiently than recurrent scans. This is why both PD arms look
slow per mixer but beat GDN2 at block/model-proxy level.

## Mixer measurements

| Mixer | Params | Target ms | Target tok/s | GDN2 ms | Time / GDN2 | Speed / GDN2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mamba-3 b=3 | 3,473,888 | 15.821 | 517,788 | 34.927 | 0.453 | **2.208x** |
| xLSTM mLSTM | 4,208,648 | 16.825 | 486,891 | 31.871 | 0.528 | **1.894x** |
| KDA | 4,487,248 | 27.260 | 300,510 | 34.136 | 0.799 | **1.252x** |
| KDA + gated conv | 4,493,392 | 29.720 | 275,639 | 32.868 | 0.904 | **1.106x** |
| GDN2 | 6,568,016 | 31.843 | 257,264 | 31.816 | 1.001 | 0.999x |
| native PD-SSM | 10,756,096 | 32.343 | 253,285 | 31.997 | 1.011 | 0.989x |
| Mamba-3 SISO PD | 9,734,320 | 32.863 | 249,277 | 31.739 | 1.035 | 0.966x |
| xLSTM sLSTM | 2,107,392 | 57.089 | 143,494 | 31.844 | 1.793 | 0.558x |
| KDA Householder R=2 | 6,608,976 | 84.798 | 96,606 | 31.576 | 2.686 | 0.372x |

These values include the latest optimization round:

- native PD densifies all four scan inputs inside the compiled region and uses
  explicit unfused post-convolution projections;
- Mamba-3 SISO PD moves C normalization/readout work to the consumer-side graph
  and uses explicit unfused projections, which won a same-process A/B by 7.6%;
- native PD uses chunk size 64;
- the SISO CUDA scan is kept opaque to Dynamo with
  `torch.compiler.disable`.

## Complete block measurements

Each row is a complete recurrent block: mixer, solved FFN and norms. `High` and
`low` refer only to the `/32` FFN-width allocation; the mixer is unchanged.
Every target is paired with a high-width GDN2 block in the same process.

### High-width and distinct-role blocks

| Block | Layer | FFN width | Params | Target ms | Target tok/s | GDN2 ms | Time / GDN2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mamba-3 b=3 high | 0 | 4,800 | 18,221,536 | 43.652 | 187,666 | 53.319 | 0.819 |
| xLSTM mLSTM high | 0 | 4,672 | 18,563,080 | 45.742 | 179,091 | 54.204 | 0.844 |
| native PD high | 0 | 2,432 | 18,229,248 | 49.252 | 166,329 | 56.806 | 0.867 |
| Mamba-3 SISO PD high | 0 | 2,752 | 18,190,512 | 47.379 | 172,905 | 51.528 | 0.919 |
| KDA high | 0 | 4,480 | 18,251,856 | 54.420 | 150,533 | 55.751 | 0.976 |
| GDN2 high | 0 | 3,808 | 18,268,240 | 54.819 | 149,437 | 54.879 | 0.999 |
| KDA + gated conv high | 0 | 4,480 | 18,258,000 | 57.380 | 142,769 | 56.224 | 1.021 |
| xLSTM sLSTM high | 6 | 4,672 | 16,461,824 | 88.621 | 92,439 | 57.481 | 1.542 |
| KDA Householder R=2 high | 0 | 3,776 | 18,210,896 | 102.282 | 80,092 | 54.732 | 1.869 |
| shared attention | 3 | 4,608 | 17,305,088 | 49.857 | 164,310 | 55.515 | 0.898 |

### Low-width blocks

| Block | Layer | FFN width | Params | Target ms | Target tok/s | GDN2-high ms | Time / GDN2-high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mamba-3 b=3 low | 9 | 4,768 | 18,123,232 | 46.902 | 174,663 | 57.517 | 0.815 |
| xLSTM mLSTM low | 10 | 4,640 | 18,464,776 | 48.027 | 170,570 | 57.034 | 0.842 |
| native PD low | 8 | 2,400 | 18,130,944 | 48.150 | 170,135 | 56.216 | 0.857 |
| Mamba-3 SISO PD low | 14 | 2,720 | 18,092,208 | 47.462 | 172,601 | 51.608 | 0.920 |
| KDA low | 4 | 4,448 | 18,153,552 | 54.159 | 151,258 | 55.400 | 0.978 |
| GDN2 low | 1 | 3,776 | 18,169,936 | 54.911 | 149,188 | 54.837 | 1.001 |
| KDA + gated conv low | 2 | 4,448 | 18,159,696 | 55.979 | 146,339 | 55.622 | 1.006 |
| xLSTM sLSTM low | 14 | 4,640 | 16,363,520 | 87.803 | 93,300 | 57.353 | 1.531 |
| KDA Householder R=2 low | 10 | 3,744 | 18,112,592 | 110.133 | 74,383 | 57.177 | 1.926 |

## Interpretation

Mamba-3 b=3 remains the fastest arm. Native PD is second in the final-model
proxy despite being roughly tied with GDN2 per mixer, because its FFN is much
narrower. The SISO PD arm is now about 6.7% ahead of GDN2 in the model proxy;
xLSTM and KDA remain effectively tied with GDN2 at this resolution.

The two real outliers are:

- xLSTM sLSTM, which is slow and memory-heavy but occupies only two layers;
- KDA Householder R=2, which occupies all twelve recurrent layers and measured
  about 3.6 GiB peak allocation for one mixer / block in prior clean sweeps.
  Twelve such blocks are a serious A100-40GB memory risk and may force a smaller
  rank microbatch, which would invalidate direct throughput comparison.

## Limits

This is consumer sm120 hardware, not production sm80. It excludes embeddings,
LM head, FSDP communication and optimizer work. The production eight-A100 smoke
is authoritative.

MFU is intentionally absent: no documented dense BF16 peak exists for this
laptop's power configuration. FLOP/s is also omitted because the mixers use
incompatible `num_flops_per_token` conventions. Wall time at identical geometry
and same-process control ratios are the only sound local comparisons.

# AWS operations and cost guardrails

**Status:** read-only preflight complete on 2026-07-31. No instance, volume, data transfer, or other AWS resource has been created.

This document translates the user's instance decisions into an executable, cost-aware operating contract. It uses only the ready sbsandbox account in us-east-1. A fresh read-only capacity and price check is mandatory immediately before any requested launch.

## Approved-by-plan instance assignments

| Phase | User-selected instance | Purpose | Do not use it for |
|---|---|---|---|
| Phase 0 | g6e.xlarge | GPU semantic tests, BF16 numerical tests, and small-LM shape preflight | multi-seed training or target-model claims |
| Phase 1 | p5.48xlarge | Eight concurrent synthetic-probe workers and fast wall-clock completion | single-run debugging or an underfilled batch |
| Phase 2 | not selected | Decided only after the Phase-1 gate and its measured runtime/memory evidence | any pre-approved launch |

## Read-only AWS facts

The following were queried through sb-aws on 2026-07-31 in account sbsandbox, region us-east-1:

| Instance | Accelerator / memory | Host resources | Linux On-Demand rate | Instance-type offering |
|---|---|---|---:|---|
| g6e.xlarge | 1× NVIDIA L40S, 45,776 MiB reported GPU memory | 4 vCPU, 32 GiB RAM, up to 20 Gbps network, 5,000 Mbps EBS | $1.861/hour | offered in us-east-1a/b/c/d |
| p5.48xlarge | 8× NVIDIA H100, 81,920 MiB per GPU | 192 vCPU, 2 TiB RAM, 3,200 Gbps network, 80,000 Mbps EBS | $55.04/hour | offered in us-east-1a/b/c/d/e/f |

Regional On-Demand quota is 768 vCPU for both P instances and G/VT instances. One p5.48xlarge consumes 192 P vCPUs, so the quota permits up to four such nodes in principle. An instance-type offering is not a capacity reservation; capacity must be rechecked at launch time.

These rates exclude EBS, public IPv4, data transfer, storage, logging, and any future paid service. The plan assumes no NAT gateway, load balancer, or provisioned EBS IOPS/throughput unless a later approved launch specifically adds one.

## EC2 value assessment

### Phase 0 — g6e.xlarge

**Why it fits:** The L40S is the same GPU class on which the existing DP-R implementation was validated. One 48 GB-class GPU is enough for the correctness suite, and the small host footprint limits idle cost while code/tests are being repaired.

**Advantages:** low hourly cost; known L40S compatibility; 48 GB GPU memory; no distributed setup; suitable for repeated test/debug cycles.

**Limitations:** only four vCPUs, one GPU, and no useful seed parallelism. Do not attempt a broad Phase-1 matrix on it.

**Necessity/value:** appropriate and close to the smallest GPU shape that can exercise the actual Triton path. It is not a target-scale benchmark platform.

**Cost equation:** Phase-0 instance charge is \(1.861\times H_0\) USD, where \(H_0\) is elapsed allocated node hours. The operator sets a Phase-0 time cap in the launch request; the runbook does not assume a duration.

### Phase 1 — p5.48xlarge

**Why it fits:** Eight H100s allow one independent synthetic-probe worker per GPU. The Phase-1 triage matrix has many independent jobs, so the node minimizes wall-clock time when it is kept full.

**Advantages:** eight 80 GB H100s; enough host memory and network for eight isolated workers; two or more waves can complete quickly; supports a clean per-GPU seed/arm schedule.

**Limitations:** $55.04/hour is expensive; it is overpowered for one debug run; capacity can be volatile; the work must be designed to occupy at least six GPUs throughout material waves.

**Necessity/value:** not required to establish the science—smaller GPUs could run the probes—but selected by the user for lower wall-clock time. Its value depends on high occupancy. If fewer than six independent Phase-1 jobs are ready, do not request this node yet; finish local/static preparation or wait until a full wave is available.

**Cost equation:** Phase-1 instance charge is \(55.04\times H_1\) USD. For \(N\) equal-duration probe jobs, eight workers, measured per-job duration \(t\), and setup/teardown \(o\):

\[
H_1 \approx \left\lceil\frac{N}{8}\right\rceil t+o.
\]

Use a timed eight-worker smoke wave to measure \(t\) before committing to the full matrix. Do not extrapolate cost from L40S or R4 measurements.

## Mandatory pre-launch checklist

The experiment operator must complete and record every item below before asking for launch approval:

1. Read-only sb-aws check: account identity, us-east-1 quota, instance-type offering, and current Linux On-Demand rate.
2. Confirm the exact source revision, image digest, Phase/manifest ID, result destination, and expected worker count.
3. Estimate node hours and total instance cost from the actual Phase-0/Phase-1 matrix and the most recent measured smoke runtime.
4. Confirm that no data will be downloaded to a workstation and identify any approved remote result destination. If there is no approved destination, keep only the small declared result bundle and ask before transfer.
5. For p5.48xlarge, show an eight-slot wave plan. Fewer than six occupied slots is a no-launch condition unless the program owner explicitly accepts the waste.
6. Obtain explicit user approval for the exact region, instance type/count, maximum runtime, result destination, and spend ceiling.

Only after approval may the operator create an EC2 instance or submit a job. Use sb-aws for AWS access; do not bypass it with a local AWS CLI or credentials.

## On-node operating rules

- One process owns one GPU. Set the GPU assignment explicitly; never rely on an implicit default device.
- Record GPU UUID, driver, CUDA, PyTorch, Triton, FLA, git revision, image digest, and manifest ID in every result.
- Compile and warm kernels before timed measurements. Separate compilation/setup time from training time.
- Fail fast on a NaN, Inf, incorrect manifest hash, missing expected task output, or GPU-memory headroom below the declared threshold.
- Do not rerun a failed seed silently. Preserve its logs and mark the seed failed in the manifest.
- Treat current AZ offerings as metadata, not a guarantee. If provisioning fails, report it; do not change instance type or region without a new approval.

## Live-status reporting during any future run

After a user-approved launch, report an initial ETA, then update at each wave transition and at least every 30 minutes with instance state, completed/total jobs, measured jobs/hour, observed cost, and revised ETA. Do not stop, replace, or resize an instance without a new explicit approval.

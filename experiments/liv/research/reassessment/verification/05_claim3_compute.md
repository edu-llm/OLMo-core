# Claim 3 — FarmShare compute envelope (48 h / 4 concurrent GPUs vs 6 h / 1 GPU)

**Verdict: CONFIRMED on walltime + concurrency. The sub-claim "`-c 8 --mem=48G` is rejected" is REFUTED.**
**Verifier:** verification agent, 2026-08-01. All numbers below are literal cluster output.
**Cluster access:** `ssh -S /tmp/farmshare-ericrcwu.sock ericrcwu@login.farmshare.stanford.edu`
Login node at time of survey: `rice-03`, `Sat Aug  1 09:48:53 PDT 2026`, user `ericrcwu`.
**No GPU job was enqueued.** Every scheduling probe used `sbatch --test-only`, which does not submit.

---

## 0. Executive answer

| Question | Answer | Binding constraint |
|---|---|---|
| Max walltime, `-p gpu` default QOS | **48 h** (`2-00:00:00`) | partition `MaxTime` |
| Max walltime, `-p gpu --qos=long` | **7 d** (`7-00:00:00`) | QOS `long` `MaxWall`; partition limit is overridden by the `PartitionTimeLimit` flag |
| Max concurrent GPUs per user | **4** | partition QOS `gpu` `MaxTRESPU=gres/gpu=4` |
| Max concurrent *jobs* per user | **4** | QOS `gpu` `MaxJobsPU=4` |
| Max *queued+running* jobs per user | **32** | QOS `gpu` `MaxSubmitPU=32` |
| Multi-GPU single-node | **YES, up to `--gres=gpu:4` on one node** | node has 4 GPUs; 5+ hits `QOSMaxGRESPerUser` |
| Multi-*node* GPU | **NO** — `-N2 --gres=gpu:4` = 8 GPUs > 4 cap | `QOSMaxGRESPerUser` |
| `-c 8 --mem=48G` | **ACCEPTED** — silently auto-bumped to 14 CPUs | not rejected; `MaxMemPerCPU=4000` causes a *bump*, not a denial |
| GPU hardware | **NVIDIA L40S, CC 8.9, 48 GB, 4/node × 6 nodes = 24 cluster-wide** | `oat-[01-06]` ActiveFeatures |
| Preemption | **OFF** (`PreemptMode=OFF`, partition and global) | jobs are never preempted |

**The "6 h / 1 GPU" figure exists nowhere in the Slurm configuration.** Nearest real numbers are
`DefaultTime=02:00:00` (the default if you omit `-t`) and QOS `normal`'s `MaxTRESPU=gres/gpu=1`.
Provenance of the false claim: `/Users/ericwu/Developer/Capstone_LLM/docs/liv-brainlift-experiment-design.md:1401`
— *"Given FarmShare allocates 1 GPU per job with a 6-hour…"*. It was never measured.

**Origin of the "1 GPU" half:** it is true *if you submit to `-p normal`.* Partition `normal` also
contains the `oat-*` GPU nodes, but its partition QOS is `normal`, whose `MaxTRESPU` is
`cpu=512,gres/gpu=1`. Verified: `-p normal --gres=gpu:2` → `QOSMaxGRESPerUser`. So the plan's author
likely submitted GPU work to `-p normal` and inferred a 1-GPU cluster cap. **Always use `-p gpu`.**

---

## 1. Raw cluster output

### 1.1 `sinfo -o "%P %a %l %D %N %G %m %c"`

```
PARTITION AVAIL TIMELIMIT NODES NODELIST GRES MEMORY CPUS
normal* up 2-00:00:00 14 barley-[01-04],rye-[01-02],wheat-[01-08] (null) 191000+ 80+
normal* up 2-00:00:00 6 oat-[01-06] gpu:4(S:0-1) 256000 64
bigmem up 2-00:00:00 2 rye-[01-02] (null) 1536000 512
gpu up 2-00:00:00 6 oat-[01-06] gpu:4(S:0-1) 256000 64
```

### 1.2 `sinfo -N -o "%N %P %G %c %m %t"` — GPU node inventory

```
NODELIST PARTITION GRES CPUS MEMORY STATE
oat-01 gpu gpu:4(S:0-1) 64 256000 mix
oat-02 gpu gpu:4(S:0-1) 64 256000 mix
oat-03 gpu gpu:4(S:0-1) 64 256000 alloc
oat-04 gpu gpu:4(S:0-1) 64 256000 mix
oat-05 gpu gpu:4(S:0-1) 64 256000 mix
oat-06 gpu gpu:4(S:0-1) 64 256000 mix
```

(Non-GPU nodes elided: `barley-[01-04]` 128c/384G, `rye-[01-02]` 512c/1536G, `wheat-[01-08]` 80c/191G,
of which `wheat-02,03,05` are `drain*`.)

**Total GPUs on the cluster = 6 nodes × 4 = 24 L40S.** Confirmed by
`scontrol show partition gpu` → `TRES=...,gres/gpu=24`.

### 1.3 GPU model — `scontrol show node oat-01`

```
NodeName=oat-01 Arch=x86_64 CoresPerSocket=16
   Gres=gpu:4(S:0-1)
   AvailableFeatures="CPU_MNF:INTEL,CPU_GEN:SPR,CPU_SKU:6426Y,CPU_FRQ:2.50GHz,GPU_GEN:LOV,GPU_BRD:TESLA,GPU_SKU:L40S,GPU_MEM:48GB,GPU_CC:8.9,CLASS:SH4_G4FP32
   ActiveFeatures="...GPU_SKU:L40S,GPU_MEM:48GB,GPU_CC:8.9,CLASS:SH4_G4FP32
   State=MIXED ThreadsPerCore=2 TmpDisk=7000000
   CfgTRES=cpu=64,mem=250G,billing=64,gres/gpu=4
   AllocTRES=cpu=46,mem=168G,gres/gpu=3
```

`GPU_SKU:L40S`, `GPU_CC:8.9` (sm_89), `GPU_MEM:48GB`. Matches the task brief. Note `ThreadsPerCore=2`
— "CPUs" in Slurm here are hyperthreads, 32 physical cores per node.
No NVLink is advertised; L40S is a PCIe card, so multi-GPU scaling is over PCIe, not NVLink.

### 1.4 `scontrol show partition` (all three)

```
PartitionName=normal
   AllowGroups=ALL AllowAccounts=ALL AllowQos=normal,long,interactive,dev
   AllocNodes=ALL Default=YES QoS=normal
   DefaultTime=02:00:00 DisableRootJobs=NO Exclusive=NO GraceTime=0 Hidden=NO
   MaxNodes=UNLIMITED MaxTime=2-00:00:00 MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED MaxCPUsPerSocket=UNLIMITED
   Nodes=barley-[01-04],oat-[01-06],rye-[01-02],wheat-[01-08]
   PriorityJobFactor=100 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=2560 TotalNodes=20 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerCPU=2400 MaxMemPerCPU=4000
   TRES=cpu=2560,mem=7672000M,node=20,billing=2560,gres/gpu=24

PartitionName=bigmem
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=bigmem
   DefaultTime=02:00:00 DisableRootJobs=NO Exclusive=NO GraceTime=0 Hidden=NO
   MaxNodes=UNLIMITED MaxTime=2-00:00:00 MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED MaxCPUsPerSocket=UNLIMITED
   Nodes=rye-[01-02]
   PriorityJobFactor=300 PriorityTier=3 RootOnly=NO ReqResv=NO OverSubscribe=NO
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=1024 TotalNodes=2 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerCPU=3000 MaxMemPerCPU=3000
   TRES=cpu=1024,mem=3000G,node=2,billing=1024

PartitionName=gpu
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=gpu
   DefaultTime=02:00:00 DisableRootJobs=NO Exclusive=NO GraceTime=0 Hidden=NO
   MaxNodes=UNLIMITED MaxTime=2-00:00:00 MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED MaxCPUsPerSocket=UNLIMITED
   Nodes=oat-[01-06]
   PriorityJobFactor=500 PriorityTier=5 RootOnly=NO ReqResv=NO OverSubscribe=NO
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=384 TotalNodes=6 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerCPU=4000 MaxMemPerCPU=4000
   TRES=cpu=384,mem=1500G,node=6,billing=384,gres/gpu=24
```

Key reads: `gpu` `MaxTime=2-00:00:00` = **48 h**. `AllowQos=ALL` (so `--qos=long` is legal here).
`PreemptMode=OFF`. `OverSubscribe=NO`. `MaxNodes=UNLIMITED`. `MaxMemPerCPU=4000` (MB).
`PriorityTier=5` — the `gpu` partition outranks `normal` (tier 1) and `bigmem` (tier 3).

### 1.5 `sacctmgr show qos ... -P` (all QOS)

```
Name|Priority|MaxWall|MaxTRES|MaxTRESPU|MaxJobsPU|MaxSubmitPU|GrpTRES|Flags
normal|100|||cpu=512,gres/gpu=1|128|1024||DenyOnLimit
interactive|200|||cpu=16,mem=64G|4|4||DenyOnLimit
dev|300|08:00:00||cpu=8,mem=32G|1|1||DenyOnLimit
long|0|7-00:00:00||cpu=128|4|32||DenyOnLimit,PartitionTimeLimit
caddyshack|400|||cpu=8,mem=32G|1|1||DenyOnLimit
bigmem|300|||mem=1.50T|4|32||DenyOnLimit
gpu|500|||gres/gpu=4|4|32||DenyOnLimit
```

- QOS `gpu`: **no `MaxWall`** (confirms the child agents), `MaxTRESPU=gres/gpu=4`,
  `MaxJobsPU=4`, `MaxSubmitPU=32`, no `GrpTRES`.
- QOS `long`: `MaxWall=7-00:00:00` **and the `PartitionTimeLimit` flag**, which is what lets a job
  exceed the partition's own 48 h `MaxTime`. `MaxTRESPU=cpu=128` (no GPU entry — but the *partition*
  QOS `gpu` still applies its `gres/gpu=4`, verified in §1.9).
- QOS `normal`: `gres/gpu=1` — **this is the origin of the "1 GPU" myth.**

### 1.6 The user's own association — this is what actually binds

```
=== sacctmgr show assoc user=ericrcwu ===
Cluster|Account|User|Partition|QOS|MaxJobs|MaxSubmit|MaxWall|MaxTRES|GrpTRES
farmshare|operator|ericrcwu||bigmem,caddyshack,dev,gpu,interactive,long,normal|||||

=== sacctmgr show user ericrcwu withassoc -P ===
User|Def Acct|Admin|Cluster|Account|Partition|Share|Priority|MaxJobs|MaxNodes|MaxCPUs|MaxSubmit|MaxWall|MaxCPUMins|QOS|Def QOS
ericrcwu|operator|None|farmshare|operator||1||||||||bigmem,caddyshack,dev,gpu,interactive,long,normal|
```

**The user's association imposes NO limits of its own** — `MaxJobs`, `MaxSubmit`, `MaxWall`,
`MaxTRES`, `GrpTRES` are all empty. The user is entitled to all 7 QOS including `long` and `gpu`.
So the binding limit is `min(partition MaxTime, QOS MaxWall)` with nothing from the association.

### 1.7 `scontrol show config` (relevant lines)

```
AccountingStorageTRES   = cpu,mem,energy,node,billing,fs/disk,vmem,pages,gres/gpu,gres/gpumem,gres/gpuutil
DefMemPerNode           = UNLIMITED
EnforcePartLimits       = ALL
GresTypes               = gpu
JobRequeue              = 1
MaxArraySize            = 1024
MaxJobCount             = 16384
MaxJobId                = 67043328
MaxMemPerNode           = UNLIMITED
PreemptMode             = OFF
PriorityDecayHalfLife   = 7-00:00:00
PriorityFavorSmall      = no
PriorityType            = priority/multifactor
PriorityWeightAge       = 1000000000
PriorityWeightAssoc     = 0
PriorityWeightFairShare = 5000000
PriorityWeightJobSize   = 1000000
PriorityWeightPartition = 20000
PriorityWeightQOS       = 10000
PriorityWeightTRES      = cpu=10000,mem=30000,GRES/gpu=50000
SchedulerType           = sched/backfill
SelectType              = select/cons_tres
SelectTypeParameters    = CR_CORE_MEMORY
```

`PriorityWeightAge=1e9` **dominates everything else by 200×** over fairshare (5e6). This cluster is
effectively FIFO-by-age with backfill. Consequence: **`--qos=long` costs you almost nothing in
priority** (QOS weight is only 1e4), and a low fairshare hurts far less than being submitted late.
`JobRequeue=1` = jobs are requeueable by default; `PreemptMode=OFF` = nothing will preempt you.
`MaxArraySize=1024`. `EnforcePartLimits=ALL` = over-limit jobs are rejected at submit, not left pending.

### 1.8 `--test-only` probes: the memory/CPU recipe (the disputed sub-claim)

```
### T1: HANDOFF recipe -p gpu --gres=gpu:1 -c 8 --mem=48G -t 6:00:00
sbatch: Job 1671430 to start at 2026-08-01T20:10:23 a using 8 processors on nodes oat-05 in partition gpu
--- exit=0
### T2: same but -t 48:00:00
sbatch: Job 1671431 to start at 2026-08-01T20:10:23 a using 8 processors on nodes oat-05 in partition gpu
--- exit=0
### T3: -c 8 --mem=32G (=8*4000MB)
sbatch: Job 1671432 to start at 2026-08-01T20:10:23 a using 8 processors on nodes oat-05 in partition gpu
--- exit=0
### T4: -c 13 --mem=48G
sbatch: Job 1671433 to start at 2026-08-01T19:10:23 a using 14 processors on nodes oat-05 in partition gpu
--- exit=0
### T5: -c 12 --mem=48000M
sbatch: Job 1671434 to start at 2026-08-01T20:10:23 a using 12 processors on nodes oat-05 in partition gpu
--- exit=0
```

Control probes proving `--test-only` *does* reject things (so T1–T5 acceptances are meaningful):

```
### R1: -t 49:00:00 (over 48h partition MaxTime), default qos
allocation failure: Requested time limit is invalid (missing or exceeds some limit)
--- exit=1
### R2: --mem=300G (over node 250G)
allocation failure: Requested node configuration is not available
--- exit=1
### R3: --gres=gpu:5 one node
sbatch: error: QOSMaxGRESPerUser
allocation failure: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
--- exit=1
### R4: --mem-per-cpu=6G (over MaxMemPerCPU=4000)
sbatch: Job 1671437 to start at 2026-08-01T20:10:56 a using 8 processors on nodes oat-05 in partition gpu
--- exit=0
### R5: --gres=gpu:8 one node
sbatch: error: QOSMaxGRESPerUser
allocation failure: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
--- exit=1
```

**`-c 8 --mem=48G` is NOT rejected.** Neither is `--mem-per-cpu=6G` (R4), which nominally exceeds
`MaxMemPerCPU=4000`. Slurm's `MaxMemPerCPU` on this cluster behaves as documented upstream: when the
request exceeds it, Slurm **automatically increases the CPU count** to cover the memory rather than
denying the job. Empirically confirmed on a *real completed* job in §1.11: requested `cpu=8,mem=48G`,
allocated `cpu=14,mem=48G`.

### 1.9 `--test-only` probes: multi-GPU and the `long` QOS

```
### G2: --gres=gpu:2 -N1
sbatch: Job 1671438 to start at 2026-08-01T19:55:50 a using 16 processors on nodes oat-03 in partition gpu
--- exit=0
### G4: --gres=gpu:4 -N1
sbatch: Job 1671439 to start at 2026-08-01T11:55:50 a using 32 processors on nodes oat-03 in partition gpu
--- exit=0
### G4x2nodes: -N2 --gres=gpu:4 (8 gpus total)
sbatch: error: QOSMaxGRESPerUser
--- exit=1
### L1: --qos=long -t 5-00:00:00 gpu:1
sbatch: Job 1671440 to start at 2026-08-01T20:11:14 a using 8 processors on nodes oat-05 in partition gpu
--- exit=0
### L2: --qos=long -t 7-00:00:00 gpu:1
sbatch: Job 1671441 to start at 2026-08-01T20:11:14 a using 8 processors on nodes oat-05 in partition gpu
--- exit=0
### L3: --qos=long -t 8-00:00:00 gpu:1
sbatch: error: QOSMaxWallDurationPerJobLimit
--- exit=1
### L4: --qos=long --gres=gpu:2
sbatch: Job 1671442 to start at 2026-08-01T19:55:50 a using 16 processors on nodes oat-03 in partition gpu
--- exit=0
### L5: --qos=long --gres=gpu:4 -N1 -t 7d
sbatch: Job 1671444 to start at 2026-08-01T11:55:50 a using 32 processors on nodes oat-03 in partition gpu
--- exit=0
### L6: --qos=long --gres=gpu:5
sbatch: error: QOSMaxGRESPerUser
--- exit=1
### L7: --qos=long -N2 --gres=gpu:4 (8 total)
sbatch: error: QOSMaxGRESPerUser
--- exit=1
### L8: --qos=long -c 130 (over cpu=128 MaxTRESPU)
sbatch: error: QOSMaxCpuPerUserLimit
--- exit=1
### N1: -p normal --gres=gpu:1 -t 48h
sbatch: Job 1671443 to start at 2026-08-01T09:51:14 a using 8 processors on nodes oat-05 in partition normal
--- exit=0
### N2: -p normal --gres=gpu:2
sbatch: error: QOSMaxGRESPerUser
--- exit=1
### Nlong: -p normal --qos=long --gres=gpu:1 -t 7d
sbatch: Job 1671451 to start at 2026-08-01T09:57:21 a using 8 processors on nodes oat-05 in partition normal
--- exit=0
### Nlong2: -p normal --qos=long --gres=gpu:2
sbatch: error: QOSMaxGRESPerUser
--- exit=1
```

Reads: `--qos=long` on `-p gpu` gives **7 days** and **still allows 4 GPUs** (the partition QOS `gpu`
supplies the `gres/gpu=4` cap; QOS `long` supplies the 7-day `MaxWall` and a `cpu=128` cap). 8 days is
rejected. `-p normal` caps you at **1 GPU** regardless of QOS.

### 1.10 `--test-only` probes: submit/array caps

```
### A24: --array=0-23 gpu:1        → accepted (Job 1671449)
### A27: --array=0-26 (27 jobs; user already had 5 queued → exactly 32) → accepted (Job 1671450)
### A28: --array=0-27 (→33)        → sbatch: error: QOSMaxSubmitJobPerUserLimit
### A32: --array=0-31 (→37)        → sbatch: error: QOSMaxSubmitJobPerUserLimit
### A48pct4: --array=0-47%4        → sbatch: error: QOSMaxSubmitJobPerUserLimit
```

**Critical operational trap:** `MaxSubmitPU=32` counts **array elements**, and `%4` throttling does
**not** exempt them. `--array=0-47%4` is rejected outright. Any campaign with >32 runs must be
submitted in waves of ≤32 elements (minus whatever is already queued), not as one throttled array.

### 1.11 The user's ACTUAL GPU history — settles "KDA already ran 20-hour jobs"

`sacct -u ericrcwu -S 2026-06-01 -X -o ... -P | grep gres/gpu` → **570 GPU jobs**. Longest:

```
1662404_13|run_lm_grid.sbatch|gpu|normal|billing=18,cpu=18,gres/gpu=1,mem=64G,node=1|20:00:20|20:00:00|TIMEOUT|2026-07-27T14:47:09
1662404_4 |run_lm_grid.sbatch|gpu|normal|...gres/gpu=1,mem=64G,node=1|11:58:38|20:00:00|COMPLETED|2026-07-26T21:48:18
1662404_10|run_lm_grid.sbatch|gpu|normal|...gres/gpu=1,mem=64G,node=1|11:57:38|20:00:00|COMPLETED|2026-07-27T09:47:02
1662404_1 |run_lm_grid.sbatch|gpu|normal|...gres/gpu=1,mem=64G,node=1|11:53:31|20:00:00|COMPLETED|2026-07-26T15:00:16
1662404_7 |run_lm_grid.sbatch|gpu|normal|...gres/gpu=1,mem=64G,node=1|11:53:22|20:00:00|COMPLETED|2026-07-27T02:53:47
1671018_0 |core6-stage0a     |gpu|normal|...gres/gpu=1,mem=48G,node=1|09:25:52|20:00:00|COMPLETED|2026-07-31T18:23:26
1671018_1 |core6-stage0a     |gpu|normal|...gres/gpu=1,mem=48G,node=1|08:42:32|20:00:00|COMPLETED|2026-07-31T18:23:26
```

**Job `1662404_13` ran 20 h 00 m 20 s against a `20:00:00` TimeLimit and hit TIMEOUT.** A 6-hour cap
could not have produced that row. The claim is empirically confirmed. Fourteen more jobs in that
array carried 20 h limits, and `1671018_*` (as recently as 2026-07-31) also used `-t 20:00:00`.

**And the `-c 8 --mem=48G` bump, on a real job:**

```
JobID|ReqTRES|AllocTRES|ReqCPUS|AllocCPUS|NCPUS
1671018_0|billing=8,cpu=8,gres/gpu=1,mem=48G,node=1|billing=14,cpu=14,...,gres/gpu=1,mem=48G,node=1|13|14|14
1662404_0|billing=8,cpu=8,gres/gpu=1,mem=64G,node=1|billing=18,cpu=18,...,gres/gpu=1,mem=64G,node=1|...
```

Requested `cpu=8`, **allocated `cpu=14`** and it ran to COMPLETED. This is the definitive refutation.

### 1.12 Live contention — `squeue -p gpu`

```
     JOBID PARTITION     USER ST       TIME  TIME_LEFT  NODES     NODELIST(REASON) TRES_PER_NODE
   1671330       gpu   ysinan PD       0:00 2-00:00:00      1         (Dependency) gres/gpu:1
   1671337       gpu   hmolin PD       0:00    8:00:00      1         (Dependency) gres/gpu:2
   1671386       gpu  iriscai PD       0:00    6:00:00      1         (Dependency) gres/gpu:1
1671411_[0       gpu ericrcwu PD       0:00    4:00:00      1         (Dependency) gres/gpu:1
   1671186       gpu   hmolin PD       0:00   12:00:00      1        (JobHeldUser) gres/gpu:3
   1671182       gpu   hmolin PD       0:00   12:00:00      1        (JobHeldUser) gres/gpu:3
   1671005       gpu   hmolin PD       0:00   12:00:00      1        (JobHeldUser) gres/gpu:3
   1669087       gpu calebs06 PD       0:00 2-00:00:00      1        (JobHeldUser) gres/gpu:1
   1669343       gpu   wgpeng  R 1-21:53:51    2:06:09      1               oat-03 gres/gpu:4
   1670899       gpu calebs06  R    9:24:45 1-14:35:15      1               oat-04 gres/gpu:1
   1670900       gpu calebs06  R    7:28:30 1-16:31:30      1               oat-02 gres/gpu:1
   1671329       gpu   ysinan  R    2:47:17 1-21:12:43      1               oat-01 gres/gpu:1
   1670561       gpu   ysinan  R 1-04:58:41   18:01:19      1               oat-04 gres/gpu:1
   1671384       gpu   nzhao2  R    1:50:50   10:09:10      2          oat-[01,06] gres/gpu:2
   1664922       gpu  banksaj  R 4-00:55:24   23:04:36      1               oat-06 gres/gpu:1
   1671312       gpu   hmolin  R    6:13:00    1:47:00      1               oat-05 gres/gpu:2
 1671406_5       gpu ericrcwu  R      16:24    1:13:36      1               oat-02 gres/gpu:1
```

Note `1669343` (`wgpeng`, `gres/gpu:4` on one node, 48 h limit) — **4-GPU single-node jobs demonstrably
run here.** And `1664922` (`banksaj`) has been **running 4 days 55 min**:

```
JobId=1664922 JobName=jl7b-fit-r   UserId=banksaj  Account=operator QOS=long
   Requeue=1 Restarts=0
   RunTime=4-00:55:41 TimeLimit=5-00:00:00
   Partition=gpu   NumNodes=1 NumCPUs=16
   AllocTRES=cpu=16,mem=56G,node=1,billing=16,gres/gpu=1
```

**A live, 5-day-TimeLimit GPU job on this partition, via `--qos=long`.** Independent proof that the
48 h partition cap is not the hard ceiling.

Current GPU occupancy: `sinfo -p gpu -O NodeList,Gres,GresUsed`

```
oat-01          gpu:4(S:0-1)        gpu:3(IDX:0-2)          mixed
oat-[02,06]     gpu:4(S:0-1)        gpu:4(IDX:0-3)          mixed
oat-04          gpu:4(S:0-1)        gpu:2(IDX:0-1)          mixed
oat-05          gpu:4(S:0-1)        gpu:3(IDX:0-1,3)        mixed
oat-03          gpu:4(S:0-1)        gpu:4(IDX:0-3)          allocated
```

**20 of 24 GPUs busy (83%)** at survey time. 4 free (1 on oat-01, 2 on oat-04, 1 on oat-05).

### 1.13 Cluster-wide QOS usage on `-p gpu`, last 30 days

```
      3 gpu|dev
   2811 gpu|gpu
     11 gpu|long
   2699 gpu|normal
```

`long` is used by **11 jobs out of 5,524** — it is almost entirely unclaimed headroom. Also note
2,699 jobs went in as QOS `normal` on the `gpu` partition (i.e. `-p normal` with a GPU gres), each
self-limited to 1 GPU. The community is largely making the same mistake the plan made.

Examples of other users' long-running 4-GPU jobs (all `QOS=gpu`, `TimeLimit=2-00:00:00`):

```
1630075|wgpeng|gpu|gpu|billing=64,cpu=64,gres/gpu=4,mem=250G,node=1|2-00:00:06|2-00:00:00|TIMEOUT
1632154|wgpeng|gpu|gpu|billing=64,cpu=64,gres/gpu=4,mem=250G,node=1|2-00:00:17|2-00:00:00|TIMEOUT
1629692|nkozak|gpu|gpu|billing=16,cpu=16,gres/gpu=4,mem=62.50G,node=1|2-00:00:28|2-00:00:00|TIMEOUT
1629621|wgpeng|gpu|gpu|billing=64,cpu=64,gres/gpu=4,mem=250G,node=1|1-18:50:10|2-00:00:00|COMPLETED
1632312|hmolin |gpu|gpu|billing=42,cpu=42,gres/gpu=3,mem=160G,node=1|2-00:00:13|2-00:00:00|TIMEOUT
```

The full 48 h × 4 GPU envelope is routinely used to exhaustion by other users.

### 1.14 Fairshare

```
Account|User|RawShares|NormShares|RawUsage|NormUsage|EffectvUsage|FairShare|LevelFS|GrpTRESMins|TRESRunMins
operator|ericrcwu|1|0.000294|16143667|0.013077|0.013077|0.005285|0.022459||cpu=596,...,gres/gpu=74,...
```

`FairShare=0.005285` is low (the user has consumed 1.3% of cluster usage against a 0.03% share) —
but with `PriorityWeightFairShare=5e6` vs `PriorityWeightAge=1e9`, fairshare contributes at most
0.5% of the priority a one-week-old job accrues from age. **Fairshare is not a practical brake here.**
`PriorityDecayHalfLife=7-00:00:00`, so heavy usage washes out in a week.

### 1.15 Queue-wait empirics (user's own 570 GPU jobs since 2026-07-01)

```
n=570 min=0.0 p25=0.0 median=1.2 p75=5.8 p90=15.8 max=1698.6 (minutes)
```

Median wait **1.2 min**; p90 **16 min**. But this is dominated by short jobs. The *long* jobs waited
much longer — the 20 h array `1662404`:

```
1662404_14 wait_min=1698.6  (28.3 h)
1662404_13 wait_min=1426.9  (23.8 h)
1662404_12 wait_min=1290.9  (21.5 h)
1662404_11 wait_min=1258.0  (21.0 h)
1662404_10 wait_min=1126.8  (18.8 h)
1662404_9  wait_min=850.9
1662404_8  wait_min=849.7
1662404_7  wait_min=713.5
1662404_6  wait_min=441.3
1662404_5  wait_min=408.3
1662404_4  wait_min=408.1
```

Most of that "wait" is self-inflicted: only 4 can run at once, so elements 4-14 queue behind 0-3.
**The right metric is achieved concurrency.** Array `1662404` (15 elements) start/end times:

```
1662404_0 |2026-07-26T15:00:16|2026-07-26T21:48:07
1662404_3 |2026-07-26T15:00:16|2026-07-26T21:48:32
1662404_2 |2026-07-26T15:00:16|2026-07-26T22:21:18
1662404_1 |2026-07-26T15:00:16|2026-07-27T02:53:47
1662404_4 |2026-07-26T21:48:18|2026-07-27T09:46:56
...
1662404_13|2026-07-27T14:47:09|2026-07-28T10:47:29
```

- 4 elements started **simultaneously at 15:00:16** — *4 concurrent GPUs is real, not nominal.*
- Sum of elapsed = **131.1 GPU-h**; span submit→last-finish = **43.79 h**.
- **Achieved sustained concurrency = 131.1 / 43.79 = 2.99 GPUs, i.e. 75% of the nominal 4.**

That 0.75 derate (gaps between one element finishing and the next being dispatched) is the honest
"realistic queueing" factor and is what I use below.

### 1.16 Other checks

- `/etc/motd` — does not exist. `/etc/slurm/` contains only `slurm.jwks` (mode `-r--------`, root).
  `/usr/local/doc` does not exist. **No local policy document contradicts or supplements the above.**
- Scratch: `truenas.farmshare.sunet:/mnt/tank/scratch/users  106T  37T  69T  35%` — 69 TB free,
  no checkpoint-storage constraint.
- `PreemptMode=OFF` at both partition and global scope: **`gpu` jobs are never preempted.**
  `JobRequeue=1` means requeue-on-node-failure is on by default, but there is no preemption to requeue from.
- No hidden partition exists (`scontrol show partition` returns exactly 3, none `Hidden=YES`).

---

## 2. The six precise answers

### 2.1 Max walltime on the GPU partition

| Layer | Value |
|---|---|
| Partition `gpu` `MaxTime` | `2-00:00:00` = **48 h** |
| QOS `gpu` (partition QOS) `MaxWall` | **none** |
| QOS `long` `MaxWall` | `7-00:00:00` = **168 h**, with flag `PartitionTimeLimit` |
| User association `MaxWall` | **none** (empty) |

**Default path (`-p gpu`, no `--qos`): the binding limit is 48 h, set by the partition `MaxTime`.**
Verified: `-t 49:00:00` → `Requested time limit is invalid`.

**With `--qos=long`: the binding limit is 168 h (7 days), set by QOS `long` `MaxWall`.** The
`PartitionTimeLimit` flag makes the job QOS's `MaxWall` *override* the partition's 48 h rather than
intersect with it. Verified two ways: `--test-only -t 7-00:00:00` accepted, `-t 8-00:00:00` rejected
with `QOSMaxWallDurationPerJobLimit`; and job `1664922` is live at RunTime 4-00:55 / TimeLimit 5-00:00.

The child agents' "48 h" is correct for the default path but is **not the ceiling** — they missed the
7-day `long` QOS, which is the more useful number for this plan.

### 2.2 Max concurrent GPUs per user

**4 GPUs, and separately 4 jobs.** From partition QOS `gpu`: `MaxTRESPU=gres/gpu=4`, `MaxJobsPU=4`.

These are two independent caps and they are *not* the same thing:
- 4 × (1-GPU job) = 4 GPUs, 4 jobs → **legal, both caps exactly saturated.**
- 1 × (4-GPU job) = 4 GPUs, 1 job → **legal, GPU cap saturated, 3 job slots unused and useless.**
- 2 × (2-GPU job) = 4 GPUs, 2 jobs → **legal.**
- 4 × (2-GPU job) = 8 GPUs → **illegal** (`QOSMaxGRESPerUser`), even though jobs ≤ 4.

So **`gres/gpu=4` is the real currency; `MaxJobsPU=4` only bites if you run 1-GPU jobs**, in which case
the two caps coincide. Empirically saturated: array `1662404` had exactly 4 elements start at the
same second (§1.15).

Additional cap: **`MaxSubmitPU=32`** (queued + running, counting array elements). Verified at the
exact boundary — with 5 jobs already queued, `--array=0-26` (27 more = 32) was accepted and
`--array=0-27` (33) was rejected.

### 2.3 Are multi-GPU single-node jobs permitted?

**Yes.** `--gres=gpu:2 -N1` and `--gres=gpu:4 -N1` both pass `--test-only` (jobs 1671438, 1671439),
and other users run them for real (`wgpeng` job 1669343, `gres/gpu:4` on `oat-03`, 48 h limit;
`nkozak` 1629692, 4 GPUs, TIMEOUT at 2-00:00:28).

- **Largest GPU node has 4 GPUs.** All six `oat-*` nodes are identical: `gpu:4(S:0-1)`, 64 CPUs, 250 GB.
- **`--gres=gpu:4` on one node schedules routinely.** It is exactly the per-user cap, so it is the
  maximum single job possible.
- **Multi-node is impossible for this user**: `-N2 --gres=gpu:4` = 8 GPUs, rejected with
  `QOSMaxGRESPerUser` (both with default QOS and `--qos=long`). So **no distributed multi-node training.**
- Interconnect caveat: L40S is a PCIe card with no NVLink advertised. A 4-GPU DDP job all-reduces over
  PCIe, so expect ~80-90% scaling efficiency, not ~100%. I use **0.85** below.

### 2.4 Is `-c 8 --mem=48G` really rejected?

**No. REFUTED.** It is accepted and it runs. Two independent proofs:

1. `sbatch --test-only -p gpu --gres=gpu:1 -c 8 --mem=48G -t 48:00:00` → `Job 1671431 to start at ...`,
   exit 0. (And the control probes R1/R2/R3/R5 show `--test-only` genuinely rejects illegal requests,
   so this acceptance is meaningful.)
2. Real completed job `1671018_0`: `ReqTRES=cpu=8,...,mem=48G` → `AllocTRES=cpu=14,...,mem=48G`,
   ran 09:25:52, `COMPLETED`.

**What `MaxMemPerCPU=4000` actually does:** when `mem / cpus > 4000 MB`, Slurm *silently raises the
CPU count* to satisfy the ratio. It does not deny the job. So the HANDOFF recipe works — but it
quietly bills you for 14 CPUs, not 8.

**Correct minimum `-c` for 48 G:**
- `--mem=48G` is 49,152 MB → `ceil(49152/4000) = 13` CPUs minimum; the scheduler rounds to **14** because
  `ThreadsPerCore=2` forces an even CPU count. (Verified: `-c 13 --mem=48G` → "using 14 processors".)
- `--mem=48000M` is 48,000 MB → exactly **12** CPUs. (Verified: `-c 12 --mem=48000M` → "using 12 processors".)

**Corrected submit line to put in HANDOFF and the probe docstrings:**

```bash
# Default path — 48 h ceiling, explicit CPU count so the allocation is not silently inflated
sbatch -p gpu --gres=gpu:1 -c 14 --mem=48G -t 48:00:00 ...
#   (equivalently, and tidier: -c 12 --mem=48000M)

# Long path — 7 day ceiling, same 4-GPU cap
sbatch -p gpu --qos=long --gres=gpu:1 -c 14 --mem=48G -t 7-00:00:00 ...

# Max single job — all 4 GPUs on one node, 7 days
sbatch -p gpu --qos=long --gres=gpu:4 -N 1 -c 32 --mem=120G -t 7-00:00:00 ...
```

The material bug in the old recipe is **not** that it fails — it is that `-c 8 --mem=48G` charges
`billing=14` while the script's dataloader is configured for 8 workers. You get 14 CPUs and use 8.

### 2.5 Any other caps the plan would trip

| Cap | Value | Does the plan trip it? |
|---|---|---|
| `MaxSubmitPU` (QOS `gpu`) | **32**, counts array elements | **YES — hard trip.** Stage (a) is 24 runs (fits, barely); the reassessment's replacement is 32 runs (fits only with an empty queue); `--array=0-47%4` is **rejected outright**. Must submit in waves. |
| `MaxJobsPU` (QOS `gpu`) | 4 | Yes, by design — this is the concurrency limit. |
| `MaxTRESPU` (QOS `gpu`) | `gres/gpu=4` | Yes, by design. |
| `MaxTRESPU` (QOS `long`) | `cpu=128` | Only if you run 4 concurrent jobs at >32 CPUs each. `-c 14` × 4 = 56, safe. Verified `-c 130` → `QOSMaxCpuPerUserLimit`. |
| `MaxMemPerCPU` | 4000 MB | Not a rejection, an auto-bump. See §2.4. |
| `GrpTRES` | **none anywhere** | No. |
| Association limits | **all empty** | No. |
| `MaxArraySize` | 1024 | No (`MaxSubmitPU=32` bites 32× sooner). |
| Node memory | 250 GB usable | No — a 750M model + AdamW is ~12 GB. |
| GPU memory | 48 GB/card | No — 750M bf16 weights 1.5 GB + fp32 optimizer states ~9 GB + grads; fits comfortably. |
| Preemption | `PreemptMode=OFF` | **No — jobs on `gpu` are NOT preemptible.** No requeue-on-preempt to worry about. |
| Fairshare | `FairShare=0.0053` (low) | Negligible: `PriorityWeightAge=1e9` swamps `PriorityWeightFairShare=5e6` by 200×. |
| Scratch space | 69 TB free | No. |

**Is there a higher-limit partition or adjacent QOS?** Yes, and the prior agents missed it:
- **`--qos=long` on `-p gpu`** raises walltime 48 h → **168 h** at no GPU-count cost and negligible
  priority cost. Only 11 jobs cluster-wide used it in 30 days — it is free headroom.
- There is **no** partition with more than 4 GPUs/user or more than 24 GPUs total. `bigmem` has no GPUs.
  `normal` has the same GPU nodes but a *worse* QOS (`gres/gpu=1`) and lower `PriorityTier` (1 vs 5).
- **Never submit GPU work to `-p normal`.** It costs you 3 of your 4 GPUs and drops you a priority tier.

### 2.6 Real-world throughput

- **Typical queue wait, 1-GPU job:** median **1.2 min**, p75 5.8 min, p90 16 min (n=570).
  Short jobs land essentially instantly — the cluster's backfill scheduler is generous to them.
- **Long jobs are different.** A 20 h 1-GPU job in a saturated array waited 7-28 h, but that is queueing
  behind *your own* 4-job cap, not against other users.
- **Is 4 concurrent achievable, or nominal? Achievable.** Array `1662404` started 4 elements in the
  same second, and sustained **2.99 GPUs averaged over a 43.8 h span** — **75% of nominal.**
  The 25% loss is dispatch latency between one element ending and the next starting, not contention.
- Current cluster load is **20/24 GPUs busy (83%)**, with other users routinely holding 4-GPU × 48 h
  jobs to TIMEOUT. So the 4 slots are attainable but you will sometimes wait hours for the 4th.

**Planning factor: 4 nominal concurrent GPUs, 3.0 realistic.** I use both below.

---

## 3. Feasibility

### 3.1 Assumptions, stated explicitly

- **FLOPs = 6ND** (N params, D tokens), the project's convention.
- **N for the headline arm = 354,483,968** (`L0`, frozen). I use **total** parameters. If you use
  non-embedding N instead (vocab 65536 × d 1024 = 67.1 M embedding, tied), every 350M number below
  drops ~19%; untied, ~38%. **Stated so the reader can rescale.**
- **L40S bf16 peak: 181.05 TFLOP/s dense.** NVIDIA's L40S datasheet lists BF16 Tensor Core as
  *362.05 TFLOPS\**, where the asterisk is **with structural sparsity**. Dense (sparsity off) is
  **half that: 181.05 TFLOP/s.** I use **181 TFLOP/s** as peak throughout. (This also happens to be
  the number the reassessment implicitly used: its "45 TFLOP/s = 25% MFU" implies peak 180.)
- **MFU: 25% realistic (45.3 TFLOP/s), 40% optimistic (72.4 TFLOP/s).** Per the reassessment. Sanity
  check from this cluster's own data: the KDA gated-delta-net arm achieved ~18.4 TFLOP/s ≈ 10% MFU,
  so 25% for a GEMM-dominant conv/GQA hybrid is a genuine (not conservative-to-the-point-of-useless)
  planning figure, and 40% would require real kernel work.
- **Multi-GPU DDP scaling: 0.85** (PCIe all-reduce, no NVLink on L40S).
- **Realistic concurrency: 3.0** of the nominal 4 (measured, §1.15).
- One 48 h × 4-GPU window = **192 L40S-hours**.

### 3.2 Per-run cost and walltime

| Stage | N | D | FLOPs/run | h/run @25% | h/run @40% | h/run, 4 GPU @25% | h/run, 4 GPU @40% |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) rank screen | 150 M | 10 B | 9.00e18 | **55.2** | 34.5 | 16.2 | 10.1 |
| (b) confirm | 354.5 M | 20 B | 4.254e19 | **260.8** | 163.2 | 76.7 | 48.0 |
| (c) headline | 750 M | 50 B | 2.250e20 | **1379.7** | 863.3 | 405.8 | 253.9 |
| (d) reassessment replacement | 354.5 M | 2 B | 4.254e18 | **26.1** | 16.3 | 7.7 | 4.8 |

### 3.3 Campaign totals

| Stage | Runs | Total L40S-h @25% | @40% | 192-h windows @25% | @40% | Calendar d, perfect (4 GPU) @25% | @40% | Calendar d, realistic (3.0 GPU) @25% | @40% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **(a)** rank screen 24 × 150M × 10B | 24 | **1,324** | 828 | 6.9 | 4.3 | **13.8** | 8.6 | **18.4** | 11.5 |
| **(b)** confirm 5 × 350M × 20B | 5 | **1,304** | 816 | 6.8 | 4.3 | **13.6** | 8.5 | **18.1** | 11.3 |
| **(c)** headline 3 × 750M × 50B | 3 | **4,139** | 2,590 | 21.6 | 13.5 | **43.1** | 27.0 | **57.5** | 36.0 |
| **(a)+(b)+(c) full program** | 32 | **6,767** | 4,234 | 35.2 | 22.1 | **70.5** | 44.1 | **94.0** | 58.8 |
| **(d)** replacement: 4 arms × 350M × 2B × 8 seeds | 32 | **834** | 522 | 4.3 | 2.7 | **8.7** | 5.4 | **11.6** | 7.3 |

### 3.4 What is now feasible that was not, and what still is not

**Newly feasible under 48 h / 4 GPU (was impossible under 6 h / 1 GPU):**

- **(d) The reassessment's replacement study — 4 arms × 350M × 2B × 8 seeds.** Each run is
  **26.1 h @25% MFU**, which is *impossible* in a 6 h job and *comfortable* in a 48 h job with
  22 h of headroom. The whole 32-run campaign is **834 L40S-h ≈ 12 calendar days** realistic,
  **8.7 days** perfect. **This is the single biggest win and it is unambiguous.** Under the old belief
  this study could not be run at 350M at all — you would have been forced down to the 20-50M scale the
  reassessment retreated to.
- **(a) The rank screen, 24 × 150M × 10B.** 55.2 h/run @25% narrowly *exceeds* 48 h — but it fits in
  `--qos=long` (168 h) with 3× headroom, or in a 4-GPU job at 16.2 h, or trivially at 40% MFU (34.5 h).
  Campaign: **1,324 L40S-h ≈ 18 calendar days** realistic. **Feasible, if slow.**
- **(b) The confirm stage, 5 × 350M × 20B.** 260.8 h/run @25% exceeds even the 7-day `long` QOS on
  1 GPU. But at **4 GPUs it is 76.7 h**, which fits inside `--qos=long` (168 h) with 2.2× headroom.
  Campaign: **1,304 L40S-h ≈ 18 calendar days** realistic — but see the caveat in §3.6, because
  running 4-GPU jobs consumes your entire concurrency budget and serializes the 5 runs.
  **Feasible with a caveat.**

**Still not feasible:**

- **(c) The headline stage, 3 × 750M × 50B.** **4,139 L40S-h @25%, 2,590 @40%.** At realistic
  concurrency that is **57.5 calendar days** (25% MFU) or **36 days** (40% MFU) of *saturated,
  uninterrupted* FarmShare. A single run is **1,380 h on one GPU** — 8.2× the 7-day QOS ceiling — or
  **406 h even on all four GPUs**, still 2.4× over the 7-day ceiling. There is no submit form that
  runs one of these to completion. **This stage remains out of reach on FarmShare** and needs either
  checkpoint-resume across ~3 seven-day jobs per run (and ~2 months of calendar), or the SB-AWS
  8-GPU box.
- **The full (a)+(b)+(c) program.** **6,767 L40S-h @25%**, i.e. **94 calendar days realistic /
  70.5 perfect.** At the optimistic 40% MFU it is still **58.8 / 44.1 days.**

**Quantifying the gap to the "8-day 8×A100" program.** The design doc costed the program at 8×A100
@40% MFU as 2.5 h/arm (150M/10B) + ~12 h/arm (350M/20B) + 2.6 d/arm (750M/50B). Reconstructing:
8 × A100 × 312 TFLOP/s × 0.40 = **998.4 TFLOP/s aggregate**, giving 2.50 h / 11.7 h / 62.6 h — which
reproduces the doc's numbers, so the comparison is like-for-like. Total program on that box:
24(2.50) + 5(11.7) + 3(62.6) = **306 h of 8×A100 wall = 12.8 days**, or **2,450 A100-GPU-hours.**

The same program on FarmShare's 4 L40S is **6,767 L40S-h (25% MFU) / 4,234 (40%)**. The aggregate
throughput ratio is the whole story:

| | aggregate TFLOP/s | program wall-clock |
|---|---:|---:|
| 8 × A100 @40% MFU | 998.4 | **12.8 days** |
| 4 × L40S @40% MFU (perfect queue) | 289.6 | **44.1 days** |
| 4 × L40S @25% MFU (perfect queue) | 181.0 | **70.5 days** |
| 3.0 × L40S @25% MFU (realistic) | 135.8 | **94.0 days** |

**FarmShare at its best is 3.4× slower than the 8×A100 box; at its realistic worst, 7.3× slower.**
So: *the relaxation helps a lot — it is a 4× throughput gain and, more importantly, it unblocks
per-job walltimes that 6 h forbade entirely — but the 13-day 8×A100 program becomes a 6-to-13-week
FarmShare program, which is not a capstone timeline.* The honest framing is that **FarmShare can now
host stages (a), (b), and the reassessment's replacement (d), but cannot host stage (c).**

### 3.5 Checkpoint-resume requirement

The plan has no checkpoint-resume story. Here is exactly where that bites.

| Stage | 1 GPU, 48 h cap | 1 GPU, 7 d (`--qos=long`) | 4 GPU, 48 h | 4 GPU, 7 d |
|---|---|---|---|---|
| (a) 150M/10B | @25% **55.2 h — NEEDS RESUME** (2 chunks); @40% 34.5 h fits | fits both (55.2 h ≪ 168 h) | fits both (16.2 / 10.1 h) | fits |
| (b) 350M/20B | 260.8 / 163.2 h — **NEEDS RESUME** (6 / 4 chunks) | @25% 260.8 h **NEEDS RESUME** (2 chunks); @40% 163.2 h fits with only a **3% margin — unsafe** | @25% 76.7 h **NEEDS RESUME**; @40% 48.0 h — **exactly at the limit, unsafe** | **fits both** (76.7 / 48.0 h ≪ 168 h) |
| (c) 750M/50B | **NEEDS RESUME** (29 / 18 chunks) | **NEEDS RESUME** (9 / 6 chunks) | **NEEDS RESUME** (9 / 6 chunks) | **NEEDS RESUME** (3 / 2 chunks) |
| (d) 350M/2B | **fits** (26.1 / 16.3 h) | fits | fits | fits |

**Bottom line on checkpointing:**
- **(d) needs no checkpoint-resume at all** at 25% MFU on a plain 48 h 1-GPU job. This is another
  reason it is the right next experiment.
- **(a) needs none if you add `--qos=long`.** One flag removes the requirement.
- **(b) needs none if you use `--gres=gpu:4 --qos=long`.** But that serializes the 5 runs
  (see §3.6), so in practice you will want resume anyway.
- **(c) requires checkpoint-resume unconditionally** — no combination of GPUs and QOS on this cluster
  fits a single 750M/50B run. Minimum 2 chunks (4 GPU, 7 d, 40% MFU); realistically 9.
- Independent of walltime: the project memory records that **"a machine has died mid-run"**, and this
  survey found a real 20 h job (`1662404_13`) lost to TIMEOUT after 20 h of compute. **Checkpoint-resume
  is worth building regardless**, and 69 TB of free scratch means storage is not a reason not to.

### 3.6 Strategic note: 4 × 1-GPU beats 1 × 4-GPU for this workload

Because `MaxTRESPU=gres/gpu=4` is a *user-wide* cap, a single 4-GPU job consumes your entire budget.
So:

- **Total GPU-hours go UP** with multi-GPU, by 1/0.85 ≈ **1.18×** (PCIe all-reduce overhead), while
  aggregate throughput stays flat at 4 GPUs.
- **Multi-GPU buys exactly one thing: lower per-run walltime**, which matters only when a single run
  must fit under a ceiling.
- For a throughput-bound campaign of independent runs — which is what (a) and (d) are —
  **4 concurrent 1-GPU jobs is strictly better.**
- Use `--gres=gpu:4` only for stage (b) and (c), where a single run otherwise cannot fit,
  and expect to pay ~18% more GPU-hours for the privilege.

### 3.7 Operational corrections the plan needs

1. Replace `-c 8 --mem=48G` with **`-c 14 --mem=48G`** (or `-c 12 --mem=48000M`) so the allocation is
   explicit rather than silently inflated to 14.
2. Always `-p gpu`, **never `-p normal`** — `normal` caps you at 1 GPU and is a lower priority tier.
   This is almost certainly where the "1 GPU" belief came from.
3. Always pass `-t` explicitly. `DefaultTime=02:00:00` — an omitted `-t` silently gives you 2 h.
4. Add `--qos=long` for anything over 48 h. It costs ~nothing in priority and only 11 jobs
   cluster-wide used it last month.
5. **Never use `--array=N-M%K` for more than 32 elements.** `%K` throttling does not exempt elements
   from `MaxSubmitPU=32`; `--array=0-47%4` is rejected at submit. Submit in waves of ≤32.
6. Note `MaxTRESPU cpu=128` under QOS `long`: 4 concurrent jobs at `-c 32` = 128, exactly at the cap.
   With `-c 14` you are at 56 and safe.

---

## 4. Bottom line

### CONFIRMED — with two corrections and one significant addition

**"FarmShare is 48 h / 4 concurrent GPUs, not 6 h / 1 GPU": CONFIRMED.**
- 48 h: `scontrol show partition gpu` → `MaxTime=2-00:00:00`; `-t 49:00:00` rejected.
- 4 concurrent GPUs: QOS `gpu` `MaxTRESPU=gres/gpu=4`, `MaxJobsPU=4`; `--gres=gpu:5` rejected;
  4 array elements observed starting in the same second.
- "KDA already ran 20-hour jobs": **CONFIRMED** — job `1662404_13` ran `20:00:20` against a
  `20:00:00` TimeLimit. A 6 h cap is arithmetically impossible.
- The "6 h" figure appears **nowhere in Slurm** and traces to an unverified assertion at
  `/Users/ericwu/Developer/Capstone_LLM/docs/liv-brainlift-experiment-design.md:1401`.

**Correction 1 — the sub-claim "`--mem=48G -c 8` is rejected because `MaxMemPerCPU=4000`, so the
printed recipe cannot run": REFUTED.** It is accepted and runs; Slurm auto-bumps CPUs 8→14. Real
completed job `1671018_0` proves it. The recipe's flaw is silent over-allocation, not failure.
The prior agents over-claimed here and the reassessment's Table row 9 should be amended.

**Correction 2 — 48 h is not the ceiling.** `--qos=long` gives **7 days** on the same partition with
the same 4-GPU cap, via the `PartitionTimeLimit` flag. Live proof: job `1664922` at RunTime 4-00:55
/ TimeLimit 5-00:00. The prior agents missed this, and it is the difference between stage (b) needing
checkpoint-resume and not.

**Addition — the newly-discovered binding constraint is `MaxSubmitPU=32`,** not walltime. It counts
array elements, `%K` throttling does not exempt them, and it is the cap that will actually break a
24-or-32-run campaign script. Nobody has flagged this.

**Consequence for the program, in one line:** the relaxation makes the reassessment's replacement
study (4 arms × 350M × 2B × 8 seeds, **834 L40S-h, ~12 calendar days**, no checkpointing needed) and
the rank screen (**1,324 L40S-h, ~18 days**) genuinely runnable at full 350M scale — which the 6 h
belief had ruled out entirely, forcing a retreat to 20-50M. But the headline 3 × 750M × 50B stage is
**4,139 L40S-h ≈ 57 realistic calendar days**, with no submit form on this cluster capable of
completing a single run, and it stays out of reach. **The 13-day 8×A100 program is a 6-to-13-week
FarmShare program: a 3.4× gap at best, 7.3× realistic.**

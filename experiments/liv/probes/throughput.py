"""Measure real training throughput for the LIV pilot arms on one L40S.

WHY MEASURE INSTEAD OF ESTIMATING
---------------------------------
Both open questions -- how long the pilot takes, and whether AWS is worth it -- reduce to
tokens/sec. A spec-sheet estimate can be off by 2-3x for a 350M model: low arithmetic
intensity, kernel-launch overhead, and (for ``N-narrow``) an unusual head dimension all cost
real time that FLOPs counting does not see.

Reports **achieved TFLOPS next to tok/s**, because that is the number that can be checked
against the card's peak. A result above peak means the measurement is broken, not that the
model is fast -- the discipline that a cache-resident microbenchmark on this same project
skipped, at a cost of a day and three wrong documents.

The step here uses a naive fp32 cross-entropy, which materializes a
``micro_bs x seq x vocab`` fp32 tensor (6.6 GB at micro_bs=8). OLMo-core's real trainer uses
a fused/chunked loss instead, so every number below is a **lower bound** on the throughput of
the actual run.
"""

import json
import math
import time
import traceback

import torch
import torch.nn.functional as F

from olmo_core.nn.transformer.liv_arms import VOCAB_SIZE, build_arm

SEQ = 4096
ARMS = ["L0", "F-r128", "G-grouped", "N-narrow"]
MICRO_BS = [1, 2, 4]
WARMUP, TIMED = 4, 12

# NVIDIA L40S datasheet, dense (non-sparse) BF16 tensor-core throughput.
L40S_BF16_DENSE_TFLOPS = 362.0
LN_VOCAB = math.log(VOCAB_SIZE)


def bench(name: str, micro_bs: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = build_arm(name)
    model = cfg.build(init_device="cuda")
    # build() does NOT initialize -- it constructs modules and (on a real device) leaves
    # parameter memory uninitialized. Without this call the first loss is ~900 instead of
    # ln(vocab)=10.83, and the timing measures garbage-magnitude activations.
    model.init_weights(max_seq_len=SEQ, device=torch.device("cuda"))
    model.train()
    flops_per_token = model.num_flops_per_token(SEQ)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
    )

    gen = torch.Generator(device="cuda").manual_seed(0)
    losses = []
    t0 = None

    for i in range(WARMUP + TIMED):
        if i == WARMUP:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        toks = torch.randint(
            0, VOCAB_SIZE, (micro_bs, SEQ + 1), device="cuda", generator=gen, dtype=torch.long
        )
        x, y = toks[:, :-1], toks[:, 1:]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tok = micro_bs * SEQ * TIMED
    tok_s = tok / dt
    tflops = tok_s * flops_per_token / 1e12

    return {
        "arm": name,
        "micro_bs": micro_bs,
        "params": n_params,
        "flops_per_token": flops_per_token,
        "step_ms": 1000 * dt / TIMED,
        "tokens_per_sec": tok_s,
        "achieved_tflops": tflops,
        "mfu_pct": 100 * tflops / L40S_BF16_DENSE_TFLOPS,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "loss_step0": losses[0],
        "loss_last": losses[-1],
    }


def main() -> None:
    print("device :", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("memory : {:.1f} GiB   sm_{}{}".format(props.total_memory / 2**30, props.major, props.minor))
    print("torch  :", torch.__version__)
    print("seq    :", SEQ, "  vocab:", VOCAB_SIZE, "  ln(vocab) = {:.3f}".format(LN_VOCAB))
    print()

    results = []
    for name in ARMS:
        for mbs in MICRO_BS:
            try:
                r = bench(name, mbs)
            except torch.cuda.OutOfMemoryError:
                print("{:<10} mbs={:<2} OOM".format(name, mbs))
                results.append({"arm": name, "micro_bs": mbs, "error": "OOM"})
                torch.cuda.empty_cache()
                continue
            except Exception as exc:  # unusual head dims can break attention kernels
                print("{:<10} mbs={:<2} FAILED: {}".format(name, mbs, exc))
                traceback.print_exc()
                results.append({"arm": name, "micro_bs": mbs, "error": repr(exc)})
                torch.cuda.empty_cache()
                continue

            # A throughput number is meaningless if the model is not actually learning the
            # task; an untrained model on uniform-random tokens must start at ~ln(vocab).
            ok_loss = abs(r["loss_step0"] - LN_VOCAB) < 0.6
            flag = "" if ok_loss else "  <-- BAD loss_step0"
            results.append(r)
            print(
                "{:<10} mbs={:<2} {:>7.0f} tok/s  {:>6.1f} TFLOPS ({:>4.1f}% MFU)  "
                "{:>7.1f} ms/step  {:>5.1f} GiB  loss0={:.2f}{}".format(
                    r["arm"], r["micro_bs"], r["tokens_per_sec"], r["achieved_tflops"],
                    r["mfu_pct"], r["step_ms"], r["peak_mem_gib"], r["loss_step0"], flag,
                )
            )
            assert r["achieved_tflops"] < L40S_BF16_DENSE_TFLOPS, (
                "achieved TFLOPS exceeds the card's dense peak -- the measurement is wrong"
            )
            # A throughput number from an uninitialized model is not a throughput number.
            assert ok_loss, (
                "loss_step0={:.2f} but ln(vocab)={:.2f}: the model is not properly "
                "initialized, so this timing is meaningless".format(r["loss_step0"], LN_VOCAB)
            )

    ok = [r for r in results if "error" not in r]
    if ok:
        best = max(ok, key=lambda r: r["tokens_per_sec"])
        print()
        print("BEST: {} at micro_bs={} -> {:,.0f} tok/s ({:.1f}% MFU, {:.1f} GiB)".format(
            best["arm"], best["micro_bs"], best["tokens_per_sec"], best["mfu_pct"],
            best["peak_mem_gib"]))
        l0 = [r for r in ok if r["arm"] == "L0"]
        if l0:
            slowest = min(l0, key=lambda r: r["tokens_per_sec"])
            fastest = max(l0, key=lambda r: r["tokens_per_sec"])
            print("L0 (bounds the schedule): {:,.0f} tok/s at best micro_bs={}".format(
                fastest["tokens_per_sec"], fastest["micro_bs"]))
            del slowest

    with open("/scratch/users/ericrcwu/liv/throughput_results.json", "w") as fh:
        json.dump({"device": torch.cuda.get_device_name(0), "seq": SEQ,
                   "l40s_peak_tflops": L40S_BF16_DENSE_TFLOPS, "results": results}, fh, indent=2)
    print("\nwrote throughput_results.json")


if __name__ == "__main__":
    main()

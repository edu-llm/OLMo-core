#!/usr/bin/env python3
import wandb

api = wandb.Api()
run = api.run("eduLLM/curriculum-moe/adfc5d7f824bcd5530ed3ad404e2912e")
print("state:", run.state, "_step:", run.summary.get("_step"))
hist = run.history(keys=["eval/bpb/arc_challenge_val_rc_5shot_bpb"], samples=1000)
print("eval history rows:", len(hist))
if len(hist):
    print(hist.tail(3).to_string())

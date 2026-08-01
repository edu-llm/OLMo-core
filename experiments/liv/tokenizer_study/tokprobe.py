import json, sys, unicodedata
p="/scratch/users/ericrcwu/liv/tok/lfm2_tokenizer.json"
d=json.load(open(p))
print("TOP KEYS:", list(d.keys()))
print("version:", d.get("version"))
print("truncation:", d.get("truncation"))
print("padding:", d.get("padding"))
print("\n=== normalizer ===");  print(json.dumps(d.get("normalizer"), ensure_ascii=False)[:2000])
print("\n=== pre_tokenizer ==="); print(json.dumps(d.get("pre_tokenizer"), ensure_ascii=False)[:3000])
print("\n=== post_processor ==="); print(json.dumps(d.get("post_processor"), ensure_ascii=False)[:2000])
print("\n=== decoder ==="); print(json.dumps(d.get("decoder"), ensure_ascii=False)[:2000])
m=d["model"]
print("\n=== model keys ===", list(m.keys()))
for k in m:
    if k not in ("vocab","merges"):
        print("  model.%s = %r" % (k, m[k]))
print("model.type =", m.get("type"))
print("len(model.vocab) =", len(m["vocab"]))
print("len(model.merges) =", len(m.get("merges",[])))
at=d.get("added_tokens",[])
print("len(added_tokens) =", len(at))
print("SUM =", len(m["vocab"])+len(at))
print("\n=== added_tokens (all) ===")
for t in at:
    print(" ", t["id"], repr(t["content"]), "special=",t.get("special"))
print("\n=== first 25 merges ===")
for x in m.get("merges",[])[:25]: print("  ", x)
print("\n=== vocab ids 0..40 ===")
inv={v:k for k,v in m["vocab"].items()}
for i in range(0,41):
    print("  ", i, repr(inv.get(i)))
print("\n=== max id in model.vocab:", max(m["vocab"].values()), " min:", min(m["vocab"].values()))

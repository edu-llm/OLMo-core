import numpy as np
from collections import Counter
# AR-Hits slice size on OUR corpus, both tokenizers, using the Zoology definition.
tr=np.load("/scratch/users/ericrcwu/kda/lm/data/train.npy",mmap_mode="r")
va=np.load("/scratch/users/ericrcwu/kda/lm/data/val.npy",mmap_mode="r")
print("building bigram freq table over a 200M-token slice of train (GPT-2 ids)...",flush=True)
n=200_000_000
a=np.asarray(tr[:n],dtype=np.int64)
big=a[:-1]*50257+a[1:]
cnt=Counter()
CH=20_000_000
for i in range(0,len(big),CH):
    cnt.update(big[i:i+CH].tolist())
print("distinct bigrams:",len(cnt),flush=True)
# eval pass over val, seq len 4096
v=np.asarray(va[:4096*256],dtype=np.int64).reshape(256,4096)
hits=0;tot=0
for row in v:
    b=row[:-1]*50257+row[1:]
    seen=set()
    for j,bg in enumerate(b.tolist()):
        tot+=1
        if bg in seen and cnt.get(bg,0)<=1250: hits+=1
        seen.add(bg)
print(f"AR-Hits slice on OUR corpus (GPT-2, 4096-ctx, 256 seqs): {hits:,}/{tot:,} = {100*hits/tot:.2f}%   (Zoology on Pile: 6.4%)")

"""Confirm OLMo-core reads the header-stripped corpus exactly as np.load does."""
import numpy as np
from olmo_core.data import NumpyFSLDatasetConfig, TokenizerConfig

SEQ = 4096
tok = TokenizerConfig.gpt2()
cfg = NumpyFSLDatasetConfig(
    paths=["/scratch/users/ericrcwu/liv/data/train_raw.bin"],
    sequence_length=SEQ,
    tokenizer=tok,
    work_dir="/scratch/users/ericrcwu/liv/work",
)
ds = cfg.build()
ds.prepare()
print("instances      :", format(len(ds), ","))
print("tokens covered :", format(len(ds) * SEQ, ","))
print("expected       : 1,200,000,000 -> floor to seq multiple:",
      format((1_200_000_000 // SEQ) * SEQ, ","))
assert len(ds) == 1_200_000_000 // SEQ, len(ds)

truth = np.load("/scratch/users/ericrcwu/kda/lm/data/train.npy", mmap_mode="r")
for idx in (0, 1, 12345, len(ds) - 1):
    got = ds[idx]["input_ids"].numpy()
    exp = np.asarray(truth[idx * SEQ : (idx + 1) * SEQ], dtype=got.dtype)
    assert np.array_equal(got, exp), f"instance {idx} differs from np.load ground truth"
    print(f"instance {idx:>7}: matches np.load exactly (first ids {got[:5].tolist()})")

mx = max(int(ds[i]["input_ids"].max()) for i in (0, 1, 500, 12345, len(ds) - 1))
print("max id in sample:", mx, "< vocab 50304 ->", mx < 50304)
print("\nDATA PATH OK")

"""End-to-end proof that the ``split`` label-mask is honored by OLMo-core's training loss.

Self-contained: it synthesizes a tiny token stream with a few known fact spans, writes shards in
the exact format ``colmlm/prepare_data.py`` produces (headerless uint16 tokens + bool mask), then
exercises the *real* training path:

    NumpyFSLDataset(label_mask_paths=...)   ->  batch carries "label_mask"
    olmo_core.data.utils.get_labels(batch)  ->  fact targets become -100  (this is exactly what
                                                TransformerTrainModule.train_batch calls)
    model(input_ids, labels=..., ignore_index=-100, loss_reduction="sum",
          loss_div_factor=num_tokens_for_loss)  ->  cross_entropy_loss(ignore_index=-100)

It proves, with numbers, that fact tokens are removed from both the numerator and denominator of
the loss (so they produce zero gradient and don't skew the mean). PR #44 adds an equivalent check
as a pytest (``src/test/colmlm_train_test.py``); this is the standalone runnable version.

    python verify_loss_masking.py
"""

import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

# Windows only: the data-loader dep `bettermap` imports multiprocessing.ForkProcess, which does not
# exist on win32. On Linux (the platform) the real module loads and this guard is a no-op.
if sys.platform == "win32" and "bettermap" not in sys.modules:
    _stub = types.ModuleType("bettermap")

    def _m(fn, *its, **_):
        return list(map(fn, *its))

    _stub.__getattr__ = lambda n: (_m if not (n.startswith("__") and n.endswith("__")) else (_ for _ in ()).throw(AttributeError(n)))  # type: ignore
    sys.modules["bettermap"] = _stub

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from olmo_core.data import NumpyDatasetDType, NumpyFSLDatasetConfig, TokenizerConfig  # noqa: E402
from olmo_core.data.utils import get_labels  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

VOCAB = 49152
SEQ = 128
N_SEQ = 6
TOTAL = SEQ * N_SEQ  # exact multiple -> no dropped remainder
IGNORE = -100

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  {'OK ' if cond else 'ERR'} {name}{(' -- ' + detail) if detail else ''}")


# --- 1. Synthesize a token stream + a fact mask (False == fact token, dropped from loss) ---
tokens = np.random.randint(1, 2000, size=TOTAL, dtype=np.uint16)
tokens[::SEQ] = 0  # an EOS at the start of each packed sequence, like real doc packing
mask = np.ones(TOTAL, dtype=np.bool_)
FACT_SPANS = [(40, 55), (130, 140), (300, 312), (500, 520), (600, 605)]
for a, b in FACT_SPANS:
    mask[a:b] = False
n_fact = int((~mask).sum())
print(f"synthetic stream: {TOTAL} tokens, {n_fact} fact tokens ({n_fact / TOTAL:.2%}), {len(FACT_SPANS)} spans")

work = Path(tempfile.mkdtemp())
(work / "tokens").mkdir()
(work / "masks").mkdir()
tokens_path = work / "tokens" / "train-00000.bin"
mask_path = work / "masks" / "train-00000.mask.bin"
tokens.astype("<u2").tofile(tokens_path)      # headerless uint16 LE  (matches prepare_data.py)
mask.tofile(mask_path)                          # headerless bool      (matches prepare_data.py)

tok = TokenizerConfig(vocab_size=VOCAB, eos_token_id=0, bos_token_id=0, pad_token_id=0,
                      identifier="HuggingFaceTB/SmolLM2-135M")


def make_ds(with_mask: bool):
    return NumpyFSLDatasetConfig(
        paths=[str(tokens_path.as_posix())],
        sequence_length=SEQ,
        tokenizer=tok,
        dtype=NumpyDatasetDType.uint16,
        work_dir=str((work / ("split" if with_mask else "base")).as_posix()),
        label_mask_paths=[str(mask_path.as_posix())] if with_mask else None,
    ).build()


base_ds, split_ds = make_ds(False), make_ds(True)
check("dataset built, N sequences", len(split_ds) == N_SEQ, f"{len(split_ds)} seqs")

# --- 2. Collate a batch exactly as the loader would (stack the packed sequences) ---
input_ids = torch.stack([split_ds[i]["input_ids"] for i in range(N_SEQ)]).long()
label_mask = torch.stack([split_ds[i]["label_mask"] for i in range(N_SEQ)])
check("split item exposes label_mask", "label_mask" in split_ds[0] and "label_mask" not in base_ds[0])

# The mask the loader produced must equal the mask we wrote (flattened, in order).
loader_mask_flat = label_mask.reshape(-1).numpy()
check("data loader carries mask through unchanged",
      bool(np.array_equal(loader_mask_flat, mask)),
      f"{int((loader_mask_flat != mask).sum())} mismatched of {TOTAL}")

# --- 3. get_labels (the exact fn TransformerTrainModule.train_batch calls) ---
labels_split = get_labels({"input_ids": input_ids, "label_mask": label_mask}, label_ignore_index=IGNORE)
labels_base = get_labels({"input_ids": input_ids}, label_ignore_index=IGNORE)

split_keep = labels_split != IGNORE
base_keep = labels_base != IGNORE
fact_targets = base_keep & ~split_keep  # targets base trains on but split drops

# Base only ignores the last (shifted) position of each sequence; split additionally drops facts.
n_base_ignored = int((~base_keep).sum())
n_split_ignored = int((~split_keep).sum())
check("base ignores only last-token-per-seq", n_base_ignored == N_SEQ, f"{n_base_ignored} (expected {N_SEQ})")
check("split drops the fact targets too", int(fact_targets.sum()) > 0 and n_split_ignored > n_base_ignored,
      f"split ignores {n_split_ignored}, base {n_base_ignored}, extra={int(fact_targets.sum())}")

# Every extra dropped target must be a (shifted) fact token -- nothing else is touched.
shifted_fact = F.pad(~label_mask[:, 1:], (0, 1), value=False)  # fact at t+1 -> target at t
check("dropped targets == shifted fact tokens exactly",
      bool((fact_targets & ~shifted_fact).sum() == 0) and bool((shifted_fact & ~fact_targets).sum() == 0),
      f"{int((fact_targets ^ (shifted_fact & base_keep)).sum())} misaligned")

masked_pct_split = float((~split_keep).float().mean())
masked_pct_base = float((~base_keep).float().mean())
print(f"  -> train/masked labels (%): split={masked_pct_split:.4%}  base={masked_pct_base:.4%}")

# --- 4. Real model forward + real loss, exactly as train_batch does it ---
print("building smollm2_135M (cpu)...")
model = TransformerConfig.smollm2_135M(vocab_size=VOCAB).build(init_device="cpu")
model.eval()

num_tokens_split = split_keep.sum()
num_tokens_base = base_keep.sum()

with torch.no_grad():
    # This is the literal call from TransformerTrainModule.model_forward -> model(...).
    logits, loss_split, ce_split, _z = model(
        input_ids,
        labels=labels_split,
        ignore_index=IGNORE,
        loss_reduction="sum",
        loss_div_factor=num_tokens_split,
        return_logits=True,
    )

V = logits.shape[-1]
flat_logits = logits.reshape(-1, V).float()

# Manual reference: mean CE over ONLY the unmasked targets.
manual_split = F.cross_entropy(flat_logits[split_keep.reshape(-1)],
                               labels_split.reshape(-1)[split_keep.reshape(-1)], reduction="mean")
check("model split-loss == mean CE over unmasked targets only",
      torch.allclose(loss_split.float(), manual_split, atol=1e-3, rtol=1e-3),
      f"model={float(loss_split):.6f} manual={float(manual_split):.6f}")

# Decomposition: split loss == base loss with exactly the fact-target contributions removed.
ce_none = F.cross_entropy(flat_logits, labels_base.reshape(-1), reduction="none",
                          ignore_index=IGNORE).reshape(N_SEQ, SEQ)
sum_all = ce_none.sum()
sum_fact = ce_none[fact_targets].sum()
loss_base_manual = sum_all / num_tokens_base
loss_split_manual = (sum_all - sum_fact) / num_tokens_split
check("split loss = (base_sum - fact_sum) / unmasked_count  (facts removed from num & denom)",
      torch.allclose(loss_split.float(), loss_split_manual.float(), atol=1e-3, rtol=1e-3),
      f"split={float(loss_split):.6f} decomposed={float(loss_split_manual):.6f}")

# Facts genuinely carry loss under base -> dropping them is a real change, not a no-op.
check("fact targets have real loss mass under base (so masking actually matters)",
      float(sum_fact) > 0.0,
      f"base_loss={float(loss_base_manual):.4f} split_loss={float(loss_split):.4f} "
      f"fact_mass={float(sum_fact):.2f} over {int(fact_targets.sum())} targets")

print("\nRESULT:", "LOSS MASKING VERIFIED END-TO-END" if all(results) else "PROBLEMS ABOVE")
sys.exit(0 if all(results) else 1)

"""Build a diffusion run's config locally, with corpus resolution stubbed out.

    python .edullm/verify_diffusion_config.py [--kernels]

WHAT THIS BUYS, AND WHAT IT CANNOT. `edullm check` validates the submission form; it does not
build the model. `train_diffusion.py --dry-run` would build it, but only on a machine with the
corpus reader (`edullm_data`) installed, which is the image and not a laptop. So the whole path
from the flags to a built `ExperimentConfig` -- the factory wrap, the block_overrides, the train
module swap, the dotted-override merge -- was reachable nowhere before a GPU had been billed.
This stubs `resolve_corpus` and exercises the rest.

It cannot reach anything that needs `flash-linear-attention`, which is not installable here. That
is what `run-kernels-diffusion-1xa10g.yaml` is for.

Run it before every submit. Two submissions have already died in the first seconds on things a
config build would have shown.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import train_diffusion as td  # noqa: E402
import train_gdn2 as gdn2  # noqa: E402
import train_on_corpus as toc  # noqa: E402

from olmo_core.data import NumpyDatasetDType, TokenizerConfig  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

KERNELS = "--kernels" in sys.argv

CORPUS = toc.Corpus(
    dataset_id="pretrain/regmix-10b",
    version="v1",
    paths=[f"s3://edullm-data/pretrain/regmix-10b/v1/shard{i:04d}.npy" for i in range(41)],
    dtype=NumpyDatasetDType.uint32,
    tokenizer=TokenizerConfig.dolma2(),
    rows=None,
)

toc.resolve_corpus = gdn2.resolve_corpus = td.gdn2.resolve_corpus = lambda **kw: CORPUS

if KERNELS:
    argv = [
        "verify",
        "--save-folder",
        "/tmp/verify",
        "--model-factory",
        "olmo2_190M",
        "--init-method",
        "fan_in",
        "--n-heads",
        "12",
        "--head-dim",
        "64",
        "--expand-v",
        "1.0",
        "--optimizer",
        "muon_h",
        "--learning-rate",
        "0.01",
        "--adamw-learning-rate",
        "8.2e-4",
        "--param-dtype",
        "bfloat16",
        "--sequence-length",
        "512",
        "--steps",
        "20",
        "--warmup-steps",
        "2",
        "--global-batch-size",
        "8192",
        "--rank-microbatch-size",
        "2048",
    ]
    expect_layers, expect_recurrent, expect_attention = 12, 9, 3
else:
    argv = [
        "verify",
        "--save-folder",
        "/tmp/verify",
        "--model-factory",
        "olmo2_370M_moe",
        "--init-method",
        "fan_in",
        "--n-heads",
        "16",
        "--head-dim",
        "64",
        "--expand-v",
        "1.0",
        "--moe-top-k",
        "8",
        "--moe-num-experts",
        "32",
        "--moe-hidden-size",
        "512",
        "--optimizer",
        "muon_h",
        "--learning-rate",
        "0.01",
        "--adamw-learning-rate",
        "8.2e-4",
        "--param-dtype",
        "bfloat16",
        "--sequence-length",
        "4096",
        "--steps",
        "19073",
        "--warmup-steps",
        "1908",
        "--global-batch-size",
        "524288",
        "--rank-microbatch-size",
        "8192",
        "--mixture",
        "on",
        "--mixture-total",
        "10000000000",
    ]
    expect_layers, expect_recurrent, expect_attention = 16, 12, 4

argv += [
    "--dataset-id",
    "pretrain/regmix-10b",
    "--dataset-version",
    "v1",
    "--dataset-tokenizer",
    "tokenizer/dolma2-bpe",
    "--work-dir",
    "/tmp/cache",
]

opts = td.build_parser().parse_args(argv)
config = td.build_config(opts, ["trainer.callbacks.checkpointer.max_checkpoints=null"])

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


module = config.train_module
check(
    "diffusion train module", type(module).__name__.startswith("Diffusion"), type(module).__name__
)
check("MuonH optimizer", type(module.optim).__name__ == "MuonHConfig", type(module.optim).__name__)
check(
    "fan_in init (MuonH reads its radius off W_0)",
    str(config.model.init_method) == "fan_in",
    str(config.model.init_method),
)
check(
    "bfloat16",
    str(module.dp_config.param_dtype).endswith("bfloat16"),
    str(module.dp_config.param_dtype),
)
check(
    "context parallelism off (a reversed scan needs the sequence whole)", module.cp_config is None
)

overrides = config.model.block_overrides or {}
recurrent = sum(
    1 for b in overrides.values() if type(b.sequence_mixer).__name__.startswith("GatedDelta")
)
attention = len(overrides) - recurrent
forward = sum(
    1
    for b in overrides.values()
    if type(b.sequence_mixer).__name__.startswith("GatedDelta")
    and not b.sequence_mixer.reverse_scan
)
check("layer count", config.model.n_layers == expect_layers, str(config.model.n_layers))
check(
    "hybrid layout",
    (recurrent, attention) == (expect_recurrent, expect_attention),
    f"{recurrent} GDN-2 + {attention} attention ({forward} forward / {recurrent - forward} reverse)",
)
check(
    "every attention block is non-causal",
    all(
        b.sequence_mixer.causal is False
        for b in overrides.values()
        if not type(b.sequence_mixer).__name__.startswith("GatedDelta")
    ),
)
check(
    "every GDN-2 block is noise-conditioned",
    all(
        b.sequence_mixer.noise_conditioned
        for b in overrides.values()
        if type(b.sequence_mixer).__name__.startswith("GatedDelta")
    ),
)

tokenizer = CORPUS.tokenizer
mask_id = module.diffusion.mask_token_id
check(
    "MASK id sits in the free vocab padding",
    tokenizer.vocab_size <= mask_id < tokenizer.padded_vocab_size(),
    f"{mask_id} in [{tokenizer.vocab_size}, {tokenizer.padded_vocab_size()})",
)

if not KERNELS:
    baseline = TransformerConfig.olmo2_370M(vocab_size=tokenizer.padded_vocab_size())
    gdn2.swap_sequence_mixer(baseline, gdn2.mixer_config(opts))
    target = baseline.num_non_embedding_params
    active = config.model.num_active_non_embedding_params
    check(
        "active params matched to the AR baseline",
        0.95 < active / target < 1.05,
        f"{active/1e6:.1f}M vs {target/1e6:.1f}M, ratio {active/target:.3f}",
    )

config.as_config_dict()
check("config serialises", True)

print()
if failures:
    print(f"DO NOT SUBMIT. Failed: {', '.join(failures)}")
    sys.exit(1)
print("Config builds and every structural check passes.")

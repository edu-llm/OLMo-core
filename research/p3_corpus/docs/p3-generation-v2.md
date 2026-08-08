# P3 six-family generation transaction

`scripts/build_p3_generation.py` is the only production entry point for the
repaired corpus. It requires exactly these siblings, in this order:
`metamath`, `mizar`, `thproofs`, `prf2`, `enigma`, `isabelle`.

The builder work root and corpus transaction root must be fresh external
locations under `/tmp`. Family builders receive only a newly created directory
under the work root. Final bytes enter the immutable generation only through the
descriptor-safe transaction writer.

Each production builder command is a closed adapter for the repository's actual
builder script and argparse surface. Its manifest declares every recursive
staging entry, including directories, file format, schema, and row source root.
After the process exits, the orchestrator rejects undeclared or missing entries,
wrong kinds, mixed schemas or roots, symlinks, temporary files, and special
files. Every declared JSON object must contain its exact internal
`schema_version` and every adapter-declared contract field; a valid filename or
manifest declaration cannot authorize schema-less JSON. Metamath split output
specifically requires `metamath-heldout-v2` with family, mode, exactly requested
held facts, and the local-assumption contract. Builder logs are kept outside the
declared output directory. Production
configuration cannot encode callbacks or test seams, and command arguments
containing test/bypass/skip/legacy-production switches are forbidden.

## Production status

The command refuses before creating the corpus root until its technical inputs
are supplied and accepted:

- finalized production MML input, source-manifest, quality-filter, and
  schema-generation roots for all four pooled siblings;
- an accepted current-source builder/root for the direct `mizar` sibling;
- six production source manifests using `p3-family-source-manifest/v2`.

No placeholder technical root is accepted. License status is recorded honestly
as metadata and any required human/legal approval is a separate pre-upload
decision. The optional generated-output `metamath_valid` metric is disabled and
does not gate source-backed token-data generation.

## One-line commands

Production-input dry validation performs no build and creates no output. It
fully reads and validates all six manifests, policy and tokenizer schemas,
content roots, the loaded tokenizer seal, the current Mizar index bytes, all
closed builder commands and readable command inputs, and the disjoint `/tmp`
work/transaction roots. It emits a canonical
`p3-generation-preflight/v2` JSON report whose
`preflight_root_sha256` binds every validated path, root, command, blocker, and
error. It returns nonzero while any blocker remains and never reports
placeholders as ready.

```bash
python scripts/build_p3_generation.py --dry-run --corpus-root /tmp/p3-generation-root --work-root /tmp/p3-generation-work --generation-id <ID> --tokenizer-seal <TOKENIZER-SEAL.json> --tokenizer-path <VENDORED-TOKENIZER> --policies <POLICIES.json> --source-manifest metamath=<METAMATH.json> --source-manifest mizar=<MIZAR.json> --source-manifest thproofs=<THPROOFS.json> --source-manifest prf2=<PRF2.json> --source-manifest enigma=<ENIGMA.json> --source-manifest isabelle=<ISABELLE.json> --mizar-semantic-index <MIZAR.sqlite>
```

Real build, only after every blocker is resolved:

```bash
python scripts/build_p3_generation.py --corpus-root /tmp/p3-generation-root --work-root /tmp/p3-generation-work --generation-id <ID> --tokenizer-seal <TOKENIZER-SEAL.json> --tokenizer-path <VENDORED-TOKENIZER> --policies <POLICIES.json> --source-manifest metamath=<METAMATH.json> --source-manifest mizar=<MIZAR.json> --source-manifest thproofs=<THPROOFS.json> --source-manifest prf2=<PRF2.json> --source-manifest enigma=<ENIGMA.json> --source-manifest isabelle=<ISABELLE.json> --mizar-semantic-index <MIZAR.sqlite>
```

Production verification of the generation selected by `CURRENT`:

```bash
PYTHONPATH=scripts python scripts/verify_corpus.py --corpus /tmp/p3-generation-root --mizar-semantic-index <MIZAR.sqlite>
```

Production verification first resolves and validates the immutable transaction,
then independently reconstructs `sidecars/schemas.json` and
`sidecars/precheck.json` from actual files and rows. It reruns the production MML
contract loader, current Mizar index checks, Metamath held
name/statement/local/target/own-proof isolation, and Isabelle direct/local/
theorem/goal/state/target/trajectory isolation with typed-drop accounting.
It also reapplies the declared internal schema and required-field contract to
all six source links, all three heldout objects, and tokenizer, policy, schema,
occurrence, and precheck sidecars, including the nested family heldout
contracts. Stored sidecar booleans are never treated as proof.

Read-only diagnosis of the stale v2 layout (always non-production and never
returns production-clean):

```bash
PYTHONPATH=scripts python scripts/verify_corpus.py --legacy-audit --corpus corpus --mizar-html2 <LEGACY-HTML2>
```

Deterministic six-family synthetic transaction:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_build_p3_generation.py::test_synthetic_six_family_generation_is_reproducible_and_deep_clean -q
```

The direct unit API exposes this exact fixed fault inventory:

- `raw_builder:<family>` and `builder_complete:<family>` for all six families;
- `split_builder:start:metamath`, `split_builder:complete:metamath`,
  `split_builder:start:isabelle`, and `split_builder:complete:isabelle`;
- `split_builder:start:mml` and `split_builder:complete:mml` around the pooled
  four-family contract builder;
- `normalization:<family>` and `partition:<family>` for Metamath and Isabelle;
- `family_split:complete:<family>` for all six final family partitions;
- `mml_partition`, `precheck`, and `final_copy:<declared-path>` for every
  transaction output.

This seam is not accepted by production manifests or exposed by the CLI. Every
injected failure preserves the previous `CURRENT` and quarantines both
transaction and external builder staging. Split-start, split-completion, and
family-split-completion failures occur before any declared final copy.

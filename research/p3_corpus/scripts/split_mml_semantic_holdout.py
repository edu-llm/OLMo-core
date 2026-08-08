"""Plan one pooled semantic holdout across Mizar and MPTP representations.

This module is deliberately separate from the current corpus splitter. It reads
the already quality-filtered raw ``mizar``, ``thproofs``, ``prf2``, and
``enigma`` JSONL shards, validates their provenance and schemas, excludes
direct-Mizar-covered ``thproofs`` trajectories, removes exact ATP duplicates in
``prf2``-then-``enigma`` order, applies the sealed Qwen ``text + EOS`` limit,
and draws one deterministic set of 1,000 semantic classes.

Planning is side-effect free. :func:`write_partition_atomically` can publish a
complete four-shard plan to a new directory while preserving every native JSONL
row byte-for-byte; holdout metadata is written only to manifests and sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from build_atp_shard import (
    ProofStep,
    is_refutation_formula,
    render_target,
    source_dependencies,
)

SHARD_ORDER = ("mizar", "thproofs", "prf2", "enigma")
REPRESENTATION_BY_SHARD = {
    "mizar": "mizar",
    "thproofs": "mizar",
    "prf2": "atp",
    "enigma": "atp",
}
SEED = 20260801
REQUESTED_CLASSES = 1_000
MAX_TEXT_PLUS_EOS_TOKENS = 16_384

MANIFEST_SCHEMA_VERSION = "mml-semantic-holdout-manifest-v7"
POLICY_VERSION = "mml-semantic-holdout-policy-v8"
MAPPING_VERSION = "mml-semantic-name-map-v2"
STATEMENT_HASH_VERSION = "mml-semantic-statement-v4"
ATP_DEDUPLICATION_POLICY = "mml-atp-exact-structured-v5"
MIZAR_THPROOFS_DEDUPLICATION_POLICY = "mml-direct-mizar-before-thproofs-v1"
ENIGMA_VARIANT_GROUPING_POLICY = "mml-enigma-theorem-variant-grouping-v1"
COMPATIBILITY_SCHEMA_VERSION = "mml-semantic-holdout-compat-v7"
SOURCE_IDENTITY_POLICY_VERSION = "mml-source-identity-policy-v3"
LOADER_CONTRACT_SCHEMA_VERSION = "mml-semantic-holdout-loader-v7"
CONTRACT_TUPLE_SCHEMA_VERSION = "mml-semantic-holdout-contract-tuple-v5"
CANONICALIZATION_CONTRACT_VERSION = "mml-semantic-canonicalization-v4"
EDULLM_DATA_COMMIT = "38bf831a6c3f445e394784018441fd59288b876c"

APPROVED_TOKENIZER_ID = "Qwen/Qwen2.5-0.5B"
APPROVED_TOKENIZER_JSON_SHA256 = (
    "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8"
)
APPROVED_TOKENIZER_CONFIG_SHA256 = (
    "ddb9f850ca6559a928bb25d511f72e3c6eff81395334a4e0eeec670448333d09"
)
APPROVED_TOKENIZER_BEHAVIOR_SHA256 = (
    "aa90434a251a434bbc938ddb3be6683a73fa94150377b5ccd2cbd7880358661a"
)
APPROVED_TOKENIZERS_VERSION = "0.22.2"
APPROVED_EOS_TOKEN_ID = 151643

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MIZAR_THEOREM_RE = re.compile(r"^([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*):([1-9]\d*)$")
MIZAR_DEFINITION_RE = re.compile(r"^([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*):def_([1-9]\d*)$")
ATP_MML_RE = re.compile(r"^([td])([1-9]\d*)_([a-z][a-z0-9]*(?:_[a-z0-9]+)*)$")
ATP_WRAPPERS = frozenset({"prf2", "enigma"})
ALTERNATE_SUFFIX_RE = re.compile(r"#\d+$")
BOOKKEEPING_RE = re.compile(r"^(?:dt_|(?:cc|fc|rc)\d*_|redefinition_|fraenkel_|rq|spc)")

COMMON_REQUIRED_FIELDS = (
    "id",
    "theorem",
    "facts",
    "cited",
    "goal",
    "target",
    "text",
    "mask_start",
    "mask_end",
)
ATP_REQUIRED_FIELDS = ("local_inputs", "goal_name", "proof_steps")

MAPPING_POLICY = {
    "version": MAPPING_VERSION,
    "mizar": {
        "theorem": "ARTICLE:N",
        "definition": "ARTICLE:def_N",
    },
    "atp": {
        "theorem": "tN_article",
        "definition": "dN_article",
        "article_grammar": "canonical lowercase MPTP atom",
        "record_identity_normalization": {
            "wrappers": ["prf2", "enigma"],
            "terminal_alternate_suffix": "#N",
        },
        "excluded_from_mapping": [
            "l",
            "scheme instances containing __",
            "e",
            "c",
            "de",
            "ie",
            "rd",
            "generated registrations",
            "malformed names",
        ],
    },
    "unmapped": "representation-scoped singleton",
}

ATP_PARENT_OCCURRENCE_POLICY = {
    "source_occurrences": "preserve ordered recursively flattened leaf parents exactly",
    "repeated_occurrences": (
        "accepted only when rule, parent_sources, and parents exactly replay source_dependencies"
    ),
    "dependency_graph": "unique parent names; occurrence multiplicity has no DAG effect",
    "stored_bytes_rewritten": False,
}

ENIGMA_VARIANT_GROUPING_DESCRIPTION = {
    "policy": ENIGMA_VARIANT_GROUPING_POLICY,
    "identity": "wrapper-stripped decoded ATP theorem with terminal #N removed",
    "suffixes": ["base", "#1", "#2", "#3", "#4", "all terminal #N"],
    "route_together": True,
    "eval_propagation": (
        "if any eligible proof variant exposes a selected class, every eligible "
        "variant of that theorem routes to eval"
    ),
    "quality_and_dedup_drops_preserved": True,
    "native_rows_rewritten": False,
}

HOLDOUT_POLICY = {
    "version": POLICY_VERSION,
    "seed": SEED,
    "requested_classes": REQUESTED_CLASSES,
    "tail_row_citation_counts": [1, 2],
    "selection_algorithm": "sorted-tail-python-mt19937-sample-v1",
    "shards": list(SHARD_ORDER),
    "representations": REPRESENTATION_BY_SHARD,
    "fact_scope": {
        "bookkeeping_prefixes": BOOKKEEPING_RE.pattern,
        "bookkeeping_location": "local_inputs",
        "other_stable_named_premises": "eligible",
    },
    "atp_parent_occurrences": ATP_PARENT_OCCURRENCE_POLICY,
    "enigma_variant_grouping": ENIGMA_VARIANT_GROUPING_DESCRIPTION,
    "ordering": [
        "approved source identity and global ID validation",
        "schema and ATP-v2 deep integrity validation",
        "direct-Mizar-before-thproofs exact trajectory deduplication",
        "text plus EOS eligibility",
        "ordered exact ATP deduplication",
        "statement consistency",
        "pooled row-citation counting",
        "class draw",
        "exposure partition",
        "ENIGMA theorem-variant eval propagation",
    ],
}

CANONICALIZATION_POLICY = {
    "version": CANONICALIZATION_CONTRACT_VERSION,
    "mizar": {
        "algorithm": "layout-collapsed-outside-quotes",
        "version": "v1",
    },
    "atp": {
        "algorithm": (
            "ordered-delimiter-validation-then-layout-removal-and-complete-outer-strip"
        ),
        "version": "v4",
        "repeat": True,
        "quote_and_escape_aware": True,
        "ordered_delimiter_stack": ["()", "[]", "{}"],
        "reject_escape_outside_quotes": True,
        "preserve_malformed_canonical_text": True,
    },
    "cross_representation_text_comparison": False,
}

_PRODUCTION_SOURCE_IDENTITY_EVIDENCE = {
    "mizar": {
        "status": "finalized",
        "source_manifest_schema": "p3-family-source-manifest/v2",
        "release": "Mizar 8.1.15 / MML 5.94.1493",
        "approved_tree_sha256": [
            "3d1af5b3e840aca5631541b42510b35c1b15dfa988af70ce463f58c899e88714",
            "1f725c9943aeee2c21c6fe63484bc00336bdc442ec454ccfc810032d7de12781",
            "fce0eda226231de221ff2e7b3c9fa0699ec259d3e647e53eb9589b181dbf7877",
        ],
        "input_rows": 55_353,
        "input_sha256": "54206c1fe89d09dec7ec36c927612439b687814ba95e1086e4b09db036ad486f",
        "source_manifest_root_sha256": (
            "fa21f98fa551ae3e54b17e4e31aacebfde48c0be3ea8b99f5ff85f4ee08fb762"
        ),
        "quality_filter_root_sha256": (
            "9fb4b02b9c632d0dfdf5f8730798b25a981a7da46bc0c06f770ee3df14ee7d7d"
        ),
        "schema_generation_root_sha256": (
            "ea8deb4c5912f9b10f5da674fcd86c9f8c8b5cf521522ad70b6168a5bf554242"
        ),
        "semantic_index_sha256": (
            "8deb18e7ab38d7d42d852828667a7f0b8000f3141b5bad7cbd940b617f9bd835"
        ),
        "acceptance_roots": {
            "recovery_identity_set_sha256": (
                "048f47cf87e6eaeccf87f3aafb202236373dea000719ae221c5ee33896dad8cd"
            ),
            "recovery_identity_source_order_sha256": (
                "6d113a43ff0b0af8aae13325908d2507b9b63aadcc01d50c37d73e29549396fa"
            ),
            "recovery_source_binding_sha256": (
                "790c86db30604c5836be70e28df527bb6c1a41b30620cfaf327122db047be65c"
            ),
            "recovery_text_hash_sequence_sha256": (
                "e116af514ee5cc7fc3415d01a68ec42037206849f3b62e6bdddbfefe4637659f"
            ),
            "recovery_token_sequence_sha256": (
                "3391cb491f1e7e8ec23b7725d27ceb95b4d5d51bd5856a8e46372507102d5ca4"
            ),
        },
        "source_snapshots": [
            {
                "reference": (
                    "http://grid01.ciirc.cvut.cz/~mptp/8.1.15_5.94.1493/"
                    "html_abstr.8.1.15_5.94.1493.tar.gz"
                ),
                "sha256": (
                    "e988481577e4e5cc25a5c96c4e86a7de612447088b20781a2680b0e6fc974eee"
                ),
            },
            {
                "reference": (
                    "https://mizar.uwb.edu.pl/~softadm/pub/system/i386-linux/"
                    "mizar-8.1.15_5.94.1493-i386-linux.tar"
                ),
                "sha256": (
                    "cfc32c3e05d5d93c595934e26d4d3b4e399f95a75da7df08359eb9ee73ae6e2e"
                ),
            },
            {
                "reference": (
                    "http://grid01.ciirc.cvut.cz/~mptp/8.1.15_5.94.1493/thproofs.tar.gz"
                ),
                "sha256": (
                    "665b17fea382d23168998a4bd1fd91736baf59c1fa3927f8c656d9886fdc3433"
                ),
            },
        ],
    },
    "thproofs": {
        "status": "finalized",
        "source_manifest_schema": "mizar-current-sources-v1",
        "release": "Mizar 8.1.15 / MML 5.94.1493",
        "approved_tree_sha256": [
            "3d1af5b3e840aca5631541b42510b35c1b15dfa988af70ce463f58c899e88714",
            "1f725c9943aeee2c21c6fe63484bc00336bdc442ec454ccfc810032d7de12781",
            "fce0eda226231de221ff2e7b3c9fa0699ec259d3e647e53eb9589b181dbf7877",
        ],
        "input_rows": 50_743,
        "input_sha256": "8bdb66128d6385f03d1b70d064bbc089f28c2f4f529f405d761e728080657854",
        "source_manifest_root_sha256": (
            "17d2aa537ef9cf05e9acb26573143cec2de9ec8d7ae272b4324a09becd25ff63"
        ),
        "quality_filter_root_sha256": (
            "895dc288d07504d0f6106a8641aeb4a268a86caa84dcdf1a55078939da60eb74"
        ),
        "schema_generation_root_sha256": (
            "52429daebe81fa70cde000ec1fffe1fb8af0ee9a2421af3d089ce0b14e75a62b"
        ),
        "semantic_index_sha256": (
            "8deb18e7ab38d7d42d852828667a7f0b8000f3141b5bad7cbd940b617f9bd835"
        ),
        "source_snapshots": [
            {
                "reference": (
                    "http://grid01.ciirc.cvut.cz/~mptp/8.1.15_5.94.1493/"
                    "html_abstr.8.1.15_5.94.1493.tar.gz"
                ),
                "sha256": (
                    "e988481577e4e5cc25a5c96c4e86a7de612447088b20781a2680b0e6fc974eee"
                ),
            },
            {
                "reference": (
                    "https://mizar.uwb.edu.pl/~softadm/pub/system/i386-linux/"
                    "mizar-8.1.15_5.94.1493-i386-linux.tar"
                ),
                "sha256": (
                    "cfc32c3e05d5d93c595934e26d4d3b4e399f95a75da7df08359eb9ee73ae6e2e"
                ),
            },
            {
                "reference": (
                    "http://grid01.ciirc.cvut.cz/~mptp/8.1.15_5.94.1493/thproofs.tar.gz"
                ),
                "sha256": (
                    "665b17fea382d23168998a4bd1fd91736baf59c1fa3927f8c656d9886fdc3433"
                ),
            },
        ],
    },
    "prf2": {
        "status": "finalized",
        "source_manifest_schema": "atp-build-source-v2",
        "approved_tree_sha256": [
            "263f830438ebf7210f2d5d960867d5eb22553f55d3cd353f0b547a9b114941a5",
        ],
        "input_rows": 24_797,
        "input_sha256": "fdde1aececef6de1c88cac8e17945c7a55491bd7bb527784c26099f67f63ab3d",
        "source_manifest_root_sha256": (
            "029f23d490c96521bf90e29c12581459b831d991fef597c58f81ce39abcd4fdf"
        ),
        "quality_filter_root_sha256": (
            "1d4ba91cfc3152cade7a3743d34801f654bdce1a682872d6ba35c403cdab6ab6"
        ),
        "schema_generation_root_sha256": (
            "bd0caede34f0dea92401bc9a306ae1ef82f81a26ed3bedeecd6e6a4a653a1a60"
        ),
        "source_snapshots": [
            {
                "reference": "/tmp/p3-source-audit/prf2.tar.gz",
                "sha256": (
                    "e57c9e799132ee0c05bb8e956620f8c9e49346b78eb8d1eaa0bf3d210d89a7d5"
                ),
            },
        ],
    },
    "enigma": {
        "status": "finalized",
        "source_manifest_schema": "atp-build-source-v2",
        "approved_tree_sha256": [
            "73b13a8461e2f98ae00513f1f764654998db2fab7de698a7809dc87f5ddbe3fe",
            "1f216d4b1f6c59242845b69695c534554df905e468d4f3e4756430736dd6d73d",
            "534158d350bb14ef898d85554e676722bb00b7adc9acf2e2c061b7de10f965d5",
            "314bc45d17237e9ce55bf47613e60ecebccc9b7649f19667ccf3d0ff69961e7c",
        ],
        "input_rows": 29_166,
        "input_sha256": "7fddf832938404f6e76f33fae06a6e8731b923cde65d9c32795288ac4250a3f7",
        "source_manifest_root_sha256": (
            "c33d5b87696e56276f8c2eb81fd1acb274ab766308e60a96e6ee999d6d76fd2e"
        ),
        "quality_filter_root_sha256": (
            "cdd95f8ab0314ec2b11b44273fd412c7445e2ab61e73d43a33e84c158b7c4ce8"
        ),
        "schema_generation_root_sha256": (
            "bd0caede34f0dea92401bc9a306ae1ef82f81a26ed3bedeecd6e6a4a653a1a60"
        ),
        "acceptance_roots": {
            "acceptance_audit_sha256": (
                "b3189c4a8589beb7d066c27246adf3257adadccb0729fecd2732a4c10165c5c8"
            ),
            "alternative_proof_policy_root_sha256": (
                "646a531f28aeb7c5a8ef78de97e8000e866816508a6e8413b51474d6c5cf4669"
            ),
            "selected_occurrence_root_sha256": (
                "587b81fa01ba84b4245745f10e83c89d9fa46771558c907148a00864afdee7e0"
            ),
        },
        "source_snapshots": [
            {
                "reference": "/tmp/p3-source-audit/mzr01.tar.gz",
                "sha256": (
                    "06ddbf863ff7f6c3421d158f106a33b5932dc2b5dfee8d5ba0fb7bab027afcd0"
                ),
            },
            {
                "reference": "/tmp/p3-source-audit/mzr02.tar.gz",
                "sha256": (
                    "5c02524146b90028712cda6fffae362ac5ea2d40f74b993ea4968edf1b4f06ba"
                ),
            },
            {
                "reference": "/tmp/p3-source-audit/mzr03.tar.gz",
                "sha256": (
                    "4280e5ed25f5ec3052a53449c0959e8adae913ee8fb310afa4a8702ce9907dcd"
                ),
            },
            {
                "reference": "/tmp/p3-source-audit/mzr08.tar.gz",
                "sha256": (
                    "8135d36c26e020f16d36a1dc7d1828d597fd92694bb538ef489fca568a42ff7d"
                ),
            },
        ],
    },
}

ATP_DEDUPLICATION_DESCRIPTION = {
    "policy": ATP_DEDUPLICATION_POLICY,
    "priority": ["prf2", "enigma"],
    "scope": "eligible ATP rows",
    "signature": {
        "ignores": [
            "record id",
            "prf2/enigma theorem wrapper",
            "terminal #N",
            "map order",
        ],
        "includes": [
            "normalized theorem identity",
            "goal",
            "goal_name",
            "facts",
            "local_inputs",
            "structured proof steps",
        ],
        "legacy": "conservative target bytes after line-ending normalization",
    },
}

MIZAR_THPROOFS_DEDUPLICATION_DESCRIPTION = {
    "policy": MIZAR_THPROOFS_DEDUPLICATION_POLICY,
    "priority": ["mizar", "thproofs"],
    "scope": "eligible Mizar-representation theorem trajectories",
    "identity": {
        "theorem": "exact canonical Mizar theorem identity",
        "goal": "UTF-8 text with CRLF normalized to LF",
        "target": "UTF-8 text with CRLF normalized to LF",
    },
    "routing": {
        "identical_thproof_trajectory": "direct_mizar_trajectory_duplicate",
        "same_theorem_different_goal_or_target": "fail closed",
        "thproof_only_trajectory": "preserve",
    },
}

PRODUCTION_DEDUPLICATION_DESCRIPTIONS = {
    "mizar": {
        "policy": MIZAR_THPROOFS_DEDUPLICATION_POLICY,
        "role": "priority source",
        "description": MIZAR_THPROOFS_DEDUPLICATION_DESCRIPTION,
    },
    "thproofs": {
        "policy": MIZAR_THPROOFS_DEDUPLICATION_POLICY,
        "role": "exclude direct-Mizar-covered trajectories",
        "description": MIZAR_THPROOFS_DEDUPLICATION_DESCRIPTION,
    },
    "prf2": {
        "policy": ATP_DEDUPLICATION_POLICY,
        "role": "priority source",
        "description": ATP_DEDUPLICATION_DESCRIPTION,
    },
    "enigma": {
        "policy": ATP_DEDUPLICATION_POLICY,
        "role": "exclude exact prf2/earlier-ENIGMA duplicates",
        "description": ATP_DEDUPLICATION_DESCRIPTION,
    },
}


class HoldoutError(RuntimeError):
    """A correctness or provenance gate that makes a plan unusable."""


class ShardSource(Protocol):
    """A replayable byte stream for one native JSONL shard."""

    name: str
    logical_path: str
    expected_input_sha256: str
    source_snapshots: Sequence[SourceSnapshot]
    source_manifest_root_sha256: str
    quality_filter_root_sha256: str
    schema_generation_root_sha256: str

    def iter_lines(self) -> Iterator[bytes]:
        """Yield native JSONL lines without rewriting them."""


@dataclass(frozen=True)
class SourceSnapshot:
    """One immutable source archive or repository reference."""

    reference: str
    sha256: str


@dataclass(frozen=True)
class ApprovedShardSource:
    """Exact generated-input and upstream roots approved for one shard."""

    input_sha256: str
    source_snapshots: tuple[SourceSnapshot, ...]
    source_manifest_root_sha256: str
    quality_filter_root_sha256: str
    schema_generation_root_sha256: str
    acceptance_roots: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SourceIdentityPolicy:
    """Closed source identity table supplied by tests or production code."""

    policy_id: str
    shards: Mapping[str, ApprovedShardSource]
    test_only: bool = False
    deduplication_roots: Mapping[str, str] | None = None


@dataclass(frozen=True)
class MemoryShardSource:
    """Replayable in-memory shard used by synthetic tests and small callers."""

    name: str
    logical_path: str
    lines: tuple[bytes, ...]
    expected_input_sha256: str
    source_snapshots: tuple[SourceSnapshot, ...]
    source_manifest_root_sha256: str
    quality_filter_root_sha256: str
    schema_generation_root_sha256: str

    def iter_lines(self) -> Iterator[bytes]:
        """Yield the original synthetic bytes."""

        yield from self.lines


@dataclass(frozen=True)
class PathShardSource:
    """Replayable file-backed shard used by the command-line entry point."""

    name: str
    logical_path: str
    path: Path
    expected_input_sha256: str
    source_snapshots: tuple[SourceSnapshot, ...]
    source_manifest_root_sha256: str
    quality_filter_root_sha256: str
    schema_generation_root_sha256: str

    def iter_lines(self) -> Iterator[bytes]:
        """Stream the native file in binary mode."""

        with self.path.open("rb") as source_file:
            yield from source_file


@dataclass(frozen=True)
class TokenizerSeam:
    """Sealed tokenizer metadata plus an injectable exact-length counter."""

    seal: Mapping[str, Any]
    count_text_plus_eos: Callable[[str], int]


@dataclass(frozen=True)
class PolicyPins:
    """Expected hashes for every policy that can change class identity."""

    policy_sha256: str
    mapping_sha256: str
    atp_deduplication_sha256: str


@dataclass(frozen=True)
class SemanticIdentity:
    """The semantic class assigned to one native fact or theorem name."""

    class_id: str
    kind: str
    representation: str
    raw_name: str
    native_name: str
    mapped: bool


@dataclass(frozen=True, order=True)
class ExposureHit:
    """One concrete path from a row to a selected semantic class."""

    class_id: str
    field: str
    native_name: str = ""

    def as_dict(self) -> dict[str, str]:
        """Return a stable sidecar representation."""

        result = {"class_id": self.class_id, "field": self.field}
        if self.native_name:
            result["native_name"] = self.native_name
        return result


@dataclass(frozen=True)
class HeldoutExposure:
    """All selected-class exposure paths visible in one native row."""

    direct_native_citation: tuple[ExposureHit, ...] = ()
    mapped_cross_representation: tuple[ExposureHit, ...] = ()
    own_theorem: tuple[ExposureHit, ...] = ()
    statement_alias: tuple[ExposureHit, ...] = ()
    visible_target: tuple[ExposureHit, ...] = ()
    theorem_variant_group: tuple[ExposureHit, ...] = ()

    @property
    def should_eval(self) -> bool:
        """Whether the row must be excluded from training."""

        return any(
            (
                self.direct_native_citation,
                self.mapped_cross_representation,
                self.own_theorem,
                self.statement_alias,
                self.visible_target,
                self.theorem_variant_group,
            )
        )

    def paths_dict(self) -> dict[str, list[dict[str, str]]]:
        """Return deterministic path metadata for an eval sidecar."""

        return {
            "direct_native_citation": [
                hit.as_dict() for hit in self.direct_native_citation
            ],
            "mapped_cross_representation": [
                hit.as_dict() for hit in self.mapped_cross_representation
            ],
            "own_theorem": [hit.as_dict() for hit in self.own_theorem],
            "statement_alias": [hit.as_dict() for hit in self.statement_alias],
            "visible_target": [hit.as_dict() for hit in self.visible_target],
            "theorem_variant_group": [
                hit.as_dict() for hit in self.theorem_variant_group
            ],
        }


@dataclass(frozen=True)
class ExposureIndex:
    """Selected class members and statement aliases, separated by syntax."""

    selected_class_ids: frozenset[str]
    members_by_class: Mapping[str, Mapping[str, frozenset[str]]]
    statement_classes_by_representation: Mapping[str, Mapping[str, frozenset[str]]]
    canonical_statement_classes_by_representation: Mapping[
        str, Mapping[str, frozenset[str]]
    ]


@dataclass(frozen=True)
class RowPlan:
    """The immutable route for one source line."""

    line_number: int
    row_id: str
    disposition: str
    text_plus_eos_tokens: int
    native_row_sha256: str
    exposure_sidecar_sha256: str | None
    drop_reason: str | None = None
    exposure: HeldoutExposure | None = None
    theorem_variant_group: str | None = None
    variant_group_eval_propagated: bool = False


@dataclass(frozen=True)
class PartitionPlan:
    """A complete four-shard plan with no output side effects."""

    manifest: dict[str, Any]
    compatibility_projections: dict[str, dict[str, Any]]
    rows: dict[str, tuple[RowPlan, ...]]
    sealed_manifest_root_sha256: str
    sealed_route_plan_root_sha256: str


@dataclass(frozen=True)
class PublishedArtifact:
    """One fully validated file in a published holdout partition."""

    path: Path
    sha256: str
    bytes: int
    rows: int
    schema: str


@dataclass(frozen=True)
class FamilyPaths:
    """Canonical train, eval, and dropped paths for one native family."""

    train: Path
    eval: Path
    dropped: Path


@dataclass(frozen=True)
class ValidatedHoldoutContract:
    """Typed, fully verified contract for shared verifier and evaluator code."""

    root: Path
    production: bool
    test_only: bool
    authoritative_root: str
    manifest: dict[str, Any]
    projections: dict[str, dict[str, Any]]
    artifacts: Mapping[str, PublishedArtifact]
    family_paths: Mapping[str, FamilyPaths]
    exposure_index: Mapping[tuple[str, str], Mapping[str, Any]]
    tokenizer_root_sha256: str
    source_root_sha256: str
    quality_filter_roots_by_shard: Mapping[str, str]
    schema_generation_roots_by_shard: Mapping[str, str]
    deduplication_roots_by_shard: Mapping[str, str]
    acceptance_roots_by_shard: Mapping[str, Mapping[str, str]]

    @property
    def selected_class_ids(self) -> frozenset[str]:
        """Return every selected semantic class ID."""

        return frozenset(
            record["class_id"] for record in self.manifest["class_records"]
        )

    def projection(self, family: str) -> dict[str, Any]:
        """Return one validated legacy-family projection."""

        if family not in {"mizar", "atp"}:
            raise KeyError(family)
        return self.projections[family]


HoldoutContract = ValidatedHoldoutContract


@dataclass(frozen=True)
class _PreparedRow:
    shard: str
    line_number: int
    record: dict[str, Any]
    text_plus_eos_tokens: int
    native_row_sha256: str
    pre_drop_reason: str | None
    atp_signature: str | None


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def production_deduplication_root(shard: str) -> str:
    """Return the code-derived deduplication-policy root for one native shard."""

    try:
        description = PRODUCTION_DEDUPLICATION_DESCRIPTIONS[shard]
    except KeyError as error:
        raise HoldoutError(f"unknown production MML shard {shard!r}") from error
    return _json_sha256(description)


def _require_production_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise HoldoutError(f"{label} is missing or is not a SHA-256 root")
    if len(set(value)) == 1:
        raise HoldoutError(f"{label} is an unfinished placeholder root")
    return value


def _normalized_production_snapshots(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise HoldoutError("production source snapshots are missing")
    snapshots = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"reference", "sha256"}:
            raise HoldoutError("production source snapshot is malformed")
        reference = item["reference"]
        if not isinstance(reference, str) or not reference.strip():
            raise HoldoutError("production source snapshot reference is missing")
        snapshots.append(
            {
                "reference": reference.strip(),
                "sha256": _require_production_digest(
                    item["sha256"],
                    label=f"production source snapshot {reference!r}",
                ),
            }
        )
    return sorted(snapshots, key=lambda item: (item["reference"], item["sha256"]))


def _normalized_acceptance_roots(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise HoldoutError("production acceptance roots are malformed")
    roots: dict[str, str] = {}
    for name, root in value.items():
        if not isinstance(name, str) or not name.strip():
            raise HoldoutError("production acceptance-root name is malformed")
        roots[name.strip()] = _require_production_digest(
            root,
            label=f"production acceptance root {name!r}",
        )
    return dict(sorted(roots.items()))


def _production_source_record_payload(
    shard: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "shard": shard,
        "source_manifest_schema": record["source_manifest_schema"],
        "input_rows": record["input_rows"],
        "input_sha256": record["input_sha256"],
        "source_snapshots": _normalized_production_snapshots(
            record["source_snapshots"]
        ),
        "source_manifest_root_sha256": record["source_manifest_root_sha256"],
        "quality_filter_root_sha256": record["quality_filter_root_sha256"],
        "schema_generation_root_sha256": record["schema_generation_root_sha256"],
        "deduplication_root_sha256": record["deduplication_root_sha256"],
        "acceptance_roots": _normalized_acceptance_roots(
            record.get("acceptance_roots")
        ),
    }


def _seal_production_source_record(
    shard: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = dict(record)
    sealed["status"] = "finalized"
    sealed["source_snapshots"] = _normalized_production_snapshots(
        sealed["source_snapshots"]
    )
    sealed["acceptance_roots"] = _normalized_acceptance_roots(
        sealed.get("acceptance_roots")
    )
    sealed["deduplication_root_sha256"] = production_deduplication_root(shard)
    sealed["finalization_root_sha256"] = _json_sha256(
        _production_source_record_payload(shard, sealed)
    )
    return sealed


def _family_source_manifest_root(manifest: Mapping[str, Any]) -> str:
    def without_recursive_roots(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: without_recursive_roots(item)
                for key, item in value.items()
                if key
                not in {
                    "manifest_root_sha256",
                    "source_manifest_root_sha256",
                }
            }
        if isinstance(value, list):
            return [without_recursive_roots(item) for item in value]
        return value

    payload = _canonical_json_bytes(without_recursive_roots(manifest))
    return hashlib.sha256(payload + b"\n").hexdigest()


def finalize_production_source_record(
    shard: str,
    *,
    raw_path: str | os.PathLike[str],
    source_manifest: Mapping[str, Any],
    acceptance_roots: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one deterministic, table-ready record from fresh local build bytes.

    This is the only finalization seam for pending production roots. It reads but
    never rewrites the raw JSONL, verifies the generic family source manifest,
    derives the shard's code-owned deduplication root, and seals all roots into a
    finalization digest.
    """

    if shard not in SHARD_ORDER:
        raise HoldoutError(f"unknown production MML shard {shard!r}")
    path = Path(raw_path)
    if path.name != f"{shard}.jsonl" or path.parent.name != "raw":
        raise HoldoutError(f"{shard}: production raw path must be raw/{shard}.jsonl")
    if not isinstance(source_manifest, Mapping):
        raise HoldoutError(f"{shard}: production source manifest is not an object")
    if source_manifest.get("schema_version") != "p3-family-source-manifest/v2":
        raise HoldoutError(f"{shard}: production source manifest schema is not v2")
    if source_manifest.get("family") != shard:
        raise HoldoutError(f"{shard}: production source manifest family mismatch")
    if source_manifest.get("test_only") is not False:
        raise HoldoutError(f"{shard}: production source manifest is test-only")
    acceptance = source_manifest.get("source_verifier_acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("accepted") is not True:
        raise HoldoutError(f"{shard}: source verifier has not accepted the candidate")
    actual_manifest_root = _family_source_manifest_root(source_manifest)
    declared_manifest_root = _require_production_digest(
        source_manifest.get("manifest_root_sha256"),
        label=f"{shard}: source-manifest root",
    )
    if declared_manifest_root != actual_manifest_root:
        raise HoldoutError(f"{shard}: source-manifest root drift")
    metadata = source_manifest.get("row_source_metadata")
    if not isinstance(metadata, Mapping):
        raise HoldoutError(f"{shard}: row source metadata is missing")
    if metadata.get("source_manifest_root_sha256") != declared_manifest_root:
        raise HoldoutError(f"{shard}: row source-manifest root drift")
    if not isinstance(metadata.get("source_roots"), Mapping) or not metadata[
        "source_roots"
    ]:
        raise HoldoutError(f"{shard}: native source roots are missing")
    quality_root = _require_production_digest(
        metadata.get("quality_filter_root_sha256"),
        label=f"{shard}: quality-filter root",
    )
    schema_root = _require_production_digest(
        metadata.get("schema_generation_root_sha256"),
        label=f"{shard}: schema-generation root",
    )
    snapshots = _normalized_production_snapshots(
        source_manifest.get("source_snapshots")
    )
    digest = hashlib.sha256()
    rows = 0
    try:
        with path.open("rb") as raw_file:
            for line in raw_file:
                if not line.strip():
                    raise HoldoutError(f"{shard}: raw input contains a blank row")
                digest.update(line)
                rows += 1
    except OSError as error:
        raise HoldoutError(f"{shard}: cannot read fresh raw input {path}") from error
    if rows < 1:
        raise HoldoutError(f"{shard}: fresh raw input is empty")
    record = {
        "source_manifest_schema": source_manifest["schema_version"],
        "input_rows": rows,
        "input_sha256": digest.hexdigest(),
        "source_snapshots": snapshots,
        "source_manifest_root_sha256": declared_manifest_root,
        "quality_filter_root_sha256": quality_root,
        "schema_generation_root_sha256": schema_root,
        "acceptance_roots": _normalized_acceptance_roots(acceptance_roots),
    }
    return _seal_production_source_record(shard, record)


def _validated_production_source_record(
    shard: str,
    record: Mapping[str, Any],
) -> ApprovedShardSource:
    if record.get("status") != "finalized":
        raise HoldoutError(f"{shard}: production source identity is not finalized")
    required = {
        "source_manifest_schema",
        "input_rows",
        "input_sha256",
        "source_snapshots",
        "source_manifest_root_sha256",
        "quality_filter_root_sha256",
        "schema_generation_root_sha256",
        "deduplication_root_sha256",
        "acceptance_roots",
        "finalization_root_sha256",
    }
    if not required <= set(record):
        raise HoldoutError(f"{shard}: finalized production source roots are incomplete")
    rows = record["input_rows"]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise HoldoutError(f"{shard}: finalized production input row count is invalid")
    source_manifest_schema = record["source_manifest_schema"]
    if not isinstance(source_manifest_schema, str) or not source_manifest_schema:
        raise HoldoutError(f"{shard}: finalized source-manifest schema is missing")
    for key, label in (
        ("input_sha256", "raw input SHA-256"),
        ("source_manifest_root_sha256", "source-manifest root"),
        ("quality_filter_root_sha256", "quality-filter root"),
        ("schema_generation_root_sha256", "schema-generation root"),
        ("deduplication_root_sha256", "deduplication root"),
        ("finalization_root_sha256", "finalization root"),
    ):
        _require_production_digest(record[key], label=f"{shard}: production {label}")
    expected_deduplication = production_deduplication_root(shard)
    if record["deduplication_root_sha256"] != expected_deduplication:
        raise HoldoutError(f"{shard}: production deduplication root drift")
    expected_finalization = _json_sha256(
        _production_source_record_payload(shard, record)
    )
    if record["finalization_root_sha256"] != expected_finalization:
        raise HoldoutError(f"{shard}: production source finalization root drift")
    snapshots = tuple(
        SourceSnapshot(reference=item["reference"], sha256=item["sha256"])
        for item in _normalized_production_snapshots(record["source_snapshots"])
    )
    acceptance_roots = tuple(
        _normalized_acceptance_roots(record["acceptance_roots"]).items()
    )
    return ApprovedShardSource(
        input_sha256=str(record["input_sha256"]),
        source_snapshots=snapshots,
        source_manifest_root_sha256=str(record["source_manifest_root_sha256"]),
        quality_filter_root_sha256=str(record["quality_filter_root_sha256"]),
        schema_generation_root_sha256=str(record["schema_generation_root_sha256"]),
        acceptance_roots=acceptance_roots,
    )


def verify_finalized_production_source_record(
    shard: str,
    record: Mapping[str, Any],
) -> bool:
    """Return whether a production table record is complete and drift-free."""

    try:
        _validated_production_source_record(shard, record)
    except (HoldoutError, KeyError, TypeError):
        return False
    return True


PRODUCTION_SOURCE_IDENTITY_TABLE = {
    shard: (
        _seal_production_source_record(shard, record)
        if record.get("status") == "finalized"
        else {
            **record,
            "deduplication_root_sha256": production_deduplication_root(shard),
        }
    )
    for shard, record in _PRODUCTION_SOURCE_IDENTITY_EVIDENCE.items()
}


def _json_line_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value) + b"\n").hexdigest()


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def digest_lines(lines: Iterable[bytes]) -> str:
    """Hash exact line bytes in their supplied order."""

    digest = hashlib.sha256()
    for line in lines:
        if not isinstance(line, bytes):
            raise TypeError("JSONL sources must yield bytes")
        digest.update(line)
    return digest.hexdigest()


def approved_tokenizer_seal() -> dict[str, Any]:
    """Return the sealed Qwen tokenizer metadata used by the 16k corpus."""

    return {
        "identity": APPROVED_TOKENIZER_ID,
        "tokenizer_json_sha256": APPROVED_TOKENIZER_JSON_SHA256,
        "tokenizer_config_sha256": APPROVED_TOKENIZER_CONFIG_SHA256,
        "behavior_digest": APPROVED_TOKENIZER_BEHAVIOR_SHA256,
        "tokenizers_version": APPROVED_TOKENIZERS_VERSION,
        "eos_token_id": APPROVED_EOS_TOKEN_ID,
        "max_text_plus_eos_tokens": MAX_TEXT_PLUS_EOS_TOKENS,
    }


def current_policy_pins() -> PolicyPins:
    """Return the exact policy hashes required by this implementation."""

    return PolicyPins(
        policy_sha256=_json_sha256(HOLDOUT_POLICY),
        mapping_sha256=_json_sha256(MAPPING_POLICY),
        atp_deduplication_sha256=_json_sha256(ATP_DEDUPLICATION_DESCRIPTION),
    )


def _versioned_contract_component(
    version: str,
    description: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "version": version,
        "sha256": _json_sha256(description),
    }


def canonical_contract_tuple(
    source_identity_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact implementation contract accepted by the loader."""

    pins = current_policy_pins()
    source_policy_sha256 = source_identity_policy.get("policy_sha256")
    source_policy_id = source_identity_policy.get("policy_id")
    source_test_only = source_identity_policy.get("test_only")
    if (
        not isinstance(source_policy_sha256, str)
        or not SHA256_RE.fullmatch(source_policy_sha256)
        or not isinstance(source_policy_id, str)
        or not isinstance(source_test_only, bool)
    ):
        raise HoldoutError("source identity policy cannot form an exact contract tuple")
    statement_hash_description = {
        "version": STATEMENT_HASH_VERSION,
        "canonicalization_sha256": _json_sha256(CANONICALIZATION_POLICY),
        "schemes": {
            "mizar": "mizar-layout-v1",
            "atp": "tptp-ordered-delimiters-complete-outer-formula-v4",
        },
        "digest": "sha256",
        "payload_separator": "NUL",
    }
    components: dict[str, dict[str, Any]] = {
        "manifest": _versioned_contract_component(
            MANIFEST_SCHEMA_VERSION,
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "root": "canonical-json-sha256-excluding-manifest-root-v1",
                "inventory": "exact-canonical-path-order-v1",
            },
        ),
        "loader": _versioned_contract_component(
            LOADER_CONTRACT_SCHEMA_VERSION,
            {
                "schema_version": LOADER_CONTRACT_SCHEMA_VERSION,
                "verification": "exact-artifacts-routes-sidecars-projections-v2",
                "atp_parent_occurrence_policy_sha256": _json_sha256(
                    ATP_PARENT_OCCURRENCE_POLICY
                ),
                "enigma_variant_grouping_policy_sha256": _json_sha256(
                    ENIGMA_VARIANT_GROUPING_DESCRIPTION
                ),
            },
        ),
        "compatibility": _versioned_contract_component(
            COMPATIBILITY_SCHEMA_VERSION,
            {
                "schema_version": COMPATIBILITY_SCHEMA_VERSION,
                "derivation": "authoritative-root-linked-v2",
                "atp_parent_occurrence_policy_sha256": _json_sha256(
                    ATP_PARENT_OCCURRENCE_POLICY
                ),
                "enigma_variant_grouping_policy_sha256": _json_sha256(
                    ENIGMA_VARIANT_GROUPING_DESCRIPTION
                ),
            },
        ),
        "policy": {
            "version": POLICY_VERSION,
            "sha256": pins.policy_sha256,
        },
        "mapping": {
            "version": MAPPING_VERSION,
            "sha256": pins.mapping_sha256,
        },
        "statement_hash": _versioned_contract_component(
            STATEMENT_HASH_VERSION,
            statement_hash_description,
        ),
        "atp_deduplication": {
            "version": ATP_DEDUPLICATION_POLICY,
            "sha256": pins.atp_deduplication_sha256,
        },
        "mizar_thproofs_deduplication": _versioned_contract_component(
            MIZAR_THPROOFS_DEDUPLICATION_POLICY,
            MIZAR_THPROOFS_DEDUPLICATION_DESCRIPTION,
        ),
        "enigma_variant_grouping": _versioned_contract_component(
            ENIGMA_VARIANT_GROUPING_POLICY,
            ENIGMA_VARIANT_GROUPING_DESCRIPTION,
        ),
        "source_policy": {
            "version": SOURCE_IDENTITY_POLICY_VERSION,
            "sha256": source_policy_sha256,
            "policy_id": source_policy_id,
            "test_only": source_test_only,
        },
        "canonicalization": _versioned_contract_component(
            CANONICALIZATION_CONTRACT_VERSION,
            CANONICALIZATION_POLICY,
        ),
    }
    return {
        "schema_version": CONTRACT_TUPLE_SCHEMA_VERSION,
        "edullm_data_commit": EDULLM_DATA_COMMIT,
        "components": components,
    }


def _snapshot_records(
    snapshots: Sequence[SourceSnapshot],
) -> list[dict[str, str]]:
    return sorted(
        (
            {
                "reference": str(snapshot.reference).strip(),
                "sha256": str(snapshot.sha256).lower(),
            }
            for snapshot in snapshots
        ),
        key=lambda item: (item["reference"], item["sha256"]),
    )


def _source_policy_deduplication_roots(
    policy: SourceIdentityPolicy,
) -> dict[str, str]:
    roots = (
        {
            shard: production_deduplication_root(shard)
            for shard in policy.shards
        }
        if policy.deduplication_roots is None
        else dict(policy.deduplication_roots)
    )
    if set(roots) != set(policy.shards):
        raise HoldoutError(
            "source identity policy deduplication roots must match its shards"
        )
    for shard, root in roots.items():
        _require_production_digest(
            root,
            label=f"{shard}: source-policy deduplication root",
        )
        if root != production_deduplication_root(shard):
            raise HoldoutError(f"{shard}: source-policy deduplication root drift")
    return {shard: roots[shard] for shard in sorted(roots)}


def _source_policy_payload(policy: SourceIdentityPolicy) -> dict[str, Any]:
    return {
        "version": SOURCE_IDENTITY_POLICY_VERSION,
        "policy_id": policy.policy_id,
        "test_only": policy.test_only,
        "deduplication_roots": _source_policy_deduplication_roots(policy),
        "shards": {
            shard: {
                "input_sha256": approved.input_sha256,
                "source_snapshots": _snapshot_records(approved.source_snapshots),
                "source_manifest_root_sha256": (approved.source_manifest_root_sha256),
                "quality_filter_root_sha256": approved.quality_filter_root_sha256,
                "schema_generation_root_sha256": (
                    approved.schema_generation_root_sha256
                ),
                "acceptance_roots": dict(approved.acceptance_roots),
            }
            for shard, approved in sorted(policy.shards.items())
        },
    }


def production_source_policy() -> SourceIdentityPolicy:
    """Return the production source policy only after every root is finalized."""

    if set(PRODUCTION_SOURCE_IDENTITY_TABLE) != set(SHARD_ORDER):
        missing = sorted(set(SHARD_ORDER) - set(PRODUCTION_SOURCE_IDENTITY_TABLE))
        extra = sorted(set(PRODUCTION_SOURCE_IDENTITY_TABLE) - set(SHARD_ORDER))
        raise HoldoutError(
            "production source identity table is not exact; "
            f"missing={missing}, extra={extra}"
        )
    unfinished = []
    approved = {}
    for shard in SHARD_ORDER:
        record = PRODUCTION_SOURCE_IDENTITY_TABLE[shard]
        if record.get("status") != "finalized":
            unfinished.append(shard)
            continue
        approved[shard] = _validated_production_source_record(shard, record)
    if unfinished:
        raise HoldoutError(
            "production source identity table is not finalized for "
            + ", ".join(sorted(unfinished))
        )
    roots = {
        shard: str(
            PRODUCTION_SOURCE_IDENTITY_TABLE[shard][
                "deduplication_root_sha256"
            ]
        )
        for shard in SHARD_ORDER
    }
    return SourceIdentityPolicy(
        policy_id="production-mml-source-policy-v3",
        shards=approved,
        test_only=False,
        deduplication_roots=roots,
    )


def production_approved_shard_source(shard: str) -> ApprovedShardSource:
    """Return one independently finalized production shard without inventing siblings."""

    record = PRODUCTION_SOURCE_IDENTITY_TABLE.get(shard)
    if record is None:
        raise HoldoutError(f"{shard}: production source identity is not finalized")
    return _validated_production_source_record(shard, record)


def validate_production_shard_source(
    shard: str,
    source: ShardSource,
    *,
    test_only: bool,
) -> None:
    """Fail closed unless one planner input matches its finalized production roots."""

    if test_only:
        raise HoldoutError(
            f"{shard}: production planner refuses test-only source policy"
        )
    approved = production_approved_shard_source(shard)
    if source.name != shard or source.logical_path != f"raw/{shard}.jsonl":
        raise HoldoutError(f"{shard}: production logical raw input path is invalid")
    if source.expected_input_sha256 != approved.input_sha256:
        raise HoldoutError(f"{shard}: production raw input SHA-256 drift")
    if tuple(source.source_snapshots) != approved.source_snapshots:
        raise HoldoutError(f"{shard}: production source snapshots drift")
    if source.source_manifest_root_sha256 != approved.source_manifest_root_sha256:
        raise HoldoutError(f"{shard}: production source-manifest root drift")
    if source.quality_filter_root_sha256 != approved.quality_filter_root_sha256:
        raise HoldoutError(f"{shard}: production quality-filter root drift")
    if source.schema_generation_root_sha256 != approved.schema_generation_root_sha256:
        raise HoldoutError(f"{shard}: production schema-generation root drift")


def source_root(ordered_inputs: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered generated inputs and every upstream generation root."""

    return _json_sha256(list(ordered_inputs))


def _validate_policy_pins(pins: PolicyPins) -> None:
    expected = current_policy_pins()
    if pins.mapping_sha256 != expected.mapping_sha256:
        raise HoldoutError(
            "mapping policy SHA-256 drift: "
            f"expected {expected.mapping_sha256}, got {pins.mapping_sha256}"
        )
    if pins.policy_sha256 != expected.policy_sha256:
        raise HoldoutError(
            "holdout policy SHA-256 drift: "
            f"expected {expected.policy_sha256}, got {pins.policy_sha256}"
        )
    if pins.atp_deduplication_sha256 != expected.atp_deduplication_sha256:
        raise HoldoutError(
            "ATP deduplication policy SHA-256 drift: "
            f"expected {expected.atp_deduplication_sha256}, "
            f"got {pins.atp_deduplication_sha256}"
        )


def _validated_tokenizer_seal(seal: Mapping[str, Any]) -> dict[str, Any]:
    expected = approved_tokenizer_seal()
    if not isinstance(seal, Mapping):
        raise HoldoutError("tokenizer seal is missing")
    for key, expected_value in expected.items():
        if seal.get(key) != expected_value:
            raise HoldoutError(
                f"tokenizer seal mismatch for {key}: "
                f"expected {expected_value!r}, got {seal.get(key)!r}"
            )
    return expected


def _decode_tptp_atom(name: str) -> str:
    value = str(name).strip()
    if len(value) < 2 or value[0] != value[-1] or value[0] not in ("'", '"'):
        return value
    body = value[1:-1]
    decoded: list[str] = []
    index = 0
    while index < len(body):
        if body[index] == "\\" and index + 1 < len(body):
            index += 1
        decoded.append(body[index])
        index += 1
    return "".join(decoded)


def _normalize_atp_theorem_name(name: str) -> tuple[str, str]:
    raw = str(name).strip()
    value = raw
    prefix, separator, rest = value.partition(":")
    if separator and prefix.lower() in ATP_WRAPPERS:
        value = rest
    value = ALTERNATE_SUFFIX_RE.sub("", value)
    return raw, _decode_tptp_atom(value)


def _singleton_class_id(representation: str, native_name: str) -> str:
    payload = (
        f"{MAPPING_VERSION}\0representation-singleton\0{representation}\0{native_name}"
    )
    return (
        f"mml:v1:singleton:{representation}:"
        f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
    )


def semantic_identity(
    name: str,
    *,
    representation: str,
    theorem_identity: bool = False,
) -> SemanticIdentity:
    """Map one approved native name or assign a representation singleton."""

    if representation not in {"mizar", "atp"}:
        raise ValueError(f"unsupported representation {representation!r}")
    raw_name = str(name).strip()
    if representation == "atp" and theorem_identity:
        _, native_name = _normalize_atp_theorem_name(raw_name)
    elif representation == "atp":
        native_name = _decode_tptp_atom(raw_name)
    else:
        native_name = raw_name

    if representation == "mizar":
        theorem_match = MIZAR_THEOREM_RE.fullmatch(native_name)
        if theorem_match is not None:
            article, number = theorem_match.groups()
            return SemanticIdentity(
                class_id=f"mml:v1:theorem:{article}:{number}",
                kind="theorem",
                representation=representation,
                raw_name=raw_name,
                native_name=native_name,
                mapped=True,
            )
        definition_match = MIZAR_DEFINITION_RE.fullmatch(native_name)
        if definition_match is not None:
            article, number = definition_match.groups()
            return SemanticIdentity(
                class_id=f"mml:v1:definition:{article}:{number}",
                kind="definition",
                representation=representation,
                raw_name=raw_name,
                native_name=native_name,
                mapped=True,
            )
    else:
        match = ATP_MML_RE.fullmatch(native_name)
        if match is not None and "__" not in native_name:
            prefix, number, article = match.groups()
            kind = "theorem" if prefix.lower() == "t" else "definition"
            return SemanticIdentity(
                class_id=f"mml:v1:{kind}:{article.upper()}:{number}",
                kind=kind,
                representation=representation,
                raw_name=raw_name,
                native_name=native_name,
                mapped=True,
            )

    return SemanticIdentity(
        class_id=_singleton_class_id(representation, native_name),
        kind="representation_singleton",
        representation=representation,
        raw_name=raw_name,
        native_name=native_name,
        mapped=False,
    )


def _canonical_without_layout(statement: str) -> str:
    out: list[str] = []
    quote: str | None = None
    escaped = False
    for char in str(statement).strip():
        if quote is not None:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
            out.append(char)
        elif not char.isspace():
            out.append(char)
    return "".join(out)


TPTP_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
TPTP_CLOSE_TO_OPEN = {close: open_ for open_, close in TPTP_OPEN_TO_CLOSE.items()}


def _canonical_native_tptp_text(statement: str) -> str:
    return str(statement).replace("\r\n", "\n").replace("\r", "\n").strip()


def _tptp_delimiter_pairs(text: str) -> dict[int, int] | None:
    stack: list[tuple[str, int]] = []
    pairs: dict[int, int] = {}
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char == "\\":
            return None
        if char in ("'", '"'):
            quote = char
            continue
        if char in TPTP_OPEN_TO_CLOSE:
            stack.append((char, index))
            continue
        if char in TPTP_CLOSE_TO_OPEN:
            if not stack or stack[-1][0] != TPTP_CLOSE_TO_OPEN[char]:
                return None
            _, opening_index = stack.pop()
            pairs[opening_index] = index
    if stack or quote is not None or escaped:
        return None
    return pairs


def _has_complete_balanced_tptp_outer_parentheses(formula: str) -> bool:
    pairs = _tptp_delimiter_pairs(formula)
    return (
        pairs is not None
        and formula.startswith("(")
        and pairs.get(0) == len(formula) - 1
    )


def _canonical_tptp_formula(statement: str) -> str:
    native = _canonical_native_tptp_text(statement)
    if _tptp_delimiter_pairs(native) is None:
        return native
    formula = _canonical_without_layout(native)
    while _has_complete_balanced_tptp_outer_parentheses(formula):
        formula = formula[1:-1]
    return formula


def _canonical_collapsed_layout(statement: str) -> str:
    out: list[str] = []
    quote: str | None = None
    escaped = False
    pending_space = False
    for char in str(statement).strip():
        if quote is not None:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            if pending_space and out:
                out.append(" ")
            pending_space = False
            quote = char
            out.append(char)
        elif char.isspace():
            pending_space = True
        else:
            if pending_space and out:
                out.append(" ")
            pending_space = False
            out.append(char)
    return "".join(out)


def canonical_statement(statement: str, *, representation: str) -> str:
    """Canonicalize layout without crossing the Mizar/TPTP syntax boundary."""

    if representation == "atp":
        return _canonical_tptp_formula(statement)
    if representation == "mizar":
        return _canonical_collapsed_layout(statement)
    raise ValueError(f"unsupported representation {representation!r}")


def statement_digest(representation: str, statement: str) -> str:
    """Hash a statement in a representation-specific namespace."""

    scheme = (
        "tptp-ordered-delimiters-complete-outer-formula-v4"
        if representation == "atp"
        else "mizar-layout-v1"
    )
    payload = "\0".join(
        (
            STATEMENT_HASH_VERSION,
            representation,
            scheme,
            canonical_statement(statement, representation=representation),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_atp_mapping(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, Mapping):
        return []
    return sorted(
        (
            _decode_tptp_atom(str(name)),
            canonical_statement(str(statement), representation="atp"),
        )
        for name, statement in value.items()
    )


def _canonical_atp_step(step: Any) -> Any:
    if not isinstance(step, Mapping):
        return step
    canonical: dict[str, Any] = {}
    for key, value in step.items():
        if key in {"formula", "source"}:
            canonical[str(key)] = canonical_statement(str(value), representation="atp")
        elif key == "parent_sources" and isinstance(value, list):
            canonical[str(key)] = [
                canonical_statement(str(parent), representation="atp")
                for parent in value
            ]
        else:
            canonical[str(key)] = value
    return canonical


def mizar_thproofs_trajectory_signature(record: Mapping[str, Any]) -> str:
    """Hash the theorem, goal, and target used to identify one Mizar trajectory."""

    payload = {
        "policy": MIZAR_THPROOFS_DEDUPLICATION_POLICY,
        "theorem": str(record.get("theorem", "")),
        "goal": str(record.get("goal", "")).replace("\r\n", "\n"),
        "target": str(record.get("target", "")).replace("\r\n", "\n"),
    }
    return _json_sha256(payload)


def exact_atp_signature(record: Mapping[str, Any]) -> str:
    """Hash one exact ATP proof independent of wrapper, ID, and map order."""

    _, theorem = _normalize_atp_theorem_name(str(record.get("theorem", "")))
    proof_steps = record.get("proof_steps")
    structured = isinstance(proof_steps, list) and bool(proof_steps)
    signature: dict[str, Any] = {
        "theorem": theorem,
        "goal": canonical_statement(str(record.get("goal", "")), representation="atp"),
        "goal_name": record.get("goal_name"),
        "facts": _canonical_atp_mapping(record.get("facts", {})),
        "local_inputs": _canonical_atp_mapping(record.get("local_inputs", {})),
    }
    if structured:
        signature["schema"] = "atp-v2-structured"
        signature["proof_steps"] = [_canonical_atp_step(step) for step in proof_steps]
    else:
        signature["schema"] = "atp-legacy-conservative"
        signature["target"] = (
            str(record.get("target", "")).replace("\r\n", "\n").strip()
        )
    payload = ATP_DEDUPLICATION_POLICY.encode("utf-8") + b"\0"
    payload += _canonical_json_bytes(signature)
    return hashlib.sha256(payload).hexdigest()


ATP_STEP_FIELDS = (
    "name",
    "role",
    "formula",
    "rule",
    "parents",
    "parent_sources",
    "source",
)


def _require_well_formed_tptp_delimiters(value: Any, *, where: str) -> None:
    if not isinstance(value, str) or _tptp_delimiter_pairs(value) is None:
        raise HoldoutError(f"{where}: malformed TPTP delimiters")


def validate_atp_v2_record(
    record: Mapping[str, Any],
    *,
    where: str = "ATP row",
) -> None:
    """Fail closed on the complete replay-relevant ATP-v2 integrity contract."""

    facts = record.get("facts")
    local_inputs = record.get("local_inputs")
    goal_name = record.get("goal_name")
    if not isinstance(facts, Mapping) or not isinstance(local_inputs, Mapping):
        raise HoldoutError(f"{where}: malformed ATP global/local supply")
    if not isinstance(goal_name, str) or not goal_name.strip():
        raise HoldoutError(f"{where}: malformed ATP goal supply")
    for supply_name, supply in (("facts", facts), ("local_inputs", local_inputs)):
        for name, formula in supply.items():
            _require_well_formed_tptp_delimiters(
                name,
                where=f"{where}: {supply_name} name",
            )
            _require_well_formed_tptp_delimiters(
                formula,
                where=f"{where}: {supply_name} formula {name}",
            )
    _require_well_formed_tptp_delimiters(
        goal_name,
        where=f"{where}: goal name",
    )
    _require_well_formed_tptp_delimiters(
        record.get("goal"),
        where=f"{where}: goal formula",
    )
    _require_well_formed_tptp_delimiters(
        record.get("target"),
        where=f"{where}: target source structure",
    )
    global_names = set(facts)
    local_names = set(local_inputs)
    overlap = global_names & local_names
    if overlap:
        raise HoldoutError(
            f"{where}: global/local supply overlap {sorted(overlap)[:3]}"
        )
    if goal_name in global_names or goal_name in local_names:
        raise HoldoutError(f"{where}: goal/global/local supply overlap {goal_name}")

    proof_steps = record.get("proof_steps")
    if not isinstance(proof_steps, list) or not proof_steps:
        raise HoldoutError(f"{where}: proof_steps must be a nonempty ATP-v2 list")

    parsed_steps: list[ProofStep] = []
    for index, raw_step in enumerate(proof_steps, 1):
        if not isinstance(raw_step, Mapping):
            raise HoldoutError(f"{where}: ATP step {index} is not an object")
        missing = [field for field in ATP_STEP_FIELDS if field not in raw_step]
        if missing:
            raise HoldoutError(
                f"{where}: missing ATP step fields at step {index}: {missing}"
            )
        for field in ("name", "role", "formula", "rule", "source"):
            if not isinstance(raw_step[field], str) or not raw_step[field].strip():
                raise HoldoutError(f"{where}: ATP step {index} has malformed {field}")
        for field in ("name", "formula", "source"):
            _require_well_formed_tptp_delimiters(
                raw_step[field],
                where=f"{where}: ATP step {index} {field}",
            )
        parents = raw_step["parents"]
        parent_sources = raw_step["parent_sources"]
        if (
            not isinstance(parents, list)
            or not all(isinstance(parent, str) and parent for parent in parents)
            or not isinstance(parent_sources, list)
            or not all(isinstance(parent, str) and parent for parent in parent_sources)
        ):
            raise HoldoutError(f"{where}: ATP step {index} has malformed parents")
        for field, values in (("parents", parents), ("parent_sources", parent_sources)):
            for parent_index, value in enumerate(values, 1):
                _require_well_formed_tptp_delimiters(
                    value,
                    where=(f"{where}: ATP step {index} {field} item {parent_index}"),
                )
        parsed_source = source_dependencies(raw_step["source"])
        if parsed_source is None:
            raise HoldoutError(
                f"{where}: ATP step {index} has no trusted derived source"
            )
        rule, parsed_parent_sources, parsed_parents = parsed_source
        if (
            rule != raw_step["rule"]
            or parsed_parent_sources != parent_sources
            or parsed_parents != parents
        ):
            raise HoldoutError(f"{where}: ATP source-parent mismatch at step {index}")
        parsed_steps.append(
            ProofStep(
                name=raw_step["name"],
                role=raw_step["role"],
                formula=raw_step["formula"],
                rule=raw_step["rule"],
                parents=list(parents),
                parent_sources=list(parent_sources),
                source=raw_step["source"],
            )
        )

    step_names = [step.name for step in parsed_steps]
    duplicates = sorted(
        name for name, count in Counter(step_names).items() if count > 1
    )
    if duplicates:
        raise HoldoutError(f"{where}: duplicate ATP step {duplicates[:3]}")
    supplied = global_names | local_names | {goal_name}
    collisions = supplied & set(step_names)
    if collisions:
        raise HoldoutError(f"{where}: ATP step/supply overlap {sorted(collisions)[:3]}")
    all_steps = set(step_names)
    seen: set[str] = set()
    for step in parsed_steps:
        for parent in dict.fromkeys(step.parents):
            if parent in supplied or parent in seen:
                continue
            if parent in all_steps:
                raise HoldoutError(f"{where}: late parent {step.name} <- {parent}")
            raise HoldoutError(f"{where}: unresolved parent {step.name} <- {parent}")
        seen.add(step.name)

    if render_target(parsed_steps) != record.get("target"):
        raise HoldoutError(f"{where}: ATP target reconstruction mismatch")
    if not is_refutation_formula(parsed_steps[-1].formula):
        raise HoldoutError(f"{where}: final semantic formula is not $false")


def _selected_statement_hits(
    statement: str,
    *,
    representation: str,
    field: str,
    index: ExposureIndex,
) -> set[ExposureHit]:
    if not statement:
        return set()
    digest = statement_digest(representation, statement)
    return {
        ExposureHit(class_id=class_id, field=field)
        for class_id in index.statement_classes_by_representation.get(
            representation, {}
        ).get(digest, ())
    }


def _visible_target_candidates(record: Mapping[str, Any]) -> Iterator[str]:
    proof_steps = record.get("proof_steps")
    if isinstance(proof_steps, list):
        for step in proof_steps:
            if isinstance(step, Mapping) and isinstance(step.get("formula"), str):
                yield step["formula"]
    explicit = record.get("visible_targets")
    if isinstance(explicit, str):
        yield explicit
    elif isinstance(explicit, list):
        for item in explicit:
            if isinstance(item, str):
                yield item
    target = record.get("target")
    if isinstance(target, str) and target.strip():
        yield target
        yield from (line.strip() for line in target.splitlines() if line.strip())


MIZAR_EXPOSURE_TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|'
    r"'(?:\\.|[^'\\])*'|"
    r"[A-Za-z_][A-Za-z0-9_]*|"
    r"\d+|:=|::|<=|>=|<>|\S"
)


def _mizar_exposure_tokens(text: str) -> tuple[str, ...]:
    return tuple(MIZAR_EXPOSURE_TOKEN_RE.findall(str(text)))


def _contains_token_sequence(
    haystack: Sequence[str],
    needle: Sequence[str],
) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        tuple(haystack[index : index + width]) == tuple(needle)
        for index in range(len(haystack) - width + 1)
    )


def _embedded_mizar_target_hits(
    target: str,
    *,
    index: ExposureIndex,
) -> set[ExposureHit]:
    target_tokens = _mizar_exposure_tokens(target)
    hits: set[ExposureHit] = set()
    for statement, class_ids in index.canonical_statement_classes_by_representation.get(
        "mizar", {}
    ).items():
        if _contains_token_sequence(
            target_tokens,
            _mizar_exposure_tokens(statement),
        ):
            hits.update(
                ExposureHit(class_id=class_id, field="target_embedded")
                for class_id in class_ids
            )
    return hits


def classify_exposure(
    record: Mapping[str, Any],
    *,
    shard: str,
    index: ExposureIndex,
) -> HeldoutExposure:
    """Classify every direct, mapped, own, alias, and target exposure path."""

    representation = REPRESENTATION_BY_SHARD.get(shard)
    if representation is None:
        raise ValueError(f"unsupported shard {shard!r}")
    direct: set[ExposureHit] = set()
    own: set[ExposureHit] = set()
    aliases: set[ExposureHit] = set()
    targets: set[ExposureHit] = set()

    for field in ("facts", "local_inputs"):
        values = record.get(field, {})
        if not isinstance(values, Mapping):
            continue
        for raw_name, statement in values.items():
            identity = semantic_identity(
                str(raw_name),
                representation=representation,
            )
            if identity.class_id in index.selected_class_ids:
                direct.add(
                    ExposureHit(
                        class_id=identity.class_id,
                        field=field,
                        native_name=identity.native_name,
                    )
                )
            elif isinstance(statement, str):
                aliases |= _selected_statement_hits(
                    statement,
                    representation=representation,
                    field=field,
                    index=index,
                )

    theorem = semantic_identity(
        str(record.get("theorem", "")),
        representation=representation,
        theorem_identity=True,
    )
    if theorem.class_id in index.selected_class_ids:
        own.add(
            ExposureHit(
                class_id=theorem.class_id,
                field="theorem",
                native_name=theorem.native_name,
            )
        )
    else:
        goal = record.get("goal")
        if isinstance(goal, str):
            aliases |= _selected_statement_hits(
                goal,
                representation=representation,
                field="goal",
                index=index,
            )

    for statement in _visible_target_candidates(record):
        targets |= _selected_statement_hits(
            statement,
            representation=representation,
            field="target",
            index=index,
        )
    if representation == "mizar" and isinstance(record.get("target"), str):
        targets |= _embedded_mizar_target_hits(
            record["target"],
            index=index,
        )

    cross: set[ExposureHit] = set()
    for hit in direct | own:
        represented = index.members_by_class.get(hit.class_id, {})
        if any(
            other != representation and members
            for other, members in represented.items()
        ):
            cross.add(
                ExposureHit(
                    class_id=hit.class_id,
                    field="semantic_mapping",
                    native_name=hit.native_name,
                )
            )

    return HeldoutExposure(
        direct_native_citation=tuple(sorted(direct)),
        mapped_cross_representation=tuple(sorted(cross)),
        own_theorem=tuple(sorted(own)),
        statement_alias=tuple(sorted(aliases)),
        visible_target=tuple(sorted(targets)),
    )


def _validate_text_mapping(
    value: Any,
    *,
    where: str,
    field: str,
    allow_empty: bool = False,
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise HoldoutError(f"{where}: {field} must be an object")
    if not value and not allow_empty:
        raise HoldoutError(f"{where}: {field} must be nonempty")
    for name, statement in value.items():
        if not isinstance(name, str) or not name.strip():
            raise HoldoutError(f"{where}: {field} contains a malformed name")
        if not isinstance(statement, str) or not statement.strip():
            raise HoldoutError(
                f"{where}: {field} contains an empty statement for {name!r}"
            )
    return value


def _validate_row_schema(
    record: Any,
    *,
    shard: str,
    line_number: int,
) -> dict[str, Any]:
    where = f"{shard}:{line_number}"
    if not isinstance(record, dict):
        raise HoldoutError(f"{where}: row must be a JSON object")
    missing = [field for field in COMMON_REQUIRED_FIELDS if field not in record]
    if missing:
        raise HoldoutError(f"{where}: missing required fields {missing}")
    for field in ("id", "theorem", "goal", "target", "text"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise HoldoutError(f"{where}: {field} must be nonempty text")
    for field in ("mask_start", "mask_end"):
        if isinstance(record[field], bool) or not isinstance(record[field], int):
            raise HoldoutError(f"{where}: {field} must be an integer")
    if not (0 <= record["mask_start"] <= record["mask_end"] <= len(record["text"])):
        raise HoldoutError(f"{where}: mask offsets are out of bounds")

    facts = _validate_text_mapping(record["facts"], where=where, field="facts")
    cited = record["cited"]
    if not isinstance(cited, list) or not all(
        isinstance(name, str) and name.strip() for name in cited
    ):
        raise HoldoutError(f"{where}: cited must be a list of names")
    missing_citations = set(cited) - set(facts)
    if missing_citations:
        raise HoldoutError(
            f"{where}: cited names lack statements: {sorted(missing_citations)[:3]}"
        )

    if shard in {"prf2", "enigma"}:
        if record.get("schema_version") != "atp-v2":
            raise HoldoutError(f"{where}: ATP rows must use schema_version atp-v2")
        missing_atp = [field for field in ATP_REQUIRED_FIELDS if field not in record]
        if missing_atp:
            raise HoldoutError(f"{where}: missing ATP fields {missing_atp}")
        _validate_text_mapping(
            record["local_inputs"],
            where=where,
            field="local_inputs",
            allow_empty=True,
        )
        if not isinstance(record["goal_name"], str) or not record["goal_name"].strip():
            raise HoldoutError(f"{where}: goal_name must be nonempty text")
        proof_steps = record["proof_steps"]
        if not isinstance(proof_steps, list) or not proof_steps:
            raise HoldoutError(f"{where}: proof_steps must be a nonempty list")
        validate_atp_v2_record(record, where=where)
        for raw_name in facts:
            native_name = _decode_tptp_atom(raw_name)
            if BOOKKEEPING_RE.match(native_name):
                raise HoldoutError(
                    f"{where}: bookkeeping name {raw_name!r} appears in global facts"
                )
    return record


def _source_digest_and_lines(source: ShardSource) -> tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    for line in source.iter_lines():
        if not isinstance(line, bytes):
            raise HoldoutError(f"{source.name}: source yielded a non-byte line")
        digest.update(line)
        lines += 1
    return digest.hexdigest(), lines


def _validate_sources(
    sources: Mapping[str, ShardSource],
    *,
    source_policy: SourceIdentityPolicy,
) -> list[dict[str, Any]]:
    if set(sources) != set(SHARD_ORDER):
        missing = sorted(set(SHARD_ORDER) - set(sources))
        extra = sorted(set(sources) - set(SHARD_ORDER))
        raise HoldoutError(
            f"exactly four native shards are required; missing={missing}, extra={extra}"
        )
    if set(source_policy.shards) != set(SHARD_ORDER):
        raise HoldoutError("source identity policy must approve exactly four shards")
    deduplication_roots = _source_policy_deduplication_roots(source_policy)
    if not source_policy.test_only:
        approved_production = production_source_policy()
        if source_policy != approved_production:
            raise HoldoutError(
                "non-test source identity policy is not production-approved"
            )
    metadata = []
    for shard in SHARD_ORDER:
        source = sources[shard]
        approved = source_policy.shards[shard]
        if source.name != shard:
            raise HoldoutError(
                f"source key {shard!r} disagrees with source name {source.name!r}"
            )
        expected = str(source.expected_input_sha256).lower()
        if not SHA256_RE.fullmatch(expected):
            raise HoldoutError(
                f"{shard}: expected input SHA-256 is missing or malformed"
            )
        if expected != approved.input_sha256:
            raise HoldoutError(
                f"{shard}: input SHA-256 is not approved by source identity policy"
            )
        snapshots = tuple(source.source_snapshots)
        if not snapshots:
            raise HoldoutError(f"{shard}: at least one source snapshot is required")
        snapshot_records = []
        for snapshot in snapshots:
            reference = str(snapshot.reference).strip()
            digest = str(snapshot.sha256).lower()
            if not reference or not SHA256_RE.fullmatch(digest):
                raise HoldoutError(
                    f"{shard}: source snapshot reference or SHA-256 is missing"
                )
            snapshot_records.append({"reference": reference, "sha256": digest})
        snapshot_records.sort(key=lambda item: (item["reference"], item["sha256"]))
        if snapshot_records != _snapshot_records(approved.source_snapshots):
            raise HoldoutError(
                f"{shard}: source snapshots do not match approved source manifest"
            )
        roots = {
            "source_manifest_root_sha256": source.source_manifest_root_sha256,
            "quality_filter_root_sha256": source.quality_filter_root_sha256,
            "schema_generation_root_sha256": source.schema_generation_root_sha256,
            "deduplication_root_sha256": deduplication_roots[shard],
        }
        approved_roots = {
            "source_manifest_root_sha256": approved.source_manifest_root_sha256,
            "quality_filter_root_sha256": approved.quality_filter_root_sha256,
            "schema_generation_root_sha256": approved.schema_generation_root_sha256,
            "deduplication_root_sha256": production_deduplication_root(shard),
        }
        for key, value in roots.items():
            normalized = str(value).lower()
            label = key.removesuffix("_root_sha256").replace("_", "-")
            if not SHA256_RE.fullmatch(normalized):
                raise HoldoutError(f"{shard}: {label} root is missing or malformed")
            if normalized != str(approved_roots[key]).lower():
                raise HoldoutError(
                    f"{shard}: {label} root is not approved by source identity policy"
                )
            roots[key] = normalized
        actual, line_count = _source_digest_and_lines(source)
        if actual != expected:
            raise HoldoutError(
                f"{shard}: input SHA-256 mismatch; expected {expected}, got {actual}"
            )
        metadata.append(
            {
                "shard": shard,
                "representation": REPRESENTATION_BY_SHARD[shard],
                "logical_path": source.logical_path,
                "sha256": actual,
                "rows": line_count,
                "source_snapshots": snapshot_records,
                "acceptance_roots": dict(approved.acceptance_roots),
                **roots,
            }
        )
    return metadata


def _iter_parsed_rows(
    source: ShardSource,
    *,
    tokenizer: TokenizerSeam,
    token_cache: dict[tuple[str, int], int],
) -> Iterator[tuple[int, dict[str, Any], int, str]]:
    digest = hashlib.sha256()
    for line_number, line in enumerate(source.iter_lines(), 1):
        if not isinstance(line, bytes):
            raise HoldoutError(f"{source.name}:{line_number}: source line is not bytes")
        digest.update(line)
        native_row_sha256 = hashlib.sha256(line).hexdigest()
        try:
            text_line = line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise HoldoutError(
                f"{source.name}:{line_number}: JSONL is not valid UTF-8"
            ) from error
        if not text_line.strip():
            raise HoldoutError(f"{source.name}:{line_number}: blank JSONL row")
        try:
            parsed = json.loads(text_line)
        except json.JSONDecodeError as error:
            raise HoldoutError(f"{source.name}:{line_number}: invalid JSON") from error
        record = _validate_row_schema(
            parsed,
            shard=source.name,
            line_number=line_number,
        )
        cache_key = (source.name, line_number)
        token_count = token_cache.get(cache_key)
        if token_count is None:
            try:
                token_count = tokenizer.count_text_plus_eos(record["text"])
            except Exception as error:
                raise HoldoutError(
                    f"{source.name}:{line_number}: tokenizer length check failed"
                ) from error
            if (
                isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count < 1
            ):
                raise HoldoutError(
                    f"{source.name}:{line_number}: tokenizer returned invalid length "
                    f"{token_count!r}"
                )
            token_cache[cache_key] = token_count
        yield line_number, record, token_count, native_row_sha256
    actual = digest.hexdigest()
    expected = str(source.expected_input_sha256).lower()
    if actual != expected:
        raise HoldoutError(
            f"{source.name}: input changed or iteration is nondeterministic; "
            f"expected {expected}, got {actual}"
        )


def _validate_global_row_ids(
    sources: Mapping[str, ShardSource],
    *,
    tokenizer: TokenizerSeam,
    token_cache: dict[tuple[str, int], int],
) -> None:
    seen: dict[str, tuple[str, int]] = {}
    for shard in SHARD_ORDER:
        for line_number, record, _, _ in _iter_parsed_rows(
            sources[shard],
            tokenizer=tokenizer,
            token_cache=token_cache,
        ):
            row_id = record["id"]
            previous = seen.get(row_id)
            if previous is not None:
                raise HoldoutError(
                    f"duplicate raw row id {row_id!r}: "
                    f"{previous[0]}:{previous[1]} and {shard}:{line_number}"
                )
            seen[row_id] = (shard, line_number)


def _eligible_direct_mizar_trajectories(
    sources: Mapping[str, ShardSource],
    *,
    tokenizer: TokenizerSeam,
    token_cache: dict[tuple[str, int], int],
) -> dict[str, str]:
    trajectories: dict[str, str] = {}
    locations: dict[str, int] = {}
    for line_number, record, token_count, _ in _iter_parsed_rows(
        sources["mizar"],
        tokenizer=tokenizer,
        token_cache=token_cache,
    ):
        if token_count > MAX_TEXT_PLUS_EOS_TOKENS:
            continue
        theorem = str(record["theorem"])
        if theorem in trajectories:
            raise HoldoutError(
                f"mizar:{line_number}: duplicate direct Mizar theorem trajectory "
                f"{theorem!r}; first seen on line {locations[theorem]}"
            )
        trajectories[theorem] = mizar_thproofs_trajectory_signature(record)
        locations[theorem] = line_number
    return trajectories


def _iter_prepared_rows(
    sources: Mapping[str, ShardSource],
    *,
    tokenizer: TokenizerSeam,
    token_cache: dict[tuple[str, int], int],
    dedup_stats: dict[str, Any] | None = None,
) -> Iterator[_PreparedRow]:
    direct_mizar_trajectories = _eligible_direct_mizar_trajectories(
        sources,
        tokenizer=tokenizer,
        token_cache=token_cache,
    )
    seen_atp_signatures: set[str] = set()
    kept_atp_signatures: list[str] = []
    atp_duplicate_counts = {"prf2": 0, "enigma": 0}
    mizar_duplicate_counts = {"mizar": 0, "thproofs": 0}
    thproofs_trajectories = 0
    thproofs_only_trajectories = 0
    for shard in SHARD_ORDER:
        source = sources[shard]
        for line_number, record, token_count, native_row_sha256 in _iter_parsed_rows(
            source,
            tokenizer=tokenizer,
            token_cache=token_cache,
        ):
            drop_reason = None
            signature = None
            if shard == "thproofs":
                thproofs_trajectories += 1
                theorem = str(record["theorem"])
                direct_signature = direct_mizar_trajectories.get(theorem)
                if direct_signature is None:
                    thproofs_only_trajectories += 1
                else:
                    thproofs_signature = mizar_thproofs_trajectory_signature(record)
                    if thproofs_signature != direct_signature:
                        raise HoldoutError(
                            f"thproofs:{line_number}: theorem {theorem!r} disagrees "
                            "with its direct Mizar trajectory"
                        )
                    drop_reason = "direct_mizar_trajectory_duplicate"
                    mizar_duplicate_counts["thproofs"] += 1
            if drop_reason is None and token_count > MAX_TEXT_PLUS_EOS_TOKENS:
                drop_reason = "overlength"
            elif drop_reason is None and shard in {"prf2", "enigma"}:
                signature = exact_atp_signature(record)
                if signature in seen_atp_signatures:
                    drop_reason = "exact_atp_duplicate"
                    atp_duplicate_counts[shard] += 1
                else:
                    seen_atp_signatures.add(signature)
                    kept_atp_signatures.append(signature)
            yield _PreparedRow(
                shard=shard,
                line_number=line_number,
                record=record,
                text_plus_eos_tokens=token_count,
                native_row_sha256=native_row_sha256,
                pre_drop_reason=drop_reason,
                atp_signature=signature,
            )
    if dedup_stats is not None:
        dedup_stats.update(
            {
                "mizar_thproofs": {
                    "direct_mizar_trajectories": len(direct_mizar_trajectories),
                    "thproofs_trajectories": thproofs_trajectories,
                    "thproofs_only_trajectories": thproofs_only_trajectories,
                    "duplicates_by_shard": mizar_duplicate_counts,
                    "duplicates_total": sum(mizar_duplicate_counts.values()),
                },
                "atp": {
                    "duplicates_by_shard": atp_duplicate_counts,
                    "duplicates_total": sum(atp_duplicate_counts.values()),
                    "kept_signature_root_sha256": hashlib.sha256(
                        "\n".join(kept_atp_signatures).encode("ascii")
                    ).hexdigest(),
                    "kept_signatures": len(kept_atp_signatures),
                },
            }
        )


def _row_identity_statements(
    record: Mapping[str, Any],
    *,
    shard: str,
) -> Iterator[tuple[SemanticIdentity, str]]:
    representation = REPRESENTATION_BY_SHARD[shard]
    for raw_name, statement in record["facts"].items():
        yield (
            semantic_identity(str(raw_name), representation=representation),
            str(statement),
        )
    yield (
        semantic_identity(
            str(record["theorem"]),
            representation=representation,
            theorem_identity=True,
        ),
        str(record["goal"]),
    )


def _collect_stability(
    sources: Mapping[str, ShardSource],
    *,
    tokenizer: TokenizerSeam,
    token_cache: dict[tuple[str, int], int],
) -> tuple[
    frozenset[tuple[str, str]],
    frozenset[tuple[str, str]],
]:
    native_digests: dict[tuple[str, str], set[str]] = defaultdict(set)
    class_representation_digests: dict[tuple[str, str], set[str]] = defaultdict(set)
    class_representation_names: dict[tuple[str, str], set[str]] = defaultdict(set)
    for prepared in _iter_prepared_rows(
        sources,
        tokenizer=tokenizer,
        token_cache=token_cache,
    ):
        if prepared.pre_drop_reason is not None:
            continue
        representation = REPRESENTATION_BY_SHARD[prepared.shard]
        for identity, statement in _row_identity_statements(
            prepared.record,
            shard=prepared.shard,
        ):
            digest = statement_digest(representation, statement)
            native_digests[(representation, identity.native_name)].add(digest)
            class_key = (identity.class_id, representation)
            class_representation_digests[class_key].add(digest)
            class_representation_names[class_key].add(identity.native_name)

    unstable_native = frozenset(
        key for key, digests in native_digests.items() if len(digests) > 1
    )
    unstable_classes = frozenset(
        key
        for key, digests in class_representation_digests.items()
        if len(digests) > 1 and len(class_representation_names[key]) > 1
    )
    return unstable_native, unstable_classes


def _row_disagreement_reason(
    record: Mapping[str, Any],
    *,
    shard: str,
    unstable_native: frozenset[tuple[str, str]],
    unstable_classes: frozenset[tuple[str, str]],
) -> str | None:
    observations = tuple(_row_identity_statements(record, shard=shard))
    if any(
        (identity.representation, identity.native_name) in unstable_native
        for identity, _ in observations
    ):
        return "statement_disagreement"
    if any(
        (identity.class_id, identity.representation) in unstable_classes
        for identity, _ in observations
    ):
        return "class_member_disagreement"
    return None


def _new_class_observation() -> dict[str, Any]:
    return {
        "kinds": set(),
        "members_by_shard": defaultdict(lambda: defaultdict(set)),
        "statement_digests_by_representation": defaultdict(set),
        "canonical_statements_by_representation": defaultdict(set),
        "source_snapshot_references_by_shard": defaultdict(set),
    }


def _add_class_observations(
    observations: dict[str, dict[str, Any]],
    *,
    record: Mapping[str, Any],
    shard: str,
    source: ShardSource,
) -> None:
    representation = REPRESENTATION_BY_SHARD[shard]
    snapshot_refs = {snapshot.reference for snapshot in source.source_snapshots}
    for identity, statement in _row_identity_statements(record, shard=shard):
        class_observation = observations.setdefault(
            identity.class_id,
            _new_class_observation(),
        )
        class_observation["kinds"].add(identity.kind)
        class_observation["members_by_shard"][shard][identity.native_name].add(
            identity.raw_name
        )
        class_observation["statement_digests_by_representation"][representation].add(
            statement_digest(representation, statement)
        )
        class_observation["canonical_statements_by_representation"][representation].add(
            canonical_statement(statement, representation=representation)
        )
        class_observation["source_snapshot_references_by_shard"][shard].update(
            snapshot_refs
        )


def draw_tail_classes(
    counts: Mapping[str, int],
    *,
    seed: int = SEED,
    requested: int = REQUESTED_CLASSES,
) -> tuple[tuple[str, ...], int]:
    """Draw exactly 1,000 classes from pooled one/two-row citation tail."""

    if seed != SEED:
        raise HoldoutError(f"approved seed is {SEED}, got {seed}")
    if requested != REQUESTED_CLASSES:
        raise HoldoutError(
            f"approved request is exactly {REQUESTED_CLASSES}, got {requested}"
        )
    tail = sorted(class_id for class_id, count in counts.items() if count in (1, 2))
    if len(tail) < requested:
        raise HoldoutError(
            f"insufficient tail: need {requested} semantic classes, found {len(tail)}"
        )
    selected_once = random.Random(seed).sample(tail, requested)
    selected_twice = random.Random(seed).sample(tail, requested)
    if selected_once != selected_twice:
        raise HoldoutError("deterministic class draw check failed")
    return tuple(sorted(selected_once)), len(tail)


def _count_classes_and_observe(
    sources: Mapping[str, ShardSource],
    *,
    tokenizer: TokenizerSeam,
    token_cache: dict[tuple[str, int], int],
    unstable_native: frozenset[tuple[str, str]],
    unstable_classes: frozenset[tuple[str, str]],
) -> tuple[Counter[str], dict[str, dict[str, Any]]]:
    counts: Counter[str] = Counter()
    observations: dict[str, dict[str, Any]] = {}
    for prepared in _iter_prepared_rows(
        sources,
        tokenizer=tokenizer,
        token_cache=token_cache,
    ):
        if prepared.pre_drop_reason is not None:
            continue
        disagreement = _row_disagreement_reason(
            prepared.record,
            shard=prepared.shard,
            unstable_native=unstable_native,
            unstable_classes=unstable_classes,
        )
        if disagreement is not None:
            continue
        representation = REPRESENTATION_BY_SHARD[prepared.shard]
        cited_classes = {
            semantic_identity(
                name,
                representation=representation,
            ).class_id
            for name in prepared.record["cited"]
        }
        counts.update(cited_classes)
        _add_class_observations(
            observations,
            record=prepared.record,
            shard=prepared.shard,
            source=sources[prepared.shard],
        )
    return counts, observations


def _build_exposure_index(
    selected: Sequence[str],
    observations: Mapping[str, Mapping[str, Any]],
) -> ExposureIndex:
    selected_ids = frozenset(selected)
    members_by_class: dict[str, dict[str, frozenset[str]]] = {}
    statement_classes: dict[str, dict[str, set[str]]] = {
        "mizar": defaultdict(set),
        "atp": defaultdict(set),
    }
    canonical_statement_classes: dict[str, dict[str, set[str]]] = {
        "mizar": defaultdict(set),
        "atp": defaultdict(set),
    }
    for class_id in selected:
        observation = observations[class_id]
        represented_members: dict[str, set[str]] = {
            "mizar": set(),
            "atp": set(),
        }
        for shard, members in observation["members_by_shard"].items():
            represented_members[REPRESENTATION_BY_SHARD[shard]].update(members)
        members_by_class[class_id] = {
            representation: frozenset(sorted(members))
            for representation, members in represented_members.items()
            if members
        }
        for representation, digests in observation[
            "statement_digests_by_representation"
        ].items():
            for digest in digests:
                statement_classes[representation][digest].add(class_id)
        for representation, statements in observation[
            "canonical_statements_by_representation"
        ].items():
            for statement in statements:
                canonical_statement_classes[representation][statement].add(class_id)
    return ExposureIndex(
        selected_class_ids=selected_ids,
        members_by_class=members_by_class,
        statement_classes_by_representation={
            representation: {
                digest: frozenset(sorted(class_ids))
                for digest, class_ids in sorted(digests.items())
            }
            for representation, digests in statement_classes.items()
        },
        canonical_statement_classes_by_representation={
            representation: {
                statement: frozenset(sorted(class_ids))
                for statement, class_ids in sorted(statements.items())
            }
            for representation, statements in canonical_statement_classes.items()
        },
    )


def _class_record(
    class_id: str,
    *,
    count: int,
    observation: Mapping[str, Any],
    route_summary: Mapping[str, Any],
) -> dict[str, Any]:
    kinds = sorted(observation["kinds"])
    if len(kinds) != 1:
        raise HoldoutError(f"class {class_id} has inconsistent kinds {kinds}")
    members_by_shard = {}
    for shard in SHARD_ORDER:
        members = observation["members_by_shard"].get(shard)
        if not members:
            continue
        members_by_shard[shard] = [
            {
                "native_name": native_name,
                "raw_names": sorted(raw_names),
            }
            for native_name, raw_names in sorted(members.items())
        ]
    return {
        "class_id": class_id,
        "kind": kinds[0],
        "selected_tail_row_citations": count,
        "native_members_by_shard": members_by_shard,
        "statement_digests_by_representation": {
            representation: sorted(digests)
            for representation, digests in sorted(
                observation["statement_digests_by_representation"].items()
            )
        },
        "source_snapshot_references_by_shard": {
            shard: sorted(references)
            for shard, references in sorted(
                observation["source_snapshot_references_by_shard"].items()
            )
        },
        "route_totals": dict(route_summary["route_totals"]),
        "route_root_sha256": route_summary["route_root_sha256"],
    }


def _empty_projection_counts() -> dict[str, dict[str, int]]:
    return {
        disposition: {"rows": 0, "text_plus_eos_tokens": 0}
        for disposition in ("train", "eval", "drop")
    }


def _exposure_sidecar_record(
    shard: str,
    route: RowPlan,
) -> dict[str, Any]:
    if route.disposition != "eval" or route.exposure is None:
        raise HoldoutError(
            f"{shard}:{route.line_number}: eval route lacks exposure metadata"
        )
    return {
        "shard": shard,
        "line_number": route.line_number,
        "row_id": route.row_id,
        "native_row_sha256": route.native_row_sha256,
        "paths": route.exposure.paths_dict(),
    }


def _drop_sidecar_record(shard: str, route: RowPlan) -> dict[str, Any]:
    if route.disposition != "drop" or not route.drop_reason:
        raise HoldoutError(
            f"{shard}:{route.line_number}: drop route lacks reason metadata"
        )
    return {
        "shard": shard,
        "line_number": route.line_number,
        "row_id": route.row_id,
        "native_row_sha256": route.native_row_sha256,
        "reason": route.drop_reason,
    }


def _route_record(shard: str, route: RowPlan) -> dict[str, Any]:
    record = {
        "line_number": route.line_number,
        "row_id": route.row_id,
        "native_row_sha256": route.native_row_sha256,
        "disposition": route.disposition,
        "drop_reason": route.drop_reason,
        "text_plus_eos_tokens": route.text_plus_eos_tokens,
        "exposure_sidecar_sha256": route.exposure_sidecar_sha256,
    }
    if shard == "enigma":
        record["theorem_variant_group"] = route.theorem_variant_group
        record["variant_group_eval_propagated"] = (
            route.variant_group_eval_propagated
        )
    return record


def route_plan_root(row_routes: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """Hash the complete ordered route plan for all four native shards."""

    ordered = {shard: list(row_routes.get(shard, ())) for shard in SHARD_ORDER}
    return _json_sha256(ordered)


def mizar_thproofs_duplicate_route_root(
    row_routes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    """Hash every exact thproof disposition superseded by direct Mizar."""

    duplicates = [
        {"shard": "thproofs", **dict(route)}
        for route in row_routes.get("thproofs", ())
        if route.get("disposition") == "drop"
        and route.get("drop_reason") == "direct_mizar_trajectory_duplicate"
    ]
    return _json_sha256(duplicates)


def enigma_variant_grouping_summary(
    row_routes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize and seal theorem-variant routing from authoritative routes."""

    routes = list(row_routes.get("enigma", ()))
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for route in routes:
        group = route.get("theorem_variant_group")
        if not isinstance(group, str) or not group:
            raise HoldoutError("ENIGMA route lacks normalized theorem-variant group")
        groups[group].append(route)
    for group, members in groups.items():
        eligible = {
            route.get("disposition")
            for route in members
            if route.get("disposition") != "drop"
        }
        if eligible == {"train", "eval"}:
            raise HoldoutError(f"ENIGMA theorem variants split across train/eval: {group}")
    route_records = [
        {
            "line_number": route.get("line_number"),
            "row_id": route.get("row_id"),
            "theorem_variant_group": route.get("theorem_variant_group"),
            "disposition": route.get("disposition"),
            "variant_group_eval_propagated": route.get(
                "variant_group_eval_propagated"
            ),
        }
        for route in routes
    ]
    return {
        "policy": ENIGMA_VARIANT_GROUPING_POLICY,
        "policy_sha256": _json_sha256(ENIGMA_VARIANT_GROUPING_DESCRIPTION),
        "route_together": True,
        "groups_total": len(groups),
        "groups_with_multiple_variants": sum(
            len(members) > 1 for members in groups.values()
        ),
        "groups_routed_eval": sum(
            any(route.get("disposition") == "eval" for route in members)
            for members in groups.values()
        ),
        "rows_promoted_to_eval": sum(
            route.get("variant_group_eval_propagated") is True for route in routes
        ),
        "route_root_sha256": _json_sha256(route_records),
    }


def artifact_inventory_root(
    inventory: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the ordered publication inventory bound by the manifest."""

    return _json_sha256(list(inventory))


def _artifact_inventory_path_error(inventory: Any) -> str | None:
    if not isinstance(inventory, list):
        return "manifest artifact inventory is missing"
    paths: list[str] = []
    for item in inventory:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            return "manifest artifact inventory has a malformed path"
        path = item["path"]
        normalized = PurePosixPath(path)
        if (
            not path
            or normalized.is_absolute()
            or normalized.as_posix() != path
            or "\\" in path
            or any(part in {".", ".."} for part in normalized.parts)
        ):
            return f"manifest artifact path is not normalized: {path!r}"
        paths.append(path)
    if len(paths) != len(set(paths)):
        return "manifest artifact inventory contains duplicate paths"
    if paths != sorted(paths):
        return "manifest artifact inventory is not in canonical path order"
    return None


def _require_canonical_artifact_inventory(inventory: Any) -> None:
    error = _artifact_inventory_path_error(inventory)
    if error is not None:
        raise HoldoutError(error)


def _native_artifact_schema(shard: str) -> str:
    if shard in {"prf2", "enigma"}:
        return "atp-v2-jsonl"
    return "mizar-native-jsonl-v1"


def _planned_artifact_inventory(
    sources: Mapping[str, ShardSource],
    rows: Mapping[str, Sequence[RowPlan]],
) -> list[dict[str, Any]]:
    output_directories = {
        "train": "shards",
        "eval": "eval",
        "drop": "dropped",
    }
    states: dict[str, dict[str, Any]] = {}
    for shard in SHARD_ORDER:
        for directory in output_directories.values():
            path = f"{directory}/{shard}.jsonl"
            states[path] = {
                "digest": hashlib.sha256(),
                "bytes": 0,
                "rows": 0,
                "schema": _native_artifact_schema(shard),
            }
    for path, schema in (
        ("sidecars/eval_exposure.jsonl", "mml-eval-exposure-sidecar-v2"),
        ("sidecars/drop_reasons.jsonl", "mml-drop-reasons-sidecar-v1"),
    ):
        states[path] = {
            "digest": hashlib.sha256(),
            "bytes": 0,
            "rows": 0,
            "schema": schema,
        }

    def update(path: str, payload: bytes) -> None:
        state = states[path]
        state["digest"].update(payload)
        state["bytes"] += len(payload)
        state["rows"] += 1

    for shard in SHARD_ORDER:
        routes = rows[shard]
        seen = 0
        for line_number, native_line in enumerate(sources[shard].iter_lines(), 1):
            seen += 1
            if line_number > len(routes):
                raise HoldoutError(f"{shard}: source grew while sealing inventory")
            route = routes[line_number - 1]
            if (
                route.line_number != line_number
                or hashlib.sha256(native_line).hexdigest() != route.native_row_sha256
            ):
                raise HoldoutError(
                    f"{shard}:{line_number}: source changed while sealing inventory"
                )
            update(
                f"{output_directories[route.disposition]}/{shard}.jsonl",
                native_line,
            )
            if route.disposition == "eval":
                update(
                    "sidecars/eval_exposure.jsonl",
                    _canonical_json_bytes(_exposure_sidecar_record(shard, route))
                    + b"\n",
                )
            elif route.disposition == "drop":
                update(
                    "sidecars/drop_reasons.jsonl",
                    _canonical_json_bytes(_drop_sidecar_record(shard, route)) + b"\n",
                )
        if seen != len(routes):
            raise HoldoutError(f"{shard}: source shrank while sealing inventory")

    inventory = [
        {
            "path": path,
            "sha256": state["digest"].hexdigest(),
            "bytes": state["bytes"],
            "rows": state["rows"],
            "schema": state["schema"],
            "hash_binding": "exact-bytes",
        }
        for path, state in sorted(states.items())
    ]
    inventory.extend(
        [
            {
                "path": "heldout/mml.json",
                "schema": MANIFEST_SCHEMA_VERSION,
                "rows": 1,
                "hash_binding": "manifest-semantic-root",
                "sha256": "$manifest_root_sha256",
                "bytes": "$canonical_pretty_json",
            },
            {
                "path": "heldout/mizar.json",
                "schema": COMPATIBILITY_SCHEMA_VERSION,
                "rows": 1,
                "hash_binding": "derived-root-linked-projection",
                "sha256": "$derived_projection_sha256",
                "bytes": "$canonical_pretty_json",
            },
            {
                "path": "heldout/atp.json",
                "schema": COMPATIBILITY_SCHEMA_VERSION,
                "rows": 1,
                "hash_binding": "derived-root-linked-projection",
                "sha256": "$derived_projection_sha256",
                "bytes": "$canonical_pretty_json",
            },
        ]
    )
    return sorted(inventory, key=lambda record: record["path"])


def _route_rows(
    sources: Mapping[str, ShardSource],
    *,
    tokenizer: TokenizerSeam,
    token_cache: dict[tuple[str, int], int],
    unstable_native: frozenset[tuple[str, str]],
    unstable_classes: frozenset[tuple[str, str]],
    exposure_index: ExposureIndex,
) -> tuple[
    dict[str, tuple[RowPlan, ...]],
    dict[str, Any],
    Counter[str],
    dict[str, Any],
]:
    rows: dict[str, list[RowPlan]] = {shard: [] for shard in SHARD_ORDER}
    by_shard = {shard: _empty_projection_counts() for shard in SHARD_ORDER}
    drop_reasons: Counter[str] = Counter()
    dedup_stats: dict[str, Any] = {}
    enigma_eval_classes: dict[str, set[str]] = defaultdict(set)
    for prepared in _iter_prepared_rows(
        sources,
        tokenizer=tokenizer,
        token_cache=token_cache,
        dedup_stats=dedup_stats,
    ):
        drop_reason = prepared.pre_drop_reason
        exposure = None
        if drop_reason is None:
            drop_reason = _row_disagreement_reason(
                prepared.record,
                shard=prepared.shard,
                unstable_native=unstable_native,
                unstable_classes=unstable_classes,
            )
        if drop_reason is not None:
            disposition = "drop"
            drop_reasons[drop_reason] += 1
        else:
            exposure = classify_exposure(
                prepared.record,
                shard=prepared.shard,
                index=exposure_index,
            )
            disposition = "eval" if exposure.should_eval else "train"
        theorem_variant_group = None
        if prepared.shard == "enigma":
            _, theorem_variant_group = _normalize_atp_theorem_name(
                str(prepared.record["theorem"])
            )
            if disposition == "eval" and exposure is not None:
                for path in (
                    exposure.direct_native_citation,
                    exposure.mapped_cross_representation,
                    exposure.own_theorem,
                    exposure.statement_alias,
                    exposure.visible_target,
                ):
                    enigma_eval_classes[theorem_variant_group].update(
                        hit.class_id for hit in path
                    )
        provisional = RowPlan(
            line_number=prepared.line_number,
            row_id=prepared.record["id"],
            disposition=disposition,
            text_plus_eos_tokens=prepared.text_plus_eos_tokens,
            native_row_sha256=prepared.native_row_sha256,
            exposure_sidecar_sha256=None,
            drop_reason=drop_reason,
            exposure=exposure,
            theorem_variant_group=theorem_variant_group,
        )
        exposure_digest = None
        if disposition == "eval":
            exposure_digest = _json_line_sha256(
                _exposure_sidecar_record(prepared.shard, provisional)
            )
        rows[prepared.shard].append(
            RowPlan(
                line_number=prepared.line_number,
                row_id=prepared.record["id"],
                disposition=disposition,
                text_plus_eos_tokens=prepared.text_plus_eos_tokens,
                native_row_sha256=prepared.native_row_sha256,
                exposure_sidecar_sha256=exposure_digest,
                drop_reason=drop_reason,
                exposure=exposure,
                theorem_variant_group=theorem_variant_group,
            )
        )
        projected = by_shard[prepared.shard][disposition]
        projected["rows"] += 1
        projected["text_plus_eos_tokens"] += prepared.text_plus_eos_tokens

    for index, route in enumerate(rows["enigma"]):
        group = route.theorem_variant_group
        selected_classes = enigma_eval_classes.get(group or "")
        if route.disposition != "train" or not selected_classes:
            continue
        exposure = HeldoutExposure(
            theorem_variant_group=tuple(
                ExposureHit(
                    class_id=class_id,
                    field="enigma_theorem_variant_group",
                    native_name=str(group),
                )
                for class_id in sorted(selected_classes)
            )
        )
        provisional = RowPlan(
            line_number=route.line_number,
            row_id=route.row_id,
            disposition="eval",
            text_plus_eos_tokens=route.text_plus_eos_tokens,
            native_row_sha256=route.native_row_sha256,
            exposure_sidecar_sha256=None,
            exposure=exposure,
            theorem_variant_group=group,
            variant_group_eval_propagated=True,
        )
        rows["enigma"][index] = RowPlan(
            **{
                **provisional.__dict__,
                "exposure_sidecar_sha256": _json_line_sha256(
                    _exposure_sidecar_record("enigma", provisional)
                ),
            }
        )
        by_shard["enigma"]["train"]["rows"] -= 1
        by_shard["enigma"]["train"]["text_plus_eos_tokens"] -= (
            route.text_plus_eos_tokens
        )
        by_shard["enigma"]["eval"]["rows"] += 1
        by_shard["enigma"]["eval"]["text_plus_eos_tokens"] += (
            route.text_plus_eos_tokens
        )

    totals = _empty_projection_counts()
    for shard in SHARD_ORDER:
        for disposition in totals:
            totals[disposition]["rows"] += by_shard[shard][disposition]["rows"]
            totals[disposition]["text_plus_eos_tokens"] += by_shard[shard][disposition][
                "text_plus_eos_tokens"
            ]
    projections = {
        "native_shard_order": list(SHARD_ORDER),
        "by_shard": by_shard,
        "totals": totals,
    }
    return (
        {shard: tuple(shard_rows) for shard, shard_rows in rows.items()},
        projections,
        drop_reasons,
        dedup_stats,
    )


def _class_route_summaries(
    rows: Mapping[str, Sequence[RowPlan]],
    selected: Sequence[str],
) -> dict[str, dict[str, Any]]:
    route_keys: dict[str, list[dict[str, Any]]] = {
        class_id: [] for class_id in selected
    }
    totals = {class_id: {"train": 0, "eval": 0, "drop": 0} for class_id in selected}
    for shard in SHARD_ORDER:
        for route in rows[shard]:
            exposure = route.exposure
            if exposure is None:
                continue
            class_ids = {
                hit.class_id
                for path in (
                    exposure.direct_native_citation,
                    exposure.mapped_cross_representation,
                    exposure.own_theorem,
                    exposure.statement_alias,
                    exposure.visible_target,
                    exposure.theorem_variant_group,
                )
                for hit in path
            }
            for class_id in sorted(class_ids):
                if class_id not in route_keys:
                    continue
                totals[class_id][route.disposition] += 1
                route_keys[class_id].append(
                    {
                        "shard": shard,
                        **_route_record(shard, route),
                    }
                )
    return {
        class_id: {
            "route_totals": totals[class_id],
            "route_root_sha256": _json_sha256(route_keys[class_id]),
        }
        for class_id in selected
    }


def _manifest_root(manifest_without_root: Mapping[str, Any]) -> str:
    return _json_sha256(manifest_without_root)


def _has_exact_contract_tuple(manifest: Mapping[str, Any]) -> bool:
    try:
        pins = current_policy_pins()
        source_policy = manifest["source_identity_policy"]
        expected = canonical_contract_tuple(source_policy)
        return (
            manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
            and manifest["loader_contract"]["schema_version"]
            == LOADER_CONTRACT_SCHEMA_VERSION
            and manifest["policy_version"] == POLICY_VERSION
            and manifest["policy_sha256"] == pins.policy_sha256
            and manifest["mapping_version"] == MAPPING_VERSION
            and manifest["mapping_sha256"] == pins.mapping_sha256
            and manifest["statement_hash_version"] == STATEMENT_HASH_VERSION
            and manifest["atp_deduplication"]["policy"] == ATP_DEDUPLICATION_POLICY
            and manifest["atp_deduplication"]["policy_sha256"]
            == pins.atp_deduplication_sha256
            and manifest["mizar_thproofs_deduplication"]["policy"]
            == MIZAR_THPROOFS_DEDUPLICATION_POLICY
            and manifest["mizar_thproofs_deduplication"]["policy_sha256"]
            == _json_sha256(MIZAR_THPROOFS_DEDUPLICATION_DESCRIPTION)
            and manifest["enigma_variant_grouping"]["policy"]
            == ENIGMA_VARIANT_GROUPING_POLICY
            and manifest["enigma_variant_grouping"]["policy_sha256"]
            == _json_sha256(ENIGMA_VARIANT_GROUPING_DESCRIPTION)
            and manifest["statement_canonicalization"] == CANONICALIZATION_POLICY
            and manifest["contract_tuple"] == expected
            and manifest["contract_tuple_sha256"] == _json_sha256(expected)
        )
    except (HoldoutError, KeyError, TypeError):
        return False


def verify_manifest_root(manifest: Mapping[str, Any]) -> bool:
    """Verify the deterministic root over the authoritative manifest body."""

    root = manifest.get("manifest_root_sha256")
    if not isinstance(root, str):
        return False
    body = dict(manifest)
    body.pop("manifest_root_sha256", None)
    if root != _manifest_root(body):
        return False
    if not _has_exact_contract_tuple(manifest):
        return False
    try:
        if manifest["route_plan_root_sha256"] != route_plan_root(
            manifest["row_routes"]
        ):
            return False
        if manifest["source_root_sha256"] != source_root(manifest["ordered_inputs"]):
            return False
        if manifest["tokenizer_root_sha256"] != _json_sha256(
            manifest["tokenizer_seal"]
        ):
            return False
        if manifest["loader_contract"]["schema_version"] != (
            LOADER_CONTRACT_SCHEMA_VERSION
        ):
            return False
        if _artifact_inventory_path_error(manifest["artifact_inventory"]) is not None:
            return False
        if manifest["artifact_inventory_root_sha256"] != artifact_inventory_root(
            manifest["artifact_inventory"]
        ):
            return False
        ordered_inputs = manifest["ordered_inputs"]
        if manifest["quality_filter_root_sha256"] != _json_sha256(
            [
                {
                    "shard": record["shard"],
                    "quality_filter_root_sha256": record["quality_filter_root_sha256"],
                }
                for record in ordered_inputs
            ]
        ):
            return False
        if manifest["schema_generation_root_sha256"] != _json_sha256(
            [
                {
                    "shard": record["shard"],
                    "schema_generation_root_sha256": record[
                        "schema_generation_root_sha256"
                    ],
                }
                for record in ordered_inputs
            ]
        ):
            return False
        if manifest["deduplication_root_sha256"] != _json_sha256(
            [
                {
                    "shard": record["shard"],
                    "deduplication_root_sha256": record[
                        "deduplication_root_sha256"
                    ],
                }
                for record in ordered_inputs
            ]
        ):
            return False
        if manifest["acceptance_root_sha256"] != _json_sha256(
            [
                {
                    "shard": record["shard"],
                    "acceptance_roots": record["acceptance_roots"],
                }
                for record in ordered_inputs
            ]
        ):
            return False
        mizar_thproofs = manifest["mizar_thproofs_deduplication"]
        if mizar_thproofs["duplicate_route_root_sha256"] != (
            mizar_thproofs_duplicate_route_root(manifest["row_routes"])
        ):
            return False
        duplicate_count = sum(
            1
            for route in manifest["row_routes"]["thproofs"]
            if route.get("disposition") == "drop"
            and route.get("drop_reason") == "direct_mizar_trajectory_duplicate"
        )
        direct_mizar_count = sum(
            1
            for route in manifest["row_routes"]["mizar"]
            if route.get("text_plus_eos_tokens", MAX_TEXT_PLUS_EOS_TOKENS + 1)
            <= MAX_TEXT_PLUS_EOS_TOKENS
        )
        thproofs_count = len(manifest["row_routes"]["thproofs"])
        if (
            mizar_thproofs["priority"] != ["mizar", "thproofs"]
            or mizar_thproofs["direct_mizar_trajectories"]
            != direct_mizar_count
            or mizar_thproofs["thproofs_trajectories"] != thproofs_count
            or mizar_thproofs["thproofs_only_trajectories"]
            != thproofs_count - duplicate_count
            or mizar_thproofs["duplicates_by_shard"]
            != {"mizar": 0, "thproofs": duplicate_count}
            or mizar_thproofs["duplicates_total"] != duplicate_count
        ):
            return False
        if manifest["enigma_variant_grouping"] != enigma_variant_grouping_summary(
            manifest["row_routes"]
        ):
            return False
    except (KeyError, TypeError):
        return False
    return True


MAPPED_CLASS_RE = re.compile(
    r"^mml:v1:(theorem|definition):([A-Z][A-Z0-9_]*):([1-9]\d*)$"
)


def _canonical_projection_alias(class_id: str, family: str) -> str | None:
    match = MAPPED_CLASS_RE.fullmatch(class_id)
    if match is None:
        return None
    kind, article, number = match.groups()
    if family == "mizar":
        infix = "" if kind == "theorem" else "def_"
        return f"{article}:{infix}{number}"
    prefix = "t" if kind == "theorem" else "d"
    return f"{prefix}{number}_{article.lower()}"


def derive_compatibility_projections(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Derive both root-linked legacy projections from one authoritative manifest."""

    root = manifest["manifest_root_sha256"]
    class_records = manifest["class_records"]
    groups = {
        "mizar": ("mizar", "thproofs"),
        "atp": ("prf2", "enigma"),
    }
    projections = {}
    for family, shards in groups.items():
        names: set[str] = set()
        projection_classes = []
        statement_hashes: set[str] = set()
        for record in class_records:
            class_names = {
                member["native_name"]
                for shard, members in record["native_members_by_shard"].items()
                if shard in shards
                for member in members
            }
            canonical_alias = _canonical_projection_alias(
                record["class_id"],
                family,
            )
            if canonical_alias is not None:
                class_names.add(canonical_alias)
            names.update(class_names)
            class_hashes = sorted(
                record["statement_digests_by_representation"].get(family, ())
            )
            statement_hashes.update(class_hashes)
            projection_classes.append(
                {
                    "class_id": record["class_id"],
                    "kind": record["kind"],
                    "native_names": sorted(class_names),
                    "statement_hashes": class_hashes,
                    "route_totals": record["route_totals"],
                    "route_root_sha256": record["route_root_sha256"],
                }
            )
        body = {
            "schema_version": COMPATIBILITY_SCHEMA_VERSION,
            "family": family,
            "facts": sorted(names),
            "shards": list(shards),
            "classes": projection_classes,
            "statement_hashes": sorted(statement_hashes),
            "canonicalization": manifest["statement_canonicalization"][family],
            "atp_parent_occurrence_policy_sha256": _json_sha256(
                manifest["loader_contract"]["atp_parent_occurrence_policy"]
            ),
            "enigma_variant_grouping_policy_sha256": _json_sha256(
                manifest["loader_contract"]["enigma_variant_grouping_policy"]
            ),
            "mapping": {
                "version": manifest["mapping_version"],
                "sha256": manifest["mapping_sha256"],
            },
            "contract_tuple_schema_version": manifest["contract_tuple"][
                "schema_version"
            ],
            "contract_tuple_sha256": manifest["contract_tuple_sha256"],
            "source_root_sha256": manifest["source_root_sha256"],
            "acceptance_root_sha256": manifest["acceptance_root_sha256"],
            "tokenizer_root_sha256": manifest["tokenizer_root_sha256"],
            "route_totals": manifest["partition_projections"]["totals"],
            "route_plan_root_sha256": manifest["route_plan_root_sha256"],
            "derived_from_selected_classes": len(class_records),
            "authoritative_manifest_root_sha256": root,
        }
        projections[family] = {
            **body,
            "projection_root_sha256": _json_sha256(body),
        }
    return projections


def plan_semantic_holdout(
    sources: Mapping[str, ShardSource],
    *,
    tokenizer: TokenizerSeam,
    policy_pins: PolicyPins,
    source_policy: SourceIdentityPolicy,
) -> PartitionPlan:
    """Build a deterministic all-or-nothing four-shard partition plan."""

    _validate_policy_pins(policy_pins)
    tokenizer_seal = _validated_tokenizer_seal(tokenizer.seal)
    ordered_inputs = _validate_sources(
        sources,
        source_policy=source_policy,
    )
    token_cache: dict[tuple[str, int], int] = {}
    _validate_global_row_ids(
        sources,
        tokenizer=tokenizer,
        token_cache=token_cache,
    )

    unstable_native, unstable_classes = _collect_stability(
        sources,
        tokenizer=tokenizer,
        token_cache=token_cache,
    )
    counts, observations = _count_classes_and_observe(
        sources,
        tokenizer=tokenizer,
        token_cache=token_cache,
        unstable_native=unstable_native,
        unstable_classes=unstable_classes,
    )
    selected, tail_size = draw_tail_classes(counts)
    exposure_index = _build_exposure_index(selected, observations)
    rows, projections, drop_reasons, dedup_stats = _route_rows(
        sources,
        tokenizer=tokenizer,
        token_cache=token_cache,
        unstable_native=unstable_native,
        unstable_classes=unstable_classes,
        exposure_index=exposure_index,
    )
    row_routes = {
        shard: [_route_record(shard, route) for route in rows[shard]]
        for shard in SHARD_ORDER
    }
    dedup_stats["mizar_thproofs"]["duplicate_route_root_sha256"] = (
        mizar_thproofs_duplicate_route_root(row_routes)
    )
    route_root = route_plan_root(row_routes)
    projections["route_plan_root_sha256"] = route_root
    for shard in SHARD_ORDER:
        projections["by_shard"][shard]["route_root_sha256"] = _json_sha256(
            row_routes[shard]
        )
    class_route_summaries = _class_route_summaries(rows, selected)

    class_records = [
        _class_record(
            class_id,
            count=counts[class_id],
            observation=observations[class_id],
            route_summary=class_route_summaries[class_id],
        )
        for class_id in selected
    ]
    atp_deduplication = {
        "policy": ATP_DEDUPLICATION_POLICY,
        "policy_sha256": policy_pins.atp_deduplication_sha256,
        "priority": ["prf2", "enigma"],
        **dedup_stats["atp"],
    }
    mizar_thproofs_deduplication = {
        "policy": MIZAR_THPROOFS_DEDUPLICATION_POLICY,
        "policy_sha256": _json_sha256(
            MIZAR_THPROOFS_DEDUPLICATION_DESCRIPTION
        ),
        "priority": ["mizar", "thproofs"],
        **dedup_stats["mizar_thproofs"],
    }
    enigma_variant_grouping = enigma_variant_grouping_summary(row_routes)
    source_policy_payload = _source_policy_payload(source_policy)
    source_identity_policy = {
        "policy_id": source_policy.policy_id,
        "test_only": source_policy.test_only,
        "injected_test_seams": source_policy.test_only,
        "policy_sha256": _json_sha256(source_policy_payload),
    }
    contract_tuple = canonical_contract_tuple(source_identity_policy)
    source_root_sha256 = source_root(ordered_inputs)
    tokenizer_root_sha256 = _json_sha256(tokenizer_seal)
    quality_filter_root_sha256 = _json_sha256(
        [
            {
                "shard": record["shard"],
                "quality_filter_root_sha256": record["quality_filter_root_sha256"],
            }
            for record in ordered_inputs
        ]
    )
    schema_generation_root_sha256 = _json_sha256(
        [
            {
                "shard": record["shard"],
                "schema_generation_root_sha256": record[
                    "schema_generation_root_sha256"
                ],
            }
            for record in ordered_inputs
        ]
    )
    deduplication_root_sha256 = _json_sha256(
        [
            {
                "shard": record["shard"],
                "deduplication_root_sha256": record[
                    "deduplication_root_sha256"
                ],
            }
            for record in ordered_inputs
        ]
    )
    acceptance_root_sha256 = _json_sha256(
        [
            {
                "shard": record["shard"],
                "acceptance_roots": record["acceptance_roots"],
            }
            for record in ordered_inputs
        ]
    )
    artifact_inventory = _planned_artifact_inventory(sources, rows)
    manifest_body: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "mapping_version": MAPPING_VERSION,
        "statement_hash_version": STATEMENT_HASH_VERSION,
        "policy_sha256": policy_pins.policy_sha256,
        "mapping_sha256": policy_pins.mapping_sha256,
        "seed": SEED,
        "requested_classes": REQUESTED_CLASSES,
        "actual_classes": len(selected),
        "tail_row_citation_counts": [1, 2],
        "tail_classes_available": tail_size,
        "class_records": class_records,
        "ordered_inputs": ordered_inputs,
        "source_identity_policy": source_identity_policy,
        "source_root_sha256": source_root_sha256,
        "quality_filter_root_sha256": quality_filter_root_sha256,
        "schema_generation_root_sha256": schema_generation_root_sha256,
        "deduplication_root_sha256": deduplication_root_sha256,
        "acceptance_root_sha256": acceptance_root_sha256,
        "tokenizer_seal": tokenizer_seal,
        "tokenizer_root_sha256": tokenizer_root_sha256,
        "atp_deduplication": atp_deduplication,
        "mizar_thproofs_deduplication": mizar_thproofs_deduplication,
        "enigma_variant_grouping": enigma_variant_grouping,
        "row_routes": row_routes,
        "route_plan_root_sha256": route_root,
        "partition_projections": projections,
        "drop_reason_counts": dict(sorted(drop_reasons.items())),
        "loader_contract": {
            "schema_version": LOADER_CONTRACT_SCHEMA_VERSION,
            "publication_mode": (
                "test_only" if source_policy.test_only else "production"
            ),
            "exact_directories": [
                "shards",
                "eval",
                "dropped",
                "heldout",
                "sidecars",
            ],
            "inventory_policy": "exact-files-no-extras-v1",
            "native_hash_policy": "exact-line-bytes-sha256-v1",
            "atp_parent_occurrence_policy": ATP_PARENT_OCCURRENCE_POLICY,
            "enigma_variant_grouping_policy": ENIGMA_VARIANT_GROUPING_DESCRIPTION,
            "heldout_hash_policy": (
                "semantic-self-root-and-derived-root-linked-projections-v1"
            ),
        },
        "contract_tuple": contract_tuple,
        "contract_tuple_sha256": _json_sha256(contract_tuple),
        "artifact_inventory": artifact_inventory,
        "artifact_inventory_root_sha256": artifact_inventory_root(artifact_inventory),
        "statement_canonicalization": CANONICALIZATION_POLICY,
        "partition_policy": {
            "eval": [
                "direct native citation",
                "mapped cross-representation exposure",
                "own theorem",
                "same-representation statement alias",
                "structured proof formula or visible target",
                "ENIGMA theorem-variant group propagation",
            ],
            "drop": [
                "direct_mizar_trajectory_duplicate",
                "exact_atp_duplicate",
                "statement_disagreement",
                "class_member_disagreement",
                "overlength",
            ],
            "native_rows_rewritten": False,
            "metadata_location": "manifest and sidecars",
        },
    }
    if len(selected) != REQUESTED_CLASSES:
        raise HoldoutError(
            f"exact draw failed: requested {REQUESTED_CLASSES}, got {len(selected)}"
        )
    root = _manifest_root(manifest_body)
    manifest = {**manifest_body, "manifest_root_sha256": root}
    if not verify_manifest_root(manifest):
        raise HoldoutError("manifest root nondeterminism detected")
    compatibility = derive_compatibility_projections(manifest)
    return PartitionPlan(
        manifest=manifest,
        compatibility_projections=compatibility,
        rows=rows,
        sealed_manifest_root_sha256=root,
        sealed_route_plan_root_sha256=route_root,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_pretty_json_bytes(value))


def write_partition_atomically(
    plan: PartitionPlan,
    *,
    sources: Mapping[str, ShardSource],
    output: str | os.PathLike[str],
) -> None:
    """Publish a complete new partition directory or publish nothing."""

    output_path = Path(output)
    if output_path.exists():
        raise HoldoutError(f"output already exists: {output_path}")
    _require_canonical_artifact_inventory(plan.manifest.get("artifact_inventory"))
    if plan.manifest.get("manifest_root_sha256") != (plan.sealed_manifest_root_sha256):
        raise HoldoutError("sealed manifest root changed before publication")
    if plan.manifest.get("route_plan_root_sha256") != (
        plan.sealed_route_plan_root_sha256
    ):
        raise HoldoutError("sealed route plan root changed before publication")
    if not verify_manifest_root(plan.manifest):
        raise HoldoutError("manifest root hash drift detected before publication")
    if set(plan.rows) != set(SHARD_ORDER):
        raise HoldoutError("route plan shard set changed before publication")
    recomputed_routes = {
        shard: [_route_record(shard, route) for route in plan.rows[shard]]
        for shard in SHARD_ORDER
    }
    if recomputed_routes != plan.manifest.get("row_routes"):
        raise HoldoutError("route plan mutation detected before publication")
    if route_plan_root(recomputed_routes) != plan.manifest.get(
        "route_plan_root_sha256"
    ):
        raise HoldoutError("route plan root mismatch before publication")
    for shard in SHARD_ORDER:
        for route in plan.rows[shard]:
            if route.disposition not in {"train", "eval", "drop"}:
                raise HoldoutError(
                    f"{shard}:{route.line_number}: invalid route disposition"
                )
            if route.disposition == "eval":
                payload = _exposure_sidecar_record(shard, route)
                if _json_line_sha256(payload) != route.exposure_sidecar_sha256:
                    raise HoldoutError(
                        f"{shard}:{route.line_number}: exposure sidecar digest drift"
                    )
            elif route.exposure_sidecar_sha256 is not None:
                raise HoldoutError(
                    f"{shard}:{route.line_number}: non-eval route has exposure digest"
                )
            if route.disposition == "drop" and not route.drop_reason:
                raise HoldoutError(
                    f"{shard}:{route.line_number}: drop route lacks a reason"
                )
            if route.disposition != "drop" and route.drop_reason is not None:
                raise HoldoutError(
                    f"{shard}:{route.line_number}: non-drop route has a drop reason"
                )
    expected_compatibility = derive_compatibility_projections(plan.manifest)
    if plan.compatibility_projections != expected_compatibility:
        raise HoldoutError("compatibility projection drift detected before publication")
    input_records = {
        item["shard"]: item for item in plan.manifest.get("ordered_inputs", [])
    }
    if set(sources) != set(SHARD_ORDER) or set(input_records) != set(SHARD_ORDER):
        raise HoldoutError("plan/source shard set changed after planning")
    for shard in SHARD_ORDER:
        actual, line_count = _source_digest_and_lines(sources[shard])
        expected = input_records[shard]["sha256"]
        expected_rows = input_records[shard]["rows"]
        if actual != expected or line_count != expected_rows:
            raise HoldoutError(
                f"{shard}: source changed after planning; "
                f"expected {expected}/{expected_rows}, got {actual}/{line_count}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.tmp.",
            dir=str(output_path.parent),
        )
    )
    try:
        for directory in ("shards", "eval", "dropped", "heldout", "sidecars"):
            (temporary / directory).mkdir()
        eval_sidecar = (temporary / "sidecars" / "eval_exposure.jsonl").open(
            "w", encoding="utf-8"
        )
        drop_sidecar = (temporary / "sidecars" / "drop_reasons.jsonl").open(
            "w", encoding="utf-8"
        )
        try:
            for shard in SHARD_ORDER:
                routes = plan.rows[shard]
                output_directories = {
                    "train": "shards",
                    "eval": "eval",
                    "drop": "dropped",
                }
                handles = {
                    disposition: (
                        temporary / output_directories[disposition] / f"{shard}.jsonl"
                    ).open("wb")
                    for disposition in ("train", "eval", "drop")
                }
                try:
                    seen_lines = 0
                    for line_number, native_line in enumerate(
                        sources[shard].iter_lines(), 1
                    ):
                        seen_lines += 1
                        if line_number > len(routes):
                            raise HoldoutError(
                                f"{shard}: more rows than the partition plan"
                            )
                        route = routes[line_number - 1]
                        if route.line_number != line_number:
                            raise HoldoutError(
                                f"{shard}: row order changed after planning"
                            )
                        native_hash = hashlib.sha256(native_line).hexdigest()
                        if native_hash != route.native_row_sha256:
                            raise HoldoutError(
                                f"{shard}:{line_number}: native row hash changed"
                            )
                        try:
                            native_record = json.loads(native_line)
                        except (UnicodeDecodeError, json.JSONDecodeError) as error:
                            raise HoldoutError(
                                f"{shard}:{line_number}: native row is no longer JSON"
                            ) from error
                        if native_record.get("id") != route.row_id:
                            raise HoldoutError(
                                f"{shard}:{line_number}: route row id changed"
                            )
                        handles[route.disposition].write(native_line)
                        if route.disposition == "eval":
                            payload = _exposure_sidecar_record(shard, route)
                            if _json_line_sha256(payload) != (
                                route.exposure_sidecar_sha256
                            ):
                                raise HoldoutError(
                                    f"{shard}:{line_number}: "
                                    "exposure sidecar digest drift"
                                )
                            eval_sidecar.write(
                                json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                        elif route.disposition == "drop":
                            drop_sidecar.write(
                                json.dumps(
                                    _drop_sidecar_record(shard, route),
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                    if seen_lines != len(routes):
                        raise HoldoutError(
                            f"{shard}: fewer rows than the partition plan"
                        )
                finally:
                    for handle in handles.values():
                        handle.close()
        finally:
            eval_sidecar.close()
            drop_sidecar.close()

        _write_json(temporary / "heldout" / "mml.json", plan.manifest)
        for family, projection in plan.compatibility_projections.items():
            _write_json(
                temporary / "heldout" / f"{family}.json",
                projection,
            )
        load_holdout_contract(
            temporary,
            production=not plan.manifest["source_identity_policy"]["test_only"],
        )
        os.replace(temporary, output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_authoritative_holdout(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load and verify ``heldout/mml.json`` for shared verifier/evaluator use."""

    requested = Path(path)
    manifest_path = (
        requested / "heldout" / "mml.json" if requested.is_dir() else requested
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HoldoutError(
            f"invalid authoritative holdout manifest: {manifest_path}"
        ) from error
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise HoldoutError(f"authoritative holdout must use {MANIFEST_SCHEMA_VERSION}")
    _require_canonical_artifact_inventory(manifest.get("artifact_inventory"))
    if not _has_exact_contract_tuple(manifest):
        raise HoldoutError("authoritative holdout contract tuple is not exact")
    if not verify_manifest_root(manifest):
        raise HoldoutError("authoritative holdout manifest root is invalid")
    return manifest


def _expected_publication_schemas() -> dict[str, str]:
    schemas = {
        f"{directory}/{shard}.jsonl": _native_artifact_schema(shard)
        for directory in ("shards", "eval", "dropped")
        for shard in SHARD_ORDER
    }
    schemas.update(
        {
            "sidecars/eval_exposure.jsonl": "mml-eval-exposure-sidecar-v2",
            "sidecars/drop_reasons.jsonl": "mml-drop-reasons-sidecar-v1",
            "heldout/mml.json": MANIFEST_SCHEMA_VERSION,
            "heldout/mizar.json": COMPATIBILITY_SCHEMA_VERSION,
            "heldout/atp.json": COMPATIBILITY_SCHEMA_VERSION,
        }
    )
    return schemas


def _validate_publication_mode(
    manifest: Mapping[str, Any],
    *,
    production: bool,
) -> bool:
    policy = manifest.get("source_identity_policy")
    loader = manifest.get("loader_contract")
    if not isinstance(policy, Mapping) or not isinstance(loader, Mapping):
        raise HoldoutError("manifest lacks source or loader policy")
    ordered_inputs = manifest.get("ordered_inputs")
    if not isinstance(ordered_inputs, list) or [
        record.get("shard") for record in ordered_inputs
    ] != list(SHARD_ORDER):
        raise HoldoutError("manifest source inputs are not exact and ordered")
    test_only = policy.get("test_only") is True
    injected = policy.get("injected_test_seams") is True
    marked_test = loader.get("publication_mode") == "test_only"
    if loader.get("atp_parent_occurrence_policy") != ATP_PARENT_OCCURRENCE_POLICY:
        raise HoldoutError("loader ATP parent-occurrence policy is not exact")
    if (
        loader.get("enigma_variant_grouping_policy")
        != ENIGMA_VARIANT_GROUPING_DESCRIPTION
    ):
        raise HoldoutError("loader ENIGMA variant-grouping policy is not exact")
    if production:
        if test_only or injected or marked_test:
            raise HoldoutError(
                "production loader refuses test-only or injected source seams"
            )
        approved = production_source_policy()
        expected_payload = _source_policy_payload(approved)
        if policy.get("policy_id") != approved.policy_id or policy.get(
            "policy_sha256"
        ) != _json_sha256(expected_payload):
            raise HoldoutError("production source identity policy is not approved")
        for record in ordered_inputs:
            approved_shard = approved.shards[record["shard"]]
            approved_deduplication_root = _source_policy_deduplication_roots(
                approved
            )[record["shard"]]
            if (
                record.get("sha256") != approved_shard.input_sha256
                or record.get("source_snapshots")
                != _snapshot_records(approved_shard.source_snapshots)
                or record.get("source_manifest_root_sha256")
                != approved_shard.source_manifest_root_sha256
                or record.get("quality_filter_root_sha256")
                != approved_shard.quality_filter_root_sha256
                or record.get("schema_generation_root_sha256")
                != approved_shard.schema_generation_root_sha256
                or record.get("deduplication_root_sha256")
                != approved_deduplication_root
            ):
                raise HoldoutError(
                    f"{record['shard']}: production source roots are not approved"
                )
            digests = [
                record.get("source_manifest_root_sha256"),
                record.get("quality_filter_root_sha256"),
                record.get("schema_generation_root_sha256"),
                record.get("deduplication_root_sha256"),
                *(
                    snapshot.get("sha256")
                    for snapshot in record.get("source_snapshots", ())
                    if isinstance(snapshot, Mapping)
                ),
            ]
            if any(
                not isinstance(digest, str)
                or not SHA256_RE.fullmatch(digest)
                or len(set(digest)) == 1
                for digest in digests
            ):
                raise HoldoutError(
                    "production loader refuses placeholder or unfinished source roots"
                )
    elif test_only:
        policy_id = str(policy.get("policy_id", "")).lower()
        if (
            not injected
            or not marked_test
            or not any(marker in policy_id for marker in ("test", "synthetic"))
        ):
            raise HoldoutError("test-only source policy is not clearly marked")
    return test_only


def _inventory_records(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    inventory = manifest.get("artifact_inventory")
    _require_canonical_artifact_inventory(inventory)
    assert isinstance(inventory, list)
    if manifest.get("artifact_inventory_root_sha256") != artifact_inventory_root(
        inventory
    ):
        raise HoldoutError("manifest artifact inventory root is invalid")
    expected_schemas = _expected_publication_schemas()
    expected_bindings = {
        path: (
            "exact-bytes"
            if not path.startswith("heldout/")
            else (
                "manifest-semantic-root"
                if path == "heldout/mml.json"
                else "derived-root-linked-projection"
            )
        )
        for path in expected_schemas
    }
    records: dict[str, Mapping[str, Any]] = {}
    for item in inventory:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise HoldoutError("manifest artifact inventory has a malformed record")
        path = item["path"]
        if path in records:
            raise HoldoutError(f"manifest artifact inventory duplicates {path}")
        if path not in expected_schemas or item.get("schema") != expected_schemas[path]:
            raise HoldoutError(f"manifest artifact schema mismatch for {path}")
        if item.get("hash_binding") != expected_bindings[path]:
            raise HoldoutError(f"manifest artifact hash binding mismatch for {path}")
        if expected_bindings[path] == "exact-bytes":
            if (
                not isinstance(item.get("sha256"), str)
                or not SHA256_RE.fullmatch(item["sha256"])
                or isinstance(item.get("bytes"), bool)
                or not isinstance(item.get("bytes"), int)
                or item["bytes"] < 0
                or isinstance(item.get("rows"), bool)
                or not isinstance(item.get("rows"), int)
                or item["rows"] < 0
            ):
                raise HoldoutError(f"manifest artifact metrics are invalid for {path}")
        elif (
            item.get("rows") != 1
            or item.get("bytes") != "$canonical_pretty_json"
            or (
                path == "heldout/mml.json"
                and item.get("sha256") != "$manifest_root_sha256"
            )
            or (
                path != "heldout/mml.json"
                and item.get("sha256") != "$derived_projection_sha256"
            )
        ):
            raise HoldoutError(
                f"manifest derived artifact binding is invalid for {path}"
            )
        records[path] = item
    if set(records) != set(expected_schemas):
        raise HoldoutError("manifest artifact inventory is incomplete")
    return records


def _actual_publication_paths(root: Path) -> set[str]:
    required_directories = {"shards", "eval", "dropped", "heldout", "sidecars"}
    if not root.is_dir() or root.is_symlink():
        raise HoldoutError(f"publication root is not a directory: {root}")
    top_level = {entry.name for entry in root.iterdir()}
    if top_level != required_directories:
        raise HoldoutError(
            "publication inventory top-level mismatch: "
            f"missing={sorted(required_directories - top_level)}, "
            f"extra={sorted(top_level - required_directories)}"
        )
    paths: set[str] = set()
    for directory in sorted(required_directories):
        parent = root / directory
        if not parent.is_dir() or parent.is_symlink():
            raise HoldoutError(
                f"publication inventory directory is invalid: {directory}"
            )
        for entry in parent.iterdir():
            if not entry.is_file() or entry.is_symlink():
                raise HoldoutError(
                    f"publication inventory contains a non-file: "
                    f"{entry.relative_to(root)}"
                )
            paths.add(entry.relative_to(root).as_posix())
    return paths


def _read_jsonl_objects(
    data: bytes,
    *,
    path: str,
) -> tuple[list[bytes], list[dict[str, Any]]]:
    lines = data.splitlines(keepends=True)
    objects = []
    for line_number, line in enumerate(lines, 1):
        if not line.endswith(b"\n"):
            raise HoldoutError(f"{path}:{line_number}: JSONL row lacks final newline")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HoldoutError(f"{path}:{line_number}: invalid JSONL") from error
        if not isinstance(value, dict):
            raise HoldoutError(f"{path}:{line_number}: JSONL row is not an object")
        objects.append(value)
    return lines, objects


def _validate_exact_artifact(
    root: Path,
    relative_path: str,
    record: Mapping[str, Any],
) -> tuple[bytes, PublishedArtifact]:
    path = root / relative_path
    data = path.read_bytes()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if record.get("hash_binding") == "exact-bytes":
        if actual_sha256 != record.get("sha256"):
            raise HoldoutError(
                f"artifact SHA-256 mismatch for {relative_path}: "
                f"expected {record.get('sha256')}, got {actual_sha256}"
            )
        if len(data) != record.get("bytes"):
            raise HoldoutError(f"artifact byte count mismatch for {relative_path}")
        rows = len(data.splitlines())
        if rows != record.get("rows"):
            raise HoldoutError(f"artifact row count mismatch for {relative_path}")
    else:
        rows = 1
    return data, PublishedArtifact(
        path=path,
        sha256=actual_sha256,
        bytes=len(data),
        rows=rows,
        schema=str(record["schema"]),
    )


def _validate_route_totals(manifest: Mapping[str, Any]) -> None:
    by_shard = {shard: _empty_projection_counts() for shard in SHARD_ORDER}
    drop_reasons: Counter[str] = Counter()
    for shard in SHARD_ORDER:
        routes = manifest["row_routes"][shard]
        if [route.get("line_number") for route in routes] != list(
            range(1, len(routes) + 1)
        ):
            raise HoldoutError(
                f"{shard}: route source lines are not complete and ordered"
            )
        for route in routes:
            disposition = route.get("disposition")
            if disposition not in {"train", "eval", "drop"}:
                raise HoldoutError(f"{shard}: route has invalid disposition")
            tokens = route.get("text_plus_eos_tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
                raise HoldoutError(f"{shard}: route has invalid token count")
            by_shard[shard][disposition]["rows"] += 1
            by_shard[shard][disposition]["text_plus_eos_tokens"] += tokens
            if disposition == "drop":
                reason = route.get("drop_reason")
                if not isinstance(reason, str) or not reason:
                    raise HoldoutError(f"{shard}: dropped route lacks a reason")
                drop_reasons[reason] += 1
            elif route.get("drop_reason") is not None:
                raise HoldoutError(f"{shard}: non-drop route has a drop reason")
        projected = manifest["partition_projections"]["by_shard"][shard]
        for disposition in ("train", "eval", "drop"):
            if projected.get(disposition) != by_shard[shard][disposition]:
                raise HoldoutError(f"{shard}: partition totals do not match routes")
        if projected.get("route_root_sha256") != _json_sha256(routes):
            raise HoldoutError(f"{shard}: projected route root is invalid")
    totals = _empty_projection_counts()
    for shard in SHARD_ORDER:
        for disposition in totals:
            totals[disposition]["rows"] += by_shard[shard][disposition]["rows"]
            totals[disposition]["text_plus_eos_tokens"] += by_shard[shard][disposition][
                "text_plus_eos_tokens"
            ]
    if manifest["partition_projections"].get("totals") != totals:
        raise HoldoutError("partition totals do not reconcile with row routes")
    if manifest["partition_projections"].get("route_plan_root_sha256") != manifest.get(
        "route_plan_root_sha256"
    ):
        raise HoldoutError("partition route root does not match authoritative routes")
    if manifest.get("drop_reason_counts") != dict(sorted(drop_reasons.items())):
        raise HoldoutError("drop reason totals do not reconcile with row routes")


def _validate_native_route_files(
    manifest: Mapping[str, Any],
    data_by_path: Mapping[str, bytes],
) -> None:
    directory_by_disposition = {
        "train": "shards",
        "eval": "eval",
        "drop": "dropped",
    }
    seen_ids: set[str] = set()
    input_rows = {
        record["shard"]: record["rows"] for record in manifest["ordered_inputs"]
    }
    for shard in SHARD_ORDER:
        routes = manifest["row_routes"][shard]
        if input_rows.get(shard) != len(routes):
            raise HoldoutError(f"{shard}: source row total does not match routes")
        for disposition, directory in directory_by_disposition.items():
            path = f"{directory}/{shard}.jsonl"
            lines, objects = _read_jsonl_objects(data_by_path[path], path=path)
            expected_routes = [
                route for route in routes if route["disposition"] == disposition
            ]
            if len(lines) != len(expected_routes):
                raise HoldoutError(f"{path}: rows do not match route totals")
            for native_line, record, route in zip(
                lines,
                objects,
                expected_routes,
                strict=True,
            ):
                native_sha256 = hashlib.sha256(native_line).hexdigest()
                if native_sha256 != route.get("native_row_sha256"):
                    raise HoldoutError(f"{path}: native row bytes do not match route")
                if record.get("id") != route.get("row_id"):
                    raise HoldoutError(f"{path}: native row id does not match route")
                if record["id"] in seen_ids:
                    raise HoldoutError(f"{path}: duplicate published row id")
                seen_ids.add(record["id"])
                _validate_row_schema(
                    record,
                    shard=shard,
                    line_number=route["line_number"],
                )


def _validate_sidecar_files(
    manifest: Mapping[str, Any],
    data_by_path: Mapping[str, bytes],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    eval_routes = [
        (shard, route)
        for shard in SHARD_ORDER
        for route in manifest["row_routes"][shard]
        if route["disposition"] == "eval"
    ]
    eval_lines, eval_objects = _read_jsonl_objects(
        data_by_path["sidecars/eval_exposure.jsonl"],
        path="sidecars/eval_exposure.jsonl",
    )
    if len(eval_lines) != len(eval_routes):
        raise HoldoutError("eval exposure sidecar count does not match eval routes")
    exposure_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for line, payload, (shard, route) in zip(
        eval_lines,
        eval_objects,
        eval_routes,
        strict=True,
    ):
        expected_identity = {
            "shard": shard,
            "line_number": route["line_number"],
            "row_id": route["row_id"],
            "native_row_sha256": route["native_row_sha256"],
        }
        if any(payload.get(key) != value for key, value in expected_identity.items()):
            raise HoldoutError("eval exposure sidecar ID does not match route")
        if hashlib.sha256(line).hexdigest() != route.get("exposure_sidecar_sha256"):
            raise HoldoutError("eval exposure sidecar digest does not match route")
        paths = payload.get("paths")
        if not isinstance(paths, dict) or not any(paths.values()):
            raise HoldoutError("eval exposure sidecar lacks exposure paths")
        key = (shard, route["row_id"])
        if key in exposure_index:
            raise HoldoutError("eval exposure sidecar contains a duplicate ID")
        exposure_index[key] = payload

    drop_routes = [
        (shard, route)
        for shard in SHARD_ORDER
        for route in manifest["row_routes"][shard]
        if route["disposition"] == "drop"
    ]
    drop_lines, drop_objects = _read_jsonl_objects(
        data_by_path["sidecars/drop_reasons.jsonl"],
        path="sidecars/drop_reasons.jsonl",
    )
    if len(drop_lines) != len(drop_routes):
        raise HoldoutError("drop sidecar count does not match dropped routes")
    for line, payload, (shard, route) in zip(
        drop_lines,
        drop_objects,
        drop_routes,
        strict=True,
    ):
        expected = {
            "shard": shard,
            "line_number": route["line_number"],
            "row_id": route["row_id"],
            "native_row_sha256": route["native_row_sha256"],
            "reason": route["drop_reason"],
        }
        if payload != expected or line != _canonical_json_bytes(expected) + b"\n":
            raise HoldoutError("drop sidecar row does not match dropped route")
    return exposure_index


def load_holdout_contract(
    root: str | os.PathLike[str],
    *,
    production: bool = True,
) -> ValidatedHoldoutContract:
    """Verify every published artifact and return the shared typed contract."""

    root_path = Path(root)
    manifest = load_authoritative_holdout(root_path)
    test_only = _validate_publication_mode(manifest, production=production)
    inventory = _inventory_records(manifest)
    actual_paths = _actual_publication_paths(root_path)
    if actual_paths != set(inventory):
        raise HoldoutError(
            "publication inventory file mismatch: "
            f"missing={sorted(set(inventory) - actual_paths)}, "
            f"extra={sorted(actual_paths - set(inventory))}"
        )

    data_by_path: dict[str, bytes] = {}
    artifacts: dict[str, PublishedArtifact] = {}
    for relative_path, record in inventory.items():
        data, artifact = _validate_exact_artifact(
            root_path,
            relative_path,
            record,
        )
        data_by_path[relative_path] = data
        artifacts[relative_path] = artifact
    if data_by_path["heldout/mml.json"] != _pretty_json_bytes(manifest):
        raise HoldoutError("authoritative manifest bytes are not canonical")

    expected_projections = derive_compatibility_projections(manifest)
    projections = {}
    for family in ("mizar", "atp"):
        path = f"heldout/{family}.json"
        try:
            projection = json.loads(data_by_path[path])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HoldoutError(f"invalid {family} holdout projection") from error
        if projection != expected_projections[family] or data_by_path[
            path
        ] != _pretty_json_bytes(expected_projections[family]):
            raise HoldoutError(
                f"{family} compatibility projection is stale or not root-linked"
            )
        projections[family] = projection

    _validate_route_totals(manifest)
    _validate_native_route_files(manifest, data_by_path)
    exposure_index = _validate_sidecar_files(manifest, data_by_path)
    ordered_inputs = {record["shard"]: record for record in manifest["ordered_inputs"]}
    return ValidatedHoldoutContract(
        root=root_path,
        production=production,
        test_only=test_only,
        authoritative_root=manifest["manifest_root_sha256"],
        manifest=manifest,
        projections=projections,
        artifacts=artifacts,
        family_paths={
            shard: FamilyPaths(
                train=root_path / "shards" / f"{shard}.jsonl",
                eval=root_path / "eval" / f"{shard}.jsonl",
                dropped=root_path / "dropped" / f"{shard}.jsonl",
            )
            for shard in SHARD_ORDER
        },
        exposure_index=exposure_index,
        tokenizer_root_sha256=manifest["tokenizer_root_sha256"],
        source_root_sha256=manifest["source_root_sha256"],
        quality_filter_roots_by_shard={
            shard: ordered_inputs[shard]["quality_filter_root_sha256"]
            for shard in SHARD_ORDER
        },
        schema_generation_roots_by_shard={
            shard: ordered_inputs[shard]["schema_generation_root_sha256"]
            for shard in SHARD_ORDER
        },
        deduplication_roots_by_shard={
            shard: ordered_inputs[shard]["deduplication_root_sha256"]
            for shard in SHARD_ORDER
        },
        acceptance_roots_by_shard={
            shard: ordered_inputs[shard]["acceptance_roots"]
            for shard in SHARD_ORDER
        },
    )


def refresh_compatibility_projections(
    root: str | os.PathLike[str],
) -> None:
    """Safely replace stale projections from the authoritative manifest only."""

    root_path = Path(root)
    manifest = load_authoritative_holdout(root_path)
    projections = derive_compatibility_projections(manifest)
    heldout = root_path / "heldout"
    if not heldout.is_dir():
        raise HoldoutError(f"heldout directory is missing: {heldout}")
    originals: dict[Path, bytes | None] = {}
    temporary: dict[Path, Path] = {}
    for family in ("mizar", "atp"):
        destination = heldout / f"{family}.json"
        originals[destination] = (
            destination.read_bytes() if destination.exists() else None
        )
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{family}.json.",
            dir=str(heldout),
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        _write_json(temp_path, projections[family])
        temporary[destination] = temp_path
    replaced: list[Path] = []
    try:
        for destination in (heldout / "mizar.json", heldout / "atp.json"):
            os.replace(temporary[destination], destination)
            replaced.append(destination)
        load_holdout_contract(
            root_path,
            production=not manifest["source_identity_policy"]["test_only"],
        )
    except Exception:
        for destination in replaced:
            original = originals[destination]
            if original is None:
                destination.unlink(missing_ok=True)
            else:
                descriptor, rollback_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.rollback.",
                    dir=str(heldout),
                )
                with os.fdopen(descriptor, "wb") as rollback:
                    rollback.write(original)
                os.replace(rollback_name, destination)
        raise
    finally:
        for temp_path in temporary.values():
            temp_path.unlink(missing_ok=True)


def _load_provenance(
    path: Path,
) -> tuple[dict[str, Any], PolicyPins, dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HoldoutError(f"invalid provenance manifest: {path}") from error
    if document.get("schema_version") != "mml-semantic-holdout-inputs-v1":
        raise HoldoutError(
            "provenance manifest must use mml-semantic-holdout-inputs-v1"
        )
    pins_raw = document.get("policy_pins")
    if not isinstance(pins_raw, dict):
        raise HoldoutError("provenance manifest lacks policy_pins")
    try:
        pins = PolicyPins(
            policy_sha256=pins_raw["policy_sha256"],
            mapping_sha256=pins_raw["mapping_sha256"],
            atp_deduplication_sha256=pins_raw["atp_deduplication_sha256"],
        )
    except KeyError as error:
        raise HoldoutError("provenance manifest has incomplete policy_pins") from error
    shards = document.get("shards")
    if not isinstance(shards, dict):
        raise HoldoutError("provenance manifest lacks shard records")
    tokenizer_seal = document.get("tokenizer_seal")
    if not isinstance(tokenizer_seal, dict):
        raise HoldoutError("provenance manifest lacks tokenizer_seal")
    return shards, pins, tokenizer_seal


def _path_sources(
    raw_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, PathShardSource]:
    sources = {}
    for shard in SHARD_ORDER:
        record = provenance.get(shard)
        if not isinstance(record, dict):
            raise HoldoutError(f"provenance manifest lacks shard {shard}")
        snapshots_raw = record.get("source_snapshots")
        if not isinstance(snapshots_raw, list):
            raise HoldoutError(f"{shard}: source_snapshots are missing")
        snapshots = []
        for item in snapshots_raw:
            if not isinstance(item, dict):
                raise HoldoutError(f"{shard}: malformed source snapshot")
            snapshots.append(
                SourceSnapshot(
                    reference=str(item.get("reference", "")),
                    sha256=str(item.get("sha256", "")),
                )
            )
        sources[shard] = PathShardSource(
            name=shard,
            logical_path=f"raw/{shard}.jsonl",
            path=raw_dir / f"{shard}.jsonl",
            expected_input_sha256=str(record.get("input_sha256", "")),
            source_snapshots=tuple(snapshots),
            source_manifest_root_sha256=str(
                record.get("source_manifest_root_sha256", "")
            ),
            quality_filter_root_sha256=str(
                record.get("quality_filter_root_sha256", "")
            ),
            schema_generation_root_sha256=str(
                record.get("schema_generation_root_sha256", "")
            ),
        )
    return sources


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for a future isolated rebuild."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        shard_provenance, pins, expected_tokenizer = _load_provenance(args.provenance)
        source_policy = production_source_policy()
        from build_isabelle_shard import (  # local sealed tokenizer seam
            _tokenizer_metadata,
            load_vendored_tokenizer,
        )

        backend = load_vendored_tokenizer(args.tokenizer)
        actual_tokenizer = _validated_tokenizer_seal(_tokenizer_metadata(backend))
        if _validated_tokenizer_seal(expected_tokenizer) != actual_tokenizer:
            raise HoldoutError("tokenizer provenance does not match loaded tokenizer")
        tokenizer = TokenizerSeam(
            seal=actual_tokenizer,
            count_text_plus_eos=lambda text: (
                len(backend.encode(text, add_special_tokens=False).ids) + 1
            ),
        )
        sources = _path_sources(args.raw_dir, shard_provenance)
        plan = plan_semantic_holdout(
            sources,
            tokenizer=tokenizer,
            policy_pins=pins,
            source_policy=source_policy,
        )
        write_partition_atomically(plan, sources=sources, output=args.out)
    except (HoldoutError, OSError, TypeError, ValueError) as error:
        print(f"semantic holdout refused: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.out}: {REQUESTED_CLASSES} pooled semantic classes, "
        f"root {plan.manifest['manifest_root_sha256']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

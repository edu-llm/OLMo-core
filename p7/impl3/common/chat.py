"""Chat formatting + assistant-only loss masking (PRD §2.4).

The tokenizer renders the OLMo-2 (Tulu) chat template: roles ``<|system|>`` /
``<|user|>`` / ``<|assistant|>``, with BOS = EOS = ``<|endoftext|>``. We build the
input ids by hand (rather than via ``apply_chat_template`` with a return-assistant-
mask) so the loss mask is explicit and auditable:

    labels = -100 on everything EXCEPT assistant content + its trailing EOS.

The ``<|assistant|>\\n`` header itself is masked, so the training target decodes to
exactly the tutor turns + EOS, and the training assistant header matches the
inference generation prompt (``<|assistant|>\\n``) — no train/serve mismatch.
"""
from __future__ import annotations

import torch

IGNORE = -100


def make_tokenize_fn(tokenizer, max_len: int):
    """Return a ``example -> {input_ids, labels, attention_mask}`` map fn.

    ``example`` must have a ``messages`` list of ``{role, content}`` dicts. A leading
    ``system`` message is optional (pedagogy examples carry one; general examples do
    not — PRD §2.4).
    """
    nl = tokenizer("\n", add_special_tokens=False)["input_ids"]

    def enc(s: str):
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    def tokenize_conversation(example):
        ids = [tokenizer.bos_token_id]
        labels = [IGNORE]
        for m in example["messages"]:
            role, content = m["role"], m["content"]
            if role == "assistant":
                head = enc("<|assistant|>\n")
                body = enc(content) + [tokenizer.eos_token_id]
                ids += head + body + nl
                labels += [IGNORE] * len(head) + body + [IGNORE] * len(nl)
            else:
                tag = "<|system|>\n" if role == "system" else "<|user|>\n"
                seg = enc(tag + content + "\n")
                ids += seg
                labels += [IGNORE] * len(seg)
        return {
            "input_ids": ids[:max_len],
            "labels": labels[:max_len],
            "attention_mask": [1] * len(ids[:max_len]),
        }

    return tokenize_conversation


def has_loss_tokens(example) -> bool:
    """True if any label is not IGNORE (used to drop degenerate rows)."""
    return any(t != IGNORE for t in example["labels"])


def make_collate_fn(tokenizer, extra_keys=()):
    """Right-pad a batch. ``extra_keys`` (e.g. ("weights",)) are padded with 0.0
    to the batch max length and returned as float tensors — used by the Impl-3
    weighted trainer."""
    pad = tokenizer.pad_token_id

    def collate(batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        ii, ll, aa = [], [], []
        extra = {k: [] for k in extra_keys}
        for x in batch:
            n = maxlen - len(x["input_ids"])
            ii.append(x["input_ids"] + [pad] * n)
            ll.append(x["labels"] + [IGNORE] * n)
            aa.append(x["attention_mask"] + [0] * n)
            for k in extra_keys:
                # Rows without an explicit weight (e.g. the eval split) default to 1.0
                # on every position -> the weighted loss reduces to the standard mean
                # CE. Padding positions are 0.0 but are masked out by labels=-100.
                w = list(x.get(k, [1.0] * len(x["input_ids"])))
                extra[k].append(w + [0.0] * n)
        out = {
            "input_ids": torch.tensor(ii, dtype=torch.long),
            "labels": torch.tensor(ll, dtype=torch.long),
            "attention_mask": torch.tensor(aa, dtype=torch.long),
        }
        for k in extra_keys:
            out[k] = torch.tensor(extra[k], dtype=torch.float)
        return out

    return collate

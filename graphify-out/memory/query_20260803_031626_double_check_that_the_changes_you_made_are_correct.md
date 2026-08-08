---
type: "query"
date: "2026-08-03T03:16:26.377677+00:00"
question: "double check that the changes you made are correct by fully scanning the code"
contributor: "graphify"
outcome: "useful"
source_nodes: ["DerivedMaskTrainModule", "TransformerTrainModule", "run_eval.py"]
---

# Q: double check that the changes you made are correct by fully scanning the code

## Answer

Expanded from original query via graph vocabulary: [derived, mask, shifted, labels, supervision, padding, eos, separator, packed, document, qwen, loss]. Full branch review found four release blockers: DerivedMaskTrainModule calls register_buffer although TrainModule is not torch.nn.Module; TransformerTrainModule construction reinitializes and destroys HF weights loaded beforehand; the fixed global-token divisor is divided again by FSDP AVG across eight ranks; and run_eval targets a legacy Metamath-only corpus while the published corpus has six families, with 78 percent of Metamath gold targets rejected by its grounding rule. Token packing and shifted-label algebra were otherwise consistent.

## Outcome

- Signal: useful

## Source Nodes

- DerivedMaskTrainModule
- TransformerTrainModule
- run_eval.py
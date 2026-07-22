  


# OLMo-core

#### Building blocks for OLMo modeling and training



> ## edu-llm fork — adds 370M training on AWS
>
> This branch is a **temporary template** that adds **370M (and 190M) model training on AWS** for
> the AlphaAI/edu-llm project — a SLURM training path layered on top of upstream OLMo-core. The
> changes are **purely additive — new files only, no upstream code is modified.** If you are
> **not** using the AWS training path, this fork behaves exactly like stock OLMo-core and nothing
> below affects you.
>
> **New files:**
>
> - `src/scripts/aws/` — a shared SLURM submit kit (`submit.sh` → `train-job.sbatch` →
> `entrypoint.sh`), an ECR build script, and an env template. Runs the plain `train` command
> under `torchrun` with S3 for data/checkpoints, **bypassing the Beaker/Gantry** `launch` **path**.
> - `src/scripts/train/OLMo2/OLMo2-190M.py` and `OLMo2-370M.py` — small-model pre-training for
> fast hypothesis iteration.
> - `src/scripts/train/sft/Olmo-2-370M-SFT.py` — 370M SFT, decoupled from Beaker.
>
> **Evidence / provenance:** the 190M/370M architectures are official ladder configs from
> `src/olmo_core/nn/transformer/config.py`; the pre-training scripts derive from the official
> `OLMo2-1B.py`; the SFT script derives from `sft/Olmo-2-7B-SFT.py`.
>
> **Caveats (apply only to these new scripts):**
>
> - *Pre-train (*`OLMo2-370M.py`*):* `lr=8e-4` and batch size are **heuristic** scalings of the 1B
> recipe (1B uses `4e-4` / `512×seq`) — verify with a short run. `MAX_DURATION` is inherited at
> `4e12` tokens; override it per run.
> - *SFT (*`Olmo-2-370M-SFT.py`*):* `lr=8e-5` is inherited verbatim from the **7B** recipe and is
> likely **too low** for 370M — sweep it.
> - *Both:* syntax-checked but **not yet dry-run validated**; the `olmo_core.model_ladder` module
> is absent here (so the `ladder/*.py` scripts don't run); GPU compute is not yet provisioned.
>
> See `[src/scripts/aws/README.md](src/scripts/aws/README.md)` for details.



## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system and hardware.

For development, we recommend installing from source:

```bash
git clone https://github.com/allenai/OLMo-core.git
cd OLMo-core
pip install -e .[all]
```

Or you can install from PyPI with:

```bash
pip install ai2-olmo-core
```

There are a number of optional dependencies that must be installed to use certain functionality as well, including:

- [flash-attn](https://github.com/Dao-AILab/flash-attention), [ring-flash-attn](https://github.com/zhuzilin/ring-flash-attention), and [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) for the corresponding attention backends.
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) for a low-memory "fused-linear" loss implementation.
- [torchao](https://github.com/pytorch/ao) for float8 training.
- [grouped_gemm](https://github.com/tgale96/grouped_gemm) for dropless mixture-of-experts (MoE) models. You may need to compile from source until [PR #21](https://github.com/tgale96/grouped_gemm/pull/21) is released (post v0.1.6).
- [QuACK](https://github.com/Dao-AILab/quack) for some CuTe-based kernels.

The published [Docker images](https://github.com/orgs/allenai/packages?repo_name=OLMo-core) contain all core and optional dependencies, and are regularly tested on our in-house H100 clusters.
But there are several things to keep in mind if you intend to use these images:

- They do not come with the OLMo-core package installed, only its dependencies, to accommodate for regular code changes.
- They may not work on your own cluster if you have different hardware or driver/CUDA versions.

If the published images do not work for your use-case for any of the above reasons, you could adapt our [Dockerfile](https://github.com/allenai/OLMo-core/blob/main/src/Dockerfile) to build your own images.

## Official training scripts

Official training scripts for released models can be found in `[src/scripts/official/](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official)`.

These scripts are meant to be launched with `torchrun`, or with OLMo-core's Beaker launch CLI if you have access to Beaker.

For example:

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/save/checkpoints
```

You can override most configuration options from the command-line. For example, to override the learning rate you could launch the script like this:

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/save/checkpoints \
  --train_module.optim.lr=6e-3
```

To continue annealing from a checkpoint, we use a separate script which can be launched like this:

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-anneal.py \
  --save-folder=/path/to/save/checkpoints \
  --checkpoint=https://storage.googleapis.com/ai2-llm/peteish32/step721901
```



### Available Training Scripts


| Model Family | Directory                                                                                                  | Description                                                   |
| ------------ | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **OLMo-2**   | `[src/scripts/official/OLMo2/](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official/OLMo2)` | Training scripts and model card for OLMo-2 32B models         |
| **OLMo-3**   | `[src/scripts/official/OLMo3/](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official/OLMo3)` | Training scripts and model cards for OLMo-3 7B and 32B models |




## Inference



### With Hugging Face Transformers

You can use our Hugging Face [transformers](https://github.com/huggingface/transformers) integration to run inference on the OLMo checkpoints:

```bash
pip install transformers>=4.57.0
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1125-32B")
tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1125-32B")
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# inputs = {k: v.to('cuda') for k,v in inputs.items()} # optional verifying cuda
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=1.0, top_p=0.7)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

Alternatively, with the Hugging Face pipeline abstraction:

```python
from transformers import pipeline
olmo_pipe = pipeline("text-generation", model="allenai/Olmo-3-1125-32B")
print(olmo_pipe("Language modeling is"))
```



### With vLLM

[vLLM](https://docs.vllm.ai/en/latest/) provides high-throughput inference for OLMo models. You can use it for offline batched inference:

```bash
pip install vllm>=0.11.0
```

```python
from vllm import LLM, SamplingParams
llm = LLM(model="allenai/Olmo-3-1125-32B")
sampling_params = SamplingParams(temperature=1.0, top_p=0.7)
prompts = ["Language modeling is"]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

For more details, see the [vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart/#offline-batched-inference).

### With Olmo-core (beta)

Autoregressive generation is supported directly in Olmo-core. Using this capability, we provide a chat-loop demo that can be used to interact with models in an interactive chat session:

```bash
python -m olmo_core.generate.chat https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage3/step11921/ --max-new-tokens 512
```



## Evaluation

Additional tools for evaluating OLMo models are available at the [OLMo Eval](https://github.com/allenai/OLMo-eval) and [olmes](https://github.com/allenai/olmes) repositories.

## Development

The Python library source code is located in `src/olmo_core`. The corresponding tests are located in `src/test`. The library docs are located in `docs`. You can build the docs locally with `make docs`.

Code checks:

- We use `pytest` to run tests. You can run all tests with `pytest -v src/test`. You can also point `pytest` at a specific test file to run it individually.
- We use `isort` and `black` for code formatting. Ideally you should integrate these into your editor, but you can also run them manually or configure them with a pre-commit hook. To validate that all files are formatted correctly, run `make style-check`.
- We use `ruff` as our primary linter. You can run it with `make lint-check`.
- We use `mypy` as our type checker. You can run it with `make type-check`.



## Citing

```bibtex
@misc{olmo20242olmo2furious,
      title={{2 OLMo 2 Furious}},
      author={{Team OLMo} and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2501.00656},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.00656},
}
```


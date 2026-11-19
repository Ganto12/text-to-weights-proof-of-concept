# Text to weights in 2 seconds

**Describe a task in one sentence. Get a LoRA for it in under 2 seconds, with no training step.**

This is my rebuild of the meta-model from Roi Cohen and Ben Shapira's post, [Can we compress backpropagation into a model?](https://roicohen.org/blog/compress-backpropagation/), on Qwen3.5-2B.

| Measure | Result |
|---|---|
| Description → merged model | **1.87 s** |
| Gain over a random update of the same size | **+5.3 points** (second seed: +4.7) |
| Output | a standard 16.9 MB LoRA that loads in plain `peft` |

## How it works

> **Task description** → frozen text encoder → **meta-model** → rank-8 LoRA → **Qwen3.5-2B, adapted**

1. **Teachers.** Train one small LoRA per task: 282 tasks from Super-NaturalInstructions.
2. **Stage 1.** The meta-model learns to write each teacher's LoRA from its task description.
3. **Stage 2.** It is then trained on the task loss directly, through the frozen base model.
4. **Use.** Give it a description it has never seen. It writes a LoRA for all 186 projection matrices.

## Results

Five held-out benchmarks, second-seed model. No training task comes from any test benchmark's source dataset, which is a stricter hold-out than the original post.

| Benchmark | Base model, given the description | With the written update |
|---|---|---|
| BBH logical deduction (3 objects) | 56.7 | **74.7** |
| GSM8K (final number) | 5.2 | **8.2** |
| WinoGrande | 53.6 | **56.6** |
| **Mean of all 5** | 48.6 | **51.9** |

## Quickstart

The scripts expect the repo at `/workspace/poc` and one 24 GB GPU (an RTX 3090 is enough).

```bash
bash scripts/setup_box.sh                    # packages
bash scripts/download_models.sh              # Qwen3.5-2B, bge-large-en-v1.5, Text-to-LoRA task configs
python scripts/build_train_tasks.py          # training tasks
python scripts/build_heldout.py              # decontaminated held-out benchmarks
python scripts/train_teachers.py             # one LoRA per task
bash scripts/run_dose.sh 282 strict data/train_tasks.json   # stage 1, stage 2, scoring
```

Then turn any description into a LoRA. See [USAGE.md](USAGE.md).

## Repository

| File | What it does |
|---|---|
| `scripts/meta_model.py` | the meta-model: encoder, trunk and per-projection heads |
| `scripts/train_meta.py` | stage 1 and stage 2 training |
| `scripts/train_teachers.py` | one teacher LoRA per task |
| `scripts/eval_arms.py` | scores the base model, the written update and a random update |
| `scripts/export_peft.py` | description in, standard PEFT adapter out |
| `scripts/delta_hooks.py` | applies a written update to the base model |

## Next

More training tasks and a wider benchmark set.

## Credits

Method and original results: Roi Cohen and Ben Shapira. Task configs from [Text-to-LoRA](https://github.com/SakanaAI/text-to-lora) (Sakana AI).

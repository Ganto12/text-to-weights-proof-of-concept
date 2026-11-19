"""Integration demo: description -> written update -> standard PEFT LoRA adapter -> plain transformers.

Usage: python export_peft.py --meta <meta-model ckpt> --description "<one sentence>" --out <adapter dir>
Checks: the PEFT-loaded model (no hooks) matches the hook path on 20 prompts; times every step;
merges the adapter (zero extra inference cost) and checks the merged model too.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/workspace/poc/scripts")
from common import ROOT, load_base, log, prompt_ids, write_json  # noqa
import delta_hooks as dh
from meta_model import MetaWriter

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--description", required=True)
ap.add_argument("--bench", default="comve", help="held-out benchmark whose POOL rows give the 20 test prompts")
ap.add_argument("--out", default=f"{ROOT}/peft_export/adapter")
args = ap.parse_args()

T = {}
t0 = time.time()
tok, model = load_base()
targets = dh.find_targets(model)
T["load_base_s"] = time.time() - t0

t0 = time.time()
writer = MetaWriter.load(args.meta, targets)
T["load_meta_model_s"] = time.time() - t0

torch.cuda.synchronize(); t0 = time.time()
delta = writer.write(args.description)
torch.cuda.synchronize(); T["write_update_s"] = time.time() - t0

# ---- export as a standard PEFT LoRA adapter (rank 8, alpha 16, the 186 projection matrices) ----
t0 = time.time()
os.makedirs(args.out, exist_ok=True)
from safetensors.torch import save_file
tensors = {}
for n, (A, B) in delta.items():
    tensors[f"base_model.model.{n}.lora_A.weight"] = A.detach().to(torch.bfloat16).cpu().contiguous()
    tensors[f"base_model.model.{n}.lora_B.weight"] = B.detach().to(torch.bfloat16).cpu().contiguous()
save_file(tensors, f"{args.out}/adapter_model.safetensors")
cfg = dict(peft_type="LORA", task_type="CAUSAL_LM", r=dh.RANK, lora_alpha=dh.ALPHA, lora_dropout=0.0, bias="none",
           target_modules=sorted(delta.keys()), use_rslora=False, use_dora=False, fan_in_fan_out=False,
           init_lora_weights=True, base_model_name_or_path="Qwen/Qwen3.5-2B", inference_mode=True,
           description=args.description)
json.dump(cfg, open(f"{args.out}/adapter_config.json", "w"), indent=1)
T["export_adapter_s"] = time.time() - t0
size_mb = os.path.getsize(f"{args.out}/adapter_model.safetensors") / 1e6

# ---- 20 prompts from the benchmark's POOL rows (never test items) ----
rows = [json.loads(l) for l in open(f"{ROOT}/data/heldout/{args.bench}_pool.jsonl")][:20]
prompts = [prompt_ids(tok, r["text"]) for r in rows]


@torch.no_grad()
def run(m):
    outs, last = [], []
    for p in prompts:
        x = torch.tensor([p]).cuda()
        last.append(m(input_ids=x).logits[0, -1].float())
        g = m.generate(input_ids=x, max_new_tokens=16, do_sample=False, use_cache=True)
        outs.append(tok.decode(g[0, len(p):], skip_special_tokens=True))
    return outs, torch.stack(last)


hooks = dh.DeltaHooks(targets)
hooks.set(delta)
hook_out, hook_logits = run(model)
hooks.clear(); hooks.remove()
base_out, base_logits = run(model)

# ---- plain transformers + peft, no hooks ----
from peft import PeftModel
t0 = time.time()
peft_model = PeftModel.from_pretrained(model, args.out)
peft_model.eval()
T["peft_load_s"] = time.time() - t0
peft_out, peft_logits = run(peft_model)

t0 = time.time()
merged = peft_model.merge_and_unload()
T["merge_s"] = time.time() - t0
merged_out, merged_logits = run(merged)

res = dict(
    description=args.description, adapter_dir=args.out, adapter_mb=round(size_mb, 1), timings_s={k: round(v, 3) for k, v in T.items()},
    description_to_merged_model_s=round(T["write_update_s"] + T["export_adapter_s"] + T["peft_load_s"] + T["merge_s"], 2),
    peft_vs_hook_same_greedy=sum(a == b for a, b in zip(peft_out, hook_out)),
    peft_vs_hook_logit_max_abs=round(float((peft_logits - hook_logits).abs().max()), 4),
    peft_vs_hook_argmax_agree=float((peft_logits.argmax(-1) == hook_logits.argmax(-1)).float().mean()),
    merged_vs_hook_same_greedy=sum(a == b for a, b in zip(merged_out, hook_out)),
    merged_vs_hook_argmax_agree=float((merged_logits.argmax(-1) == hook_logits.argmax(-1)).float().mean()),
    update_changes_output_vs_base=sum(a != b for a, b in zip(hook_out, base_out)),
    n_prompts=len(prompts), examples=[dict(prompt=rows[i]["text"][:120], base=base_out[i], update=hook_out[i], peft=peft_out[i]) for i in range(3)],
)
write_json(res, f"{ROOT}/peft_export/result.json")
log("[peft] " + json.dumps({k: v for k, v in res.items() if k != "examples"}))

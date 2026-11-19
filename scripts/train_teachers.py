"""Teacher adapters (stage 1 targets) and, with --mode baseline, the LoRA baselines.

Fixed recipe (PLAN.md section 3): rank 8, alpha 16, no dropout, all 192 projection matrices,
AdamW lr 2e-4, cosine with 10 warmup steps, batch 16 (2 x 8), max 512 tokens, answer-token loss.
Teacher: up to 500 train rows, 2 epochs; pass = valid loss at least 10% below base.
Workers share one GPU; each takes tasks i where i % nworkers == worker.
"""
import argparse
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, "/workspace/poc/scripts")
from common import (ROOT, collate, encode_rows, length_grouped_batches, load_base, log, mean_answer_loss,  # noqa
                    answer_loss, read_json, write_json)
import delta_hooks as dh

ap = argparse.ArgumentParser()
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--nworkers", type=int, default=1)
ap.add_argument("--limit", type=int, default=0, help="stop after this many tasks (0 = all)")
ap.add_argument("--mode", choices=["teacher", "baseline"], default="teacher")
ap.add_argument("--rows", type=int, default=500)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tasks", default=f"{ROOT}/data/train_tasks.json")
ap.add_argument("--follow", action="store_true", help="train tasks as the data build writes them")
ap.add_argument("--reverse", action="store_true", help="walk the task list from the end (second worker)")
args = ap.parse_args()

LR, WARMUP, MICRO, ACCUM = 2e-4, 10, 8, 2

tok, model = load_base(grad_ckpt=True)
targets = dh.find_targets(model)
hooks = dh.DeltaHooks(targets)
pad = tok.pad_token_id

def seeded_order():
    """Same candidate order as build_train_tasks.py (name/id drops, seed-0 shuffle)."""
    import random as _r
    import re as _re
    import yaml
    import build_train_tasks as btt
    cfg = yaml.safe_load(open(f"{ROOT}/t2l/configs/hyper_lora_decontam_lol_tasks.yaml"))
    order = []
    for lid in cfg["train_ds_names"]:
        short = yaml.safe_load(open(f"{ROOT}/t2l/tasks/{lid}/metadata.yaml"))["task_name"].split("/", 1)[1]
        num = _re.match(r"task(\d+)_", short).group(1)
        if not (btt.DROP_RE.search(short) or num in btt.DROP_IDS):
            order.append(short)
    _r.Random(0).shuffle(order)
    return order


def passes_filters(task):
    p = f"{ROOT}/data/sni/{task}.json"
    if not os.path.exists(p):
        return None
    rec = read_json(p)
    st = rec["_stats"]
    return (st["n_train"] >= 300 and st["n_valid"] >= 100 and st["median_answer_tokens"] <= 128
            and len(rec["descriptions"]) >= 128)


def follow_tasks():
    """Yield tasks for this worker as the data build writes them. After the 13-gram scan
    (data/train_tasks.json exists), only final tasks count, and any teacher trained before the
    scan on a task that lost rows is retrained (its old metrics renamed, never deleted)."""
    order = seeded_order()
    mine = [t for i, t in enumerate(order) if i % args.nworkers == args.worker]
    while True:
        final = read_json(f"{ROOT}/data/train_tasks.json")["tasks"] if os.path.exists(f"{ROOT}/data/train_tasks.json") else None
        pending = False
        for t in mine:
            if final is not None and t not in final:
                continue
            mp = f"{ROOT}/teachers/{t}/metrics.json"
            if os.path.exists(mp):
                if final is not None and read_json(f"{ROOT}/data/sni/{t}.json")["_stats"].get("ngram_rows_removed"):
                    if not read_json(mp).get("post_scan"):
                        os.replace(mp, mp.replace("metrics.json", "metrics_prescan_stale.json"))
                        log(f"[teach w{args.worker}] {t}: retrain, 13-gram scan removed rows after pre-scan training")
                        yield t
                continue
            ok = passes_filters(t)
            if ok is None:
                pending = True
                continue
            if ok:
                yield t
        if final is not None and not pending:
            return
        if os.path.exists(f"{ROOT}/data/train_tasks_candidates.json") and final is None:
            cands = set(read_json(f"{ROOT}/data/train_tasks_candidates.json")["tasks"])
            mine = [t for t in mine if t in cands]
        time.sleep(30)


if args.follow:
    task_iter = follow_tasks()
else:
    tasks = read_json(args.tasks)["tasks"]
    if args.reverse:
        tasks = tasks[::-1]
    task_iter = [t for i, t in enumerate(tasks) if i % args.nworkers == args.worker]
log(f"[teach w{args.worker}] mode {args.mode}, follow={args.follow}")
done = 0
for task in task_iter:
    out_dir = f"{ROOT}/teachers/{task}"
    if os.path.exists(f"{out_dir}/metrics.json"):
        continue
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    post_scan = os.path.exists(f"{ROOT}/data/train_tasks.json")  # decided BEFORE the rows are read
    rec = read_json(f"{ROOT}/data/sni/{task}.json")
    rows = rec["train"][:args.rows]
    enc_tr, drop_tr = encode_rows(tok, rows)
    enc_va, drop_va = encode_rows(tok, rec["valid"][:100])
    if len(enc_tr) < 50 or len(enc_va) < 20:
        write_json(dict(task=task, skipped="too few rows under 512 tokens", n_train=len(enc_tr), n_valid=len(enc_va)),
                   f"{out_dir}/metrics.json")
        log(f"[teach w{args.worker}] {task}: SKIP few rows ({len(enc_tr)} train, {len(enc_va)} valid)")
        continue
    hooks.clear()
    base_val = mean_answer_loss(model, enc_va, pad)

    delta = dh.new_lora(targets, seed=args.seed)
    params = dh.delta_params(delta)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    batches = []
    for ep in range(args.epochs):
        batches += length_grouped_batches(enc_tr, MICRO, seed=args.seed * 100 + ep)
    total_steps = math.ceil(len(batches) / ACCUM)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    hooks.set(delta)
    model.train()
    n_tok, step, losses = 0, 0, []
    for bi, ids in enumerate(batches):
        inp, lab, att = collate(enc_tr, ids, pad)
        loss = answer_loss(model, inp, lab, att) / ACCUM
        loss.backward()
        losses.append(loss.item() * ACCUM)
        n_tok += int(att.sum().item())
        if (bi + 1) % ACCUM == 0 or bi == len(batches) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
    model.eval()
    teach_val = mean_answer_loss(model, enc_va, pad)
    hooks.clear()
    dh.save_delta(delta, f"{out_dir}/adapter.safetensors")
    secs = time.time() - t0
    norms = dh.frob_norms(delta)
    m = dict(task=task, mode=args.mode, post_scan=post_scan, base_val_loss=base_val, teacher_val_loss=teach_val,
             rel_drop=(base_val - teach_val) / max(base_val, 1e-9), passed=bool(teach_val <= 0.9 * base_val),
             n_train=len(enc_tr), n_train_dropped_long=drop_tr, n_valid=len(enc_va), steps=step,
             train_tokens=n_tok, seconds=round(secs, 1), tokens_per_s=round(n_tok / secs, 1),
             first_loss=sum(losses[:4]) / max(len(losses[:4]), 1), last_loss=sum(losses[-4:]) / max(len(losses[-4:]), 1),
             delta_frob_total=math.sqrt(sum(v * v for v in norms.values())), gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    write_json(m, f"{out_dir}/metrics.json")
    done += 1
    log(f"[teach w{args.worker}] {task}: base {base_val:.3f} -> {teach_val:.3f} ({m['rel_drop']*100:.0f}% drop, "
        f"{'PASS' if m['passed'] else 'fail'}) {step} steps {secs:.0f}s {m['tokens_per_s']:.0f} tok/s mem {m['gpu_mem_gb']}GB")
    del delta, params, opt
    torch.cuda.empty_cache()
    if args.limit and done >= args.limit:
        break
log(f"[teach w{args.worker}] shard finished, {done} trained this run")

"""Train the meta-model. --stage 1: match teacher updates. --stage 2: task loss through the frozen base.

Stage 1 needs no base forward. Stage 2 adds the written update through hooks and backprops
answer-token loss through the frozen bf16 base (gradient checkpointing) into the meta-model.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, "/workspace/text-to-weights/scripts")
from common import (ROOT, answer_loss, collate, encode_rows, load_base, log, mean_answer_loss, read_json,  # noqa
                    write_json)
import delta_hooks as dh
from meta_model import Encoder, MetaModel, product_rel_loss
import timeline as tl

PROBE_N = 40        # probe items per benchmark, from each benchmark's POOL rows (never test items)
PROBE_EVERY = 1000  # stage 2 steps between probes (plus step 0 and the final step)


def load_probes():
    man = read_json(f"{ROOT}/data/heldout_manifest.json")
    probes = {}
    keep = set(args.probe_benches.split(",")) if args.probe_benches else None
    for b, m in man.items():
        if "error" in m or (keep and b not in keep):
            continue
        rows = [json.loads(l) for l in open(f"{ROOT}/data/heldout/{b}_pool.jsonl")][:(args.probe_n or PROBE_N)]
        if rows:
            probes[b] = (m["description"], rows)
    # ComVE first: the quick-reproduction headline curve
    return dict(sorted(probes.items(), key=lambda kv: kv[0] != "comve"))


@torch.no_grad()
def run_probe(step, phase, model, base, tok, hooks, targets, enc, probes, arms=("C", "R")):
    """Score update arms on the probe sets (pool rows). Logged as split='probe' in the timeline."""
    from eval_arms import MAXNEW, generate, parse_and_score
    from common import prompt_ids
    t0 = time.time()
    base.eval(); model.eval()
    means = {a: [] for a in arms}
    for b, (desc, rows) in probes.items():
        prompts = [prompt_ids(tok, r["text"]) for r in rows]
        if model is not None:
            dC = {n: (A.detach(), B.detach()) for n, (A, B) in clamp(model(enc([desc])[0], targets)).items()}
        for arm in arms:
            hooks.clear()
            if arm == "A0":
                pass
            elif arm == "A":
                prompts_a = [prompt_ids(tok, f"{desc}\n\n{r['text']}") for r in rows]
            elif arm == "C":
                hooks.set(dC)
            elif arm == "R":
                hooks.set(dh.random_like_norm(dC, seed=0))
            reps = generate(base, tok, prompts_a if arm == "A" else prompts, MAXNEW[rows[0]["kind"]], args.probe_bs)
            res = [parse_and_score(rows[0]["kind"], rep, r) for rep, r in zip(reps, rows)]
            if b == "comve" and arm in ("C", "R"):
                log(f"[probe] {arm} comve sample replies: {[x[:80] for x in reps[:3]]}")
            acc = 100 * sum(ok for _, ok in res) / len(rows)
            unp = sum(pred == "" for pred, _ in res)
            tl.append(phase, "accuracy", round(acc, 2), arm=arm, benchmark=b, split="probe", step=step, n=len(rows), unparsed=unp, note=(args.label or None))
            means[arm].append(acc)
        hooks.clear()
    for arm, v in means.items():
        tl.append(phase, "accuracy_mean20", round(sum(v) / len(v), 2), arm=arm, benchmark="_mean", split="probe", step=step, n=len(v), note=(args.label or None))
    secs = time.time() - t0
    tl.append(phase, "probe_seconds", round(secs, 1), step=step, note=(args.label or None))
    log(f"[probe] step {step} {phase}: " + ", ".join(f"{a} {sum(v)/len(v):.1f}" for a, v in means.items()) + f" ({secs:.0f}s)")
    base.train(); model.train()
    return {a: sum(v) / len(v) for a, v in means.items()}

from train_meta_val import VAL_TASKS

ap = argparse.ArgumentParser()
ap.add_argument("--stage", type=int, required=True)
ap.add_argument("--steps", type=int, default=0)
ap.add_argument("--init", default="")
ap.add_argument("--out", default="")
ap.add_argument("--label", default="", help="timeline phase prefix, e.g. 'early_100t_'")
ap.add_argument("--probe-every", type=int, default=0)
ap.add_argument("--probe-benches", default="", help="comma list; default all 20")
ap.add_argument("--probe-n", type=int, default=0, help="pool rows per probe benchmark (default PROBE_N)")
ap.add_argument("--stop-gain", type=float, default=0.0, help="stop when ComVE probe C >= A + this (retry mode)")
ap.add_argument("--max-minutes", type=float, default=0.0, help="stop stage 2 after this many minutes")
ap.add_argument("--lr", type=float, default=2e-5, help="stage 2 learning rate (stability fix 20:25 UTC: was 1e-4)")
ap.add_argument("--tasks-file", default="data/train_tasks.json")
ap.add_argument("--max-teachers", type=int, default=0, help="use only the first N passing teachers in list order")
ap.add_argument("--extra-tasks", default="", help="comma list of extra task ids (post-style set)")
ap.add_argument("--no-clamp", action="store_true")
ap.add_argument("--no-format-line", action="store_true")
ap.add_argument("--probe-bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=0, help="meta-model init and data-order seed (robustness re-runs)")
args = ap.parse_args()


def embed_descriptions(enc, tasks):
    p = f"{ROOT}/data/desc_emb.pt"
    cache = torch.load(p) if os.path.exists(p) else {}
    todo = [t for t in tasks if t not in cache]
    for t in todo:
        rec = read_json(f"{ROOT}/data/sni/{t}.json")
        cache[t] = enc(rec["descriptions"][:128]).cpu()
    if todo:
        torch.save(cache, p)
    return cache


CLAMP_NORM = None


def clamp(delta, max_norm=None):
    """Stability fix: scale the written update down (through B, differentiably) so its total
    Frobenius norm is at most the mean teacher norm."""
    m = max_norm if max_norm is not None else CLAMP_NORM
    if not m:
        return delta
    sq = 0.0
    for A, B in delta.values():
        Af, Bf = A.float(), B.float()
        sq = sq + torch.trace((Bf.t() @ Bf) @ (Af @ Af.t()))
    norm = dh.SCALE * torch.sqrt(sq.clamp_min(1e-12))
    k = torch.clamp(m / norm, max=1.0)
    return {n: (A, B * k) for n, (A, B) in delta.items()}


def format_line(rec):
    """Benchmark-style format instruction for an SNI training task, from its own train answers."""
    outs = [r["output"] for r in rec.get("train", [])[:300]]
    labels = sorted(set(outs))
    if outs and len(labels) <= 10 and all(len(x.split()) <= 5 for x in labels):
        return "Answer with one of: " + ", ".join(labels) + "."
    if outs and all(x.replace(".", "", 1).replace("-", "", 1).isdigit() for x in outs):
        return "Answer with the number only."
    return "Answer with the answer only."


def save(model, shapes, path, extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), shapes=shapes, **extra), path)


def main():
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tasks = read_json(f"{ROOT}/{args.tasks_file}")["tasks"]
    if args.extra_tasks:
        tasks = tasks + [t for t in args.extra_tasks.split(",") if t and t not in tasks]
    passing = [t for t in tasks if os.path.exists(f"{ROOT}/teachers/{t}/metrics.json")
               and read_json(f"{ROOT}/teachers/{t}/metrics.json").get("passed")]
    log(f"[meta s{args.stage}] {len(passing)} passing teachers of {len(tasks)} tasks")
    if args.max_teachers:
        strict = [t for t in passing if t not in set(args.extra_tasks.split(","))]
        extra = [t for t in passing if t in set(args.extra_tasks.split(","))]
        passing = strict[:args.max_teachers] + extra
    global CLAMP_NORM
    norms = [read_json(f"{ROOT}/teachers/{t}/metrics.json").get("delta_frob_total") for t in passing]
    norms = [x for x in norms if x]
    CLAMP_NORM = None if args.no_clamp else sum(norms) / len(norms)
    log(f"[meta s{args.stage}] using {len(passing)} teachers; clamp norm {CLAMP_NORM}")
    enc = Encoder()
    embs = embed_descriptions(enc, passing)

    if args.stage == 1:
        # shapes without loading the base: read one teacher
        any_delta = dh.load_delta(f"{ROOT}/teachers/{passing[0]}/adapter.safetensors", device="cpu")
        tmeta = read_json(f"{ROOT}/data/target_meta.json")  # written by stage 2 / check script
        targets = {n: dict(layer=v["layer"], type=v["type"], din=v["din"], dout=v["dout"]) for n, v in tmeta.items()}
        shapes = {}
        for v in targets.values():
            shapes[v["type"]] = (v["din"], v["dout"])
        model = MetaModel(shapes).cuda()
        log(f"[meta s1] params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        from meta_model import layout, stack_delta, stacked_rel_loss
        lay = layout(targets)
        rnd = random.Random(0 + 1000 * args.seed)
        held = passing[-10:]
        train = passing[:-10]
        # all teachers stacked on the GPU once (about 17 MB each in bf16), with ||BA||^2 precomputed
        teach, teach_tt = {}, {}
        for t in passing:
            teach[t], teach_tt[t] = stack_delta(dh.load_delta(f"{ROOT}/teachers/{t}/adapter.safetensors", device="cpu"), lay)
        emb_gpu = {t: embs[t].cuda() for t in passing}
        log(f"[meta s1] teachers on GPU: {len(teach)}, mem {torch.cuda.memory_allocated()/1e9:.2f} GB")

        def batch(ts, pick):
            e = torch.stack([emb_gpu[t][pick(t)] for t in ts])
            tgt = {k: (torch.stack([teach[t][k][0] for t in ts]), torch.stack([teach[t][k][1] for t in ts])) for k in lay}
            tt = {k: torch.stack([teach_tt[t][k] for t in ts]) for k in lay}
            return e, tgt, tt

        steps = args.steps or 4000
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)))
        os.makedirs(f"{ROOT}/meta", exist_ok=True)
        logf = open(f"{ROOT}/meta/{args.label}stage1_log.jsonl", "a")
        t0 = time.time()
        for step in range(steps):
            ts = [rnd.choice(train) for _ in range(16)]
            e, tgt, tt = batch(ts, lambda t: rnd.randrange(len(emb_gpu[t])))
            loss = stacked_rel_loss(model.forward_stacked(e, lay), tgt, tt)
            loss.backward()
            loss_acc = loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 100 == 0 or step == steps - 1:
                with torch.no_grad():
                    e, tgt, tt = batch(held, lambda t: 0)
                    pred = model.forward_stacked(e, lay)
                    per = []
                    for i in range(len(held)):
                        p_i = {k: (v[0][i:i + 1], v[1][i:i + 1]) for k, v in pred.items()}
                        g_i = {k: (v[0][i:i + 1], v[1][i:i + 1]) for k, v in tgt.items()}
                        per.append(stacked_rel_loss(p_i, g_i, {k: v[i:i + 1] for k, v in tt.items()}).item())
                rec = dict(step=step, train_rel=loss_acc, held_median_rel=sorted(per)[len(per) // 2],
                           steps_per_s=round((step + 1) / (time.time() - t0), 2))
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                log(f"[meta s1] {rec}")
                tl.append("stage1", "train_loss", round(loss_acc, 4), step=step, note=(args.label or None))
                tl.append("stage1", "val_loss", round(rec["held_median_rel"], 4), step=step, n=len(held),
                          note=(args.label or "") + "median relative distance, 10 held-back teachers")
        if rec["held_median_rel"] >= 0.7:
            log(f"[meta s1] WARN stage-1 bar missed (median {rec['held_median_rel']:.3f} >= 0.7); proceeding to stage 2 as pre-registered")
        save(model, shapes, args.out or f"{ROOT}/meta/stage1/ckpt_final.pt", dict(stage=1, steps=steps))
        return

    # stage 2
    tok, base = load_base(grad_ckpt=True)
    targets = dh.find_targets(base)
    hooks = dh.DeltaHooks(targets)
    shapes = dh.type_shapes(targets)
    model = MetaModel(shapes).cuda()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cuda")["state_dict"])
        log(f"[meta s2] init from {args.init}")
    steps = args.steps or 6000
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    enc_cache = {}

    def task_rows(t, split, n):
        key = (t, split)
        if key not in enc_cache:
            rec = read_json(f"{ROOT}/data/sni/{t}.json")
            rows = rec[split][:n]
            if not args.no_format_line:
                fl = format_line(rec)
                rows = [dict(r, input=f"{r['input']}\n{fl}") for r in rows]
            enc_cache[key] = encode_rows(tok, rows)[0]
        return enc_cache[key]

    val_emb = {t: enc(read_json(f"{ROOT}/data/sni/{t}.json")["descriptions"][:1])[0] for t in VAL_TASKS
               if os.path.exists(f"{ROOT}/data/sni/{t}.json")}
    val_base = {}
    for t in val_emb:
        val_base[t] = mean_answer_loss(base, task_rows(t, "test", 100), tok.pad_token_id)
    log(f"[meta s2] validation base losses {val_base}")
    probes = load_probes()
    log(f"[meta s2] probe sets: {len(probes)} benchmarks x up to {PROBE_N} pool rows")
    for t, v in val_base.items():
        tl.append("stage2", "val_loss_base", round(v, 4), benchmark=t, split="sni_val", step=0, note=(args.label or None))
    p0 = run_probe(0, "stage2", model, base, tok, hooks, targets, enc, probes, arms=("A0", "A", "C", "R"))
    a_probe = p0["A"]
    t_start = time.time()
    stop_reason = None
    rnd = random.Random(1 + 1000 * args.seed)
    out_dir = args.out or f"{ROOT}/meta/stage2"
    os.makedirs(out_dir, exist_ok=True)
    logf = open(f"{out_dir}/log.jsonl", "a")
    t0 = time.time(); ntok = 0
    for step in range(steps):
        model.train(); base.train()
        acc = 0.0
        for _ in range(4):
            t = rnd.choice(passing)
            e = embs[t][rnd.randrange(len(embs[t]))].cuda()
            rows = task_rows(t, "train", 1000)
            ids = rnd.sample(range(len(rows)), min(8, len(rows)))
            inp, lab, att = collate(rows, ids, tok.pad_token_id)
            hooks.set(clamp(model(e, targets)))
            loss = answer_loss(base, inp, lab, att) / 4
            loss.backward()
            acc += loss.item(); ntok += int(att.sum())
        hooks.clear()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            log(f"[meta s2] step {step} loss {acc:.4f} {ntok/(time.time()-t0):.0f} tok/s")
            tl.append("stage2", "train_loss", round(acc, 4), step=step, note=(args.label or None))
        if args.stop_gain and (step + 1) % (args.probe_every or PROBE_EVERY) == 0:
            pc = run_probe(step + 1, "stage2", model, base, tok, hooks, targets, enc, probes)
            mins = (time.time() - t_start) / 60
            if pc["C"] >= a_probe + args.stop_gain:
                stop_reason = f"ComVE probe C {pc['C']:.1f} >= A {a_probe:.1f} + {args.stop_gain}"
            elif args.max_minutes and mins >= args.max_minutes:
                stop_reason = f"time cap {args.max_minutes} min (probe C {pc['C']:.1f}, A {a_probe:.1f})"
            save(model, shapes, f"{out_dir}/ckpt_{step+1}.pt", dict(stage=2, step=step + 1, clamp_norm=CLAMP_NORM))
            if stop_reason:
                save(model, shapes, f"{out_dir}/ckpt_stop.pt", dict(stage=2, step=step + 1, stop_reason=stop_reason, clamp_norm=CLAMP_NORM))
                write_json(dict(step=step + 1, reason=stop_reason, minutes=round(mins, 1)), f"{out_dir}/stop.json")
                log(f"[meta s2] STOP at step {step+1}: {stop_reason}")
                return
            continue
        if (step + 1) % 500 == 0 or step == steps - 1:
            model.eval(); base.eval()
            wins, vl = 0, {}
            with torch.no_grad():
                for t, e in val_emb.items():
                    hooks.set(clamp(model(e.cuda(), targets)))
                    vl[t] = mean_answer_loss(base, task_rows(t, "test", 100), tok.pad_token_id)
                    hooks.clear()
                    wins += vl[t] < val_base[t]
            rec = dict(step=step + 1, train_loss=acc, val_wins=wins, val=vl, tok_per_s=ntok / (time.time() - t0))
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log(f"[meta s2] VAL step {step+1}: written update beats base on {wins}/{len(vl)}")
            tl.append("stage2", "val_loss", round(sum(vl.values()) / len(vl), 4), step=step + 1, split="sni_val", n=len(vl), note=(args.label or "") + "mean answer loss, 6 SNI validation tasks")
            tl.append("stage2", "val_wins_of6", wins, step=step + 1, split="sni_val", n=len(vl), note=(args.label or None))
            if (step + 1) % (args.probe_every or PROBE_EVERY) == 0 or step == steps - 1:
                run_probe(step + 1, "stage2", model, base, tok, hooks, targets, enc, probes)
            save(model, shapes, f"{out_dir}/ckpt_{step+1}.pt", dict(stage=2, step=step + 1, clamp_norm=CLAMP_NORM))
            if step + 1 == 1500 and wins < 4:
                log("[meta s2] EARLY STOP RULE: fewer than 4/6 validation wins at step 1500 - stopping for debug")
                break
    save(model, shapes, f"{out_dir}/ckpt_final.pt", dict(stage=2, step=step + 1, clamp_norm=CLAMP_NORM))
    # Pre-registered: the verdict checkpoint is the one with the lowest mean SNI-validation loss.
    recs = [json.loads(l) for l in open(f"{out_dir}/log.jsonl")]
    best = min(recs, key=lambda r: sum(r["val"].values()) / len(r["val"]))
    write_json(dict(selected=f"{out_dir}/ckpt_{best['step']}.pt", step=best["step"],
                    rule="lowest mean loss on the 6 SNI validation tasks; benchmarks never used"), f"{out_dir}/selected.json")
    log(f"[meta s2] selected checkpoint step {best['step']}")


if __name__ == "__main__":
    main()

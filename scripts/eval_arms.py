"""Score arms on the 20 held-out benchmarks (PLAN.md section 5).

Base arms: A0 (bare item), A (description + item), B (in-context examples + item).
Update arms: pass --delta-from <file pattern> (LoRA baselines) or --meta <ckpt> with
--arm C|R|S|G. All arms use the same item text, parser and greedy decoding.
Generation is batched with left padding only if a startup self-test shows identical
replies to one-at-a-time decoding; otherwise batch size 1.
"""
import argparse
import json
import os
import re
import string
import sys
import time

import torch

sys.path.insert(0, "/workspace/text-to-weights/scripts")
from common import ROOT, load_base, log, prompt_ids, read_json, write_json  # noqa
import delta_hooks as dh

MAXNEW = dict(letter=32, digit=32, yesno=32, exact=32, number=16, squad=48, ner_json=192)  # math keeps the post's 16-token cap


def norm_squad(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _option_texts(item):
    """Numbered or lettered option lines in the item text: {"1": text, ...} or {"A": text, ...}."""
    opts = {}
    for line in item["text"].splitlines():
        m = re.match(r"^\s*([12]|[A-J])\.\s+(.*)$", line)
        if m:
            opts[m.group(1)] = m.group(2)
    return opts


def _match_option(reply, item):
    """Fallback (logged, applied to every arm alike): if the reply copies one option's text instead
    of giving its number or letter, map it to that option by word overlap (needs > 0.5 and a clear winner)."""
    rw = set(re.findall(r"[a-z0-9']+", reply.lower()))
    if not rw:
        return ""
    best = sorted(((len(rw & set(re.findall(r"[a-z0-9']+", t.lower()))) / max(1, len(set(re.findall(r"[a-z0-9']+", t.lower())))), k)
                   for k, t in _option_texts(item).items()), reverse=True)
    if best and best[0][0] > 0.5 and (len(best) == 1 or best[0][0] - best[1][0] > 0.1):
        return best[0][1]
    return ""


def parse_and_score(kind, reply, item):
    r = reply.strip()
    r = re.sub(r"^(the )?(final )?answer( is)?\s*[:\-]?\s*", "", r, flags=re.I)  # "Answer: X" (seen in arm B replies)
    if kind == "letter":
        lone = re.fullmatch(r"\(?([a-jA-J])\)?[.)]?", r)  # a reply that is just one letter, any case (01:57 UTC fix)
        if lone:
            pred = lone.group(1).upper()
            return pred, pred == item["gold"]
        m = re.search(r"\(?\b([A-J])\b\)?", r)
        pred = m.group(1) if m else _match_option(r, item)
        return pred, pred == item["gold"]
    if kind == "digit":
        m = re.search(r"[12]", r)
        pred = m.group(0) if m else _match_option(r, item)
        return pred, pred == item["gold"]
    if kind == "yesno":
        m = re.search(r"\b(yes|no)\b", r.lower())
        pred = m.group(1) if m else ""
        return pred, pred == item["gold"]
    if kind == "exact":
        g = item["gold"].strip()
        first = r.splitlines()[0].strip() if r else ""
        if re.fullmatch(r"\([A-Z]\)", g):
            m = re.search(r"\(?\b([A-R])\b\)?", first)
            pred = f"({m.group(1)})" if m else first
        else:
            pred = first.rstrip(".").strip()
        return pred, pred.lower() == g.lower()
    if kind == "number":
        m = re.search(r"-?\d[\d,]*\.?\d*", r)
        pred = m.group(0).replace(",", "").rstrip(".") if m else ""
        try:
            ok = abs(float(pred) - float(item["gold"])) < 1e-6
        except ValueError:
            ok = False
        return pred, ok
    if kind == "squad":
        first = r.splitlines()[0] if r else ""
        return first, any(norm_squad(first) == norm_squad(g) for g in item.get("golds", [item["gold"]]))
    if kind == "ner_json":
        m = re.search(r"\{.*\}", r, re.S)
        try:
            pred = json.loads(m.group(0)) if m else {}
            gold = json.loads(item["gold"])
            ok = all(sorted(map(str, pred.get(k, []) or [])) == sorted(gold[k]) for k in gold)
        except Exception:  # noqa
            pred, ok = {}, False
        return json.dumps(pred), ok
    raise ValueError(kind)


def build_user(arm, bench, item, desc, pool, tok, icl_budget=4096, icl_cap=84):
    if arm == "A":
        return f"{desc}\n\n{item['text']}"
    if arm == "B":
        shots, used = [], len(prompt_ids(tok, item["text"])) + 64
        for ex in pool:
            s = f"{ex['text']}\nAnswer: {ex['gold'] if ex['kind'] != 'ner_json' else ex['gold']}"
            n = len(tok(s, add_special_tokens=False)["input_ids"]) + 4
            if used + n > icl_budget or len(shots) >= icl_cap:
                break
            shots.append(s); used += n
        item["_n_shots"] = len(shots)
        return "Here are solved examples.\n\n" + "\n\n".join(shots) + "\n\nNow solve this one.\n\n" + item["text"]
    return item["text"]


@torch.no_grad()
def generate(model, tok, prompts, max_new, bs):
    outs = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for b0 in range(0, len(order), bs):
        idx = order[b0:b0 + bs]
        L = max(len(prompts[i]) for i in idx)
        inp = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            p = prompts[i]
            inp[j, L - len(p):] = torch.tensor(p); att[j, L - len(p):] = 1
        g = model.generate(input_ids=inp.cuda(), attention_mask=att.cuda(), max_new_tokens=max_new, do_sample=False, use_cache=True,
                           pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            outs[i] = tok.decode(g[j, L:], skip_special_tokens=True)
    return outs


def pick_batch_size(model, tok, items):
    prompts = [prompt_ids(tok, it["text"]) for it in items[:16]]
    single = [generate(model, tok, [p], 16, 1)[0] for p in prompts]
    batched = generate(model, tok, prompts, 16, 16)
    same = sum(a == b for a, b in zip(single, batched))
    log(f"[eval] left-padding self-test: {same}/16 identical")
    return 16 if same == 16 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--benches", default="all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--delta", default="", help="adapter file; {bench} is substituted")
    ap.add_argument("--meta", default="", help="meta-model checkpoint (arms C, R, S, G)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    manifest = read_json(f"{ROOT}/data/heldout_manifest.json")
    benches = [b for b in manifest if "error" not in manifest[b]] if args.benches == "all" else args.benches.split(",")
    tok, model = load_base()
    targets = dh.find_targets(model)
    hooks = dh.DeltaHooks(targets)
    writer = None
    if args.meta:
        from meta_model import MetaWriter  # written for stage 1/2
        writer = MetaWriter.load(args.meta, targets)
    name = args.arm + (f"_{args.tag}" if args.tag else "")
    out_p = f"{ROOT}/evals/{name}.json"
    res = read_json(out_p) if os.path.exists(out_p) else {}
    bs = None
    derange = {b: benches[(i + 1) % len(benches)] for i, b in enumerate(benches)}  # arm S: fixed derangement
    for b in benches:
        if b in res:
            continue
        t0 = time.time()
        items = [json.loads(l) for l in open(f"{ROOT}/data/heldout/{b}.jsonl")]
        if args.limit:
            items = items[:args.limit]
        pool = [json.loads(l) for l in open(f"{ROOT}/data/heldout/{b}_pool.jsonl")]
        desc = manifest[b]["description"]
        hooks.clear()
        if args.delta:
            hooks.set(dh.load_delta(args.delta.format(bench=b)))
        elif writer is not None:
            src = {"C": desc, "S": manifest[derange[b]]["description"], "G": "Answer the question correctly.",
                   "R": desc}[args.arm]
            delta = writer.write(src)
            if args.arm == "R":
                delta = dh.random_like_norm(delta, seed=0)
            hooks.set(delta)
        if bs is None:
            bs = pick_batch_size(model, tok, items)
        prompts = [prompt_ids(tok, build_user(args.arm, b, it, desc, pool, tok)) for it in items]
        kind = items[0]["kind"]
        replies = generate(model, tok, prompts, MAXNEW[kind], bs if args.arm != "B" else max(1, bs // 4))
        rows, correct, unparsed = [], 0, 0
        for it, rep in zip(items, replies):
            pred, ok = parse_and_score(kind, rep, it)
            correct += ok; unparsed += pred == ""
            rows.append(dict(id=it["id"], reply=rep, pred=pred, gold=it["gold"], ok=bool(ok), shots=it.get("_n_shots")))
        os.makedirs(f"{ROOT}/evals/raw/{name}", exist_ok=True)
        with open(f"{ROOT}/evals/raw/{name}/{b}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        res[b] = dict(score=round(100 * correct / len(items), 2), n=len(items), unparsed=unparsed,
                      seconds=round(time.time() - t0, 1))
        write_json(res, out_p)
        log(f"[eval {name}] {b}: {res[b]['score']:.1f} (n={len(items)}, unparsed {unparsed}) {res[b]['seconds']}s")
    hooks.clear()
    log(f"[eval {name}] mean over {len(res)}: {sum(v['score'] for v in res.values()) / max(len(res), 1):.2f}")


if __name__ == "__main__":
    main()

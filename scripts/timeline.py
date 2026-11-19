"""Append-only results timeline: /workspace/poc/evals/timeline.jsonl.

Row: ts_utc, phase, step, arm, benchmark, split, metric, value, n, ci_low, ci_high, unparsed.
`python timeline.py --sync` backfills every evals/<arm>.json score not yet in the file.
"""
import argparse
import fcntl
import glob
import json
import math
import os
import time

ROOT = "/workspace/poc"
PATH = f"{ROOT}/evals/timeline.jsonl"


def wilson(acc_pct, n, z=1.96):
    if not n:
        return None, None
    p = acc_pct / 100.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 2), round(100 * (c + h), 2)


def append(phase, metric, value, arm=None, benchmark=None, split=None, step=None, n=None,
           ci_low=None, ci_high=None, unparsed=None, ts=None, note=None):
    if metric == "accuracy" and n and ci_low is None:
        ci_low, ci_high = wilson(value, n)
    row = dict(ts_utc=ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), phase=phase, step=step, arm=arm,
               benchmark=benchmark, split=split, metric=metric, value=value, n=n, ci_low=ci_low, ci_high=ci_high,
               unparsed=unparsed, note=note)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def _existing_keys():
    keys = set()
    if os.path.exists(PATH):
        for line in open(PATH):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            keys.add((r["phase"], r["step"], r["arm"], r["benchmark"], r["split"], r["metric"], r.get("note")))
    return keys


def sync(cost):
    keys = _existing_keys()
    added = 0
    for p in sorted(glob.glob(f"{ROOT}/evals/*.json")):
        arm = os.path.basename(p)[:-5]
        if arm.startswith("verdict"):
            continue
        res = json.load(open(p))
        res = {b: v for b, v in res.items() if isinstance(v, dict) and "score" in v}
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(p)))
        phase = ("quick_read" if ("early" in arm or "quick" in arm or arm.endswith("_comve")) else
                 "base_arms" if arm in ("A0", "A", "B") or arm.startswith("L") else "final")
        arm_base, _, tag = arm.partition("_")  # dashboard wants plain arm ids; the tag goes to "note"
        if tag.startswith("dose_t"):  # dose_t<N>_<set>[_s<step>] -> phase dose, note teachers=N; set=...
            parts = tag.split("_")
            phase = "dose"
            extra = []
            for x in parts[3:]:
                if x.startswith("seed"): extra.append(f"seed={x[4:]}")
                elif x.startswith("s2x"): extra.append(f"stage2_steps={x[3:]}")
                elif x == "stage1only": extra.append("ckpt=stage1only")
                elif x.startswith("s") and x[1:].isdigit(): extra.append(f"ckpt=s{x[1:]}")
            tag = f"teachers={parts[1][1:]}; set={parts[2]}" + "".join(f"; {e}" for e in extra)
        for b, v in res.items():
            k = (phase, None, arm_base, b, "test", "accuracy", tag or None)
            if k in keys:
                continue
            append(phase, "accuracy", v["score"], arm=arm_base, benchmark=b, split="test", n=v["n"],
                   unparsed=v.get("unparsed"), ts=ts, note=tag or None)
            added += 1
    tf = next((f for f in ("train_tasks_all_plus_post.json", "train_tasks_all.json", "train_tasks.json")
               if os.path.exists(f"{ROOT}/data/{f}")), None)
    tasks = json.load(open(f"{ROOT}/data/{tf}"))["tasks"] if tf else []
    done = sum(os.path.exists(f"{ROOT}/teachers/{t}/metrics.json") for t in tasks)
    append("teachers", "teachers_done", done, n=len(tasks))
    if cost is not None:
        append("progress", "box_dollars", round(cost, 3))
    print(f"[timeline] backfilled {added} score rows; teachers_done {done}/{len(tasks)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--cost", type=float, default=None)
    a = ap.parse_args()
    if a.sync:
        sync(a.cost)

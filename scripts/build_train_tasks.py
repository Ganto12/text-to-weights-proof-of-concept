"""Build the Track 1 training-task set from Text-to-LoRA's SNI list.

Order: drop held-out sources by name -> shuffle (seed 0) -> download in that order ->
strip to the bare item -> size filters -> keep the first N_KEEP that pass (a uniform
random draw of the passing set). N_KEEP has a margin over 200 because the 13-gram scan
in build_heldout.py may drop more; that script freezes the final 200.
"""
import glob
import json
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

sys.path.insert(0, "/workspace/text-to-weights/scripts")
from common import ROOT, log, write_json  # noqa

N_KEEP = int(os.environ.get("N_KEEP", "230"))  # dose-response run: N_KEEP=999 keeps every passing task
MIN_TRAIN = 300
MIN_VALID = 100
MAX_MEDIAN_ANS_TOK = 128
CAP = dict(train=1000, valid=200, test=200)

# Held-out benchmark sources (PLAN.md 2.1, contamination rule 1). Matched against the SNI task name.
DROP_PATTERNS = [
    r"mmmlu", r"mmlu", r"semeval_2020_task4", r"comve", r"commonsense_validation", r"scitail",
    r"squad", r"xquad", r"svamp", r"obqa", r"openbookqa", r"winogrande", r"\barc\b", r"_arc_",
    r"commonsenseqa", r"commonsense_qa", r"csqa", r"cosmos?_?qa_?csqa", r"toxigen", r"conll", r"gsm8k",
    r"multiarith", r"bigbench", r"big_bench", r"bbh", r"date_understanding", r"boolean_expression",
    r"logical_deduction", r"navigate", r"sports_understanding", r"tracking_shuffled",
]
DROP_RE = re.compile("|".join(DROP_PATTERNS))
# Audit 2026-11-18: dropped by SNI task number after reading SNI "Source" fields.
# 1135 X-CSQA English = CommonsenseQA; 861/865/867 ASDiv/MAWPS (MultiArith, SVAMP seeds); 085/087 AddSub arithmetic;
# 201 MNLI (shares premises with validation task202); 1581 eQASC (same source as validation task039).
DROP_IDS = {"1135", "861", "865", "867", "085", "087", "201", "1581"}
MARK = "Now complete the following example -"


def bare_input(full):
    if MARK in full:
        s = full.split(MARK)[-1]
        how = "marker"
    else:
        s = full
        how = "nomarker"
    s = s.strip()
    if s.startswith("Input:"):
        s = s[len("Input:"):]
    s = re.sub(r"\s*Output:\s*$", "", s)
    return s.strip(), how


def load_task(name, retries=8):
    from datasets import load_dataset
    for k in range(retries):
        try:
            return load_dataset(name)
        except Exception as e:  # noqa
            log(f"[build] {name} attempt {k+1} failed: {type(e).__name__}: {str(e)[:200]}")
            time.sleep(min(120, 10 * 2 ** k))  # HF 429s for unauthenticated clients: back off hard
    return None


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3.5-2B")
    cfg = yaml.safe_load(open(f"{ROOT}/t2l/configs/hyper_lora_decontam_lol_tasks.yaml"))
    lol_ids = cfg["train_ds_names"]
    meta = {}
    for lid in lol_ids:
        m = yaml.safe_load(open(f"{ROOT}/t2l/tasks/{lid}/metadata.yaml"))
        meta[lid] = dict(task=m["task_name"], descriptions=m["descriptions"][:128])
    report = dict(pool=len(lol_ids), dropped_by_name=[], dropped_by_filter={}, marker_fallback={})
    cands = []
    for lid in lol_ids:
        short = meta[lid]["task"].split("/", 1)[1]
        num = re.match(r"task(\d+)_", short).group(1)
        if DROP_RE.search(short) or num in DROP_IDS:
            report["dropped_by_name"].append(short)
        else:
            cands.append(lid)
    log(f"[build] pool {len(lol_ids)}, dropped by name {len(report['dropped_by_name'])}, candidates {len(cands)}")
    random.Random(0).shuffle(cands)

    os.makedirs(f"{ROOT}/data/sni", exist_ok=True)
    kept = []

    def process(lid):
        name = meta[lid]["task"]
        short = name.split("/", 1)[1]
        out_p = f"{ROOT}/data/sni/{short}.json"
        if os.path.exists(out_p):
            return lid, json.load(open(out_p))["_stats"], None
        ds = load_task(name)
        if ds is None:
            return lid, None, "download_failed"
        rec, fallback = {}, 0
        for split in ("train", "valid", "test"):
            rows = []
            if split in ds:
                d = ds[split].shuffle(seed=0).select(range(min(CAP[split], len(ds[split]))))
                for r in d:
                    b, how = bare_input(r["input"])
                    fallback += how != "marker"
                    outs = r["output"] if isinstance(r["output"], list) else [r["output"]]
                    if not b or not outs or not outs[0].strip():
                        continue
                    rows.append(dict(input=b, output=outs[0].strip(), outputs=[o.strip() for o in outs], id=r.get("id")))
            rec[split] = rows
        ans_tok = [len(tok(r["output"], add_special_tokens=False)["input_ids"]) for r in rec["train"][:300]] or [0]
        stats = dict(n_train=len(rec["train"]), n_valid=len(rec["valid"]), n_test=len(rec["test"]),
                     median_answer_tokens=statistics.median(ans_tok), marker_fallback=fallback)
        rec["_stats"] = stats
        rec["task"] = short
        rec["lol_id"] = lid
        rec["descriptions"] = meta[lid]["descriptions"]
        with open(out_p, "w") as f:
            json.dump(rec, f)
        return lid, stats, None

    # Download in the seeded order, 8 at a time, until N_KEEP pass.
    i = 0
    with ThreadPoolExecutor(4) as ex:
        while len(kept) < N_KEEP and i < len(cands):
            chunk = cands[i:i + 16]
            i += 16
            for lid, stats, err in ex.map(process, chunk):
                short = meta[lid]["task"].split("/", 1)[1]
                why = err
                if stats and not why:
                    if stats["n_train"] < MIN_TRAIN: why = "few_train"
                    elif stats["n_valid"] < MIN_VALID: why = "few_valid"
                    elif stats["median_answer_tokens"] > MAX_MEDIAN_ANS_TOK: why = "long_answers"
                    elif len(meta[lid]["descriptions"]) < 128: why = "few_descriptions"
                if why:
                    report["dropped_by_filter"].setdefault(why, []).append(short)
                elif len(kept) < N_KEEP:
                    kept.append(short)
                    if stats["marker_fallback"]:
                        report["marker_fallback"][short] = stats["marker_fallback"]
            log(f"[build] processed {i}/{len(cands)} candidates, kept {len(kept)}, "
                f"filtered {sum(len(v) for v in report['dropped_by_filter'].values())}")
    report["kept_before_ngram_scan"] = kept
    suffix = os.environ.get("OUT_SUFFIX", "")
    write_json(dict(stage="pre_ngram_scan", tasks=kept), f"{ROOT}/data/train_tasks_candidates{suffix}.json")
    write_json(report, f"{ROOT}/data/decontam_report_names{suffix}.json")
    log(f"[build] done: {len(kept)} candidates written; name drops {len(report['dropped_by_name'])}; "
        f"filter drops { {k: len(v) for k, v in report['dropped_by_filter'].items()} }")


if __name__ == "__main__":
    main()

"""Build the 20 held-out benchmarks (PLAN.md 2.2), then scan training tasks for overlap.

Each item: {id, text (item shown to every arm, incl. the fixed format line), gold, kind, scan_text}.
Pools (rows for the in-context arm and LoRA baselines) come from each benchmark's own train
split (BBH: 50 carved-off items). Pools never share an item with the test set (checked).
"""
import io
import json
import os
import random
import re
import string
import sys
import urllib.request

sys.path.insert(0, "/workspace/poc/scripts")
from common import ROOT, log, write_json, read_json  # noqa

N_TEST = 500
N_POOL = 900
SEED = 0
L = "ABCDEFGHIJ"

DESCRIPTIONS = {
    "openbookqa": "Answer an elementary science multiple-choice question that needs one core science fact plus common knowledge, replying with the letter of the correct option only.",
    "arc_easy": "Answer a grade-school science exam multiple-choice question about basic physical, life or earth science, replying with the letter of the correct option only.",
    "arc_challenge": "Answer a hard grade-school science exam multiple-choice question that needs multi-step reasoning, replying with the letter of the correct option only.",
    "commonsenseqa": "Answer a five-option everyday commonsense question about people, objects and situations, replying with the letter of the most sensible option only.",
    "mmlu": "Answer a four-option multiple-choice exam question from one of many academic and professional subjects, replying with the letter of the correct option only.",
    "winogrande": "Read a sentence with a blank and two candidate names or nouns, and reply with the number 1 or 2 of the option that fills the blank sensibly.",
    "comve": "Given two statements that differ in a few words, reply with the number 1 or 2 of the statement that goes against common sense.",
    "scitail": "Decide whether a science premise sentence supports a hypothesis sentence, replying with the single word yes if it does and no if it does not.",
    "toxigen": "Decide whether a short statement about a social or demographic group is hateful or toxic toward that group, replying with the single word yes or no.",
    "bbh_boolean_expressions": "Evaluate a Boolean expression built from True, False, not, and, or and parentheses, replying with True or False only.",
    "bbh_date_understanding": "Answer a multiple-choice question that requires date arithmetic from a short scenario, replying with the letter of the option giving the correct date in parentheses.",
    "bbh_logical_deduction_three_objects": "Use the clues about the order of three objects to answer which option is correct, replying with the option letter in parentheses only.",
    "bbh_navigate": "Follow a list of step and turn instructions and decide whether they bring you back to the starting point, replying with Yes or No only.",
    "bbh_sports_understanding": "Decide whether a sentence about a named athlete doing a sports action is plausible, replying with yes or no only.",
    "bbh_tracking_shuffled_objects_three_objects": "Track which object each of three people holds after a sequence of pairwise swaps, replying with the letter of the correct option in parentheses only.",
    "squad": "Answer a question about a Wikipedia passage by copying the shortest exact span of the passage that answers it, with nothing else.",
    "conll_ner_json": "List the named entities in a news sentence as one JSON object with keys PER, ORG, LOC and MISC, each mapping to a list of strings.",
    "gsm8k_final_number": "Solve a grade-school math word problem and reply with the final number only, on one line, with no working, units or explanation.",
    "svamp_final_number": "Solve a short one- or two-step arithmetic word problem and reply with the final number only, with no working or explanation.",
    "multiarith_final_number": "Solve a multi-step arithmetic word problem about everyday quantities and reply with the final number only, with no working or explanation.",
}


def ld(*args, **kw):
    from datasets import load_dataset
    return load_dataset(*args, **kw)


def mc_item(q, choices, gold_idx, fmt="Answer with the letter."):
    opts = "\n".join(f"{L[i]}. {c}" for i, c in enumerate(choices))
    return dict(text=f"Question: {q}\n{opts}\n{fmt}", gold=L[gold_idx], kind="letter",
                scan_text=q + " " + " ".join(choices))


def b_openbookqa(split):
    d = ld("allenai/openbookqa", "main", split="test" if split == "test" else "train")
    return [dict(id=r["id"], **mc_item(r["question_stem"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]))) for r in d]


def b_arc(cfg):
    def f(split):
        d = ld("allenai/ai2_arc", cfg, split="test" if split == "test" else "train")
        out = []
        for r in d:
            labels = r["choices"]["label"]
            if r["answerKey"] not in labels:
                continue
            out.append(dict(id=r["id"], **mc_item(r["question"], r["choices"]["text"], labels.index(r["answerKey"]))))
        return out
    return f


def b_commonsenseqa(split):
    d = ld("tau/commonsense_qa", split="validation" if split == "test" else "train")
    return [dict(id=r["id"], **mc_item(r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]))) for r in d]


def b_mmlu(split):
    if split == "test":
        d = list(ld("cais/mmlu", "all", split="test"))
    else:
        d = list(ld("cais/mmlu", "all", split="dev")) + list(ld("cais/mmlu", "all", split="validation"))
    return [dict(id=f"{r['subject']}-{i}", **mc_item(r["question"], r["choices"], r["answer"])) for i, r in enumerate(d)]


def b_winogrande(split):
    d = ld("allenai/winogrande", "winogrande_xl", split="validation" if split == "test" else "train")
    return [dict(id=f"wg-{split}-{i}", text=f"{r['sentence']}\n1. {r['option1']}\n2. {r['option2']}\nAnswer with 1 or 2.",
                 gold=r["answer"], kind="digit", scan_text=r["sentence"]) for i, r in enumerate(d)]


COMVE_BASE = "https://raw.githubusercontent.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation/master/ALL%20data/"
COMVE_FILES = {
    "test": ("Test%20Data/subtaskA_test_data.csv", "Test%20Data/subtaskA_gold_answers.csv"),
    "pool": ("Training%20%20Data/subtaskA_data_all.csv", "Training%20%20Data/subtaskA_answers_all.csv"),
}


def _csv(url):
    import csv
    with urllib.request.urlopen(url, timeout=60) as r:
        return list(csv.reader(io.StringIO(r.read().decode("utf-8-sig"))))


def b_comve(split):
    data_f, ans_f = COMVE_FILES["test" if split == "test" else "pool"]
    rows = _csv(COMVE_BASE + data_f)
    ans = {a[0]: a[1] for a in _csv(COMVE_BASE + ans_f) if len(a) >= 2}
    out = []
    for r in rows[1:]:
        if len(r) < 3 or r[0] not in ans:
            continue
        # gold answer = index (0/1) of the statement against common sense -> report as 1/2
        out.append(dict(id=f"comve-{r[0]}", text=f"1. {r[1]}\n2. {r[2]}\nAnswer with 1 or 2.",
                        gold=str(int(ans[r[0]]) + 1), kind="digit", scan_text=r[1] + " " + r[2]))
    return out


def b_scitail(split):
    d = ld("allenai/scitail", "snli_format", split="test" if split == "test" else "train")
    return [dict(id=f"sci-{split}-{i}", text=f"Premise: {r['sentence1']}\nHypothesis: {r['sentence2']}\nAnswer with yes or no.",
                 gold="yes" if r["gold_label"] == "entailment" else "no", kind="yesno",
                 scan_text=r["sentence1"] + " " + r["sentence2"]) for i, r in enumerate(d)]


def b_toxigen(split):
    d = ld("skg/toxigen-data", "annotated", split="test" if split == "test" else "train")
    out = []
    for i, r in enumerate(d):
        lab = (r["toxicity_ai"] + r["toxicity_human"]) > 5.5  # lm-evaluation-harness rule
        out.append(dict(id=f"tox-{split}-{i}", text=f"Statement: {r['text']}\nAnswer with yes or no.",
                        gold="yes" if lab else "no", kind="yesno", scan_text=r["text"]))
    return out


def b_bbh(task):
    def f(split):
        d = list(ld("lukaemon/bbh", task, split="test"))
        idx = list(range(len(d)))
        random.Random(SEED).shuffle(idx)
        pool_idx, test_idx = idx[:100], idx[100:]  # 100 pool rows so L16/L64/L100 never touch test items
        use = test_idx if split == "test" else pool_idx
        return [dict(id=f"{task}-{i}", text=f"{d[i]['input']}\nAnswer with the final answer only.",
                     gold=d[i]["target"], kind="exact", scan_text=d[i]["input"]) for i in use]
    return f


def b_squad(split):
    d = ld("rajpurkar/squad", split="validation" if split == "test" else "train")
    return [dict(id=r["id"], text=f"Passage: {r['context']}\nQuestion: {r['question']}\nAnswer with a span copied from the passage.",
                 gold=r["answers"]["text"][0], golds=r["answers"]["text"], kind="squad",
                 scan_text=r["question"]) for r in d]


NER_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def b_conll(split):
    d = None
    for name in ("eriktks/conll2003", "tner/conll2003"):
        try:
            # both repos are script-based; read Hugging Face's auto-converted parquet branch instead
            d = ld(name, split="test" if split == "test" else "train", revision="refs/convert/parquet")
            log(f"[heldout] conll from {name}")
            break
        except Exception as e:  # noqa
            log(f"[heldout] conll {name} failed: {str(e)[:200]}")
    if d is None:
        raise RuntimeError("no CoNLL source")
    tag_key = "ner_tags" if "ner_tags" in d.column_names else "tags"
    feat = d.features[tag_key]
    names = getattr(getattr(feat, "feature", None), "names", None)
    if not names:  # tner/conll2003 label2id
        tner = {"O": 0, "B-ORG": 1, "B-MISC": 2, "B-PER": 3, "I-PER": 4, "B-LOC": 5, "I-ORG": 6, "I-MISC": 7, "I-LOC": 8}
        names = [k for k, _ in sorted(tner.items(), key=lambda kv: kv[1])]
    log(f"[heldout] conll tag names {names}")
    out = []
    for i, r in enumerate(d):
        ents = {"PER": [], "ORG": [], "LOC": [], "MISC": []}
        cur, typ = [], None
        for w, t in list(zip(r["tokens"], r[tag_key])) + [("", 0)]:
            tag = names[t] if isinstance(t, int) else t
            if tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and tag[2:] != typ):
                if cur:
                    ents[typ].append(" ".join(cur))
                cur, typ = ([w], tag[2:]) if tag != "O" else ([], None)
            else:
                cur.append(w)
        if not any(ents.values()):
            continue
        sent = " ".join(r["tokens"])
        out.append(dict(id=f"conll-{split}-{i}", text=f"Sentence: {sent}\nAnswer with a JSON object.",
                        gold=json.dumps(ents), kind="ner_json", scan_text=sent))
    return out


def _num(s):
    s = str(s).replace(",", "").strip()
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def b_gsm8k(split):
    d = ld("openai/gsm8k", "main", split="test" if split == "test" else "train")
    return [dict(id=f"gsm-{split}-{i}", text=f"Question: {r['question']}\nAnswer with the number only.",
                 gold=_num(r["answer"].split("####")[-1]), kind="number", scan_text=r["question"]) for i, r in enumerate(d)]


def b_svamp(split):
    d = ld("ChilleD/SVAMP", split="test" if split == "test" else "train")
    out = []
    for i, r in enumerate(d):
        q = r.get("question_concat") or (r["Body"].strip() + " " + r["Question"].strip())
        out.append(dict(id=f"svamp-{split}-{i}", text=f"Question: {q}\nAnswer with the number only.",
                        gold=_num(r["Answer"]), kind="number", scan_text=q))
    return out


def b_multiarith(split):
    d = ld("ChilleD/MultiArith", split="test" if split == "test" else "train")
    return [dict(id=f"ma-{split}-{i}", text=f"Question: {r['question']}\nAnswer with the number only.",
                 gold=_num(r["final_ans"]), kind="number", scan_text=r["question"]) for i, r in enumerate(d)]


BENCH = {
    "openbookqa": b_openbookqa, "arc_easy": b_arc("ARC-Easy"), "arc_challenge": b_arc("ARC-Challenge"),
    "commonsenseqa": b_commonsenseqa, "mmlu": b_mmlu, "winogrande": b_winogrande, "comve": b_comve,
    "scitail": b_scitail, "toxigen": b_toxigen,
    "bbh_boolean_expressions": b_bbh("boolean_expressions"), "bbh_date_understanding": b_bbh("date_understanding"),
    "bbh_logical_deduction_three_objects": b_bbh("logical_deduction_three_objects"), "bbh_navigate": b_bbh("navigate"),
    "bbh_sports_understanding": b_bbh("sports_understanding"),
    "bbh_tracking_shuffled_objects_three_objects": b_bbh("tracking_shuffled_objects_three_objects"),
    "squad": b_squad, "conll_ner_json": b_conll, "gsm8k_final_number": b_gsm8k,
    "svamp_final_number": b_svamp, "multiarith_final_number": b_multiarith,
}

_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def words(s):
    return s.lower().translate(_PUNCT).split()


def ngrams(ws, n=13):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def main():
    os.makedirs(f"{ROOT}/data/heldout", exist_ok=True)
    manifest = {}
    for b, fn in BENCH.items():
        p_test, p_pool = f"{ROOT}/data/heldout/{b}.jsonl", f"{ROOT}/data/heldout/{b}_pool.jsonl"
        if os.path.exists(p_test) and os.path.exists(p_pool):
            manifest[b] = read_json(f"{ROOT}/data/heldout/{b}.meta.json")
            continue
        try:
            test = fn("test")
            try:
                pool = fn("pool")
            except Exception as e:  # noqa
                log(f"[heldout] WARN {b}: pool failed ({str(e)[:200]}); arms B and L will skip it")
                pool = []
        except Exception as e:  # noqa
            log(f"[heldout] FAILED {b}: {type(e).__name__}: {str(e)[:300]}")
            manifest[b] = dict(error=str(e)[:300])
            continue
        rng = random.Random(SEED)
        rng.shuffle(test)
        test = test[:N_TEST]
        test_keys = {" ".join(words(t["scan_text"])) for t in test}
        pool = [r for r in pool if " ".join(words(r["scan_text"])) not in test_keys]
        random.Random(SEED + 1).shuffle(pool)
        pool = pool[:N_POOL]
        desc = DESCRIPTIONS[b]
        nw = len(desc.split())
        assert 14 <= nw <= 31, (b, nw)
        with open(p_test, "w") as f:
            for r in test:
                f.write(json.dumps(r) + "\n")
        with open(p_pool, "w") as f:
            for r in pool:
                f.write(json.dumps(r) + "\n")
        manifest[b] = dict(n_test=len(test), n_pool=len(pool), description=desc, description_words=nw,
                           gold_examples=[t["gold"] for t in test[:3]])
        write_json(manifest[b], f"{ROOT}/data/heldout/{b}.meta.json")
        log(f"[heldout] {b}: test {len(test)} pool {len(pool)} desc {nw} words; e.g. gold {manifest[b]['gold_examples']}")
    write_json(manifest, f"{ROOT}/data/heldout_manifest.json")
    missing = [b for b, m in manifest.items() if "error" in m]
    log(f"[heldout] benchmarks ok {len(manifest) - len(missing)}/20, missing {missing}")

    # 13-gram scan of training-task candidates against held-out TEST items, plus
    # exact match of whole held-out questions of 6+ words (short items escape 13-grams).
    # n -> set of n-grams. 13-grams from every item; short items (6-12 words) add 8-grams,
    # or their whole text as a 6- or 7-gram, so they cannot slip under the 13-gram window.
    gsets = {13: set(), 8: set(), 7: set(), 6: set()}
    for b in BENCH:
        p = f"{ROOT}/data/heldout/{b}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            ws = words(json.loads(line)["scan_text"])
            gsets[13] |= ngrams(ws, 13)
            if 8 <= len(ws) < 13:
                gsets[8] |= ngrams(ws, 8)
            elif len(ws) in (6, 7):
                gsets[len(ws)].add(" ".join(ws))
    suffix = os.environ.get("OUT_SUFFIX", "")
    cap = int(os.environ.get("MAX_TASKS", "200"))
    cands = read_json(f"{ROOT}/data/train_tasks_candidates{suffix}.json")["tasks"]
    scan, final = {}, []
    for t in cands:
        p = f"{ROOT}/data/sni/{t}.json"
        rec = read_json(p)
        hits = 0
        for split in ("train", "valid", "test"):
            keep = []
            for r in rec[split]:
                ws = words(r["input"] + " " + r["output"])
                hit = any(gs and (ngrams(ws, n) & gs) for n, gs in gsets.items())
                if hit:
                    hits += 1
                else:
                    keep.append(r)
            rec[split] = keep
        total = sum(len(rec[s]) for s in ("train", "valid", "test")) + hits
        frac = hits / max(total, 1)
        scan[t] = dict(hits=hits, rows=total, frac=round(frac, 4))
        if frac > 0.01:
            continue
        if hits:
            rec["_stats"]["ngram_rows_removed"] = rec["_stats"].get("ngram_rows_removed", 0) + hits
            write_json(rec, p)
            mp = f"{ROOT}/teachers/{t}/metrics.json"
            if os.path.exists(mp):
                os.replace(mp, mp.replace("metrics.json", "metrics_prescan_stale.json"))
                log(f"[heldout] {t}: {hits} rows removed; teacher marked stale for retraining")
        if len(rec["train"]) >= 300 and len(rec["valid"]) >= 100 and len(final) < cap:
            final.append(t)
    write_json(dict(tasks=final, n=len(final), seed=0, note="first 200 candidates (seeded order) that survive the 13-gram scan"),
               f"{ROOT}/data/train_tasks{suffix}.json")
    rep = read_json(f"{ROOT}/data/decontam_report_names{suffix}.json")
    rep["ngram_scan"] = scan
    rep["ngram_dropped_tasks"] = [t for t, s in scan.items() if s["frac"] > 0.01]
    rep["final_tasks"] = len(final)
    write_json(rep, f"{ROOT}/data/decontam_report{suffix}.json")
    log(f"[heldout] scan: {len(rep['ngram_dropped_tasks'])} tasks dropped (>1% hit), "
        f"{sum(1 for s in scan.values() if 0 < s['frac'] <= 0.01)} tasks had hit rows removed; final {len(final)} tasks")


if __name__ == "__main__":
    main()

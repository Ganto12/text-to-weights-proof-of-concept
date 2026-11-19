"""Shared helpers: model loading, chat formatting, batching, answer-token loss."""
import json
import os
import random
import time

import torch

ROOT = "/workspace/poc"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf_cache")
MODEL_DIR = f"{ROOT}/models/Qwen3.5-2B"
MAX_LEN = 512


def log(*a):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}]", *a, flush=True)


def load_base(grad_ckpt=False):
    from transformers import AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
            log(f"[load_base] loaded with {cls_name}: {type(model).__name__}")
            break
        except Exception as e:  # noqa
            log(f"[load_base] {cls_name} failed: {type(e).__name__}: {str(e)[:300]}")
    if model is None:
        raise RuntimeError("could not load Qwen3.5-2B")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


def prompt_ids(tok, user_text):
    msgs = [{"role": "user", "content": user_text}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False)["input_ids"]


def answer_ids(tok, answer):
    end = "<|im_end|>"
    return tok(answer + end, add_special_tokens=False)["input_ids"]


def encode_rows(tok, rows, max_len=MAX_LEN):
    """rows: [{'input','output'}] -> list of (ids, n_prompt). Drops rows over max_len."""
    out, dropped = [], 0
    for r in rows:
        p = prompt_ids(tok, r["input"])
        a = answer_ids(tok, r["output"])
        if len(p) + len(a) > max_len:
            dropped += 1
            continue
        out.append((p + a, len(p)))
    return out, dropped


def length_grouped_batches(encoded, bs, seed, group=64):
    idx = list(range(len(encoded)))
    random.Random(seed).shuffle(idx)
    batches = []
    for g0 in range(0, len(idx), group):
        g = sorted(idx[g0:g0 + group], key=lambda i: len(encoded[i][0]))
        batches += [g[i:i + bs] for i in range(0, len(g), bs)]
    random.Random(seed + 1).shuffle(batches)
    return batches


def collate(encoded, ids, pad_id, device="cuda"):
    L = max(len(encoded[i][0]) for i in ids)
    inp = torch.full((len(ids), L), pad_id, dtype=torch.long)
    lab = torch.full((len(ids), L), -100, dtype=torch.long)
    att = torch.zeros((len(ids), L), dtype=torch.long)
    for j, i in enumerate(ids):
        seq, n_p = encoded[i]
        inp[j, :len(seq)] = torch.tensor(seq)
        lab[j, n_p:len(seq)] = torch.tensor(seq[n_p:])
        att[j, :len(seq)] = 1
    return inp.to(device), lab.to(device), att.to(device)


def _answer_logits(model, inp, lab, att):
    """Run the decoder, then the LM head only on positions that predict answer tokens."""
    h = model.model(input_ids=inp, attention_mask=att, use_cache=False)
    h = h.last_hidden_state if hasattr(h, "last_hidden_state") else h[0]
    tgt = lab[:, 1:]
    mask = tgt != -100
    logits = model.get_output_embeddings()(h[:, :-1][mask]).float()
    return logits, tgt[mask]


def answer_loss(model, inp, lab, att):
    """Mean cross-entropy over answer tokens (right padding, so causal context is clean)."""
    logits, tgt = _answer_logits(model, inp, lab, att)
    return torch.nn.functional.cross_entropy(logits, tgt)


@torch.no_grad()
def mean_answer_loss(model, encoded, pad_id, bs=16):
    tot, n = 0.0, 0
    for b0 in range(0, len(encoded), bs):
        ids = list(range(b0, min(b0 + bs, len(encoded))))
        inp, lab, att = collate(encoded, ids, pad_id)
        logits, tgt = _answer_logits(model, inp, lab, att)
        l = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
        tot += l.item(); n += tgt.numel()
    return tot / max(n, 1)


def read_json(p):
    with open(p) as f:
        return json.load(f)


def write_json(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, p)

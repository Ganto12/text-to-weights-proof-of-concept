"""Shared low-rank weight-update machinery for teachers, LoRA baselines and the meta-model.

An update is a dict {module_name: (A, B)} with A: (r, d_in), B: (d_out, r).
Effective change: dW = scale * B @ A, scale = alpha / r = 16 / 8 = 2.
Training and evaluation add it through forward hooks (exact). merge() is only used
for the merged-weights equivalence check.
"""
import math
import re

import torch
import torch.nn as nn

RANK = 8
ALPHA = 16
SCALE = ALPHA / RANK

TYPES = [
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
    "linear_attn.in_proj_b", "linear_attn.out_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
_PAT = re.compile(r"layers\.(\d+)\.(" + "|".join(re.escape(t) for t in TYPES) + r")$")
EXPECTED_TARGETS = 186  # 18 linear-attention layers x 8 + 6 full-attention layers x 7


def find_targets(model):
    out = {}
    for name, mod in model.named_modules():
        if any(s in name for s in ("visual", "vision", "mtp")):
            continue
        m = _PAT.search(name)
        if m and isinstance(mod, nn.Linear):
            out[name] = dict(module=mod, layer=int(m.group(1)), type=TYPES.index(m.group(2)),
                             din=mod.in_features, dout=mod.out_features)
    if len(out) != EXPECTED_TARGETS:
        raise RuntimeError(f"[delta_hooks] found {len(out)} target matrices, expected {EXPECTED_TARGETS}")
    return out


def type_shapes(targets):
    """{type_index: (din, dout)} - checks every layer agrees."""
    shapes = {}
    for t in targets.values():
        s = (t["din"], t["dout"])
        if shapes.setdefault(t["type"], s) != s:
            raise RuntimeError(f"[delta_hooks] inconsistent shape for type {TYPES[t['type']]}")
    return shapes


class DeltaHooks:
    def __init__(self, targets, scale=SCALE):
        self.targets = targets
        self.scale = scale
        self.delta = {}
        self.handles = [t["module"].register_forward_hook(self._make(n)) for n, t in targets.items()]

    def _make(self, name):
        def hook(mod, inp, out):
            d = self.delta.get(name)
            if d is None:
                return out
            A, B = d
            x = inp[0]
            return out + self.scale * ((x @ A.to(x.dtype).t()) @ B.to(x.dtype).t())
        return hook

    def set(self, delta):
        self.delta = delta

    def clear(self):
        self.delta = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def new_lora(targets, rank=RANK, device="cuda", seed=0):
    """Standard LoRA init: A Kaiming-uniform, B zero. fp32 trainable leaves."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    delta = {}
    for n, t in targets.items():
        A = torch.empty(rank, t["din"])
        bound = 1.0 / math.sqrt(t["din"])  # kaiming_uniform with a=sqrt(5)
        A.uniform_(-bound, bound, generator=g)
        B = torch.zeros(t["dout"], rank)
        delta[n] = (A.to(device).requires_grad_(True), B.to(device).requires_grad_(True))
    return delta


def delta_params(delta):
    return [p for ab in delta.values() for p in ab]


@torch.no_grad()
def frob_norms(delta, scale=SCALE):
    """Exact ||scale * B A||_F per matrix via r x r Gram matrices."""
    out = {}
    for n, (A, B) in delta.items():
        A = A.float(); B = B.float()
        out[n] = scale * torch.sqrt(torch.clamp(torch.trace((B.t() @ B) @ (A @ A.t())), min=0)).item()
    return out


@torch.no_grad()
def random_like_norm(delta, seed, scale=SCALE, rank=RANK):
    """Random rank-r update with the same Frobenius norm per matrix as `delta`."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    target = frob_norms(delta, scale)
    out = {}
    for n, (A, B) in delta.items():
        Ar = torch.randn(rank, A.shape[1], generator=g)
        Br = torch.randn(B.shape[0], rank, generator=g)
        cur = scale * torch.sqrt(torch.trace((Br.t() @ Br) @ (Ar @ Ar.t()))).item()
        k = math.sqrt(target[n] / cur) if cur > 0 else 0.0
        out[n] = ((Ar * k).to(A.device, A.dtype), (Br * k).to(B.device, B.dtype))
    return out


def save_delta(delta, path):
    from safetensors.torch import save_file
    t = {}
    for n, (A, B) in delta.items():
        t[n + ".A"] = A.detach().to(torch.bfloat16).cpu().contiguous()
        t[n + ".B"] = B.detach().to(torch.bfloat16).cpu().contiguous()
    save_file(t, path)


def load_delta(path, device="cuda", dtype=torch.float32):
    from safetensors.torch import load_file
    t = load_file(path)
    names = sorted({k[:-2] for k in t})
    return {n: (t[n + ".A"].to(device, dtype), t[n + ".B"].to(device, dtype)) for n in names}


@torch.no_grad()
def merge(targets, delta, scale=SCALE, sign=1.0):
    for n, (A, B) in delta.items():
        W = targets[n]["module"].weight
        W.add_((sign * scale * (B.float() @ A.float())).to(W.dtype))

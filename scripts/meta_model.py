"""Meta-model: description embedding -> rank-8 update for all 192 projection matrices.

Text-to-LoRA-style (PLAN.md section 4): frozen gte-large-en-v1.5 CLS vector -> trunk with
layer and projection-type embeddings -> one head per projection type emitting A and B.
B-part of every head starts at zero, so the first written update is exactly zero.
"""
import torch
import torch.nn as nn

import delta_hooks as dh

GTE_DIR = "/workspace/poc/models/bge-large-en-v1.5"  # gte-large remote code breaks on transformers 5; bge is native BERT, 1024-d CLS


def _fix_gte_buffers(model):
    """transformers 5 loads remote-code models on the meta device, so gte's non-persistent
    buffers (position_ids, rotary inv_freq / cos / sin caches) come back as garbage.
    Rebuild them exactly as the model's own constructors do."""
    emb = model.embeddings
    n = emb.position_ids.shape[-1]
    emb.register_buffer("position_ids", torch.arange(n), persistent=False)
    rot = emb.rotary_emb
    kw = dict(scaling_factor=rot.scaling_factor, mixed_b=rot.mixed_b) if hasattr(rot, "scaling_factor") else {}
    fresh = type(rot)(rot.dim, rot.max_position_embeddings, rot.base, **kw)
    for name, buf in fresh.named_buffers():
        rot.register_buffer(name, buf.clone(), persistent=False)


class Encoder:
    """Frozen description encoder (CLS of last layer, 1024-d)."""

    def __init__(self, device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(GTE_DIR)
        self.model = AutoModel.from_pretrained(GTE_DIR).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            out.append(self.model(**b).last_hidden_state[:, 0].float())
        return torch.cat(out)


class Block(nn.Module):
    def __init__(self, d, drop=0.05):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Dropout(drop), nn.Linear(d, d))

    def forward(self, x):
        return x + self.net(x)


class MetaModel(nn.Module):
    def __init__(self, shapes, n_layers=24, rank=dh.RANK, d_emb=1024, d_task=256, d_hidden=512, d_out=256):
        super().__init__()
        self.shapes = {int(k): tuple(v) for k, v in shapes.items()}  # type -> (din, dout)
        self.rank = rank
        self.task_proj = nn.Sequential(nn.Linear(d_emb, d_task), nn.LayerNorm(d_task))
        self.layer_emb = nn.Embedding(n_layers, 32)
        self.type_emb = nn.Embedding(len(dh.TYPES), 32)
        self.inp = nn.Linear(d_task + 64, d_hidden)
        self.blocks = nn.Sequential(Block(d_hidden), Block(d_hidden))
        self.out = nn.Sequential(nn.LayerNorm(d_hidden), nn.Linear(d_hidden, d_out), nn.SiLU())
        self.headA = nn.ModuleDict()
        self.headB = nn.ModuleDict()
        for t, (din, dout) in self.shapes.items():
            hA = nn.Linear(d_out, rank * din)
            nn.init.normal_(hA.weight, std=1e-3)
            nn.init.uniform_(hA.bias, -din ** -0.5, din ** -0.5)  # Bias-HyperInit: same bound as the teachers' A init
            hB = nn.Linear(d_out, dout * rank)
            nn.init.zeros_(hB.weight); nn.init.zeros_(hB.bias)
            self.headA[str(t)] = hA
            self.headB[str(t)] = hB

    def forward_stacked(self, embs, lay):
        """embs (T, 1024); lay = layout(targets). Returns {type: (A (T,k,r,din), B (T,k,dout,r))}."""
        T = embs.shape[0]
        dev = embs.device
        types = [t for t, items in lay.items() for _ in items]
        layers = [l for items in lay.values() for _, l in items]
        ty = torch.tensor(types, device=dev)
        ly = torch.tensor(layers, device=dev)
        M = len(types)
        task = self.task_proj(embs)
        h = torch.cat([task[:, None, :].expand(T, M, -1), self.layer_emb(ly)[None].expand(T, -1, -1),
                       self.type_emb(ty)[None].expand(T, -1, -1)], dim=-1)
        h = self.out(self.blocks(self.inp(h)))
        out, i = {}, 0
        for t, items in lay.items():
            k = len(items)
            din, dout = self.shapes[t]
            ht = h[:, i:i + k]
            i += k
            out[t] = (self.headA[str(t)](ht).view(T, k, self.rank, din), self.headB[str(t)](ht).view(T, k, dout, self.rank))
        return out

    def forward(self, emb, targets):
        """emb: (1024,) one task. Returns {name: (A, B)} for every target matrix."""
        lay = layout(targets)
        st = self.forward_stacked(emb.unsqueeze(0), lay)
        delta = {}
        for t, items in lay.items():
            A, B = st[t]
            for j, (n, _) in enumerate(items):
                delta[n] = (A[0, j], B[0, j])
        return delta


def layout(targets):
    """{type: [(name, layer), ...]} sorted by type then layer; the fixed matrix order for stacking."""
    lay = {}
    for n, t in targets.items():
        lay.setdefault(t["type"], []).append((n, t["layer"]))
    return {k: sorted(v, key=lambda x: x[1]) for k, v in sorted(lay.items())}


def stack_delta(delta, lay, device="cuda", dtype=torch.bfloat16):
    """Teacher update -> {type: (A (k,r,din), B (k,dout,r))} on device, plus ||BA||^2 per matrix."""
    st, tt = {}, {}
    for t, items in lay.items():
        A = torch.stack([delta[n][0] for n, _ in items]).to(device, torch.float32)
        B = torch.stack([delta[n][1] for n, _ in items]).to(device, torch.float32)
        tt[t] = ((B.transpose(-1, -2) @ B) * (A @ A.transpose(-1, -2)).transpose(-1, -2)).sum((-1, -2))
        st[t] = (A.to(dtype), B.to(dtype))
    return st, tt


def stacked_rel_loss(pred, tgt, tt):
    """Mean over tasks and matrices of ||B'A' - BA||^2 / ||BA||^2 (exact, r x r Gram matrices)."""
    tot, cnt = 0.0, 0
    for t, (Ap, Bp) in pred.items():
        A, B = tgt[t]
        A = A.float(); B = B.float()
        pp = ((Bp.transpose(-1, -2) @ Bp) * (Ap @ Ap.transpose(-1, -2)).transpose(-1, -2)).sum((-1, -2))
        pt = ((B.transpose(-1, -2) @ Bp) * (Ap @ A.transpose(-1, -2)).transpose(-1, -2)).sum((-1, -2))
        l = (pp - 2 * pt + tt[t]) / tt[t].clamp_min(1e-12)
        tot = tot + l.sum()
        cnt += l.numel()
    return tot / cnt


def product_rel_loss(pred, target, scale=dh.SCALE):
    """Mean over matrices of ||B'A' - BA||^2 / ||BA||^2, exact via r x r Gram matrices."""
    losses = []
    for n, (A, B) in target.items():
        Ap, Bp = pred[n]
        Ap = Ap.float(); Bp = Bp.float(); A = A.float(); B = B.float()
        tt = torch.trace((B.t() @ B) @ (A @ A.t()))
        pp = torch.trace((Bp.t() @ Bp) @ (Ap @ Ap.t()))
        pt = torch.trace((Bp.t() @ B) @ (A @ Ap.t()))
        losses.append((pp - 2 * pt + tt) / tt.clamp_min(1e-12))
    return torch.stack(losses).mean()


class MetaWriter:
    """Inference wrapper used by eval_arms.py."""

    def __init__(self, model, encoder, targets):
        self.model, self.encoder, self.targets = model, encoder, targets

    @classmethod
    def load(cls, path, targets):
        ck = torch.load(path, map_location="cuda")
        m = MetaModel(ck["shapes"]).cuda()
        m.load_state_dict(ck["state_dict"])
        m.eval()
        w = cls(m, Encoder(), targets)
        w.clamp_norm = ck.get("clamp_norm")
        return w

    clamp_norm = None

    @torch.no_grad()
    def write(self, text):
        emb = self.encoder([text])[0]
        d = {n: (A.detach(), B.detach()) for n, (A, B) in self.model(emb, self.targets).items()}
        if self.clamp_norm:
            sq = sum(float(torch.trace((B.float().t() @ B.float()) @ (A.float() @ A.float().t()))) for A, B in d.values())
            norm = dh.SCALE * sq ** 0.5
            if norm > self.clamp_norm:
                k = self.clamp_norm / norm
                d = {n: (A, B * k) for n, (A, B) in d.items()}
        return d

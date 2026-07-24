#!/usr/bin/env python3
"""PyTorch control arm for ddp-bench's `olmo` model.

Mirrors ddp-bench/src/models/olmo.rs exactly — architecture, deviations
included (MultiheadAttention projection biases, trimmed seq, the staged
olmo-mix slice) — so the flodl-vs-PyTorch comparison is self-controlled.
AI2 publishes no loss curves for the tiny OLMo configs; this script IS
the reference curve.

Run after ddp-bench has staged the shards (any `--model olmo` run):

    python scripts/olmo_control.py [--data-dir data] [--epochs 3]

Prints per-epoch mean train CE and held-out C4-en eval CE (same eval
data as the Rust side).
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants mirroring models/olmo.rs (keep in sync by hand).
D_MODEL = 768
N_HEADS = 12
HEAD_DIM = D_MODEL // N_HEADS
N_LAYERS = 12
MLP_HIDDEN = 8 * D_MODEL  # gate-split to 3072
EMBEDDING_SIZE = 50_304
LN_EPS = 1e-6
ROPE_THETA = 10_000.0
SEQ_LEN = 256
LR = 6.0e-4
BATCH_SIZE = 4


class Rope:
    """Duplicated-halves rotate-half tables, matching flodl's RotaryEmbedding."""

    def __init__(self, head_dim, max_seq, device):
        half = head_dim // 2
        inv_freq = ROPE_THETA ** (-2.0 * np.arange(half) / head_dim)
        angles = np.outer(np.arange(max_seq), inv_freq)  # [seq, half]
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=1)
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=1)
        self.cos = torch.tensor(cos, dtype=torch.float32, device=device)
        self.sin = torch.tensor(sin, dtype=torch.float32, device=device)

    def apply(self, x):  # x: [B, H, S, D]
        s = x.shape[2]
        cos = self.cos[:s].view(1, 1, s, -1)
        sin = self.sin[:s].view(1, 1, s, -1)
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * sin


class Block(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.rope = rope
        self.attn_norm = nn.RMSNorm(D_MODEL, eps=LN_EPS)
        # bias=True intentionally: flodl's MultiheadAttention carries
        # projection biases (documented deviation from the OLMo config).
        self.q = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.k = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.v = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.o = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.ff_norm = nn.RMSNorm(D_MODEL, eps=LN_EPS)
        self.ff_proj = nn.Linear(D_MODEL, MLP_HIDDEN, bias=False)
        self.ff_out = nn.Linear(MLP_HIDDEN // 2, D_MODEL, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        h = self.attn_norm(x)
        q = self.q(h).view(b, s, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(h).view(b, s, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(h).view(b, s, N_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = self.rope.apply(q), self.rope.apply(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(b, s, D_MODEL)
        x = x + self.o(a)

        h = self.ff_norm(x)
        proj = self.ff_proj(h)
        val, gate = proj.chunk(2, dim=-1)  # OLMo: x, gate = chunk(2)
        return x + self.ff_out(F.silu(gate) * val)


class Olmo(nn.Module):
    def __init__(self, device):
        super().__init__()
        rope = Rope(HEAD_DIM, SEQ_LEN, device)
        self.emb = nn.Embedding(EMBEDDING_SIZE, D_MODEL)
        self.blocks = nn.ModuleList(Block(rope) for _ in range(N_LAYERS))
        self.final_norm = nn.RMSNorm(D_MODEL, eps=LN_EPS)
        self.head = nn.Linear(D_MODEL, EMBEDDING_SIZE, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.final_norm(x))


def windows(path):
    """Non-overlapping SEQ_LEN windows with shift-by-one targets, like TokenShards."""
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    n = (len(tokens) - 1) // SEQ_LEN
    return tokens, n


def batch_at(tokens, order, step, device):
    idx = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
    xs = np.stack([tokens[i * SEQ_LEN:i * SEQ_LEN + SEQ_LEN] for i in idx]).astype(np.int64)
    ys = np.stack([tokens[i * SEQ_LEN + 1:i * SEQ_LEN + SEQ_LEN + 1] for i in idx]).astype(np.int64)
    return (torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="ddp-bench data dir (holds olmo/)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_path = Path(args.data_dir) / "olmo" / "books-part-0-00000.head.npy"
    eval_path = Path(args.data_dir) / "olmo" / "c4-val-part-0-00000.head.npy"
    for p in (train_path, eval_path):
        if not p.exists():
            sys.exit(f"missing {p} — run any `ddp-bench --model olmo` first to stage the shards")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", file=sys.stderr)

    model = Olmo(device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.1f}M", file=sys.stderr)

    train_tokens, n_train = windows(train_path)
    eval_tokens, n_eval = windows(eval_path)
    steps = n_train // BATCH_SIZE  # drop_last, matching the Rust side
    total = steps * args.epochs

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    warmup = max(total // 20, 1)

    def lr_at(step):  # 5% linear warmup, cosine to 0.1x — mirrors olmo.rs
        if step < warmup:
            return LR * (step + 1) / warmup
        t = (step - warmup) / max(total - warmup, 1)
        return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    rng = np.random.RandomState(args.seed)
    step_no = 0
    for epoch in range(args.epochs):
        order = rng.permutation(n_train)
        model.train()
        losses = []
        t0 = time.time()
        for s in range(steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step_no)
            x, y = batch_at(train_tokens, order, s, device)
            loss = F.cross_entropy(model(x).view(-1, EMBEDDING_SIZE), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step_no += 1

        model.eval()
        eval_losses = []
        eval_order = np.arange(n_eval)
        with torch.no_grad():
            for s in range(n_eval // BATCH_SIZE):
                x, y = batch_at(eval_tokens, eval_order, s, device)
                l = F.cross_entropy(model(x).view(-1, EMBEDDING_SIZE), y.view(-1))
                eval_losses.append(l.item())

        print(f"epoch {epoch}: train_ce={np.mean(losses):.4f} "
              f"eval_ce={np.mean(eval_losses):.4f} "
              f"({time.time() - t0:.1f}s, {steps} batches)")

    print(f"done: eval={np.mean(eval_losses):.4f}")


if __name__ == "__main__":
    main()

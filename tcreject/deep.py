"""Learned-representation channel.

A small 1D-CNN encoder is pre-trained *without labels* (SimCLR / NT-Xent) on raw
windows, then its embedding is scored exactly like the hand-crafted feature
spaces (density scoring with the same conformal machinery). The encoder is
retrained in every fold on windows outside that fold's held-out groups, so a
held-out group's own signal never influences the representation used to score it.

PyTorch is imported lazily; the rest of the package works without it.
A custom ``embedder`` (callable ``(W_pretrain, seed) -> (W -> embeddings)``) can be
injected for testing.
"""
import numpy as np

from .conformal import channel_pvalues, Fold


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError as e:  # pragma: no cover
        raise ImportError("The deep channel requires PyTorch (pip install torch).") from e


def resolve_device(device):
    torch, _, _ = _torch()
    if device != "auto":
        return device
    if torch.cuda.is_available():
        print(f"    [deep] using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("    [deep] no CUDA device found, using CPU")
    return "cpu"


def _build_encoder(in_channels, embed_dim):
    torch, nn, F = _torch()

    class TSEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, embed_dim, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(embed_dim), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.embed_dim = embed_dim
            self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                      nn.Linear(embed_dim, embed_dim))

        def forward(self, x, project=False):
            z = self.backbone(x).squeeze(-1)
            if project:
                return F.normalize(self.proj(z), dim=1)
            return z

    return TSEncoder()


def _augment(x, rng):
    """x: (B, T, C). Jitter, per-window/channel scaling, channel dropout, light time-warp."""
    B, T, C = x.shape
    out = x.copy()
    out = out + rng.normal(0, 0.05 * (out.std() + 1e-6), size=out.shape)
    scale = rng.normal(1.0, 0.1, size=(B, 1, C))
    out = out * scale
    if C > 1:
        mask = rng.random((B, 1, C)) > 0.1
        out = out * mask
    warp_frac = rng.uniform(0.9, 1.0, size=B)
    warped = np.empty_like(out)
    for i in range(B):
        span = max(int(T * warp_frac[i]), 8)
        start = rng.integers(0, T - span + 1)
        seg = out[i, start:start + span, :]
        idx = np.linspace(0, span - 1, T)
        lo = np.floor(idx).astype(int).clip(0, span - 1)
        hi = np.ceil(idx).astype(int).clip(0, span - 1)
        frac = (idx - lo)[:, None]
        warped[i] = seg[lo] * (1 - frac) + seg[hi] * frac
    return warped


def _nt_xent(z1, z2, temperature):
    torch, _, F = _torch()
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], dim=0)
    return F.cross_entropy(sim, targets)


def train_encoder(W, cfg, seed=0, device="cpu", verbose=False):
    """Self-supervised (label-free) contrastive pre-training on raw windows W: (N, T, C)."""
    torch, _, _ = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    N, T, C = W.shape
    enc = _build_encoder(C, cfg.deep_embed_dim).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=cfg.deep_lr)
    enc.train()
    idx_all = np.arange(N)
    for ep in range(cfg.deep_epochs):
        rng.shuffle(idx_all)
        losses = []
        for s in range(0, N, cfg.deep_batch_size):
            b = idx_all[s:s + cfg.deep_batch_size]
            if len(b) < 4:
                continue
            xb = W[b]
            t1 = torch.tensor(_augment(xb, rng), dtype=torch.float32, device=device).transpose(1, 2)
            t2 = torch.tensor(_augment(xb, rng), dtype=torch.float32, device=device).transpose(1, 2)
            loss = _nt_xent(enc(t1, project=True), enc(t2, project=True), cfg.deep_temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if verbose and losses:
            print(f"      [deep encoder] epoch {ep + 1}/{cfg.deep_epochs} loss={np.mean(losses):.3f}")
    enc.eval()
    return enc


def embed(enc, W, device="cpu", batch_size=256):
    torch, _, _ = _torch()
    if len(W) == 0:
        return np.zeros((0, enc.embed_dim))
    outs = []
    with torch.no_grad():
        for s in range(0, len(W), batch_size):
            t = torch.tensor(W[s:s + batch_size], dtype=torch.float32, device=device).transpose(1, 2)
            outs.append(enc(t, project=False).cpu().numpy())
    return np.concatenate(outs, axis=0)


def torch_embedder(cfg, device="auto", verbose=False):
    """Default embedder: trains a fresh encoder on W_pretrain and returns W -> embeddings."""
    dev = resolve_device(device)

    def make(W_pretrain, seed):
        enc = train_encoder(W_pretrain, cfg, seed=seed, device=dev, verbose=verbose)
        return lambda W: embed(enc, W, device=dev)

    return make


def deep_pvalues(W, y, folds, groups, cfg, embedder, seed=0, verbose=True):
    """Out-of-fold conformal p-values on the learned embedding.

    For every fold an encoder is pre-trained on all windows outside the fold's
    held-out groups, fit windows and target windows are embedded, embeddings are
    standardised on the fit windows, and the standard density conformal scoring
    is applied.
    """
    n = len(W)
    acc = np.zeros(n)
    hits = np.zeros(n)
    for i, f in enumerate(folds):
        fit, tgt = f.fit, f.target
        if len(fit) < 20 or len(set(y[fit].tolist())) < 2:
            continue
        pre = np.where(~np.isin(groups, list(f.held_groups)))[0]
        if verbose:
            print(f"    [deep] fold {i + 1}/{len(folds)}: pre-training on {len(pre)} windows "
                  f"(held-out groups excluded)", flush=True)
        emb = embedder(W[pre], seed)
        Zfit, Ztgt = emb(W[fit]), emb(W[tgt])
        mu, sd = Zfit.mean(0), Zfit.std(0) + 1e-6
        Z = np.zeros((n, Zfit.shape[1]))
        Z[fit] = (Zfit - mu) / sd
        Z[tgt] = (Ztgt - mu) / sd
        sub = channel_pvalues(Z, y, [Fold(fit=fit, target=tgt, held_groups=f.held_groups)],
                              groups, "density", cfg)["p"]
        ok = ~np.isnan(sub[tgt])
        acc[tgt[ok]] += sub[tgt[ok]]
        hits[tgt[ok]] += 1
    p = np.full(n, np.nan)
    scored = hits > 0
    p[scored] = acc[scored] / hits[scored]
    return p


def auxiliary_embedding(W, groups, exclude_groups, cfg, embedder, seed=0):
    """One extra encoder, pre-trained without the weight-probe validation groups,
    used only so the deep channel can be weighted by the same probes as every
    other channel. Returns embeddings for every window."""
    pre = np.where(~np.isin(groups, list(exclude_groups)))[0]
    emb = embedder(W[pre], seed)
    return emb(W)

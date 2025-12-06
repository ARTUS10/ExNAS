# ============================================================
# CNN (SmallCNN) - Colab Ready (CPU) + Calibration + Fallback
# ============================================================
#
# Copyright (c) 2025 José María Lancho Rodríguez
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================

import math, time, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------- Seeds ----------
def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_seeds(42)

# ---------- Data (MNIST or CIFAR10) ----------
def get_dataloaders(dataset_name="MNIST", batch_size=128, num_workers=2):
    from torchvision import datasets, transforms
    ds = dataset_name.upper()
    if ds == "MNIST":
        in_ch, img_size = 1, 28
        mean, std = (0.1307,), (0.3081,)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tf)
        test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tf)
    elif ds == "CIFAR10":
        in_ch, img_size = 3, 32
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=tf)
        test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    else:
        raise ValueError("dataset_name debe ser 'MNIST' o 'CIFAR10'.")

    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train, test, in_ch, img_size

# ---------- Model ----------
class SmallCNN(nn.Module):
    """Conv1 → ReLU → Conv2 → ReLU → AvgPool(2) → Flatten → FC-10"""
    def __init__(self, in_ch=1, num_classes=10, img_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.AvgPool2d(2, 2)
        with torch.no_grad():
            d = torch.zeros(1, in_ch, img_size, img_size)
            x = F.relu(self.conv1(d)); x = F.relu(self.conv2(x)); x = self.pool(x)
            flat_dim = x.view(1, -1).shape[1]
        self.fc = nn.Linear(flat_dim, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------- experiential memory ----------
class LayerMemory:
    """
    1D signatures (CPU) with top-k by cosine similarity and temporal decay.
    If 'dim' is None, uses original dimension.
    """
    def __init__(self, dim=None, top_k=16, decay=0.95, maxlen=1024):
        self.dim = dim
        self.top_k = top_k
        self.decay = decay
        self.signatures = deque(maxlen=maxlen)  # 1D tensors on CPU
        self.scores = deque(maxlen=maxlen)      # floats
        self.projector = None                   # (F x dim) for optional projection
    def _ensure_projector(self, feat_dim, device):
        if self.dim is None: return None
        if self.projector is None or self.projector.shape[0] != feat_dim:
            P = torch.randn(feat_dim, self.dim, device=device) / math.sqrt(max(1, feat_dim))
            self.projector = P
        return self.projector
    @torch.no_grad()
    def _compress(self, features):
        if features.ndim == 4:  # Conv: [B,C,H,W] → (C,)
            sig = features.mean(dim=(0, 2, 3)).detach().float().cpu()
            if self.dim is not None and sig.shape[0] != self.dim:
                if sig.shape[0] > self.dim: sig = sig[: self.dim]
                else: sig = torch.cat([sig, torch.zeros(self.dim - sig.shape[0])], dim=0)
            return sig
        # FC: [B,F] → (F,),  with optional projection
        device = features.device
        sig = features.mean(dim=0)
        Fdim = sig.shape[0]
        if self.dim is not None and self.dim < Fdim:
            P = self._ensure_projector(Fdim, device=device)
            sig = (sig @ P).detach()
        sig = sig.float().cpu()
        if self.dim is not None and sig.shape[0] != self.dim:
            if sig.shape[0] > self.dim: sig = sig[: self.dim]
            else: sig = torch.cat([sig, torch.zeros(self.dim - sig.shape[0])], dim=0)
        return sig
    @torch.no_grad()
    def add_signature(self, features, score=1.0):
        sig = self._compress(features)
        self.signatures.append(sig); self.scores.append(float(score))
    @torch.no_grad()
    def query_top_k(self, features):
        if not self.signatures: return []
        q = self._compress(features)    # CPU
        sims, qn = [], (q.norm().item() + 1e-8)
        total = len(self.signatures)
        for i, s in enumerate(self.signatures):
            m = min(len(q), len(s)); q_m, s_m = q[:m], s[:m]
            sim = float((q_m * s_m).sum().item() / ((qn) * (s_m.norm().item() + 1e-8)))
            sims.append((sim * (self.decay ** (total - i - 1)), self.scores[i]))
        sims.sort(key=lambda t: t[0], reverse=True)
        return sims[: min(self.top_k, len(sims))]

# ----------  Partial computation helpers  ----------
def conv2_forward_subset(x, conv, idx_out):
    """Conv2 sólo para canales idx_out; recompone tensor con ceros en el resto."""
    w = conv.weight.index_select(0, idx_out)
    b = conv.bias.index_select(0, idx_out) if conv.bias is not None else None
    y_sel = F.conv2d(x, w, b, stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
    B, k, H, W = y_sel.shape; C = conv.out_channels
    y = x.new_zeros(B, C, H, W); y[:, idx_out, :, :] = y_sel
    return y
def linear_forward_subset(x, linear, idx_feat):
    """FC usando sólo columnas idx_feat de W."""
    W_sel = linear.weight[:, idx_feat]; b = linear.bias
    x_sel = x.index_select(1, idx_feat)
    return F.linear(x_sel, W_sel, b)

# ---------- Conv2 gate calibration (Ridge) ----------
@torch.no_grad()
def calibrate_gate_conv2_linear(base_model, loader, steps=8, lam=1e-3, device="cpu"):
    """
    Ajusta gate_conv2 para que prediga la 'importancia' (L2/espacial) de Conv2
    a partir de GAP(Conv1). Usa ridge: W = (X^T X + λI)^(-1) X^T Y
    X: (N, 32)  ; Y: (N, 64)
    """
    Xs, Ys, n_collected = [], [], 0
    base_model.eval()
    for xb, _ in loader:
        xb = xb.to(device)
        x1 = F.relu(base_model.conv1(xb))                       # (B,32,H,W)
        x2 = base_model.conv2(x1)                               # (B,64,H,W) pre-ReLU
        X = x1.mean(dim=(2, 3))                                 # (B,32) GAP
        Y = x2.pow(2).mean(dim=(2, 3)).sqrt()                   # (B,64) L2 espatial
        Xs.append(X.cpu()); Ys.append(Y.cpu())
        n_collected += X.shape[0]
        steps -= 1
        if steps <= 0: break
    X = torch.cat(Xs, dim=0)  # (N,32)
    Y = torch.cat(Ys, dim=0)  # (N,64)
    N, D_in = X.shape; D_out = Y.shape[1]
    I = torch.eye(D_in)
    W = torch.linalg.solve(X.T @ X + lam * I, X.T @ Y)          # (32,64)
    return W.T.contiguous()                                     # return as (64,32) (Linear weight)

# ---------- ExNAS compute-aware + fallback ----------
class ExNASWrapper(nn.Module):
    """
    Gating compute-aware:
      - Conv2: subset of out-ch before computing the layer.
      - FC:   subset of features (W columns).
    Extras:
      - Stable memory (ALWAYS based on Conv1).
      - Prior gate_conv2 calibration (optional).
      - Uncertainty fallback: re-evaluates only uncertain samples with baseline.
    """
    def __init__(self, base_model,
                 bl=0.20, Bg=0.05, top_k=16,
                 mem_dim_fc=128, mem_decay=0.95,
                 min_keep_conv2=24,     # +accuracy
                 min_keep_fc=8192,      # +accuracy (MNIST total=12544)
                 fallback_thresh=0.60   # re-evaluate with baseline if max prob < 0.60
                 ):
        super().__init__()
        self.base = base_model
        self.bl, self.Bg = float(bl), float(Bg)
        self.top_k = int(top_k)
        self.fallback_thresh = float(fallback_thresh) if fallback_thresh is not None else None

        # Memories
        self.mem_conv2 = LayerMemory(dim=None,       top_k=top_k, decay=mem_decay)  # uses Conv1 (32)
        self.mem_fc    = LayerMemory(dim=mem_dim_fc, top_k=top_k, decay=mem_decay)

        # Cheap gate for Conv2: GAP Conv1 → Linear(32→64)
        in_ch  = self.base.conv2.in_channels   # 32
        out_ch = self.base.conv2.out_channels  # 64
        self.gate_conv2 = nn.Linear(in_ch, out_ch, bias=False)
        with torch.no_grad():
            # stable replicated identity
            W = torch.zeros(out_ch, in_ch)
            I = torch.eye(in_ch)
            rows = min(out_ch, in_ch)
            W[:rows, :] = I[:rows, :]
            if out_ch > rows:
                rep = min(out_ch - rows, in_ch)
                W[rows:rows+rep, :] = I[:rep, :]
            self.gate_conv2.weight.copy_(W)

        self.min_keep_conv2 = int(min_keep_conv2)
        self.min_keep_fc    = int(min_keep_fc)

        self.active_fracs = {"conv2": [], "fc": []}
        self.last_masks   = {"conv2": None, "fc": None}

    @staticmethod
    def _n_keep(n_units, bl, Bg, n_min):
        n_local = max(1, int(n_units * bl))
        n_keep  = max(n_min, int(n_local * (1.0 - Bg)))
        return min(n_units, n_keep)

    def _scores_conv2(self, x_conv1):
        sig = x_conv1.abs().mean(dim=(0, 2, 3))          # (32,)
        pred = F.softplus(self.gate_conv2(sig))          # (64,)
        sims = self.mem_conv2.query_top_k(x_conv1)
        mem_factor = 1.0 + (sum(s for s, _ in sims) / len(sims) if sims else 0.0)
        return pred * mem_factor

    def _scores_fc(self, x_flat):
        return x_flat.abs().mean(dim=0)                  # (F,)

    def forward(self, x0):
        # Conv1
        x1 = F.relu(self.base.conv1(x0))

        # ---- Gating Conv2 ----
        scores_c = self._scores_conv2(x1)                            # (64,)
        n_keep_c = self._n_keep(scores_c.numel(), self.bl, self.Bg, self.min_keep_conv2)
        idx_c = torch.topk(scores_c, k=n_keep_c, largest=True).indices
        mask_c = torch.zeros_like(scores_c, dtype=torch.bool); mask_c[idx_c] = True
        self.last_masks["conv2"] = mask_c.detach().cpu()
        self.active_fracs["conv2"].append(mask_c.float().mean().item())

        # update Conv2 memory ALWAYS with Conv1 
        with torch.no_grad():
            self.mem_conv2.add_signature(x1, score=float(scores_c.mean().item()))

        # Conv2 parcial + ReLU
        x2 = F.relu(conv2_forward_subset(x1, self.base.conv2, idx_c))

        # Pool + Flatten
        x_flat = self.base.pool(x2).view(x2.size(0), -1)             # (B,F)

        # ---- Gating FC ----
        scores_f = self._scores_fc(x_flat)                            # (F,)
        n_keep_f = self._n_keep(scores_f.numel(), self.bl, self.Bg, self.min_keep_fc)
        idx_f = torch.topk(scores_f, k=n_keep_f, largest=True).indices
        mask_f = torch.zeros_like(scores_f, dtype=torch.bool); mask_f[idx_f] = True
        self.last_masks["fc"] = mask_f.detach().cpu()
        self.active_fracs["fc"].append(mask_f.float().mean().item())

        # FC parcial
        out = linear_forward_subset(x_flat, self.base.fc, idx_f)

        # FC Memory
        with torch.no_grad():
            self.mem_fc.add_signature(x_flat, score=float(scores_f.mean().item()))

        # ---- FUncertainty fallback (optional) ----
        if self.fallback_thresh is not None:
            with torch.no_grad():
                probs = out.softmax(dim=1)
                low = probs.max(dim=1).values < self.fallback_thresh
                if low.any():
                    full_logits = self.base(x0[low])
                    out[low] = full_logits

        return out

    def get_stats(self):
        conv2_active = float(np.mean(self.active_fracs["conv2"])) if self.active_fracs["conv2"] else 0.0
        fc_active    = float(np.mean(self.active_fracs["fc"]))    if self.active_fracs["fc"]    else 0.0
        return {"conv2_active": conv2_active, "fc_active": fc_active}

# ---------- Training and evaluation ----------
def train_baseline(model, train_loader, device, epochs=1, lr=1e-3, log_every=100):
    model.to(device); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    running = 0.0
    for ep in range(1, epochs+1):
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits = model(xb)
            loss = crit(logits, yb); loss.backward(); opt.step()
            running += loss.item()
            if (i+1) % log_every == 0:
                print(f"[epoch {ep}/{epochs}] step {i+1} loss {running/log_every:.4f}")
                running = 0.0

@torch.no_grad()
def evaluate(model, data_loader, device, warmup_batches=5):
    model.to(device); model.eval()
    # warmup
    it = iter(data_loader)
    for _ in range(min(warmup_batches, len(data_loader))):
        xb, _ = next(it, (None, None))
        if xb is None: break
        _ = model(xb.to(device))
    # measurement
    correct, total = 0, 0
    t0 = time.time()
    for xb, yb in data_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.numel()
    elapsed = time.time() - t0
    acc = correct / total if total else 0.0
    tput = total / max(elapsed, 1e-8)
    return acc, elapsed, tput

# ---------- Main experiment ----------
def run_experiment(
    dataset_name="MNIST",
    train_epochs=1,
    batch_size=128,
    # More conservative budgets for better Acc/Speed
    bl=0.20, Bg=0.05,
    top_k=16,
    mem_dim_fc=128,
    mem_decay=0.95,
    min_keep_conv2=24,
    min_keep_fc=8192,
    fallback_thresh=0.60,
    calib_steps=8, calib_lam=1e-3,
    num_workers=2
):
    print("="*70)
    print(f"ExNAS CNN - {dataset_name} | Presupuestos: bl={bl}, Bg={Bg}, top-k={top_k}")
    print("="*70)
    device = torch.device("cpu")

    train_loader, test_loader, in_ch, img_size = get_dataloaders(
        dataset_name=dataset_name, batch_size=batch_size, num_workers=num_workers
    )

    # Baseline
    base = SmallCNN(in_ch=in_ch, num_classes=10, img_size=img_size)
    print("\n[Entrenamiento baseline]")
    train_baseline(base, train_loader, device, epochs=train_epochs, lr=1e-3, log_every=100)

    print("\n[Evaluación BASELINE]")
    acc_b, t_b, tp_b = evaluate(base, test_loader, device)
    print(f"Accuracy:   {acc_b:.4f}")
    print(f"Tiempo:     {t_b:.3f} s")
    print(f"Throughput: {tp_b:.1f} muestras/s")

    # ExNAS
    exnas = ExNASWrapper(
        base_model=base, bl=bl, Bg=Bg, top_k=top_k,
        mem_dim_fc=mem_dim_fc, mem_decay=mem_decay,
        min_keep_conv2=min_keep_conv2, min_keep_fc=min_keep_fc,
        fallback_thresh=fallback_thresh
    )

    # Quick Conv2 gate calibration (improves channel selection)
    print("\n[Calibración rápida de gate Conv2]")
    W = calibrate_gate_conv2_linear(base, train_loader, steps=calib_steps, lam=calib_lam, device=str(device))
    with torch.no_grad():
        exnas.gate_conv2.weight.copy_(W)

    print("\n[Evaluación ExNAS (compute-aware + fallback)]")
    acc_e, t_e, tp_e = evaluate(exnas, test_loader, device)
    stats = exnas.get_stats()
    print(f"Accuracy:   {acc_e:.4f}")
    print(f"Tiempo:     {t_e:.3f} s")
    print(f"Throughput: {tp_e:.1f} muestras/s")
    print(f"Conv2 activo medio: {100.0*stats['conv2_active']:.2f}%")
    print(f"FC   activo medio: {100.0*stats['fc_active']:.2f}%")

    print("\n[Comparación]")
    d_time = (t_b - t_e) / max(t_b, 1e-8) * 100.0
    d_tp   = (tp_e - tp_b) / max(tp_b, 1e-8) * 100.0
    d_acc  = (acc_e - acc_b) * 100.0
    print(f"Δ Tiempo:     {d_time:+.2f}%")
    print(f"Δ Throughput: {d_tp:+.2f}%")
    print(f"Δ Accuracy:   {d_acc:+.2f} pp")
    print("\nListo ✅")

# ---------- Run ----------
if __name__ == "__main__":
    run_experiment(
        dataset_name="MNIST",
        train_epochs=1,
        batch_size=128,
        bl=0.20, Bg=0.05,        # good starting point
        top_k=16,
        mem_dim_fc=128,
        mem_decay=0.95,
        min_keep_conv2=24,       # ↑ if you want more accuracy
        min_keep_fc=8192,        # ↑ if you want more accuracy
        fallback_thresh=0.60,    # lower to 0.55 if you want more speed
        calib_steps=8, calib_lam=1e-3,
        num_workers=2
    )

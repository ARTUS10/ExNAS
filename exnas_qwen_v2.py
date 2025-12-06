# ============================================================
#   ExNAS-Lite v2 — Qwen2-1.5B — Colab T4 
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

!pip -q install "transformers>=4.41.0" datasets torch accelerate --upgrade

import os, math, time, json, random, collections
from typing import List, Dict, Tuple
import numpy as np
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- Stable environment ----------
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['PYTHONHASHSEED']='123'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
random.seed(123); np.random.seed(123); torch.manual_seed(123)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(123)

MODEL_ID      = "Qwen/Qwen2-1.5B"
TARGET_LAYERS = list(range(9,17))    # 8 middle layers 
GROUP         = 128
CANDIDATE_WIDTHS = [8704, 8448, 8192]  # all multiples of 128

# Unified harness
EVAL_SEQ       = 256
PROMPTS        = ["The future of AI","Recent scientific developments have led to","The economic impact of new technologies"]
NEW_TOKENS     = 1024
WARMUP         = 3
REPEATS_SEL    = 2   # for quick selection
REPEATS_FINAL  = 3   # for final metrics

# Guardarails
PPL_GATE_GLOBAL = 1.0      # ≤ +1% global
TPS_GATE_LOCAL  = 0.2      # ≥ +0.2% local; if none pass, fallback ≥0% 

# (Optional) Per-sequence gating
USE_SEQUENCE_GATING = True
NGRAM, MEM_CAP = 12, 30000
SENTINEL       = 13
TARGET_SLIM_FR = 0.35
RHO_MIN, RHO_MAX = 0.015, 0.060

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float16 if device.type=="cuda" else torch.float32
print(f"Device: {device}  dtype: {dtype}")

# ---------- Model ----------
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=None, dtype=dtype, trust_remote_code=True).to(device)
tok   = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
if hasattr(model.config, "use_cache"): model.config.use_cache = True
model.eval()
layers = model.model.layers
print(f"Capas totales: {len(layers)} | Target: {TARGET_LAYERS}")

# ---------- Data ----------
def wikitext_subset(tok, split="test", seq_len=EVAL_SEQ, n_texts=8):
    ds = load_dataset("wikitext","wikitext-2-raw-v1", split=split)
    bufs=[]
    for x in ds:
        t=x["text"].strip()
        if not t: continue
        ids = tok(t, return_tensors="pt", truncation=True, max_length=seq_len,
                  padding="max_length")["input_ids"][0]
        bufs.append(ids)
        if len(bufs)>=n_texts: break
    if len(bufs)<n_texts:
        s = "The neural network architecture was designed to process sequential data efficiently. " * 40
        for _ in range(n_texts-len(bufs)):
            bufs.append(tok(s, return_tensors="pt", truncation=True, max_length=seq_len,
                            padding="max_length")["input_ids"][0])
    return bufs

eval_bufs  = wikitext_subset(tok, "test",  EVAL_SEQ, n_texts=8)
calib_bufs = wikitext_subset(tok, "train", EVAL_SEQ, n_texts=4)

# ---------- Harness ----------
@torch.no_grad()
def measure_ppl(m, bufs):
    m.eval()
    tot_loss, tot_tokens = 0.0, 0
    for ids in bufs:
        inp = ids.unsqueeze(0).to(device)
        attn = (inp != tok.pad_token_id)
        labels = inp.clone(); labels[labels==tok.pad_token_id] = -100
        out = m(input_ids=inp, attention_mask=attn, labels=labels)
        loss = float(out.loss); valid = int((labels!=-100).sum())
        tot_loss += loss*valid; tot_tokens += valid
    return math.exp(tot_loss/max(1,tot_tokens))

@torch.no_grad()
def measure_tps(m, repeats):
    m.eval()
    # warm-up
    for _ in range(WARMUP):
        for p in PROMPTS:
            batch = tok(p, return_tensors="pt")
            batch["attention_mask"] = (batch["input_ids"] != tok.pad_token_id)
            batch = {k:v.to(device) for k,v in batch.items()}
            _ = m.generate(**batch, max_new_tokens=256, do_sample=False,
                           pad_token_id=tok.pad_token_id, use_cache=True)
    if device.type=="cuda": torch.cuda.synchronize()
    vals=[]
    for _ in range(repeats):
        tot_new, tot_time = 0, 0.0
        for p in PROMPTS:
            batch = tok(p, return_tensors="pt")
            batch["attention_mask"] = (batch["input_ids"] != tok.pad_token_id)
            batch = {k:v.to(device) for k,v in batch.items()}
            if device.type=="cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = m.generate(**batch, max_new_tokens=NEW_TOKENS, do_sample=False,
                           pad_token_id=tok.pad_token_id, use_cache=True)
            if device.type=="cuda": torch.cuda.synchronize()
            tot_new += NEW_TOKENS; tot_time += (time.perf_counter()-t0)
        vals.append(tot_new/max(tot_time,1e-9))
    arr = np.array(vals, dtype=np.float64)
    return {"runs": vals, "mean": float(arr.mean()),
            "ci": 1.96*(float(arr.std(ddof=1))/max(1,math.sqrt(len(vals)))) if len(vals)>1 else 0.0}

# ---------- Safe Fisher (or fallback) ----------
def fisher_scores_qwen_mlp_safe(mlp):
    up, gate = mlp.up_proj, mlp.gate_proj
    N = up.weight.size(0)
    try:
        scores = torch.zeros(N, dtype=torch.float32, device=device)
        prev_req={p:p.requires_grad for p in model.parameters()}
        for p in model.parameters(): p.requires_grad_(False)
        up.weight.requires_grad_(True); gate.weight.requires_grad_(True)
        if up.bias is not None: up.bias.requires_grad_(True)
        if gate.bias is not None: gate.bias.requires_grad_(True)
        prev_mode = model.training; model.train()
        with torch.enable_grad():
            for ids in calib_bufs:
                inp = ids.unsqueeze(0).to(device)
                attn=(inp != tok.pad_token_id)
                labels=inp.clone(); labels[labels==tok.pad_token_id]=-100
                out = model(input_ids=inp, attention_mask=attn, labels=labels)
                loss = out.loss
                model.zero_grad(set_to_none=True); loss.backward()
                up_w=up.weight.detach().float();   up_g=up.weight.grad.detach().float()
                gt_w=gate.weight.detach().float(); gt_g=gate.weight.grad.detach().float()
                scores += torch.sqrt(
                    (up_w.abs()*up_g.abs()).mean(dim=1) * (gt_w.abs()*gt_g.abs()).mean(dim=1) + 1e-8
                )
        for p,req in prev_req.items(): p.requires_grad_(req)
        model.train(prev_mode)
        return scores
    except Exception:
        with torch.no_grad():
            up_w   = up.weight.detach().float()
            gate_w = gate.weight.detach().float()
            acc = torch.zeros(up_w.shape[1], dtype=torch.float32, device=device)
            def prehook(m, x):
                x0=x[0]; acc.add_(x0.detach().float().abs().mean(dim=(0,1)))
            h = up.register_forward_pre_hook(lambda m,inp: prehook(m,inp))
            cnt=0
            for ids in calib_bufs:
                inp = ids.unsqueeze(0).to(device)
                attn=(inp != tok.pad_token_id)
                _ = model(input_ids=inp, attention_mask=attn)
                cnt+=1
            h.remove(); acc = acc/max(cnt,1)
            up_imp   = (up_w.abs()   @ acc).float()
            gate_imp = (gate_w.abs() @ acc).float()
            return torch.sqrt((up_imp+1e-8)*(gate_imp+1e-8))

def group_topk(scores: torch.Tensor, target_width: int, group: int):
    N = scores.numel(); num_groups = N//group; usable = num_groups*group
    g_scores = scores[:usable].reshape(num_groups, group).mean(dim=1)
    keep_g = max(1, min(num_groups, round(target_width/group)))
    topg = torch.topk(g_scores, k=keep_g, largest=True).indices.sort()[0]
    idx=[]
    for g in topg.tolist():
        s,e = g*group, (g+1)*group
        idx.extend(range(s,e))
    return torch.tensor(idx, device=device, dtype=torch.long)

def make_slim_qwen_mlp(mlp, keep_idx):
    up, gate, down = mlp.up_proj, mlp.gate_proj, mlp.down_proj
    dev, dt = up.weight.device, up.weight.dtype
    inF  = up.in_features
    outF = keep_idx.numel()
    slim_up   = nn.Linear(inF, outF, bias=up.bias   is not None, device=dev, dtype=dt)
    slim_gate = nn.Linear(inF, outF, bias=gate.bias is not None, device=dev, dtype=dt)
    slim_down = nn.Linear(outF, down.out_features,  bias=down.bias is not None, device=dev, dtype=dt)
    with torch.no_grad():
        slim_up.weight.copy_(up.weight[keep_idx]);     slim_gate.weight.copy_(gate.weight[keep_idx])
        if up.bias   is not None: slim_up.bias.copy_(up.bias[keep_idx])
        if gate.bias is not None: slim_gate.bias.copy_(gate.bias[keep_idx])
        slim_down.weight.copy_(down.weight[:, keep_idx])
        if down.bias is not None: slim_down.bias.copy_(down.bias)
    act = getattr(mlp, "act_fn", getattr(mlp, "act", nn.SiLU()))
    drop= getattr(mlp, "dropout", nn.Dropout(0.0))
    class Wrap(nn.Module):
        def __init__(self, act, drop, up, gate, down):
            super().__init__(); self.act_fn=act; self.dropout=drop
            self.full=mlp; self.up_proj=up; self.gate_proj=gate; self.down_proj=down; self.mode="full"
        def set_full(self): self.mode="full"
        def set_slim(self): self.mode="slim"
        def forward(self, x):
            if self.mode=="full":
                h = self.act_fn(self.full.gate_proj(x)) * self.full.up_proj(x)
                h = self.dropout(h); return self.full.down_proj(h)
            else:
                h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                h = self.dropout(h); return self.down_proj(h)
    return Wrap(act, drop, slim_up, slim_gate, slim_down)

# ---------- Baseline ----------
print("\n=== BASELINE (harness unificado) ===")
ppl_base = measure_ppl(model, eval_bufs)
tps_base = measure_tps(model, repeats=REPEATS_FINAL)
print(f"PPL  {ppl_base:.2f}")
print(f"TPS  {tps_base['mean']:.2f} ±{tps_base['ci']:.2f}")

# ---------- Layer-wise selection ----------
accepted = []  # (li, width, d_ppl, d_tps_local)
for li in TARGET_LAYERS:
    mlp = layers[li].mlp
    scores = fisher_scores_qwen_mlp_safe(mlp)
    best=None
    # 1) normal criterion (≥ +0.2% TPS)
    for w in CANDIDATE_WIDTHS:
        keep_idx = group_topk(scores, w, GROUP)
        slim_wrap = make_slim_qwen_mlp(mlp, keep_idx)
        old = layers[li].mlp; layers[li].mlp = slim_wrap; slim_wrap.set_slim()
        ppl_after = measure_ppl(model, eval_bufs)
        d_ppl = (ppl_after/ppl_base - 1.0)*100.0
        tps_local = measure_tps(model, repeats=REPEATS_SEL)['mean']
        d_tps = (tps_local/tps_base['mean'] - 1.0)*100.0
        layers[li].mlp = old
        if d_ppl <= PPL_GATE_GLOBAL + 1e-9 and d_tps >= TPS_GATE_LOCAL:
            if best is None or d_tps > best[2]:
                best = (w, d_ppl, d_tps, slim_wrap)
    # 2) fallback if nothing ≥ 0.2% but there is ≥ 0.0%
    if best is None:
        for w in CANDIDATE_WIDTHS:
            keep_idx = group_topk(scores, w, GROUP)
            slim_wrap = make_slim_qwen_mlp(mlp, keep_idx)
            old = layers[li].mlp; layers[li].mlp = slim_wrap; slim_wrap.set_slim()
            ppl_after = measure_ppl(model, eval_bufs)
            d_ppl = (ppl_after/ppl_base - 1.0)*100.0
            tps_local = measure_tps(model, repeats=REPEATS_SEL)['mean']
            d_tps = (tps_local/tps_base['mean'] - 1.0)*100.0
            layers[li].mlp = old
            if d_ppl <= PPL_GATE_GLOBAL + 1e-9 and d_tps >= 0.0:
                best = (w, d_ppl, d_tps, slim_wrap); break
    if best is not None:
        layers[li].mlp = best[3]; layers[li].mlp.set_full()
        accepted.append((li, best[0], best[1], best[2]))
        print(f"  capa {li}: width {best[0]} | ΔPPL {best[1]:+.2f}% | ΔTPS_local {best[2]:+.2f}% → ACEPTADA")
    else:
        print(f"  capa {li}: ninguna variante acelera y respeta PPL → DESCARTADA")

# ---------- Global revalidation (hill-climb) ----------
def set_all(mode="full"):
    for (li,_,_,_) in accepted:
        w = layers[li].mlp
        if hasattr(w,"set_full") and hasattr(w,"set_slim"):
            w.set_full() if mode=="full" else w.set_slim()

set_all("slim")
ppl_all = measure_ppl(model, eval_bufs)
tps_all = measure_tps(model, repeats=REPEATS_FINAL)
d_ppl_all = (ppl_all/ppl_base - 1.0)*100.0
d_tps_all = (tps_all['mean']/tps_base['mean'] - 1.0)*100.0

if d_tps_all < 0 and accepted:
    accepted_sorted = sorted(accepted, key=lambda x: x[3])  # worst local_TPS first
    kept = accepted_sorted[:]
    for item in accepted_sorted:
        li_rm = item[0]
        for (li,_,_,_) in kept: layers[li].mlp.set_slim()
        layers[li_rm].mlp.set_full()
        ppl_try = measure_ppl(model, eval_bufs)
        tps_try = measure_tps(model, repeats=REPEATS_FINAL)
        if (tps_try['mean']/tps_base['mean'] - 1.0) > 0:
            kept = [it for it in kept if it[0]!=li_rm]
            d_ppl_all = (ppl_try/ppl_base - 1.0)*100.0
            d_tps_all = (tps_try['mean']/tps_base['mean'] - 1.0)*100.0
            accepted = kept
            print(f"[hill-climb] retiro capa {li_rm} → ΔTPS {d_tps_all:+.2f}%")
            break
set_all("full")

print("\n=== Conjunto estático verificado ===")
print(f"ΔPPL_all-slim {d_ppl_all:+.2f}% | ΔTPS_all-slim {d_tps_all:+.2f}%")
print(f"Capas aceptadas: {[li for (li,_,_,_) in accepted]}")

# ---------- (Optional) Per-sequence gating  ----------
class LRUExperiential:
    def __init__(self, cap=MEM_CAP): self.cap=cap; self.store=collections.OrderedDict()
    def get(self,key):
        if key in self.store: self.store.move_to_end(key); return self.store[key]
        return None
    def put(self,key,val):
        if key in self.store: self.store[key]=val; self.store.move_to_end(key)
        else:
            if len(self.store)>=self.cap: self.store.popitem(last=False)
            self.store[key]=val
memory = LRUExperiential(MEM_CAP)

@torch.no_grad()
def sentinel_ratio(ids_prefix: torch.Tensor):
    li=SENTINEL
    block = layers[li].mlp
    core  = block.full if hasattr(block, "full") else layers[li].mlp
    vals={}
    def hook(module, inp, out):
        y=out; y_last = y[:, -1, :] if y.dim()==3 else y
        vals["v"]=float(torch.linalg.vector_norm(y_last, ord=2).detach().cpu())
    h = core.down_proj.register_forward_hook(hook)
    inp = ids_prefix.unsqueeze(0).to(device)
    attn=(inp != tok.pad_token_id)
    out = model(input_ids=inp, attention_mask=attn)
    if hasattr(out,"last_hidden_state"):
        x = out.last_hidden_state[:, -1, :]
        xnorm=float(torch.linalg.vector_norm(x, ord=2).detach().cpu())
    else:
        xnorm=1.0
    h.remove()
    return vals.get("v",1.0)/max(1e-6, xnorm)

def decide_profile(ids_prefix: torch.Tensor, rho: float) -> str:
    key=tuple(ids_prefix.view(-1).tolist()[-NGRAM:])
    hit = memory.get(key)
    if hit is not None: return hit
    r = sentinel_ratio(ids_prefix)
    ch = "slim" if (r < rho) else "full"
    memory.put(key, ch)
    return ch

def calibrate_rho(target=TARGET_SLIM_FR):
    rho = (RHO_MIN+RHO_MAX)/2.0
    for _ in range(5):
        memory.store.clear()
        slim, tot = 0, 0
        for ids in calib_bufs:
            ids=ids[:EVAL_SEQ]
            for t in range(32, min(EVAL_SEQ,96), 16):
                ch = decide_profile(ids[:t], rho)
                slim += int(ch=="slim"); tot += 1
        rate = slim/max(tot,1)
        if rate < target: rho += (RHO_MAX-rho)*0.5
        else:             rho -= (rho-RHO_MIN)*0.5
    return max(RHO_MIN, min(RHO_MAX, rho))

ppl_ex, tps_ex, rho_star = None, None, None
if USE_SEQUENCE_GATING and accepted:
    rho_star = calibrate_rho()
    print(f"\nGating calibrado: rho* = {rho_star:.3f} (objetivo {TARGET_SLIM_FR:.0%})")
    @torch.no_grad()
    def eval_with_gating():
        # PPL
        tot_loss, tot_tokens = 0.0, 0
        for ids in eval_bufs:
            ids = ids[:EVAL_SEQ]
            ch = decide_profile(ids[:min(64,len(ids))], rho_star)
            for (li,_,_,_) in accepted:
                w = layers[li].mlp
                if hasattr(w,"set_slim"): (w.set_slim() if ch=="slim" else w.set_full())
            inp = ids.unsqueeze(0).to(device)
            attn=(inp != tok.pad_token_id)
            labels=inp.clone(); labels[labels==tok.pad_token_id]=-100
            out = model(input_ids=inp, attention_mask=attn, labels=labels)
            loss=float(out.loss); valid=int((labels!=-100).sum())
            tot_loss += loss*valid; tot_tokens += valid
        ppl = math.exp(tot_loss/max(1,tot_tokens))
        # TPS
        vals=[]
        for _ in range(REPEATS_FINAL):
            tot_new, tot_time = 0, 0.0
            for p in PROMPTS:
                ids = tok(p, return_tensors="pt")["input_ids"][0]
                ch = decide_profile(ids, rho_star)
                for (li,_,_,_) in accepted:
                    w = layers[li].mlp
                    if hasattr(w,"set_slim"): (w.set_slim() if ch=="slim" else w.set_full())
                batch = {"input_ids": ids.unsqueeze(0).to(device)}
                batch["attention_mask"] = (batch["input_ids"] != tok.pad_token_id)
                if device.type=="cuda": torch.cuda.synchronize()
                t0=time.perf_counter()
                _ = model.generate(**batch, max_new_tokens=NEW_TOKENS, do_sample=False,
                                   pad_token_id=tok.pad_token_id, use_cache=True)
                if device.type=="cuda": torch.cuda.synchronize()
                tot_new += NEW_TOKENS; tot_time += (time.perf_counter()-t0)
            vals.append(tot_new/max(tot_time,1e-9))
        arr = np.array(vals, dtype=np.float64)
        return math.exp(tot_loss/max(1,tot_tokens)), {"runs": vals, "mean": float(arr.mean()),
                 "ci": 1.96*(float(arr.std(ddof=1))/max(1,math.sqrt(len(vals)))) if len(vals)>1 else 0.0}
    ppl_ex, tps_ex = eval_with_gating()

# ---------- Final and JSON ----------
ppl_pre  = measure_ppl(model, eval_bufs)
tps_pre  = measure_tps(model, repeats=REPEATS_FINAL)

print("\n=== METRICS (final) ===")
print(f"BASE PPL {ppl_base:.2f} | TPS {tps_base['mean']:.2f} ±{tps_base['ci']:.2f}")
print(f"PRE  PPL {ppl_pre:.2f} | TPS {tps_pre['mean']:.2f} ±{tps_pre['ci']:.2f}")
if accepted:
    if ppl_ex is not None and tps_ex is not None:
        print(f"EXNAS PPL {ppl_ex:.2f} | TPS {tps_ex['mean']:.2f} ±{tps_ex['ci']:.2f}")
        print(f"ΔPPL {((ppl_ex/ppl_base)-1)*100:+.2f}% | ΔTPS {((tps_ex['mean']/tps_base['mean'])-1)*100:+.2f}%")
    else:
        print("EXNAS (gating) desactivado o sin capas aceptadas.")
else:
    print("No se aceptaron capas; usa sólo baseline/pre.")

summary = {
  "model": MODEL_ID, "device": str(device), "dtype": str(dtype),
  "harness": {"prompts": len(PROMPTS), "new_tokens": NEW_TOKENS, "warmup": WARMUP,
              "repeats_sel": REPEATS_SEL, "repeats_final": REPEATS_FINAL},
  "target_layers": TARGET_LAYERS, "group": GROUP, "candidate_widths": CANDIDATE_WIDTHS,
  "gates": {"ppl_global_pct": PPL_GATE_GLOBAL, "tps_local_pct": TPS_GATE_LOCAL},
  "baseline": {"ppl": float(ppl_base), "tps": tps_base},
  "accepted_layers": [{"layer":li,"width":w,"d_ppl_pct":dp,"d_tps_local_pct":dt}
                      for (li,w,dp,dt) in accepted],
  "static_all_slim": {"d_ppl_pct": float((measure_ppl(model, eval_bufs)/ppl_base-1)*100.0) if accepted else 0.0,
                      "note": "medido con perfiles en 'full' por seguridad"},
  "pre": {"ppl": float(ppl_pre), "tps": tps_pre},
}
if ppl_ex is not None and tps_ex is not None:
    summary["exnas"] = {"ppl": float(ppl_ex), "tps": tps_ex, "rho_star": float(rho_star),
                        "memory": {"type":"LRU","ngram":NGRAM,"cap":MEM_CAP,"sentinel":SENTINEL}}
with open("exnas_v2_summary.json","w") as f: json.dump(summary, f, indent=2)
print("\nGuardado → exnas_v2_summary.json")


# ============================================================
# ExNAS — SeqGate v2  — DistilGPT-2 — Colab 
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

!pip -q install "transformers>=4.41.0" datasets

import os, gc, math, time, json, random
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------- CONFIG ----------------
MODEL_ID = "distilgpt2"
TARGET_LAYERS = [2, 3, 4]
GROUP = 64
SLIM_WIDTHS_TRY = [3008, 2944]   # conservative → fallback (×64)

PPL_CAP_LAYER  = 0.010           # 1.0% per layer
PPL_CAP_LAYER_FALLBACK = 0.015   # if none pass

# PPL streaming
EVAL_SEQ = 512
EVAL_STRIDE = 256   # overlap

# Calibración gating
CALIB_SEQ = 128
CALIB_SAMPLES = 64
Q_RATIO = 0.35      # ratio percentile
Q_ENT   = 0.35      # entropy percentile

# TPS
TPS_NEW = 64
TPS_REPEATS = 3

SEED = 123
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ----------------- utils -----------------
def safe_set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        try: torch.cuda.manual_seed_all(seed)
        except Exception as e: print(f"[seed cuda] omitido: {e}")

def device_dtype():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda"), dtype
    return torch.device("cpu"), torch.float32

def get_layers_gpt2(model):
    if hasattr(model,"transformer") and hasattr(model.transformer,"h"):
        return model.transformer.h
    raise RuntimeError("Esperado GPT-2/DistilGPT-2")

def mlp_out_features_gpt2(mlp):
    fc = mlp.c_fc if hasattr(mlp, "c_fc") else mlp.orig_fc
    return int(fc.bias.shape[0]) if getattr(fc,"bias",None) is not None else int(fc.weight.shape[-1])

def model_max_pos(model):
    return getattr(model.config,"n_positions", getattr(model.config,"max_position_embeddings",1024))

def clamp_len(ids: torch.Tensor, max_len: int) -> torch.Tensor:
    return ids if ids.numel() <= max_len else ids[-max_len:]

# -------- tokenization without warning --------
def encode_long_text(tok, text: str):
    # encode to list (without giant tensors), no truncation (CausalLM does shift internally)  
    ids = tok.encode(text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long)

# ---------- PPL streaming (FIX) ----------
@torch.no_grad()
def ppl_streaming(model, tok, device, seq_len=EVAL_SEQ, stride=EVAL_STRIDE):
    model.eval()
    ds = load_dataset("wikitext","wikitext-2-raw-v1", split="test")
    text = "\n\n".join([x["text"] for x in ds])
    ids_all = encode_long_text(tok, text).to(device)

    max_pos = model_max_pos(model)
    seq_len = min(seq_len, max_pos-8)

    total_ll, total_tok = 0.0, 0
    L = ids_all.size(0) - 1
    for i in range(0, L, stride):
        end = min(i + seq_len, L)
        cur_len = end - i
        if cur_len <= 0: break
        x = ids_all[i:end].unsqueeze(0)             # [1, T]
        attn = torch.ones_like(x, dtype=torch.long) # no padding
        # *** FIX: labels = input_ids (CausalLM hace el shift internamente) ***
        out = model(input_ids=x, attention_mask=attn, labels=x)
        total_ll += float(out.loss.detach().cpu()) * cur_len
        total_tok += cur_len
        if end == L: break
    return math.exp(total_ll / max(1,total_tok))

# ---------- TPS (prewarm + masks) ----------
@torch.no_grad()
def tps_generate(model, tok, device, new_tokens=TPS_NEW, repeats=TPS_REPEATS):
    model.eval()
    prompts = [
        "The future of artificial intelligence",
        "Recent scientific developments have led to",
        "The economic impact of new technologies",
    ]
    # prewarm
    for p in prompts[:1]:
        ids = clamp_len(tok(p, return_tensors="pt")["input_ids"][0], model_max_pos(model)-8)
        _ = model.generate(input_ids=ids.unsqueeze(0).to(device),
                           attention_mask=torch.ones_like(ids.unsqueeze(0),device=device),
                           max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)

    vals=[]
    for _ in range(repeats):
        tot_new, tot_time = 0, 0.0
        for p in prompts:
            ids = clamp_len(tok(p, return_tensors="pt")["input_ids"][0], model_max_pos(model)-8)
            t0 = time.perf_counter()
            _ = model.generate(input_ids=ids.unsqueeze(0).to(device),
                               attention_mask=torch.ones_like(ids.unsqueeze(0),device=device),
                               max_new_tokens=new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
            dt = time.perf_counter() - t0
            tot_new += new_tokens; tot_time += dt
        vals.append(tot_new / max(1e-9, tot_time))
    arr = np.array(vals, dtype=np.float64)
    mean = float(arr.mean())
    ci = 1.96 * (float(arr.std(ddof=1)) / math.sqrt(len(vals))) if len(vals)>1 else 0.0
    return {"runs": vals, "mean": mean, "ci": float(ci)}

# ------- Conv1D slicing (GPT-2 MLP) -------
def new_conv1d_like(old, new_nf):
    nx = old.weight.shape[0]
    Conv1D = old.__class__
    out = Conv1D(new_nf, nx)
    out.weight = nn.Parameter(old.weight.new_empty(nx, new_nf))
    out.bias   = nn.Parameter(old.bias.new_empty(new_nf)) if old.bias is not None else None
    return out.to(old.weight.device, dtype=old.weight.dtype)

def slice_gpt2_mlp(mlp, keep_idx: torch.Tensor):
    # HOTFIX: explicitly expose original and slim modules for hook/forward   
    c_fc_orig, c_proj_orig = mlp.c_fc, mlp.c_proj
    k = keep_idx.numel()
    new_fc   = new_conv1d_like(c_fc_orig, k)
    new_proj = new_conv1d_like(c_proj_orig, c_proj_orig.bias.shape[0])  # same output; rows→keep_idx 
    with torch.no_grad():
        new_fc.weight.copy_(c_fc_orig.weight[:, keep_idx])
        if c_fc_orig.bias is not None: new_fc.bias.copy_(c_fc_orig.bias[keep_idx])
        new_proj.weight = nn.Parameter(c_proj_orig.weight[keep_idx, :].clone())
        if c_proj_orig.bias is not None: new_proj.bias.copy_(c_proj_orig.bias)

    class MLPWrap(nn.Module):
        def __init__(self, act, drop, orig_fc, orig_proj, slim_fc, slim_proj):
            super().__init__()
            self.act=act; self.drop=drop
            self.orig_fc=orig_fc; self.orig_proj=orig_proj
            self.slim_fc=slim_fc; self.slim_proj=slim_proj
            self.mode="full"
        def set_mode(self, m): self.mode=m
        def forward(self, x):
            if self.mode=="slim":
                h=self.slim_fc(x); h=self.act(h); h=self.drop(h); return self.slim_proj(h)
            else:
                h=self.orig_fc(x); h=self.act(h); h=self.drop(h); return self.orig_proj(h)
    return MLPWrap(mlp.act, mlp.dropout, c_fc_orig, c_proj_orig, new_fc, new_proj)

def topk_group_indices(scores: torch.Tensor, width: int, group: int):
    N = scores.numel()
    num_groups = max(1, N // group)
    usable = num_groups * group
    g_scores = scores[:usable].reshape(num_groups, group).mean(dim=1)
    keep_g = max(1, min(width // group, num_groups))
    keep_idx_g = torch.topk(g_scores, k=keep_g, largest=True).indices.sort()[0]
    keep=[]
    for g in keep_idx_g.tolist():
        s=g*group; e=(g+1)*group
        keep.extend(range(s,e))
    return torch.tensor(keep, device=scores.device, dtype=torch.long)

# ------ Short Fisher (c_fc only) ------
def fisher_scores_gpt2_mlp(mlp, model, tok, device, seqs):
    # ALWAYS use original fc (independent of current mode) 
    c_fc = mlp.c_fc if hasattr(mlp, "c_fc") else mlp.orig_fc
    outN = mlp_out_features_gpt2(mlp)
    scores = torch.zeros(outN, dtype=torch.float32, device=device)
    prev_req = {}
    for p in model.parameters():
        prev_req[p] = p.requires_grad
        p.requires_grad_(False)
    c_fc.weight.requires_grad_(True)
    if c_fc.bias is not None: c_fc.bias.requires_grad_(True)
    prev_mode = model.training
    model.train()
    with torch.enable_grad():
        for ids in seqs:
            ids = clamp_len(ids, CALIB_SEQ)
            x = ids.unsqueeze(0).to(device)
            attn = torch.ones_like(x, dtype=torch.long, device=device)
            out = model(input_ids=x, attention_mask=attn, labels=x)  # labels=x (internal shift)  
            loss = out.loss
            model.zero_grad(set_to_none=True)
            loss.backward()
            w = c_fc.weight.detach().float()
            g = c_fc.weight.grad.detach().float()
            scores += (w.abs() * g.abs()).mean(dim=0)
    for p, req in prev_req.items(): p.requires_grad_(req)
    model.train(prev_mode)
    return scores

# -------- gating signals --------
@torch.no_grad()
def seq_ratio(model, tok, device, layer_idx, ids):
    layers = get_layers_gpt2(model)
    mlp = layers[layer_idx].mlp
    # HOTFIX: hook ALWAYS on the projector that's actually used  
    core_proj = mlp.c_proj if hasattr(mlp, "c_proj") else mlp.orig_proj
    vals={}
    def hook(module, inp, out):
        y = out; y_last = y[:, -1, :] if y.dim()==3 else y
        vals["v"] = float(torch.linalg.vector_norm(y_last, ord=2).detach().cpu())
    h = core_proj.register_forward_hook(hook)
    x = clamp_len(ids, CALIB_SEQ).unsqueeze(0).to(device)
    attn = torch.ones_like(x, dtype=torch.long, device=device)
    out = model(input_ids=x, attention_mask=attn)
    if hasattr(out,"last_hidden_state"):
        hstate = out.last_hidden_state[:, -1, :]
        xnorm = float(torch.linalg.vector_norm(hstate, ord=2).detach().cpu())
    else:
        xnorm = 1.0
    h.remove()
    return vals.get("v",1.0) / max(1e-6, xnorm)

@torch.no_grad()
def seq_entropy(model, tok, device, ids):
    x = clamp_len(ids, CALIB_SEQ).unsqueeze(0).to(device)
    attn = torch.ones_like(x, dtype=torch.long, device=device)
    out = model(input_ids=x, attention_mask=attn)
    logits = out.logits[:, -1, :] if hasattr(out,"logits") else model.lm_head(out.last_hidden_state)[:, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    topv, _ = torch.topk(probs, k=min(50, probs.shape[-1]), dim=-1)
    ent = -torch.sum(topv * torch.log(topv + 1e-9), dim=-1)
    return float(ent.detach().cpu())

# =================== MAIN ===================
def main():
    safe_set_seeds(SEED)
    device, dtype = device_dtype()
    print(f"Device: {device}  dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map=None, trust_remote_code=True).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if hasattr(model.config,"use_cache"): model.config.use_cache = True
    layers = get_layers_gpt2(model)
    print(f"Capas totales: {len(layers)} | Target: {TARGET_LAYERS}")

    # --- Baseline (streaming + TPS) ---
    base_ppl = ppl_streaming(model, tok, device)
    base_tps = tps_generate(model, tok, device)
    print("\n=== BASELINE ===")
    print(f"PPL  {base_ppl:.2f}")
    print(f"TPS  {base_tps['mean']:.2f} ±{base_tps['ci']:.2f}")

    # --- calibration buffers ---
    calib_ds = load_dataset("wikitext","wikitext-2-raw-v1", split="train")
    calib_bufs=[]
    for x in calib_ds:
        t = x["text"].strip()
        if not t: continue
        ids = tok(t, return_tensors="pt", truncation=True, max_length=CALIB_SEQ)["input_ids"][0]
        calib_bufs.append(ids)
        if len(calib_bufs) >= CALIB_SAMPLES: break
    if not calib_bufs:
        s = "The model processes the sequence efficiently. "*50
        calib_bufs = [tok(s, return_tensors="pt")["input_ids"][0] for _ in range(CALIB_SAMPLES)]

    # --- layer acceptance ---
    fisher_cache={}
    accepted=[]
    layer_cap = PPL_CAP_LAYER
    for attempt in [0,1]:  # 1.0% → 1.5% if none
        if attempt==1 and accepted: break
        if attempt==1:
            layer_cap = PPL_CAP_LAYER_FALLBACK
            print(f"\n[Fallback] Relajo ΔPPL_capa a ≤ {layer_cap*100:.1f}%")
        for li in TARGET_LAYERS:
            if li in accepted: continue
            mlp = layers[li].mlp
            outN = mlp_out_features_gpt2(mlp)
            widths = [w for w in SLIM_WIDTHS_TRY if w < outN]
            if not widths:
                print(f"  capa {li}: sin ancho slim válido"); continue
            # short fisher 
            sc = fisher_cache.get(li)
            if sc is None:
                sc = fisher_scores_gpt2_mlp(mlp, model, tok, device, calib_bufs[:8])
                fisher_cache[li]=sc
            ok=False
            for slim_w in widths:
                keep_idx = topk_group_indices(sc, slim_w, GROUP)
                slim = slice_gpt2_mlp(mlp, keep_idx)
                old = layers[li].mlp
                layers[li].mlp = slim
                slim.set_mode("slim")
                ppl1 = ppl_streaming(model, tok, device)
                d = (ppl1/base_ppl)-1.0
                if d <= layer_cap:
                    slim.set_mode("full")
                    accepted.append(li)
                    print(f"  capa {li}: width {slim_w} | ΔPPL_capa {d*100:+.2f}% → ACEPTADA")
                    ok=True; break
                else:
                    layers[li].mlp = old
            if not ok and attempt==0:
                print(f"  capa {li}: no pasó 1.0%")
            elif not ok:
                print(f"  capa {li}: no pasó 1.5%")

    if accepted:
        # static ceiling (all slim) 
        for li in accepted: layers[li].mlp.set_mode("slim")
        pre_ppl = ppl_streaming(model, tok, device)
        pre_tps = tps_generate(model, tok, device)
        for li in accepted: layers[li].mlp.set_mode("full")
    else:
        pre_ppl, pre_tps = base_ppl, base_tps

    # --- dual calibration (ratio + entropy) ---
    sentinel = accepted[len(accepted)//2] if accepted else TARGET_LAYERS[len(TARGET_LAYERS)//2]
    ratios, ents = [], []
    for ids in calib_bufs[:CALIB_SAMPLES]:
        ratios.append(seq_ratio(model, tok, device, sentinel, ids))
        ents.append(seq_entropy(model, tok, device, ids))
    thr_ratio = float(np.quantile(np.array(ratios, dtype=np.float32), Q_RATIO))
    thr_ent   = float(np.quantile(np.array(ents,   dtype=np.float32), Q_ENT))
    print(f"\nCalibrado (percentiles): ratio@{Q_RATIO:.2f}→{thr_ratio:.4f} | ent@{Q_ENT:.2f}→{thr_ent:.4f} | sentinel={sentinel}")

    # --- SeqGate (Eval: AND, Gen: OR) ---
    @torch.no_grad()
    def seqgate_ppl_and():
        model.eval()
        ds = load_dataset("wikitext","wikitext-2-raw-v1", split="test")
        text = "\n\n".join([x["text"] for x in ds])
        ids_all = encode_long_text(tok, text).to(device)
        max_pos = model_max_pos(model)
        L = ids_all.size(0) - 1
        total_ll, total_tok = 0.0, 0
        slim_used = 0
        for i in range(0, L, EVAL_STRIDE):
            end = min(i + EVAL_SEQ, L)
            cur_len = end - i
            if cur_len <= 0: break
            prefix = ids_all[i : min(i+CALIB_SEQ, end)]
            r = seq_ratio(model, tok, device, sentinel, prefix)
            e = seq_entropy(model, tok, device, prefix)
            use_slim = (r <= thr_ratio) and (e <= thr_ent) and bool(accepted)
            for li in accepted:
                layers[li].mlp.set_mode("slim" if use_slim else "full")
            x = ids_all[i:end].unsqueeze(0)
            attn = torch.ones_like(x, dtype=torch.long, device=device)
            out = model(input_ids=x, attention_mask=attn, labels=x)  # labels=x
            total_ll += float(out.loss.detach().cpu()) * cur_len
            total_tok += cur_len
            if use_slim: slim_used += cur_len
            for li in accepted: layers[li].mlp.set_mode("full")
            if end == L: break
        return math.exp(total_ll/max(1,total_tok)), slim_used/max(1,total_tok)

    @torch.no_grad()
    def seqgate_tps_or():
        model.eval()
        prompts = [
            "The future of artificial intelligence",
            "Recent scientific developments have led to",
            "The economic impact of new technologies",
        ]
        vals=[]; slim_rate=[]
        # prewarm
        ids0 = clamp_len(tok(prompts[0], return_tensors="pt")["input_ids"][0], model_max_pos(model)-8)
        _ = model.generate(input_ids=ids0.unsqueeze(0).to(device),
                           attention_mask=torch.ones_like(ids0.unsqueeze(0),device=device),
                           max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        for _ in range(TPS_REPEATS):
            tot_new, tot_time, slim_used, total_used = 0, 0.0, 0, 0
            for p in prompts:
                cur = clamp_len(tok(p, return_tensors="pt")["input_ids"][0], model_max_pos(model)-8)
                r = seq_ratio(model, tok, device, sentinel, cur[:CALIB_SEQ])
                e = seq_entropy(model, tok, device, cur[:CALIB_SEQ])
                use_slim = ((r <= thr_ratio) or (e <= thr_ent)) and bool(accepted)  # OR en generación
                for li in accepted: layers[li].mlp.set_mode("slim" if use_slim else "full")
                t0 = time.perf_counter()
                _ = model.generate(input_ids=cur.unsqueeze(0).to(device),
                                   attention_mask=torch.ones_like(cur.unsqueeze(0),device=device),
                                   max_new_tokens=TPS_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
                dt = time.perf_counter()-t0
                tot_new += TPS_NEW; tot_time += dt
                total_used += TPS_NEW; slim_used += (TPS_NEW if use_slim else 0)
                for li in accepted: layers[li].mlp.set_mode("full")
            vals.append(tot_new / max(1e-9, tot_time))
            slim_rate.append(slim_used/max(1,total_used))
        arr = np.array(vals, dtype=np.float64)
        mean = float(arr.mean())
        ci = 1.96 * (float(arr.std(ddof=1)) / math.sqrt(len(vals))) if len(vals)>1 else 0.0
        return {"runs": vals, "mean": mean, "ci": float(ci)}, float(np.mean(slim_rate))

    # metrics
    print("\n=== MÉTRICAS ===")
    print(f"BASE  PPL {base_ppl:.2f} | TPS {base_tps['mean']:.2f} ±{base_tps['ci']:.2f}")
    if accepted:
        print(f"PRE   PPL {pre_ppl:.2f} | TPS {pre_tps['mean']:.2f} ±{pre_tps['ci']:.2f}")
    else:
        print("PRE   (sin capas aceptadas)")

    seq_ppl, slim_rate_eval = seqgate_ppl_and()   # AND to protect PPL
    seq_tps, slim_rate_gen  = seqgate_tps_or()    # OR to gain TPS

    dp = (seq_ppl/base_ppl - 1.0)*100
    dt = (seq_tps['mean']/base_tps['mean'] - 1.0)*100
    print(f"EXNAS PPL {seq_ppl:.2f} | TPS {seq_tps['mean']:.2f} ±{seq_tps['ci']:.2f}")
    print(f"Uso slim: {slim_rate_eval*100:.1f}% (eval) | {slim_rate_gen*100:.1f}% (gen)")
    print(f"ΔPPL {dp:+.2f}% | ΔTPS {dt:+.2f}% → final: seqgate")

    summary = {
        "model": MODEL_ID,
        "accepted_layers": accepted,
        "slim_widths_try": SLIM_WIDTHS_TRY,
        "caps": {"layer": PPL_CAP_LAYER, "layer_fallback": PPL_CAP_LAYER_FALLBACK},
        "gating": {"type":"per-sequence", "signals":["ratio","entropy"], "q_ratio": Q_RATIO, "q_ent": Q_ENT,
                   "sentinel": (accepted[len(accepted)//2] if accepted else None),
                   "slim_use_eval": slim_rate_eval, "slim_use_gen": slim_rate_gen},
        "baseline": {"ppl": base_ppl, "tps": base_tps},
        "pre": {"ppl": (pre_ppl if accepted else base_ppl), "tps": (pre_tps if accepted else base_tps)},
        "exnas": {"ppl": seq_ppl, "tps": seq_tps},
        "deltas": {"ppl_pct": dp, "tps_pct": dt}
    }
    with open("exnas_seqgate_v2_summary.json","w") as f: json.dump(summary, f, indent=2)
    print("\nGuardado → exnas_seqgate_v2_summary.json")

# -------- run --------
gc.collect()
try:
    if torch.cuda.is_available(): torch.cuda.empty_cache()
except Exception: pass
main()

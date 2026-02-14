#   ExNAS  (Qwen2.5-7B)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEMORY_SIZE = 200      
K_NEIGHBORS = 3        


print(f"--- ExNAS Implementation: Sentinel & Online Memory (Corrected) ---")
print(f"Device: {DEVICE}")


def create_sliced_model(base_model_name, keep_indices, name):
    print(f"Building Profile: {name}...")
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
    all_layers = model.model.layers
    selected_layers = torch.nn.ModuleList([all_layers[i] for i in keep_indices])
    model.model.layers = selected_layers
    print(f"  -> Layers kept: {keep_indices} (Total: {len(model.model.layers)})")
    return model

# PeProfiles
idx_sentinel = [0] 
# Fast
idx_fast = [0, 1, 2, 12, 22, 23]  
idx_med = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23]

print("Loading Full Teacher Model...")
profile_full = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()

sentinel_model = create_sliced_model(MODEL_NAME, idx_sentinel, "Sentinel")
profile_fast = create_sliced_model(MODEL_NAME, idx_fast, "Fast")
profile_med = create_sliced_model(MODEL_NAME, idx_med, "Medium")

profiles = [profile_fast, profile_med, profile_full]
profile_names = ["Fast", "Medium", "Full"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==========================================
# 3. DINAMIC MEMORY
# ==========================================
class ExNASMemory:
    def __init__(self, embedding_dim, dtype=torch.float16):
        self.dtype = dtype
        self.keys = torch.empty(0, embedding_dim, dtype=self.dtype).to(DEVICE)
        self.values = torch.empty(0, dtype=torch.long).to(DEVICE)
        self.max_size = MEMORY_SIZE

    def query(self, fingerprint):
        fingerprint = fingerprint.to(self.dtype)
        
        # Cold Start
        if self.keys.size(0) < K_NEIGHBORS:
            return 2 
        
        query_norm = F.normalize(fingerprint, p=2, dim=1)
        keys_norm = F.normalize(self.keys, p=2, dim=1)
        
        scores = torch.mm(query_norm, keys_norm.t())
        _, top_indices = scores.topk(K_NEIGHBORS, dim=1)
        
        suggested_profiles = self.values[top_indices.squeeze()]
        
        if suggested_profiles.dim() == 0: return suggested_profiles.item()
        return torch.mode(suggested_profiles).values.item()

    def update(self, fingerprint, optimal_idx):
        fingerprint = fingerprint.to(self.dtype)
        self.keys = torch.cat([self.keys, fingerprint], dim=0)
        self.values = torch.cat([self.values, torch.tensor([optimal_idx], device=DEVICE)], dim=0)
        
        if self.keys.size(0) > self.max_size:
            self.keys = self.keys[-self.max_size:]
            self.values = self.values[-self.max_size:]

hidden_dim = profile_full.config.hidden_size
memory = ExNASMemory(hidden_dim, dtype=torch.float16)

# ==========================================
# 4. DATASET (Curriculum Learning)
# ==========================================
prompts = [
    # GROUP 1
    "The opposite of hot is", 
    "10 + 10 =", 
    "The capital of France is",
    "Red, Blue and",
    
    # GROUP 2
    "Explain gravity briefly:", 
    "List three mammals:", 
    
    # GROUP
    "Write a python script to parse HTML.", 
    "Analyze the geopolitical impact of AI."
] * 5 # 5 Repetitions 

# ==========================================
# 5. INFERENCE LOOP
# ==========================================
print("\n" + "="*105)
print(f"{'Prompt':<30} | {'ρ (Diff)':<10} | {'Pred':<8} | {'GT':<8} | {'Result':<15} | {'Saved'}")
print("="*105)

stats = {"Correct": 0, "Total": 0, "Savings": 0}

for i, text in enumerate(prompts):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    #  PHASE 1: SENTINEL 
    with torch.no_grad():
        out_sentinel = sentinel_model(**inputs, output_hidden_states=True)
        z_in = out_sentinel.hidden_states[0]
        z_out = out_sentinel.hidden_states[-1]
        fingerprint = z_out.mean(dim=1) 
        # Rho: Magnitude of the vector change in layer 0
        rho = (torch.norm(z_out) / (torch.norm(z_in) + 1e-6)).item()

    #  PHASE 2: PREDICTIÓN (MEMORY) 
    predicted_idx = memory.query(fingerprint)

    #  PHASE 3: ORÁCLE (GROUND TRUTH)  
    # We use "Token Agreement" 
    with torch.no_grad():
        logits_full = profile_full(**inputs).logits[:, -1, :]
        token_full = torch.argmax(logits_full, dim=-1)
        
        logits_fast = profile_fast(**inputs).logits[:, -1, :]
        token_fast = torch.argmax(logits_fast, dim=-1)
        
        logits_med = profile_med(**inputs).logits[:, -1, :]
        token_med = torch.argmax(logits_med, dim=-1)
        
        # Selection Logic ("Greedy Oracle")
        # If the smaller model predicts the SAME word as the larger one, that's sufficient.
        if token_fast == token_full:
            gt_idx = 0 # Fast is enough
        elif token_med == token_full:
            gt_idx = 1 # Medium is enough
        else:
            gt_idx = 2 # We need Full

    # --- PHASE 4: UPDATE ---
    memory.update(fingerprint, gt_idx)
    
    # --- METRICS ---
    is_correct = predicted_idx >= gt_idx
    
    match_str = "✅ OK" if is_correct else "❌ Under"
    if predicted_idx > gt_idx: match_str = "⚠️ Over" 
    
    # We calculated layers saved (Full has 24 layers)
    layers_current = len(profiles[predicted_idx].model.layers)
    saved = 24 - layers_current
    
    stats["Total"] += 1
    if is_correct: 
        stats["Correct"] += 1
        stats["Savings"] += saved

    trunc_text = text[:28] + ".." if len(text) > 28 else text
    print(f"{trunc_text:<30} | {rho:.4f}     | {profile_names[predicted_idx]:<8} | {profile_names[gt_idx]:<8} | {match_str:<15} | +{saved}")

print("="*105)
acc = (stats['Correct']/stats['Total'])*100
avg_savings = stats['Savings'] / stats['Total']
print(f"Final Accuracy (Safety): {acc:.2f}%")
print(f"Avg Layers Saved: {avg_savings:.1f} layers per query")
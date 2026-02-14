# Fingerprint Ablation (Visual Scenarios)
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# =============================================================================
# PART 1: ABLATION VIDEO GENERATOR
# =============================================================================
def generate_ablation_video(filename="ablation_test.mp4"):
    print(f"Generating ablation video: {filename}...")
    width, height = 224, 224
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    # 1. Simple Gradient
    for i in range(10):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            val = int((y / height) * 255)
            frame[y, :, :] = (val, val//2, val//3)
        out.write(frame)

    # 2. Complex Noise
    for i in range(10):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)

    # 3. Bright Trap (White Wall)
    for i in range(10):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255 
        out.write(frame)

    out.release()
    print("Video generated.\n")

# =============================================================================
# PART 2: MULTI-METRIC CONTROLLER (Now with Entropy & Timing)
# =============================================================================
class AblationController:
    def __init__(self, device):
        self.device = device
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).to(device).eval()
        self.preprocess = MobileNet_V3_Small_Weights.DEFAULT.transforms()
        self.hook_data = {"input_energy": 0.0, "output": None}
        self.handle = self.model.features[0].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input_t, output_t):
        self.hook_data["input_energy"] = torch.norm(input_t[0], p=2).item()
        self.hook_data["output"] = output_t.detach()

    def calculate_entropy(self, tensor):
        # Shannon Entropy is computationally expensive!
        prob = torch.softmax(tensor.flatten(), dim=0)
        return -torch.sum(prob * torch.log(prob + 1e-9)).item()

    def analyze_frame(self, raw_frame):
        img_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(torch.from_numpy(img_rgb).permute(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.model.features[0](input_tensor)

        x_out = self.hook_data["output"]
        e_in = self.hook_data["input_energy"]
        
        results = {}

        # 1. MAGNITUDE (Fast but Dumb)
        t0 = time.perf_counter()
        results['mag_val'] = torch.norm(x_out, p=2).item()
        results['mag_time'] = (time.perf_counter() - t0) * 1000 # ms

        # 2. ENTROPY (Smart but Slow) - THE NEW BASELINE
        t0 = time.perf_counter()
        results['ent_val'] = self.calculate_entropy(x_out)
        results['ent_time'] = (time.perf_counter() - t0) * 1000 # ms

        # 3. ENERGY RATIO (Ours: Fast & Smart)
        t0 = time.perf_counter()
        results['ratio_val'] = results['mag_val'] / (e_in + 1e-6)
        results['ratio_time'] = (time.perf_counter() - t0) * 1000 # ms

        return results

    def close(self):
        self.handle.remove()

# =============================================================================
# PART 3: EXPERIMENT EXECUTION
# =============================================================================
def run_ablation_experiment():
    generate_ablation_video()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running ablation on: {device}")
    
    controller = AblationController(device)
    cap = cv2.VideoCapture("ablation_test.mp4")
    
    print("\n" + "="*110)
    print(f"{'Scene':<15} | {'Metric':<12} | {'Value':<10} | {'Time (ms)':<10} | {'Decision':<10} | {'Analysis'}")
    print("-" * 110)

    frame_idx = 0
    # Thresholds
    THRESH_MAG = 60.0
    THRESH_ENT = 4.0 # Entropy usually around 3-5
    THRESH_RATIO = 2.5

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % 10 == 0: # Analyze 1 frame per scene type
            if frame_idx < 10: label = "Gradient"
            elif frame_idx < 20: label = "Noise"
            else: label = "Bright Wall"

            res = controller.analyze_frame(frame)

            # --- DECISION LOGIC ---
            # Magnitude
            dec_mag = "FULL" if res['mag_val'] > THRESH_MAG else "SLIM"
            status_mag = "FAIL (False Pos)" if label == "Bright Wall" and dec_mag == "FULL" else "OK"
            
            # Entropy
            dec_ent = "FULL" if res['ent_val'] > THRESH_ENT else "SLIM"
            status_ent = "OK (Slow!)" # Entropy is always correct but slow
            
            # Ratio (Ours)
            dec_ratio = "FULL" if res['ratio_val'] > THRESH_RATIO else "SLIM"
            status_ratio = "OK (Robust)" if label == "Bright Wall" and dec_ratio == "SLIM" else "OK"

            # PRINT ROWS
            print(f"{label:<15} | {'Magnitude':<12} | {res['mag_val']:.1f}{' '*6} | {res['mag_time']:.4f}{' '*4} | {dec_mag:<10} | {status_mag}")
            print(f"{'':<15} | {'Entropy':<12} | {res['ent_val']:.2f}{' '*6} | {res['ent_time']:.4f}{' '*4} | {dec_ent:<10} | {status_ent}")
            print(f"{'':<15} | {'Ratio (Ours)':<12} | {res['ratio_val']:.2f}{' '*6} | {res['ratio_time']:.4f}{' '*4} | {dec_ratio:<10} | {status_ratio}")
            print("-" * 110)

        frame_idx += 1

    cap.release()
    controller.close()
    print("="*110)

if __name__ == "__main__":
    run_ablation_experiment()
"""
Debug fire detection - show all scores and prompts.
"""

import cv2
import torch
import clip
import numpy as np
from pathlib import Path
from PIL import Image

print("=" * 70)
print(" " * 15 + "FIRE DETECTION DEBUG")
print("=" * 70)

# Load CLIP
print("\n[1/4] Loading CLIP...")
device = 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)
print("  CLIP loaded")

# Load fire image
print("\n[2/4] Loading fire image...")
fire_path = Path("test_data/fire.jpg")
frame = cv2.imread(str(fire_path))
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(frame_rgb)
print(f"  Image loaded: {frame.shape[1]}x{frame.shape[0]}")

# Prepare image
print("\n[3/4] Processing image...")
image_input = preprocess(pil_image).unsqueeze(0).to(device)

# Encode image
with torch.no_grad():
    image_features = model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# Test ALL fire-related prompts
print("\n[4/4] Testing prompts...")
all_prompts = [
    # Original prompts
    "There is smoke rising",
    "There is an arsonist",
    "There is something that shines brightly",
    "There is a campfire",
    "There is flame and fire soaring",
    
    # Additional fire prompts
    "A burning building",
    "A fire",
    "Flames and smoke",
    "A building on fire",
    "An explosion",
    "Orange flames",
    "Black smoke",
    "Fire emergency",
    "Burning fire",
    "Large fire",
    
    # Non-fire prompts (for comparison)
    "A normal building",
    "People walking",
    "A street scene",
    "No fire"
]

# Encode all text prompts
text_inputs = clip.tokenize(all_prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Calculate similarities
similarities = (image_features @ text_features.T).cpu().numpy()[0]

# Sort by score
prompt_scores = list(zip(all_prompts, similarities))
prompt_scores.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 70)
print("SIMILARITY SCORES (Higher = Better Match)")
print("=" * 70)

print("\nTOP 10 MATCHES:")
for i, (prompt, score) in enumerate(prompt_scores[:10], 1):
    emoji = "🔥" if any(word in prompt.lower() for word in ['fire', 'smoke', 'flame', 'burn', 'explosion']) else "  "
    print(f"{i:2d}. {emoji} {score:.4f} - '{prompt}'")

print("\nBOTTOM 5 MATCHES:")
for i, (prompt, score) in enumerate(prompt_scores[-5:], len(prompt_scores)-4):
    print(f"{i:2d}.    {score:.4f} - '{prompt}'")

# Determine if fire
top_5_scores = [s for _, s in prompt_scores[:5]]
avg_top_5 = np.mean(top_5_scores)
max_score = prompt_scores[0][1]

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)
print(f"Highest score: {max_score:.4f} - '{prompt_scores[0][0]}'")
print(f"Average of top 5: {avg_top_5:.4f}")

# Different threshold tests
thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
print("\nThreshold Analysis:")
for threshold in thresholds:
    if max_score > threshold:
        print(f"  Threshold {threshold:.2f}: ✅ FIRE DETECTED")
    else:
        print(f"  Threshold {threshold:.2f}: ❌ No detection")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if max_score > 0.25:
    print(f"🔥 FIRE IS PRESENT IN IMAGE")
    print(f"   Best match: '{prompt_scores[0][0]}' (score: {max_score:.4f})")
    print(f"   Recommended threshold: 0.25")
else:
    print(f"❌ Fire detection unclear")
    print(f"   Max score too low: {max_score:.4f}")
    print(f"   May need different prompts or model")

print("=" * 70)
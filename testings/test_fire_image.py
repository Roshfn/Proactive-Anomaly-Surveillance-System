"""
Test arson detection with actual fire image.
"""

import cv2
from pathlib import Path
from src.stage2_anomaly_recognition.scene_analyzer import SceneAnalyzer
from src.stage1_human_tracking.tracklet import Tracklet
import torch

print("=" * 70)
print(" " * 15 + "FIRE IMAGE TEST")
print("=" * 70)

# Initialize scene analyzer
print("\n[1/3] Loading CLIP model...")
scene_analyzer = SceneAnalyzer(device='cpu')
print("  CLIP loaded")

# Load fire image
print("\n[2/3] Loading fire image...")
fire_image_path = Path("test_data/fire.jpg")

if not fire_image_path.exists():
    print(f"  Fire image not found at {fire_image_path}")
    print("  Please save uploaded fire.jpg to test_data/fire.jpg")
    exit(1)

frame = cv2.imread(str(fire_image_path))
print(f"  Image loaded: {frame.shape[1]}x{frame.shape[0]}")

# Analyze for fire (no tracklets needed for frame-level detection)
print("\n[3/3] Analyzing for fire...")
arson_events = scene_analyzer.detect_arson(frame, [], fire_threshold=0.25)

print(f"\nResults:")
print(f"  Arson events detected: {len(arson_events)}")

if arson_events:
    for event in arson_events:
        print(f"\n  🔥 FIRE DETECTED!")
        print(f"    Fire score: {event['fire_score']:.3f}")
        print(f"    Confidence: {event['fire_score']*100:.1f}%")
        print(f"    Matched prompt: '{event['matched_prompt']}'")
        print(f"    Patches analyzed: {event['num_patches_analyzed']}")
else:
    print("\n  No fire detected")

# Save result
output_frame = frame.copy()

if arson_events:
    status = f"FIRE DETECTED! Score: {arson_events[0]['fire_score']:.2f}"
    color = (0, 0, 255)
    cv2.putText(output_frame, status, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(output_frame, arson_events[0]['matched_prompt'], (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

output_path = Path("outputs/fire_detection_result.jpg")
cv2.imwrite(str(output_path), output_frame)
print(f"\n  Output saved: {output_path}")

print("\n" + "=" * 70)
print(" " * 15 + "FIRE DETECTION TEST COMPLETE")
print("=" * 70)
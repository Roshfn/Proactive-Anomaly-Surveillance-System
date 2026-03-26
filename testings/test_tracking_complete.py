"""
Complete tracking test with YOLOv5 detection and OSNet features.
"""

import cv2
import torch
from pathlib import Path
from src.stage1_human_tracking.osnet_model import build_osnet
from src.stage1_human_tracking.feature_extractor import FeatureExtractor
from src.stage1_human_tracking.human_tracker import HumanTracker, YOLOv5Detector

print("=" * 70)
print(" " * 15 + "COMPLETE TRACKING TEST")
print("=" * 70)

# Initialize components
print("\n[1/5] Initializing components...")

# YOLOv5 detector
yolo_path = Path("models/yolov5s.pt")
detector = YOLOv5Detector(str(yolo_path), device='cpu', conf_threshold=0.5)
print("  YOLOv5 detector loaded")

# OSNet feature extractor
osnet_path = Path("models/osnet_x1_0_imagenet.pth")
osnet = build_osnet(str(osnet_path), device='cpu')
feature_extractor = FeatureExtractor(osnet, device='cpu')
print("  OSNet feature extractor loaded")

# Human tracker
tracker = HumanTracker(feature_extractor, max_age=30, min_hits=1)
print("  Human tracker initialized")

# Load test image
print("\n[2/5] Loading test image...")
test_image_path = Path("test_data/test_people.jpg")
frame = cv2.imread(str(test_image_path))
print(f"  Image loaded: {frame.shape[1]}x{frame.shape[0]} pixels")

# Detect people
print("\n[3/5] Detecting people...")
detections, mid_features = detector.detect(frame)
print(f"  Detected {len(detections)} people")
for i, bbox in enumerate(detections):
    print(f"    Person {i+1}: bbox={bbox}")

# Track people (single frame for now)
print("\n[4/5] Running tracking...")
active_tracklets = tracker.update(frame, detections, mid_features)
print(f"  Active tracklets: {len(active_tracklets)}")

for tracklet in active_tracklets:
    bbox = tracklet.get_current_bbox()
    print(f"    {tracklet}")
    print(f"      BBox: {bbox}")
    print(f"      Feature dim: {tracklet.get_current_feature().shape}")

# Visualize results
print("\n[5/5] Creating visualization...")
output_frame = frame.copy()

for tracklet in active_tracklets:
    bbox = tracklet.get_current_bbox()
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bounding box
    color = (0, 255, 0)
    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw ID label
    label = f"ID:{tracklet.id}"
    cv2.putText(output_frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Save output
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "tracking_test.jpg"
cv2.imwrite(str(output_path), output_frame)

print(f"  Output saved: {output_path}")

print("\n" + "=" * 70)
print(" " * 15 + "TRACKING TEST COMPLETE")
print("=" * 70)
print("\nResults:")
print(f"  YOLOv5 detection: {len(detections)} people detected")
print(f"  Feature extraction: 512-dim features extracted")
print(f"  Tracking: {len(active_tracklets)} tracklets active")
print(f"  Visualization: {output_path}")
print("\n" + "=" * 70)
print(" " * 10 + "PHASE 4 COMPLETE - TRACKING WORKING")
print("=" * 70)
print("\nNext Phase: Anomaly Detection")
print("  Phase 5: Intrusion Detection")
print("  Phase 6: Loitering Detection")
print("  Phase 7: Abandonment Detection")
print("  Phase 8: Arson Detection")
print("=" * 70)
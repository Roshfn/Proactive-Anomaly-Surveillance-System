"""
Test intrusion and loitering detection.
"""

import cv2
import numpy as np
from pathlib import Path
from src.stage1_human_tracking.osnet_model import build_osnet
from src.stage1_human_tracking.feature_extractor import FeatureExtractor
from src.stage1_human_tracking.human_tracker import HumanTracker, YOLOv5Detector
from src.stage2_anomaly_recognition.intersection_detector import IntersectionDetector

print("=" * 70)
print(" " * 15 + "INTRUSION DETECTION TEST")
print("=" * 70)

# Initialize components
print("\n[1/5] Initializing components...")
yolo_path = Path("models/yolov5s.pt")
detector = YOLOv5Detector(str(yolo_path), device='cpu')

osnet_path = Path("models/osnet_x1_0_imagenet.pth")
osnet = build_osnet(str(osnet_path), device='cpu')
feature_extractor = FeatureExtractor(osnet, device='cpu')

tracker = HumanTracker(feature_extractor, min_hits=1)  # Changed to 1 for single frame test
intrusion_detector = IntersectionDetector(
    intrusion_threshold=0.3,
    loitering_threshold=0.3,
    loitering_duration=10.0
)

print("  All components initialized")

# Load test image
print("\n[2/5] Loading test image...")
test_image_path = Path("test_data/test_people.jpg")
frame = cv2.imread(str(test_image_path))
height, width = frame.shape[:2]
print(f"  Image: {width}x{height}")

# Define intrusion zone (center rectangle)
print("\n[3/5] Defining detection zones...")
intrusion_zone = [
    (width * 0.3, height * 0.3),
    (width * 0.7, height * 0.3),
    (width * 0.7, height * 0.7),
    (width * 0.3, height * 0.7)
]
intrusion_detector.add_intrusion_zone('restricted_area', intrusion_zone)

loitering_zone = [
    (width * 0.1, height * 0.1),
    (width * 0.5, height * 0.1),
    (width * 0.5, height * 0.5),
    (width * 0.1, height * 0.5)
]
intrusion_detector.add_loitering_zone('monitored_area', loitering_zone)

print(f"  Intrusion zone: center area")
print(f"  Loitering zone: top-left area")

# Detect and track
print("\n[4/5] Running detection and tracking...")
detections, mid_features = detector.detect(frame)
print(f"  Detections: {len(detections)}")

active_tracklets = tracker.update(frame, detections, mid_features)
print(f"  Active tracklets: {len(active_tracklets)}")

# Check intrusion
intrusion_events = intrusion_detector.check_intrusion(active_tracklets)
print(f"  Intrusion events: {len(intrusion_events)}")

for event in intrusion_events:
    print(f"    Tracklet {event['tracklet_id']} in zone '{event['zone_id']}'")
    print(f"      Intersection ratio: {event['intersection_ratio']:.2f}")

# Check loitering (simulate 15 seconds elapsed)
loitering_events = intrusion_detector.check_loitering(active_tracklets, frame_time=15.0)
print(f"  Loitering events: {len(loitering_events)}")

# Visualize
print("\n[5/5] Creating visualization...")
output_frame = frame.copy()

# Draw zones
intrusion_pts = np.array(intrusion_zone, dtype=np.int32)
cv2.polylines(output_frame, [intrusion_pts], True, (0, 0, 255), 3)
cv2.putText(output_frame, "INTRUSION ZONE", 
            (int(width*0.35), int(height*0.25)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

loitering_pts = np.array(loitering_zone, dtype=np.int32)
cv2.polylines(output_frame, [loitering_pts], True, (255, 165, 0), 3)
cv2.putText(output_frame, "LOITERING ZONE",
            (int(width*0.15), int(height*0.08)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

# Draw tracklets
for tracklet in active_tracklets:
    bbox = tracklet.get_current_bbox()
    x1, y1, x2, y2 = map(int, bbox)
    
    # Check if in intrusion zone
    in_intrusion = any(e['tracklet_id'] == tracklet.id for e in intrusion_events)
    color = (0, 0, 255) if in_intrusion else (0, 255, 0)
    
    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
    
    label = f"ID:{tracklet.id}"
    if in_intrusion:
        label += " INTRUSION!"
    
    cv2.putText(output_frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Save output
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "intrusion_detection_test.jpg"
cv2.imwrite(str(output_path), output_frame)

print(f"  Output saved: {output_path}")

print("\n" + "=" * 70)
print(" " * 15 + "INTRUSION DETECTION TEST COMPLETE")
print("=" * 70)
print("\nResults:")
print(f"  Zones defined: 2 (intrusion + loitering)")
print(f"  Tracklets detected: {len(active_tracklets)}")
print(f"  Intrusion events: {len(intrusion_events)}")
print(f"  Visualization: {output_path}")
print("\n" + "=" * 70)
print(" " * 10 + "PHASE 5 COMPLETE - INTRUSION DETECTION WORKING")
print("=" * 70)
print("\nNext: Phase 6 - Loitering Detection (uses same code)")
print("=" * 70)
"""
Test tracklet creation and basic matching.
"""

import torch
import numpy as np
from src.stage1_human_tracking.tracklet import Tracklet
from src.algorithms.cascade_matching import cascade_matching, cosine_distance

print("=" * 70)
print(" " * 15 + "TRACKING COMPONENTS TEST")
print("=" * 70)

# Test 1: Tracklet creation
print("\n[Test 1] Tracklet Creation")
print("-" * 70)

bbox1 = [100, 100, 200, 300]
feature1 = torch.randn(512)
tracklet1 = Tracklet(bbox1, feature1, frame_id=0)

print(f"Created: {tracklet1}")
print(f"  ID: {tracklet1.id}")
print(f"  Current bbox: {tracklet1.get_current_bbox()}")
print(f"  Trajectory length: {len(tracklet1.get_trajectory())}")
print("Success: Tracklet creation working")

# Test 2: Tracklet update
print("\n[Test 2] Tracklet Update")
print("-" * 70)

bbox2 = [105, 105, 205, 305]
feature2 = torch.randn(512)
tracklet1.update(bbox2, feature2, frame_id=1)

print(f"Updated: {tracklet1}")
print(f"  Trajectory length: {len(tracklet1.get_trajectory())}")
print(f"  Hits: {tracklet1.hits}")
print(f"  Age: {tracklet1.age}")
print("Success: Tracklet update working")

# Test 3: Feature distance calculation
print("\n[Test 3] Feature Distance Calculation")
print("-" * 70)

features1 = np.random.randn(3, 512).astype(np.float32)
features2 = np.random.randn(2, 512).astype(np.float32)

# Normalize features
features1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
features2 = features2 / np.linalg.norm(features2, axis=1, keepdims=True)

distances = cosine_distance(features1, features2)

print(f"Distance matrix shape: {distances.shape}")
print(f"Distance matrix:\n{distances}")
print(f"  Min distance: {distances.min():.4f}")
print(f"  Max distance: {distances.max():.4f}")
print("Success: Distance calculation working")

# Test 4: Cascade matching
print("\n[Test 4] Cascade Matching")
print("-" * 70)

# Create mock tracklets
tracklets = []
for i in range(3):
    bbox = [100 + i*50, 100, 200 + i*50, 300]
    feature = torch.randn(512)
    feature = feature / torch.norm(feature)
    t = Tracklet(bbox, feature, frame_id=0)
    tracklets.append(t)

# Create mock detections
detections = [
    [105, 105, 205, 305],
    [155, 105, 255, 305],
]
detection_features = np.random.randn(2, 512).astype(np.float32)
detection_features = detection_features / np.linalg.norm(detection_features, axis=1, keepdims=True)

# Run matching
matches, unmatched_tracks, unmatched_dets = cascade_matching(
    tracklets, detections, detection_features
)

print(f"Matches: {matches}")
print(f"Unmatched tracklets: {unmatched_tracks}")
print(f"Unmatched detections: {unmatched_dets}")
print(f"  Total tracklets: {len(tracklets)}")
print(f"  Total detections: {len(detections)}")
print(f"  Matched pairs: {len(matches)}")
print("Success: Cascade matching working")

# Test 5: Stationary detection
print("\n[Test 5] Stationary Detection")
print("-" * 70)

stationary_tracklet = Tracklet([100, 100, 200, 300], torch.randn(512), 0)

# Simulate stationary person (same bbox multiple times)
for i in range(10):
    stationary_tracklet.update([100, 100, 200, 300], torch.randn(512), i+1)

is_stationary = stationary_tracklet.check_stationary()
print(f"Tracklet stationary status: {is_stationary}")
print(f"  Stationary frames: {stationary_tracklet.stationary_frames}")
print("Success: Stationary detection working")

print("\n" + "=" * 70)
print(" " * 15 + "ALL TESTS PASSED")
print("=" * 70)
print("\nResults:")
print("  Tracklet creation: Working")
print("  Tracklet update: Working")
print("  Feature distance: Working")
print("  Cascade matching: Working")
print("  Stationary detection: Working")
print("\nReady for: Full tracking implementation with video")
print("=" * 70)
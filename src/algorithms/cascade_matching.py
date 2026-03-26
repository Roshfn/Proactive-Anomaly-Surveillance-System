"""
Cascade Matching Algorithm for ID Assignment.

Implements the matching strategy used in Deep SORT:
1. Match by appearance similarity (feature distance)
2. Use Hungarian algorithm for optimal assignment
3. Cascade through different stages based on time since update
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def cosine_distance(features1, features2):
    """
    Calculate cosine distance matrix between two feature sets.
    
    Args:
        features1: Tensor of shape (N, D)
        features2: Tensor of shape (M, D)
    
    Returns:
        distance_matrix: Numpy array of shape (N, M)
    """
    # Cosine similarity
    similarity = np.dot(features1, features2.T)
    
    # Convert to distance (0 = same, 2 = opposite)
    distance = 1.0 - similarity
    
    return distance


def euclidean_distance(features1, features2):
    """
    Calculate Euclidean distance matrix between two feature sets.
    
    Args:
        features1: Array of shape (N, D)
        features2: Array of shape (M, D)
    
    Returns:
        distance_matrix: Array of shape (N, M)
    """
    distances = np.zeros((len(features1), len(features2)))
    
    for i, feat1 in enumerate(features1):
        for j, feat2 in enumerate(features2):
            distances[i, j] = np.linalg.norm(feat1 - feat2)
    
    return distances


def iou_distance(bboxes1, bboxes2):
    """
    Calculate IoU distance matrix between two bbox sets.
    
    Args:
        bboxes1: List of bboxes [(x1,y1,x2,y2), ...]
        bboxes2: List of bboxes [(x1,y1,x2,y2), ...]
    
    Returns:
        distance_matrix: Array of shape (N, M)
    """
    distances = np.zeros((len(bboxes1), len(bboxes2)))
    
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            iou = calculate_iou(bbox1, bbox2)
            distances[i, j] = 1.0 - iou  # Convert IoU to distance
    
    return distances


def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def cascade_matching(tracklets, detections, detection_features, 
                     max_distance=0.7, max_cascade_age=30):
    """
    Perform cascade matching between tracklets and detections.
    
    Args:
        tracklets: List of Tracklet objects
        detections: List of detection bboxes
        detection_features: Array of detection features (N, 512)
        max_distance: Maximum feature distance for valid match
        max_cascade_age: Maximum age for cascade levels
    
    Returns:
        matches: List of (tracklet_idx, detection_idx) pairs
        unmatched_tracklets: List of unmatched tracklet indices
        unmatched_detections: List of unmatched detection indices
    """
    if len(tracklets) == 0 or len(detections) == 0:
        return [], list(range(len(tracklets))), list(range(len(detections)))
    
    matches = []
    unmatched_detections = list(range(len(detections)))
    
    # Cascade through different time-since-update levels
    for cascade_level in range(max_cascade_age):
        if len(unmatched_detections) == 0:
            break
        
        # Get tracklets at this cascade level
        tracklet_indices = [
            i for i, t in enumerate(tracklets)
            if t.time_since_update == cascade_level and t.state == 'tracked'
        ]
        
        if len(tracklet_indices) == 0:
            continue
        
        # Build cost matrix
        tracklet_features = np.array([
            tracklets[i].get_current_feature().cpu().numpy()
            for i in tracklet_indices
        ])
        
        detection_features_subset = detection_features[unmatched_detections]
        
        cost_matrix = cosine_distance(tracklet_features, detection_features_subset)
        
        # Apply gating (reject matches with distance > threshold)
        cost_matrix[cost_matrix > max_distance] = max_distance + 1e5
        
        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid matches
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= max_distance:
                tracklet_idx = tracklet_indices[row]
                detection_idx = unmatched_detections[col]
                matches.append((tracklet_idx, detection_idx))
        
        # Update unmatched detections
        matched_detection_indices = [col for row, col in zip(row_indices, col_indices)
                                     if cost_matrix[row, col] <= max_distance]
        unmatched_detections = [
            d for i, d in enumerate(unmatched_detections)
            if i not in matched_detection_indices
        ]
    
    # Find unmatched tracklets
    matched_tracklet_indices = [m[0] for m in matches]
    unmatched_tracklets = [
        i for i in range(len(tracklets))
        if i not in matched_tracklet_indices and tracklets[i].state == 'tracked'
    ]
    
    return matches, unmatched_tracklets, unmatched_detections
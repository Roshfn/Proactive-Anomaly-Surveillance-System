"""
Tracklet: Data structure for tracking individual persons across frames.

Stores:
- Bounding box history
- Feature history
- Trajectory for ATM calculation
- Tracking state
"""

import numpy as np
from collections import deque


class Tracklet:
    """
    Represents a tracked person with history of detections.
    """
    
    _id_counter = 0
    
    def __init__(self, bbox, feature, frame_id):
        """
        Initialize a new tracklet.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            feature: Initial 512-dim feature vector
            frame_id: Frame number where tracklet was created
        """
        self.id = Tracklet._id_counter
        Tracklet._id_counter += 1
        
        self.bboxes = deque(maxlen=30)
        self.features = deque(maxlen=30)
        self.frame_ids = deque(maxlen=30)
        
        self.bboxes.append(bbox)
        self.features.append(feature)
        self.frame_ids.append(frame_id)
        
        self.state = 'tracked'
        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        
        self.is_stationary = False
        self.stationary_frames = 0
    
    def update(self, bbox, feature, frame_id):
        """
        Update tracklet with new detection.
        
        Args:
            bbox: New bounding box
            feature: New feature vector
            frame_id: Current frame number
        """
        self.bboxes.append(bbox)
        self.features.append(feature)
        self.frame_ids.append(frame_id)
        
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
        self.state = 'tracked'
    
    def mark_missed(self):
        """Mark tracklet as not detected in current frame."""
        self.time_since_update += 1
        self.age += 1
        
        if self.time_since_update > 3:
            self.state = 'lost'
    
    def get_current_bbox(self):
        """Get most recent bounding box."""
        if len(self.bboxes) == 0:
            return None
        return self.bboxes[-1]
    
    def get_current_feature(self):
        """Get most recent feature vector."""
        if len(self.features) == 0:
            return None
        return self.features[-1]
    
    def get_trajectory(self, max_length=30):
        """
        Get trajectory as list of center points.
        
        Args:
            max_length: Maximum number of points to return
        
        Returns:
            List of (x, y) center points
        """
        trajectory = []
        for bbox in list(self.bboxes)[-max_length:]:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            trajectory.append((cx, cy))
        return trajectory
    
    def get_top_left_trajectory(self, max_length=30):
        """
        Get trajectory as list of top-left points (for ATM calculation).
        
        Returns:
            List of (x, y) top-left points
        """
        trajectory = []
        for bbox in list(self.bboxes)[-max_length:]:
            x1, y1, x2, y2 = bbox
            trajectory.append((x1, y1))
        return trajectory
    
    def check_stationary(self, iou_threshold=0.9, min_frames=5):
        """
        Check if tracklet represents a stationary person.
        
        Uses IoU between consecutive bounding boxes.
        
        Args:
            iou_threshold: Minimum IoU to consider stationary
            min_frames: Minimum frames to confirm stationary state
        
        Returns:
            True if person is stationary
        """
        if len(self.bboxes) < min_frames:
            return False
        
        recent_bboxes = list(self.bboxes)[-min_frames:]
        ious = []
        
        for i in range(len(recent_bboxes) - 1):
            iou = self._calculate_iou(recent_bboxes[i], recent_bboxes[i+1])
            ious.append(iou)
        
        avg_iou = np.mean(ious)
        
        if avg_iou > iou_threshold:
            self.stationary_frames += 1
            if self.stationary_frames >= min_frames:
                self.is_stationary = True
                return True
        else:
            self.stationary_frames = 0
            self.is_stationary = False
        
        return False
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bboxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def __repr__(self):
        return f"Tracklet(id={self.id}, state={self.state}, age={self.age}, hits={self.hits})"
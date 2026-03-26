"""
Human Tracker: Combines detection, feature extraction, and matching.

Implements the complete Stage 1 pipeline:
1. Receive detections from YOLOv5
2. Extract identity features (OSNet)
3. Extract RoI features (YOLOv5 mid-level)
4. Couple features (512-dim)
5. Cascade matching for ID assignment
6. Update tracklets
"""

import torch
import numpy as np
from src.stage1_human_tracking.tracklet import Tracklet
from src.algorithms.cascade_matching import cascade_matching


class HumanTracker:
    """
    Manages tracking of multiple people across frames.
    """
    
    def __init__(self, feature_extractor, max_age=30, min_hits=3):
        """
        Args:
            feature_extractor: FeatureExtractor instance
            max_age: Maximum frames to keep tracklet without update
            min_hits: Minimum hits before tracklet is confirmed
        """
        self.feature_extractor = feature_extractor
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracklets = []
        self.frame_count = 0
    
    def update(self, frame, detections, mid_features=None):
        """
        Update tracker with new frame detections.
        
        Args:
            frame: Current frame image (H, W, 3)
            detections: List of bounding boxes [(x1,y1,x2,y2), ...]
            mid_features: Mid-level features from YOLOv5 (optional)
        
        Returns:
            active_tracklets: List of tracklets in tracked state
        """
        self.frame_count += 1
        
        # Extract features for detections
        if len(detections) > 0:
            identity_features = self.feature_extractor.extract_identity_features(
                frame, detections
            )
            
            if mid_features is not None:
                roi_features = self.feature_extractor.extract_roi_features(
                    mid_features, detections
                )
            else:
                # If no mid_features, use zero features
                roi_features = torch.zeros(len(detections), 256, 
                                          device=identity_features.device)
            
            # Couple features
            coupled_features = self.feature_extractor.couple_features(
                identity_features, roi_features
            )
        else:
            coupled_features = torch.empty(0, 512)
        
        # Perform cascade matching
        matches, unmatched_tracklets, unmatched_detections = cascade_matching(
            self.tracklets, detections, coupled_features.cpu().numpy()
        )
        
        # Update matched tracklets
        for tracklet_idx, detection_idx in matches:
            self.tracklets[tracklet_idx].update(
                detections[detection_idx],
                coupled_features[detection_idx],
                self.frame_count
            )
        
        # Mark unmatched tracklets as missed
        for tracklet_idx in unmatched_tracklets:
            self.tracklets[tracklet_idx].mark_missed()
        
        # Create new tracklets for unmatched detections
        for detection_idx in unmatched_detections:
            new_tracklet = Tracklet(
                detections[detection_idx],
                coupled_features[detection_idx],
                self.frame_count
            )
            self.tracklets.append(new_tracklet)
        
        # Remove dead tracklets
        self.tracklets = [
            t for t in self.tracklets
            if t.time_since_update <= self.max_age
        ]
        
        # Check stationary state for each tracklet
        for tracklet in self.tracklets:
            tracklet.check_stationary()
        
        # Return active tracklets (confirmed by min_hits)
        active_tracklets = [
            t for t in self.tracklets
            if t.hits >= self.min_hits and t.state == 'tracked'
        ]
        
        return active_tracklets
    
    def get_all_tracklets(self):
        """Get all tracklets regardless of state."""
        return self.tracklets
    
    def reset(self):
        """Reset tracker state."""
        self.tracklets = []
        self.frame_count = 0
        Tracklet._id_counter = 0


class YOLOv5Detector:
    """
    Wrapper for YOLOv5 detection with mid-level feature extraction.
    """
    
    def __init__(self, model_path, device='cpu', conf_threshold=0.5):
        """
        Args:
            model_path: Path to YOLOv5 weights
            device: Device for inference
            conf_threshold: Confidence threshold for detections
        """
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                    path=model_path)
        self.model.conf = conf_threshold
        self.model.to(device)
        self.model.eval()
    
    def detect(self, frame):
        """
        Detect people in frame.
        
        Args:
            frame: Input image (H, W, 3) in BGR format
        
        Returns:
            bboxes: List of bounding boxes for detected people
            mid_features: Mid-level feature map (placeholder for now)
        """
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        
        # Filter for person class
        people = detections[detections['name'] == 'person']
        
        bboxes = []
        for _, person in people.iterrows():
            x1 = int(person['xmin'])
            y1 = int(person['ymin'])
            x2 = int(person['xmax'])
            y2 = int(person['ymax'])
            bboxes.append([x1, y1, x2, y2])
        
        # TODO: Extract actual mid-level features from YOLOv5
        # For now, return None (tracker will use zero RoI features)
        mid_features = None
        
        return bboxes, mid_features
"""
Luggage Tracker: Top-down approach for detecting abandoned objects.

Paper approach:
1. Global detection (YOLOv5 on full frame)
2. Local detection (YOLOv5 on human patches)
3. Merge detections (NMS)
4. Assign ownership (bipartite matching with dynamic threshold)
5. Detect abandonment (distance + time)
"""

import numpy as np
import cv2
import torch


class LuggageTracker:
    """
    Tracks luggage and detects abandonment.
    """
    
    def __init__(self, detector, luggage_classes=['handbag', 'backpack', 'suitcase'],
                 abandonment_duration=10.0):
        """
        Args:
            detector: YOLOv5Detector instance
            luggage_classes: List of object classes to track
            abandonment_duration: Time in seconds before considering abandoned
        """
        self.detector = detector
        self.luggage_classes = luggage_classes
        self.abandonment_duration = abandonment_duration
        
        self.luggage_ownership = {}
        self.luggage_timers = {}
        self.luggage_history = {}
    
    def detect_luggage_global(self, frame):
        """
        Global luggage detection on full frame.
        
        Args:
            frame: Full frame image
        
        Returns:
            luggage_detections: List of [x1, y1, x2, y2, class_name]
        """
        results = self.detector.model(frame)
        detections = results.pandas().xyxy[0]
        
        luggage = []
        for _, det in detections.iterrows():
            if det['name'] in self.luggage_classes:
                x1 = int(det['xmin'])
                y1 = int(det['ymin'])
                x2 = int(det['xmax'])
                y2 = int(det['ymax'])
                luggage.append([x1, y1, x2, y2, det['name']])
        
        return luggage
    
    def detect_luggage_local(self, frame, tracklets):
        """
        Local luggage detection on human-centered patches.
        
        Args:
            frame: Full frame image
            tracklets: List of Tracklet objects
        
        Returns:
            luggage_detections: List of [x1, y1, x2, y2, class_name]
        """
        local_detections = []
        
        for tracklet in tracklets:
            bbox = tracklet.get_current_bbox()
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Expand bbox for local search (1.5x height)
            height = y2 - y1
            width = x2 - x1
            
            expand_h = int(height * 0.5)
            expand_w = int(width * 0.5)
            
            px1 = max(0, x1 - expand_w)
            py1 = max(0, y1 - expand_h)
            px2 = min(frame.shape[1], x2 + expand_w)
            py2 = min(frame.shape[0], y2 + expand_h)
            
            # Extract patch
            patch = frame[py1:py2, px1:px2]
            
            if patch.size == 0:
                continue
            
            # Detect in patch
            results = self.detector.model(patch)
            detections = results.pandas().xyxy[0]
            
            # Transform coordinates back to full frame
            for _, det in detections.iterrows():
                if det['name'] in self.luggage_classes:
                    lx1 = int(det['xmin']) + px1
                    ly1 = int(det['ymin']) + py1
                    lx2 = int(det['xmax']) + px1
                    ly2 = int(det['ymax']) + py1
                    local_detections.append([lx1, ly1, lx2, ly2, det['name']])
        
        return local_detections
    
    def merge_detections(self, global_detections, local_detections, iou_threshold=0.5):
        """
        Merge global and local detections using NMS.
        
        Args:
            global_detections: List from global detection
            local_detections: List from local detection
            iou_threshold: IoU threshold for NMS
        
        Returns:
            merged_detections: Deduplicated list
        """
        all_detections = global_detections + local_detections
        
        if len(all_detections) == 0:
            return []
        
        # Simple NMS
        merged = []
        used = [False] * len(all_detections)
        
        for i, det1 in enumerate(all_detections):
            if used[i]:
                continue
            
            used[i] = True
            bbox1 = det1[:4]
            
            # Check against remaining detections
            for j in range(i + 1, len(all_detections)):
                if used[j]:
                    continue
                
                bbox2 = all_detections[j][:4]
                iou = self._calculate_iou(bbox1, bbox2)
                
                if iou > iou_threshold:
                    used[j] = True
            
            merged.append(det1)
        
        return merged
    
    def assign_ownership(self, luggage_detections, tracklets):
        """
        Assign luggage to nearest person using dynamic threshold.
        
        Dynamic threshold: 2 × person bbox width
        
        Args:
            luggage_detections: List of luggage bboxes
            tracklets: List of Tracklet objects
        
        Returns:
            ownership: Dict {luggage_idx: tracklet_id or None}
        """
        ownership = {}
        
        for lug_idx, luggage in enumerate(luggage_detections):
            lx1, ly1, lx2, ly2 = luggage[:4]
            lug_center = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)
            
            min_distance = float('inf')
            owner_id = None
            
            for tracklet in tracklets:
                bbox = tracklet.get_current_bbox()
                if bbox is None:
                    continue
                
                px1, py1, px2, py2 = bbox
                person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                person_width = px2 - px1
                
                # Dynamic threshold: 2 × person width
                threshold = 2 * person_width
                
                # Calculate distance
                distance = np.sqrt(
                    (lug_center[0] - person_center[0])**2 +
                    (lug_center[1] - person_center[1])**2
                )
                
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    owner_id = tracklet.id
            
            ownership[lug_idx] = owner_id
        
        return ownership
    
    def check_abandonment(self, luggage_detections, ownership, tracklets, frame_time):
        """
        Check for abandoned luggage.
        
        Args:
            luggage_detections: List of luggage bboxes
            ownership: Dict {luggage_idx: tracklet_id}
            tracklets: List of active tracklets
            frame_time: Current time in seconds
        
        Returns:
            abandonment_events: List of event dictionaries
        """
        abandonment_events = []
        active_tracklet_ids = {t.id for t in tracklets}
        
        for lug_idx, owner_id in ownership.items():
            luggage_bbox = luggage_detections[lug_idx]
            
            # Luggage has no owner or owner left scene
            if owner_id is None or owner_id not in active_tracklet_ids:
                
                if lug_idx not in self.luggage_timers:
                    self.luggage_timers[lug_idx] = frame_time
                
                elapsed = frame_time - self.luggage_timers[lug_idx]
                
                if elapsed >= self.abandonment_duration:
                    abandonment_events.append({
                        'type': 'abandonment',
                        'luggage_bbox': luggage_bbox[:4],
                        'luggage_class': luggage_bbox[4],
                        'previous_owner': owner_id,
                        'duration': elapsed,
                        'timestamp': frame_time
                    })
            else:
                # Luggage still with owner, reset timer
                if lug_idx in self.luggage_timers:
                    del self.luggage_timers[lug_idx]
        
        return abandonment_events
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes."""
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
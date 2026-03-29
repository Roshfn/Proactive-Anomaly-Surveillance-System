"""
PASS-CCTV: Integrated anomaly surveillance system.

Combines all 4 anomaly detectors:
1. Intrusion Detection
2. Loitering Detection
3. Abandonment Detection
4. Arson Detection
"""

import cv2
import time
from pathlib import Path
from src.stage1_human_tracking.osnet_model import build_osnet
from src.stage1_human_tracking.feature_extractor import FeatureExtractor
from src.stage1_human_tracking.human_tracker import HumanTracker, YOLOv5Detector
from src.stage2_anomaly_recognition.intersection_detector import IntersectionDetector
from src.stage2_anomaly_recognition.luggage_tracker import LuggageTracker
from src.stage2_anomaly_recognition.scene_analyzer import SceneAnalyzer


class PASSCCTVSystem:
    """
    Complete PASS-CCTV surveillance system.
    """
    
    def __init__(self, device='cpu', enable_intrusion=True, enable_loitering=True,
                 enable_abandonment=True, enable_arson=True):
        """
        Initialize PASS-CCTV system.
        
        Args:
            device: Device for inference
            enable_intrusion: Enable intrusion detection
            enable_loitering: Enable loitering detection
            enable_abandonment: Enable abandonment detection
            enable_arson: Enable arson detection
        """
        self.device = device
        
        print("PASS-CCTV SYSTEM")
        print("Initializing Components...")

        
        # Stage 1: Human Detection & Tracking
        print("\n[Stage 1] Loading detection and tracking models...")
        yolo_path = Path("models/yolov5s.pt")
        self.detector = YOLOv5Detector(str(yolo_path), device=device)
        print(" YOLOv5 detector loaded")
        
        osnet_path = Path("models/osnet_x1_0_imagenet.pth")
        osnet = build_osnet(str(osnet_path), device=device)
        feature_extractor = FeatureExtractor(osnet, device=device)
        self.tracker = HumanTracker(feature_extractor, max_age=30, min_hits=3)
        print(" OSNet feature extractor loaded")
        print(" Human tracker initialized")
        
        # Stage 2: Anomaly Recognition
        print("\n[Stage 2] Loading anomaly detectors...")
        
        self.enable_intrusion = enable_intrusion
        self.enable_loitering = enable_loitering
        self.enable_abandonment = enable_abandonment
        self.enable_arson = enable_arson
        
        if enable_intrusion or enable_loitering:
            self.intersection_detector = IntersectionDetector(
                intrusion_threshold=0.3,
                loitering_threshold=0.3,
                loitering_duration=10.0
            )
            print(" Intersection detector loaded (intrusion + loitering)")
        
        if enable_abandonment:
            self.luggage_tracker = LuggageTracker(
                self.detector,
                abandonment_duration=10.0
            )
            print(" Luggage tracker loaded (abandonment)")
        
        if enable_arson:
            self.scene_analyzer = SceneAnalyzer(device=device)
            print(" Scene analyzer loaded (arson)")
        
        print("PASS-CCTV System Ready")
        
        self.frame_count = 0
        self.start_time = None
    
    def add_intrusion_zone(self, zone_id, vertices):
        """Add intrusion detection zone."""
        if self.enable_intrusion:
            self.intersection_detector.add_intrusion_zone(zone_id, vertices)
    
    def add_loitering_zone(self, zone_id, vertices):
        """Add loitering detection zone."""
        if self.enable_loitering:
            self.intersection_detector.add_loitering_zone(zone_id, vertices)
    
    def process_frame(self, frame):
        """
        Process single frame through entire pipeline.
        
        Args:
            frame: Input frame (H, W, 3) BGR format
        
        Returns:
            results: Dictionary with all detection results
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        self.frame_count += 1
        frame_time = time.time() - self.start_time
        
        results = {
            'frame_id': self.frame_count,
            'timestamp': frame_time,
            'detections': [],
            'tracklets': [],
            'intrusion_events': [],
            'loitering_events': [],
            'abandonment_events': [],
            'arson_events': []
        }
        
        # Stage 1: Detection and Tracking
        detections, mid_features = self.detector.detect(frame)
        active_tracklets = self.tracker.update(frame, detections, mid_features)
        
        results['detections'] = detections
        results['tracklets'] = active_tracklets
        
        # Stage 2: Anomaly Recognition
        
        # Intrusion Detection
        if self.enable_intrusion and hasattr(self, 'intersection_detector'):
            intrusion_events = self.intersection_detector.check_intrusion(active_tracklets)
            results['intrusion_events'] = intrusion_events
        
        # Loitering Detection
        if self.enable_loitering and hasattr(self, 'intersection_detector'):
            loitering_events = self.intersection_detector.check_loitering(
                active_tracklets, frame_time
            )
            results['loitering_events'] = loitering_events
        
        # Abandonment Detection
        if self.enable_abandonment and hasattr(self, 'luggage_tracker'):
            # Detect luggage
            global_luggage = self.luggage_tracker.detect_luggage_global(frame)
            local_luggage = self.luggage_tracker.detect_luggage_local(frame, active_tracklets)
            merged_luggage = self.luggage_tracker.merge_detections(global_luggage, local_luggage)
            
            # Assign ownership
            ownership = self.luggage_tracker.assign_ownership(merged_luggage, active_tracklets)
            
            # Check abandonment
            abandonment_events = self.luggage_tracker.check_abandonment(
                merged_luggage, ownership, active_tracklets, frame_time
            )
            results['abandonment_events'] = abandonment_events
            results['luggage'] = merged_luggage
        
        # Arson Detection (run every 30 frames to save compute)
        if self.enable_arson and hasattr(self, 'scene_analyzer'):
            if self.frame_count % 30 == 0:  # Check every second at 30 FPS
                arson_events = self.scene_analyzer.detect_arson(
                    frame, active_tracklets, fire_threshold=0.25
                )
                results['arson_events'] = arson_events
        
        return results
    
    def visualize_results(self, frame, results, show_zones=True):
        """
        Draw detection results on frame.
        
        Args:
            frame: Input frame
            results: Results dictionary from process_frame
            show_zones: Whether to draw detection zones
        
        Returns:
            output_frame: Annotated frame
        """
        import numpy as np
        output_frame = frame.copy()
        
        # Draw zones if enabled
        if show_zones:
            if self.enable_intrusion and hasattr(self, 'intersection_detector'):
                for zone_id, vertices in self.intersection_detector.intrusion_zones.items():
                    pts = np.array(vertices, dtype=np.int32)
                    cv2.polylines(output_frame, [pts], True, (0, 0, 255), 2)
                    cv2.putText(output_frame, f"INTRUSION: {zone_id}",
                               (int(vertices[0][0]), int(vertices[0][1])-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if self.enable_loitering and hasattr(self, 'intersection_detector'):
                for zone_id, vertices in self.intersection_detector.loitering_zones.items():
                    pts = np.array(vertices, dtype=np.int32)
                    cv2.polylines(output_frame, [pts], True, (255, 165, 0), 2)
                    cv2.putText(output_frame, f"LOITERING: {zone_id}",
                               (int(vertices[0][0]), int(vertices[0][1])-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        # Draw tracklets
        for tracklet in results['tracklets']:
            bbox = tracklet.get_current_bbox()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Check if involved in any anomaly
            color = (0, 255, 0)  # Green by default
            label = f"ID:{tracklet.id}"
            
            # Check intrusion
            for event in results['intrusion_events']:
                if event['tracklet_id'] == tracklet.id:
                    color = (0, 0, 255)  # Red
                    label += " INTRUSION!"
            
            # Check loitering
            for event in results['loitering_events']:
                if event['tracklet_id'] == tracklet.id:
                    color = (0, 165, 255)  # Orange
                    label += f" LOITER:{int(event['duration'])}s"
            
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw luggage if abandonment enabled
        if self.enable_abandonment and 'luggage' in results:
            for luggage in results['luggage']:
                x1, y1, x2, y2 = map(int, luggage[:4])
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(output_frame, luggage[4], (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Show arson alert
        if results['arson_events']:
            cv2.putText(output_frame, "FIRE DETECTED!",
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(output_frame, f"Score: {results['arson_events'][0]['fire_score']:.2f}",
                       (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add frame info
        info_text = f"Frame: {results['frame_id']} | People: {len(results['tracklets'])}"
        cv2.putText(output_frame, info_text, (10, output_frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_frame
    
    def reset(self):
        """Reset system state."""
        self.tracker.reset()
        self.frame_count = 0
        self.start_time = None
        
        if hasattr(self, 'intersection_detector'):
            self.intersection_detector.reset()
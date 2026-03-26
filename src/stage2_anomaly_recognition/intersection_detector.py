"""
Intersection Detector: Detects intrusion and loitering.

Intrusion: Person enters restricted zone (instant)
Loitering: Person remains in zone for extended duration (10 seconds)
"""

import time
from src.algorithms.polygon_clipping import calculate_intersection_ratio


class IntersectionDetector:
    """
    Detects intrusion and loitering based on zone intersection.
    """
    
    def __init__(self, intrusion_threshold=0.5, loitering_threshold=0.5, 
                 loitering_duration=10.0):
        """
        Args:
            intrusion_threshold: Minimum intersection ratio for intrusion
            loitering_threshold: Minimum intersection ratio for loitering
            loitering_duration: Duration in seconds to confirm loitering
        """
        self.intrusion_zones = {}
        self.loitering_zones = {}
        
        self.intrusion_threshold = intrusion_threshold
        self.loitering_threshold = loitering_threshold
        self.loitering_duration = loitering_duration
        
        self.loitering_timers = {}
    
    def add_intrusion_zone(self, zone_id, vertices):
        """
        Add intrusion detection zone.
        
        Args:
            zone_id: Unique zone identifier
            vertices: List of (x, y) polygon vertices
        """
        self.intrusion_zones[zone_id] = vertices
    
    def add_loitering_zone(self, zone_id, vertices):
        """
        Add loitering detection zone.
        
        Args:
            zone_id: Unique zone identifier
            vertices: List of (x, y) polygon vertices
        """
        self.loitering_zones[zone_id] = vertices
    
    def check_intrusion(self, tracklets):
        """
        Check for intrusion events.
        
        Args:
            tracklets: List of active Tracklet objects
        
        Returns:
            intrusion_events: List of (tracklet_id, zone_id, ratio) tuples
        """
        intrusion_events = []
        
        for tracklet in tracklets:
            bbox = tracklet.get_current_bbox()
            if bbox is None:
                continue
            
            for zone_id, zone_vertices in self.intrusion_zones.items():
                ratio = calculate_intersection_ratio(bbox, zone_vertices)
                
                if ratio > self.intrusion_threshold:
                    intrusion_events.append({
                        'type': 'intrusion',
                        'tracklet_id': tracklet.id,
                        'zone_id': zone_id,
                        'intersection_ratio': ratio,
                        'bbox': bbox,
                        'timestamp': time.time()
                    })
        
        return intrusion_events
    
    def check_loitering(self, tracklets, frame_time=None):
        """
        Check for loitering events.
        
        Args:
            tracklets: List of active Tracklet objects
            frame_time: Current frame timestamp (seconds since start)
        
        Returns:
            loitering_events: List of loitering event dictionaries
        """
        if frame_time is None:
            frame_time = time.time()
        
        loitering_events = []
        current_tracklets = set()
        
        for tracklet in tracklets:
            bbox = tracklet.get_current_bbox()
            if bbox is None:
                continue
            
            current_tracklets.add(tracklet.id)
            
            for zone_id, zone_vertices in self.loitering_zones.items():
                ratio = calculate_intersection_ratio(bbox, zone_vertices)
                
                key = (tracklet.id, zone_id)
                
                if ratio > self.loitering_threshold:
                    if key not in self.loitering_timers:
                        self.loitering_timers[key] = {
                            'start_time': frame_time,
                            'zone_id': zone_id,
                            'tracklet_id': tracklet.id
                        }
                    else:
                        elapsed = frame_time - self.loitering_timers[key]['start_time']
                        
                        if elapsed >= self.loitering_duration:
                            loitering_events.append({
                                'type': 'loitering',
                                'tracklet_id': tracklet.id,
                                'zone_id': zone_id,
                                'duration': elapsed,
                                'bbox': bbox,
                                'timestamp': frame_time
                            })
                else:
                    if key in self.loitering_timers:
                        del self.loitering_timers[key]
        
        # Clean up timers for tracklets no longer present
        keys_to_remove = [
            key for key in self.loitering_timers
            if key[0] not in current_tracklets
        ]
        for key in keys_to_remove:
            del self.loitering_timers[key]
        
        return loitering_events
    
    def reset(self):
        """Reset all timers."""
        self.loitering_timers = {}
"""
Sutherland-Hodgman Polygon Clipping Algorithm.

Used to calculate intersection between bounding box and alert zone.
Paper reference: Equation 10
"""

import numpy as np


def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    """
    Clip subject polygon against clip polygon.
    
    Args:
        subject_polygon: List of (x, y) vertices
        clip_polygon: List of (x, y) vertices
    
    Returns:
        clipped_polygon: List of (x, y) vertices of intersection
    """
    def inside_edge(point, edge_start, edge_end):
        """Check if point is on the left side of edge (inside)."""
        return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) >= \
               (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    
    def line_intersection(p1, p2, p3, p4):
        """Calculate intersection point of two line segments."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    output_polygon = list(subject_polygon)
    
    for i in range(len(clip_polygon)):
        if len(output_polygon) == 0:
            break
        
        input_polygon = output_polygon
        output_polygon = []
        
        edge_start = clip_polygon[i]
        edge_end = clip_polygon[(i + 1) % len(clip_polygon)]
        
        for j in range(len(input_polygon)):
            current_vertex = input_polygon[j]
            previous_vertex = input_polygon[j - 1]
            
            current_inside = inside_edge(current_vertex, edge_start, edge_end)
            previous_inside = inside_edge(previous_vertex, edge_start, edge_end)
            
            if current_inside:
                if not previous_inside:
                    intersection = line_intersection(
                        previous_vertex, current_vertex,
                        edge_start, edge_end
                    )
                    if intersection:
                        output_polygon.append(intersection)
                output_polygon.append(current_vertex)
            elif previous_inside:
                intersection = line_intersection(
                    previous_vertex, current_vertex,
                    edge_start, edge_end
                )
                if intersection:
                    output_polygon.append(intersection)
    
    return output_polygon


def polygon_area(polygon):
    """
    Calculate area of polygon using shoelace formula.
    
    Args:
        polygon: List of (x, y) vertices
    
    Returns:
        area: Polygon area
    """
    if len(polygon) < 3:
        return 0.0
    
    area = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        area += x1 * y2 - x2 * y1
    
    return abs(area) / 2.0


def bbox_to_polygon(bbox):
    """
    Convert bounding box to polygon vertices.
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        polygon: [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
    """
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def calculate_intersection_ratio(bbox, zone_polygon):
    """
    Calculate intersection ratio between bbox and zone (Equation 10).
    
    Intersection(B_k_i, Z) = Area(B_k_i ∩ Z) / Area(B_k_i)
    
    Args:
        bbox: [x1, y1, x2, y2]
        zone_polygon: List of (x, y) zone vertices
    
    Returns:
        ratio: Intersection ratio [0, 1]
    """
    bbox_polygon = bbox_to_polygon(bbox)
    bbox_area = polygon_area(bbox_polygon)
    
    if bbox_area == 0:
        return 0.0
    
    intersection_polygon = sutherland_hodgman_clip(bbox_polygon, zone_polygon)
    intersection_area = polygon_area(intersection_polygon)
    
    ratio = intersection_area / bbox_area
    
    return min(1.0, max(0.0, ratio))
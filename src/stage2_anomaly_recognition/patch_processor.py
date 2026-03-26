"""
Patch Processor: Extracts image patches for CLIP analysis.

Extracts three types of patches:
1. Frame patch: Entire resized frame
2. Trajectory patches: Regions around each tracked person
3. Stop region patches: Enlarged regions around stationary persons (1.5x height)
"""

import cv2
import numpy as np
from PIL import Image


class PatchProcessor:
    """
    Processes frames to extract patches for scene analysis.
    """
    
    def __init__(self, input_size=(224, 224)):
        """
        Args:
            input_size: Target size for CLIP input (width, height)
        """
        self.input_size = input_size
    
    def extract_frame_patch(self, frame):
        """
        Extract and resize entire frame.
        
        Args:
            frame: Input frame (H, W, 3) BGR format
        
        Returns:
            patch: Resized frame as PIL Image
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize maintaining aspect ratio
        pil_image = self._resize_maintain_aspect(pil_image)
        
        return pil_image
    
    def extract_trajectory_patches(self, frame, tracklets):
        """
        Extract patches around each tracked person.
        
        Args:
            frame: Input frame (H, W, 3) BGR format
            tracklets: List of Tracklet objects
        
        Returns:
            patches: List of PIL Images
        """
        patches = []
        
        for tracklet in tracklets:
            bbox = tracklet.get_current_bbox()
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop region
            crop = frame[y1:y2, x1:x2]
            
            # Convert to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL
            pil_image = Image.fromarray(crop_rgb)
            
            # Resize
            pil_image = self._resize_maintain_aspect(pil_image)
            
            patches.append(pil_image)
        
        return patches
    
    def extract_stop_region_patches(self, frame, tracklets, enlarge_factor=1.5):
        """
        Extract enlarged patches around stationary persons.
        
        Args:
            frame: Input frame (H, W, 3) BGR format
            tracklets: List of Tracklet objects
            enlarge_factor: Factor to enlarge region (1.5 = 1.5x person height)
        
        Returns:
            patches: List of PIL Images
        """
        patches = []
        
        for tracklet in tracklets:
            # Only process stationary tracklets
            if not tracklet.is_stationary:
                continue
            
            bbox = tracklet.get_current_bbox()
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate person height
            height = y2 - y1
            width = x2 - x1
            
            # Enlarge region
            expand_h = int(height * (enlarge_factor - 1.0) / 2)
            expand_w = int(width * (enlarge_factor - 1.0) / 2)
            
            # Calculate enlarged bbox
            ex1 = max(0, x1 - expand_w)
            ey1 = max(0, y1 - expand_h)
            ex2 = min(frame.shape[1], x2 + expand_w)
            ey2 = min(frame.shape[0], y2 + expand_h)
            
            if ex2 <= ex1 or ey2 <= ey1:
                continue
            
            # Crop enlarged region
            crop = frame[ey1:ey2, ex1:ex2]
            
            # Convert to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL
            pil_image = Image.fromarray(crop_rgb)
            
            # Resize
            pil_image = self._resize_maintain_aspect(pil_image)
            
            patches.append(pil_image)
        
        return patches
    
    def _resize_maintain_aspect(self, pil_image):
        """
        Resize image maintaining aspect ratio with padding.
        
        Args:
            pil_image: PIL Image
        
        Returns:
            resized: PIL Image of size self.input_size
        """
        # Calculate aspect ratio
        width, height = pil_image.size
        target_width, target_height = self.input_size
        
        # Calculate scale to fit within target size
        scale = min(target_width / width, target_height / height)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        resized = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create new image with padding
        new_image = Image.new('RGB', self.input_size, (0, 0, 0))
        
        # Paste resized image in center
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_image.paste(resized, (paste_x, paste_y))
        
        return new_image
"""
Feature Extractor: Combines OSNet identity features with YOLOv5 RoI features.

This implements the Feature Coupling method from the paper (Equation 2-4):
- Extract identity features (hif) using OSNet
- Extract RoI features (hrf) from YOLOv5 mid-level representations
- Couple features: F = Concat(hif, hrf)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class FeatureExtractor:
    """
    Extracts and couples identity features and RoI features for robust tracking.
    """
    
    def __init__(self, osnet_model, device='cpu'):
        """
        Args:
            osnet_model: Pre-trained OSNet model
            device: Computation device ('cpu' or 'cuda')
        """
        self.osnet = osnet_model
        self.device = device
        self.input_size = (256, 128)  # Height x Width for OSNet
    
    def extract_identity_features(self, image, bboxes):
        """
        Extract identity features (hif) using OSNet.
        
        Args:
            image: Input image (H, W, 3) in BGR format
            bboxes: List of bounding boxes [(x1, y1, x2, y2), ...]
        
        Returns:
            features: Tensor of shape (N, 256) where N is number of bboxes
        """
        if len(bboxes) == 0:
            return torch.empty(0, 256, device=self.device)
        
        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop person region
            crop = image[y1:y2, x1:x2]
            
            # Resize to OSNet input size
            crop = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
            
            # Convert BGR to RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Normalize (ensure float32, not float64)
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            # To tensor (C, H, W) - explicitly use float32
            crop = torch.from_numpy(crop).permute(2, 0, 1).float()
            crops.append(crop)
        
        if len(crops) == 0:
            return torch.empty(0, 256, device=self.device)
        
        # Batch process
        batch = torch.stack(crops).to(self.device)
        
        with torch.no_grad():
            features = self.osnet(batch, return_feats=True)
        
        # L2 normalization
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def extract_roi_features(self, mid_features, bboxes, output_size=16):
        """
        Extract RoI features (hrf) from YOLOv5 mid-level representations.
        
        Args:
            mid_features: Mid-level feature map from YOLOv5 (C, H, W)
            bboxes: List of bounding boxes in original image coordinates
            output_size: Size of RoI pooled features
        
        Returns:
            features: Tensor of shape (N, 256) where N is number of bboxes
        """
        if len(bboxes) == 0 or mid_features is None:
            return torch.empty(0, 256, device=self.device)
        
        # Get feature map dimensions
        _, feat_h, feat_w = mid_features.shape
        
        roi_features = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # Map bbox coordinates to feature map scale
            # Assuming mid_features is from stride 8 layer
            fx1 = int(x1 / 8)
            fy1 = int(y1 / 8)
            fx2 = int(x2 / 8)
            fy2 = int(y2 / 8)
            
            # Ensure valid coordinates
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(feat_w, fx2), min(feat_h, fy2)
            
            if fx2 <= fx1 or fy2 <= fy1:
                # Invalid RoI, use zero features
                roi_feat = torch.zeros(256, device=self.device)
            else:
                # Extract RoI from feature map
                roi = mid_features[:, fy1:fy2, fx1:fx2]
                
                # Adaptive average pooling to fixed size
                roi = F.adaptive_avg_pool2d(roi.unsqueeze(0), (output_size, output_size))
                
                # Global average pooling
                roi_feat = F.adaptive_avg_pool2d(roi, (1, 1)).squeeze()
                
                # Ensure 256 dimensions
                if roi_feat.numel() != 256:
                    roi_feat = F.adaptive_avg_pool1d(roi_feat.unsqueeze(0), 256).squeeze()
            
            roi_features.append(roi_feat)
        
        if len(roi_features) == 0:
            return torch.empty(0, 256, device=self.device)
        
        features = torch.stack(roi_features)
        
        # L2 normalization
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def couple_features(self, identity_features, roi_features):
        """
        Couple identity and RoI features (Equation 4).
        
        F_k_i = Concat(hif_k_i, hrf_k_i)
        
        Args:
            identity_features: Tensor of shape (N, 256)
            roi_features: Tensor of shape (N, 256)
        
        Returns:
            coupled_features: Tensor of shape (N, 512)
        """
        if identity_features.size(0) == 0:
            return torch.empty(0, 512, device=self.device)
        
        # Concatenate along feature dimension
        coupled = torch.cat([identity_features, roi_features], dim=1)
        
        # L2 normalization of coupled features
        coupled = F.normalize(coupled, p=2, dim=1)
        
        return coupled
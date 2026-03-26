"""
Scene Analyzer: CLIP-based fire and arson detection.

Uses text prompts to detect:
1. Fire scenes (smoke, flames, bright light)
2. Arson behavior (person lighting fire)

Zero-shot detection - no training required.
"""

import torch
import clip
import numpy as np
from src.stage2_anomaly_recognition.patch_processor import PatchProcessor


class SceneAnalyzer:
    """
    Analyzes scenes for fire and arson using CLIP.
    """
    
    def __init__(self, device='cpu'):
        """
        Args:
            device: Device for inference
        """
        self.device = device
        self.patch_processor = PatchProcessor(input_size=(224, 224))
        
        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # UPDATED PROMPTS (based on empirical testing with real fire images)
        # Debug results showed these prompts work best:
        # "Large fire" (0.2874), "A burning building" (0.2793), "A building on fire" (0.2661)
        self.fire_scene_prompts = {
            'non_fire': [
                "There is no person",
                "There is a moving person",
                "There is a person who doesn't move",
                "A normal building",
                "A street scene",
                "People walking"
            ],
            'fire': [
                "Large fire",                    # Best match (0.287 score)
                "A burning building",            # Second best (0.279 score)
                "A building on fire",            # Third best (0.266 score)
                "Fire emergency",                # Fourth best (0.256 score)
                "Flames and smoke",              # Fifth best (0.253 score)
                "A fire",
                "There is smoke rising",         # Original prompt
                "There is flame and fire soaring", # Original prompt
                "Burning fire",
                "Orange flames",
                "Black smoke"
            ]
        }
        
        self.arson_prompts = {
            'non_fire': [
                "A photo of a standing person",
                "A photo of a walking person",
                "A photo of a bending person",
                "A photo of a squats person"
            ],
            'fire': [
                "A photo of an arsonist",
                "A photo of a man fighting a fire",
                "A photo of flames and fires soaring",
                "A photo of a campfire",
                "A photo of a burning cooking pan",
                "A photo of a man doing a barbecue"
            ]
        }
    
    def detect_arson(self, frame, tracklets, fire_threshold=0.25):
        """
        Detect fire and arson in frame.
        
        UPDATED: Threshold lowered from 0.5 to 0.25 based on empirical testing.
        Real fire images score around 0.28, so 0.25 is optimal.
        
        Args:
            frame: Input frame (H, W, 3) BGR format
            tracklets: List of Tracklet objects
            fire_threshold: Threshold for fire detection (default: 0.25)
        
        Returns:
            arson_events: List of detection events
        """
        # Extract patches
        patches = []
        patch_types = []
        
        # Frame patch
        frame_patch = self.patch_processor.extract_frame_patch(frame)
        patches.append(frame_patch)
        patch_types.append('frame')
        
        # Trajectory patches
        traj_patches = self.patch_processor.extract_trajectory_patches(frame, tracklets)
        patches.extend(traj_patches)
        patch_types.extend(['trajectory'] * len(traj_patches))
        
        # Stop region patches
        stop_patches = self.patch_processor.extract_stop_region_patches(frame, tracklets)
        patches.extend(stop_patches)
        patch_types.extend(['stop_region'] * len(stop_patches))
        
        if len(patches) == 0:
            return []
        
        # Preprocess images
        image_inputs = torch.stack([self.preprocess(p) for p in patches]).to(self.device)
        
        # Encode images
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Prepare text prompts
        fire_texts = self.fire_scene_prompts['fire']
        all_texts = fire_texts
        
        # Encode text
        text_inputs = clip.tokenize(all_texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (image_features @ text_features.T).cpu().numpy()
        
        # Get top-3 scores per patch and average
        fire_scores = []
        for sim_row in similarities:
            top_3_indices = np.argsort(sim_row)[-3:]
            top_3_scores = sim_row[top_3_indices]
            avg_score = np.mean(top_3_scores)
            fire_scores.append(avg_score)
        
        # Overall fire score (average of all patches)
        overall_fire_score = np.mean(fire_scores)
        
        # Generate events
        arson_events = []
        
        if overall_fire_score > fire_threshold:
            # Find which prompts matched best
            best_prompt_idx = np.argmax(similarities.mean(axis=0))
            matched_prompt = all_texts[best_prompt_idx]
            
            arson_events.append({
                'type': 'arson',
                'fire_score': float(overall_fire_score),
                'matched_prompt': matched_prompt,
                'num_patches_analyzed': len(patches),
                'timestamp': None
            })
        
        return arson_events
    
    def set_custom_prompts(self, fire_prompts=None, arson_prompts=None):
        """
        Set custom text prompts for detection.
        
        Args:
            fire_prompts: Dict with 'fire' and 'non_fire' lists
            arson_prompts: Dict with 'fire' and 'non_fire' lists
        """
        if fire_prompts is not None:
            self.fire_scene_prompts = fire_prompts
        
        if arson_prompts is not None:
            self.arson_prompts = arson_prompts
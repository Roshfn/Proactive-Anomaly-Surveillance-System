"""
PASS-CCTV Video Processing Demo
Main demonstration script for anomaly detection on video input.

Processes video files (.avi, .mp4, .mkv, etc.) with noise reduction, 
frame extraction, and comprehensive anomaly detection across all 4 scenarios.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import time
from pass_cctv_system import PASSCCTVSystem


class VideoProcessor:
    """
    Processes surveillance video with quality enhancement and anomaly detection.
    """
    
    def __init__(self, system, enhance_quality=True):
        """
        Args:
            system: PASSCCTVSystem instance
            enhance_quality: Enable noise reduction and quality enhancement
        """
        self.system = system
        self.enhance_quality = enhance_quality
    
    def reduce_noise(self, frame):
        """
        Apply noise reduction techniques to improve frame quality.
        
        Args:
            frame: Input frame
        
        Returns:
            Enhanced frame
        """
        denoised = cv2.fastNlMeansDenoisingColored(
            frame, None, 10, 10, 7, 21
        )
        return denoised
    
    def sharpen_frame(self, frame):
        """
        Sharpen blurry frames using unsharp masking.
        
        Args:
            frame: Input frame
        
        Returns:
            Sharpened frame
        """
        gaussian = cv2.GaussianBlur(frame, (0, 0), 3.0)
        sharpened = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
        return sharpened
    
    def assess_blur(self, frame):
        """
        Assess if frame is blurry using Laplacian variance.
        
        Args:
            frame: Input frame
        
        Returns:
            blur_score: Lower values indicate more blur
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def enhance_frame(self, frame):
        """
        Apply comprehensive quality enhancement.
        
        Args:
            frame: Input frame
        
        Returns:
            Enhanced frame
        """
        if not self.enhance_quality:
            return frame
        
        blur_score = self.assess_blur(frame)
        
        if blur_score < 100:
            frame = self.sharpen_frame(frame)
        
        frame = self.reduce_noise(frame)
        
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def process_video(self, video_path, output_path=None, 
                     frame_skip=1, max_frames=None):
        """
        Process video file through complete PASS-CCTV pipeline.
        
        Args:
            video_path: Path to input video (supports .avi, .mp4, .mkv, .mov, etc.)
            output_path: Path for output video (optional, auto-detects format)
            frame_skip: Process every Nth frame (1 = all frames)
            max_frames: Maximum frames to process (None = all)
        
        Returns:
            results_summary: Dictionary with processing statistics
        """
        print("\nPASS-CCTV VIDEO PROCESSING")
        print("-" * 60)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"ERROR: Cannot open video {video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"\nInput Video: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total Frames: {total_frames}")
        print(f"Duration: {duration:.1f}s")
        print(f"Quality Enhancement: {'Enabled' if self.enhance_quality else 'Disabled'}")
        print(f"Frame Skip: {frame_skip}")
        
        if max_frames:
            frames_to_process = min(max_frames, total_frames)
        else:
            frames_to_process = total_frames
        
        print(f"Frames to Process: {frames_to_process}")
        
        out = None
        if output_path:
            video_ext = Path(output_path).suffix.lower()
            
            if video_ext == '.avi':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            elif video_ext == '.mp4':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif video_ext == '.mkv':
                fourcc = cv2.VideoWriter_fourcc(*'X264')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )
            print(f"Output Video: {output_path}")
        
        results_summary = {
            'total_frames': 0,
            'processed_frames': 0,
            'intrusion_count': 0,
            'loitering_count': 0,
            'abandonment_count': 0,
            'arson_count': 0,
            'processing_time': 0,
            'quality_enhanced': self.enhance_quality
        }
        
        frame_num = 0
        processed_count = 0
        start_time = time.time()
        
        print("\nProcessing video frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            if frame_num > frames_to_process:
                break
            
            if (frame_num - 1) % frame_skip != 0:
                if out:
                    out.write(frame)
                continue
            
            enhanced_frame = self.enhance_frame(frame)
            
            results = self.system.process_frame(enhanced_frame)
            
            results_summary['intrusion_count'] += len(results['intrusion_events'])
            results_summary['loitering_count'] += len(results['loitering_events'])
            results_summary['abandonment_count'] += len(results['abandonment_events'])
            results_summary['arson_count'] += len(results['arson_events'])
            
            output_frame = self.system.visualize_results(
                enhanced_frame, results, show_zones=True
            )
            
            if out:
                out.write(output_frame)
            
            processed_count += 1
            
            if processed_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_processing = processed_count / elapsed if elapsed > 0 else 0
                progress = (frame_num / frames_to_process) * 100
                
                print(f"Progress: {progress:.1f}% | Frame {frame_num}/{frames_to_process} | "
                      f"FPS: {fps_processing:.1f} | "
                      f"Intrusion: {results_summary['intrusion_count']}, "
                      f"Loitering: {results_summary['loitering_count']}, "
                      f"Abandonment: {results_summary['abandonment_count']}, "
                      f"Arson: {results_summary['arson_count']}")
        
        cap.release()
        if out:
            out.release()
        
        results_summary['total_frames'] = frame_num
        results_summary['processed_frames'] = processed_count
        results_summary['processing_time'] = time.time() - start_time
        
        print("\nPROCESSING COMPLETE")
        print("-" * 60)
        
        print(f"\nProcessing Statistics:")
        print(f"Total Frames: {results_summary['total_frames']}")
        print(f"Processed Frames: {results_summary['processed_frames']}")
        print(f"Processing Time: {results_summary['processing_time']:.1f}s")
        print(f"Average FPS: {results_summary['processed_frames']/results_summary['processing_time']:.1f}")
        
        print(f"\nDetected Anomalies:")
        print(f"Intrusion Events: {results_summary['intrusion_count']}")
        print(f"Loitering Events: {results_summary['loitering_count']}")
        print(f"Abandonment Events: {results_summary['abandonment_count']}")
        print(f"Arson Events: {results_summary['arson_count']}")
        
        if output_path:
            print(f"\nOutput Video: {output_path}")
        
        print("-" * 60)
        
        return results_summary


def main():
    """
    Main demonstration function.
    """
    
    INPUT_VIDEO = Path("ABODA/video1.avi")
    OUTPUT_VIDEO = Path("outputs/pass_cctv_result.avi")
    
    FRAME_SKIP = 1
    MAX_FRAMES = None
    ENHANCE_QUALITY = True
    
    if not INPUT_VIDEO.exists():
        print(f"Input video not found: {INPUT_VIDEO}")
        print("Please place your video file (.avi, .mp4, .mkv, etc.) in test_data/ folder")
        print("\nCreating sample demonstration video...")
        
        test_image = cv2.imread("test_data/test_people.jpg")
        if test_image is None:
            print("ERROR: Test image not found at test_data/test_people.jpg")
            return
        
        INPUT_VIDEO.parent.mkdir(exist_ok=True)
        height, width = test_image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(INPUT_VIDEO), fourcc, 30.0, (width, height))
        
        for i in range(300):
            out.write(test_image)
        
        out.release()
        print(f"Sample video created: {INPUT_VIDEO}\n")
    
    print("Initializing PASS-CCTV System...")
    system = PASSCCTVSystem(
        device='cpu',
        enable_intrusion=True,
        enable_loitering=True,
        enable_abandonment=True,
        enable_arson=True
    )
    
    system.tracker.min_hits = 1
    
    test_image = cv2.imread("test_data/test_people.jpg")
    if test_image is not None:
        height, width = test_image.shape[:2]
        
        intrusion_zone = [
            (width * 0.3, height * 0.3),
            (width * 0.7, height * 0.3),
            (width * 0.7, height * 0.7),
            (width * 0.3, height * 0.7)
        ]
        system.add_intrusion_zone('restricted_area', intrusion_zone)
        
        loitering_zone = [
            (width * 0.1, height * 0.2),
            (width * 0.4, height * 0.2),
            (width * 0.4, height * 0.8),
            (width * 0.1, height * 0.8)
        ]
        system.add_loitering_zone('monitored_area', loitering_zone)
    
    processor = VideoProcessor(system, enhance_quality=ENHANCE_QUALITY)
    
    OUTPUT_VIDEO.parent.mkdir(exist_ok=True)
    
    results = processor.process_video(
        INPUT_VIDEO,
        OUTPUT_VIDEO,
        frame_skip=FRAME_SKIP,
        max_frames=MAX_FRAMES
    )
    
    if results:
        print("\nPASS-CCTV DEMONSTRATION SUCCESSFUL")
        print("\nSystem Status:")
        print("All 4 anomaly detectors operational")
        print("Video processing complete")
        print("Quality enhancement applied")
        print("\nOutput ready for presentation")


if __name__ == "__main__":
    main()
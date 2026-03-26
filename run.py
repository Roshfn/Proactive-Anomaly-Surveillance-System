"""
PASS-CCTV Flexible Video Processor
Supports .avi, .mp4, and other video formats with auto-detection.
"""

import cv2
import sys
from pathlib import Path
from demo_video_processing import VideoProcessor
from pass_cctv_system import PASSCCTVSystem


def process_any_video(input_path, output_path=None, enhance=True, skip=1, max_frames=None):
    """
    Process any video format through PASS-CCTV system.
    
    Args:
        input_path: Path to input video (.avi, .mp4, etc.)
        output_path: Output path (auto-generates if None)
        enhance: Enable quality enhancement
        skip: Process every Nth frame
        max_frames: Limit frames to process
    """
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: Video not found at {input_path}")
        return None
    
    if output_path is None:
        output_path = Path("outputs") / f"processed_{input_path.name}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(exist_ok=True)
    
    print("Initializing PASS-CCTV System...")
    system = PASSCCTVSystem(
        device='cpu',
        enable_intrusion=True,
        enable_loitering=True,
        enable_abandonment=True,
        enable_arson=True
    )
    
    system.tracker.min_hits = 1
    
    cap = cv2.VideoCapture(str(input_path))
    if cap.isOpened():
        ret, first_frame = cap.read()
        if ret:
            height, width = first_frame.shape[:2]
            
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
        cap.release()
    
    processor = VideoProcessor(system, enhance_quality=enhance)
    
    results = processor.process_video(
        input_path,
        output_path,
        frame_skip=skip,
        max_frames=max_frames
    )
    
    return results


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python run_video.py <input_video> [output_video] [options]")
        print("\nOptions:")
        print("  --no-enhance     Disable quality enhancement")
        print("  --skip N         Process every Nth frame (default: 1)")
        print("  --max-frames N   Process maximum N frames")
        print("\nExamples:")
        print("  python run_video.py test_data/video.avi")
        print("  python run_video.py test_data/video.avi outputs/result.avi")
        print("  python run_video.py test_data/video.avi --skip 2 --max-frames 300")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = None
    enhance = True
    skip = 1
    max_frames = None
    
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--no-enhance':
            enhance = False
        elif arg == '--skip' and i + 1 < len(sys.argv):
            skip = int(sys.argv[i + 1])
            i += 1
        elif arg == '--max-frames' and i + 1 < len(sys.argv):
            max_frames = int(sys.argv[i + 1])
            i += 1
        elif output_video is None and not arg.startswith('--'):
            output_video = arg
        
        i += 1
    
    process_any_video(input_video, output_video, enhance, skip, max_frames)
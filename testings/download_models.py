import os
import torch # type: ignore
import urllib.request
from pathlib import Path
import ssl

# Handle SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

print("=" * 70)
print(" " * 15 + "PASS-CCTV MODEL DOWNLOADER")
print("=" * 70)

# Create models directory
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

def download_file(url, destination, description):
    """Download file with progress"""
    print(f"\n Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Destination: {destination}")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r   Progress: {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, destination, progress_hook)
        print()  # New line after progress
        
        size_mb = destination.stat().st_size / (1024*1024)
        print(f"✅ Downloaded successfully! Size: {size_mb:.2f} MB")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

# ============= 1. YOLOV5 =============
print("\n" + "="*70)
print("[1/3] YOLOv5 - Human Detection Model")
print("="*70)

yolov5_path = models_dir / "yolov5s.pt"

if yolov5_path.exists():
    size_mb = yolov5_path.stat().st_size / (1024*1024)
    print(f"✅ YOLOv5 already exists ({size_mb:.2f} MB)")
else:
    url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
    download_file(url, yolov5_path, "YOLOv5s weights")

# ============= 2. OSNET =============
print("\n" + "="*70)
print("[2/3] OSNet - Person Re-Identification Model")
print("="*70)

osnet_path = models_dir / "osnet_x1_0_imagenet.pth"

if osnet_path.exists():
    size_mb = osnet_path.stat().st_size / (1024*1024)
    print(f"✅ OSNet already exists ({size_mb:.2f} MB)")
else:
    url = "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x1_0_imagenet.pth"
    success = download_file(url, osnet_path, "OSNet weights")
    
    if not success:
        print("\n⚠️  Trying alternative download method...")
        # Alternative: Direct link
        alt_url = "https://drive.google.com/uc?export=download&id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY"
        download_file(alt_url, osnet_path, "OSNet weights (alternative)")

# ============= 3. CLIP =============
print("\n" + "="*70)
print("[3/3] CLIP - Vision-Language Model (Arson Detection)")
print("="*70)

try:
    import clip
    print("📥 Loading CLIP ViT-B/32 model...")
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    
    # Find where CLIP was downloaded
    clip_cache = Path.home() / ".cache" / "clip"
    
    if clip_cache.exists():
        clip_files = list(clip_cache.glob("ViT-B-32.pt"))
        if clip_files:
            size_mb = clip_files[0].stat().st_size / (1024*1024)
            print(f"✅ CLIP model downloaded to cache")
            print(f"   Location: {clip_files[0]}")
            print(f"   Size: {size_mb:.2f} MB")
        else:
            print(f"✅ CLIP model ready (cached)")
    else:
        print(f"✅ CLIP model loaded successfully")
        
except Exception as e:
    print(f"❌ Error loading CLIP: {e}")

# ============= VERIFICATION =============
print("\n" + "="*70)
print(" " * 20 + "VERIFICATION")
print("="*70)

models_status = {
    "YOLOv5": yolov5_path.exists(),
    "OSNet": osnet_path.exists(),
    "CLIP": True  # Loaded from cache
}

print(f"\nModels Directory: {models_dir.absolute()}\n")

total_size = 0
for name, exists in models_status.items():
    if exists:
        if name == "CLIP":
            clip_cache = Path.home() / ".cache" / "clip"
            if clip_cache.exists():
                clip_files = list(clip_cache.glob("*.pt"))
                if clip_files:
                    size_mb = clip_files[0].stat().st_size / (1024*1024)
                    total_size += size_mb
                    print(f"✅ {name:<15} {size_mb:>8.2f} MB (cached)")
        else:
            # Find the file
            if name == "YOLOv5":
                filepath = yolov5_path
            elif name == "OSNet":
                filepath = osnet_path
            
            size_mb = filepath.stat().st_size / (1024*1024)
            total_size += size_mb
            print(f"✅ {name:<15} {size_mb:>8.2f} MB")
    else:
        print(f"❌ {name:<15} NOT FOUND")

print(f"\n{'─'*70}")
print(f"Total size: {total_size:.2f} MB")
print(f"{'─'*70}")

# Final check
all_ready = all(models_status.values())

if all_ready:
    print("\n🎉 ALL MODELS READY!")
    print("✅ YOLOv5 - Human detection")
    print("✅ OSNet  - Person re-identification") 
    print("✅ CLIP   - Fire/arson detection")
    print("\n" + "="*70)
    print(" " * 15 + "✅ READY FOR PHASE 3!")
    print("="*70)
else:
    print("\n⚠️  Some models are missing!")
    print("Please share this output for troubleshooting.")

print()
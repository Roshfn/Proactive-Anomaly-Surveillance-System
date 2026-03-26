import torch
import cv2
from pathlib import Path
import numpy as np

print("=" * 70)
print(" " * 15 + "YOLOV5 DETECTION TEST")
print("=" * 70)

# Load YOLOv5 model
print("\n[1/3] Loading YOLOv5 model...")
model_path = Path("models/yolov5s.pt")

if not model_path.exists():
    print(f"❌ Model not found at: {model_path}")
    exit(1)

try:
    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), force_reload=False)
    model.conf = 0.5  # Confidence threshold
    print(f"✅ YOLOv5 model loaded successfully")
    # Fixed: Get device differently
    device = next(model.model.parameters()).device
    print(f"   Device: {device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Load test image
print("\n[2/3] Loading test image...")
test_image_path = Path("test_data/test_people.jpg")

if not test_image_path.exists():
    print(f"❌ Test image not found at: {test_image_path}")
    print("   Please run download_test_image.py first")
    exit(1)

img = cv2.imread(str(test_image_path))
print(f"✅ Image loaded: {img.shape[1]}x{img.shape[0]} pixels")

# Run detection
print("\n[3/3] Running detection...")
try:
    results = model(img)
    
    # Get detections
    detections = results.pandas().xyxy[0]
    people = detections[detections['name'] == 'person']
    
    print(f"✅ Detection complete!")
    print(f"   Total objects detected: {len(detections)}")
    print(f"   People detected: {len(people)}")
    
    if len(people) > 0:
        print("\n📊 Detected People:")
        for idx, person in people.iterrows():
            conf = person['confidence']
            x1, y1, x2, y2 = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])
            print(f"   Person {idx+1}: Confidence={conf:.2f}, BBox=[{x1},{y1},{x2},{y2}]")
    
    # Draw bounding boxes
    img_with_boxes = img.copy()
    for idx, person in people.iterrows():
        x1, y1, x2, y2 = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])
        conf = person['confidence']
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Person {conf:.2f}"
        cv2.putText(img_with_boxes, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save output
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "yolov5_detection_test.jpg"
    
    cv2.imwrite(str(output_path), img_with_boxes)
    
    print(f"\n💾 Output saved to: {output_path}")
    
    # Display summary
    print("\n" + "=" * 70)
    print(" " * 20 + "TEST SUMMARY")
    print("=" * 70)
    print(f"✅ YOLOv5 model: Working")
    print(f"✅ People detected: {len(people)}")
    print(f"✅ Output image: {output_path}")
    print("\n" + "=" * 70)
    print(" " * 15 + "✅ PHASE 3 COMPLETE!")
    print("=" * 70)
    print("\nNext: Check the output image to verify detection quality")
    print(f"Location: {output_path.absolute()}")
    print("=" * 70)

except Exception as e:
    print(f"❌ Error during detection: {e}")
    import traceback
    traceback.print_exc()
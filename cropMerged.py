# finall.py
import os
import json
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
DOCS_DIR = r"C:\Users\Rou\Desktop\Checkbox_Detection\AH3"  # Full images folder
OUTPUT_IMAGES = r"C:\Users\Rou\Desktop\Checkbox_Detection\cropped_rows"  # Cropped strips
VISUAL_DIR = r"C:\Users\Rou\Desktop\Checkbox_Detection\visualized"       # Crop visualization
FINAL_OUTPUT = r"C:\Users\Rou\Desktop\Checkbox_Detection\final_detection" # Final merged detections
PROCESSED_FILE = r"C:\Users\Rou\Desktop\Checkbox_Detection\processed_docs.json"
MODEL_PATH = r"C:\Users\MAJED\Desktop\Checkbox_Detection\runs\detect\train12\weights\best.pt"

MARGIN_Y_TOP = 100
MARGIN_Y_BOTTOM = 100
ROW_THRESHOLD = 200
MAX_WIDTH = 2000
CONFIDENCE = 0.5
IOU_THRESHOLD = 0.35

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT, exist_ok=True)

Image.MAX_IMAGE_PIXELS = None

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def merge_boxes_iou(boxes, classes, iou_thresh=0.5):
    boxes = np.array(boxes)
    classes = np.array(classes)
    if len(boxes) == 0:
        return [], []
    keep = []
    idxs = np.argsort(boxes[:, 0])
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        ious = []
        for j in idxs[1:]:
            xx1 = max(boxes[i,0], boxes[j,0])
            yy1 = max(boxes[i,1], boxes[j,1])
            xx2 = min(boxes[i,2], boxes[j,2])
            yy2 = min(boxes[i,3], boxes[j,3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
            area_j = (boxes[j,2]-boxes[j,0])*(boxes[j,3]-boxes[j,1])
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0
            ious.append(iou)
        idxs = idxs[1:]
        idxs = idxs[np.array(ious) < iou_thresh]
    return boxes[keep].tolist(), classes[keep].tolist()

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load or initialize processed docs
if os.path.exists(PROCESSED_FILE):
    with open(PROCESSED_FILE, "r") as f:
        processed_docs = json.load(f)
else:
    processed_docs = {}

# ---------------------------
# MAIN LOOP
# ---------------------------
for img_file in os.listdir(DOCS_DIR):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(DOCS_DIR, img_file)
    image = Image.open(img_path)
    width_original, height_original = image.size

    # Resize if too large
    if width_original > MAX_WIDTH:
        scale = MAX_WIDTH / width_original
        new_height = int(height_original * scale)
        image = image.resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)
        width_original, height_original = image.size

    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)

    # Check if already cropped
    if img_file in processed_docs and processed_docs[img_file].get("cropped", False):
        print(f"[INFO] {img_file} already cropped. Skipping cropping...")
        # Optionally, you could load previous crops here if needed
        crops = []          # No new crops
        positions = [(0,0)] # Run detection on full image
    else:
        # Crop into horizontal strips
        crops = []
        positions = []
        y_start = 0
        crop_idx = 0
        while y_start < height_original:
            top = max(0, y_start - MARGIN_Y_TOP)
            bottom = min(height_original, y_start + ROW_THRESHOLD + MARGIN_Y_BOTTOM)
            crop = image.crop((0, top, width_original, bottom))
            crops.append(crop)
            positions.append((0, top))

            out_img_path = os.path.join(OUTPUT_IMAGES, f"{os.path.splitext(img_file)[0]}_row{crop_idx}.jpg")
            crop.save(out_img_path, quality=95)
            draw.rectangle([0, top, width_original, bottom], outline="red", width=3)
            print(f"[DEBUG] Crop {crop_idx}: y=({top},{bottom}) -> saved as {out_img_path}")

            y_start += ROW_THRESHOLD
            crop_idx += 1

        # Save visualization
        vis_img.save(os.path.join(VISUAL_DIR, img_file), quality=95)

        # Mark as cropped
        processed_docs[img_file] = {"cropped": True}
        with open(PROCESSED_FILE, "w") as f:
            json.dump(processed_docs, f, indent=4)

    # ---------------------------
    # DETECTION ON CROPS OR FULL IMAGE
    # ---------------------------
    all_boxes = []
    all_classes = []
    if crops:
        # Run on cropped strips
        for crop, (x_offset, y_offset) in zip(crops, positions):
            results = model.predict(source=crop, conf=CONFIDENCE, imgsz=850, verbose=False)
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            # Adjust coordinates to full image
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = box
                all_boxes.append([x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset])
                all_classes.append(cls)
    else:
        # Run on full image
        results = model.predict(source=image, conf=CONFIDENCE, imgsz=850, verbose=False)
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        all_boxes.extend(boxes)
        all_classes.extend(classes)

    # Merge overlapping boxes
    merged_boxes, merged_classes = merge_boxes_iou(all_boxes, all_classes, iou_thresh=IOU_THRESHOLD)

    # Draw final boxes on full image
    final_img = image.copy()
    draw_final = ImageDraw.Draw(final_img)
    for box, cls in zip(merged_boxes, merged_classes):
        x1, y1, x2, y2 = box
        color = "green" if cls == 0 else "red"
        draw_final.rectangle([x1, y1, x2, y2], outline=color, width=2)

    out_final_path = os.path.join(FINAL_OUTPUT, img_file)
    final_img.save(out_final_path, quality=95)
    print(f"[OK] {img_file}: {len(crops) if crops else 1} crops processed, {len(merged_boxes)} final boxes drawn.")

print("\nALL DONE!")

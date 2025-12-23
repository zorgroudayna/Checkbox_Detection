import os
from PIL import Image, ImageDraw

# -------------------------
# CONFIG
# -------------------------
DOCS_DIR = r"C:\Users\MAJED\Desktop\Checkbox_Detection\AH3"      # Original documents folder
LABELS_DIR = r"C:\Users\MAJED\Desktop\Checkbox_Detection\lab"   # YOLO labels folder
OUTPUT_IMAGES = "cropped/images"                       # Cropped patches output
OUTPUT_LABELS = "cropped/labels"                       # Cropped labels output
VISUAL_DIR = "visualized"                               # Images with rectangles for visualization

MARGIN_Y_TOP = 100    # Margin above checkbox row
MARGIN_Y_BOTTOM = 100 # Margin below checkbox row
ROW_THRESHOLD = 30    # Vertical distance to consider same row
MAX_WIDTH = 2000      # Max width to resize large images

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

Image.MAX_IMAGE_PIXELS = None  # allow very large images

# -------------------------
# Helper: read YOLO labels
# -------------------------
def read_yolo_label(label_path, img_width, img_height):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            cls, x_center, y_center, w, h = map(float, line.strip().split())
            x_center_px = x_center * img_width
            y_center_px = y_center * img_height
            w_px = w * img_width
            h_px = h * img_height
            boxes.append({
                "class": int(cls),
                "x_center": x_center_px,
                "y_center": y_center_px,
                "w": w_px,
                "h": h_px
            })
    return boxes

# -------------------------
# Main loop
# -------------------------
total_crops = 0  # counter for total cropped images

for img_file in os.listdir(DOCS_DIR):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(DOCS_DIR, img_file)
    label_path = os.path.join(LABELS_DIR, os.path.splitext(img_file)[0] + ".txt")

    if not os.path.exists(label_path):
        print(f"[WARNING] Label not found for {img_file}, skipping.")
        continue

    image = Image.open(img_path)
    width_original, height_original = image.size

    # Resize if too large
    if width_original > MAX_WIDTH:
        scale = MAX_WIDTH / width_original
        new_height = int(height_original * scale)
        image = image.resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)
        width_original, height_original = image.size

    boxes = read_yolo_label(label_path, width_original, height_original)

    # Keep all checkboxes (both checked and unchecked)
    checkbox_boxes = [b for b in boxes if b["class"] == 0 or b["class"] == 1]

    # Sort by y_center
    checkbox_boxes.sort(key=lambda b: b["y_center"])

    # Group checkboxes by vertical proximity (rows)
    lines = []
    current_line = []
    current_y = None
    for box in checkbox_boxes:
        if current_y is None:
            current_line.append(box)
            current_y = box["y_center"]
        elif abs(box["y_center"] - current_y) <= ROW_THRESHOLD:
            current_line.append(box)
            current_y = sum(b["y_center"] for b in current_line)/len(current_line)
        else:
            lines.append(current_line)
            current_line = [box]
            current_y = box["y_center"]
    if current_line:
        lines.append(current_line)

    # Visualization image
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)

    # Crop each line with fixed height and include all checkboxes inside the crop
    for line_idx, line_boxes in enumerate(lines):
        # Determine crop boundaries (full width)
        top = min(b["y_center"] - b["h"]/2 for b in line_boxes) - MARGIN_Y_TOP
        bottom = max(b["y_center"] + b["h"]/2 for b in line_boxes) + MARGIN_Y_BOTTOM
        top = max(0, top)
        bottom = min(height_original, bottom)
        left = 0
        right = width_original

        # Draw rectangle for visualization
        draw.rectangle([left, top, right, bottom], outline="red", width=3)

        # Crop image
        cropped = image.crop((left, top, right, bottom))
        crop_w, crop_h = right - left, bottom - top

        # Get all checkboxes inside this crop (both classes 0 and 1)
        boxes_in_crop = [b for b in boxes if (b["y_center"] >= top and b["y_center"] <= bottom) and (b["class"] == 0 or b["class"] == 1)]

        # Save YOLO labels relative to cropped image
        out_label_path = os.path.join(
            OUTPUT_LABELS, f"{os.path.splitext(img_file)[0]}_line{line_idx}.txt"
        )
        with open(out_label_path, "w") as f:
            for b in boxes_in_crop:
                x_center = (b["x_center"] - left) / crop_w
                y_center = (b["y_center"] - top) / crop_h
                w = b["w"] / crop_w
                h = b["h"] / crop_h
                f.write(f"{b['class']} {x_center} {y_center} {w} {h}\n")

        # Save cropped image
        out_img_path = os.path.join(
            OUTPUT_IMAGES, f"{os.path.splitext(img_file)[0]}_line{line_idx}.jpg"
        )
        cropped.save(out_img_path, quality=95)
        total_crops += 1

    # Save visualization
    vis_img.save(os.path.join(VISUAL_DIR, img_file), quality=95)
    print(f"[OK] {img_file}: {len(lines)} rows cropped, labels include all checkboxes inside each crop.")

print(f"\nDONE! Total cropped images: {total_crops}")

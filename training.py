from ultralytics import YOLO
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # prevent decompression bomb errors
if __name__ == "__main__":
    # Charger le modèle

    # Load model
    model = YOLO('yolo11s.pt')

    # Train with optimized parameters
    model.train(
        data="data_split/data.yaml",
        epochs=100,
        patience=20,
        imgsz=600,
        batch=16,
        device=0,
        workers=8,
        pretrained=True,
        amp=True,
        cache='disk',
        optimizer="AdamW",
        lr0=0.01,
        cos_lr=True,
        augment=True,
        mosaic=0.0,
        mixup=0.0,
    )


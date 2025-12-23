import os
import shutil
import random

data_folder = r"C:\Users\Rou\Desktop\Checkbox_Detection\cropped"          # Original data
split_folder = "data_splitcroped"   # New folder for split

# Get all images that have corresponding labels
all_images = [f for f in os.listdir(os.path.join(data_folder, "images"))
              if os.path.exists(os.path.join(data_folder, "labels", f.rsplit(".",1)[0]+".txt"))]

random.shuffle(all_images)

n = len(all_images)
train_split = int(0.8 * n)
val_split = int(0.9 * n)

splits = {
    "train": all_images[:train_split],
    "val": all_images[train_split:val_split],
    "test": all_images[val_split:]
}

for split, files in splits.items():
    os.makedirs(os.path.join(split_folder, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(split_folder, split, "labels"), exist_ok=True)

    for img in files:
        label = img.rsplit(".",1)[0] + ".txt"

        # Copy images and labels
        shutil.copy2(os.path.join(data_folder, "images", img),
                     os.path.join(split_folder, split, "images", img))
        shutil.copy2(os.path.join(data_folder, "labels", label),
                     os.path.join(split_folder, split, "labels", label))

print("✅ Data split completed! Saved in folder:", split_folder)


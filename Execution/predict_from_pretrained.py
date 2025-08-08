import os
import torch
from pathlib import Path
import shutil
from ultralytics import YOLO
from converted_backlash import converted_backlash

def move_labels_to_split_dir(results_dir, split_images_dir):
    labels_dir = Path(results_dir) / "labels"
    print(f"Moving labels: Checking source label folder {labels_dir}")
    if not labels_dir.exists():
        print(f"[Warning] {labels_dir} does not exist, no label files to move.")
        return

    target_labels_dir = Path(split_images_dir).parent / "labels"
    print(f"Target label folder: {target_labels_dir}")
    os.makedirs(target_labels_dir, exist_ok=True)

    label_files = list(labels_dir.glob('*.txt'))
    print(f"Found {len(label_files)} label files, starting to move...")
    for label_file in label_files:
        shutil.move(str(label_file), str(target_labels_dir / label_file.name))

def predict_on_splits(model_weights, split_dirs, device=None):
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO(converted_backlash(model_weights))

    result_summary = {}

    for split, img_dir in split_dirs.items():
        img_dir_conv = converted_backlash(img_dir)
        
        if not os.path.isdir(img_dir_conv):
            print(f"[Warning] {split} split folder does not exist: {img_dir_conv}, skipping automatically.")
            continue

        imgs = [f for f in os.listdir(img_dir_conv) if f.lower().endswith(('.jpg', '.png'))]
        if not imgs:
            print(f"[Warning] No images in {split} split, skipping.")
            continue

        print(f"\n[Process] Running prediction on {split} split, folder: {img_dir_conv}, total {len(imgs)} images.")

        results = model.predict(
            source=img_dir_conv,
            task='segment',
            save=True,
            save_txt=True,
            save_conf=True,
            imgsz=640,
            device=device,  
            project='runs/segment',
            name=f'predict_{split}',
            exist_ok=True
        )

        results_dir = f"runs/segment/predict_{split}"
        move_labels_to_split_dir(results_dir, img_dir_conv)
        result_summary[split] = len(results) if results else 0

    print("All labels processed and saved to the specified paths!")
    return result_summary

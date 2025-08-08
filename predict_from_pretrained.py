import os
import torch
from pathlib import Path
import shutil
from ultralytics import YOLO
from converted_backlash import converted_backlash

def move_labels_to_split_dir(results_dir, split_images_dir):
    labels_dir = Path(results_dir) / "labels"
    print(f"搬移標籤: 檢查標籤來源資料夾 {labels_dir}")
    if not labels_dir.exists():
        print(f"[警告] {labels_dir} 不存在，無標籤檔案可搬移。")
        return

    target_labels_dir = Path(split_images_dir).parent / "labels"
    print(f"目標標籤資料夾: {target_labels_dir}")
    os.makedirs(target_labels_dir, exist_ok=True)

    label_files = list(labels_dir.glob('*.txt'))
    print(f"找到 {len(label_files)} 個標籤檔案，開始搬移...")
    for label_file in label_files:
        shutil.move(str(label_file), str(target_labels_dir / label_file.name))

def predict_on_splits(model_weights, split_dirs, device=None):
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO(converted_backlash(model_weights))

    result_summary = {}

    for split, img_dir in split_dirs.items():
        img_dir_conv = converted_backlash(img_dir)
        
        if not os.path.isdir(img_dir_conv):
            print(f"[警告] {split}集資料夾不存在: {img_dir_conv}，自動跳過。")
            continue

        imgs = [f for f in os.listdir(img_dir_conv) if f.lower().endswith(('.jpg', '.png'))]
        if not imgs:
            print(f"[警告] {split}集沒有圖片，跳過。")
            continue

        print(f"\n[流程] 執行{split}集預測，資料夾: {img_dir_conv}, 共{len(imgs)}張圖。")

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

    print("labels都處理完成且存入指定路徑中！")
    return result_summary

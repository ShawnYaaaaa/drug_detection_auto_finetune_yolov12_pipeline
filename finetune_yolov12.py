from ultralytics import YOLO
import os
import yaml
import torch
import re
import numpy as np
import glob
import cv2
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, RandomRotate90, 
    ShiftScaleRotate, RandomBrightnessContrast, Blur, ColorJitter, 
    GaussNoise, RandomResizedCrop
)
from converted_backlash import converted_backlash

def finetune_yolov12(model_weights, data_yaml, epochs=50, batch_size=16, device=None):
    print("\nYOLO v12  finetuning...")
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model_weights, task="segment")
    print(f"device: {model.device}")

    results = model.train(
        data = data_yaml,
        task = "segment",
        epochs = epochs,
        batch = batch_size,
        imgsz = 640,
        scale = 0.5,
        device=device,
        workers=4, 
        cache=True,
        lr0=0.0005, 
        warmup_epochs=10,
        save=True,
        plots=True,
    )
    best_weights = "runs/segment/train/weights/best.pt"
    if not os.path.isfile(best_weights):
        print(f"[Error] Best Checkpoint not found: {best_weights}")
        best_weights = None  
    return best_weights

def train_dir_sort_key(x):
    m = re.search(r'train(\d*)', os.path.basename(x))
    return int(m.group(1)) if m and m.group(1).isdigit() else 0


def find_latest_train_weights(runs_segment_dir="runs/segment"):
    train_dirs = glob.glob(os.path.join(runs_segment_dir, "train*"))
    if not train_dirs:
        print("[Error] Train files were not found.")
        return None

    train_dirs.sort(key=lambda x: os.path.getmtime(x))
    latest_train_dir = train_dirs[-1]
    best_weights_path = os.path.join(latest_train_dir, "weights", "best.pt")

    if os.path.isfile(best_weights_path):
        print(f"[Info] Best point found：{best_weights_path}")
        return best_weights_path
    else:
        print(f"[Error] Checkpoint not found:{best_weights_path}")
        return None


def valid_the_finetuned(model_weights=None, data_yaml=None, device=None):
    
    if model_weights is None:
        model_weights = find_latest_train_weights()
        if model_weights is None:
            raise FileNotFoundError("Usable weight for validation were not found. Please set model_weights directly.")
    else:
        model_weights = converted_backlash(model_weights)

    print("\nYOLO v12 Validating...")
    model = YOLO(converted_backlash(model_weights))
    print(f"model_weights_path:{model_weights}")
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {model.device}")
    
    

    data_path = os.path.normpath(converted_backlash(data_yaml))
    print("Classes:", model.names)
    print("Num of Classes:", len(model.names))

    results = model.val(
        data=data_path,  
        task="segment",
        imgsz=640,
        batch=16,  
        device=device,  
        split="test",  
        save=True,  
        save_txt=True,  
        save_conf=True, 
        max_det = 300,
    )

    print("Test Set Metrics:")
    print(f"Precision (P): {float(np.mean(results.box.p)):.3f}")
    print(f"Recall (R): {float(np.mean(results.box.r)):.3f}") 
    print(f"mAP@50: {float(results.box.map50):.3f}") 
    print(f"mAP@50:95: {float(results.box.map):.3f}") 

    for i, name in enumerate(results.box.p):
        print(f"Class {name}:")
        print(f"  Precision: {float(results.box.p[i]):.3f}")
        print(f"  Recall: {float(results.box.r[i]):.3f}")
        print(f"  mAP@50:95: {float(results.box.maps[i]):.3f}")

def predict_the_finetuned(model_weights=None, test_data_path=None, device=None):
    if model_weights is None:
        model_weights = find_latest_train_weights()
        if model_weights is None:
            raise FileNotFoundError("Usable weight not found. Please set model_weights directly.")
    else:
        model_weights = converted_backlash(model_weights)

    ("\nYOLO v12 Predicting...")
    model = YOLO(converted_backlash(model_weights))
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {model.device}")
    test_package_dir = test_data_path

    results = model.predict(
        source = test_package_dir,
        task = 'segment',
        save = True,
        save_txt = True,
        save_conf = True,
        imgsz = 640,
        device = device,
    )

    
    


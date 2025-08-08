import os
import sys
import argparse
from ultralytics import YOLO
from predict_from_pretrained  import predict_on_splits
from converted_backlash import converted_backlash
from Preprocessing_labels import main_process_labels
from finetune_yolov12 import finetune_yolov12, valid_the_finetuned, predict_the_finetuned, find_latest_train_weights



def main():
    # if len(sys.argv) == 1:
    #     sys.argv += ['--do_train']

    parser = argparse.ArgumentParser(description="YOLOv12 微調與後續操作")
    parser.add_argument('--do_preprocess', action='store_true', help="是否執行前處理")
    parser.add_argument('--epochs', type=int, default=5, help="訓練Epoch數")
    parser.add_argument('--batch_size', type=int, default=16, help="訓練 batch size")
    parser.add_argument('--device', type=str, default=None, help="使用的GPU裝置號")
    parser.add_argument('--data_yaml', type=str, default=None, help="資料集 yaml 路徑（可省略）")
    parser.add_argument('--do_train', action='store_true', help="是否執行微調")
    parser.add_argument('--do_valid', action='store_true', help="是否執行驗證")
    parser.add_argument('--do_predict', action='store_true', help="是否執行預測")
    args = parser.parse_args()

    device_param = args.device
    if not device_param or device_param =="None":
        device_param = None

    
    train_yaml = converted_backlash(r"C:\SHAWN\Drug_detection\Processed_datas_v1_20\data.yaml")
    test_yaml = converted_backlash(r"C:\SHAWN\Drug_detection\Processed_datas_v1_20\data.yaml")

    
    if args.data_yaml:
        data_yaml_path = converted_backlash(args.data_yaml)
    elif args.do_train:
        data_yaml_path = train_yaml
    elif args.do_valid or args.do_predict:
        data_yaml_path = test_yaml
    elif args.do_preprocess:
        data_yaml_path = None
    else:
        print("未指定任何流程操作，也沒有給 data_yaml 路徑，請指定後重試。")
        return

    print(f"使用的資料集 YAML 路徑為：{data_yaml_path}")
    
    model_weights_path = converted_backlash(r"C:\SHAWN\Drug_detection\yolov12s-seg_finetuned_15class_drugs.pt")
    best_weights = None
    
    split_images_dirs = {
        'train': converted_backlash(r"...yourdataset\train\images"),
        'valid': converted_backlash(r"...yourdataset\valid\images"),
        'test' : converted_backlash(r"...yourdataset\test\images"),
    }

    
    split_labels_dirs = {
        split: converted_backlash(path.replace('images', 'labels'))
        for split, path in split_images_dirs.items()
    }

    
    for split, p in split_images_dirs.items():
        print(f"Split: {split}, Images Path: {p}, Exists: {os.path.isdir(p)}")
        print(f"Split: {split}, Labels Path: {split_labels_dirs[split]}, Exists: {os.path.isdir(split_labels_dirs[split])}")

    
    for split, p in split_images_dirs.items():
        print(f"Split: {split}, Images Path: {p}, Exists: {os.path.isdir(p)}")
        print(f"Split: {split}, Labels Path: {split_labels_dirs[split]}, Exists: {os.path.isdir(split_labels_dirs[split])}")


    
    if args.do_preprocess:
        summary = predict_on_splits(model_weights_path, split_images_dirs, device=device_param)
        
        print("\n[main.py]預測流程結束！各集處理摘要：")
        for split, count in summary.items():
            print(f"  - {split:<7}: {count} 張圖片已預測")
            
        for split, label_folder in split_labels_dirs.items():
            print(f"\n開始處理 {split} split 的 labels：{label_folder}")
            main_process_labels(
                folder_path=label_folder,
                class_num=15,        
                do_fix_seg=True,    
                do_clean=True,      
                do_check=True,      
                backup=False       
        )    
        return

    
    if args.do_train:
        best_weights = finetune_yolov12(
            model_weights=model_weights_path,
            data_yaml=data_yaml_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device_param
        )
    if best_weights is None:
        best_weights = find_latest_train_weights()
        if best_weights is None:
            best_weights = model_weights_path
    
    if args.do_valid:
        print("\n開始執行驗證階段")
        valid_the_finetuned(best_weights, data_yaml_path, device=device_param)

    if args.do_predict:
        print("\n開始執行預測階段")
        test_data_path = r"F:\Shawn\Drug_detection\Others_datas\Photo_myself\Processed_datas_v2+v3_180\test\images"
        predict_the_finetuned(best_weights, test_data_path, device=device_param)
    

if __name__ == "__main__":
    main()

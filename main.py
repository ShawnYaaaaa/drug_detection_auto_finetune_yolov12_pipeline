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

    parser = argparse.ArgumentParser(description="YOLOv12 fine-tune")
    parser.add_argument('--do_preprocess', action='store_true', help="Whether preprocess")
    parser.add_argument('--epochs', type=int, default=50, help="Epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="training batch size")
    parser.add_argument('--device', type=str, default=None, help="Device for training")
    parser.add_argument('--data_yaml', type=str, default=None, help="data yaml path")
    parser.add_argument('--do_train', action='store_true', help="fine-tuning on new dataset's train split")
    parser.add_argument('--do_valid', action='store_true', help="Validation for test split(default)")
    parser.add_argument('--do_predict', action='store_true', help="Prediction for test split(default)")
    args = parser.parse_args()

    device_param = args.device
    if not device_param or device_param =="None":
        device_param = None

    
    yaml_path = converted_backlash(r"C:\SHAWN\Drug_detection\Processed_datas_v1_20\data.yaml")
    

    
    if args.data_yaml:
        data_yaml_path = converted_backlash(args.data_yaml)
    elif args.do_train:
        data_yaml_path = yaml_path
    elif args.do_valid or args.do_predict:
        data_yaml_path = yaml_path
    elif args.do_preprocess:
        data_yaml_path = None
    else:
        print("None of action or data_yaml path were given, please retry!")
        return

    print(f"data_yaml_path：{data_yaml_path}")
    
    model_weights_path = converted_backlash(r"...\yolov12s-seg_finetuned_15class_drugs.pt")
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
        
        print("\n[main.py]Prediction Complete! Summary for each splits:")
        for split, count in summary.items():
            print(f"  - {split:<7}: {count} images predicted")
            
        for split, label_folder in split_labels_dirs.items():
            print(f"\nDealing {split}'s split labels:{label_folder}")
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
        print("\nValidating...")
        valid_the_finetuned(best_weights, data_yaml_path, device=device_param)

    if args.do_predict:
        print("\nPredicting...")
        test_data_path = r"...yourdataset\test\images"
        predict_the_finetuned(best_weights, test_data_path, device=device_param)
    

if __name__ == "__main__":
    main()

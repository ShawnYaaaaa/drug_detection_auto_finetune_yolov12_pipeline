# Drug_detection_base_on_yolov12

An automated pipeline for fast drug recognition using YOLOv12.

***

## Overview

This project implements a complete and automated pipeline for drug recognition based on the YOLOv12 detection and segmentation framework. The pipeline supports dataset splitting, label preprocessing, fine-tuning with pretrained weights, validation, and prediction, all organized into a user-friendly command-line interface for modular task execution.

***

## Project Structure and Key Components

- **Dataset Preparation**  
  Users should pre-split their dataset into `train`, `valid`, and `test` splits and update the YAML file accordingly by modifying class names and `nc` (number of classes).

- **Core Python Modules**  
  - `converted_backlash.py`: Utility for Windows path conversions for consistent usage.  
  - `predict_from_pretrained.py`: Functions to perform predictions on different dataset splits using pretrained models.  
  - `Preprocessing_labels.py`: Handles label preprocessing tasks such as fixing class indices, removing redundant tokens, and formatting corrections.  
  - `finetune_yolov12.py`: Facilitates YOLOv12 fine-tuning, validation, and prediction with logging and device management.

- **Main Entrypoint (`main.py`)**  
  Integrates all modules and encapsulates key functionalities into command-line flags:  
  - `--do_preprocess`  
  - `--do_train`  
  - `--do_valid`  
  - `--do_predict`  

  Allows users to run different pipeline stages through simple command-line commands.

***

## Usage

### 1. Preprocess labels

- Automatically generates `labels` folders for each split aligned with `images` folders.  
- Processes label files by adjusting class indices, cleaning redundant data, and verifying format correctness.  
- Prepares clean labels for training and evaluation.

***

### 2. Train (Fine-tune)

- Fine-tunes YOLOv12 from a pretrained weight specified in the project on the current dataset.  
- Produces training logs including loss, mAP, and other metrics.  
- Runs validation on the validation split concurrently with training, outputting detailed class-wise performance.  
- Saves the best model weights (`best.pt`) and logs in the `train` output folder.

***

### 3. Validate model

- Evaluates the trained model on the default `test` split with precision, recall, mAP, and per-class metrics.  
- You may customize which split to evaluate by modifying the YAML file.  
- Validation results and logs are saved in the `valid` results folder.

***

### 4. Predict

***

## Command-line Arguments Summary

| Argument       | Description                                                       |
|----------------|-----------------------------------------------------------------|
| `--do_preprocess` | Run label preprocessing pipeline across all splits             |
| `--do_train`       | Start fine-tuning YOLOv12 training                              |
| `--do_valid`       | Run model validation on test split                              |
| `--do_predict`     | Run predictions on test split                                   |
| `--epochs`         | Number of training epochs (default: 5)                         |
| `--batch_size`     | Batch size for training (default: 16)                          |
| `--device`         | Device string to use (e.g., `cuda:0`, `cpu`)                   |
| `--data_yaml`      | Optional path to dataset YAML config file                       |

***

## Dataset YAML Format

- Must specify paths to `train`, `valid` (or `val`), and `test` image folders.  
- Define `nc` for number of classes.  
- Provide `names` list for class labels corresponding to indices.

Example snippet:


***

## Dependencies and Installation

This project depends on the following main Python packages:

- ultralytics  
- torch  
- albumentations  
- opencv-python  
- numpy  

Installation example:


***

## Notes

- Make sure your dataset splits and label formats conform to the YOLO standard before training.  
- Labels are expected to include class indices and normalized coordinates or masks.  
- The pipeline design allows flexible execution of each stage independently from the command line.  
- GPU usage is recommended for training and inference to improve speed and efficiency.

***

## Contact

For questions, issues, or contributions, please contact:  
`your.email@example.com` (replace with your actual email)

***

## Citation

If you use this project or the YOLOv12 model, please cite:

***

## Acknowledgements

This project is based on and extends the work by the YOLOv12 authors [Chien Yeh et al.](https://github.com/sunsmarterjie/yolov12). We thank them for their excellent contribution to the field.

---

You can customize and expand this README as your project evolves.


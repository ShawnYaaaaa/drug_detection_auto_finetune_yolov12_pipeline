# Drug Detection Based on YOLOv12

An automated pipeline for fast drug recognition using the YOLOv12 detection and segmentation framework.

---

## Overview

This project implements a complete and automated pipeline for drug recognition based on the YOLOv12 framework. The pipeline supports dataset splitting, label preprocessing, fine-tuning with pretrained weights, validation, and prediction. All steps are organized into a user-friendly command-line interface for modular task execution.

---

## Project Structure and Key Components

| Module/File                | Description                                                                                 |
|---------------------------|---------------------------------------------------------------------------------------------|
| **converted_backlash.py**  | Utility for Windows path conversions to ensure consistent path usage                       |
| **predict_from_pretrained.py** | Functions for making predictions on various dataset splits using pretrained models      |
| **Preprocessing_labels.py** | Handles label preprocessing: fixing class indices, removing redundant tokens, formatting corrections |
| **finetune_yolov12.py**    | Facilitates YOLOv12 fine-tuning, validation, and prediction with proper logging and device management |
| **main.py**                | Integrates all modules and provides command-line flags for modular pipeline execution      |

---

## Pipeline Stages Overview

| Stage           | Description                                                                                  |
|-----------------|----------------------------------------------------------------------------------------------|
| **Preprocess**  | Generates `labels` folders for each split aligned with `images`. Processes label files to adjust class indices, remove redundant data, and verify format correctness. Prepares clean labels for training and evaluation. |
| **Train**       | Fine-tunes YOLOv12 using pretrained weights on the current dataset. Tracks training metrics such as loss and mAP. Concurrently runs validation on the validation split and saves best model weights (`best.pt`) and logs. |
| **Validate**    | Evaluates the trained model on the test split by default, reporting precision, recall, mAP, and per-class metrics. Evaluation can be customized by adjusting the YAML config. Saves results and logs in the `valid` folder. |
| **Predict**     | Runs model inference on the test split, saving predicted images and labels into the `predict` output folder. Outputs detection results for further analysis or visualization. |

---

## Usage

### Command-Line Arguments

| Argument         | Description                                       | Default      |
|------------------|-------------------------------------------------|--------------|
| `--do_preprocess`| Run label preprocessing for all dataset splits  | -            |
| `--do_train`     | Start fine-tuning YOLOv12 training               | -            |
| `--do_valid`     | Run model validation on the test (or specified) split | -         |
| `--do_predict`   | Run predictions using the trained model on the test split | -        |
| `--epochs`       | Number of training epochs                         | 50            |
| `--batch_size`   | Batch size for training                           | 16           |
| `--device`       | Device to use for training/inference (e.g., `cuda:0`, `cpu`) | -     |
| `--data_yaml`    | Path to dataset YAML configuration file          | -            |

---

## Dataset YAML Format

- Must specify paths to `train`, `valid` (or `val`), and `test` image folders.
- Define `nc` for the number of classes.
- Provide `names` list with class labels corresponding to indices.

Example snippet:


---

## Dependencies and Installation

This project depends on the following Python packages:

- ultralytics  
- torch  
- albumentations  
- opencv-python  
- numpy  

Installation example:


---

## Notes

- Ensure dataset splits and label formats conform to the YOLO standard before training.
- Labels should include proper class indices and normalized coordinates or masks.
- Each pipeline stage can be independently executed through command-line flags.
- GPU is recommended to accelerate training and inference.

---

## Contact

For questions, issues, or contributions, please contact:  
yehshawn133@gmail.com 

---

## Citation

If you use this project or YOLOv12 model, please cite the original authors and this work accordingly.

---

## Acknowledgements

This project is based on and extends the work by the YOLOv12 authors [Yunjie Tian et al.](https://github.com/sunsmarterjie/yolov12). We thank them for their excellent contribution to the field.

---

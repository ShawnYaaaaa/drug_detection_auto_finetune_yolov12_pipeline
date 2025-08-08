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


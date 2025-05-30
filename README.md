﻿# Warehouse-Object-detection
# Warehouse Package and Label Detection

This project uses [YOLOv8](https://github.com/ultralytics/ultralytics) for detecting packages and labels in warehouse images and videos. The dataset is sourced from [Roboflow](https://universe.roboflow.com/packageandbarcode/package-and-label-detection-agjvl/dataset/4) and annotated in YOLO format.

## Dataset
- **Classes:** `label`, `package`
- **Source:** [Roboflow Universe](https://universe.roboflow.com/packageandbarcode/package-and-label-detection-agjvl/dataset/4)
- **Format:** YOLOv8 (see `data.yaml`)
- **Train images:** `train/images/`
- **Validation images:** `valid/images/`
- **Test images:** `test/images/`

## Getting Started

### 1. Clone the repository
```bash
git clone <repo-url>
cd <repo-directory>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Training
Train a YOLOv8 model on your dataset:
```bash
python main.py
```
- This uses `yolov8n.pt` as the base model and trains for 100 epochs on GPU (CUDA).

### 4. Testing on Images
Run detection on test images and save results to the `results/` directory:
```bash
python test.py
```
- Input: images from `test/images/`
- Output: annotated images in `results/`

### 5. Video Inference
Run detection on a video and save the output video:
```bash
python video_test.py
```
- Input: video from `videos/1stvideo.mp4` (edit path as needed)
- Output: `videos/output_video_with_counts.mp4`

## Model Weights
- Trained weights are saved in `runs/detect/train5/weights/best.pt` by default.
- You can change the weights path in `test.py` and `video_test.py` as needed.

## Requirements
See [requirements.txt](requirements.txt) for all dependencies.

## Notes
- The project expects a CUDA-capable GPU for training and inference.
- The dataset and code are provided under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/)

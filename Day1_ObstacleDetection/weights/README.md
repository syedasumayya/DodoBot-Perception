# Trained Weights

Put your downloaded `best.pt` (from Colab, Step 7 of `train_yolov8_coco.py`) in this folder.

## Before pushing to GitHub

Check the file size first:

- **Under ~50MB** (this YOLOv8n run was ~6.5MB, so you're fine): just `git add` it normally.
- **Over ~50MB** (e.g. if you later train a larger YOLOv8 variant): GitHub blocks files over 100MB
  by default and warns above 50MB. Use [Git LFS](https://git-lfs.com/) instead:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add weights/best.pt
git commit -m "Add trained YOLOv8 obstacle detection weights"
git push
```

## Loading this model later (e.g. Day 5 ROS2 inference node)

```python
from ultralytics import YOLO
model = YOLO("Day1_ObstacleDetection/weights/best.pt")
results = model.predict(source="your_image_or_frame.jpg", conf=0.4)
```

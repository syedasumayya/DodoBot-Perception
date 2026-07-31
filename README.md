# DoDo Bot — Perception Stack

A restaurant service robot's perception work: obstacle detection, gesture recognition, and safety-zone logic. This repo tracks daily progress.

## Status Overview

| Track | Status | Details |
|---|---|---|
| **Day 1 — Obstacle Detection (COCO bootstrap)** | ✅ Done | YOLOv8n trained, mAP50 = 0.482. See `Day1_ObstacleDetection/` |
| **Day 1 — Live ROS2 sensor recording** | 🔲 Not started | Original plan in `planning_docs/`; requires physical LiDAR + OAK-D hardware |
| **Day 2 — Restaurant video test** | 🔲 Not started | Reuses Day 1's trained model; needs a restaurant video source |
| **Day 3 — Gesture Recognition** | 🔲 Not started | Plan in `planning_docs/`; replaces an earlier emotion-detection track |
| **Safety zone (distance computation, ROS)** | 🔲 Not started | Separate ROS2 task, laptop camera, 0.3m safe/unsafe threshold |

## Repo Structure

```
DodoBot-Perception/
├── README.md                       <- you are here
├── Day1_ObstacleDetection/
│   ├── train_yolov8_coco.py        <- full working pipeline, commented
│   ├── results/
│   │   └── training_log.txt        <- per-epoch metrics, final scores
│   └── weights/
│       └── best.pt                 <- ADD THIS: your downloaded trained model
├── Day2_RestaurantVideoTest/       <- empty, next task
├── Day3_GestureRecognition/        <- empty, not started
├── SafetyZone_ROS/                 <- empty, not started
├── docs/
│   ├── Day1_Report.pdf             <- full write-up with screenshots
│   └── screenshots/                <- raw screenshots used in the report
└── planning_docs/                  <- original task-plan documents (.docx)
```

## Day 1 — What Was Actually Done

Trained a YOLOv8-nano object detector as an obstacle-detection baseline, using a
filtered COCO subset (bootstrap approach — no robot hardware needed yet).

**Result:** mAP50 = 0.482 overall. Restaurant-relevant classes performed well:
person (0.766), dining table (0.563), chair (0.547).

**Full step-by-step explanation, including every error hit and how it was
fixed, is in `docs/Day1_Report.pdf`.** Short version of the debugging journey:

1. Planned to use `fiftyone` to auto-download a filtered COCO subset — blocked
   by a broken `_Ink` import bug in fiftyone 1.19.0 (not fixable via Pillow version).
2. Switched to filtering COCO locally with `pycocotools` instead, using
   already-downloaded full COCO data.
3. Zipped and uploaded the ~315MB filtered subset (2000 images) to Colab.
4. Hit a "cannot find zipfile" error — caused by unzipping before the upload
   fully finished, not a code bug.
5. Found `dataset.yaml` was missing a `train:` key — added it manually.
6. Trained YOLOv8n for 20 epochs on a T4 GPU (~18 minutes) — completed with
   no errors.

**Known limitation to fix later:** the final dataset ended up covering 79 COCO
classes instead of the originally intended 5 (person, chair, dining table,
backpack, handbag), because the fiftyone class-filter step never actually ran.
The model still works and the 5 target classes score reasonably, but a proper
5-class-only retrain would likely improve accuracy on exactly what DoDo Bot
needs. Re-running the `pycocotools` filter step in `train_yolov8_coco.py`
(Step 1 comments) with `MAX_IMAGES_PER_CLASS` raised would fix this.

## How to Reproduce

1. Open `Day1_ObstacleDetection/train_yolov8_coco.py` in Google Colab.
2. Follow the numbered steps in the file — Step 1 runs locally on your machine
   (filtering COCO), Steps 2 onward run in Colab.
3. Enable GPU: `Runtime` → `Change runtime type` → `T4 GPU`.

## Next Steps

- Day 2: point `trained_model.predict(source=...)` at a restaurant video to
  test detection in a realistic setting, and check floor-level detections.
- Day 3: build the gesture recognizer (MediaPipe landmarks + classifier) per
  `planning_docs/DoDoBot_Day3_GestureRecognition.docx`.
- Safety zone: separate ROS2 node computing distance-to-obstacle from camera/
  LiDAR input, triggering a direction change below 0.3m.

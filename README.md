# DoDo Bot — Perception Stack

A restaurant service robot's perception system: detecting obstacles on the
floor and recognizing hand gestures from customers, so the robot can
navigate safely and respond to simple commands.

This repo contains two trained models, the datasets and pipelines used to
build them, and full documentation of how each one works.

---

## Models

### 1. Obstacle Detection

**What it does:** Detects people, chairs, dining tables, and other
restaurant-floor objects in a camera frame, so the robot can identify and
avoid obstacles as it moves.

**Architecture:** YOLOv8-nano, fine-tuned from pretrained COCO weights.

**Dataset:** A filtered subset of [COCO](https://cocodataset.org/) (2,000
images), built locally with `pycocotools` and covering 79 object classes
relevant to an indoor restaurant setting — with a particular focus on
`person`, `chair`, `dining table`, `backpack`, and `handbag`.

**How it works:** The model takes a single camera frame as input and
outputs bounding boxes with class labels and confidence scores for every
object it recognizes. It was trained for 20 epochs on a T4 GPU (~18
minutes).

**Accuracy:**

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| person | 0.872 | 0.632 | 0.766 |
| dining table | 0.764 | 0.467 | 0.563 |
| chair | 0.673 | 0.474 | 0.547 |
| backpack | 0.586 | 0.220 | 0.272 |
| handbag | 0.564 | 0.114 | 0.196 |
| **All 79 classes (overall)** | **0.676** | **0.440** | **0.482** |

**Real-world validation:** The trained model was also tested against a
restaurant video it had never seen during training — it detected chairs,
tables, and people consistently across all 183 frames, and correctly
flagged floor-level obstacles using a bounding-box-position heuristic. See
`ObstacleDetection/video_validation/`.

**Known limitation:** This model only recognizes its 79 trained COCO
classes — it does not detect arbitrary/unknown objects (e.g. random
litter, dropped items outside those categories). Closing that gap requires
either an open-vocabulary detector or, more practically for a robot,
distance-based sensing (LiDAR/depth) as a hardware-level safety net that
doesn't depend on knowing what an object is — see **Next Phase** below.

Full write-up: [`docs/ObstacleDetection_Report.pdf`](docs/ObstacleDetection_Report.pdf)

---

### 2. Gesture Recognition

**What it does:** Recognizes static hand gestures (e.g. stop, thumbs up,
OK sign) from a single image, so the robot can respond to simple visual
commands from customers or staff.

**Architecture:** MediaPipe hand-landmark extraction (21 landmarks × x/y/z
= 63 features) feeding into a Random Forest classifier (200 trees) — a
lightweight pipeline chosen specifically so it can run in real time on the
robot's onboard hardware without needing a GPU.

**Dataset:** [HaGRID](https://github.com/hukenovs/hagrid) (Hand Gesture
Recognition Image Dataset), 300 images per class across 9 gesture classes.

**How it works:** MediaPipe locates a hand in the image and extracts 21
3D keypoints (fingertips, joints, wrist). Those 63 numbers — not the raw
image — are what the classifier is trained on, which keeps the model
small and fast.

**Gesture set and robot response:**

| Gesture | Robot Response |
|---|---|
| Open palm | Halt movement |
| Fist | Emergency stop |
| OK sign | Task-complete confirmation |
| Peace / V | Menu / bill request |
| Thumbs up | Order confirmed |
| Thumbs down | Order cancelled / issue flagged |
| Call gesture | Alert staff |
| Index pointing | Directional command |
| Mute/quiet | Do-not-disturb request |

**Accuracy:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| palm | 0.97 | 1.00 | 0.98 |
| fist | 0.93 | 0.98 | 0.95 |
| ok | 1.00 | 0.96 | 0.98 |
| peace | 0.98 | 0.98 | 0.98 |
| like | 0.94 | 0.85 | 0.90 |
| dislike | 0.98 | 0.98 | 0.98 |
| call | 0.83 | 0.95 | 0.88 |
| one | 0.89 | 0.89 | 0.89 |
| mute | 0.96 | 0.85 | 0.90 |
| **Overall accuracy** | | | **0.94** |

**How this number was reached:** an earlier 10-class version (which
included a `two_up` gesture) scored only 89%, because `two_up` and `peace`
were visually near-identical in landmark space and got confused with each
other constantly — confirmed by testing both a Random Forest and a neural
network, which hit the same wall. Dropping `two_up` (keeping the more
useful `peace` gesture) raised accuracy to 94%, with `peace`'s own score
jumping from 0.66 to 0.98.

Full write-up: [`docs/GestureRecognition_Report.pdf`](docs/GestureRecognition_Report.pdf)

---

## Repo Structure

```
DodoBot-Perception/
├── README.md
├── ObstacleDetection/
│   ├── train_yolov8_coco.py          <- full training pipeline, commented
│   ├── video_validation/README.md    <- real-world video test
│   ├── results/training_log.txt      <- per-epoch metrics
│   └── weights/best.pt               <- trained model
├── GestureRecognition/
│   ├── train_gesture_classifier.py   <- full training pipeline, commented
│   ├── results/classification_reports.txt
│   └── weights/gesture_classifier_final.pkl
├── docs/
│   ├── ObstacleDetection_Report.pdf  <- full write-up incl. debugging log
│   ├── GestureRecognition_Report.pdf <- full write-up incl. debugging log
│   └── screenshots/
├── planning_docs/                    <- original project planning documents
└── SafetyZone_ROS/                   <- next phase, see below
```

## How to Reproduce

Both models were built and trained in Google Colab.

1. **Obstacle Detection** — open `ObstacleDetection/train_yolov8_coco.py`,
   follow the numbered steps (Step 1 runs locally to filter COCO; the rest
   runs in Colab). Enable GPU: `Runtime` → `Change runtime type` → `T4 GPU`.
2. **Gesture Recognition** — open `GestureRecognition/train_gesture_classifier.py`
   and follow the numbered steps. Requires MediaPipe's current Tasks API
   (`HandLandmarker`) — the older `mp.solutions.hands` interface was
   removed as of MediaPipe 1.0.0.

## Next Phase

- **Safety-zone distance sensing (ROS2):** a separate node computing
  distance-to-obstacle from camera/LiDAR input, triggering a direction
  change when something is within 0.3m — the hardware-level safety net
  that catches obstacles regardless of whether the vision model recognizes
  them.
- **On-robot deployment:** loading both trained models onto the robot's
  onboard camera (OAK-D) for real-time inference.
- **Idle/no-gesture class:** HaGRID doesn't include a baseline
  "no gesture" class — needed so the robot can distinguish "customer is
  gesturing" from "no one is gesturing."

"""
DoDo Bot — Obstacle Detection Bootstrap
Trains a YOLOv8-nano object detector on a filtered COCO subset.

Run this in Google Colab (GPU runtime recommended: Runtime > Change runtime type > T4 GPU).
This is the consolidated, working version of the pipeline after debugging the
fiftyone/Pillow import error and the missing `train:` key in dataset.yaml.

Pipeline:
  1. Filter COCO locally (pycocotools) -> coco_subset_yolo.zip   [run on your laptop, not here]
  2. Upload coco_subset_yolo.zip to Colab session storage
  3. Unzip + verify structure
  4. Fix / write dataset.yaml
  5. Train YOLOv8n
  6. Evaluate + run inference on sample images
  7. Download trained weights (best.pt)
"""

# ---------------------------------------------------------------------------
# STEP 1 (reference only — this part runs on your LAPTOP, not in Colab)
# ---------------------------------------------------------------------------
# See planning_docs/ for the original Day 1 plan. Short version of what was
# actually run locally to filter full COCO down to a manageable subset:
#
#   pip install pycocotools tqdm
#
#   from pycocotools.coco import COCO
#   coco = COCO("annotations/instances_train2017.json")
#   cat_ids = coco.getCatIds(catNms=["person","chair","dining table","backpack","handbag"])
#   ... filter images/annotations containing those classes, copy to coco_subset/,
#   ... write annotations_filtered.json, then zip the folder.
#
# NOTE: the zip that actually got produced and used ended up being a broader
# COCO-derived YOLO export (79 classes, not just the 5 targeted) — see the
# "Known Issues" section in docs/ObstacleDetection_Report.pdf for why, and what to redo
# if a narrower 5-class model is needed later.


# ---------------------------------------------------------------------------
# STEP 2 — Upload coco_subset_yolo.zip into Colab
# ---------------------------------------------------------------------------
# Manual step: click the folder icon in Colab's left sidebar -> upload icon ->
# select coco_subset_yolo.zip. Wait for the upload spinner to fully disappear
# before continuing (uploading large files is the #1 source of "cannot find
# zipfile" errors if you unzip too early).


# ---------------------------------------------------------------------------
# STEP 3 — Unzip and verify folder structure
# ---------------------------------------------------------------------------
import os

os.system("unzip -q /content/coco_subset_yolo.zip -d /content/")

print(os.listdir("/content/coco_subset_yolo"))
# Expected: ['labels', 'dataset.yaml', 'images']

# Deeper check — confirms images/labels actually contain files, and reveals
# whether they're nested inside a val/ subfolder (they were, in our run):
def show_tree(path, prefix="", max_items=10):
    items = sorted(os.listdir(path))
    print(prefix + os.path.basename(path) + "/")
    for item in items[:max_items]:
        full = os.path.join(path, item)
        if os.path.isdir(full):
            show_tree(full, prefix + "  ", max_items)
        else:
            print(prefix + "  " + item)
    if len(items) > max_items:
        print(prefix + f"  ... and {len(items) - max_items} more")

show_tree("/content/coco_subset_yolo")


# ---------------------------------------------------------------------------
# STEP 4 — Fix dataset.yaml
# ---------------------------------------------------------------------------
# The zip's dataset.yaml was missing a `train:` key (only had `val:`).
# For tonight's baseline we point both train and val at the same images/val
# folder. This is fine for a quick working pipeline; NOT ideal for a real
# generalization benchmark since it validates on data it also trained on —
# do a proper train/val split before treating these numbers as final.

yaml_content = """path: /content/coco_subset_yolo
train: images/val
val: images/val
names:
  0: sandwich
  1: car
  2: chair
  3: dining table
  4: person
  5: wine glass
  6: handbag
  7: backpack
  8: bottle
  9: cell phone
  10: laptop
  11: tv
  12: keyboard
  13: cup
  14: pizza
  15: sports ball
  16: book
  17: suitcase
  18: traffic light
  19: potted plant
  20: bicycle
  21: umbrella
  22: donut
  23: cake
  24: teddy bear
  25: bowl
  26: remote
  27: clock
  28: apple
  29: spoon
  30: knife
  31: bench
  32: surfboard
  33: orange
  34: refrigerator
  35: tie
  36: truck
  37: bird
  38: fork
  39: couch
  40: hot dog
  41: broccoli
  42: oven
  43: sink
  44: banana
  45: vase
  46: bed
  47: bus
  48: dog
  49: cow
  50: parking meter
  51: horse
  52: elephant
  53: skateboard
  54: motorcycle
  55: giraffe
  56: kite
  57: train
  58: stop sign
  59: microwave
  60: scissors
  61: baseball bat
  62: carrot
  63: snowboard
  64: frisbee
  65: baseball glove
  66: skis
  67: tennis racket
  68: cat
  69: airplane
  70: toothbrush
  71: mouse
  72: boat
  73: fire hydrant
  74: sheep
  75: toilet
  76: zebra
  77: hair drier
  78: bear
"""
with open("/content/coco_subset_yolo/dataset.yaml", "w") as f:
    f.write(yaml_content)

print(open("/content/coco_subset_yolo/dataset.yaml").read())


# ---------------------------------------------------------------------------
# STEP 5 — Install ultralytics and train YOLOv8n
# ---------------------------------------------------------------------------
os.system("pip install ultralytics -q")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # pretrained COCO weights, fine-tuned on our subset

results = model.train(
    data="/content/coco_subset_yolo/dataset.yaml",
    epochs=20,
    imgsz=640,
    name="dodo_obstacle_baseline",
)

# Result from the run this repo documents:
#   20 epochs, ~18 min on a T4 GPU
#   Overall mAP50 = 0.482
#   Key obstacle classes: person 0.766, dining table 0.563, chair 0.547
#   (Full per-class table in docs/ObstacleDetection_Report.pdf and results/training_log.txt)


# ---------------------------------------------------------------------------
# STEP 6 — Evaluate + run inference on sample images
# ---------------------------------------------------------------------------
from IPython.display import Image as IPImage, display

display(IPImage(filename="/content/runs/detect/dodo_obstacle_baseline/results.png"))

trained_model = YOLO("/content/runs/detect/dodo_obstacle_baseline/weights/best.pt")
predict_results = trained_model.predict(
    source="/content/coco_subset_yolo/images/val",
    conf=0.4,
    save=True,
)
print("Annotated predictions saved to:", predict_results[0].save_dir)


# ---------------------------------------------------------------------------
# STEP 7 — Download trained weights
# ---------------------------------------------------------------------------
from google.colab import files
files.download("/content/runs/detect/dodo_obstacle_baseline/weights/best.pt")

# Save best.pt into ObstacleDetection/weights/ in this repo after downloading.
# NOTE: if best.pt is over ~50MB, use Git LFS to push it to GitHub — see README.md.

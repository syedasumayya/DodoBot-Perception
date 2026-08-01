# Restaurant Video Validation

No separate training happened today — this reuses the trained model from
`ObstacleDetection/weights/best.pt` and tests it against a novel video
it never saw during training.

## What was done

1. Sourced a restaurant floor video (chairs, tables, people walking) from
   a free stock footage site (Pexels), since AI video generation tools
   weren't producing usable output at the time.
2. Ran `trained_model.predict(source="video.mp4", conf=0.4, save=True)` —
   Ultralytics automatically processes the video frame-by-frame.
3. Checked floor-level detections separately using a simple heuristic:
   any detection whose bounding-box vertical center falls in the bottom
   30% of the frame is treated as "floor region."

## Code used

```python
from ultralytics import YOLO

trained_model = YOLO("/content/best.pt")

video_results = trained_model.predict(
    source="/content/your_restaurant_video.mp4",
    conf=0.4,
    save=True,
)

# Floor-level check
for i, r in enumerate(video_results):
    h = r.orig_shape[0]
    floor_hits = []
    for box in r.boxes:
        cls_name = trained_model.names[int(box.cls)]
        y_center = float(box.xywh[0][1])
        if y_center > h * 0.7:
            floor_hits.append(f"{cls_name} ({float(box.conf):.2f})")
    if floor_hits:
        print(f"Frame {i+1}: {floor_hits}")
```

## Results

- Video processed cleanly across all 183 frames, no errors.
- **Obstacle detection**: `chair` and `dining table` detected in nearly
  every frame; `person` detected consistently (4-14 people per frame as
  they moved through the scene); `bench` picked up in later frames despite
  not being explicitly emphasized in training.
- **Floor-level detection**: consistently caught chairs and dining tables
  near the bottom of frame across all sampled frames, with strong
  confidence (many detections 0.8-0.94).
- Minor note: `person` sometimes appears in the floor-region check too —
  expected, since a person's feet/legs naturally sit in the lower part of
  the frame even though the person as a whole isn't a "floor object."

## Known limitation

The annotated output video and full per-frame log weren't saved into this
repo (only observed in the Colab session). Re-run the code above with a
saved video source to regenerate and archive the annotated output if
needed for a demo.

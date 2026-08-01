# Trained Gesture Classifier

Put your downloaded `gesture_classifier_final.pkl` (from Colab, Step 6 of
`train_gesture_classifier.py`) in this folder.

This is a small file (Random Forest model), well under GitHub's size limits —
just `git add` it normally, no Git LFS needed.

## Loading this model later

```python
import pickle
import numpy as np

with open("GestureRecognition/weights/gesture_classifier_final.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
classes = saved["classes"]  # ["palm", "fist", "ok", "peace", "like", "dislike", "call", "one", "mute"]

# landmarks = extract_landmarks(...)  # 63-value array from MediaPipe HandLandmarker
prediction = model.predict([landmarks])
print(classes[prediction[0]])
```

## Dependency note

Extracting landmarks at inference time requires MediaPipe's **Tasks API**
(`HandLandmarker`), not the older `mp.solutions.hands` interface, which was
removed as of MediaPipe 1.0.0. See `train_gesture_classifier.py` Step 2 for
the working extraction code and the required `hand_landmarker.task` model
file download.

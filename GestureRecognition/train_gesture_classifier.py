"""
DoDo Bot — Gesture Recognition
Trains a lightweight gesture classifier using MediaPipe hand landmarks + HaGRID dataset.

Run this in Google Colab (GPU optional — only the neural network comparison step
benefits from it; landmark extraction and Random Forest are CPU-only).

Pipeline:
  1. Download HaGRID subset via kagglehub
  2. Extract MediaPipe hand landmarks (63 features per image) using the NEW
     Tasks API (mp.solutions.hands is deprecated as of MediaPipe 1.0.0)
  3. Train + compare Random Forest vs. small neural network
  4. Diagnose class confusion via confusion matrix
  5. Drop the confused class (two_up), retrain, finalize
  6. Save + download the final model
"""

import os
import numpy as np
import cv2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# STEP 1 — Download HaGRID dataset
# ---------------------------------------------------------------------------
# !pip install kagglehub mediapipe -q

import kagglehub
path = kagglehub.dataset_download("innominate817/hagrid-classification-512p")
inner_path = os.path.join(path, "hagrid-classification-512p")
print(os.listdir(inner_path))
# Real HaGRID classes found: three2, ok, call, one, palm, fist, stop_inverted,
# three, peace, two_up_inverted, dislike, peace_inverted, like, four, rock,
# two_up, mute, stop
# (Original Day 3 plan assumed classes like "wave"/"beckoning"/"both hands up"
#  which don't exist in HaGRID since it's a static single-hand dataset. Mapped
#  to the closest available static gestures instead — see README for the
#  full mapping table and reasoning.)


# ---------------------------------------------------------------------------
# STEP 2 — Extract MediaPipe hand landmarks (NEW Tasks API)
# ---------------------------------------------------------------------------
# IMPORTANT: mp.solutions.hands (the API in the original Day 3 plan) throws
#   AttributeError: module 'mediapipe' has no attribute 'solutions'
# as of MediaPipe 1.0.0 — the legacy "solutions" interface was removed.
# Fixed by switching to the Tasks API (HandLandmarker) below.

# !wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        coords = []
        for lm in landmarks:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords)  # 21 landmarks x 3 = 63 features
    return None

# Initial 10-class attempt (see Step 4 for why two_up was later dropped)
GESTURE_CLASSES = ["palm", "fist", "ok", "peace", "like", "dislike", "call", "one", "two_up", "mute"]
MAX_PER_CLASS = 300

X, y = [], []
for label, gesture in enumerate(GESTURE_CLASSES):
    folder_path = os.path.join(inner_path, gesture)
    files = os.listdir(folder_path)[:MAX_PER_CLASS]
    print(f"Processing '{gesture}': {len(files)} images")
    for img_file in tqdm(files):
        landmarks = extract_landmarks(os.path.join(folder_path, img_file))
        if landmarks is not None:
            X.append(landmarks)
            y.append(label)

X = np.array(X)
y = np.array(y)
print("Feature shape:", X.shape, "Label shape:", y.shape)
# Result: (2777, 63) — 2777/3000 images (92.5%) had a detectable hand.


# ---------------------------------------------------------------------------
# STEP 3 — Train + compare Random Forest vs. neural network
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print("--- Random Forest (10 classes) ---")
print(classification_report(y_val, y_pred, target_names=GESTURE_CLASSES))
# Result: 89% overall accuracy

# Neural network comparison
from tensorflow.keras import layers, models

nn_model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(63,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(GESTURE_CLASSES), activation='softmax')
])
nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=32)

y_pred_nn = np.argmax(nn_model.predict(X_val), axis=1)
print("--- Neural Network (10 classes) ---")
print(classification_report(y_val, y_pred_nn, target_names=GESTURE_CLASSES))
# Result: 88% overall accuracy — essentially tied with Random Forest.


# ---------------------------------------------------------------------------
# STEP 4 — Diagnose class confusion
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_val, y_pred)
print(pd.DataFrame(cm, index=GESTURE_CLASSES, columns=GESTURE_CLASSES))
# Finding: `peace` and `two_up` are heavily confused with each other in BOTH
# models (21-26 misclassifications each way). Since both classifiers — one
# tree-based, one a neural network — hit the same wall on the same class
# pair, the issue is in the features/gesture choice, not the model:
# `peace` and `two_up` are genuinely very close in 3D landmark space (both
# are "two fingers extended," just different finger pairs).
#
# Decision: drop `two_up`. In the original gesture table, `peace` maps to
# "menu/bill request" (intuitive, likely to be used) while `two_up` was an
# arbitrary custom mapping to "navigation command" — `one` (pointing) can
# reasonably cover directional intent instead.


# ---------------------------------------------------------------------------
# STEP 5 — Retrain on final 9 classes
# ---------------------------------------------------------------------------
FINAL_CLASSES = ["palm", "fist", "ok", "peace", "like", "dislike", "call", "one", "mute"]
two_up_label = GESTURE_CLASSES.index("two_up")

keep_mask = y != two_up_label
X_final = X[keep_mask]
y_final_raw = y[keep_mask]

old_to_new = {old_label: new_idx for new_idx, old_label in
              enumerate([i for i in range(len(GESTURE_CLASSES)) if i != two_up_label])}
y_final = np.array([old_to_new[label] for label in y_final_raw])

print("Final dataset shape:", X_final.shape, y_final.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

final_clf = RandomForestClassifier(n_estimators=200, random_state=42)
final_clf.fit(X_train, y_train)
y_pred_final = final_clf.predict(X_val)

print("--- FINAL Random Forest (9 classes) ---")
print(classification_report(y_val, y_pred_final, target_names=FINAL_CLASSES))
# Result: 94% overall accuracy — peace f1-score jumped from 0.66 to 0.98.


# ---------------------------------------------------------------------------
# STEP 6 — Save + download final model
# ---------------------------------------------------------------------------
import pickle

with open("gesture_classifier_final.pkl", "wb") as f:
    pickle.dump({"model": final_clf, "classes": FINAL_CLASSES}, f)

from google.colab import files
files.download("gesture_classifier_final.pkl")

# Save this .pkl into GestureRecognition/weights/ in this repo after downloading.

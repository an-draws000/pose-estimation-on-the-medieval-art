import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.utils import compute_class_weight

import pickle

def normalize_keypoints(xy, bbox):
    x_min, y_min, w, h = bbox
    xy_norm = xy.copy().astype(float)
    xy_norm[:, 0] = (xy[:, 0] - x_min) / (w + 1e-6)
    xy_norm[:, 1] = (xy[:, 1] - y_min) / (h + 1e-6)
    return xy_norm.flatten()  

def angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def build_features(xy_raw, xy_norm):
    feats = []
    feats.extend(xy_norm.flatten()) 

    feats.append(angle(xy_raw[9],  xy_raw[7],  xy_raw[5]))   # left elbow
    feats.append(angle(xy_raw[10], xy_raw[8],  xy_raw[6]))   # right elbow
    feats.append(angle(xy_raw[13], xy_raw[11], xy_raw[5]))   # left hip
    feats.append(angle(xy_raw[14], xy_raw[12], xy_raw[6]))    # right hip
    feats.append(angle(xy_raw[15], xy_raw[13], xy_raw[11]))  # left knee
    feats.append(angle(xy_raw[16], xy_raw[14], xy_raw[12]))  # right knee

    return feats

ANNO_DIR  = Path("C:/Users/admin/Desktop/Dataset/")
def load_data(anno_dir):
    
    X = []
    y = []
    SKIP_CLASSES = {"objects"} 
    
    for anno_file in anno_dir.glob("*.json"):
        with open(anno_file) as f:
            data = json.load(f) 

    
        cat_lookup = {cat["id"]: cat["name"] for cat in data["categories"]}

        
        for ann in data["annotations"]:

            label = cat_lookup[ann["category_id"]]

            if label in SKIP_CLASSES:
                continue

            if "keypoints" not in ann or len(ann["keypoints"]) == 0:
                continue
            kp = np.array(ann["keypoints"]).reshape(-1, 3)  
            if len(kp) < 17:
                pad = np.zeros((17 - len(kp), 3))
                kp = np.vstack([kp, pad])

            xy = kp[:, :2]  # (17, 2)
            x_min, y_min, w, h = [float(v) for v in ann["bbox"]]

    
            normalized_kp = normalize_keypoints(xy, (x_min, y_min, w, h)).reshape(17, 2)
           
            X.append(build_features(xy, normalized_kp))
            
            y.append(label)

    return np.array(X), np.array(y)

X, y = load_data(ANNO_DIR)
print(f"Samples: {len(y)}")
print(f"Classes: {set(y)}")
print(f"Feature shape: {X.shape}")

classes = np.unique(y)
weights = compute_class_weight("balanced", classes=classes, y=y)
weight_dict = dict(zip(classes, weights))

weight_dict["student standing"] *= 2

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", class_weight=weight_dict, probability=True))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
# print(f"CV F1: {scores.mean():.3f} ± {scores.std():.3f}")

# # Train on full dataset
# pipeline.fit(X, y)

param_grid = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto", 0.01, 0.001]
}

grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro")
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")
print(f"Best F1: {grid.best_score_:.3f}")

y_pred = cross_val_predict(grid.best_estimator_, X, y, cv=cv)
cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))  
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.best_estimator_.classes_)
disp.plot(ax=ax, xticks_rotation=45)  
plt.tight_layout()                     
plt.savefig("SVC_man_cm.png", dpi=150, bbox_inches="tight")
plt.show()

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

print("\nClassification Report:")
print(classification_report(y, y_pred))

from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline

pipeline_proba = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced",
                C=grid.best_params_["clf__C"],
                gamma=grid.best_params_["clf__gamma"]))
])

y_prob = cross_val_predict(pipeline_proba, X, y, cv=cv, method='predict_proba')
classes_list = np.unique(y)
y_bin = label_binarize(y, classes=classes_list)
print(f"Macro ROC-AUC: {roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro'):.3f}")
for i, cls in enumerate(classes_list):
    print(f"  ROC-AUC [{cls}]: {roc_auc_score(y_bin[:, i], y_prob[:, i]):.3f}")

import random
random.seed(42)
np.random.seed(42)

best_model = grid.best_estimator_
best_model.fit(X, y)

with open("SVC_man_no_aug.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved.")
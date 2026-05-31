import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from ultralytics import YOLO
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.utils import compute_class_weight
import pickle
import random
random.seed(42)
np.random.seed(42)

yolo = YOLO("yolov8x-pose.pt")

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

def bbox_iou(boxA, boxB):
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0] + boxA[2], boxA[1] + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0] + boxB[2], boxB[1] + boxB[3]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    return inter_area / (boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter_area + 1e-6)

def build_features(xy_raw, xy_norm):

    feats = list(xy_norm.flatten())                            # 34 normalized coords
    feats.append(angle(xy_raw[9],  xy_raw[7],  xy_raw[5]))    # left elbow
    feats.append(angle(xy_raw[10], xy_raw[8],  xy_raw[6]))    # right elbow
    feats.append(angle(xy_raw[13], xy_raw[11], xy_raw[5]))    # left hip
    feats.append(angle(xy_raw[14], xy_raw[12], xy_raw[6]))    # right hip
    feats.append(angle(xy_raw[15], xy_raw[13], xy_raw[11]))   # left knee
    feats.append(angle(xy_raw[16], xy_raw[14], xy_raw[12]))   # right knee
    return feats

ANNO_DIR  = Path("C:/Users/admin/Desktop/Dataset/")
IMAGE_DIR = Path("C:/Users/admin/Desktop/Dataset/pictures/")
SKIP_CLASSES  = {"objects"}
IOU_THRESHOLD = 0.001

def load_data(anno_dir, image_dir):
    X, y = [], []
    skipped = 0

    for anno_file in anno_dir.glob("*.json"):
        with open(anno_file) as f:
            data = json.load(f)

        cat_lookup   = {cat["id"]: cat["name"] for cat in data["categories"]}
        image_lookup = {img["id"]: img         for img in data["images"]}

        for ann in data["annotations"]:
            label = cat_lookup[ann["category_id"]]
            if label in SKIP_CLASSES:
                continue
            if "keypoints" not in ann or len(ann["keypoints"]) == 0:
                continue

            img_info   = image_lookup[ann["image_id"]]
            image_path = image_dir / img_info["file_name"]
            if not image_path.exists():
                print(f"Missing image: {image_path}")
                skipped += 1
                continue

            coco_bbox = [float(v) for v in ann["bbox"]]  

            results = yolo(str(image_path), verbose=False)
            if not results[0].keypoints or len(results[0].keypoints.xy) == 0:
                skipped += 1
                continue

            yolo_boxes = results[0].boxes.xywh.cpu().numpy()
            best_iou, best_idx = 0, -1
            for i, box in enumerate(yolo_boxes):
                yolo_bbox = [box[0] - box[2]/2, box[1] - box[3]/2, box[2], box[3]]
                iou = bbox_iou(coco_bbox, yolo_bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_idx == -1 or best_iou < IOU_THRESHOLD:
                print(f"No IoU match: {image_path.name} (best IoU: {best_iou:.2f})")
                skipped += 1
                continue

            else:
                xy_yolo = results[0].keypoints.xy[best_idx].cpu().numpy()
                if len(xy_yolo) < 17:
                    print(f"WARNING: only {len(xy_yolo)} keypoints for {image_path.name}")
                    pad = np.zeros((17 - len(xy_yolo), 2))
                    xy_yolo = np.vstack([xy_yolo, pad])
                x_min, y_min, w, h = coco_bbox
                xy_2d = normalize_keypoints(xy_yolo, (x_min, y_min, w, h)).reshape(17, 2)

            X.append(build_features(xy_yolo, xy_norm=xy_2d))
            y.append(label)
            print(f"OK: {image_path.name} | IoU: {best_iou:.2f} | {label}")

    print(f"\nLoaded: {len(y)} samples | Skipped: {skipped}")
    return np.array(X), np.array(y)

X, y = load_data(ANNO_DIR, IMAGE_DIR)
print(f"Classes:       {set(y)}")
print(f"Feature shape: {X.shape}")


classes = np.unique(y)
weights = compute_class_weight("balanced", classes=classes, y=y)
weight_dict = dict(zip(classes, weights))
weight_dict["student standing"] *= 2  # manual boost

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", class_weight=weight_dict, probability=True))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "clf__C":     [0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto", 0.01, 0.001]
}

grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro", verbose=1)
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")
print(f"Best F1:     {grid.best_score_:.3f}")


y_pred = cross_val_predict(grid.best_estimator_, X, y, cv=cv)
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.best_estimator_.classes_)
disp.plot(ax=ax, xticks_rotation=45)
plt.tight_layout()
plt.savefig("SVC_yolo_cm.png", dpi=150, bbox_inches="tight")
plt.show()

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

print("\nClassification Report:")
print(classification_report(y, y_pred))

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

best_model = grid.best_estimator_
best_model.fit(X, y)
with open("SVC_yolo_feats.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Model saved as SVC_yolo_feats.pkl")


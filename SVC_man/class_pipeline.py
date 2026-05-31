import ultralytics
import pickle
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

yolo = ultralytics.YOLO("yolov8x-pose.pt")

with open("SVC_man_no_aug.pkl", "rb") as f:
    classifier = pickle.load(f)

ANNO_DIR     = Path("C:/Users/admin/Desktop/Dataset/")
IMAGE_DIR    = Path("C:/Users/admin/Desktop/Dataset/pictures/")
IOU_THRESHOLD = 0.001
SKIP_CLASSES  = {"objects"}

COLORS = {
    "teacher sitting":  "#e74c3c",
    "teacher standing": "#e67e22",
    "student sitting":  "#2980b9",
    "student standing": "#27ae60"
}
ALL_CLASSES = list(COLORS.keys())
SHORT       = ["T.Sit", "T.Stand", "S.Sit", "S.Stand"]

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
    feats = list(xy_norm.flatten())
    feats.append(angle(xy_raw[9],  xy_raw[7],  xy_raw[5]))
    feats.append(angle(xy_raw[10], xy_raw[8],  xy_raw[6]))
    feats.append(angle(xy_raw[13], xy_raw[11], xy_raw[5]))
    feats.append(angle(xy_raw[14], xy_raw[12], xy_raw[6]))
    feats.append(angle(xy_raw[15], xy_raw[13], xy_raw[11]))
    feats.append(angle(xy_raw[16], xy_raw[14], xy_raw[12]))
    return np.array(feats).reshape(1, -1)

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

def run_evaluation(anno_dir, image_dir):
    all_gt, all_pred, all_conf, all_proba = [], [], [], []
    skipped = 0

    for anno_file in anno_dir.glob("*.json"):
        with open(anno_file) as f:
            data = json.load(f)

        cat_lookup   = {cat["id"]: cat["name"] for cat in data["categories"]}
        image_lookup = {img["id"]: img for img in data["images"]}

        for ann in data["annotations"]:
            gt_label = cat_lookup[ann["category_id"]]
            if gt_label in SKIP_CLASSES:
                continue
            if "keypoints" not in ann or len(ann["keypoints"]) == 0:
                continue

            img_info   = image_lookup[ann["image_id"]]
            image_path = image_dir / img_info["file_name"]
            if not image_path.exists():
                skipped += 1
                continue

            coco_bbox = [float(v) for v in ann["bbox"]]

            results = yolo(str(image_path), verbose=False)
            if not results[0].keypoints or len(results[0].keypoints.xy) == 0:
                print(f"No detection: {image_path.name}")
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

            xy_yolo = results[0].keypoints.xy[best_idx].cpu().numpy()
            if len(xy_yolo) < 17:
                xy_yolo = np.vstack([xy_yolo, np.zeros((17 - len(xy_yolo), 2))])

            x_min, y_min, w, h = coco_bbox
            xy_2d    = normalize_keypoints(xy_yolo, (x_min, y_min, w, h)).reshape(17, 2)
            features = build_features(xy_yolo, xy_2d)

            proba      = classifier.predict_proba(features)[0]
            pred_label = classifier.classes_[np.argmax(proba)]
            confidence = proba.max()

            all_gt.append(gt_label)
            all_pred.append(pred_label)
            all_conf.append(float(confidence))
            all_proba.append(proba)

            print(f"GT: {gt_label:20s} | Pred: {pred_label:20s} | IoU: {best_iou:.2f} | Conf: {confidence:.1%}")

    print(f"\nMatched: {len(all_gt)} | Skipped: {skipped}")
    print(f"Predictions: {Counter(all_pred)}")
    return all_gt, all_pred, all_conf, all_proba


all_gt, all_pred, all_conf, all_proba = run_evaluation(ANNO_DIR, IMAGE_DIR)

print("\nClassification Report:")
print(classification_report(all_gt, all_pred, labels=ALL_CLASSES))

y_bin       = label_binarize(all_gt, classes=ALL_CLASSES)
y_proba_arr = np.array(all_proba)
print(f"Macro ROC-AUC: {roc_auc_score(y_bin, y_proba_arr, multi_class='ovr', average='macro'):.3f}")
for i, cls in enumerate(ALL_CLASSES):
    print(f"  ROC-AUC [{cls}]: {roc_auc_score(y_bin[:, i], y_proba_arr[:, i]):.3f}")

cm = confusion_matrix(all_gt, all_pred, labels=ALL_CLASSES)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SHORT)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.savefig("SVC_man_yolo_inference_cm.png", dpi=150, bbox_inches="tight")
plt.show()
from ultralytics import YOLO
import numpy as np
import json
from pathlib import Path

yolo = YOLO("yolov8x-pose.pt")

def normalize_keypoints(xy, bbox):
    x_min, y_min, w, h = bbox
    xy_norm = xy.copy().astype(float)
    xy_norm[:, 0] = (xy[:, 0] - x_min) / (w + 1e-6)
    xy_norm[:, 1] = (xy[:, 1] - y_min) / (h + 1e-6)
    return xy_norm.flatten()

def bbox_iou(boxA, boxB):
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0] + boxA[2], boxA[1] + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0] + boxB[2], boxB[1] + boxB[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]

    return inter_area / (boxA_area + boxB_area - inter_area + 1e-6)

def collect_pairs(anno_dir, image_dir, iou_threshold=0.001):
    SKIP_CLASSES = {"objects"}
    
    X_yolo = []    # yolo keypoints
    X_manual = []  # manual keypoints
    y = []         # labels

    for anno_file in Path(anno_dir).glob("*.json"):
        with open(anno_file) as f:
            data = json.load(f)

        cat_lookup = {cat["id"]: cat["name"] for cat in data["categories"]}
        image_lookup = {img["id"]: img for img in data["images"]}

        for ann in data["annotations"]:
            label = cat_lookup[ann["category_id"]]
            if label in SKIP_CLASSES:
                continue
            if "keypoints" not in ann or len(ann["keypoints"]) == 0:
                continue

            img_info = image_lookup[ann["image_id"]]
            image_path = Path(image_dir) / img_info["file_name"]

            if not image_path.exists():
                print(f"Image not found: {image_path}")
                continue

            # Manual keypoints
            kp_manual = np.array(ann["keypoints"]).reshape(-1, 3)
            if len(kp_manual) < 17:
                pad = np.zeros((17 - len(kp_manual), 3))
                kp_manual = np.vstack([kp_manual, pad])
            xy_manual = kp_manual[:, :2]

            # COCO bbox
            coco_bbox = [float(v) for v in ann["bbox"]]

            # Run YOLOv8
            results = yolo(str(image_path), verbose=False)
            if not results[0].keypoints or len(results[0].keypoints.xy) == 0:
                print(f"No detection: {image_path.name}")
                continue

            # Find best matching detection via IoU
            boxes = results[0].boxes.xywh.cpu().numpy()
            best_iou = 0
            best_idx = -1

            for i, box in enumerate(boxes):
                yolo_bbox = [box[0] - box[2]/2, box[1] - box[3]/2, box[2], box[3]]
                iou = bbox_iou(coco_bbox, yolo_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx == -1 or best_iou < iou_threshold:
                print(f"No IoU match for {image_path.name} (best IoU: {best_iou:.2f})")
                continue

            # YOLOv8 keypoints for matched detection
            xy_yolo = results[0].keypoints.xy[best_idx].cpu().numpy()
            if len(xy_yolo) < 17:
                pad = np.zeros((17 - len(xy_yolo), 2))
                xy_yolo = np.vstack([xy_yolo, pad])

            # Normalize both using COCO bbox
            x_min, y_min, w, h = coco_bbox
            xy_yolo_norm = normalize_keypoints(xy_yolo, (x_min, y_min, w, h))
            xy_manual_norm = normalize_keypoints(xy_manual, (x_min, y_min, w, h))

            X_yolo.append(xy_yolo_norm)
            X_manual.append(xy_manual_norm)
            y.append(label)

            print(f"Matched: {image_path.name} | IoU: {best_iou:.2f} | {label}")

    return np.array(X_yolo), np.array(X_manual), np.array(y)

X_yolo, X_manual, y = collect_pairs(
    anno_dir="C:/Users/admin/Desktop/Dataset/",
    image_dir="C:/Users/admin/Desktop/Dataset/pictures/"
)

np.save("X_yolo.npy", X_yolo)
np.save("X_manual.npy", X_manual)
print(f"\nPairs collected: {len(y)}")
print(f"Failed matches will need IoU threshold lowered or are genuinely undetectable")
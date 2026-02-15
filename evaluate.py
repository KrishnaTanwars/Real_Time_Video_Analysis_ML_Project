import argparse
import csv
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2

from app import PERSON_LABELS, VEHICLE_LABELS, detect_yolo, yolo_params_for


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate detector performance on annotated images.")
    parser.add_argument("--mode", choices=["object", "human", "vehicle"], required=True)
    parser.add_argument("--annotations", required=True, help="Path to annotations JSON")
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--profile", default="balanced", help="Model profile: fast|balanced|accurate")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def load_annotations(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "images" in payload:
        return payload["images"]
    if isinstance(payload, list):
        return payload

    raise ValueError("Annotation format must be a list or a dict with 'images' key")


def allowed_labels(mode: str):
    if mode == "human":
        return set(PERSON_LABELS)
    if mode == "vehicle":
        return set(VEHICLE_LABELS)
    return None


def match_predictions(gt_items, pred_items, iou_threshold: float):
    matches = []
    used_gt = set()
    used_pred = set()

    candidates = []
    for gt_idx, gt in enumerate(gt_items):
        for pred_idx, pred in enumerate(pred_items):
            if gt.get("label") != pred.get("label"):
                continue
            iou = compute_iou(gt.get("bbox", [0, 0, 0, 0]), pred.get("box", [0, 0, 0, 0]))
            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx))

    for iou, gt_idx, pred_idx in sorted(candidates, key=lambda row: row[0], reverse=True):
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    return matches, used_gt, used_pred


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)

    mode_labels = allowed_labels(args.mode)
    params = yolo_params_for(args.mode, args.profile)
    base_conf = min(0.20, params["conf_threshold"])

    records = load_annotations(args.annotations)

    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    confusion = defaultdict(int)
    threshold_points = []

    total_images = 0
    total_latency_ms = 0.0

    all_image_results = []

    for item in records:
        file_name = item.get("file") or item.get("image") or item.get("filename")
        if not file_name:
            continue

        image_path = os.path.join(args.images_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        gt_items = item.get("annotations", [])
        if mode_labels is not None:
            gt_items = [g for g in gt_items if g.get("label") in mode_labels]

        started = time.perf_counter()
        predictions = detect_yolo(
            image,
            allowed_labels=mode_labels,
            conf_threshold=base_conf,
            nms_threshold=params["nms_threshold"],
            input_size=params["input_size"],
            min_area_ratio=params["min_area_ratio"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        total_latency_ms += elapsed_ms
        total_images += 1

        all_image_results.append({
            "gt": gt_items,
            "pred": predictions,
        })

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    final_threshold = params["conf_threshold"]

    for threshold in thresholds:
        tp_sum = 0
        fp_sum = 0
        fn_sum = 0

        for row in all_image_results:
            gt_items = row["gt"]
            predictions = [p for p in row["pred"] if p.get("confidence", 0.0) >= threshold]

            matches, used_gt, used_pred = match_predictions(gt_items, predictions, args.iou_threshold)
            tp = len(matches)
            fp = len(predictions) - len(used_pred)
            fn = len(gt_items) - len(used_gt)

            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

        precision = safe_div(tp_sum, tp_sum + fp_sum)
        recall = safe_div(tp_sum, tp_sum + fn_sum)
        threshold_points.append({
            "threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })

    for row in all_image_results:
        gt_items = row["gt"]
        predictions = [p for p in row["pred"] if p.get("confidence", 0.0) >= final_threshold]

        matches, used_gt, used_pred = match_predictions(gt_items, predictions, args.iou_threshold)

        for gt_idx, pred_idx, _ in matches:
            gt_label = gt_items[gt_idx].get("label", "unknown")
            pred_label = predictions[pred_idx].get("label", "unknown")
            per_label[gt_label]["tp"] += 1
            confusion[(gt_label, pred_label)] += 1

        for pred_idx, pred in enumerate(predictions):
            if pred_idx in used_pred:
                continue
            pred_label = pred.get("label", "unknown")
            per_label[pred_label]["fp"] += 1
            confusion[("background", pred_label)] += 1

        for gt_idx, gt in enumerate(gt_items):
            if gt_idx in used_gt:
                continue
            gt_label = gt.get("label", "unknown")
            per_label[gt_label]["fn"] += 1
            confusion[(gt_label, "background")] += 1

    tp_total = sum(row["tp"] for row in per_label.values())
    fp_total = sum(row["fp"] for row in per_label.values())
    fn_total = sum(row["fn"] for row in per_label.values())

    precision_total = safe_div(tp_total, tp_total + fp_total)
    recall_total = safe_div(tp_total, tp_total + fn_total)
    f1_total = safe_div(2 * precision_total * recall_total, precision_total + recall_total)

    per_label_metrics = {}
    all_labels = sorted(per_label.keys())

    for label in all_labels:
        row = per_label[label]
        precision = safe_div(row["tp"], row["tp"] + row["fp"])
        recall = safe_div(row["tp"], row["tp"] + row["fn"])
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_label_metrics[label] = {
            "tp": row["tp"],
            "fp": row["fp"],
            "fn": row["fn"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    metrics = {
        "mode": args.mode,
        "profile": params["profile"],
        "input_size": params["input_size"],
        "confidence_threshold": final_threshold,
        "iou_threshold": args.iou_threshold,
        "images_evaluated": total_images,
        "avg_latency_ms": round(safe_div(total_latency_ms, total_images), 2),
        "overall": {
            "precision": round(precision_total, 4),
            "recall": round(recall_total, 4),
            "f1": round(f1_total, 4),
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
        },
        "per_label": per_label_metrics,
        "threshold_curve": threshold_points,
    }

    metrics_path = os.path.join(args.output_dir, f"metrics_{args.mode}.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    confusion_labels = sorted(set([gt for gt, _ in confusion.keys()] + [pred for _, pred in confusion.keys()]))
    if "background" not in confusion_labels:
        confusion_labels.append("background")

    confusion_path = os.path.join(args.output_dir, f"confusion_{args.mode}.csv")
    with open(confusion_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gt\\pred"] + confusion_labels)
        for gt_label in confusion_labels:
            row = [gt_label]
            for pred_label in confusion_labels:
                row.append(confusion.get((gt_label, pred_label), 0))
            writer.writerow(row)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved confusion matrix: {confusion_path}")


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

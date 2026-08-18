"""Evaluator: so sánh ground truth với predictions của TechGAR.

Tính:
  - Precision, Recall, F1 cho occupied và free
  - False-free rate (có xe nhưng báo trống)
  - False-occupied rate (trống nhưng báo có xe)
  - Transition delay (thời gian từ sự kiện thật đến khi hệ thống nhận)
  - Flicker rate (số lần đổi trạng thái sai trong 1 phút)

Usage:
  python evaluate.py experiment_test/output/two_camera_28
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_ground_truth(gt_path: Path) -> Dict[Tuple[str, str], List[dict]]:
    """Load ground truth CSV. Returns {(camera_id, slot_id): [intervals]}."""
    intervals = defaultdict(list)
    with gt_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam = f"cam{row['camera_id']}"
            slot = row["slot_id"]
            intervals[(cam, slot)].append({
                "start": int(row["start_frame"]),
                "end": int(row["end_frame"]),
                "occupied": row["occupied"].strip().lower() == "true",
                "vehicle_id": row.get("vehicle_id"),
            })
    return dict(intervals)


def load_predictions(pred_path: Path) -> Dict[int, dict]:
    """Load predictions JSONL. Returns {frame_idx: prediction_data}."""
    predictions = {}
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            predictions[data["frame_idx"]] = data
    return predictions


def gt_status_at_frame(gt_intervals: Dict, cam: str, slot: str, frame: int) -> Optional[Tuple[bool, Optional[str]]]:
    """Return (occupied, vehicle_id) at a specific frame, or None if not labeled."""
    key = (cam, slot)
    if key not in gt_intervals:
        return None
    for iv in gt_intervals[key]:
        if iv["start"] <= frame <= iv["end"]:
            return iv["occupied"], str(iv["vehicle_id"]) if iv["vehicle_id"] else None
    return None


def evaluate(session_dir: Path, fps: float = 30.0):
    gt_path = session_dir / "ground_truth_slots.csv"
    pred_path = session_dir / "predictions.jsonl"

    if not gt_path.exists():
        print(f"❌ Không tìm thấy: {gt_path}")
        sys.exit(1)
    if not pred_path.exists():
        print(f"❌ Không tìm thấy: {pred_path}")
        sys.exit(1)

    gt = load_ground_truth(gt_path)
    predictions = load_predictions(pred_path)

    if not predictions:
        print("❌ predictions.jsonl rỗng!")
        sys.exit(1)

    # Lấy danh sách tất cả (cam, slot) từ GT
    labeled_keys = set(gt.keys())
    print(f"\n📋 Ground truth: {len(labeled_keys)} cặp (cam, slot) được gán nhãn")
    for key in sorted(labeled_keys):
        intervals = gt[key]
        print(f"   {key[0]}/{key[1]}: {len(intervals)} khoảng, "
              f"frame {intervals[0]['start']}–{intervals[-1]['end']}")

    # Counters
    tp_occ = 0  # True positive: GT=occupied, Pred=occupied
    fp_occ = 0  # False positive: GT=free, Pred=occupied
    fn_occ = 0  # False negative: GT=occupied, Pred=free
    tn_occ = 0  # True negative: GT=free, Pred=free

    total_compared = 0
    frame_indices = sorted(predictions.keys())
    print(f"\n📊 Predictions: {len(frame_indices)} frame ({frame_indices[0]}–{frame_indices[-1]})")

    # Track transitions for delay measurement
    # For each (cam, slot), find GT transitions and measure when prediction catches up
    transition_delays = []

    # Track flicker
    flicker_count = 0
    prev_pred_status = {}  # (cam, slot) -> last predicted occupied

    id_total = 0
    id_match = 0
    id_mismatch = 0
    id_errors = defaultdict(int)  # GT ID -> predicted ID frequency

    for frame_idx in frame_indices:
        pred = predictions[frame_idx]
        cameras = pred.get("cameras", {})

        for cam, cam_data in cameras.items():
            slots = cam_data.get("parking_slots", {})
            for slot_id, slot_data in slots.items():
                gt_result = gt_status_at_frame(gt, cam, slot_id, frame_idx)
                if gt_result is None:
                    continue  # Không có nhãn cho slot này

                gt_occ, gt_id = gt_result
                pred_occ = slot_data.get("occupied", False)
                pred_id = str(slot_data.get("vehicle_id")) if slot_data.get("vehicle_id") is not None else None
                total_compared += 1

                if gt_occ and pred_occ:
                    tp_occ += 1
                    if gt_id and gt_id != "None" and gt_id != "null":
                        id_total += 1
                        if pred_id == gt_id:
                            id_match += 1
                        else:
                            id_mismatch += 1
                            id_errors[f"{cam}/{slot_id} GT:{gt_id} Pred:{pred_id}"] += 1
                elif not gt_occ and pred_occ:
                    fp_occ += 1
                elif gt_occ and not pred_occ:
                    fn_occ += 1
                else:
                    tn_occ += 1

                # Flicker detection
                key = (cam, slot_id)
                if key in prev_pred_status:
                    if prev_pred_status[key] != pred_occ and pred_occ != gt_occ:
                        flicker_count += 1
                prev_pred_status[key] = pred_occ

    # === Calculate metrics ===
    print(f"\n{'='*60}")
    print(f"  📊 KẾT QUẢ ĐÁNH GIÁ")
    print(f"{'='*60}")
    print(f"\n  Tổng slot-frame so sánh: {total_compared:,}")

    # Confusion matrix
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │        Confusion Matrix              │")
    print(f"  │                  Prediction           │")
    print(f"  │                Occupied   Free        │")
    print(f"  │  GT Occupied   {tp_occ:>6}   {fn_occ:>6}       │")
    print(f"  │  GT Free       {fp_occ:>6}   {tn_occ:>6}       │")
    print(f"  └─────────────────────────────────────┘")

    # Occupied metrics
    occ_precision = tp_occ / max(1, tp_occ + fp_occ)
    occ_recall = tp_occ / max(1, tp_occ + fn_occ)
    occ_f1 = 2 * occ_precision * occ_recall / max(1e-9, occ_precision + occ_recall)

    # Free metrics
    free_precision = tn_occ / max(1, tn_occ + fn_occ)
    free_recall = tn_occ / max(1, tn_occ + fp_occ)
    free_f1 = 2 * free_precision * free_recall / max(1e-9, free_precision + free_recall)

    # False rates
    false_free = fn_occ / max(1, tp_occ + fn_occ)  # Có xe nhưng báo trống
    false_occupied = fp_occ / max(1, tn_occ + fp_occ)  # Trống nhưng báo có xe

    # Balanced accuracy
    balanced_acc = (occ_recall + free_recall) / 2

    print(f"\n  {'Metric':<28} {'Occupied':>10} {'Free':>10}")
    print(f"  {'─'*50}")
    print(f"  {'Precision':<28} {occ_precision:>10.4f} {free_precision:>10.4f}")
    print(f"  {'Recall':<28} {occ_recall:>10.4f} {free_recall:>10.4f}")
    print(f"  {'F1':<28} {occ_f1:>10.4f} {free_f1:>10.4f}")
    print(f"\n  {'False-free rate':<28} {false_free:>10.4f}  (có xe nhưng báo trống)")
    print(f"  {'False-occupied rate':<28} {false_occupied:>10.4f}  (trống nhưng báo có xe)")
    print(f"  {'Balanced Accuracy':<28} {balanced_acc:>10.4f}")

    # ID metrics
    print(f"\n  {'─'*50}")
    print(f"  🆔 ID MATCHING ACCURACY")
    print(f"  {'─'*50}")
    if id_total > 0:
        id_acc = id_match / id_total
        print(f"  {'Total ID frames compared':<28} {id_total:>10}")
        print(f"  {'ID Matches (Correct)':<28} {id_match:>10}")
        print(f"  {'ID Mismatches (Wrong)':<28} {id_mismatch:>10}")
        print(f"  {'ID Accuracy':<28} {id_acc:>10.4f}")
        
        if id_mismatch > 0:
            print(f"\n  Top ID Mismatches:")
            for err, count in sorted(id_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    - {err}: {count} frames")
    else:
        print("  Không có nhãn ID trong ground truth.")

    # Flicker
    total_seconds = (frame_indices[-1] - frame_indices[0]) / fps
    total_minutes = max(0.01, total_seconds / 60)
    flicker_per_min = flicker_count / total_minutes
    print(f"\n  {'Flicker count':<28} {flicker_count:>10}")
    print(f"  {'Flicker/min':<28} {flicker_per_min:>10.1f}")

    # === Transition delay ===
    print(f"\n  {'─'*50}")
    print(f"  📐 TRANSITION DELAY")
    print(f"  {'─'*50}")

    for (cam, slot_id), intervals in sorted(gt.items()):
        for i in range(len(intervals) - 1):
            curr = intervals[i]
            nxt = intervals[i + 1]
            if curr["occupied"] != nxt["occupied"]:
                # Transition at nxt["start"]
                gt_frame = nxt["start"]
                expected = nxt["occupied"]
                # Find first prediction frame that matches
                found_frame = None
                search_end = min(gt_frame + int(fps * 10), frame_indices[-1])
                for f in range(gt_frame, search_end + 1):
                    if f in predictions:
                        pred = predictions[f]
                        cam_data = pred.get("cameras", {}).get(cam, {})
                        slot_data = cam_data.get("parking_slots", {}).get(slot_id)
                        if slot_data and slot_data.get("occupied") == expected:
                            found_frame = f
                            break

                if found_frame is not None:
                    delay_frames = found_frame - gt_frame
                    delay_ms = delay_frames / fps * 1000
                    transition_delays.append(delay_ms)
                    status_str = "occupied" if expected else "free"
                    print(f"  {cam}/{slot_id} → {status_str} @ frame {gt_frame}: "
                          f"delay = {delay_frames} frames ({delay_ms:.0f} ms)")
                else:
                    transition_delays.append(float("inf"))
                    status_str = "occupied" if expected else "free"
                    print(f"  {cam}/{slot_id} → {status_str} @ frame {gt_frame}: "
                          f"⚠️ KHÔNG NHẬN ĐƯỢC trong 10s")

    if transition_delays:
        valid_delays = [d for d in transition_delays if d != float("inf")]
        if valid_delays:
            import statistics
            avg_delay = statistics.mean(valid_delays)
            p50 = statistics.median(valid_delays)
            p95 = sorted(valid_delays)[int(len(valid_delays) * 0.95)] if len(valid_delays) >= 2 else valid_delays[0]
            print(f"\n  {'Avg delay':<28} {avg_delay:>10.0f} ms")
            print(f"  {'p50 delay':<28} {p50:>10.0f} ms")
            print(f"  {'p95 delay':<28} {p95:>10.0f} ms")
        missed = sum(1 for d in transition_delays if d == float("inf"))
        if missed:
            print(f"  {'Missed transitions':<28} {missed:>10}")

    # === Save results ===
    results = {
        "total_slot_frames": total_compared,
        "tp": tp_occ, "fp": fp_occ, "fn": fn_occ, "tn": tn_occ,
        "occupied_precision": round(occ_precision, 4),
        "occupied_recall": round(occ_recall, 4),
        "occupied_f1": round(occ_f1, 4),
        "free_precision": round(free_precision, 4),
        "free_recall": round(free_recall, 4),
        "free_f1": round(free_f1, 4),
        "false_free_rate": round(false_free, 4),
        "false_occupied_rate": round(false_occupied, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "id_accuracy": round(id_match / max(1, id_total), 4) if id_total > 0 else None,
        "flicker_count": flicker_count,
        "flicker_per_min": round(flicker_per_min, 1),
        "transition_delays_ms": [round(d, 0) for d in transition_delays if d != float("inf")],
    }

    out_path = session_dir / "evaluation_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Kết quả đã lưu: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TechGAR predictions vs ground truth")
    parser.add_argument("session_dir", help="Path to session directory containing predictions.jsonl and ground_truth_slots.csv")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (default: 30)")
    args = parser.parse_args()
    evaluate(Path(args.session_dir), fps=args.fps)

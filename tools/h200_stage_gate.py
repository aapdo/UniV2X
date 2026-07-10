#!/usr/bin/env python
"""Reject UniMM-V2X checkpoints whose validation outputs have collapsed."""

import argparse
import json
import math
import os
import pickle

import torch


POLICIES = {
    "sub_inf_stg1": {"track_nonempty_ratio": 0.05},
    "sub_vehicle_stg1": {"track_nonempty_ratio": 0.05},
    "coop_stg1": {
        "track_nonempty_ratio": 0.10,
        "lanes_iou": 0.05,
        "crossing_iou": 0.05,
        "mAP": 0.05,
        "amota": 0.05,
    },
    "coop_stg2": {
        "track_nonempty_ratio": 0.10,
        "lanes_iou": 0.05,
        "crossing_iou": 0.05,
        "mAP": 0.05,
        "amota": 0.05,
        "max_l2_3s": 3.5,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(POLICIES))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--expected-epoch", type=int)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-samples", type=int, default=100)
    return parser.parse_args()


def scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def box_count(boxes):
    if boxes is None:
        return 0
    try:
        return len(boxes)
    except TypeError:
        tensor = getattr(boxes, "tensor", None)
        return 0 if tensor is None else len(tensor)


def aggregate_map_iou(samples, name):
    intersection = 0.0
    union = 0.0
    for sample in samples:
        values = sample.get("ret_iou", {})
        intersection += scalar(values.get(f"{name}_intersection", 0.0))
        union += scalar(values.get(f"{name}_union", 0.0))
    return intersection / union if union > 0 else float("nan")


def find_metric(metrics, suffix):
    matches = []

    def visit(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}/{key}" if path else key)
        elif path == suffix or path.endswith(f"/{suffix}"):
            matches.append(scalar(value))

    visit(metrics)
    if not matches:
        return None
    return matches[0]


def load_checkpoint_summary(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "bytes": os.path.getsize(path),
        "epoch": int(checkpoint.get("epoch", 0) or 0),
        "best_loss": scalar(checkpoint.get("best_loss", float("nan"))),
    }


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def main():
    args = parse_args()
    checkpoint = load_checkpoint_summary(args.checkpoint)
    outputs = load_pickle(args.results)
    samples = outputs.get("bbox_results", outputs)
    planning = outputs.get("planning_results_computed", {})
    metrics = {}
    if args.metrics and os.path.isfile(args.metrics):
        with open(args.metrics, encoding="utf-8") as handle:
            metrics = json.load(handle)

    sample_count = len(samples)
    track_counts = [box_count(sample.get("boxes_3d")) for sample in samples]
    observed = {
        "sample_count": sample_count,
        "track_box_count": sum(track_counts),
        "track_nonempty_frames": sum(count > 0 for count in track_counts),
        "track_nonempty_ratio": (
            sum(count > 0 for count in track_counts) / sample_count
            if sample_count else 0.0
        ),
        "lanes_iou": aggregate_map_iou(samples, "lanes"),
        "crossing_iou": aggregate_map_iou(samples, "crossing"),
        "drivable_iou": aggregate_map_iou(samples, "drivable"),
        "mAP": find_metric(metrics, "mAP"),
        "amota": find_metric(metrics, "amota"),
    }
    l2 = planning.get("L2_valid", planning.get("L2"))
    if l2 is not None and len(l2) >= 6:
        observed["l2_3s"] = scalar(l2[5])

    failures = []
    if sample_count < args.min_samples:
        failures.append(f"sample_count={sample_count} < {args.min_samples}")
    if args.expected_epoch is not None and checkpoint["epoch"] != args.expected_epoch:
        failures.append(
            f"checkpoint epoch={checkpoint['epoch']} != {args.expected_epoch}"
        )
    if not math.isfinite(checkpoint["best_loss"]):
        failures.append("checkpoint best_loss is not finite")

    for key, minimum in POLICIES[args.stage].items():
        if key == "max_l2_3s":
            value = observed.get("l2_3s")
            if value is None or not math.isfinite(value) or value > minimum:
                failures.append(f"l2_3s={value} > {minimum}")
            continue
        value = observed.get(key)
        if value is None:
            failures.append(f"required metric {key} is missing")
        elif not math.isfinite(value) or value < minimum:
            failures.append(f"{key}={value} < {minimum}")

    report = {
        "stage": args.stage,
        "passed": not failures,
        "checkpoint": checkpoint,
        "observed": observed,
        "policy": POLICIES[args.stage],
        "failures": failures,
    }
    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

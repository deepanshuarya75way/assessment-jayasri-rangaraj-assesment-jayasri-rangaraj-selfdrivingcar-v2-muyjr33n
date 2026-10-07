import collections
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml


def load_yolo_dataset_stats(labels_dir: str, data_config: str | None = None) -> Dict:
    """
    Scan a YOLO-format labels directory and return basic statistics:
    - class_counts: number of instances per class id
    - num_images
    - classes: optional mapping from id -> name (from yaml if provided)
    """
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    class_counts: Dict[int, int] = collections.Counter()
    num_images = 0

    for txt_file in labels_path.rglob("*.txt"):
        num_images += 1
        with txt_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                class_counts[cls_id] += 1

    classes = None
    if data_config is not None and Path(data_config).exists():
        with open(data_config, "r") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names")
        if isinstance(names, dict):
            classes = {int(k): v for k, v in names.items()}
        elif isinstance(names, list):
            classes = {i: n for i, n in enumerate(names)}

    return {
        "class_counts": dict(class_counts),
        "num_images": num_images,
        "classes": classes,
    }


def compute_class_weights(class_counts: Dict[int, int], smoothing: float = 1.0) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights with optional smoothing.
    These can be used to:
    - reweight the loss function
    - guide a custom sampler
    """
    if not class_counts:
        return {}

    counts = np.array(list(class_counts.values()), dtype=float) + smoothing
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum()

    return {cls_id: float(w) for cls_id, w in zip(class_counts.keys(), weights)}


def summarize_distribution(stats: Dict) -> str:
    """Return a short human-readable summary of class distribution."""
    class_counts = stats.get("class_counts", {})
    classes = stats.get("classes") or {}
    num_images = stats.get("num_images", 0)

    total_instances = sum(class_counts.values())
    if total_instances == 0:
        return "No labeled objects found in the dataset."

    sorted_items: List[Tuple[int, int]] = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    lines = [f"Images: {num_images}", f"Total objects: {total_instances}", "Class distribution (top 10):"]
    for cls_id, count in sorted_items[:10]:
        name = classes.get(cls_id, f"class_{cls_id}")
        frac = count / total_instances
        lines.append(f"- {name} (id={cls_id}): {count} ({frac:.1%})")

    if len(sorted_items) > 10:
        lines.append(f"... and {len(sorted_items) - 10} more classes.")

    return "\n".join(lines)


import argparse
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from training.data_utils import load_yolo_dataset_stats, compute_class_weights, summarize_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO model for self-driving perception.")
    parser.add_argument("--data", type=str, required=True, help="Path to YOLO data config YAML.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base YOLO model weights.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Directory to save training runs (Ultralytics default is 'runs/train').",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="selfdriving-yolo",
        help="Name of this training run.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Enable basic class-balancing via loss weights.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Optional explicit path to labels directory for stats (otherwise inferred from data yaml).",
    )
    return parser.parse_args()


def infer_labels_dir_from_yaml(data_yaml: str) -> Optional[str]:
    """
    Very light heuristic to guess labels dir from a YOLO data yaml.
    We don't parse here deeply; for detailed stats use --labels_dir explicitly.
    """
    # For now, rely on user to pass --labels_dir when they want detailed stats.
    return None


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml}")

    print(f"[INFO] Using data config: {data_yaml}")
    print(f"[INFO] Loading base model: {args.model}")
    model = YOLO(args.model)

    class_weights = None
    if args.balance:
        labels_dir = args.labels_dir or infer_labels_dir_from_yaml(str(data_yaml))
        if labels_dir:
            print(f"[INFO] Computing class distribution from labels: {labels_dir}")
            stats = load_yolo_dataset_stats(labels_dir, data_config=str(data_yaml))
            print("[INFO] Dataset stats:")
            print(summarize_distribution(stats))
            class_weights = compute_class_weights(stats["class_counts"])
            print("[INFO] Computed class weights (inverse-frequency, normalized):")
            for cid, w in class_weights.items():
                print(f"  class {cid}: {w:.4f}")
        else:
            print("[WARN] --balance enabled but labels_dir could not be inferred. "
                  "Pass --labels_dir explicitly if you want class-weighted training.")

    # Ultralytics YOLO allows passing 'data' and standard training args.
    # Class weights can be provided via the 'cls' in loss, but that is more advanced;
    # For hackathon purposes, we log them and you can plug them into custom loss if needed.
    # Here we focus on a clean training entrypoint.

    print("[INFO] Starting training...")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )
    print("[INFO] Training finished.")
    print("[INFO] Best weights are typically saved as 'best.pt' under runs/train/<name>/")


if __name__ == "__main__":
    main()


import argparse
from pathlib import Path

from .data import load_csv
from .model import train_eval_save

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.gz"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(DEFAULT_DATA_PATH),
        help="Path to training CSV",
    )
    ap.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Saved model path",
    )
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    ap.add_argument("--random-state", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    model_path = Path(args.model_path)

    df = load_csv(csv_path, nrows=args.nrows)

    if args.label not in df.columns:
        raise ValueError(f"Label column '{args.label}' not found in {csv_path}")

    metrics = train_eval_save(
        df=df,
        model_path=model_path,
        random_state=args.random_state,
        test_size=args.test_size,
    )

    print(f"Saved model to: {model_path}")
    print(f"log_loss={metrics['log_loss']:.6f}")
    if "auc" in metrics:
        print(f"auc={metrics['auc']:.6f}")


if __name__ == "__main__":
    main()

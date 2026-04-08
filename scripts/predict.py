#!/usr/bin/env python3
"""
Predict the Nutriscore for a single food product.

The script trains a KNN regressor on the cleaned dataset and then
applies it to the nutritional values supplied via CLI arguments.
The predicted numeric score is mapped to an A–E letter grade.

Nutriscore grade thresholds (per 100 g):
  A: ≤ −1   B: 0–2   C: 3–10   D: 11–18   E: ≥ 19

Usage example
-------------
    python scripts/predict.py \\
        --energy 1800 \\
        --proteins 6 --fat 22 --carbohydrates 55 \\
        --sugars 28 --salt 0.8 --saturated-fat 10 \\
        --fiber 1.5 --additives 4 --pnns 2 --nova 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from nutriscore.config import DATA_CLEAN, KNN_BEST_K
from nutriscore.features.engineering import prepare_xy, scale
from nutriscore.models.train import train_knn

# Nutriscore letter-grade cut-offs (official French algorithm thresholds)
GRADE_THRESHOLDS = [
    ("A", -float("inf"), -1),
    ("B", 0, 2),
    ("C", 3, 10),
    ("D", 11, 18),
    ("E", 19, float("inf")),
]


def score_to_grade(score: float) -> str:
    for grade, low, high in GRADE_THRESHOLDS:
        if low <= score <= high:
            return grade
    return "E"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict Nutriscore for a food product",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clean", type=Path, default=DATA_CLEAN, help="Cleaned CSV used for training")
    p.add_argument("--energy", type=float, required=True, help="Energy (kJ per 100 g)")
    p.add_argument("--proteins", type=float, required=True, help="Proteins (g per 100 g)")
    p.add_argument("--fat", type=float, required=True, help="Total fat (g per 100 g)")
    p.add_argument("--carbohydrates", type=float, required=True, help="Carbohydrates (g per 100 g)")
    p.add_argument("--sugars", type=float, required=True, help="Sugars (g per 100 g)")
    p.add_argument("--salt", type=float, required=True, help="Salt (g per 100 g)")
    p.add_argument(
        "--saturated-fat",
        type=float,
        required=True,
        dest="saturated_fat",
        help="Saturated fat (g per 100 g)",
    )
    p.add_argument("--fiber", type=float, default=0.0, help="Dietary fiber (g per 100 g)")
    p.add_argument("--calcium", type=float, default=0.0, help="Calcium (g per 100 g)")
    p.add_argument("--cholesterol", type=float, default=0.0, help="Cholesterol (g per 100 g)")
    p.add_argument(
        "--trans-fat", type=float, default=0.0, dest="trans_fat", help="Trans fat (g per 100 g)"
    )
    p.add_argument("--iron", type=float, default=0.0, help="Iron (g per 100 g)")
    p.add_argument("--additives", type=int, default=0, help="Number of food additives")
    p.add_argument(
        "--pnns", type=int, default=1, choices=range(1, 10), help="PNNS group code (1–9)"
    )
    p.add_argument(
        "--nova", type=int, default=1, choices=[1, 2, 3, 4], help="Nova processing group (1–4)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.clean.exists():
        print(f"Error: cleaned dataset not found at {args.clean}.")
        print("Run `python scripts/run_pipeline.py` first to generate it.")
        sys.exit(1)

    # Train model on full clean dataset
    df = pd.read_csv(args.clean)
    X, y = prepare_xy(df)
    X_scaled, scaler = scale(X)
    knn = train_knn(X_scaled, y, k=KNN_BEST_K)

    # Build the sample vector (order must match MODEL_FEATURES)
    sample = np.array(
        [
            [
                args.pnns,
                args.nova,
                args.energy,
                args.proteins,
                args.fat,
                args.carbohydrates,
                args.sugars,
                args.salt,
                args.saturated_fat,
                args.fiber,
                args.calcium,
                args.cholesterol,
                args.trans_fat,
                args.iron,
                args.additives,
            ]
        ]
    )
    sample_scaled = scaler.transform(sample)
    score = float(knn.predict(sample_scaled)[0])
    grade = score_to_grade(score)

    print(f"\n  Predicted Nutriscore : {score:+.1f}")
    print(f"  Letter Grade        : {grade}")
    print("\n  [A] [B] [C] [D] [E]")
    print(
        f"   {'↑' if grade == 'A' else ' '}   {'↑' if grade == 'B' else ' '}   {'↑' if grade == 'C' else ' '}   {'↑' if grade == 'D' else ' '}   {'↑' if grade == 'E' else ' '}"
    )
    print()


if __name__ == "__main__":
    main()

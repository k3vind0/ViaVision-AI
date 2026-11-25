"""Entrena baseline y candidatos ML con partición por tramo sin fugas."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import ROOT
from src.data.synthetic import generate_synthetic_tramos
from src.ml.train import train_priority_models


def synthetic():
    return generate_synthetic_tramos()


def train(frame, demo, group):
    return train_priority_models(frame, demo=demo, group=group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--grupo", default="id_tramo")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv) if args.csv else synthetic()
    if not args.csv:
        frame.to_csv(ROOT / "data_sample" / "tramos_sinteticos.csv", index=False)
    print(json.dumps(train(frame, args.csv is None, args.grupo), ensure_ascii=False, indent=2))

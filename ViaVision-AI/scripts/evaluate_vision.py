"""Evalúa detecciones contra cajas verificadas de calles y recorridos nuevos."""
import argparse
import json
from pathlib import Path

from src.vision.evaluator import evaluate_manifest


def evaluate(manifest_path: Path, output_json: Path | None = None, figure_path: Path | None = None):
    return evaluate_manifest(manifest_path, output_json=output_json, figure_path=figure_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--figure", type=Path, default=None)
    args = p.parse_args()
    print(json.dumps(evaluate(args.manifest, args.output, args.figure), ensure_ascii=False, indent=2))

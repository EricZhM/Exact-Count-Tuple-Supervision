"""Unified command-line entry point for NTMP."""
from __future__ import annotations

import argparse
from pathlib import Path

from ntmp.data import prepare_manifest
from ntmp.training import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train NTMP from exact-count tuple supervision")
    parser.add_argument("--dataset", required=True, choices=["mnist", "fashionmnist", "cifar10", "svhn"])
    parser.add_argument("--data-dir", required=True, help="torchvision dataset root")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n", type=int, default=5, help="tuple length")
    parser.add_argument("--m", type=int, default=2, help="positive instances per tuple")
    parser.add_argument("--pi", type=float, default=0.8, help="positive prior of D_U")
    parser.add_argument("--correction", choices=["ure", "abs", "relu"], default="abs")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--tuple-count", type=int, default=1000)
    parser.add_argument("--reference-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256, help="complete tuples per update")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--download", action="store_true", help="allow torchvision to download data")
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    manifest_path = prepare_manifest(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=output_dir,
        seed=args.seed,
        n=args.n,
        m=args.m,
        pi=args.pi,
        tuple_count=args.tuple_count,
        reference_size=args.reference_size,
        download=args.download,
    )
    run_dir = run_training(
        manifest_path=manifest_path,
        output_dir=output_dir,
        correction=args.correction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        max_batches=args.max_batches_per_epoch,
    )
    print(f"Saved run to {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sweep roccap captures for the CCL all-reduce example.

Edit the parameter arrays below, then run from this directory:
    python sweep_roccap.py
    python sweep_roccap.py --dry-run

Each run executes torchrun + roccap_wrapper, then renames generated .cap and
.json files to unique names encoding the sweep parameters, e.g.:
    persistent_all_reduce_two_shot_8192x4096_128x128_96sms_1stage_fp32_32warps_8nproc_rank0.cap
    persistent_all_reduce_atomic_8192x4096_128x128_96sms_1stage_fp32_32warps_8nproc_rank0.cap

Notes:
  - Skip --validate in FFM (cross-rank RMA is not functional in simulation).
  - Ring variant requires block_size_n divisible by world_size (nproc_per_node).
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Sweep parameter arrays — edit these to define your sweep
# ---------------------------------------------------------------------------

NPROC_PER_NODE = [4, 8]

M_SIZES = [8192]
N_SIZES = [4096]

DATATYPES = ["fp32"]

# (block_size_m, block_size_n) pairs — each entry is one sweep point
BLOCK_SIZES: list[tuple[int, int]] = [
    (256, 128),
]

COMM_SMS = [96, 128, 144]
NUM_STAGES = [1]
NUM_WARPS = [8]

WAVES_PER_EU = [0]
HEAP_SIZE = [1 << 30]
VALIDATE = [False]

# atomic | ring | two_shot | one_shot | spinlock
ALL_REDUCE_VARIANT = ["two_shot", "one_shot"]

# Minimum .cap file size (MiB) for a capture to count as successful.
MIN_CAP_MB = 4.0

# Optional: pass extra args through to example.py (same for every run)
EXTRA_EXAMPLE_ARGS: list[str] = []


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
ROCCAP_WRAPPER = EXAMPLE_DIR / "../../scripts/roccap_wrapper.py"
EXAMPLE_SCRIPT = EXAMPLE_DIR / "example.py"


# Roccap -k filter must match the Triton kernel function name.


def roccap_kernel(all_reduce_variant: str) -> str:
    variant = all_reduce_variant.lower()
    kernel_map = {
        "atomic": "persistent_all_reduce_atomic",
        "ring": "persistent_all_reduce_ring",
        "two_shot": "persistent_all_reduce_two_shot",
        "one_shot": "persistent_all_reduce_one_shot",
        "spinlock": "persistent_all_reduce_spinlock",
    }
    if variant not in kernel_map:
        raise ValueError(f"Unknown all_reduce_variant: {all_reduce_variant}")
    return kernel_map[variant]


@dataclass(frozen=True)
class SweepConfig:
    nproc_per_node: int
    m: int
    n: int
    datatype: str
    block_size_m: int
    block_size_n: int
    comm_sms: int
    num_stages: int
    num_warps: int
    waves_per_eu: int
    heap_size: int
    validate: bool
    all_reduce_variant: str

    @property
    def kernel(self) -> str:
        return roccap_kernel(self.all_reduce_variant)

    @property
    def matrix_label(self) -> str:
        return f"{self.m}x{self.n}"

    def basename(self, rank: int) -> str:
        return (
            f"{self.kernel}_"
            f"{self.matrix_label}_"
            f"{self.block_size_m}x{self.block_size_n}_"
            f"{self.comm_sms}sms_"
            f"{self.num_stages}stage_"
            f"{self.datatype}_"
            f"{self.num_warps}warps_"
            f"{self.nproc_per_node}nproc_rank{rank}"
        )

    def example_args(self) -> list[str]:
        args = [
            "-m",
            str(self.m),
            "-n",
            str(self.n),
            "--heap_size",
            str(self.heap_size),
            "--datatype",
            self.datatype,
            "--block_size_m",
            str(self.block_size_m),
            "--block_size_n",
            str(self.block_size_n),
            "--comm_sms",
            str(self.comm_sms),
            "--num_stages",
            str(self.num_stages),
            "--num_warps",
            str(self.num_warps),
            "--waves_per_eu",
            str(self.waves_per_eu),
            "--all_reduce_variant",
            self.all_reduce_variant,
        ]
        if self.validate:
            args.append("--validate")
        args.extend(EXTRA_EXAMPLE_ARGS)
        return args

    def torchrun_cmd(self) -> list[str]:
        wrapper = ROCCAP_WRAPPER.resolve()
        example = EXAMPLE_SCRIPT.resolve()
        return [
            "torchrun",
            f"--nproc_per_node={self.nproc_per_node}",
            "--standalone",
            str(wrapper),
            "-k",
            self.kernel,
            str(example),
            *self.example_args(),
        ]


def iter_sweep_configs() -> Iterable[SweepConfig]:
    for (
        nproc_per_node,
        m,
        n,
        datatype,
        block_size,
        comm_sms,
        num_stages,
        num_warps,
        waves_per_eu,
        heap_size,
        validate,
        all_reduce_variant,
    ) in itertools.product(
        NPROC_PER_NODE,
        M_SIZES,
        N_SIZES,
        DATATYPES,
        BLOCK_SIZES,
        COMM_SMS,
        NUM_STAGES,
        NUM_WARPS,
        WAVES_PER_EU,
        HEAP_SIZE,
        VALIDATE,
        ALL_REDUCE_VARIANT,
    ):
        block_size_m, block_size_n = block_size
        if all_reduce_variant == "ring" and block_size_n % nproc_per_node != 0:
            continue
        yield SweepConfig(
            nproc_per_node,
            m,
            n,
            datatype,
            block_size_m,
            block_size_n,
            comm_sms,
            num_stages,
            num_warps,
            waves_per_eu,
            heap_size,
            validate,
            all_reduce_variant,
        )


def resolve_cap_path(cfg: SweepConfig, workdir: Path, rank: int) -> Path | None:
    pattern = f"{cfg.kernel}_rank_{rank}*.cap"
    candidates = list(workdir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def capture_passed(cfg: SweepConfig, workdir: Path, min_cap_bytes: int) -> tuple[bool, list[str]]:
    messages: list[str] = []
    all_ok = True

    for rank in range(cfg.nproc_per_node):
        cap_path = resolve_cap_path(cfg, workdir, rank)
        if cap_path is None:
            all_ok = False
            messages.append(f"rank{rank}: missing {cfg.kernel}_rank_{rank}*.cap")
            continue

        size = cap_path.stat().st_size
        size_mb = size / (1024 * 1024)
        if size <= min_cap_bytes:
            all_ok = False
            min_cap_mb = min_cap_bytes / (1024 * 1024)
            messages.append(f"rank{rank}: {cap_path.name} is {size_mb:.2f} MiB (need > {min_cap_mb:g} MiB)")
        else:
            messages.append(f"rank{rank}: {cap_path.name} is {size_mb:.2f} MiB (pass)")

    return all_ok, messages


def rename_outputs(cfg: SweepConfig, workdir: Path, output_dir: Path | None) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    dest_root = output_dir if output_dir is not None else workdir

    for rank in range(cfg.nproc_per_node):
        basename = cfg.basename(rank)
        cap_src = resolve_cap_path(cfg, workdir, rank)

        renames: list[tuple[Path, Path]] = []
        if cap_src is not None:
            renames.append((cap_src, dest_root / f"{basename}.cap"))
        renames.extend(
            [
                (
                    workdir / f"{cfg.kernel}_rank_{rank}_heap_bases.json",
                    dest_root / f"{basename}_heap_bases.json",
                ),
                (
                    workdir / f"iris_rank_{rank}_allocator_views.json",
                    dest_root / f"{basename}_allocator_views.json",
                ),
            ]
        )

        for src, dst in renames:
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            moves.append((src, dst))

    return moves


def run_one(
    cfg: SweepConfig,
    workdir: Path,
    output_dir: Path | None,
    dry_run: bool,
    min_cap_bytes: int,
) -> int:
    cmd = cfg.torchrun_cmd()
    print("\n" + "=" * 80)
    print(" ".join(cmd))
    print("=" * 80)

    if dry_run:
        for rank in range(cfg.nproc_per_node):
            print(f"  -> {cfg.basename(rank)}.cap")
            print(f"  -> {cfg.basename(rank)}_heap_bases.json")
            print(f"  -> {cfg.basename(rank)}_allocator_views.json")
        return 0

    result = subprocess.run(cmd, cwd=workdir)
    if result.returncode != 0:
        print(f"Command exited with code {result.returncode} (checking .cap sizes anyway)")

    passed, cap_messages = capture_passed(cfg, workdir, min_cap_bytes)
    for line in cap_messages:
        print(line)

    if not passed:
        print("Capture failed: one or more .cap files missing or too small", file=sys.stderr)
        return 1

    moves = rename_outputs(cfg, workdir, output_dir)
    if not moves:
        print("Warning: capture passed but no outputs found to rename", file=sys.stderr)
        return 1

    for _, dst in moves:
        print(f"Renamed -> {dst}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep roccap captures for CCL all-reduce")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=EXAMPLE_DIR,
        help="Directory to run torchrun in (default: this example directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Move renamed artifacts here (default: keep in workdir)",
    )
    parser.add_argument(
        "--min-cap-mb",
        type=float,
        default=None,
        help=f"Minimum .cap file size in MiB (default: MIN_CAP_MB={MIN_CAP_MB:g})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = args.workdir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None

    configs = list(iter_sweep_configs())
    min_cap_mb = MIN_CAP_MB if args.min_cap_mb is None else args.min_cap_mb
    min_cap_bytes = int(min_cap_mb * 1024 * 1024)
    print(f"Planned sweep runs: {len(configs)}")
    print(f"Pass criteria: each rank .cap file > {min_cap_mb:g} MiB")

    failures = 0
    for idx, cfg in enumerate(configs, start=1):
        print(f"\n[{idx}/{len(configs)}]")
        rc = run_one(cfg, workdir, output_dir, args.dry_run, min_cap_bytes)
        if rc != 0:
            failures += 1
            if not args.dry_run:
                print("Stopping sweep after failure.", file=sys.stderr)
                break

    if failures:
        print(f"\nSweep finished with {failures} failure(s).", file=sys.stderr)
        return 1

    print(f"\nSweep finished successfully ({len(configs)} run(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

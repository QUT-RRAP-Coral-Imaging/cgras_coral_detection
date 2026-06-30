#!/usr/bin/env python3
"""
batch_generate_manual_yaml.py

Runs generate_manual_yaml over every sub-folder in a source directory.

Each sub-folder is expected to contain a Metadata/ directory that holds:
  - Tile_<tile_id>_<batch_id>.yaml          (the tile-level YAML)
  - <tile_id>_<batch_id>_X_Y.yaml ...       (per-image YAMLs)

All output *_manual.yaml files are written to a single output directory.

Usage:
    python batch_generate_manual_yaml.py <source_dir> <output_dir> [options]

Options:
    --operator TEXT         Operator/user ID (default: "cgras")
    --species TEXT          Override the species field in all output YAMLs
    --num_tabs INT INT      Number of tabs (default: 20 20)
    --parent_folder TEXT    image_files_parent_folder override
    --workers INT           Parallel worker threads (default: 8)
    --dry_run               Print what would be done without writing files
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from typing import List, Optional, Tuple

# Allow importing from the same directory regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from generate_manual_yaml import generate_manual_yaml  # noqa: E402


def find_tile_yaml(metadata_dir: str) -> Optional[str]:
    """
    Return the canonical tile-level YAML path inside metadata_dir.
    Excludes files that contain '_manual' or ' (copy)' in their name.
    """
    candidates = glob(os.path.join(metadata_dir, "Tile_*.yaml"))
    valid = [
        p for p in candidates
        if "_manual" not in os.path.basename(p)
        and "(copy)" not in os.path.basename(p)
    ]
    if len(valid) == 1:
        return valid[0]
    if len(valid) == 0:
        return None
    # If multiple remain, prefer the shortest name (most likely the canonical one)
    return sorted(valid, key=lambda p: len(os.path.basename(p)))[0]


def process_folder(
    folder: str,
    output_dir: str,
    operator: str,
    num_tabs: List[int],
    parent_folder: Optional[str],
    species_override: Optional[str],
    dry_run: bool,
) -> Tuple[str, bool, str]:
    """
    Process a single source folder.
    Returns (folder_name, success, message).
    """
    folder_name = os.path.basename(folder)
    metadata_dir = os.path.join(folder, "Metadata")

    if not os.path.isdir(metadata_dir):
        return folder_name, False, f"No Metadata/ directory found in {folder}"

    tile_yaml = find_tile_yaml(metadata_dir)
    if tile_yaml is None:
        return folder_name, False, f"No canonical Tile_*.yaml found in {metadata_dir}"

    # Derive the output filename: Tile_<tile_id>_<batch_id>_manual.yaml
    basename = os.path.basename(tile_yaml)          # e.g. Tile_XXX_CG1-YYY.yaml
    stem = os.path.splitext(basename)[0]            # Tile_XXX_CG1-YYY
    output_filename = f"{stem}_manual.yaml"
    output_path = os.path.join(output_dir, output_filename)

    if dry_run:
        species_note = f" (species -> '{species_override}')" if species_override else ""
        return folder_name, True, f"[DRY RUN] Would write {output_path}{species_note}"

    try:
        out = generate_manual_yaml(
            tile_yaml_path=tile_yaml,
            operator=operator,
            num_tabs=num_tabs,
            parent_folder=parent_folder,
            output_path=output_path,
            image_dir=metadata_dir,
        )
        if species_override:
            import yaml
            with open(out, "r") as f:
                data = yaml.safe_load(f)
            data["species"] = species_override
            with open(out, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return folder_name, True, f"Written: {out}"
    except Exception as exc:
        return folder_name, False, f"ERROR: {exc}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate Tile_*_manual.yaml files from a source tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source_dir",
        help="Root directory containing one sub-folder per tile batch "
             "(e.g. /home/dtsai/cgras_data/Source/2025)",
    )
    parser.add_argument(
        "output_dir",
        help="Directory where all *_manual.yaml files will be written.",
    )
    parser.add_argument(
        "--operator",
        default="cgras",
        help="Operator/user ID (default: 'cgras')",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Override the species field in all output YAMLs (e.g. 'acropora').",
    )
    parser.add_argument(
        "--num_tabs",
        nargs=2,
        type=int,
        default=[20, 20],
        metavar=("ROWS", "COLS"),
        help="Number of settlement tabs (default: 20 20)",
    )
    parser.add_argument(
        "--parent_folder",
        default=None,
        help="Override image_files_parent_folder in every output YAML.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker threads (default: 8)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned actions without writing any files.",
    )

    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(source_dir):
        print(f"ERROR: source_dir '{source_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

    # Collect all immediate sub-folders
    folders = sorted(
        f.path
        for f in os.scandir(source_dir)
        if f.is_dir()
    )

    if not folders:
        print(f"No sub-folders found in '{source_dir}'.")
        sys.exit(0)

    print(f"Found {len(folders)} folders in '{source_dir}'.")
    print(f"Output directory: '{output_dir}'")
    if args.species:
        print(f"Species override: '{args.species}'")
    if args.dry_run:
        print("[DRY RUN MODE — no files will be written]\n")

    successes = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_folder,
                folder,
                output_dir,
                args.operator,
                args.num_tabs,
                args.parent_folder,
                args.species,
                args.dry_run,
            ): folder
            for folder in folders
        }

        for i, future in enumerate(as_completed(futures), start=1):
            folder_name, ok, msg = future.result()
            status = "✓" if ok else "✗"
            print(f"[{i:>3}/{len(folders)}] {status} {folder_name}: {msg}")
            if ok:
                successes += 1
            else:
                failures.append((folder_name, msg))

    print(f"\nDone. {successes}/{len(folders)} succeeded.")
    if failures:
        print(f"\nFailed ({len(failures)}):")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

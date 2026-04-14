#!/usr/bin/env python3
"""
generate_manual_yaml.py

Generates a Tile_*_manual.yaml file from:
  - A tile-level YAML file (e.g. Tile_2025Dec-982091078351762_CG1-251222124007.yaml)
  - All associated image-level YAML files in the same directory
    (e.g. 2025Dec-982091078351762_CG1-251222124007_0_0.yaml)

Usage:
    python generate_manual_yaml.py <tile_yaml_path> [options]

Options:
    --operator TEXT           Operator/user ID (default: "unknown")
    --num_tabs INT INT        Number of tabs as two integers, e.g. --num_tabs 20 20 (default: 20 20)
    --parent_folder TEXT      image_files_parent_folder path override
    --output TEXT             Output file path (default: <dir>/Tile_<tile_id>_<batch_id>_manual.yaml)
    --image_dir TEXT          Directory containing image YAMLs (default: same dir as tile yaml)
"""

import argparse
import os
import re
import sys
from datetime import datetime
from glob import glob

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


def parse_batch_time(batch_id: str) -> str:
    """
    Derive a human-readable datetime string from the batch_id timestamp suffix.

    The suffix format is YYMMDDHHMMSS, e.g. 'CG1-251222124007'
    -> timestamp '251222124007' -> '2025-12-22 12:40:07'
    """
    # Extract the numeric timestamp part after the last '-'
    match = re.search(r'-(\d{12})$', batch_id)
    if not match:
        raise ValueError(
            f"Cannot parse timestamp from batch_id '{batch_id}'. "
            "Expected format like 'CG1-251222124007' (suffix: YYMMDDHHMMSS)."
        )
    ts = match.group(1)  # e.g. '251222124007'
    dt = datetime.strptime(ts, "%y%m%d%H%M%S")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_image_yamls(image_dir: str, tile_id: str, batch_id: str) -> list[str]:
    """
    Find all image-level YAML files matching the pattern:
        {tile_id}_{batch_id}_{x}_{y}.yaml
    """
    pattern = os.path.join(image_dir, f"{tile_id}_{batch_id}_*.yaml")
    matches = glob(pattern)
    # Filter to only files that match the _X_Y.yaml suffix pattern
    filtered = [
        p for p in matches
        if re.search(r'_\d+_\d+\.yaml$', os.path.basename(p))
    ]
    return sorted(filtered)


def build_images_list(image_yamls: list[str]) -> list[dict]:
    """
    Build the images list [{x, y, file}, ...] sorted by (x, y).
    """
    images = []
    for path in image_yamls:
        data = load_yaml(path)
        images.append({
            "x": int(data["capture_x"]),
            "y": int(data["capture_y"]),
            "file": data["image_filename"],
        })
    images.sort(key=lambda img: (img["x"], img["y"]))
    return images


def generate_manual_yaml(
    tile_yaml_path: str,
    operator: str = "unknown",
    num_tabs: list[int] = None,
    parent_folder: str = None,
    output_path: str = None,
    image_dir: str = None,
) -> str:
    """
    Core function. Returns the output file path.
    """
    if num_tabs is None:
        num_tabs = [20, 20]

    tile_yaml_path = os.path.abspath(tile_yaml_path)
    tile_yaml_dir = os.path.dirname(tile_yaml_path)
    tile_yaml_basename = os.path.basename(tile_yaml_path)

    # -- Parse tile yaml --
    tile_data = load_yaml(tile_yaml_path)

    tile_id = tile_data["tile_id"]
    season = tile_data["season"]
    species = tile_data["species"]
    settle_time = str(tile_data["settle_time"])
    spawn_time = str(tile_data["spawn_time"])

    # tile_size and frame_size are stored inside a JSON string in 'metadata'
    metadata_raw = tile_data.get("metadata", "{}")
    if isinstance(metadata_raw, str):
        import json
        metadata = json.loads(metadata_raw)
    else:
        metadata = metadata_raw  # already a dict if pyyaml parsed it
    tile_size = metadata.get("tile_size", [280, 280])
    frame_size = metadata.get("frame_size", [294, 294])

    # -- Determine batch_id from the filename --
    # Tile yaml filename: Tile_{tile_id}_{batch_id}.yaml
    match = re.match(
        rf'^Tile_{re.escape(tile_id)}_(.+)\.yaml$',
        tile_yaml_basename
    )
    if not match:
        raise ValueError(
            f"Tile YAML filename '{tile_yaml_basename}' does not match expected pattern "
            f"'Tile_{{tile_id}}_{{batch_id}}.yaml'"
        )
    batch_id = match.group(1)
    batch_time = parse_batch_time(batch_id)

    # -- Find and parse image YAMLs --
    search_dir = image_dir if image_dir else tile_yaml_dir
    image_yamls = find_image_yamls(search_dir, tile_id, batch_id)
    if not image_yamls:
        print(
            f"WARNING: No image YAML files found in '{search_dir}' "
            f"matching pattern '{tile_id}_{batch_id}_X_Y.yaml'",
            file=sys.stderr,
        )

    images = build_images_list(image_yamls)

    # -- Build parent folder path --
    if parent_folder is None:
        parent_folder = (
            f"/home/qcr/cgras_data/Source/"
            f"{season.replace(season[4:], '')}/{tile_id}_{batch_id}/Export"
            # Simple default: user should override via --parent_folder
        )
        # Nicer default that mirrors the observed path pattern:
        year = season[:4]
        parent_folder = (
            f"/home/qcr/cgras_data/Source/{year}/{tile_id}_{batch_id}/Export"
        )

    # -- Assemble output dict --
    # Use a plain dict; we'll serialise with custom representer to preserve order
    output = {
        "tile_id": tile_id,
        "species": species,
        "settle_time": settle_time,
        "spawning_time": spawn_time,
        "season": season,
        "num_tabs": num_tabs,
        "tile_size": tile_size,
        "frame_size": frame_size,
        "batch_id": batch_id,
        "batch_time": batch_time,
        "importer_id": "YAML",
        "operator": operator,
        "image_files_parent_folder": parent_folder,
        "images": images,
    }

    # -- Serialise to YAML --
    # Use a Dumper that keeps insertion order and doesn't sort keys
    yaml_str = yaml.dump(
        output,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )

    # -- Write output file --
    if output_path is None:
        output_path = os.path.join(
            tile_yaml_dir, f"Tile_{tile_id}_{batch_id}_manual.yaml"
        )

    with open(output_path, "w") as f:
        f.write(yaml_str)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate a Tile_*_manual.yaml from tile + image YAML files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tile_yaml",
        help="Path to the tile-level YAML file, e.g. Tile_2025Dec-982091078351762_CG1-251222124007.yaml",
    )
    parser.add_argument(
        "--operator",
        default="unknown",
        help="Operator/user ID to embed in the output (default: 'unknown')",
    )
    parser.add_argument(
        "--num_tabs",
        nargs=2,
        type=int,
        default=[20, 20],
        metavar=("ROWS", "COLS"),
        help="Number of settlement tabs as two integers (default: 20 20)",
    )
    parser.add_argument(
        "--parent_folder",
        default=None,
        help=(
            "Override the image_files_parent_folder path. "
            "Defaults to /home/qcr/cgras_data/Source/<year>/<tile_id>_<batch_id>/Export"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output file path. "
            "Defaults to <tile_yaml_dir>/Tile_<tile_id>_<batch_id>_manual.yaml"
        ),
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help=(
            "Directory containing the image YAML files. "
            "Defaults to the same directory as the tile YAML."
        ),
    )

    args = parser.parse_args()

    try:
        out = generate_manual_yaml(
            tile_yaml_path=args.tile_yaml,
            operator=args.operator,
            num_tabs=args.num_tabs,
            parent_folder=args.parent_folder,
            output_path=args.output,
            image_dir=args.image_dir,
        )
        print(f"✓ Written: {out}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

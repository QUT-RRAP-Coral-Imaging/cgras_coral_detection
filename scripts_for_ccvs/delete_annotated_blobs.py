#!/usr/bin/env python3
"""
delete_annotated_blobs.py

Finds and deletes all files matching 'annotated_blob_*.jpg' in every
immediate sub-folder of a given root directory.

Usage:
    python delete_annotated_blobs.py <root_dir>

The script will print a count and list of matched files, then ask for
confirmation before deleting anything.
"""

import argparse
import os
import sys
from glob import glob


def main():
    parser = argparse.ArgumentParser(
        description="Find and delete annotated_blob_*.jpg files in sub-folders.",
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default="/home/dtsai/cgras_data/detector/data/2025Dec",
        help="Root directory to search (default: /home/dtsai/cgras_data/detector/data/2025Dec)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every matched file path before prompting.",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    if not os.path.isdir(root_dir):
        print(f"ERROR: '{root_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Collect all matching files recursively under root_dir
    matches = sorted(glob(os.path.join(root_dir, "**", "annotated_blob_*.jpg"), recursive=True))

    if not matches:
        print(f"No 'annotated_blob_*.jpg' files found under '{root_dir}'.")
        sys.exit(0)

    # Summary by sub-folder
    by_folder: dict = {}
    for path in matches:
        folder = os.path.dirname(path)
        by_folder.setdefault(folder, []).append(path)

    print(f"Found {len(matches)} file(s) across {len(by_folder)} sub-folder(s):\n")
    for folder in sorted(by_folder):
        rel = os.path.relpath(folder, root_dir)
        print(f"  {rel}: {len(by_folder[folder])} file(s)")

    if args.list:
        print()
        for path in matches:
            print(f"  {path}")

    print(f"\nTotal to delete: {len(matches)} file(s)")
    print("Type 'yes' to confirm deletion, anything else to abort: ", end="", flush=True)
    answer = input().strip().lower()

    if answer != "yes":
        print("Aborted. No files were deleted.")
        sys.exit(0)

    # Delete
    deleted = 0
    errors = []
    for path in matches:
        try:
            os.remove(path)
            deleted += 1
        except OSError as exc:
            errors.append((path, str(exc)))

    print(f"\nDeleted {deleted}/{len(matches)} file(s).")
    if errors:
        print(f"\n{len(errors)} error(s):")
        for path, msg in errors:
            print(f"  {path}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

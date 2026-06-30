import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

# Script to count how many times each coral tile (by tile_id) has been imaged,
# based on YAML config files found under root_directory, filtered by target species.

# Target species (lowercase for case-insensitive comparison)
TARGET_SPECIES = {"acropora tersa", "acropora hyacinthus"}


def contains_target_species(data) -> bool:
    """Recursively search for a 'species' key and check if it matches target species."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() == "species":
                if isinstance(value, str) and value.lower() in TARGET_SPECIES:
                    return True
                elif isinstance(value, list):
                    if any(isinstance(i, str) and i.lower() in TARGET_SPECIES for i in value):
                        return True
            if contains_target_species(value):
                return True
    elif isinstance(data, list):
        for item in data:
            if contains_target_species(item):
                return True
    return False


def count_tile_ids(root_dir: str) -> Counter:
    counts: Counter = Counter()
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("Tile_") and filename.endswith((".yaml", ".yml")):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, "r") as f:
                        data = yaml.safe_load(f)
                    if data and "tile_id" in data and contains_target_species(data):
                        counts[str(data["tile_id"])] += 1
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    return counts


def save_chart(counts: Counter, output_path: str = "tile_id_counts.png") -> None:
    labels, values = zip(*counts.most_common())

    fig_height = max(6, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bars = ax.barh(range(len(labels)), values, color="steelblue")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Image Count")
    ax.set_title("Tile Image Counts by tile_id (most imaged first) for " + ", ".join(TARGET_SPECIES))

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, format="png")
    plt.close(fig)
    print(f"Chart saved to: {output_path}")


if __name__ == "__main__":
    root_directory = "/home/dtsai/cgras_data/Source/2025"  # Change this to your root folder

    counts = count_tile_ids(root_directory)

    if not counts:
        print("No tile_id entries found.")
    else:
        print(f"{'tile_id':<25} {'count':>5}")
        print("-" * 32)
        for tile_id, count in counts.most_common():
            print(f"{tile_id:<25} {count:>5}")

        print(f"\nTotal unique tile_ids: {len(counts)}")
        print(f"Total YAML files matched: {sum(counts.values())}")

        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        save_chart(counts, str(output_dir / "tile_id_counts.png"))
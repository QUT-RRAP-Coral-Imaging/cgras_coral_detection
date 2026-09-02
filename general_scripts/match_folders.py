#!/usr/bin/env python3
"""
Script to ensure files in images folder have corresponding files in labels folder.
Removes extra files from the labels folder that don't have matching images.
Also removes image files that don't have matching label files.
Optionally filters train.txt to only include existing image files.
"""

from pathlib import Path


def get_file_stems(folder_path):
    """
    Get the file stems (filename without extension) from a folder.
    
    Args:
        folder_path (str): Path to the folder
        
    Returns:
        set: Set of file stems
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return set()
    
    return {file.stem for file in folder.iterdir() if file.is_file()}


def filter_train_txt(train_txt_path, images_folder, dry_run=False):
    """
    Filter train.txt file to only include images that exist in the images folder.
    
    Args:
        train_txt_path (str): Path to train.txt file
        images_folder (str): Path to images folder
        dry_run (bool): If True, only show what would be changed without actually modifying
    """
    train_path = Path(train_txt_path)
    images_path = Path(images_folder)
    
    if not train_path.exists():
        print(f"Error: Train file {train_txt_path} does not exist")
        return
    
    if not images_path.exists():
        print(f"Error: Images folder {images_folder} does not exist")
        return
    
    # Read existing train.txt content
    with open(train_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} entries in train.txt")
    
    # Filter lines to only include existing images
    valid_lines = []
    removed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract filename from path (handle both relative and absolute paths)
        image_filename = Path(line).name
        image_path = images_path / image_filename
        
        if image_path.exists():
            valid_lines.append(line + '\n')
        else:
            removed_lines.append(line)
    
    print(f"Found {len(valid_lines)} valid image entries")
    print(f"Found {len(removed_lines)} entries for missing images")
    
    if removed_lines:
        print(f"\nEntries for missing images:")
        for removed in removed_lines:
            print(f"  - {removed}")
    
    if not dry_run and removed_lines:
        # Write filtered content back to train.txt
        with open(train_path, 'w') as f:
            f.writelines(valid_lines)
        print(f"\nUpdated train.txt - removed {len(removed_lines)} entries for missing images")
    elif dry_run and removed_lines:
        print(f"\nDry run: Would remove {len(removed_lines)} entries from train.txt")
    elif not removed_lines:
        print("\nTrain.txt is already clean - all entries have corresponding images!")


def remove_images_without_labels(images_folder, labels_folder, dry_run=False):
    """
        Remove image files that don't have corresponding label .txt files.
    
    Args:
        images_folder (str): Path to images folder
        labels_folder (str): Path to labels folder
        dry_run (bool): If True, only show what would be removed without actually removing
    """
    images_path = Path(images_folder)
    labels_path = Path(labels_folder)
    
    if not images_path.exists():
        print(f"Error: Images folder {images_folder} does not exist")
        return
    
    if not labels_path.exists():
        print(f"Error: Labels folder {labels_folder} does not exist")
        return
    
    # Get file stems from images and label .txt files
    image_stems = get_file_stems(images_folder)
    label_stems = {file.stem for file in labels_path.glob("*.txt") if file.is_file()}
    
    print(f"Found {len(image_stems)} image files")
    print(f"Found {len(label_stems)} label files")
    
    # Remove images without corresponding labels
    images_without_labels = image_stems - label_stems
    if images_without_labels:
        print(f"\nFound {len(images_without_labels)} image files without corresponding .txt labels:")

        removed_image_count = 0
        for missing_stem in sorted(images_without_labels):
            image_files = list(images_path.glob(f"{missing_stem}.*"))

            for image_file in image_files:
                print(f"  - {image_file.name}")

                if not dry_run:
                    try:
                        image_file.unlink()
                        removed_image_count += 1
                        print(f"    Removed: {image_file}")
                    except Exception as e:
                        print(f"    Error removing {image_file}: {e}")
                else:
                    print(f"    Would remove: {image_file}")

        if dry_run:
            print(f"\nDry run completed. {len(images_without_labels)} image stems would be removed.")
        else:
            print(f"\nRemoved {removed_image_count} image files without labels.")
    else:
        print("\nNo image files without labels found. Nothing to remove.")


def process_dataset(base_dir, dry_run=False):
    base_path = Path(base_dir)
    images_folder = base_path / "data/images/train"
    labels_folder = base_path / "data/labels/train"
    train_txt_path = base_path / "train.txt"

    print(f"\nDataset root: {base_path}")
    print(f"Images folder: {images_folder}")
    print(f"Labels folder: {labels_folder}")
    print(f"Train file: {train_txt_path}")

    if dry_run:
        print("Running in DRY RUN mode - no files will be removed")

    print("-" * 50)
    remove_images_without_labels(images_folder, labels_folder, dry_run)

    print("\n" + "=" * 50)
    print("FILTERING TRAIN.TXT")
    print("=" * 50)
    filter_train_txt(train_txt_path, images_folder, dry_run)


def main():
    base_dir = "/home/java/reef_builders/Data/cslics/cvat_export/cslics_nov_2025/"
    folders = [
        "LAR01_uuid_1422724372929_both_multi-time",
        "LAR03_uuid_1421024219215_both_multi-time",
        "LAR05_uuid_1421024219458_both_multi-time",
        "LAR06_uuid_1421124265396_both_multi-time",
        "LAR08_uuid_1423124357365_21_lights_on",
        "LAR08_uuid_1423124357365_Lights_off",
        "LAR08_uuid_1423124357365_Lights_on",
        "LAR10_uuid_1423124357530_both_multi-time",
        "LAR11_uuid_1421124265569_lights_off",
        "LAR13_uuid_1321024251440_lights_off",
        "LAR13_uuid_1421024251440_both_multi",
        "LAR15_uuid_1421024251162_both_multi-time",
        "LAR17_uuid_1423124357407_lights_on"
    ]
    dry_run = False  # Set to True if you want to see what would be changed without making changes

    for folder in folders:
        process_dataset(base_dir+folder, dry_run)


if __name__ == "__main__":
    main()
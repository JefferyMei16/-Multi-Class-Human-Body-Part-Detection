"""
dataset_utils.py

Supports:
  - Open Images Dataset V6 via FiftyOne
  - Automatic export to ImageFolder format for PyTorch

Dependencies:
  pip install fiftyone
"""

#%% pip install fiftyone
import os
import fiftyone.zoo as foz
import fiftyone.types as fot


# Utility helpers

def ensure_dir(path: str):
    """Create directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# Main function: Download Open Images subset

def download_openimages_subset(
    classes,
    max_samples=500,
    export_dir= f"..{os.sep}data{os.sep}bigdata{os.sep}open_images",
    split="validation",
    dataset_name="open_images_subset"
):
    """
    Downloads a subset of Open Images V6 containing only the
    specified classes, then exports the data as a PyTorch-
    friendly ImageFolder structure.

    Args:
        classes (list of str):
            Human-readable class names (e.g., ["Dog", "Cat"]).
        max_samples (int):
            Total images to download (per split).
        export_dir (str):
            Output ImageFolder directory.
        split (str):
            "train", "validation", or "test".
        dataset_name (str):
            Name used inside FiftyOne's dataset registry.

    Returns:
        export_dir (str): Path to the ImageFolder dataset.
    """

    print("\n============================================")
    print("  Downloading Open Images subset via FiftyOne")
    print("============================================")
    print(f"Requested classes: {classes}")
    print(f"Split: {split}")
    print(f"Max samples: {max_samples}")
    print("--------------------------------------------\n")

    # ------------------------------------------
    # Step 1 — Load subset using FiftyOne Zoo
    # ------------------------------------------
    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v6",
            split=split,
            label_types=["classifications"],
            classes=classes,
            max_samples=max_samples,
            only_matching=True,
            dataset_name=dataset_name,
        )
    except Exception as e:
        print("\n[ERROR] Could not download Open Images subset.")
        print("FiftyOne error message:")
        print(e)
        print("\nMake sure:")
        print("  - You ran: pip install fiftyone")
        print("  - You're online")
        print("  - The class names exist in Open Images")
        return None

    print("\n[✓] Download completed.")
    print(f"Dataset size: {len(dataset)} images")
    print("Now exporting to ImageFolder format...\n")

    # ------------------------------------------
    # Step 2 — Export to ImageFolder format
    # ------------------------------------------
    ensure_dir(export_dir)

    try:
        dataset.export(
            export_dir=export_dir,
            dataset_type=fot.ImageDirectory,
            label_field="ground_truth",
        )
    except Exception as e:
        print("\n[ERROR] Export to ImageFolder format failed.")
        print(e)
        return None

    print("[✓] Export complete.")
    print(f"ImageFolder dataset saved at:\n    {export_dir}\n")

    # Clear the dataset from memory (optional)
    dataset.delete()

    return export_dir


# Convenience function: preview structure
def preview_imagefolder(imagefolder_dir, n=5):
    """
    Prints a few sample file paths from an ImageFolder dataset.
    """
    print("\n[Preview] Sample images:")
    count = 0

    for cls in os.listdir(imagefolder_dir):
        cls_dir = os.path.join(imagefolder_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                print("  ", os.path.join(cls_dir, fname))
                count += 1
                if count >= n:
                    return


# End of module


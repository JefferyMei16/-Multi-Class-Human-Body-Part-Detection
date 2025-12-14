#%%
# %pip install openimages
# 

# %% Cell 1: Imports and Setup
"""
Body Part Detection Project - Cell 1: Setup
Run this cell first to import libraries and set up directories
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from collections import Counter
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.types as fot

print("✓ All libraries imported successfully")
print(f"FiftyOne version: {fo.__version__}")

# Set paths
base_dir = "/root/autodl-tmp/final/data/bigdata"
images_dir = os.path.join(base_dir, "body_part_images")
annotations_dir = os.path.join(base_dir, "body_part_annotations")

# Create directories
os.makedirs(base_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

print(f"\n✓ Directories created:")
print(f"  Base: {base_dir}")
print(f"  Images: {images_dir}")
print(f"  Annotations: {annotations_dir}")

# %% Cell 2: Check Available Classes (Optional - for exploration)
"""
Body Part Detection Project - Cell 2: Explore Available Classes
OPTIONAL: Run this to see what body part classes are available in Open Images
This will help you choose which classes to download
"""

# List of body part classes to try (more commonly available in Open Images)
potential_classes = [
    "Human eye",
    "Human ear",
    "Human nose",
    "Human mouth",
    "Human hand",
    "Human foot",
    "Human arm",
    "Human leg",
    "Human face",
    "Human head",
    "Human body",
    "Human hair"
]

print("Checking availability of body part classes in Open Images V6...")
print("This may take a moment...\n")

# Note: This is just for information - we'll use the classes that work
for cls in potential_classes:
    print(f"  - {cls}")

print("\nNote: We'll use classes that have good coverage in the dataset")

# %% Cell 3C_Box_Balanced: Download with Balanced Bounding Box Distribution
"""
Download body part images with BALANCED bounding box distribution
Strategy: Download more images for underrepresented classes to balance annotation counts
FILTER: Only keep images with <10 total annotation boxes (not too crowded)
"""

import shutil
from collections import Counter

# Clean up previous download
print("Cleaning up previous download...")
if os.path.exists(images_dir):
    shutil.rmtree(images_dir)
if os.path.exists(annotations_dir):
    shutil.rmtree(annotations_dir)

os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)
print("✓ Cleaned up\n")

# Body part classes
body_part_classes = [
    "Human eye",
    "Human ear",
    "Human nose",
    "Human mouth",
    "Human hand",
    "Human foot",
    "Human arm",
    "Human leg",
    "Human face",
    "Human head",
    "Human body",
    "Human hair"
]

# Target: ~2000 boxes per class (balanced across all classes)
TARGET_BOXES_PER_CLASS = 2000
TOTAL_BOX_TARGET = TARGET_BOXES_PER_CLASS * len(body_part_classes)

# IMPORTANT: Maximum boxes per image filter
MAX_BOXES_PER_IMAGE = 10

# Estimated boxes per image for each class (from previous data)
estimated_boxes_per_image = {
    "Human face": 2.82,
    "Human hair": 3.54,
    "Human arm": 4.64,
    "Human hand": 3.24,
    "Human eye": 2.88,
    "Human mouth": 1.54,
    "Human body": 3.50,
    "Human foot": 2.64,
    "Human leg": 3.51,
    "Human ear": 1.81,
    "Human head": 3.53,
    "Human nose": 1.73,
}

print("="*70)
print("  DOWNLOADING WITH BALANCED BOUNDING BOX DISTRIBUTION")
print("="*70)
print(f"Target: {TARGET_BOXES_PER_CLASS:,} boxes per class")
print(f"Total target: {TOTAL_BOX_TARGET:,} boxes across all classes")
print(f"Strategy: Download more images for classes with fewer boxes per image")
print(f"FILTER: Only images with <{MAX_BOXES_PER_IMAGE} total boxes")
print("="*70 + "\n")

# Calculate images needed per class to get target boxes
images_per_class = {}
for body_part in body_part_classes:
    avg_boxes = estimated_boxes_per_image.get(body_part, 3.0)
    # Increase buffer to 2.0x since we'll filter out crowded images
    images_needed = int(TARGET_BOXES_PER_CLASS / avg_boxes * 2.0)
    images_per_class[body_part] = images_needed

print("Calculated image targets per class (with 2x buffer for filtering):")
print("-"*70)
print("Class".ljust(20) + "Est Boxes/Img".rjust(15) + "Images Needed".rjust(15))
print("-"*70)
for body_part in body_part_classes:
    avg = estimated_boxes_per_image.get(body_part, 3.0)
    needed = images_per_class[body_part]
    print(f"{body_part:20s} {avg:15.2f} {needed:15d}")
print("-"*70 + "\n")

# Download each class with calculated targets
all_samples = []
class_stats = {}
filtered_out = 0

for idx, body_part in enumerate(body_part_classes, 1):
    target_images = images_per_class[body_part]
    print(f"[{idx}/{len(body_part_classes)}] Downloading {body_part}...")
    print(f"  Target: {target_images} images to get ~{TARGET_BOXES_PER_CLASS} boxes")
    
    try:
        # Download this class (with extra to account for filtering)
        dataset = foz.load_zoo_dataset(
            "open-images-v6",
            split="train",
            label_types=["detections"],
            classes=[body_part],
            max_samples=target_images,
            only_matching=True,
            dataset_name=f"body_part_{body_part.replace(' ', '_')}",
        )
        
        # Filter samples and count boxes
        num_boxes = 0
        class_filtered = 0
        
        for sample in dataset:
            # Count TOTAL boxes in this image (all classes)
            total_boxes_in_image = 0
            target_class_boxes = 0
            
            if hasattr(sample, 'ground_truth') and sample.ground_truth:
                total_boxes_in_image = len(sample.ground_truth.detections)
                
                # Count boxes for target class
                for det in sample.ground_truth.detections:
                    if det.label == body_part:
                        target_class_boxes += 1
            
            # FILTER: Only keep images with <10 total boxes
            if total_boxes_in_image < MAX_BOXES_PER_IMAGE and total_boxes_in_image > 0:
                all_samples.append(sample)
                num_boxes += target_class_boxes
            else:
                class_filtered += 1
                filtered_out += 1
        
        num_images = len(dataset) - class_filtered
        
        class_stats[body_part] = {
            'images': num_images,
            'boxes': num_boxes,
            'boxes_per_image': num_boxes / num_images if num_images > 0 else 0,
            'filtered': class_filtered
        }
        
        print(f"  ✓ Downloaded: {len(dataset)} images")
        print(f"  ✓ Kept: {num_images} images (filtered out {class_filtered} crowded images)")
        print(f"  ✓ Boxes: {num_boxes} ({num_boxes/num_images:.2f} boxes/img)" if num_images > 0 else "")
        
        # Clean up
        dataset.delete()
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        class_stats[body_part] = {'images': 0, 'boxes': 0, 'boxes_per_image': 0, 'filtered': 0}

print("\n" + "="*70)
print("  DOWNLOAD SUMMARY")
print("="*70 + "\n")

# Calculate totals
total_images = sum(stats['images'] for stats in class_stats.values())
total_boxes = sum(stats['boxes'] for stats in class_stats.values())

print(f"Total images downloaded: {total_images:,}")
print(f"Total boxes downloaded: {total_boxes:,}")
print(f"Images filtered out (>{MAX_BOXES_PER_IMAGE} boxes): {filtered_out}")
print(f"Overall boxes per image: {total_boxes/total_images:.2f}\n")

print("Per-class results:")
print("-"*90)
print("Class".ljust(20) + "Images".rjust(10) + "Filtered".rjust(10) + 
      "Boxes".rjust(10) + "Boxes/Img".rjust(12) + "% of Target".rjust(12))
print("-"*90)

for body_part in sorted(class_stats.keys(), key=lambda x: class_stats[x]['boxes'], reverse=True):
    stats = class_stats[body_part]
    pct_target = (stats['boxes'] / TARGET_BOXES_PER_CLASS * 100) if TARGET_BOXES_PER_CLASS > 0 else 0
    print(f"{body_part:20s} {stats['images']:10d} {stats['filtered']:10d} {stats['boxes']:10d} "
          f"{stats['boxes_per_image']:12.2f} {pct_target:11.1f}%")

print("-"*90)
print(f"{'TOTAL':20s} {total_images:10d} {filtered_out:10d} {total_boxes:10d}")

# Calculate balance metrics
box_counts = [stats['boxes'] for stats in class_stats.values() if stats['boxes'] > 0]
if box_counts:
    max_boxes = max(box_counts)
    min_boxes = min(box_counts)
    balance_ratio = max_boxes / min_boxes if min_boxes > 0 else float('inf')
    
    print(f"\nBox Distribution Balance:")
    print(f"  Most boxes:      {max_boxes:,}")
    print(f"  Fewest boxes:    {min_boxes:,}")
    print(f"  Imbalance ratio: {balance_ratio:.2f}:1")
    
    if balance_ratio < 2.0:
        print("  ✓ Excellent balance!")
    elif balance_ratio < 3.0:
        print("  ✓ Good balance")
    elif balance_ratio < 5.0:
        print("  ⚠️  Moderate imbalance")
    else:
        print("  ⚠️  Significant imbalance")

print("="*70 + "\n")

# Create combined dataset
print("Creating combined balanced dataset (filtered for <10 boxes/image)...")

try:
    combined_dataset = fo.Dataset("balanced_body_parts_boxes_filtered")
    combined_dataset.add_samples(all_samples)
    
    print(f"✓ Combined dataset created: {len(combined_dataset)} images\n")
    
    # Verify filtering worked - count boxes per image
    boxes_per_image_counts = []
    for sample in combined_dataset:
        if hasattr(sample, 'ground_truth') and sample.ground_truth:
            total_boxes = len(sample.ground_truth.detections)
            boxes_per_image_counts.append(total_boxes)
    
    if boxes_per_image_counts:
        print(f"Boxes per image statistics (after filtering):")
        print(f"  Min:    {min(boxes_per_image_counts)}")
        print(f"  Max:    {max(boxes_per_image_counts)}")
        print(f"  Mean:   {np.mean(boxes_per_image_counts):.2f}")
        print(f"  Median: {np.median(boxes_per_image_counts):.1f}")
        
        if max(boxes_per_image_counts) < MAX_BOXES_PER_IMAGE:
            print(f"  ✓ All images have <{MAX_BOXES_PER_IMAGE} boxes!")
        else:
            print(f"  ⚠️  Warning: Some images have ≥{MAX_BOXES_PER_IMAGE} boxes")
        print()
    
    # Final box count verification
    final_det_counts = Counter()
    for sample in combined_dataset:
        if hasattr(sample, 'ground_truth') and sample.ground_truth:
            for det in sample.ground_truth.detections:
                final_det_counts[det.label] += 1
    
    print("Final bounding box counts:")
    print("-"*70)
    total_final_boxes = sum(final_det_counts.values())
    
    for cls, count in sorted(final_det_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_final_boxes * 100) if total_final_boxes > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {cls:20s}: {count:5d} boxes ({pct:5.1f}%) {bar}")
    print("-"*70)
    print(f"  {'TOTAL':20s}: {total_final_boxes:5d} boxes")
    print("="*70 + "\n")
    
    # Export images
    print("Exporting images...")
    combined_dataset.export(
        export_dir=images_dir,
        dataset_type=fot.ImageDirectory,
        label_field="positive_labels",
    )
    print("✓ Images exported\n")
    
    # Export COCO annotations
    print("Exporting COCO annotations...")
    combined_dataset.export(
        export_dir=annotations_dir,
        dataset_type=fot.COCODetectionDataset,
        label_field="ground_truth",
    )
    print("✓ Annotations exported\n")
    
    # Verify export
    coco_json = os.path.join(annotations_dir, "labels.json")
    if os.path.exists(coco_json):
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)
        
        print("Export Verification:")
        print(f"  ✓ Categories: {len(coco_data.get('categories', []))}")
        print(f"  ✓ Images: {len(coco_data.get('images', []))}")
        print(f"  ✓ Annotations: {len(coco_data.get('annotations', []))}")
        
        if len(coco_data.get('annotations', [])) > 0:
            print("\n✓✓✓ SUCCESS! Box-balanced, filtered dataset created!")
    
    # Clean up
    combined_dataset.delete()
    
    print("\n" + "="*70)
    print("  BOX-BALANCED DOWNLOAD COMPLETE (WITH FILTERING)")
    print("="*70)
    print(f"\nFiltering Summary:")
    print(f"  Images filtered out: {filtered_out}")
    print(f"  Images kept: {total_images}")
    print(f"  Max boxes per image: {max(boxes_per_image_counts) if boxes_per_image_counts else 0}")
    print(f"\nCompare to original imbalance:")
    print(f"  Original: Human face (8179) vs Human foot (203) = 40:1")
    print(f"  Now:      ~{balance_ratio:.1f}:1 imbalance ratio")
    print(f"\nNext steps:")
    print(f"  1. Run Cell 4G to analyze the balanced distribution")
    print(f"  2. Continue with EDA cells 5-9")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n✗ Export failed: {e}")
    import traceback
    traceback.print_exc()

# %%
# %% Cell 4E: Load Downloaded Data (Correct Version)
"""
Load and analyze the properly downloaded dataset
"""

print("="*70)
print("  ANALYZING DOWNLOADED DATA")
print("="*70 + "\n")

# Check images directory
if not os.path.exists(images_dir):
    print(f"ERROR: Images directory not found: {images_dir}")
else:
    class_counts = {}
    class_paths = {}
    
    for class_name in os.listdir(images_dir):
        class_path = os.path.join(images_dir, class_name)
        
        if not os.path.isdir(class_path) or class_name.startswith('.'):
            continue
        
        image_files = [f for f in os.listdir(class_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) > 0:
            class_counts[class_name] = len(image_files)
            class_paths[class_name] = [os.path.join(class_path, f) for f in image_files]
    
    if not class_counts:
        print("No class directories found. Checking for flat structure...")
        # Images might be in flat structure
        all_images = [f for f in os.listdir(images_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if all_images:
            print(f"Found {len(all_images)} images in flat structure")
            class_counts = {"all_images": len(all_images)}
            class_paths = {"all_images": [os.path.join(images_dir, f) for f in all_images]}
        else:
            print("ERROR: No images found!")
    
    if class_counts:
        print("✓ Images loaded!\n")
        print("Class Distribution:")
        print("-"*70)
        
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        total_images = sum(class_counts.values())
        
        for rank, (class_name, count) in enumerate(sorted_classes, 1):
            pct = (count / total_images) * 100
            bar = '█' * int(pct / 2)
            print(f"{rank}. {class_name:20s}: {count:4d} images ({pct:5.1f}%) {bar}")
        
        print("-"*70)
        print(f"   {'TOTAL':20s}: {total_images:4d} images (100.0%)")
        print("="*70 + "\n")
        
        # Check annotations
        coco_json = os.path.join(annotations_dir, "labels.json")
        if os.path.exists(coco_json):
            with open(coco_json, 'r') as f:
                coco_data = json.load(f)
            
            print("Annotations loaded:")
            print(f"  Categories: {len(coco_data.get('categories', []))}")
            print(f"  Images: {len(coco_data.get('images', []))}")
            print(f"  Bounding boxes: {len(coco_data.get('annotations', []))}")
        else:
            print("⚠️  No annotations file found")
        
        print("\n✓ Data loaded. Proceed to Cell 5 for detailed analysis")

# %% Cell 4F: Analyze Body Part Classes from Annotations
"""
Body Part Detection Project - Cell 4F: Count Images per Body Part Class
Analyzes the COCO annotations to see which body part classes are present
"""

print("="*70)
print("  BODY PART CLASS DISTRIBUTION FROM ANNOTATIONS")
print("="*70 + "\n")

# Target body part classes we're interested in
target_body_parts = [
    "Human eye",
    "Human ear",
    "Human nose",
    "Human mouth",
    "Human hand",
    "Human foot",
    "Human arm",
    "Human leg",
    "Human face",
    "Human head",
    "Human body",
    "Human hair"
]

# Load COCO annotations
coco_json = os.path.join(annotations_dir, "labels.json")

if not os.path.exists(coco_json):
    print(f"ERROR: COCO annotations not found at: {coco_json}")
    print("Run Cell 3C first to download the data!")
else:
    print(f"Loading annotations from: {coco_json}\n")
    
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    num_images = len(coco_data.get('images', []))
    num_annotations = len(coco_data.get('annotations', []))
    num_categories = len(coco_data.get('categories', []))
    
    print(f"Dataset Overview:")
    print(f"  Total images: {num_images}")
    print(f"  Total bounding boxes: {num_annotations}")
    print(f"  Total categories: {num_categories}")
    
    if num_categories == 0 or num_annotations == 0:
        print("\n⚠️  WARNING: No annotations found!")
        print("The download may not have included detection annotations.")
        print("Try re-running Cell 3C with different classes.")
    else:
        # Get all categories
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        print(f"\n" + "="*70)
        print("  CATEGORIES IN DATASET")
        print("="*70)
        print("\nAll categories found:")
        for cat_id, cat_name in sorted(categories.items()):
            print(f"  {cat_id:3d}: {cat_name}")
        
        # Map image_id to filename
        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Count images per category (an image can have multiple categories)
        category_image_sets = {cat_name: set() for cat_name in categories.values()}
        category_bbox_counts = Counter()
        
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']
            category_name = categories.get(category_id, 'Unknown')
            
            # Count bounding boxes
            category_bbox_counts[category_name] += 1
            
            # Track unique images per category
            if image_id in image_id_to_filename:
                filename = image_id_to_filename[image_id]
                category_image_sets[category_name].add(filename)
        
        print(f"\n" + "="*70)
        print("  STATISTICS PER CATEGORY")
        print("="*70 + "\n")
        
        # Sort by number of images descending
        category_stats = []
        for cat_name in categories.values():
            num_imgs = len(category_image_sets[cat_name])
            num_boxes = category_bbox_counts[cat_name]
            category_stats.append((cat_name, num_imgs, num_boxes))
        
        category_stats.sort(key=lambda x: x[1], reverse=True)
        
        print("Category".ljust(25) + "Images".rjust(10) + "Boxes".rjust(10) + "Avg Boxes/Img".rjust(15))
        print("-"*70)
        
        for cat_name, num_imgs, num_boxes in category_stats:
            avg_boxes = num_boxes / num_imgs if num_imgs > 0 else 0
            pct_imgs = (num_imgs / num_images * 100) if num_images > 0 else 0
            bar = '█' * int(pct_imgs / 2)
            print(f"{cat_name:25s} {num_imgs:10d} {num_boxes:10d} {avg_boxes:15.2f}  {bar}")
        
        print("-"*70)
        print(f"{'TOTAL':25s} {num_images:10d} {num_annotations:10d}")
        
        # Check which target body parts are present
        print(f"\n" + "="*70)
        print("  TARGET BODY PART CLASSES")
        print("="*70 + "\n")
        
        print("Checking for target body part classes:\n")
        
        found_classes = []
        missing_classes = []
        
        for body_part in target_body_parts:
            if body_part in categories.values():
                num_imgs = len(category_image_sets[body_part])
                num_boxes = category_bbox_counts[body_part]
                found_classes.append((body_part, num_imgs, num_boxes))
                status = "✓"
            else:
                missing_classes.append(body_part)
                status = "✗"
            
            if body_part in categories.values():
                print(f"  {status} {body_part:20s}: {num_imgs:4d} images, {num_boxes:5d} boxes")
            else:
                print(f"  {status} {body_part:20s}: NOT FOUND")
        
        print(f"\n" + "-"*70)
        print(f"Summary:")
        print(f"  Found:   {len(found_classes)}/{len(target_body_parts)} target classes")
        print(f"  Missing: {len(missing_classes)}/{len(target_body_parts)} target classes")
        
        if missing_classes:
            print(f"\nMissing classes:")
            for cls in missing_classes:
                print(f"    - {cls}")
            print(f"\nThese classes don't have detection annotations in Open Images V6")
        
        # Create class_counts and class_paths for found classes
        class_counts = {}
        class_paths = {}
        
        # Check if images are organized by class in directories
        if os.path.exists(images_dir):
            for class_name in os.listdir(images_dir):
                class_path = os.path.join(images_dir, class_name)
                
                if not os.path.isdir(class_path) or class_name.startswith('.'):
                    continue
                
                image_files = [f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if len(image_files) > 0:
                    class_counts[class_name] = len(image_files)
                    class_paths[class_name] = [os.path.join(class_path, f) for f in image_files]
        
        # If no class directories, use annotation-based grouping
        if not class_counts:
            print(f"\nImages are not organized by class directories.")
            print(f"Using annotation-based grouping...\n")
            
            for cat_name, image_set in category_image_sets.items():
                if len(image_set) > 0:
                    class_counts[cat_name] = len(image_set)
                    class_paths[cat_name] = [os.path.join(images_dir, fname) 
                                             for fname in image_set]
        
        print(f"\n" + "="*70)
        print("  FINAL CLASS COUNTS FOR ANALYSIS")
        print("="*70 + "\n")
        
        if class_counts:
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            total = sum(class_counts.values())
            
            for rank, (cls_name, count) in enumerate(sorted_classes, 1):
                pct = (count / total * 100) if total > 0 else 0
                bar = '█' * int(pct / 2)
                print(f"{rank:2d}. {cls_name:25s}: {count:4d} ({pct:5.1f}%) {bar}")
            
            print("-"*70)
            print(f"    {'TOTAL':25s}: {total:4d} (100.0%)")
        
        print("="*70 + "\n")
        print("✓ Analysis complete!")
        
        if found_classes:
            print("✓ Found some target body part classes. Proceed to Cell 5 for image analysis")
        else:
            print("⚠️  None of the target body part classes were found in the dataset")
            print("   The downloaded data has different classes (Person, Man, Woman, etc.)")
            print("   You can either:")
            print("   1. Continue analysis with the available classes")
            print("   2. Re-download with body part classes that exist")

# %% Cell 4G: Simple Body Part Class Counts
"""
Body Part Detection Project - Cell 4G: Count Each Body Part Class
Shows exactly how many images we have for each body part
"""

print("="*70)
print("  BODY PART CLASS COUNTS IN DOWNLOADED DATASET")
print("="*70 + "\n")

# Load COCO annotations
coco_json = os.path.join(annotations_dir, "labels.json")

with open(coco_json, 'r') as f:
    coco_data = json.load(f)

# Get categories
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

# Map image_id to filename
image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

# Count images per category
category_image_sets = {cat_name: set() for cat_name in categories.values()}
category_bbox_counts = Counter()

for ann in coco_data['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    category_name = categories[category_id]
    
    # Count bounding boxes
    category_bbox_counts[category_name] += 1
    
    # Track unique images per category
    if image_id in image_id_to_filename:
        filename = image_id_to_filename[image_id]
        category_image_sets[category_name].add(filename)

# Display results
print("Body Part".ljust(20) + "Images".rjust(10) + "Bounding Boxes".rjust(18) + "  Visualization")
print("="*70)

# Sort by number of images descending
sorted_categories = sorted(category_image_sets.items(), key=lambda x: len(x[1]), reverse=True)

total_unique_images = len(set().union(*category_image_sets.values()))
total_boxes = sum(category_bbox_counts.values())

for cat_name, image_set in sorted_categories:
    num_images = len(image_set)
    num_boxes = category_bbox_counts[cat_name]
    pct = (num_images / total_unique_images * 100) if total_unique_images > 0 else 0
    bar = '█' * int(pct / 2)
    
    print(f"{cat_name:20s} {num_images:10d} {num_boxes:18d}  {bar} ({pct:.1f}%)")

print("-"*70)
print(f"{'UNIQUE IMAGES':20s} {total_unique_images:10d}")
print(f"{'TOTAL BOXES':20s} {' '*10} {total_boxes:18d}")
print("="*70 + "\n")

# Note about overlapping images
print("NOTE: Images can contain multiple body parts, so the sum of images")
print("      per category may be greater than the total unique images.\n")

# Check the 12 target body parts
target_body_parts = [
    "Human eye",
    "Human ear", 
    "Human nose",
    "Human mouth",
    "Human hand",
    "Human foot",
    "Human arm",
    "Human leg",
    "Human face",
    "Human head",
    "Human body",
    "Human hair"
]

print("="*70)
print("  TARGET BODY PARTS STATUS")
print("="*70 + "\n")

found_count = 0
missing_count = 0

for body_part in target_body_parts:
    if body_part in categories.values():
        num_imgs = len(category_image_sets[body_part])
        num_boxes = category_bbox_counts[body_part]
        print(f"✓ {body_part:20s}: {num_imgs:4d} images, {num_boxes:5d} boxes")
        found_count += 1
    else:
        print(f"✗ {body_part:20s}: NOT IN DATASET")
        missing_count += 1

print(f"\n{'='*70}")
print(f"Found:   {found_count:2d} / {len(target_body_parts)} target classes")
print(f"Missing: {missing_count:2d} / {len(target_body_parts)} target classes")
print(f"{'='*70}\n")

# Create class_counts variable for later cells
class_counts = {cat_name: len(image_set) for cat_name, image_set in category_image_sets.items() if len(image_set) > 0}
class_paths = {cat_name: [os.path.join(images_dir, fname) for fname in image_set] 
               for cat_name, image_set in category_image_sets.items() if len(image_set) > 0}

print("✓ Class counts loaded. Proceed to Cell 5 for image analysis!")

# %% Cell 5: Image Size and Quality Analysis
"""
Body Part Detection Project - Cell 5: Analyze Image Properties
Analyzes image sizes, formats, and quality issues
"""

print("="*70)
print("  IMAGE SIZE AND QUALITY ANALYSIS")
print("="*70 + "\n")

# Check if we have data from previous cell
if 'class_paths' not in locals() or not class_paths:
    print("ERROR: No image data loaded. Run Cell 4 first!")
else:
    image_stats = {
        'widths': [],
        'heights': [],
        'aspects': [],
        'sizes_mb': [],
        'channels': [],
        'formats': [],
        'corrupted': []
    }
    
    print("Analyzing images (sampling up to 500 per class)...\n")
    
    total_analyzed = 0
    max_per_class = 500
    
    for class_name, image_paths in class_paths.items():
        print(f"Analyzing {class_name}...")
        
        sample_size = min(max_per_class, len(image_paths))
        success_count = 0
        
        for idx, img_path in enumerate(image_paths[:sample_size]):
            if idx % 100 == 0 and idx > 0:
                print(f"  Progress: {idx}/{sample_size}")
            
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    image_stats['widths'].append(width)
                    image_stats['heights'].append(height)
                    image_stats['aspects'].append(width / height)
                    image_stats['formats'].append(img.format)
                    
                    file_size_mb = os.path.getsize(img_path) / (1024 * 1024)
                    image_stats['sizes_mb'].append(file_size_mb)
                    
                    if img.mode == 'RGB':
                        image_stats['channels'].append(3)
                    elif img.mode == 'L':
                        image_stats['channels'].append(1)
                    else:
                        image_stats['channels'].append(len(img.getbands()))
                    
                    success_count += 1
                    total_analyzed += 1
                    
            except Exception as e:
                image_stats['corrupted'].append(img_path)
        
        print(f"  ✓ Analyzed {success_count}/{sample_size} images")
    
    print(f"\n✓ Total analyzed: {total_analyzed} images\n")
    
    if not image_stats['widths']:
        print("ERROR: No valid images found!")
    else:
        print("-"*70)
        print("IMAGE SIZE STATISTICS:")
        print("-"*70)
        print(f"  Width:")
        print(f"    Min:    {min(image_stats['widths']):5d}px")
        print(f"    Max:    {max(image_stats['widths']):5d}px")
        print(f"    Mean:   {np.mean(image_stats['widths']):7.1f}px")
        print(f"    Median: {np.median(image_stats['widths']):7.1f}px")
        print(f"    Std:    {np.std(image_stats['widths']):7.1f}px")
        print()
        print(f"  Height:")
        print(f"    Min:    {min(image_stats['heights']):5d}px")
        print(f"    Max:    {max(image_stats['heights']):5d}px")
        print(f"    Mean:   {np.mean(image_stats['heights']):7.1f}px")
        print(f"    Median: {np.median(image_stats['heights']):7.1f}px")
        print(f"    Std:    {np.std(image_stats['heights']):7.1f}px")
        print()
        print(f"  Aspect Ratio (Width/Height):")
        print(f"    Min:    {min(image_stats['aspects']):.2f}")
        print(f"    Max:    {max(image_stats['aspects']):.2f}")
        print(f"    Mean:   {np.mean(image_stats['aspects']):.2f}")
        print(f"    Median: {np.median(image_stats['aspects']):.2f}")
        print(f"    Std:    {np.std(image_stats['aspects']):.2f}")
        print()
        print(f"  File Size:")
        print(f"    Min:    {min(image_stats['sizes_mb']):.3f} MB")
        print(f"    Max:    {max(image_stats['sizes_mb']):.3f} MB")
        print(f"    Mean:   {np.mean(image_stats['sizes_mb']):.3f} MB")
        print(f"    Median: {np.median(image_stats['sizes_mb']):.3f} MB")
        print("-"*70)
        
        # Format statistics
        format_counts = Counter(image_stats['formats'])
        print("\nIMAGE FORMATS:")
        for fmt, count in format_counts.most_common():
            pct = (count / total_analyzed) * 100
            print(f"  {fmt:8s}: {count:5d} images ({pct:5.1f}%)")
        
        print("\nQUALITY CHECKS:")
        print("-"*70)
        
        small = sum(1 for w, h in zip(image_stats['widths'], image_stats['heights']) 
                   if w < 224 or h < 224)
        large = sum(1 for w, h in zip(image_stats['widths'], image_stats['heights']) 
                   if w > 2000 or h > 2000)
        extreme = sum(1 for a in image_stats['aspects'] if a < 0.5 or a > 2.0)
        grayscale = sum(1 for c in image_stats['channels'] if c == 1)
        
        print(f"  Small images (<224px):      {small:4d}", end='')
        if small > 0:
            pct = (small / total_analyzed) * 100
            print(f" ({pct:4.1f}%)")
            print(f"    ⚠️  May need upsampling or removal")
        else:
            print(" (0.0%)")
        
        print(f"  Large images (>2000px):     {large:4d}", end='')
        if large > 0:
            pct = (large / total_analyzed) * 100
            print(f" ({pct:4.1f}%)")
            print(f"    ℹ️  Consider resizing to reduce memory")
        else:
            print(" (0.0%)")
        
        print(f"  Extreme aspect ratios:      {extreme:4d}", end='')
        if extreme > 0:
            pct = (extreme / total_analyzed) * 100
            print(f" ({pct:4.1f}%)")
            print(f"    ℹ️  May require special padding/cropping")
        else:
            print(" (0.0%)")
        
        print(f"  Grayscale images:           {grayscale:4d}", end='')
        if grayscale > 0:
            pct = (grayscale / total_analyzed) * 100
            print(f" ({pct:4.1f}%)")
            print(f"    ℹ️  Convert to RGB for consistency")
        else:
            print(" (0.0%)")
        
        print(f"  Corrupted files:            {len(image_stats['corrupted']):4d}")
        if len(image_stats['corrupted']) > 0:
            print(f"    ⚠️  Remove these files before training")
        
        print("="*70 + "\n")
        
        print("✓ Analysis complete. Proceed to Cell 6 for visualizations")

# %% Cell 6: Generate Visualizations
"""
Body Part Detection Project - Cell 6: Create Distribution Plots
Generates plots for class distribution and image statistics
"""

print("Generating visualizations...\n")

# Check if we have data
if 'class_counts' not in locals() or 'image_stats' not in locals():
    print("ERROR: No data available. Run Cells 4G and 5 first!")
else:
    # 1. Class Distribution Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes = [x[0] for x in sorted_classes]
    counts = [x[1] for x in sorted_classes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    # Bar chart
    bars = ax1.barh(range(len(classes)), counts, color=colors)
    ax1.set_yticks(range(len(classes)))
    ax1.set_yticklabels(classes, fontsize=10)
    ax1.set_xlabel('Number of Images', fontsize=12)
    ax1.set_title('Body Part Class Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    for i, count in enumerate(counts):
        ax1.text(count + max(counts)*0.01, i, str(count), va='center', fontweight='bold')
    
    # Pie chart
    ax2.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 9})
    ax2.set_title('Body Part Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    class_plot_path = os.path.join(base_dir, 'body_part_class_distribution.png')
    plt.savefig(class_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {class_plot_path}")
    
    # 2. Image Size Distribution Plots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Width distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(image_stats['widths'], bins=50, color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(image_stats['widths']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(image_stats["widths"]):.0f}px')
    ax1.axvline(np.median(image_stats['widths']), color='orange', linestyle='--',
               linewidth=2, label=f'Median: {np.median(image_stats["widths"]):.0f}px')
    ax1.set_xlabel('Width (pixels)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Image Width Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Height distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(image_stats['heights'], bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(image_stats['heights']), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(image_stats["heights"]):.0f}px')
    ax2.axvline(np.median(image_stats['heights']), color='orange', linestyle='--',
               linewidth=2, label=f'Median: {np.median(image_stats["heights"]):.0f}px')
    ax2.set_xlabel('Height (pixels)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Image Height Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Aspect Ratio distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(image_stats['aspects'], bins=50, color='#95E1D3', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(image_stats['aspects']), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(image_stats["aspects"]):.2f}')
    ax3.axvline(np.median(image_stats['aspects']), color='orange', linestyle='--',
               linewidth=2, label=f'Median: {np.median(image_stats["aspects"]):.2f}')
    ax3.set_xlabel('Aspect Ratio (Width/Height)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # File Size distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(image_stats['sizes_mb'], bins=50, color='#F38181', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(image_stats['sizes_mb']), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(image_stats["sizes_mb"]):.3f}MB')
    ax4.axvline(np.median(image_stats['sizes_mb']), color='orange', linestyle='--',
               linewidth=2, label=f'Median: {np.median(image_stats["sizes_mb"]):.3f}MB')
    ax4.set_xlabel('File Size (MB)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('File Size Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 2D scatter: Width vs Height
    ax5 = fig.add_subplot(gs[2, :])
    scatter = ax5.scatter(image_stats['widths'], image_stats['heights'], 
                         c=image_stats['aspects'], cmap='viridis', 
                         alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    ax5.set_xlabel('Width (pixels)', fontsize=11)
    ax5.set_ylabel('Height (pixels)', fontsize=11)
    ax5.set_title('Image Dimensions Scatter Plot (colored by aspect ratio)', 
                 fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Aspect Ratio', fontsize=10)
    
    # Add reference lines
    max_dim = max(max(image_stats['widths']), max(image_stats['heights']))
    ax5.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='Square (1:1)')
    ax5.legend()
    
    size_plot_path = os.path.join(base_dir, 'body_part_image_analysis.png')
    plt.savefig(size_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {size_plot_path}")
    
    print("\n✓ All visualizations generated!")
    print(f"\nGenerated files:")
    print(f"  1. {class_plot_path}")
    print(f"  2. {size_plot_path}")
    print("\n✓ Proceed to Cell 7 for annotation details")

# %%
# %% Cell 7: Detailed Annotation Analysis
"""
Body Part Detection Project - Cell 7: Analyze Bounding Box Annotations
Examines the COCO format annotations in detail
"""

print("="*70)
print("  DETAILED ANNOTATION ANALYSIS")
print("="*70 + "\n")

coco_json = os.path.join(annotations_dir, "labels.json")

if not os.path.exists(coco_json):
    print(f"ERROR: COCO annotations not found at: {coco_json}")
else:
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    num_images = len(coco_data.get('images', []))
    num_annotations = len(coco_data.get('annotations', []))
    num_categories = len(coco_data.get('categories', []))
    
    print(f"COCO Dataset Summary:")
    print(f"  Images:      {num_images:,}")
    print(f"  Annotations: {num_annotations:,}")
    print(f"  Categories:  {num_categories}")
    print(f"  Avg boxes per image: {num_annotations/num_images:.2f}" if num_images > 0 else "")
    
    print("\n" + "-"*70)
    print("Categories:")
    print("-"*70)
    for cat in sorted(coco_data['categories'], key=lambda x: x['id']):
        print(f"  {cat['id']:3d}: {cat['name']}")
    
    # Count by category
    cat_counts = Counter(ann['category_id'] for ann in coco_data['annotations'])
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    print("\n" + "="*70)
    print("  BOUNDING BOXES PER CATEGORY")
    print("="*70 + "\n")
    
    total_boxes = sum(cat_counts.values())
    
    print("Category".ljust(25) + "Boxes".rjust(10) + "Percentage".rjust(12) + "  Visualization")
    print("-"*70)
    
    for cat_id, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        cat_name = categories.get(cat_id, 'Unknown')
        pct = (count / total_boxes) * 100
        bar = '█' * int(pct / 2)
        print(f"{cat_name:25s} {count:10,} {pct:11.1f}%  {bar}")
    
    print("-"*70)
    print(f"{'TOTAL':25s} {total_boxes:10,}")
    
    # Bbox statistics
    print("\n" + "="*70)
    print("  BOUNDING BOX STATISTICS")
    print("="*70 + "\n")
    
    bbox_areas = []
    bbox_widths = []
    bbox_heights = []
    bbox_aspect_ratios = []
    
    for ann in coco_data['annotations']:
        bbox = ann['bbox']  # [x, y, width, height]
        area = bbox[2] * bbox[3]
        aspect = bbox[2] / bbox[3] if bbox[3] > 0 else 0
        
        bbox_areas.append(area)
        bbox_widths.append(bbox[2])
        bbox_heights.append(bbox[3])
        bbox_aspect_ratios.append(aspect)
    
    print("Bounding Box Areas (px²):")
    print(f"  Min:    {min(bbox_areas):10,.0f}")
    print(f"  Max:    {max(bbox_areas):10,.0f}")
    print(f"  Mean:   {np.mean(bbox_areas):10,.0f}")
    print(f"  Median: {np.median(bbox_areas):10,.0f}")
    print(f"  Std:    {np.std(bbox_areas):10,.0f}")
    
    print("\nBounding Box Dimensions:")
    print(f"  Width  - Min: {min(bbox_widths):6.0f}px, Max: {max(bbox_widths):6.0f}px, Mean: {np.mean(bbox_widths):6.0f}px")
    print(f"  Height - Min: {min(bbox_heights):6.0f}px, Max: {max(bbox_heights):6.0f}px, Mean: {np.mean(bbox_heights):6.0f}px")
    
    print("\nBounding Box Aspect Ratios:")
    print(f"  Min:    {min(bbox_aspect_ratios):.2f}")
    print(f"  Max:    {max(bbox_aspect_ratios):.2f}")
    print(f"  Mean:   {np.mean(bbox_aspect_ratios):.2f}")
    print(f"  Median: {np.median(bbox_aspect_ratios):.2f}")
    
    # Size categories
    print("\nBounding Box Size Categories:")
    small_boxes = sum(1 for area in bbox_areas if area < 32*32)
    medium_boxes = sum(1 for area in bbox_areas if 32*32 <= area < 96*96)
    large_boxes = sum(1 for area in bbox_areas if area >= 96*96)
    
    print(f"  Small  (<32x32):   {small_boxes:6,} boxes ({small_boxes/len(bbox_areas)*100:5.1f}%)")
    print(f"  Medium (32-96):    {medium_boxes:6,} boxes ({medium_boxes/len(bbox_areas)*100:5.1f}%)")
    print(f"  Large  (>96x96):   {large_boxes:6,} boxes ({large_boxes/len(bbox_areas)*100:5.1f}%)")
    
    print("="*70 + "\n")
    print("✓ Annotation analysis complete. Proceed to Cell 8 for sample visualization")

# %%
# %% Cell 8: Visualize Sample Images with Annotations
"""
Body Part Detection Project - Cell 8: Show Sample Annotated Images
Displays sample images with their bounding box annotations
"""

print("="*70)
print("  SAMPLE IMAGE VISUALIZATION")
print("="*70 + "\n")

coco_json = os.path.join(annotations_dir, "labels.json")

if not os.path.exists(coco_json):
    print("ERROR: Annotations not found")
else:
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # Get category colors
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    cat_id_to_color = {cat_id: colors[i] for i, cat_id in enumerate(categories.keys())}
    
    # Find images with multiple annotations
    img_ann_counts = Counter(ann['image_id'] for ann in coco_data['annotations'])
    
    # Get top 6 images with most annotations
    top_images = img_ann_counts.most_common(6)
    
    print(f"Displaying 6 images with the most annotations...\n")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (img_id, ann_count) in enumerate(top_images):
        # Get image info
        img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
        img_filename = img_info['file_name']
        
        # Try to find the image
        img_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(img_path):
            # Try without subdirectories
            img_path = os.path.join(images_dir, os.path.basename(img_filename))
        
        if not os.path.exists(img_path):
            print(f"  Warning: Image not found: {img_filename}")
            continue
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        
        # Draw on image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        for ann in img_anns:
            bbox = ann['bbox']  # [x, y, width, height]
            cat_id = ann['category_id']
            cat_name = categories[cat_id]
            
            # Draw rectangle
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                linewidth=2, edgecolor=cat_id_to_color[cat_id],
                                facecolor='none')
            axes[idx].add_patch(rect)
            
            # Add label
            axes[idx].text(bbox[0], bbox[1]-5, cat_name, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=cat_id_to_color[cat_id], alpha=0.7),
                    fontsize=8, color='black', fontweight='bold')
        
        axes[idx].set_title(f'{ann_count} annotations', fontsize=10, fontweight='bold')
        print(f"  ✓ Image {idx+1}: {ann_count} annotations")
    
    plt.tight_layout()
    sample_viz_path = os.path.join(base_dir, 'body_part_sample_annotations.png')
    plt.savefig(sample_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {sample_viz_path}")
    print("\n✓ Proceed to Cell 9 to save final summary")

# %%
# %% Cell 9: Save Comprehensive Summary
"""
Body Part Detection Project - Cell 9: Save Final Summary JSON
Saves all statistics to a comprehensive JSON file
"""

print("Saving comprehensive summary...\n")

if 'class_counts' in locals() and 'image_stats' in locals():
    
    # Load annotation stats
    coco_json = os.path.join(annotations_dir, "labels.json")
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    cat_counts = Counter(ann['category_id'] for ann in coco_data['annotations'])
    
    summary = {
        "project": "Body Part Detection",
        "dataset_info": {
            "total_images": len(coco_data['images']),
            "total_unique_images": len(set().union(*[set(paths) for paths in class_paths.values()])),
            "total_annotations": len(coco_data['annotations']),
            "num_classes": len(class_counts),
            "classes": list(class_counts.keys()),
            "avg_boxes_per_image": len(coco_data['annotations']) / len(coco_data['images']) if len(coco_data['images']) > 0 else 0
        },
        "class_distribution": {
            "images_per_class": class_counts,
            "boxes_per_class": {categories[cat_id]: count for cat_id, count in cat_counts.items()}
        },
        "image_statistics": {
            "width": {
                "min": int(min(image_stats['widths'])),
                "max": int(max(image_stats['widths'])),
                "mean": float(np.mean(image_stats['widths'])),
                "median": float(np.median(image_stats['widths'])),
                "std": float(np.std(image_stats['widths']))
            },
            "height": {
                "min": int(min(image_stats['heights'])),
                "max": int(max(image_stats['heights'])),
                "mean": float(np.mean(image_stats['heights'])),
                "median": float(np.median(image_stats['heights'])),
                "std": float(np.std(image_stats['heights']))
            },
            "aspect_ratio": {
                "min": float(min(image_stats['aspects'])),
                "max": float(max(image_stats['aspects'])),
                "mean": float(np.mean(image_stats['aspects'])),
                "median": float(np.median(image_stats['aspects'])),
                "std": float(np.std(image_stats['aspects']))
            },
            "file_size_mb": {
                "min": float(min(image_stats['sizes_mb'])),
                "max": float(max(image_stats['sizes_mb'])),
                "mean": float(np.mean(image_stats['sizes_mb'])),
                "median": float(np.median(image_stats['sizes_mb']))
            }
        },
        "quality_issues": {
            "small_images": sum(1 for w, h in zip(image_stats['widths'], image_stats['heights']) if w < 224 or h < 224),
            "large_images": sum(1 for w, h in zip(image_stats['widths'], image_stats['heights']) if w > 2000 or h > 2000),
            "extreme_aspect_ratios": sum(1 for a in image_stats['aspects'] if a < 0.5 or a > 2.0),
            "grayscale_images": sum(1 for c in image_stats['channels'] if c == 1),
            "corrupted_files": len(image_stats['corrupted'])
        },
        "file_paths": {
            "images": images_dir,
            "annotations": annotations_dir,
            "coco_json": coco_json
        }
    }
    
    summary_path = os.path.join(base_dir, 'body_part_eda_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved: {summary_path}\n")
    
    print("="*70)
    print("  EDA COMPLETE!")
    print("="*70)
    print(f"\nDataset Summary:")
    print(f"  Total Images:     {summary['dataset_info']['total_images']:,}")
    print(f"  Body Part Classes: {summary['dataset_info']['num_classes']}")
    print(f"  Total Annotations: {summary['dataset_info']['total_annotations']:,}")
    print(f"  Avg Boxes/Image:   {summary['dataset_info']['avg_boxes_per_image']:.2f}")
    
    print(f"\nGenerated Files:")
    print(f"  1. {class_plot_path}")
    print(f"  2. {size_plot_path}")
    if 'sample_viz_path' in locals():
        print(f"  3. {sample_viz_path}")
        print(f"  4. {summary_path}")
    else:
        print(f"  3. {summary_path}")
    
    print(f"\nNext Steps:")
    print(f"  - Review class distribution for imbalances")
    print(f"  - Check quality issues")
    print(f"  - Split dataset into train/val/test sets")
    print(f"  - Implement data augmentation")
    print(f"  - Train detection/segmentation model")
    print("="*70)
else:
    print("ERROR: No data to save. Run previous cells first!")
# %%
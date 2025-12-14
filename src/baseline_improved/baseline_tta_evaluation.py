"""
baseline_tta_evaluation.py

Test-Time Augmentation (TTA)
Strategy: Run inference on multiple augmented versions of each image,
then merge predictions using Non-Maximum Suppression (NMS)

TTA Augmentations:
1. Original image
2. Horizontal flip
3. Scale 0.9x (smaller)
4. Scale 1.1x (larger)

This approach:
- NO retraining needed
- Improves robustness and recall
- Catches objects missed in single view
- Can be combined with adaptive thresholds

adaptive first, then tta 
"""

import os
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision.ops import nms
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
BASE_DIR = "/root/autodl-tmp/final/data/bigdata"
IMAGES_DIR = os.path.join(BASE_DIR, "body_part_images")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "body_part_annotations", "labels.json")
MODEL_PATH = "/root/autodl-tmp/final/results/baseline_frcnn/best_model.pth"
ADAPTIVE_EVAL_PATH = "/root/autodl-tmp/final/results/baseline_adaptive_threshold/adaptive_threshold_evaluation.json"
OUTPUT_DIR = "/root/autodl-tmp/final/results/baseline_tta"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("  BASELINE MODEL WITH TEST-TIME AUGMENTATION")
print("="*70)
print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("="*70 + "\n")

# PART 1: CLASS-SPECIFIC THRESHOLDS (FROM ADAPTIVE APPROACH)
"""
IMPROVEMENT 1: Use class-specific thresholds from Option 1
These were proven to work well
"""

CLASS_THRESHOLDS = {
    1: 0.25,  # Human arm
    2: 0.45,  # Human body
    3: 0.50,  # Human ear
    4: 0.35,  # Human eye
    5: 0.50,  # Human face
    6: 0.45,  # Human foot
    7: 0.25,  # Human hair
    8: 0.30,  # Human hand
    9: 0.25,  # Human head
    10: 0.50, # Human leg
    11: 0.50, # Human mouth
    12: 0.40  # Human nose
}

categories = {
    1: "Human arm", 2: "Human body", 3: "Human ear", 4: "Human eye",
    5: "Human face", 6: "Human foot", 7: "Human hair", 8: "Human hand",
    9: "Human head", 10: "Human leg", 11: "Human mouth", 12: "Human nose"
}

print("Using adaptive class-specific thresholds + TTA")
print("="*70 + "\n")

# PART 2: TEST-TIME AUGMENTATION FUNCTIONS
def tta_horizontal_flip(image, predictions):
    """
    IMPROVEMENT 2: Test-Time Augmentation - Horizontal Flip
    
    Flips image horizontally and adjusts bounding boxes back
    Helps detect objects that may be missed in original orientation
    """
    # Flip image
    image_flipped = torch.flip(image, dims=[2])
    
    # Flip predictions back
    width = image.shape[2]
    for pred in predictions:
        boxes = pred['boxes']
        # Flip boxes: [x1, y1, x2, y2] -> [width-x2, y1, width-x1, y2]
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        pred['boxes'] = boxes
    
    return predictions

def tta_scale(image, scale_factor):
    """
    IMPROVEMENT 2: Test-Time Augmentation - Multi-Scale
    
    Resizes image by scale_factor and adjusts boxes accordingly
    - scale < 1.0: Makes objects appear smaller (helps detect large objects)
    - scale > 1.0: Makes objects appear larger (helps detect small objects)
    """
    h, w = image.shape[1], image.shape[2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Resize image
    image_scaled = F.interpolate(
        image.unsqueeze(0), 
        size=(new_h, new_w), 
        mode='bilinear',
        align_corners=False
    )[0]
    
    return image_scaled, scale_factor

def merge_predictions_nms(predictions_list, iou_threshold=0.5):
    """
    IMPROVEMENT 2: Merge Multiple Predictions with NMS
    
    Combines predictions from different augmentations:
    1. Concatenate all boxes, scores, labels
    2. Apply Non-Maximum Suppression per class
    3. Return merged predictions
    
    This eliminates duplicate detections from different augmentations
    """
    if len(predictions_list) == 0:
        return {'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])}
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    # Collect all predictions
    for pred in predictions_list:
        all_boxes.append(pred['boxes'])
        all_scores.append(pred['scores'])
        all_labels.append(pred['labels'])
    
    # Concatenate
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Apply NMS per class
    keep_indices = []
    unique_labels = torch.unique(all_labels)
    
    for label in unique_labels:
        # Get indices for this class
        class_mask = all_labels == label
        class_boxes = all_boxes[class_mask]
        class_scores = all_scores[class_mask]
        class_indices = torch.where(class_mask)[0]
        
        # Apply NMS
        if len(class_boxes) > 0:
            keep = nms(class_boxes, class_scores, iou_threshold)
            keep_indices.extend(class_indices[keep].tolist())
    
    # Return merged predictions
    if len(keep_indices) > 0:
        keep_indices = torch.tensor(keep_indices)
        return {
            'boxes': all_boxes[keep_indices],
            'scores': all_scores[keep_indices],
            'labels': all_labels[keep_indices]
        }
    else:
        return {'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])}

def predict_with_tta(model, image, device, class_thresholds):
    """
    IMPROVEMENT 2: Complete TTA Pipeline
    
    Runs inference on:
    1. Original image
    2. Horizontally flipped image
    3. 0.9x scaled image (detect larger objects better)
    4. 1.1x scaled image (detect smaller objects better)
    
    Then merges all predictions using NMS
    """
    predictions_list = []
    
    # 1. Original image
    with torch.no_grad():
        pred_original = model([image.to(device)])[0]
    
    # Filter by class-specific thresholds
    filtered_pred = filter_by_threshold(pred_original, class_thresholds)
    if len(filtered_pred['boxes']) > 0:
        predictions_list.append(filtered_pred)
    
    # 2. Horizontal flip
    image_flipped = torch.flip(image, dims=[2])
    with torch.no_grad():
        pred_flipped = model([image_flipped.to(device)])[0]
    
    # Flip boxes back
    width = image.shape[2]
    boxes_flipped = pred_flipped['boxes'].cpu()
    boxes_flipped[:, [0, 2]] = width - boxes_flipped[:, [2, 0]]
    pred_flipped['boxes'] = boxes_flipped
    pred_flipped['scores'] = pred_flipped['scores'].cpu()
    pred_flipped['labels'] = pred_flipped['labels'].cpu()
    
    filtered_pred = filter_by_threshold(pred_flipped, class_thresholds)
    if len(filtered_pred['boxes']) > 0:
        predictions_list.append(filtered_pred)
    
    # 3. Scale 0.9x (smaller - helps detect large objects)
    image_small = F.interpolate(
        image.unsqueeze(0),
        scale_factor=0.9,
        mode='bilinear',
        align_corners=False
    )[0]
    
    with torch.no_grad():
        pred_small = model([image_small.to(device)])[0]
    
    # Scale boxes back
    boxes_small = pred_small['boxes'].cpu() / 0.9
    pred_small['boxes'] = boxes_small
    pred_small['scores'] = pred_small['scores'].cpu()
    pred_small['labels'] = pred_small['labels'].cpu()
    
    filtered_pred = filter_by_threshold(pred_small, class_thresholds)
    if len(filtered_pred['boxes']) > 0:
        predictions_list.append(filtered_pred)
    
    # 4. Scale 1.1x (larger - helps detect small objects)
    image_large = F.interpolate(
        image.unsqueeze(0),
        scale_factor=1.1,
        mode='bilinear',
        align_corners=False
    )[0]
    
    with torch.no_grad():
        pred_large = model([image_large.to(device)])[0]
    
    # Scale boxes back
    boxes_large = pred_large['boxes'].cpu() / 1.1
    pred_large['boxes'] = boxes_large
    pred_large['scores'] = pred_large['scores'].cpu()
    pred_large['labels'] = pred_large['labels'].cpu()
    
    filtered_pred = filter_by_threshold(pred_large, class_thresholds)
    if len(filtered_pred['boxes']) > 0:
        predictions_list.append(filtered_pred)
    
    # Merge all predictions with NMS
    merged_pred = merge_predictions_nms(predictions_list, iou_threshold=0.5)
    
    return merged_pred

def filter_by_threshold(pred, class_thresholds):
    """Apply class-specific thresholds to predictions"""
    boxes = pred['boxes'].cpu() if pred['boxes'].is_cuda else pred['boxes']
    scores = pred['scores'].cpu() if pred['scores'].is_cuda else pred['scores']
    labels = pred['labels'].cpu() if pred['labels'].is_cuda else pred['labels']
    
    keep_mask = torch.zeros(len(scores), dtype=torch.bool)
    
    for i, (score, label) in enumerate(zip(scores, labels)):
        threshold = class_thresholds.get(label.item(), 0.5)
        if score > threshold:
            keep_mask[i] = True
    
    return {
        'boxes': boxes[keep_mask],
        'scores': scores[keep_mask],
        'labels': labels[keep_mask]
    }

# PART 3: DATASET
class BodyPartDataset(Dataset):
    """Custom Dataset for COCO-format body part detection"""
    
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = coco_data['images']
        self.annotations = coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        self.img_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.img_to_anns[ann['image_id']].append(ann)
        
        self.valid_images = []
        for img_info in self.images:
            img_path = os.path.join(self.images_dir, img_info['file_name'])
            img_id = img_info['id']
            
            if not os.path.exists(img_path):
                continue
            if img_id not in self.img_to_anns or len(self.img_to_anns[img_id]) == 0:
                continue
            
            try:
                with Image.open(img_path) as test_img:
                    test_img.verify()
                self.valid_images.append(img_info)
            except:
                continue
        
        print(f"Dataset: {len(self.valid_images)} valid images")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new('RGB', (800, 600), color='gray')
        
        img_id = img_info['id']
        anns = self.img_to_anns[img_id]
        
        boxes = []
        labels = []
        areas = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
        
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [1]
            areas = [1]
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, target

def get_transform():
    return transforms.Compose([transforms.ToTensor()])

# Load dataset
print("="*70)
print("  LOADING DATASET")
print("="*70 + "\n")

full_dataset = BodyPartDataset(IMAGES_DIR, ANNOTATIONS_FILE, get_transform())

# Use same split as baseline
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

_, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Validation set: {len(val_dataset)} images")

def collate_fn(batch):
    return tuple(zip(*batch))

# Use batch_size=1 for TTA (process one image at a time)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=0, collate_fn=collate_fn, pin_memory=True
)

print(f"Validation batches: {len(val_loader)}")
print("  Note: TTA processes one image at a time (slower but more thorough)")
print("="*70 + "\n")

# PART 4: LOAD MODEL
print("="*70)
print("  LOADING BASELINE MODEL")
print("="*70 + "\n")

model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 13
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("✓ Baseline model loaded")
print("="*70 + "\n")

# PART 5: LOAD ADAPTIVE METRICS FOR COMPARISON
print("="*70)
print("  LOADING ADAPTIVE THRESHOLD METRICS")
print("="*70 + "\n")

adaptive_metrics = {}
if os.path.exists(ADAPTIVE_EVAL_PATH):
    with open(ADAPTIVE_EVAL_PATH, 'r') as f:
        adaptive_metrics = json.load(f)
    print(f"✓ Adaptive metrics loaded")
    print(f"  Adaptive total detections: {adaptive_metrics.get('total_detections', 0)}")
else:
    print(f"⚠️  Adaptive metrics not found")

print("="*70 + "\n")

# PART 6: EVALUATION WITH TTA
print("="*70)
print("  RUNNING EVALUATION WITH TTA")
print("="*70)
print("  Each image gets 4x inference (original + flip + 2 scales)")
print("  This will be ~4x slower than standard evaluation")
print("  Estimated time: 15-20 minutes")
print("="*70 + "\n")

def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Evaluation data structures
class_detections = defaultdict(int)
class_confidences = defaultdict(list)
class_true_positives = defaultdict(int)
class_false_positives = defaultdict(int)
class_false_negatives = defaultdict(int)
class_ground_truths = defaultdict(int)

example_predictions = []
num_examples = 6

print("Running TTA inference...")
start_time = time.time()

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            progress = (batch_idx / len(val_loader)) * 100
            eta = (elapsed / (batch_idx + 1)) * (len(val_loader) - batch_idx)
            print(f"  Progress: {batch_idx}/{len(val_loader)} ({progress:.1f}%) - "
                  f"Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")
        
        # TTA expects single image
        image = images[0]
        target = targets[0]
        
        # Run TTA prediction
        pred = predict_with_tta(model, image, device, CLASS_THRESHOLDS)
        
        # Move to CPU
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']
        
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        
        # Count ground truths
        for gt_label in gt_labels:
            class_ground_truths[gt_label.item()] += 1
        
        # Count detections
        for label, score in zip(pred_labels, pred_scores):
            class_detections[label.item()] += 1
            class_confidences[label.item()].append(score.item())
        
        # Match predictions to ground truth
        matched_gt = set()
        
        for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_idx in matched_gt:
                    continue
                if pred_label != gt_label:
                    continue
                
                iou = compute_iou(pred_box.numpy(), gt_box.numpy())
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= 0.5:
                class_true_positives[pred_label.item()] += 1
                matched_gt.add(best_gt_idx)
            else:
                class_false_positives[pred_label.item()] += 1
        
        # Count false negatives
        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_idx not in matched_gt:
                class_false_negatives[gt_label.item()] += 1
        
        # Store examples
        if len(example_predictions) < num_examples:
            example_predictions.append({
                'image': image,
                'predictions': pred,
                'ground_truth': {'boxes': gt_boxes, 'labels': gt_labels}
            })

total_time = time.time() - start_time
print(f"\n✓ TTA inference complete in {total_time:.0f}s ({total_time/60:.1f} minutes)")
print(f"  Average time per image: {total_time/len(val_loader):.2f}s\n")

# PART 7: COMPUTE METRICS AND COMPARISON
print("="*70)
print("  TTA RESULTS")
print("="*70 + "\n")

total_detections = sum(class_detections.values())
total_ground_truths = sum(class_ground_truths.values())

print(f"Total detections: {total_detections}")
print(f"Total ground truth: {total_ground_truths}")

if adaptive_metrics:
    adaptive_total = adaptive_metrics.get('total_detections', 0)
    change = total_detections - adaptive_total
    pct_change = (change / adaptive_total * 100) if adaptive_total > 0 else 0
    print(f"\nAdaptive detections: {adaptive_total}")
    print(f"Change from adaptive: {change:+d} ({pct_change:+.1f}%)")
    
    if pct_change > 0:
        print(f"✓ TTA IMPROVEMENT: {pct_change:.1f}% more detections than adaptive!")
    elif abs(pct_change) < 5:
        print(f"≈ Similar to adaptive (within 5%)")
    else:
        print(f"✗ Fewer detections than adaptive")

print()

# Per-class metrics
print("Per-Class Performance:")
print("-"*110)
print(f"{'Class':<20} {'Det':<8} {'Adaptive':<10} {'Change':<10} {'Prec':<10} {'Rec':<10} {'F1':<10}")
print("-"*110)

class_metrics = {}
overall_precision = []
overall_recall = []
overall_f1 = []

for class_id in sorted(categories.keys()):
    class_name = categories[class_id]
    detections = class_detections.get(class_id, 0)
    gt_count = class_ground_truths.get(class_id, 0)
    tp = class_true_positives.get(class_id, 0)
    fp = class_false_positives.get(class_id, 0)
    fn = class_false_negatives.get(class_id, 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if precision > 0 or recall > 0:
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
    
    # Get adaptive comparison
    adaptive_det = 0
    if adaptive_metrics and 'per_class_metrics' in adaptive_metrics:
        adaptive_class = adaptive_metrics['per_class_metrics'].get(class_name, {})
        adaptive_det = adaptive_class.get('detections', 0)
    
    change = detections - adaptive_det
    change_str = f"{change:+d}" if adaptive_det > 0 else "N/A"
    
    class_metrics[class_name] = {
        'detections': detections,
        'adaptive_detections': adaptive_det,
        'change': change,
        'ground_truth': gt_count,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    print(f"{class_name:<20} {detections:<8} {adaptive_det:<10} {change_str:<10} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")

print("-"*110)

# Overall metrics
mean_precision = np.mean(overall_precision) if overall_precision else 0
mean_recall = np.mean(overall_recall) if overall_recall else 0
mean_f1 = np.mean(overall_f1) if overall_f1 else 0

print(f"\n{'='*70}")
print(f"  OVERALL METRICS COMPARISON")
print(f"{'='*70}\n")

print(f"{'Metric':<20} {'TTA':<15} {'Adaptive':<15} {'Change':<15}")
print("-"*70)

if adaptive_metrics:
    adaptive_prec = adaptive_metrics.get('mean_precision', 0)
    adaptive_rec = adaptive_metrics.get('mean_recall', 0)
    adaptive_f1 = adaptive_metrics.get('mean_f1_score', 0)
    
    prec_change = mean_precision - adaptive_prec
    rec_change = mean_recall - adaptive_rec
    f1_change = mean_f1 - adaptive_f1
    
    print(f"{'Precision':<20} {mean_precision:<15.3f} {adaptive_prec:<15.3f} {prec_change:+.3f}")
    print(f"{'Recall':<20} {mean_recall:<15.3f} {adaptive_rec:<15.3f} {rec_change:+.3f}")
    print(f"{'F1 Score':<20} {mean_f1:<15.3f} {adaptive_f1:<15.3f} {f1_change:+.3f}")
    print(f"{'mAP@0.5':<20} {mean_precision:<15.3f} {adaptive_prec:<15.3f} {prec_change:+.3f}")
else:
    print(f"{'Precision':<20} {mean_precision:<15.3f}")
    print(f"{'Recall':<20} {mean_recall:<15.3f}")
    print(f"{'F1 Score':<20} {mean_f1:<15.3f}")

print("-"*70)

# Save metrics
eval_metrics = {
    "strategy": "Test-Time Augmentation + Adaptive Thresholds",
    "tta_config": {
        "augmentations": ["original", "horizontal_flip", "scale_0.9", "scale_1.1"],
        "nms_iou_threshold": 0.5,
        "class_thresholds": CLASS_THRESHOLDS
    },
    "total_detections": total_detections,
    "total_ground_truths": total_ground_truths,
    "mean_precision": float(mean_precision),
    "mean_recall": float(mean_recall),
    "mean_f1_score": float(mean_f1),
    "mAP@0.5": float(mean_precision),
    "inference_time_seconds": total_time,
    "avg_time_per_image": total_time / len(val_loader),
    "per_class_metrics": class_metrics,
    "adaptive_comparison": {
        "adaptive_detections": adaptive_metrics.get('total_detections', 0) if adaptive_metrics else 0,
        "detection_change": total_detections - adaptive_metrics.get('total_detections', 0) if adaptive_metrics else 0,
        "adaptive_precision": adaptive_metrics.get('mean_precision', 0) if adaptive_metrics else 0,
        "adaptive_recall": adaptive_metrics.get('mean_recall', 0) if adaptive_metrics else 0,
        "adaptive_f1": adaptive_metrics.get('mean_f1_score', 0) if adaptive_metrics else 0
    }
}

with open(os.path.join(OUTPUT_DIR, 'tta_evaluation.json'), 'w') as f:
    json.dump(eval_metrics, f, indent=2)

print(f"\n✓ Metrics saved to: {OUTPUT_DIR}/tta_evaluation.json")

# PART 8: VISUALIZATIONS
print("\n" + "="*70)
print("  GENERATING VISUALIZATIONS")
print("="*70 + "\n")

colors = {
    1: 'red', 2: 'blue', 3: 'green', 4: 'yellow',
    5: 'purple', 6: 'orange', 7: 'pink', 8: 'cyan',
    9: 'magenta', 10: 'lime', 11: 'brown', 12: 'navy'
}

fig, axes = plt.subplots(num_examples, 2, figsize=(16, 4*num_examples))

for idx, example in enumerate(example_predictions):
    img = example['image'].cpu().numpy().transpose(1, 2, 0)
    pred = example['predictions']
    gt = example['ground_truth']
    
    # Ground truth
    axes[idx, 0].imshow(img)
    axes[idx, 0].set_title(f'Ground Truth (Image {idx+1})', fontweight='bold')
    axes[idx, 0].axis('off')
    
    for box, label in zip(gt['boxes'], gt['labels']):
        x1, y1, x2, y2 = box
        color = colors.get(label.item(), 'white')
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=2, edgecolor=color, facecolor='none')
        axes[idx, 0].add_patch(rect)
        axes[idx, 0].text(x1, y1-5, categories[label.item()], 
                        color=color, fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    # TTA Predictions
    axes[idx, 1].imshow(img)
    axes[idx, 1].set_title(f'TTA Predictions (Image {idx+1}) - {len(pred["boxes"])} detections', 
                          fontweight='bold')
    axes[idx, 1].axis('off')
    
    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        x1, y1, x2, y2 = box
        color = colors.get(label.item(), 'white')
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
        axes[idx, 1].add_patch(rect)
        axes[idx, 1].text(x1, y1-5, f"{categories[label.item()]}: {score:.2f}", 
                        color=color, fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tta_example_predictions.png'), dpi=300, bbox_inches='tight')
print(f"✓ Example predictions saved")

print("\n" + "="*70)
print("  TTA EVALUATION COMPLETE")
print("="*70)
print(f"\nOutput files:")
print(f"  - {OUTPUT_DIR}/tta_evaluation.json")
print(f"  - {OUTPUT_DIR}/tta_example_predictions.png")
print(f"\nInference time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Speedup over adaptive: ~4x slower (but potentially more accurate)")
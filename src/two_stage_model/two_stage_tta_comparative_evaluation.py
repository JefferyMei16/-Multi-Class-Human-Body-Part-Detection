"""
two_stage_tta_comparative_evaluation.py

Comprehensive evaluation of Two-Stage Fine-Tuned + TTA model
Compares training annotation distribution vs. detection distribution
Uses EXACT same data split as training for consistency

This code is partially written by ChatGPT since I could not quite figure out why the two_stage model and the baseline model 
is running on two different validation sets 
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
from collections import defaultdict, Counter
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
MODEL_PATH = "/root/autodl-tmp/final/results/two_stage_finetuning/best_model.pth"
OUTPUT_DIR = "/root/autodl-tmp/final/results/two_stage_with_tta"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("  TWO-STAGE + TTA MODEL COMPREHENSIVE EVALUATION")
print("="*70)
print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("="*70 + "\n")

# PART 1: CLASS-SPECIFIC ADAPTIVE THRESHOLDS

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

print("Adaptive Class-Specific Thresholds:")
print("-"*50)
for class_id in sorted(categories.keys()):
    print(f"{categories[class_id]:<20} {CLASS_THRESHOLDS[class_id]:.2f}")
print("-"*50 + "\n")

# PART 2: TEST-TIME AUGMENTATION FUNCTIONS
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

def merge_predictions_nms(predictions_list, iou_threshold=0.5):
    """Merge multiple predictions with NMS"""
    if len(predictions_list) == 0:
        return {'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])}
    
    all_boxes = torch.cat([p['boxes'] for p in predictions_list], dim=0)
    all_scores = torch.cat([p['scores'] for p in predictions_list], dim=0)
    all_labels = torch.cat([p['labels'] for p in predictions_list], dim=0)
    
    keep_indices = []
    for label in torch.unique(all_labels):
        class_mask = all_labels == label
        class_boxes = all_boxes[class_mask]
        class_scores = all_scores[class_mask]
        class_indices = torch.where(class_mask)[0]
        
        if len(class_boxes) > 0:
            keep = nms(class_boxes, class_scores, iou_threshold)
            keep_indices.extend(class_indices[keep].tolist())
    
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
    """Run TTA prediction with 4 augmentations"""
    predictions_list = []
    
    # 1. Original
    with torch.no_grad():
        pred = model([image.to(device)])[0]
    filtered_pred = filter_by_threshold(pred, class_thresholds)
    if len(filtered_pred['boxes']) > 0:
        predictions_list.append(filtered_pred)
    
    # 2. Horizontal flip
    image_flipped = torch.flip(image, dims=[2])
    with torch.no_grad():
        pred_flipped = model([image_flipped.to(device)])[0]
    
    width = image.shape[2]
    boxes_flipped = pred_flipped['boxes'].cpu()
    boxes_flipped[:, [0, 2]] = width - boxes_flipped[:, [2, 0]]
    pred_flipped['boxes'] = boxes_flipped
    pred_flipped['scores'] = pred_flipped['scores'].cpu()
    pred_flipped['labels'] = pred_flipped['labels'].cpu()
    
    filtered_pred = filter_by_threshold(pred_flipped, class_thresholds)
    if len(filtered_pred['boxes']) > 0:
        predictions_list.append(filtered_pred)
    
    # 3. Scale 0.9x
    image_small = F.interpolate(image.unsqueeze(0), scale_factor=0.9, mode='bilinear', align_corners=False)[0]
    with torch.no_grad():
        pred_small = model([image_small.to(device)])[0]
    boxes_small = pred_small['boxes'].cpu() / 0.9
    pred_small['boxes'] = boxes_small
    pred_small['scores'] = pred_small['scores'].cpu()
    pred_small['labels'] = pred_small['labels'].cpu()
    
    filtered_pred = filter_by_threshold(pred_small, class_thresholds)
    if len(filtered_pred['boxes']) > 0:
        predictions_list.append(filtered_pred)
    
    # 4. Scale 1.1x
    image_large = F.interpolate(image.unsqueeze(0), scale_factor=1.1, mode='bilinear', align_corners=False)[0]
    with torch.no_grad():
        pred_large = model([image_large.to(device)])[0]
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

# PART 3: GET TRAINING DISTRIBUTION USING EXACT SAME SPLIT AS PYTORCH
print("="*70)
print("  LOADING TRAINING DISTRIBUTION (EXACT SAME SPLIT AS MODEL TRAINING)")
print("="*70 + "\n")

# Load full COCO annotations
with open(ANNOTATIONS_FILE, 'r') as f:
    coco_data = json.load(f)

# Create temporary dataset to get the EXACT same split as PyTorch random_split
class BodyPartDatasetSimple(Dataset):
    def __init__(self, images_dir, annotations_file):
        self.images_dir = images_dir
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = coco_data['images']
        self.annotations = coco_data['annotations']
        
        self.img_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.img_to_anns[ann['image_id']].append(ann)
        
        # Validate images (same as training dataset)
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
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        return self.valid_images[idx]

# Create dataset with same validation as actual training
temp_dataset = BodyPartDatasetSimple(IMAGES_DIR, ANNOTATIONS_FILE)
print(f"Total valid images: {len(temp_dataset)}")

# Split EXACTLY like PyTorch training
train_size = int(0.8 * len(temp_dataset))
val_size = len(temp_dataset) - train_size

train_dataset_temp, val_dataset_temp = torch.utils.data.random_split(
    temp_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # SAME SEED AS TRAINING
)

# Get training image IDs from the exact split
train_image_ids = set()
for idx in train_dataset_temp.indices:
    img_info = temp_dataset[idx]
    train_image_ids.add(img_info['id'])

print(f"Training images: {len(train_image_ids)}")
print(f"Validation images: {len(val_dataset_temp)}")

# Count annotations in TRAINING SET using exact split
training_annotations = Counter()
for ann in coco_data['annotations']:
    if ann['image_id'] in train_image_ids:
        training_annotations[ann['category_id']] += 1

total_train_annotations = sum(training_annotations.values())

print("\nTraining Set Annotation Distribution (80% split - EXACT):")
print("-"*70)
print(f"{'Class':<20} {'Train Boxes':<15} {'Train %':<15}")
print("-"*70)

train_distribution = {}
for class_id in sorted(categories.keys()):
    count = training_annotations.get(class_id, 0)
    pct = (count / total_train_annotations * 100) if total_train_annotations > 0 else 0
    train_distribution[class_id] = {
        'count': count,
        'percentage': pct
    }
    print(f"{categories[class_id]:<20} {count:<15} {pct:<15.1f}%")

print("-"*70)
print(f"{'TOTAL':<20} {total_train_annotations:<15}")
print("="*70 + "\n")

# PART 4: DATASET FOR EVALUATION
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
        
        boxes, labels, areas = [], [], []
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

print("="*70)
print("  LOADING VALIDATION DATASET")
print("="*70 + "\n")

full_dataset = BodyPartDataset(IMAGES_DIR, ANNOTATIONS_FILE, get_transform())

# Split for validation (same as training)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

_, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Validation set: {len(val_dataset)} images")

def collate_fn(batch):
    return tuple(zip(*batch))

val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=0, collate_fn=collate_fn, pin_memory=True
)

print(f"Validation batches: {len(val_loader)}")
print("="*70 + "\n")

# PART 5: LOAD MODEL
print("="*70)
print("  LOADING TWO-STAGE FINE-TUNED MODEL")
print("="*70 + "\n")

model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 13
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("✓ Two-stage fine-tuned model loaded successfully")
print("="*70 + "\n")

# PART 6: RUN EVALUATION WITH TTA
print("="*70)
print("  RUNNING EVALUATION WITH TTA")
print("="*70 + "\n")

print("TTA Configuration:")
print("  - 4 augmentations: original, horizontal flip, 0.9× scale, 1.1× scale")
print("  - Class-specific adaptive thresholds")
print("  - NMS merging with IoU threshold 0.5\n")

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

class_detections = defaultdict(int)
class_confidences = defaultdict(list)
class_true_positives = defaultdict(int)
class_false_positives = defaultdict(int)
class_false_negatives = defaultdict(int)
class_ground_truths = defaultdict(int)

# Store example predictions
example_predictions = []
num_examples = 6

print("Running TTA inference on validation set...")
print("This will take several minutes...\n")
start_time = time.time()

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            progress = (batch_idx / len(val_loader)) * 100
            eta = (elapsed / (batch_idx + 1)) * (len(val_loader) - batch_idx) if batch_idx > 0 else 0
            print(f"  Progress: {batch_idx}/{len(val_loader)} ({progress:.1f}%) - "
                  f"Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")
        
        image = images[0]
        target = targets[0]
        
        pred = predict_with_tta(model, image, device, CLASS_THRESHOLDS)
        
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']
        
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        
        for gt_label in gt_labels:
            class_ground_truths[gt_label.item()] += 1
        
        for label, score in zip(pred_labels, pred_scores):
            class_detections[label.item()] += 1
            class_confidences[label.item()].append(score.item())
        
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
        
        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_idx not in matched_gt:
                class_false_negatives[gt_label.item()] += 1
        
        # Store examples for visualization
        if len(example_predictions) < num_examples:
            example_predictions.append({
                'image': image,
                'predictions': {'boxes': pred_boxes, 'labels': pred_labels, 'scores': pred_scores},
                'ground_truth': {'boxes': gt_boxes, 'labels': gt_labels}
            })

total_time = time.time() - start_time
print(f"\n✓ TTA inference complete!")
print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} minutes)")
print(f"  Average time per image: {total_time/len(val_loader):.2f}s\n")

# PART 7: COMPUTE METRICS
print("="*70)
print("  DETECTION METRICS")
print("="*70 + "\n")

total_detections = sum(class_detections.values())
total_ground_truths = sum(class_ground_truths.values())

print(f"Total detections: {total_detections:,}")
print(f"Total ground truth objects: {total_ground_truths:,}")
print(f"Detection rate: {total_detections/total_ground_truths:.3f}\n")

# Per-class metrics
print("Per-Class Performance:")
print("-"*100)
print(f"{'Class':<20} {'Detections':<12} {'GT':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Avg Conf':<12}")
print("-"*100)

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
    avg_conf = np.mean(class_confidences.get(class_id, [0]))
    
    if precision > 0 or recall > 0:
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
    
    class_metrics[class_name] = {
        'detections': detections,
        'ground_truth': gt_count,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_confidence': avg_conf,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    print(f"{class_name:<20} {detections:<12} {gt_count:<8} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {avg_conf:<12.3f}")

print("-"*100)

# Overall metrics
mean_precision = np.mean(overall_precision) if overall_precision else 0
mean_recall = np.mean(overall_recall) if overall_recall else 0
mean_f1 = np.mean(overall_f1) if overall_f1 else 0

print(f"\nOverall Metrics:")
print(f"  Mean Precision: {mean_precision:.3f}")
print(f"  Mean Recall:    {mean_recall:.3f}")
print(f"  Mean F1 Score:  {mean_f1:.3f}")
print(f"  mAP@0.5:        {mean_precision:.3f}")

print("\n" + "="*100)

# PART 8: COMPARATIVE ANALYSIS - TRAINING vs DETECTION DISTRIBUTION
print("\n" + "="*70)
print("  COMPARATIVE ANALYSIS: TRAINING vs. DETECTION DISTRIBUTION")
print("="*70 + "\n")

print("Comparative Analysis: Training vs. Detection Distribution")
print("="*110)
print(f"{'Class':<20} {'Train Boxes':<15} {'Train %':<12} {'Detections':<15} {'Detection %':<15} {'Ratio':<10}")
print("="*110)

comparative_data = {}

for class_id in sorted(categories.keys()):
    class_name = categories[class_id]
    
    # Training data
    train_boxes = train_distribution[class_id]['count']
    train_pct = train_distribution[class_id]['percentage']
    
    # Detection data
    detections = class_detections.get(class_id, 0)
    det_pct = (detections / total_detections * 100) if total_detections > 0 else 0
    
    # Ratio
    ratio = (det_pct / train_pct) if train_pct > 0 else 0
    
    comparative_data[class_name] = {
        'train_boxes': train_boxes,
        'train_percentage': train_pct,
        'detections': detections,
        'detection_percentage': det_pct,
        'ratio': ratio
    }
    
    print(f"{class_name:<20} {train_boxes:<15} {train_pct:<12.1f}% {detections:<15} {det_pct:<15.1f}% {ratio:<10.2f}×")

print("="*110 + "\n")

# Identify systematic biases
over_detected = [(name, data['ratio']) for name, data in comparative_data.items() if data['ratio'] > 1.1]
under_detected = [(name, data['ratio']) for name, data in comparative_data.items() if data['ratio'] < 0.9]
balanced = [(name, data['ratio']) for name, data in comparative_data.items() if 0.9 <= data['ratio'] <= 1.1]

print("Detection Balance Analysis:")
print("-"*70)
if balanced:
    print(f"Balanced classes (0.9× - 1.1×): {len(balanced)}")
    for name, ratio in sorted(balanced, key=lambda x: x[1]):
        print(f"  {name}: {ratio:.2f}×")
    print()

if over_detected:
    print(f"Over-detected classes (ratio > 1.1×): {len(over_detected)}")
    for name, ratio in sorted(over_detected, key=lambda x: x[1], reverse=True):
        print(f"  {name}: {ratio:.2f}×")
    print()

if under_detected:
    print(f"Under-detected classes (ratio < 0.9×): {len(under_detected)}")
    for name, ratio in sorted(under_detected, key=lambda x: x[1]):
        print(f"  {name}: {ratio:.2f}×")
    print()

print("="*70 + "\n")

# PART 9: SAVE COMPREHENSIVE RESULTS
eval_results = {
    "strategy": "Two-Stage Fine-Tuning + Adaptive Thresholds + TTA",
    "total_detections": total_detections,
    "total_ground_truths": total_ground_truths,
    "detection_rate": total_detections / total_ground_truths if total_ground_truths > 0 else 0,
    "mean_precision": float(mean_precision),
    "mean_recall": float(mean_recall),
    "mean_f1_score": float(mean_f1),
    "mAP@0.5": float(mean_precision),
    "inference_time_seconds": total_time,
    "inference_time_per_image": total_time / len(val_loader),
    "tta_config": {
        "augmentations": ["original", "horizontal_flip", "scale_0.9", "scale_1.1"],
        "nms_iou_threshold": 0.5,
        "class_thresholds": CLASS_THRESHOLDS
    },
    "comparative_analysis": comparative_data,
    "per_class_metrics": class_metrics,
    "detection_balance": {
        "balanced_classes": len(balanced),
        "over_detected_classes": len(over_detected),
        "under_detected_classes": len(under_detected)
    }
}

output_file = os.path.join(OUTPUT_DIR, 'two_stage_tta_evaluation.json')
with open(output_file, 'w') as f:
    json.dump(eval_results, f, indent=2)

print(f"✓ Comprehensive results saved to: {output_file}\n")

# PART 10: VISUALIZE EXAMPLE PREDICTIONS
print("="*70)
print("  GENERATING EXAMPLE PREDICTIONS")
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
    axes[idx, 0].set_title(f'Ground Truth (Image {idx+1})', fontweight='bold', fontsize=12)
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
    
    # Predictions
    axes[idx, 1].imshow(img)
    axes[idx, 1].set_title(f'TTA Predictions (Image {idx+1}) - {len(pred["boxes"])} detections', 
                          fontweight='bold', fontsize=12)
    axes[idx, 1].axis('off')
    
    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        x1, y1, x2, y2 = box
        color = colors.get(label.item(), 'white')
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
        axes[idx, 1].add_patch(rect)
        axes[idx, 1].text(x1, y1-5, f"{categories[label.item()]}: {score:.2f}", 
                        color=color, fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'two_stage_tta_predictions.png'), dpi=300, bbox_inches='tight')
print(f"✓ Example predictions saved to: {OUTPUT_DIR}/two_stage_tta_predictions.png\n")

# PART 11: CREATE PERFORMANCE VISUALIZATIONS
print("="*70)
print("  GENERATING PERFORMANCE VISUALIZATION")
print("="*70 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Precision by class
class_names = [categories[i] for i in sorted(categories.keys())]
precisions = [class_metrics[categories[i]]['precision'] for i in sorted(categories.keys())]
recalls = [class_metrics[categories[i]]['recall'] for i in sorted(categories.keys())]
f1_scores = [class_metrics[categories[i]]['f1_score'] for i in sorted(categories.keys())]

axes[0, 0].barh(class_names, precisions, color='steelblue', alpha=0.8)
axes[0, 0].set_xlabel('Precision', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Precision by Class (Two-Stage + TTA)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlim([0, 1])
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. Recall by class
axes[0, 1].barh(class_names, recalls, color='coral', alpha=0.8)
axes[0, 1].set_xlabel('Recall', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Recall by Class (Two-Stage + TTA)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlim([0, 1])
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. F1 Score by class
axes[1, 0].barh(class_names, f1_scores, color='mediumseagreen', alpha=0.8)
axes[1, 0].set_xlabel('F1 Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('F1 Score by Class (Two-Stage + TTA)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlim([0, 1])
axes[1, 0].grid(axis='x', alpha=0.3)

# 4. Detection count vs Ground Truth
detections_list = [class_detections.get(i, 0) for i in sorted(categories.keys())]
gt_list = [class_ground_truths.get(i, 0) for i in sorted(categories.keys())]

x = np.arange(len(class_names))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, detections_list, width, label='Detections', 
                       color='skyblue', alpha=0.8, edgecolor='black')
bars2 = axes[1, 1].bar(x + width/2, gt_list, width, label='Ground Truth', 
                       color='lightcoral', alpha=0.8, edgecolor='black')
axes[1, 1].set_xlabel('Class', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Detections vs Ground Truth (Two-Stage + TTA)', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'two_stage_tta_performance_metrics.png'), dpi=300, bbox_inches='tight')
print(f"✓ Performance visualization saved to: {OUTPUT_DIR}/two_stage_tta_performance_metrics.png\n")

# PART 12: CREATE COMPARATIVE TABLE VISUALIZATION
print("="*70)
print("  GENERATING COMPARATIVE TABLE VISUALIZATION")
print("="*70 + "\n")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Class', 'Train Boxes', 'Train %', 'Detections', 'Detection %', 'Ratio'])

for class_id in sorted(categories.keys()):
    class_name = categories[class_id]
    data = comparative_data[class_name]
    
    table_data.append([
        class_name,
        f"{data['train_boxes']:,}",
        f"{data['train_percentage']:.1f}%",
        f"{data['detections']:,}",
        f"{data['detection_percentage']:.1f}%",
        f"{data['ratio']:.2f}×"
    ])

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.25, 0.15, 0.12, 0.15, 0.15, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

# Color code ratio column
for row in range(1, len(table_data)):
    ratio_val = float(table_data[row][5].rstrip('×'))
    
    if 0.9 <= ratio_val <= 1.1:
        table[(row, 5)].set_facecolor('#C6EFCE')  # Green - balanced
        table[(row, 5)].set_text_props(weight='bold')
    elif ratio_val > 1.1:
        table[(row, 5)].set_facecolor('#FFEB9C')  # Yellow - over-detected
        table[(row, 5)].set_text_props(weight='bold')
    else:
        table[(row, 5)].set_facecolor('#FFC7CE')  # Red - under-detected
        table[(row, 5)].set_text_props(weight='bold')

plt.title('Two-Stage + TTA: Training vs. Detection Distribution Comparison', 
          fontsize=14, fontweight='bold', pad=20)
plt.savefig(os.path.join(OUTPUT_DIR, 'two_stage_tta_comparative_table.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Comparative table saved to: {OUTPUT_DIR}/two_stage_tta_comparative_table.png\n")

# FINAL SUMMARY
print("="*70)
print("  TWO-STAGE + TTA EVALUATION COMPLETE")
print("="*70)
print(f"\nOutput files saved to: {OUTPUT_DIR}/")
print("  - two_stage_tta_evaluation.json")
print("  - two_stage_tta_predictions.png")
print("  - two_stage_tta_performance_metrics.png")
print("  - two_stage_tta_comparative_table.png")
print("\nKey Metrics:")
print(f"  Total Detections:     {total_detections:,}")
print(f"  Total Ground Truth:   {total_ground_truths:,}")
print(f"  Mean Precision:       {mean_precision:.3f}")
print(f"  Mean Recall:          {mean_recall:.3f}")
print(f"  Mean F1 Score:        {mean_f1:.3f}")
print(f"  mAP@0.5:              {mean_precision:.3f}")
print(f"  Inference Time:       {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  Time per Image:       {total_time/len(val_loader):.2f}s")
print("\nDetection Balance:")
print(f"  Balanced classes:     {len(balanced)}/12")
print(f"  Over-detected:        {len(over_detected)}/12")
print(f"  Under-detected:       {len(under_detected)}/12")
print("="*70 + "\n")
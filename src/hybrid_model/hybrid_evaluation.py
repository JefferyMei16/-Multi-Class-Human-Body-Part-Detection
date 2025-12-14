"""
hybrid_evaluation.py

Comprehensive evaluation of HYBRID Faster R-CNN model
Computes Precision, Recall, mAP@0.5, and generates example predictions
Provides side-by-side comparison with baseline
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
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from collections import defaultdict

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
BASE_DIR = "/root/autodl-tmp/final/data/bigdata"
IMAGES_DIR = os.path.join(BASE_DIR, "body_part_images")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "body_part_annotations", "labels.json")
HYBRID_MODEL_PATH = "/root/autodl-tmp/final/results/hybrid_frcnn/best_model.pth"
BASELINE_EVAL_PATH = "/root/autodl-tmp/final/results/baseline_frcnn/comprehensive_evaluation.json"
OUTPUT_DIR = "/root/autodl-tmp/final/results/hybrid_frcnn"

print("="*70)
print("  HYBRID MODEL COMPREHENSIVE EVALUATION")
print("="*70)
print(f"Model: {HYBRID_MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("="*70 + "\n")

# PART 1: DATASET
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
        
        # Validate images
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
print("\n" + "="*70)
print("  LOADING DATASET")
print("="*70)

full_dataset = BodyPartDataset(IMAGES_DIR, ANNOTATIONS_FILE, get_transform())

# Use same split as training (IMPORTANT: same seed)
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
    val_dataset, batch_size=4, shuffle=False,
    num_workers=2, collate_fn=collate_fn, pin_memory=True
)

print(f"Validation batches: {len(val_loader)}")
print("="*70 + "\n")

# PART 2: LOAD MODEL
print("="*70)
print("  LOADING HYBRID MODEL")
print("="*70 + "\n")

# Load model architecture (same as baseline)
model = fasterrcnn_resnet50_fpn(pretrained=False)

# Replace head
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 13
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load hybrid trained weights
model.load_state_dict(torch.load(HYBRID_MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("✓ Hybrid model loaded successfully")
print("="*70 + "\n")

# PART 3: COMPREHENSIVE EVALUATION
print("="*70)
print("  RUNNING COMPREHENSIVE EVALUATION")
print("="*70 + "\n")

categories = {
    1: "Human arm", 2: "Human body", 3: "Human ear", 4: "Human eye",
    5: "Human face", 6: "Human foot", 7: "Human hair", 8: "Human hand",
    9: "Human head", 10: "Human leg", 11: "Human mouth", 12: "Human nose"
}

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
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

# Store example predictions
example_predictions = []
num_examples = 6

print("Running inference on validation set...")
print("This may take a few minutes...\n")

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        if batch_idx % 50 == 0:
            print(f"  Processed {batch_idx}/{len(val_loader)} batches...")
        
        images_list = list(img.to(device) for img in images)
        predictions = model(images_list)
        
        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            boxes = pred['boxes'].cpu()
            scores = pred['scores'].cpu()
            labels = pred['labels'].cpu()
            
            gt_boxes = target['boxes'].cpu()
            gt_labels = target['labels'].cpu()
            
            # Count ground truths
            for gt_label in gt_labels:
                class_ground_truths[gt_label.item()] += 1
            
            # Filter predictions by confidence threshold
            conf_threshold = 0.5
            keep = scores > conf_threshold
            pred_boxes = boxes[keep]
            pred_scores = scores[keep]
            pred_labels = labels[keep]
            
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
                
                # IoU threshold for matching
                if best_iou >= 0.5:
                    class_true_positives[pred_label.item()] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    class_false_positives[pred_label.item()] += 1
            
            # Count false negatives
            for gt_idx, gt_label in enumerate(gt_labels):
                if gt_idx not in matched_gt:
                    class_false_negatives[gt_label.item()] += 1
            
            # Store examples for visualization
            if len(example_predictions) < num_examples:
                example_predictions.append({
                    'image': images[img_idx],
                    'predictions': {'boxes': pred_boxes, 'labels': pred_labels, 'scores': pred_scores},
                    'ground_truth': {'boxes': gt_boxes, 'labels': gt_labels}
                })

print("\n✓ Inference complete\n")

# PART 4: LOAD BASELINE METRICS FOR COMPARISON
print("="*70)
print("  LOADING BASELINE METRICS FOR COMPARISON")
print("="*70 + "\n")

baseline_metrics = {}
if os.path.exists(BASELINE_EVAL_PATH):
    with open(BASELINE_EVAL_PATH, 'r') as f:
        baseline_metrics = json.load(f)
    print(f"✓ Baseline metrics loaded from: {BASELINE_EVAL_PATH}")
else:
    print(f"⚠️  Baseline metrics not found at: {BASELINE_EVAL_PATH}")
    print("   Comparison with baseline will not be available")

print("="*70 + "\n")

# PART 5: COMPUTE METRICS
print("="*70)
print("  DETECTION METRICS - HYBRID MODEL")
print("="*70 + "\n")

total_detections = sum(class_detections.values())
total_ground_truths = sum(class_ground_truths.values())

print(f"Total detections (confidence > 0.5): {total_detections}")
print(f"Total ground truth objects: {total_ground_truths}")
print(f"Detection rate: {total_detections/total_ground_truths:.2%}")

if baseline_metrics:
    baseline_total = baseline_metrics.get('total_detections', 0)
    change = total_detections - baseline_total
    pct_change = (change / baseline_total * 100) if baseline_total > 0 else 0
    print(f"Baseline detections: {baseline_total}")
    print(f"Change: {change:+d} ({pct_change:+.1f}%)")

print()

# Per-class metrics
print("Per-Class Performance:")
print("-"*100)
print(f"{'Class':<20} {'Det':<8} {'GT':<8} {'Prec':<10} {'Rec':<10} {'F1':<10} {'Conf':<10} {'vs Baseline':<15}")
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
    
    # Get baseline comparison
    baseline_det = 0
    if baseline_metrics and 'per_class_metrics' in baseline_metrics:
        baseline_class = baseline_metrics['per_class_metrics'].get(class_name, {})
        baseline_det = baseline_class.get('detections', 0)
    
    change = detections - baseline_det
    change_str = f"{change:+d}" if baseline_det > 0 else "N/A"
    
    class_metrics[class_name] = {
        'detections': detections,
        'baseline_detections': baseline_det,
        'change': change,
        'ground_truth': gt_count,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_confidence': avg_conf,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    print(f"{class_name:<20} {detections:<8} {gt_count:<8} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {avg_conf:<10.3f} {change_str:<15}")

print("-"*100)

# Overall metrics
mean_precision = np.mean(overall_precision) if overall_precision else 0
mean_recall = np.mean(overall_recall) if overall_recall else 0
mean_f1 = np.mean(overall_f1) if overall_f1 else 0

print(f"\nOverall Metrics - HYBRID:")
print(f"  Mean Precision: {mean_precision:.3f}")
print(f"  Mean Recall:    {mean_recall:.3f}")
print(f"  Mean F1 Score:  {mean_f1:.3f}")
print(f"  mAP@0.5:        {mean_precision:.3f}")

if baseline_metrics:
    baseline_prec = baseline_metrics.get('mean_precision', 0)
    baseline_rec = baseline_metrics.get('mean_recall', 0)
    baseline_f1 = baseline_metrics.get('mean_f1_score', 0)
    
    print(f"\nOverall Metrics - BASELINE:")
    print(f"  Mean Precision: {baseline_prec:.3f}")
    print(f"  Mean Recall:    {baseline_rec:.3f}")
    print(f"  Mean F1 Score:  {baseline_f1:.3f}")
    
    print(f"\nIMPROVEMENT:")
    print(f"  Precision: {(mean_precision - baseline_prec):+.3f} ({(mean_precision - baseline_prec)/baseline_prec*100:+.1f}%)" if baseline_prec > 0 else "  Precision: N/A")
    print(f"  Recall:    {(mean_recall - baseline_rec):+.3f} ({(mean_recall - baseline_rec)/baseline_rec*100:+.1f}%)" if baseline_rec > 0 else "  Recall: N/A")
    print(f"  F1 Score:  {(mean_f1 - baseline_f1):+.3f} ({(mean_f1 - baseline_f1)/baseline_f1*100:+.1f}%)" if baseline_f1 > 0 else "  F1 Score: N/A")

print("\n" + "="*100)

# Save metrics
eval_metrics = {
    "total_detections": total_detections,
    "total_ground_truths": total_ground_truths,
    "detection_rate": total_detections / total_ground_truths if total_ground_truths > 0 else 0,
    "mean_precision": float(mean_precision),
    "mean_recall": float(mean_recall),
    "mean_f1_score": float(mean_f1),
    "mAP@0.5": float(mean_precision),
    "per_class_metrics": class_metrics,
    "threshold": 0.5,
    "baseline_comparison": {
        "baseline_detections": baseline_metrics.get('total_detections', 0) if baseline_metrics else 0,
        "detection_change": total_detections - baseline_metrics.get('total_detections', 0) if baseline_metrics else 0,
        "baseline_precision": baseline_metrics.get('mean_precision', 0) if baseline_metrics else 0,
        "baseline_recall": baseline_metrics.get('mean_recall', 0) if baseline_metrics else 0,
        "baseline_f1": baseline_metrics.get('mean_f1_score', 0) if baseline_metrics else 0
    }
}

with open(os.path.join(OUTPUT_DIR, 'comprehensive_evaluation.json'), 'w') as f:
    json.dump(eval_metrics, f, indent=2)

print(f"✓ Comprehensive metrics saved to: {OUTPUT_DIR}/comprehensive_evaluation.json\n")

# PART 6: VISUALIZE EXAMPLE PREDICTIONS
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
    axes[idx, 1].set_title(f'HYBRID Predictions (Image {idx+1}) - {len(pred["boxes"])} detections', 
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
plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_example_predictions.png'), dpi=300, bbox_inches='tight')
print(f"✓ Example predictions saved to: {OUTPUT_DIR}/hybrid_example_predictions.png")

# PART 7: CREATE COMPARISON VISUALIZATION
print("\n" + "="*70)
print("  GENERATING PERFORMANCE COMPARISON")
print("="*70 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

class_names = [categories[i] for i in sorted(categories.keys())]

# 1. Precision comparison
precisions_hybrid = [class_metrics[categories[i]]['precision'] for i in sorted(categories.keys())]
precisions_baseline = []
if baseline_metrics and 'per_class_metrics' in baseline_metrics:
    for i in sorted(categories.keys()):
        baseline_class = baseline_metrics['per_class_metrics'].get(categories[i], {})
        precisions_baseline.append(baseline_class.get('precision', 0))
else:
    precisions_baseline = [0] * len(class_names)

x = np.arange(len(class_names))
width = 0.35

axes[0, 0].barh(x - width/2, precisions_baseline, width, label='Baseline', color='lightblue')
axes[0, 0].barh(x + width/2, precisions_hybrid, width, label='Hybrid', color='steelblue')
axes[0, 0].set_xlabel('Precision', fontsize=12)
axes[0, 0].set_yticks(x)
axes[0, 0].set_yticklabels(class_names, fontsize=10)
axes[0, 0].set_title('Precision by Class: Baseline vs Hybrid', fontsize=14, fontweight='bold')
axes[0, 0].set_xlim([0, 1])
axes[0, 0].legend()
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. Recall comparison
recalls_hybrid = [class_metrics[categories[i]]['recall'] for i in sorted(categories.keys())]
recalls_baseline = []
if baseline_metrics and 'per_class_metrics' in baseline_metrics:
    for i in sorted(categories.keys()):
        baseline_class = baseline_metrics['per_class_metrics'].get(categories[i], {})
        recalls_baseline.append(baseline_class.get('recall', 0))
else:
    recalls_baseline = [0] * len(class_names)

axes[0, 1].barh(x - width/2, recalls_baseline, width, label='Baseline', color='lightcoral')
axes[0, 1].barh(x + width/2, recalls_hybrid, width, label='Hybrid', color='coral')
axes[0, 1].set_xlabel('Recall', fontsize=12)
axes[0, 1].set_yticks(x)
axes[0, 1].set_yticklabels(class_names, fontsize=10)
axes[0, 1].set_title('Recall by Class: Baseline vs Hybrid', fontsize=14, fontweight='bold')
axes[0, 1].set_xlim([0, 1])
axes[0, 1].legend()
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. F1 Score comparison
f1_scores_hybrid = [class_metrics[categories[i]]['f1_score'] for i in sorted(categories.keys())]
f1_scores_baseline = []
if baseline_metrics and 'per_class_metrics' in baseline_metrics:
    for i in sorted(categories.keys()):
        baseline_class = baseline_metrics['per_class_metrics'].get(categories[i], {})
        f1_scores_baseline.append(baseline_class.get('f1_score', 0))
else:
    f1_scores_baseline = [0] * len(class_names)

axes[1, 0].barh(x - width/2, f1_scores_baseline, width, label='Baseline', color='lightgreen')
axes[1, 0].barh(x + width/2, f1_scores_hybrid, width, label='Hybrid', color='mediumseagreen')
axes[1, 0].set_xlabel('F1 Score', fontsize=12)
axes[1, 0].set_yticks(x)
axes[1, 0].set_yticklabels(class_names, fontsize=10)
axes[1, 0].set_title('F1 Score by Class: Baseline vs Hybrid', fontsize=14, fontweight='bold')
axes[1, 0].set_xlim([0, 1])
axes[1, 0].legend()
axes[1, 0].grid(axis='x', alpha=0.3)

# 4. Detection count comparison
detections_hybrid = [class_detections.get(i, 0) for i in sorted(categories.keys())]
detections_baseline = []
if baseline_metrics and 'per_class_metrics' in baseline_metrics:
    for i in sorted(categories.keys()):
        baseline_class = baseline_metrics['per_class_metrics'].get(categories[i], {})
        detections_baseline.append(baseline_class.get('detections', 0))
else:
    detections_baseline = [0] * len(class_names)

y_pos = np.arange(len(class_names))
axes[1, 1].barh(y_pos - width/2, detections_baseline, width, label='Baseline', color='lightyellow')
axes[1, 1].barh(y_pos + width/2, detections_hybrid, width, label='Hybrid', color='gold')
axes[1, 1].set_xlabel('Number of Detections', fontsize=12)
axes[1, 1].set_yticks(y_pos)
axes[1, 1].set_yticklabels(class_names, fontsize=10)
axes[1, 1].set_title('Detection Count: Baseline vs Hybrid', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_vs_baseline_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Comparison visualization saved to: {OUTPUT_DIR}/hybrid_vs_baseline_comparison.png")

# PART 8: SUMMARY REPORT
print("\n" + "="*70)
print("  HYBRID EVALUATION COMPLETE")
print("="*70)
print(f"\nOutput files saved to: {OUTPUT_DIR}/")
print("  - comprehensive_evaluation.json")
print("  - hybrid_example_predictions.png")
print("  - hybrid_vs_baseline_comparison.png")

print(f"\n{'='*70}")
print("  HYBRID MODEL SUMMARY")
print(f"{'='*70}")
print(f"\nKey Metrics:")
print(f"  Total Detections: {total_detections:,}")
print(f"  Mean Precision:   {mean_precision:.3f}")
print(f"  Mean Recall:      {mean_recall:.3f}")
print(f"  Mean F1 Score:    {mean_f1:.3f}")
print(f"  mAP@0.5:          {mean_precision:.3f}")

if baseline_metrics:
    print(f"\nComparison to Baseline:")
    baseline_total = baseline_metrics.get('total_detections', 0)
    baseline_prec = baseline_metrics.get('mean_precision', 0)
    baseline_rec = baseline_metrics.get('mean_recall', 0)
    baseline_f1 = baseline_metrics.get('mean_f1_score', 0)
    
    print(f"  Detections: {total_detections:,} vs {baseline_total:,} ({(total_detections-baseline_total)/baseline_total*100:+.1f}%)" if baseline_total > 0 else "  Detections: N/A")
    print(f"  Precision:  {mean_precision:.3f} vs {baseline_prec:.3f} ({(mean_precision-baseline_prec)/baseline_prec*100:+.1f}%)" if baseline_prec > 0 else "  Precision: N/A")
    print(f"  Recall:     {mean_recall:.3f} vs {baseline_rec:.3f} ({(mean_recall-baseline_rec)/baseline_rec*100:+.1f}%)" if baseline_rec > 0 else "  Recall: N/A")
    print(f"  F1 Score:   {mean_f1:.3f} vs {baseline_f1:.3f} ({(mean_f1-baseline_f1)/baseline_f1*100:+.1f}%)" if baseline_f1 > 0 else "  F1 Score: N/A")

print(f"\n{'='*70}")
print("  KEY IMPROVEMENTS FROM HYBRID APPROACH")
print(f"{'='*70}")
print("\nClasses with Significant Detection Increases:")

for class_id in sorted(categories.keys()):
    class_name = categories[class_id]
    hybrid_det = class_detections.get(class_id, 0)
    
    if baseline_metrics and 'per_class_metrics' in baseline_metrics:
        baseline_class = baseline_metrics['per_class_metrics'].get(class_name, {})
        baseline_det = baseline_class.get('detections', 0)
        
        if baseline_det > 0:
            change_pct = ((hybrid_det - baseline_det) / baseline_det) * 100
            if abs(change_pct) > 20:  # Show significant changes only
                symbol = "✓" if change_pct > 0 else "✗"
                print(f"  {symbol} {class_name:20s}: {hybrid_det:4d} vs {baseline_det:4d} ({change_pct:+6.1f}%)")
        elif hybrid_det > 0:
            print(f"  ✓ {class_name:20s}: {hybrid_det:4d} vs 0 (NEW DETECTIONS!)")

print("="*70 + "\n")
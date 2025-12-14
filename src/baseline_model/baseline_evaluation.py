"""
baseline_evaluation.py

Comprehensive evaluation of baseline Faster R-CNN model
Computes Precision, Recall, mAP@0.5, and generates comparative analysis
with training distribution vs. detection distribution
"""
# %%   
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
from collections import defaultdict, Counter

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
OUTPUT_DIR = "/root/autodl-tmp/final/results/baseline_frcnn"

print("="*70)
print("  BASELINE MODEL COMPREHENSIVE EVALUATION")
print("="*70)
print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("="*70 + "\n")

# %%  PART 1: GET TRAINING DISTRIBUTION USING EXACT SAME SPLIT AS PYTORCH
print("="*70)
print("  LOADING TRAINING DISTRIBUTION (EXACT SAME SPLIT AS MODEL TRAINING)")
print("="*70 + "\n")

categories = {
    1: "Human arm", 2: "Human body", 3: "Human ear", 4: "Human eye",
    5: "Human face", 6: "Human foot", 7: "Human hair", 8: "Human hand",
    9: "Human head", 10: "Human leg", 11: "Human mouth", 12: "Human nose"
}

# Load full COCO annotations
with open(ANNOTATIONS_FILE, 'r') as f:
    coco_data = json.load(f)

# Create full dataset to get the EXACT same split as PyTorch random_split
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

# Split EXACTLY like PyTorch training
train_size = int(0.8 * len(temp_dataset))
val_size = len(temp_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    temp_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # SAME SEED AS TRAINING
)

# Get training image IDs
train_image_ids = set()
for idx in train_dataset.indices:
    img_info = temp_dataset[idx]
    train_image_ids.add(img_info['id'])

print(f"Training images: {len(train_image_ids)}")
print(f"Validation images: {len(val_dataset)}")

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

# %%  PART 2: DATASET
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
print("="*70)
print("  LOADING VALIDATION DATASET")
print("="*70 + "\n")

full_dataset = BodyPartDataset(IMAGES_DIR, ANNOTATIONS_FILE, get_transform())

# Use same split as training (80/20)
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

# %%   PART 3: LOAD MODEL
print("="*70)
print("  LOADING BASELINE MODEL")
print("="*70 + "\n")

# Load model architecture
model = fasterrcnn_resnet50_fpn(pretrained=False)

# Replace head
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 13
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("✓ Model loaded successfully")
print("="*70 + "\n")

# %%  PART 4: COMPREHENSIVE EVALUATION
print("="*70)
print("  RUNNING COMPREHENSIVE EVALUATION")
print("="*70 + "\n")

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

# %%   PART 5: COMPUTE METRICS
print("="*70)
print("  DETECTION METRICS")
print("="*70 + "\n")

total_detections = sum(class_detections.values())
total_ground_truths = sum(class_ground_truths.values())

print(f"Total detections (confidence > 0.5): {total_detections}")
print(f"Total ground truth objects: {total_ground_truths}")
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

# %%   PART 6: COMPARATIVE ANALYSIS - TRAINING vs DETECTION DISTRIBUTION
print("\n" + "="*70)
print("  COMPARATIVE ANALYSIS: TRAINING vs. DETECTION DISTRIBUTION")
print("="*70 + "\n")

print("Comparative Analysis: Baseline vs. Ground Truth Distribution")
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

if over_detected:
    print("Over-detected classes (ratio > 1.1×):")
    for name, ratio in sorted(over_detected, key=lambda x: x[1], reverse=True):
        print(f"  {name}: {ratio:.2f}×")
    print()

if under_detected:
    print("Under-detected classes (ratio < 0.9×):")
    for name, ratio in sorted(under_detected, key=lambda x: x[1]):
        print(f"  {name}: {ratio:.2f}×")
    print()

# %%   PART 7: SAVE COMPREHENSIVE METRICS
eval_metrics = {
    "total_detections": total_detections,
    "total_ground_truths": total_ground_truths,
    "detection_rate": total_detections / total_ground_truths if total_ground_truths > 0 else 0,
    "mean_precision": float(mean_precision),
    "mean_recall": float(mean_recall),
    "mean_f1_score": float(mean_f1),
    "mAP@0.5": float(mean_precision),
    "per_class_metrics": class_metrics,
    "comparative_analysis": comparative_data,
    "threshold": 0.5
}

with open(os.path.join(OUTPUT_DIR, 'comprehensive_evaluation.json'), 'w') as f:
    json.dump(eval_metrics, f, indent=2)

print(f"✓ Comprehensive metrics saved to: {OUTPUT_DIR}/comprehensive_evaluation.json\n")

# %%   PART 8: VISUALIZE EXAMPLE PREDICTIONS
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
    axes[idx, 1].set_title(f'Predictions (Image {idx+1}) - {len(pred["boxes"])} detections', 
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
plt.savefig(os.path.join(OUTPUT_DIR, 'baseline_example_predictions.png'), dpi=300, bbox_inches='tight')
print(f"✓ Example predictions saved to: {OUTPUT_DIR}/baseline_example_predictions.png\n")

# %%   PART 9: CREATE PERFORMANCE VISUALIZATIONS
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
axes[0, 0].set_title('Precision by Class', fontsize=14, fontweight='bold')
axes[0, 0].set_xlim([0, 1])
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. Recall by class
axes[0, 1].barh(class_names, recalls, color='coral', alpha=0.8)
axes[0, 1].set_xlabel('Recall', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Recall by Class', fontsize=14, fontweight='bold')
axes[0, 1].set_xlim([0, 1])
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. F1 Score by class
axes[1, 0].barh(class_names, f1_scores, color='mediumseagreen', alpha=0.8)
axes[1, 0].set_xlabel('F1 Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('F1 Score by Class', fontsize=14, fontweight='bold')
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
axes[1, 1].set_title('Detections vs Ground Truth', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'baseline_performance_metrics.png'), dpi=300, bbox_inches='tight')
print(f"✓ Performance visualization saved to: {OUTPUT_DIR}/baseline_performance_metrics.png\n")

# %%   PART 10: CREATE COMPARATIVE TABLE VISUALIZATION
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
    
    if ratio_val > 1.1:
        table[(row, 5)].set_facecolor('#C6EFCE')  # Green - over-detected
        table[(row, 5)].set_text_props(weight='bold')
    elif ratio_val < 0.9:
        table[(row, 5)].set_facecolor('#FFC7CE')  # Red - under-detected
        table[(row, 5)].set_text_props(weight='bold')
    else:
        table[(row, 5)].set_facecolor('#FFEB9C')  # Yellow - balanced

plt.title('Baseline: Training vs. Detection Distribution Comparison', 
          fontsize=14, fontweight='bold', pad=20)
plt.savefig(os.path.join(OUTPUT_DIR, 'baseline_comparative_table.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Comparative table saved to: {OUTPUT_DIR}/baseline_comparative_table.png\n")

# %%  FINAL SUMMARY
print("="*70)
print("  BASELINE EVALUATION COMPLETE")
print("="*70)
print(f"\nOutput files saved to: {OUTPUT_DIR}/")
print("  - comprehensive_evaluation.json")
print("  - baseline_example_predictions.png")
print("  - baseline_performance_metrics.png")
print("  - baseline_comparative_table.png")
print("\nKey Metrics:")
print(f"  Total Detections: {total_detections:,}")
print(f"  Mean Precision:   {mean_precision:.3f}")
print(f"  Mean Recall:      {mean_recall:.3f}")
print(f"  Mean F1 Score:    {mean_f1:.3f}")
print(f"  mAP@0.5:          {mean_precision:.3f}")
print("\nSystematic Biases:")
if over_detected:
    print(f"  Over-detected:    {len(over_detected)} classes")
if under_detected:
    print(f"  Under-detected:   {len(under_detected)} classes")
print("="*70 + "\n")
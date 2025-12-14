"""
baseline_faster_rcnn.py

Baseline Faster R-CNN model for body part detection
Uses torchvision's pre-trained Faster R-CNN with ResNet-50 FPN backbone
Minimal tuning to establish baseline performance
"""
# %% Cell  
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import time
from collections import defaultdict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
BASE_DIR = "/root/autodl-tmp/final/data/bigdata"
IMAGES_DIR = os.path.join(BASE_DIR, "body_part_images")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "body_part_annotations", "labels.json")
OUTPUT_DIR = "/root/autodl-tmp/final/results/baseline_frcnn"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("  BASELINE FASTER R-CNN FOR BODY PART DETECTION")
print("="*70)
print(f"Images directory: {IMAGES_DIR}")
print(f"Annotations file: {ANNOTATIONS_FILE}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*70 + "\n")

# PART 1: DATASET PREPARATION
class BodyPartDataset(Dataset):
    """Custom Dataset for COCO-format body part detection"""
    
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = coco_data['images']
        self.annotations = coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image_id
        self.img_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.img_to_anns[ann['image_id']].append(ann)
        
        print(f"Dataset loaded: {len(self.images)} images, "
              f"{len(self.annotations)} annotations, "
              f"{len(self.categories)} categories")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image info
        img_info = self.images[idx]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Get annotations for this image
        img_id = img_info['id']
        anns = self.img_to_anns[img_id]
        
        # Extract boxes and labels
        boxes = []
        labels = []
        areas = []
        
        for ann in anns:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': torch.zeros((len(anns),), dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transforms:
            img = self.transforms(img)
        
        return img, target

# Simple transforms (minimal preprocessing for baseline)
def get_transform(train=True):
    transforms_list = [transforms.ToTensor()]
    return transforms.Compose(transforms_list)

# Load dataset
print("\n" + "="*70)
print("  LOADING DATASET")
print("="*70)

full_dataset = BodyPartDataset(IMAGES_DIR, ANNOTATIONS_FILE, get_transform())

# Split into train/val (80/20 split)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"\nDataset split:")
print(f"  Training:   {len(train_dataset)} images")
print(f"  Validation: {len(val_dataset)} images")

# Create data loaders
def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print("="*70 + "\n")

# PART 2: MODEL ARCHITECTURE
print("="*70)
print("  MODEL ARCHITECTURE")
print("="*70 + "\n")

# Load pre-trained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one (12 body part classes + background)
num_classes = 13  # 12 classes + background
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model = model.to(device)

print("Architecture: Faster R-CNN with ResNet-50 FPN backbone")
print(f"Number of classes: {num_classes} (12 body parts + background)")
print(f"Input features to box predictor: {in_features}")
print(f"\nBackbone: ResNet-50 with Feature Pyramid Network (FPN)")
print(f"RPN: Region Proposal Network")
print(f"ROI Head: Box Predictor with classification and regression")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB (fp32)")

# Save architecture summary
arch_summary = {
    "model": "Faster R-CNN",
    "backbone": "ResNet-50 FPN",
    "num_classes": num_classes,
    "input_features": in_features,
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "pretrained": True,
    "modifications": "Replaced box predictor head for 12 body part classes"
}

with open(os.path.join(OUTPUT_DIR, "architecture_summary.json"), 'w') as f:
    json.dump(arch_summary, f, indent=2)

print("="*70 + "\n")

# PART 3: TRAINING SETUP
print("="*70)
print("  TRAINING CONFIGURATION")
print("="*70 + "\n")

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# Training parameters
num_epochs = 10
print(f"Optimizer: SGD")
print(f"  Learning rate: 0.005")
print(f"  Momentum: 0.9")
print(f"  Weight decay: 0.0005")
print(f"\nLR Scheduler: StepLR")
print(f"  Step size: 3 epochs")
print(f"  Gamma: 0.1")
print(f"\nTraining epochs: {num_epochs}")
print(f"Batch size: 4")
print("="*70 + "\n")

# PART 4: TRAINING LOOP
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    
    epoch_losses = []
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_losses.append(losses.item())
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(data_loader)}, "
                  f"Loss: {losses.item():.4f}")
    
    return np.mean(epoch_losses)

@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model on validation set"""
    model.train()  # Keep in train mode to get losses
    
    val_losses = []
    
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        val_losses.append(losses.item())
    
    return np.mean(val_losses)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'lr': [],
    'epoch_times': []
}

print("="*70)
print("  TRAINING STARTED")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"  EPOCH {epoch}/{num_epochs}")
    print(f"{'='*70}")
    
    # Train
    train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
    
    # Validate
    val_loss = evaluate(model, val_loader, device)
    
    # Update learning rate
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    epoch_time = time.time() - epoch_start
    
    # Store history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['lr'].append(current_lr)
    history['epoch_times'].append(epoch_time)
    
    print(f"\n{'='*70}")
    print(f"  EPOCH {epoch} SUMMARY")
    print(f"{'='*70}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  LR:         {current_lr:.6f}")
    print(f"  Time:       {epoch_time:.1f}s")
    print(f"{'='*70}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

print(f"\n{'='*70}")
print("  TRAINING COMPLETED")
print(f"{'='*70}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Total training time: {sum(history['epoch_times']):.1f}s")
print(f"{'='*70}\n")

# Save final model
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))

# Save training history
with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=2)

print("Models and history saved to:", OUTPUT_DIR)

# PART 5: LEARNING CURVES
print("\n" + "="*70)
print("  GENERATING LEARNING CURVES")
print("="*70 + "\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
epochs_range = range(1, num_epochs + 1)

axes[0].plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
axes[0].plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Learning rate schedule
axes[1].plot(epochs_range, history['lr'], 'g-o', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Learning Rate', fontsize=12)
axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'), dpi=300)
print(f"✓ Learning curves saved: {OUTPUT_DIR}/learning_curves.png")

# PART 6: INITIAL EVALUATION
print("\n" + "="*70)
print("  INITIAL EVALUATION METRICS")
print("="*70 + "\n")

model.eval()

# Simple evaluation: count detections and compute average confidence
total_detections = 0
class_detections = defaultdict(int)
avg_confidence = []

print("Running inference on validation set...\n")

with torch.no_grad():
    for images, targets in val_loader:
        images = list(img.to(device) for img in images)
        
        predictions = model(images)
        
        for pred in predictions:
            boxes = pred['boxes'].cpu()
            scores = pred['scores'].cpu()
            labels = pred['labels'].cpu()
            
            # Filter by confidence threshold
            keep = scores > 0.5
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            total_detections += len(boxes)
            avg_confidence.extend(scores.tolist())
            
            for label in labels:
                class_detections[label.item()] += 1

print("Baseline Evaluation Metrics:")
print("-"*70)
print(f"Total detections (confidence > 0.5): {total_detections}")
print(f"Average confidence: {np.mean(avg_confidence) if avg_confidence else 0:.3f}")
print(f"Detections per class:")

categories = {1: "Human arm", 2: "Human body", 3: "Human ear", 4: "Human eye",
              5: "Human face", 6: "Human foot", 7: "Human hair", 8: "Human hand",
              9: "Human head", 10: "Human leg", 11: "Human mouth", 12: "Human nose"}

for class_id in sorted(class_detections.keys()):
    class_name = categories.get(class_id, f"Class {class_id}")
    count = class_detections[class_id]
    pct = (count / total_detections * 100) if total_detections > 0 else 0
    print(f"  {class_name:20s}: {count:5d} ({pct:5.1f}%)")

# Save evaluation metrics
eval_metrics = {
    "total_detections": total_detections,
    "avg_confidence": float(np.mean(avg_confidence)) if avg_confidence else 0,
    "detections_per_class": dict(class_detections),
    "threshold": 0.5
}

with open(os.path.join(OUTPUT_DIR, 'initial_metrics.json'), 'w') as f:
    json.dump(eval_metrics, f, indent=2)

print("-"*70)
print(f"\n✓ Evaluation metrics saved: {OUTPUT_DIR}/initial_metrics.json")
print("\n" + "="*70)
print("  BASELINE MODEL TRAINING COMPLETE")
print("="*70)
print(f"\nOutput files saved to: {OUTPUT_DIR}/")
print("  - best_model.pth")
print("  - final_model.pth")
print("  - architecture_summary.json")
print("  - training_history.json")
print("  - learning_curves.png")
print("  - initial_metrics.json")
print("="*70 + "\n")
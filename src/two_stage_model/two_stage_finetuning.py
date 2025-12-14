"""
two_stage_finetuning.py

Two-Stage Fine-Tuning + Adaptive Thresholds + TTA

Two-Stage Training Strategy:
Stage 1 (3 epochs): Freeze backbone, oversample failing classes (arm, hair, hand, head)
                    Focus learning on detecting previously-failed classes
                    High learning rate on detection head only

Stage 2 (7 epochs): Unfreeze all, train on balanced full dataset
                    Refine all parameters together
                    Lower learning rate for stability

After Training: Evaluate with Adaptive Thresholds + TTA for best results

This combines ALL successful strategies:
- Two-stage training (this file)
- Adaptive thresholds (from Option 1)
- Test-time augmentation (from Option 3)

two_stage_finetuning.py
"""
# %%  
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from datetime import datetime
import time
from collections import defaultdict
import random

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
BASE_DIR = "/root/autodl-tmp/final/data/bigdata"
IMAGES_DIR = os.path.join(BASE_DIR, "body_part_images")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "body_part_annotations", "labels.json")
OUTPUT_DIR = "/root/autodl-tmp/final/results/two_stage_finetuning"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("  TWO-STAGE FINE-TUNING FOR BODY PART DETECTION")
print("="*70)
print(f"Images directory: {IMAGES_DIR}")
print(f"Annotations file: {ANNOTATIONS_FILE}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*70 + "\n")

# %%   PART 1: CONFIGURATION
"""
IMPROVEMENT: Two-Stage Training Strategy

Stage 1: Focus on Failing Classes
- Classes with catastrophic failure in baseline: arm (0), hair (0), head (0), hand (2)
- Oversample images containing these classes by 3x
- Freeze backbone to prevent catastrophic forgetting
- Train detection head only with high LR (0.01)
- 3 epochs to learn these difficult classes

Stage 2: Full Model Refinement
- Unfreeze all parameters
- Train on balanced full dataset
- Lower LR (0.005) for stability
- 7 epochs to refine everything together
"""

# Failing classes from baseline analysis
FAILING_CLASSES = [1, 7, 8, 9]  # Human arm, hair, hand, head
FAILING_CLASS_NAMES = ["Human arm", "Human hair", "Human hand", "Human head"]

# Training configuration
STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 7
TOTAL_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS
BATCH_SIZE = 4
OVERSAMPLE_RATE = 3  # Oversample failing classes by 3x in Stage 1

print("="*70)
print("  TWO-STAGE TRAINING CONFIGURATION")
print("="*70)
print(f"\nFailing Classes (will be oversampled in Stage 1):")
for class_id, class_name in zip(FAILING_CLASSES, FAILING_CLASS_NAMES):
    print(f"  - {class_name} (ID: {class_id})")

print(f"\nStage 1 (Focus on Failing Classes):")
print(f"  Epochs: {STAGE1_EPOCHS}")
print(f"  Strategy: Freeze backbone, oversample failing classes {OVERSAMPLE_RATE}x")
print(f"  Learning rate: 0.01 (high, detection head only)")
print(f"  Goal: Learn to detect arm, hair, hand, head")

print(f"\nStage 2 (Full Model Refinement):")
print(f"  Epochs: {STAGE2_EPOCHS}")
print(f"  Strategy: Unfreeze all, balanced training")
print(f"  Learning rate: 0.005 → 0.0005 (step decay)")
print(f"  Goal: Refine everything together without forgetting")

print(f"\nTotal training: {TOTAL_EPOCHS} epochs")
print("="*70 + "\n")

# %%   PART 2: CUSTOM SAMPLER FOR STAGE 1
class FailingClassSampler(Sampler):
    """
    IMPROVEMENT: Oversample images containing failing classes
    
    For Stage 1, we want the model to see more examples of arm, hair, hand, head.
    Images containing these classes are duplicated OVERSAMPLE_RATE times.
    This ensures balanced learning across all classes.
    """
    
    def __init__(self, dataset, failing_classes, oversample_rate=3):
        self.dataset = dataset
        self.failing_classes = failing_classes
        self.oversample_rate = oversample_rate
        
        # Build index list with oversampling
        self.indices = []
        
        oversample_count = 0
        normal_count = 0
        
        for idx in range(len(dataset)):
            # Get sample (this triggers __getitem__ but we need it to check labels)
            sample = dataset[idx]
            labels = sample[1]['labels']
            
            # Check if contains any failing class
            contains_failing = any(label.item() in failing_classes for label in labels)
            
            if contains_failing:
                # Oversample this image
                self.indices.extend([idx] * oversample_rate)
                oversample_count += 1
            else:
                # Regular sampling
                self.indices.append(idx)
                normal_count += 1
        
        print(f"\n  FailingClassSampler Statistics:")
        print(f"    Images with failing classes: {oversample_count}")
        print(f"    Normal images: {normal_count}")
        print(f"    Total samples per epoch: {len(self.indices)}")
        print(f"    Effective oversampling: {oversample_count * oversample_rate / len(dataset):.1%} of dataset")
    
    def __iter__(self):
        # Shuffle indices for each epoch
        shuffled = torch.randperm(len(self.indices)).tolist()
        return iter([self.indices[i] for i in shuffled])
    
    def __len__(self):
        return len(self.indices)

# %%   PART 3: DATASET
class BodyPartDataset(Dataset):
    """Dataset for COCO-format body part detection"""
    
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
        print("\nValidating images...")
        
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
        
        print(f"Dataset: {len(self.valid_images)} valid images, {len(self.categories)} categories")
    
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
    return T.Compose([T.ToTensor()])

# %%   PART 4: LOAD DATASET
print("\n" + "="*70)
print("  LOADING DATASET")
print("="*70)

full_dataset = BodyPartDataset(IMAGES_DIR, ANNOTATIONS_FILE, get_transform())

# Split 80/20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"\nDataset split:")
print(f"  Training:   {len(train_dataset)} images")
print(f"  Validation: {len(val_dataset)} images")

def collate_fn(batch):
    return tuple(zip(*batch))

# Create Stage 1 sampler (with oversampling)
stage1_sampler = FailingClassSampler(
    train_dataset, 
    FAILING_CLASSES, 
    oversample_rate=OVERSAMPLE_RATE
)

stage1_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler=stage1_sampler,
    num_workers=2, 
    collate_fn=collate_fn, 
    pin_memory=True
)

# Create Stage 2 loader (normal, no oversampling)
stage2_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=2, 
    collate_fn=collate_fn, 
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=2, 
    collate_fn=collate_fn, 
    pin_memory=True
)

print(f"\nDataLoaders created:")
print(f"  Stage 1: {len(stage1_loader)} batches (with oversampling)")
print(f"  Stage 2: {len(stage2_loader)} batches (normal)")
print(f"  Validation: {len(val_loader)} batches")
print("="*70 + "\n")

# %%   PART 5: MODEL ARCHITECTURE
print("="*70)
print("  MODEL ARCHITECTURE")
print("="*70 + "\n")

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace box predictor
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 13
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model = model.to(device)

print("Architecture: Faster R-CNN ResNet-50 FPN")
print(f"  Number of classes: {num_classes}")
print(f"  Pretrained: COCO weights")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

arch_summary = {
    "model": "Faster R-CNN (Two-Stage Fine-Tuned)",
    "backbone": "ResNet-50 FPN",
    "training_strategy": "Two-stage fine-tuning",
    "stage1": {
        "epochs": STAGE1_EPOCHS,
        "strategy": "Freeze backbone, oversample failing classes",
        "failing_classes": FAILING_CLASS_NAMES,
        "oversample_rate": OVERSAMPLE_RATE,
        "lr": 0.01
    },
    "stage2": {
        "epochs": STAGE2_EPOCHS,
        "strategy": "Unfreeze all, balanced training",
        "lr": "0.005 → 0.0005"
    },
    "num_classes": num_classes,
    "total_parameters": total_params,
    "trainable_parameters": trainable_params
}

with open(os.path.join(OUTPUT_DIR, "architecture_summary.json"), 'w') as f:
    json.dump(arch_summary, f, indent=2)

print("="*70 + "\n")

# %%   PART 6: TRAINING FUNCTIONS
def train_one_epoch(model, optimizer, data_loader, device, epoch, stage):
    """Train for one epoch"""
    model.train()
    epoch_losses = []
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        try:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not torch.isfinite(losses):
                continue
            
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            epoch_losses.append(losses.item())
            
            if batch_idx % 100 == 0:
                print(f"  Stage {stage}, Epoch {epoch}, Batch {batch_idx}/{len(data_loader)}, "
                      f"Loss: {losses.item():.4f}")
        
        except Exception as e:
            continue
    
    return np.mean(epoch_losses) if epoch_losses else 0.0

@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model"""
    model.train()
    val_losses = []
    
    for images, targets in data_loader:
        try:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if torch.isfinite(losses):
                val_losses.append(losses.item())
        except:
            continue
    
    return np.mean(val_losses) if val_losses else 0.0

# %%   PART 7: STAGE 1 TRAINING
print("="*70)
print("  STAGE 1: FOCUS ON FAILING CLASSES")
print("="*70)
print("  Strategy: Freeze backbone, train detection head only")
print("  Dataset: Oversample failing classes (arm, hair, hand, head)")
print("="*70 + "\n")

# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

print("✓ Backbone frozen")

# Count trainable parameters
stage1_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters (Stage 1): {stage1_trainable:,}")
print(f"  Frozen parameters: {total_params - stage1_trainable:,}\n")

# Stage 1 optimizer (high LR, detection head only)
stage1_optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.01,  # High LR for fast adaptation
    momentum=0.9,
    weight_decay=0.0005
)

# Training history
history = {
    'stage1_train_loss': [],
    'stage1_val_loss': [],
    'stage1_lr': [],
    'stage2_train_loss': [],
    'stage2_val_loss': [],
    'stage2_lr': [],
    'epoch_times': []
}

best_val_loss = float('inf')

print("Starting Stage 1 training...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

for epoch in range(1, STAGE1_EPOCHS + 1):
    epoch_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"  STAGE 1 - EPOCH {epoch}/{STAGE1_EPOCHS}")
    print(f"{'='*70}")
    
    # Train
    train_loss = train_one_epoch(model, stage1_optimizer, stage1_loader, device, epoch, stage=1)
    
    # Validate
    val_loss = evaluate(model, val_loader, device)
    
    current_lr = stage1_optimizer.param_groups[0]['lr']
    epoch_time = time.time() - epoch_start
    
    # Store history
    history['stage1_train_loss'].append(train_loss)
    history['stage1_val_loss'].append(val_loss)
    history['stage1_lr'].append(current_lr)
    history['epoch_times'].append(epoch_time)
    
    print(f"\n{'='*70}")
    print(f"  STAGE 1 - EPOCH {epoch} SUMMARY")
    print(f"{'='*70}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  LR:         {current_lr:.6f}")
    print(f"  Time:       {epoch_time:.1f}s")
    print(f"{'='*70}")
    
    # Save best model
    if val_loss < best_val_loss and val_loss > 0:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'stage1_best_model.pth'))
        print(f"  ✓ Stage 1 best model saved (val_loss: {val_loss:.4f})")

print(f"\n{'='*70}")
print("  STAGE 1 COMPLETED")
print(f"{'='*70}")
print(f"Best Stage 1 val loss: {best_val_loss:.4f}")
print("="*70 + "\n")

# %%   PART 8: STAGE 2 TRAINING
print("="*70)
print("  STAGE 2: FULL MODEL REFINEMENT")
print("="*70)
print("  Strategy: Unfreeze all parameters, balanced training")
print("  Dataset: Normal (no oversampling)")
print("="*70 + "\n")

# Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

print("✓ All parameters unfrozen")

stage2_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters (Stage 2): {stage2_trainable:,}\n")

# Stage 2 optimizer (lower LR for stability)
stage2_optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,  # Lower LR than Stage 1
    momentum=0.9,
    weight_decay=0.0005
)

# LR scheduler for Stage 2
stage2_scheduler = torch.optim.lr_scheduler.StepLR(
    stage2_optimizer,
    step_size=3,
    gamma=0.1
)

print("Starting Stage 2 training...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

for epoch in range(1, STAGE2_EPOCHS + 1):
    epoch_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"  STAGE 2 - EPOCH {epoch}/{STAGE2_EPOCHS}")
    print(f"{'='*70}")
    
    # Train
    train_loss = train_one_epoch(model, stage2_optimizer, stage2_loader, device, epoch, stage=2)
    
    # Validate
    val_loss = evaluate(model, val_loader, device)
    
    # Update LR
    stage2_scheduler.step()
    
    current_lr = stage2_optimizer.param_groups[0]['lr']
    epoch_time = time.time() - epoch_start
    
    # Store history
    history['stage2_train_loss'].append(train_loss)
    history['stage2_val_loss'].append(val_loss)
    history['stage2_lr'].append(current_lr)
    history['epoch_times'].append(epoch_time)
    
    print(f"\n{'='*70}")
    print(f"  STAGE 2 - EPOCH {epoch} SUMMARY")
    print(f"{'='*70}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  LR:         {current_lr:.6f}")
    print(f"  Time:       {epoch_time:.1f}s")
    print(f"{'='*70}")
    
    # Save best model
    if val_loss < best_val_loss and val_loss > 0:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

print(f"\n{'='*70}")
print("  STAGE 2 COMPLETED")
print(f"{'='*70}")
print(f"Best overall val loss: {best_val_loss:.4f}")
print("="*70 + "\n")

# Save final model
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))

# Save training history
with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=2)

# %%   PART 9: LEARNING CURVES
print("="*70)
print("  GENERATING LEARNING CURVES")
print("="*70 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Combine losses
all_train_loss = history['stage1_train_loss'] + history['stage2_train_loss']
all_val_loss = history['stage1_val_loss'] + history['stage2_val_loss']
all_lr = history['stage1_lr'] + history['stage2_lr']
epochs_range = range(1, TOTAL_EPOCHS + 1)

# Loss curves
axes[0, 0].plot(epochs_range, all_train_loss, 'b-o', label='Train Loss', linewidth=2)
axes[0, 0].plot(epochs_range, all_val_loss, 'r-o', label='Val Loss', linewidth=2)
axes[0, 0].axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', alpha=0.5, label='Stage 1→2')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Two-Stage Training Loss', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# LR schedule
axes[0, 1].plot(epochs_range, all_lr, 'g-o', linewidth=2)
axes[0, 1].axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', alpha=0.5, label='Stage 1→2')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Stage 1 detail
stage1_epochs = range(1, STAGE1_EPOCHS + 1)
axes[1, 0].plot(stage1_epochs, history['stage1_train_loss'], 'b-o', label='Train', linewidth=2)
axes[1, 0].plot(stage1_epochs, history['stage1_val_loss'], 'r-o', label='Val', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Loss', fontsize=12)
axes[1, 0].set_title('Stage 1: Focus on Failing Classes', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Stage 2 detail
stage2_epochs = range(STAGE1_EPOCHS + 1, TOTAL_EPOCHS + 1)
axes[1, 1].plot(stage2_epochs, history['stage2_train_loss'], 'b-o', label='Train', linewidth=2)
axes[1, 1].plot(stage2_epochs, history['stage2_val_loss'], 'r-o', label='Val', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Loss', fontsize=12)
axes[1, 1].set_title('Stage 2: Full Model Refinement', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'), dpi=300)
print("✓ Learning curves saved")

# %%   PART 10: SUMMARY
print("\n" + "="*70)
print("  TWO-STAGE TRAINING COMPLETE")
print("="*70)
print(f"\nOutput files saved to: {OUTPUT_DIR}/")
print("  - best_model.pth (best overall model)")
print("  - stage1_best_model.pth (best from stage 1)")
print("  - final_model.pth (after all training)")
print("  - architecture_summary.json")
print("  - training_history.json")
print("  - learning_curves.png")

print(f"\nTraining Summary:")
print(f"  Total epochs: {TOTAL_EPOCHS}")
print(f"  Stage 1: {STAGE1_EPOCHS} epochs (failing classes focus)")
print(f"  Stage 2: {STAGE2_EPOCHS} epochs (full refinement)")
print(f"  Best validation loss: {best_val_loss:.4f}")
print(f"  Total training time: {sum(history['epoch_times']):.1f}s ({sum(history['epoch_times'])/60:.1f} min)")

print(f"\n{'='*70}")
print("  NEXT STEP: EVALUATE WITH ADAPTIVE THRESHOLDS + TTA")
print(f"{'='*70}")
print("\nRun the evaluation script to test this model with:")
print("  1. Adaptive class-specific thresholds")
print("  2. Test-time augmentation")
print("  3. Compare against baseline and previous approaches")
print("="*70 + "\n")
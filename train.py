"""
IMPROVED Segmentation Training Script for Offroad Desert Terrain
================================================================
Key Improvements:
1. Strong data augmentation for desert terrain (color jitter, brightness, contrast, blur)
2. Combined Dice + Weighted CrossEntropy loss for better class balance
3. Mixed precision training (AMP) for memory efficiency on RTX 4050
4. CosineAnnealingLR scheduler with warmup for better convergence
5. AdamW optimizer with weight decay for better generalization
6. Per-class IoU tracking during training
7. Class weight calculation based on pixel frequency
8. Early stopping to prevent overfitting
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import os
import torchvision
from tqdm import tqdm
import random
from torch.cuda.amp import autocast, GradScaler

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# AUGMENTATION TRANSFORMS FOR DESERT TERRAIN
# ============================================================================

class RandomColorJitter:
    """Apply random color jittering to simulate different lighting conditions in desert."""
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, img):
        if random.random() > 0.5:
            img = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )(img)
        return img


class RandomGammaCorrection:
    """Apply random gamma correction to simulate different exposure levels."""
    def __init__(self, gamma_range=(0.7, 1.5)):
        self.gamma_range = gamma_range
    
    def __call__(self, img):
        if random.random() > 0.5:
            gamma = random.uniform(*self.gamma_range)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.power(img_array, gamma)
            img = Image.fromarray((img_array * 255).astype(np.uint8))
        return img


class RandomGaussianBlur:
    """Apply random Gaussian blur to simulate dust/haze in desert."""
    def __init__(self, kernel_size=5, sigma_range=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
    
    def __call__(self, img):
        if random.random() > 0.7:  # 30% chance
            sigma = random.uniform(*self.sigma_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class RandomShadow:
    """Add random shadow effects to simulate varying sun positions."""
    def __init__(self, shadow_intensity_range=(0.3, 0.7)):
        self.shadow_intensity_range = shadow_intensity_range
    
    def __call__(self, img):
        if random.random() > 0.6:  # 40% chance
            img_array = np.array(img).astype(np.float32)
            h, w = img_array.shape[:2]
            
            # Create random shadow mask
            shadow_intensity = random.uniform(*self.shadow_intensity_range)
            
            # Random shadow direction (top, bottom, left, right)
            direction = random.choice(['top', 'bottom', 'left', 'right'])
            
            if direction in ['top', 'bottom']:
                gradient = np.linspace(1, shadow_intensity, h)[:, None]
                if direction == 'bottom':
                    gradient = gradient[::-1]
            else:
                gradient = np.linspace(1, shadow_intensity, w)[None, :]
                if direction == 'right':
                    gradient = gradient[:, ::-1]
            
            img_array = img_array * gradient[:, :, None]
            img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        return img


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)

# Class names for better logging
class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset with Synchronized Augmentation
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.augment = augment
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        # Apply synchronized geometric augmentations (must be same for image and mask)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Random rotation (small angles for offroad terrain)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)

        # Apply color augmentations (only to image, not mask)
        if self.augment:
            image = RandomColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)(image)
            image = RandomGammaCorrection(gamma_range=(0.7, 1.5))(image)
            image = RandomGaussianBlur()(image)
            image = RandomShadow()(image)

        # Apply standard transforms
        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.decoder(x)


# ============================================================================
# COMBINED LOSS: Dice + Weighted CrossEntropy
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for better handling of class imbalance."""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Dice + Weighted CrossEntropy Loss for robust training."""
    def __init__(self, class_weights=None, dice_weight=0.5, ce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce


def calculate_class_weights(data_loader, device, num_classes=10):
    """
    Calculate class weights based on inverse frequency.
    This helps balance the loss for rare classes (like logs, rocks).
    """
    print("Calculating class weights from training data...")
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    
    for _, labels in tqdm(data_loader, desc="Computing weights", leave=False):
        labels = labels.squeeze(1).long()
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum().item()
    
    # Inverse frequency weighting
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (class_counts + 1e-6)  # Avoid division by zero
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("\nClass weights:")
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        print(f"  {name:<20}: {weight:.4f} (pixels: {int(class_counts[i])})")
    
    return class_weights.to(device)


# ============================================================================
# Metrics with Per-Class IoU
# ============================================================================

def compute_iou_per_class(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return both mean and per-class IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    """Evaluate all metrics on a dataset including per-class IoU."""
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_class_ious = []

    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Mixed precision inference
            with autocast():
                output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(output.to(device))
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            iou, class_iou = compute_iou_per_class(outputs, labels, num_classes=num_classes)
            dice = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_ious.append(class_iou)

    model.train()
    
    # Average per-class IoUs
    avg_class_iou = np.nanmean(all_class_ious, axis=0)
    
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies), avg_class_iou


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train', linewidth=2)
    plt.plot(history['val_loss'], label='val', linewidth=2)
    plt.title('Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='train', linewidth=2)
    plt.plot(history['val_iou'], label='val', linewidth=2)
    plt.title('IoU (Intersection over Union)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Dice and Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='train', linewidth=2)
    plt.plot(history['val_dice'], label='val', linewidth=2)
    plt.title('Dice Coefficient', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train', linewidth=2)
    plt.plot(history['val_pixel_acc'], label='val', linewidth=2)
    plt.title('Pixel Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Learning Rate
    if 'learning_rate' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['learning_rate'], linewidth=2, color='orange')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 4: Per-Class IoU Evolution
    if 'val_class_iou' in history and len(history['val_class_iou']) > 0:
        plt.figure(figsize=(14, 6))
        
        # Transpose to get time series for each class
        class_iou_over_time = np.array(history['val_class_iou']).T
        
        for i, class_iou in enumerate(class_iou_over_time):
            if not np.all(np.isnan(class_iou)):
                plt.plot(class_iou, label=class_names[i], linewidth=2, alpha=0.8)
        
        plt.title('Per-Class IoU Evolution', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend(loc='best', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved training plots to {output_dir}/")


def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    filepath = os.path.join(output_dir, 'training_history.txt')
    
    with open(filepath, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc', 'LR']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<10}\n".format(*headers))
        f.write("-" * 100 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            lr = history['learning_rate'][i] if 'learning_rate' in history else 0
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<10.6f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i],
                lr
            ))
        
        # Add per-class IoU for best epoch
        if 'val_class_iou' in history and len(history['val_class_iou']) > 0:
            best_epoch_idx = np.argmax(history['val_iou'])
            best_class_iou = history['val_class_iou'][best_epoch_idx]
            
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Per-Class IoU at Best Epoch ({best_epoch_idx + 1}):\n")
            f.write("=" * 50 + "\n")
            for i, (name, iou) in enumerate(zip(class_names, best_class_iou)):
                iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
                f.write(f"  {name:<20}: {iou_str}\n")

    print(f"Saved training history to {filepath}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Hyperparameters (optimized for RTX 4050 6GB)
    batch_size = 2  # Keep at 2 to avoid OOM on 6GB VRAM
    w = 672
    h = 378
    lr = 3e-4  # Higher initial LR for AdamW
    n_epochs = 25  # More epochs with early stopping
    warmup_epochs = 2  # Warmup for stable training
    patience = 5  # Early stopping patience
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {h}x{w}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Early stopping patience: {patience}")

    # Output directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats_improved')
    os.makedirs(output_dir, exist_ok=True)

    # Standard transforms (no augmentation here - done in dataset)
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

    # Dataset paths (relative to script location)
    data_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')

    # Create datasets WITH augmentation for training
    print("\nCreating datasets...")
    trainset = MaskDataset(data_dir=data_dir, transform=transform, 
                           mask_transform=mask_transform, augment=True)
    
    # Set num_workers based on platform (Windows has issues with multiprocessing)
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    print(f"Using num_workers={num_workers} for DataLoader")
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True if num_workers == 0 else False)

    valset = MaskDataset(data_dir=val_dir, transform=transform, 
                         mask_transform=mask_transform, augment=False)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True if num_workers == 0 else False)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # Calculate class weights from training data
    class_weights = calculate_class_weights(train_loader, device, num_classes=n_classes)

    # Load DINOv2 backbone
    print("\nLoading DINOv2 backbone...")
    BACKBONE_SIZE = "base"  # Upgraded backbone
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    # backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    # backbone_model.eval()
    # backbone_model.to(device)
    # print("Backbone loaded successfully!")
    
    backbone_model = torch.hub.load(
    repo_or_dir="facebookresearch/dinov2",
    model=backbone_name
    )

    backbone_model.eval()
    backbone_model.to(device)

# 🔒 Freeze backbone parameters
    for param in backbone_model.parameters():
        param.requires_grad = False

    print("Backbone loaded successfully and frozen!")

    # Get embedding dimension from backbone
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")
    print(f"Patch tokens shape: {output.shape}")

    # Create segmentation head
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier = classifier.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Loss function: Combined Dice + Weighted CrossEntropy
    loss_fct = CombinedLoss(class_weights=class_weights, dice_weight=0.6, ce_weight=0.4)
    print("\nUsing Combined Loss: 50% Dice + 50% Weighted CrossEntropy")

    # Optimizer: AdamW with weight decay for better generalization
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    print(f"Optimizer: AdamW (lr={lr}, weight_decay=0.01)")

    # Learning rate scheduler: Cosine Annealing with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Scheduler: Cosine Annealing with {warmup_epochs} warmup epochs")

    # Mixed precision training scaler
    scaler = GradScaler()
    print("Mixed Precision Training: Enabled (AMP)")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
        'train_pixel_acc': [],
        'val_pixel_acc': [],
        'learning_rate': [],
        'val_class_iou': []
    }

    # Early stopping
    best_val_iou = 0.0
    epochs_without_improvement = 0
    best_model_state = None

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training with IMPROVEMENTS:")
    print("  ✓ Strong data augmentation (color jitter, gamma, blur, shadow)")
    print("  ✓ Combined Dice + Weighted CE loss")
    print("  ✓ AdamW optimizer with weight decay")
    print("  ✓ Cosine annealing LR scheduler with warmup")
    print("  ✓ Mixed precision training (AMP)")
    print("  ✓ Per-class IoU tracking")
    print("  ✓ Early stopping")
    print("=" * 80 + "\n")

    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        current_lr = optimizer.param_groups[0]['lr']
        
        # Training phase
        classifier.train()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", 
                          leave=False, unit="batch")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # Mixed precision training
            with autocast():
                with torch.no_grad():
                    output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

                logits = classifier(output.to(device))
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                labels = labels.squeeze(dim=1).long()
                loss = loss_fct(outputs, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")

        # Validation phase
        classifier.eval()
        val_losses = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", 
                        leave=False, unit="batch")
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)

                with autocast():
                    output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                    logits = classifier(output.to(device))
                    outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                    labels = labels.squeeze(dim=1).long()
                    loss = loss_fct(outputs, labels)
                    
                val_losses.append(loss.item())
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Calculate metrics including per-class IoU
        train_iou, train_dice, train_pixel_acc, _ = evaluate_metrics(
            classifier, backbone_model, train_loader, device, num_classes=n_classes
        )
        val_iou, val_dice, val_pixel_acc, val_class_iou = evaluate_metrics(
            classifier, backbone_model, val_loader, device, num_classes=n_classes
        )

        # Store history
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_pixel_acc'].append(val_pixel_acc)
        history['learning_rate'].append(current_lr)
        history['val_class_iou'].append(val_class_iou)

        # Update learning rate
        scheduler.step()

        # Early stopping check
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_without_improvement = 0
            best_model_state = classifier.state_dict().copy()
            
            # Save best model
            best_model_path = os.path.join(script_dir, "segmentation_head_best.pth")
            torch.save(best_model_state, best_model_path)
        else:
            epochs_without_improvement += 1

        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix(
            train_loss=f"{epoch_train_loss:.3f}",
            val_loss=f"{epoch_val_loss:.3f}",
            val_iou=f"{val_iou:.3f}",
            best_iou=f"{best_val_iou:.3f}",
            patience=f"{epochs_without_improvement}/{patience}"
        )

        # Print per-class IoU every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n  Per-class IoU at epoch {epoch+1}:")
            for i, (name, iou) in enumerate(zip(class_names, val_class_iou)):
                iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
                print(f"    {name:<20}: {iou_str}")
            print()

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n Early stopping triggered! No improvement for {patience} epochs.")
            print(f"  Best Val IoU: {best_val_iou:.4f}")
            break

    # Load best model for final evaluation
    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)
        print("\nLoaded best model weights for final evaluation")

    # Save plots and history
    print("\nSaving training curves and history...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    # Save final model
    final_model_path = os.path.join(script_dir, "segmentation_head.pth")
    torch.save(classifier.state_dict(), final_model_path)
    print(f"Saved final model to '{final_model_path}'")
    print(f"Saved best model to '{best_model_path}'")

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Best Val IoU:      {best_val_iou:.4f} (Epoch {np.argmax(history['val_iou']) + 1})")
    print(f"Final Val Loss:    {history['val_loss'][-1]:.4f}")
    print(f"Final Val IoU:     {history['val_iou'][-1]:.4f}")
    print(f"Final Val Dice:    {history['val_dice'][-1]:.4f}")
    print(f"Final Val Accuracy:{history['val_pixel_acc'][-1]:.4f}")
    
    # Show final per-class IoU
    final_class_iou = history['val_class_iou'][-1]
    print("\nFinal Per-Class IoU:")
    for i, (name, iou) in enumerate(zip(class_names, final_class_iou)):
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        print(f"  {name:<20}: {iou_str}")
    
    print("=" * 80)
    print("\nTraining complete! 🎉")
    print(f"\nAll outputs saved to:")
    print(f"  - {output_dir}/")
    print(f"  - {final_model_path}")
    print(f"  - {best_model_path}")


if __name__ == "__main__":
    main()

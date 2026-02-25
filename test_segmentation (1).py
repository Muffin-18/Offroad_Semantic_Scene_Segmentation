"""
Test/Inference Script for Offroad Desert Terrain Segmentation
===============================================================
Performs inference on unseen images using the trained model.
Works with or without ground truth masks.
Generates colored segmentation masks and overlays.

Updated to match train_segmentation_improved.py structure
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm
from torch import amp

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Class Definitions and Color Mapping
# ============================================================================

# Must match training script EXACTLY
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

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# Color palette for visualization (RGB format) - visually distinct colors
color_palette = np.array([
    [70, 70, 70],      # Background - Dark Gray
    [34, 139, 34],     # Trees - Forest Green
    [0, 255, 127],     # Lush Bushes - Spring Green
    [255, 215, 0],     # Dry Grass - Gold
    [210, 105, 30],    # Dry Bushes - Chocolate
    [128, 128, 0],     # Ground Clutter - Olive
    [139, 69, 19],     # Logs - Saddle Brown
    [169, 169, 169],   # Rocks - Gray
    [222, 184, 135],   # Landscape - Burlywood
    [135, 206, 235],   # Sky - Sky Blue
], dtype=np.uint8)


# ============================================================================
# Utility Functions
# ============================================================================

def denormalize_image(img_tensor):
    """Denormalize image tensor for visualization."""
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask (H, W) to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


def create_overlay(image, mask, alpha=0.5):
    """Create an overlay of the mask on the original image."""
    # Ensure image is in [0, 1] range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Convert mask to color
    mask_color = mask_to_color(mask)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)
    return overlay


# ============================================================================
# Dataset for Inference (with optional ground truth)
# ============================================================================

class InferenceDataset(Dataset):
    """
    Dataset for inference that handles cases with or without ground truth masks.
    """
    def __init__(self, data_dir, transform=None, mask_transform=None, has_gt=True):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation') if has_gt else None
        self.transform = transform
        self.mask_transform = mask_transform
        self.has_gt = has_gt
        
        # Get list of image files
        self.data_ids = sorted([f for f in os.listdir(self.image_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(self.data_ids)} images in {self.image_dir}")
        if has_gt and self.masks_dir:
            print(f"Ground truth masks: {self.masks_dir}")
        else:
            print("Running inference without ground truth masks")

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)

        # Load mask if available
        if self.has_gt and self.masks_dir:
            mask_path = os.path.join(self.masks_dir, data_id)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask = convert_mask(mask)
            else:
                # Create dummy mask if file doesn't exist
                mask = Image.fromarray(np.zeros((original_size[1], original_size[0]), dtype=np.uint8))
        else:
            # Create dummy mask for inference-only mode
            mask = Image.fromarray(np.zeros((original_size[1], original_size[0]), dtype=np.uint8))

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask, data_id, original_size


# ============================================================================
# Model: Segmentation Head (Must match training exactly)
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
# Metrics (only used if ground truth is available)
# ============================================================================

def compute_iou_per_class(pred, target, num_classes=10):
    """Compute IoU for each class and return both mean and per-class IoU."""
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Visualization Functions
# ============================================================================

def save_prediction_visualization(img_tensor, pred_mask, output_path, data_id, 
                                   gt_mask=None, original_size=None):
    """
    Save visualization of prediction with optional ground truth comparison.
    
    Args:
        img_tensor: Normalized image tensor
        pred_mask: Predicted segmentation mask
        output_path: Where to save the visualization
        data_id: Image filename
        gt_mask: Optional ground truth mask
        original_size: Original image size (W, H)
    """
    # Denormalize image
    img = denormalize_image(img_tensor)
    
    # Convert masks to color
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))
    
    # Create overlay
    img_uint8 = (img * 255).astype(np.uint8)
    overlay = create_overlay(img_uint8, pred_mask.cpu().numpy().astype(np.uint8), alpha=0.5)
    
    # Determine number of subplots
    if gt_mask is not None:
        # With ground truth: show input, GT, prediction, overlay
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
        
        axes[0].imshow(img)
        axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(gt_color)
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(pred_color)
        axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay (50%)', fontsize=12, fontweight='bold')
        axes[3].axis('off')
    else:
        # Without ground truth: show input, prediction, overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(pred_color)
        axes[1].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (50%)', fontsize=12, fontweight='bold')
        axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_legend(output_path):
    """Create and save a legend showing the color coding for all classes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create patches for legend
    patches = []
    for i, (name, color) in enumerate(zip(class_names, color_palette)):
        patches.append(plt.Rectangle((0, 0), 1, 1, fc=color/255.0))
    
    ax.legend(patches, class_names, loc='center', fontsize=12, frameon=False)
    ax.axis('off')
    
    plt.title('Segmentation Class Legend', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir, has_gt):
    """Save metrics summary to a text file and create bar chart."""
    if not has_gt:
        return
    
    os.makedirs(output_dir, exist_ok=True)

    # Save text summary
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write(f"Pixel Accuracy:    {results['mean_pixel_acc']:.4f}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 60 + "\n")
        for i, (name, iou) in enumerate(zip(class_names, results['class_iou'])):
            if not np.isnan(iou):
                symbol = "good" if iou > 0.5 else "⚠" if iou > 0.3 else "✗"
                f.write(f"  {symbol} {name:<20}: {iou:.4f}\n")
            else:
                f.write(f"  bad {name:<20}: N/A\n")

    print(f"\n✓ Saved evaluation metrics to {filepath}")

    # Create bar chart for per-class IoU
    fig, ax = plt.subplots(figsize=(12, 6))

    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    bars = ax.bar(range(n_classes), valid_iou, 
                   color=[color_palette[i] / 255 for i in range(n_classes)],
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, iou) in enumerate(zip(bars, valid_iou)):
        if iou > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{iou:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('IoU Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', 
               linewidth=2, label=f'Mean IoU: {results["mean_iou"]:.3f}')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved per-class IoU chart to '{output_dir}/per_class_iou.png'")


# ============================================================================
# Main Inference Function
# ============================================================================

def main():
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(
        description='Segmentation inference script for unseen images'
    )
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join(script_dir, 'segmentation_head_best.pth'),
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages'),
                        help='Path to test images directory (should contain Color_Images folder)')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join(script_dir, 'predictions'),
                        help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for inference')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='Number of visualization comparisons to save')
    parser.add_argument('--has_gt', action='store_true',
                        help='Set if ground truth masks are available in Segmentation folder')
    args = parser.parse_args()

    # Check if ground truth exists
    segmentation_dir = os.path.join(args.data_dir, 'Segmentation')
    has_gt = os.path.exists(segmentation_dir) and len(os.listdir(segmentation_dir)) > 0
    
    if args.has_gt:
        has_gt = True
    
    print("\n" + "=" * 70)
    print("OFFROAD TERRAIN SEGMENTATION - INFERENCE")
    print("=" * 70)
    print(f"Mode: {'Evaluation (with GT)' if has_gt else 'Inference (no GT)'}")
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"Input: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 70 + "\n")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    masks_dir = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    overlays_dir = os.path.join(args.output_dir, 'overlays')
    visualizations_dir = os.path.join(args.output_dir, 'visualizations')
    
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

    # Image dimensions (must match training)
    w = int(((960 / 2) // 14) * 14)  # 476
    h = int(((540 / 2) // 14) * 14)  # 266
    print(f"Processing size: {h}x{w}")

    # Transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

    # Create dataset
    print("Loading dataset...")
    dataset = InferenceDataset(
        data_dir=args.data_dir, 
        transform=transform, 
        mask_transform=mask_transform,
        has_gt=has_gt
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0  # Windows compatibility
    )
    
    print(f"Loaded {len(dataset)} images\n")

    # Load DINOv2 backbone
    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "base"
    backbone_name = "dinov2_vitb14_reg"

    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone.eval()
    backbone.to(device)
    print("Backbone loaded\n")

    # Get embedding dimension
    sample_img, _, _, _ = dataset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = backbone.forward_features(sample_img)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")

    # Load segmentation head
    print(f"Loading segmentation head from {args.model_path}...")
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    
    if not os.path.exists(args.model_path):
        print(f"\nERROR: Model file not found: {args.model_path}")
        print("Please check the path and try again.")
        return
    
    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    print("Model loaded successfully\n")

    # Run inference
    print("=" * 70)
    print(f"Running inference on {len(dataset)} images...")
    print("=" * 70 + "\n")

    iou_scores = []
    pixel_accuracies = []
    all_class_iou = []
    vis_count = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Processing", unit="batch")
        
        for batch_idx, (imgs, labels, data_ids, original_sizes) in enumerate(pbar):
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward pass with mixed precision
            with amp.autocast(device_type='cuda'):
                output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(output)
                outputs = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

            # Get predictions
            predicted_masks = torch.argmax(outputs, dim=1)

            # Calculate metrics if ground truth is available
            if has_gt:
                labels_squeezed = labels.squeeze(dim=1).long()
                iou, class_iou = compute_iou_per_class(outputs, labels_squeezed, num_classes=n_classes)
                pixel_acc = compute_pixel_accuracy(outputs, labels_squeezed)
                
                iou_scores.append(iou)
                pixel_accuracies.append(pixel_acc)
                all_class_iou.append(class_iou)
                
                pbar.set_postfix({'IoU': f'{iou:.3f}', 'Acc': f'{pixel_acc:.3f}'})
            else:
                pbar.set_postfix({'Status': 'Inferring'})

            # Save outputs for each image in batch
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]
                original_size = original_sizes[i]  # (W, H)

                # Get prediction and resize to original size
                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
                pred_mask_resized = cv2.resize(
                    pred_mask, 
                    (original_size[0].item(), original_size[1].item()),
                    interpolation=cv2.INTER_NEAREST
                )

                # 1. Save raw prediction mask (class IDs 0-9)
                Image.fromarray(pred_mask_resized).save(
                    os.path.join(masks_dir, f'{base_name}_pred.png')
                )

                # 2. Save colored prediction mask
                pred_color = mask_to_color(pred_mask_resized)
                cv2.imwrite(
                    os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
                )

                # 3. Save overlay
                img_denorm = denormalize_image(imgs[i])
                img_denorm_resized = cv2.resize(
                    (img_denorm * 255).astype(np.uint8),
                    (original_size[0].item(), original_size[1].item())
                )
                overlay = create_overlay(img_denorm_resized, pred_mask_resized, alpha=0.5)
                cv2.imwrite(
                    os.path.join(overlays_dir, f'{base_name}_overlay.png'),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                )

                # 4. Save detailed visualization for first N samples
                if vis_count < args.num_vis:
                    gt_mask = labels[i].squeeze().long() if has_gt else None
                    save_prediction_visualization(
                        imgs[i], 
                        predicted_masks[i], 
                        os.path.join(visualizations_dir, f'{base_name}_visualization.png'),
                        data_id,
                        gt_mask=gt_mask,
                        original_size=original_size
                    )
                    vis_count += 1

    # Create and save legend
    print("\n✓ Creating class legend...")
    create_legend(os.path.join(args.output_dir, 'class_legend.png'))

    # Save metrics if ground truth was available
    if has_gt and iou_scores:
        mean_iou = np.nanmean(iou_scores)
        mean_pixel_acc = np.mean(pixel_accuracies)
        avg_class_iou = np.nanmean(all_class_iou, axis=0)

        results = {
            'mean_iou': mean_iou,
            'mean_pixel_acc': mean_pixel_acc,
            'class_iou': avg_class_iou
        }

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Mean IoU:          {mean_iou:.4f}")
        print(f"Pixel Accuracy:    {mean_pixel_acc:.4f}")
        print("\nPer-Class IoU:")
        for name, iou in zip(class_names, avg_class_iou):
            if not np.isnan(iou):
                symbol = "good" if iou > 0.5 else "⚠" if iou > 0.3 else "✗"
                print(f"  {symbol} {name:<20}: {iou:.4f}")
            else:
                print(f"  bad {name:<20}: N/A")
        print("=" * 70)

        save_metrics_summary(results, args.output_dir, has_gt=True)

    # Final summary
    print("\n" + "=" * 70)
    print("✓ INFERENCE COMPLETE!")
    print("=" * 70)
    print(f"\nProcessed {len(dataset)} images successfully")
    print(f"\nOutputs saved to: {args.output_dir}/")
    print(f"  ├── masks/              : Raw prediction masks (class IDs)")
    print(f"  ├── masks_color/        : Colored prediction masks (RGB)")
    print(f"  ├── overlays/           : Image + mask overlays")
    print(f"  ├── visualizations/     : Detailed comparisons ({args.num_vis} samples)")
    print(f"  └── class_legend.png    : Color legend for all classes")
    
    if has_gt:
        print(f"  └── evaluation_metrics.txt : Quantitative results")
        print(f"  └── per_class_iou.png      : Per-class IoU chart")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
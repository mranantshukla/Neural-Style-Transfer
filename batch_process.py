"""
Batch processing script to apply all style images to the content image.
Generates stylized images for all style combinations and saves them in a results folder.
Also plots loss vs epoch graphs for each style transfer.
"""

import os
import matplotlib.pyplot as plt
from datetime import datetime
from src.style_transfer import StyleTransfer
from src.utils import get_image_list, save_image

# Configuration
CONTENT_FOLDER = 'contentimage'
STYLE_FOLDER = 'style_images'
OUTPUT_FOLDER = 'batch_results'
PLOTS_FOLDER = 'loss_plots'

# Updated parameters for better content preservation
EPOCHS = 30
STEPS_PER_EPOCH = 5
LEARNING_RATE = 3.0  # Slightly lower for stability
CONTENT_WEIGHT = 2000  # Very low (to preserve content quality)
STYLE_WEIGHT = 0.5  # Very high


def plot_loss_curves(loss_history, save_path, style_name):
    """
    Plot loss curves (total, content, style) vs epochs.
    
    Args:
        loss_history: Dictionary with 'total_loss', 'content_loss', 'style_loss', 'epochs'
        save_path: Path to save the plot
        style_name: Name of the style image for title
    """
    plt.figure(figsize=(12, 6))
    
    epochs = loss_history['epochs']
    total_loss = loss_history['total_loss']
    content_loss = loss_history['content_loss']
    style_loss = loss_history['style_loss']
    
    plt.plot(epochs, total_loss, 'b-', label='Total Loss', linewidth=2)
    plt.plot(epochs, content_loss, 'g-', label='Content Loss', linewidth=2)
    plt.plot(epochs, style_loss, 'r-', label='Style Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'Loss vs Epochs - {style_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def batch_process_styles():
    """
    Process the content image with all available style images.
    """
    # Get content image
    content_images = get_image_list(CONTENT_FOLDER)
    if not content_images:
        print(f"Error: No content images found in {CONTENT_FOLDER}/")
        return
    
    content_image = content_images[0]  # Use the first (and only) content image
    content_path = os.path.join(CONTENT_FOLDER, content_image)
    print(f"Content image: {content_image}")
    
    # Get all style images
    style_images = get_image_list(STYLE_FOLDER)
    if not style_images:
        print(f"Error: No style images found in {STYLE_FOLDER}/")
        return
    
    print(f"Found {len(style_images)} style images")
    
    # Create output folders
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    print(f"Output folder: {OUTPUT_FOLDER}/")
    print(f"Plots folder: {PLOTS_FOLDER}/")
    print("\n" + "="*60)
    print("Starting batch processing...")
    print("="*60 + "\n")
    
    # Process each style image
    for idx, style_image in enumerate(style_images, 1):
        style_path = os.path.join(STYLE_FOLDER, style_image)
        
        print(f"[{idx}/{len(style_images)}] Processing: {style_image}")
        print(f"  Content: {content_image}")
        print(f"  Style: {style_image}")
        
        try:
            # Initialize style transfer
            style_transfer = StyleTransfer(content_path, style_path)
            
            # Generate stylized image
            print(f"  Generating... (this may take a few minutes)")
            output_image, loss_history = style_transfer.generate(
                epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH,
                learning_rate=LEARNING_RATE,
                content_weight=CONTENT_WEIGHT,
                style_weight=STYLE_WEIGHT
            )
            
            # Create output filename
            content_name = os.path.splitext(content_image)[0]
            style_name = os.path.splitext(style_image)[0]
            output_filename = f"{content_name}_styled_with_{style_name}.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Save image
            save_image(output_image, output_path)
            print(f"  [OK] Saved: {output_filename}")
            
            # Plot loss curves
            plot_filename = f"loss_{content_name}_{style_name}.png"
            plot_path = os.path.join(PLOTS_FOLDER, plot_filename)
            plot_loss_curves(loss_history, plot_path, style_image)
            print(f"  [OK] Saved loss plot: {plot_filename}")
            
        except Exception as e:
            print(f"  [ERROR] Error processing {style_image}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        print()  # Empty line for readability
    
    print("="*60)
    print(f"Batch processing complete!")
    print(f"Results saved in: {OUTPUT_FOLDER}/")
    print("="*60)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Neural Style Transfer - Batch Processing")
    print("="*60)
    print(f"Parameters:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Steps per epoch: {STEPS_PER_EPOCH}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Content weight: {CONTENT_WEIGHT}")
    print(f"  Style weight: {STYLE_WEIGHT}")
    print("="*60 + "\n")
    
    batch_process_styles()


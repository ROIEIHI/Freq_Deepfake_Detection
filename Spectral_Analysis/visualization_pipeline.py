"""
Visualization pipeline for spectral analysis of deepfake detection.
Shows step-by-step processing: RGB separation → FFT → log-magnitude → high-pass filter.

Usage:
python visualization_pipeline.py --real <path_to_real_image> --fake <path_to_fake_image> --output <output_file>

Author: Spectral Analysis Pipeline
Date: 2026-02-03
"""
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

# High-pass filter radius (from features.py)
HPF_RADIUS = 0  # High-Pass Filter Radius


def get_log_spectrum(channel):
    """Performs 2D-FFT and returns the Log-Magnitude Spectrum."""
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    return 20 * np.log(magnitude + 1e-10)


def apply_highpass_filter(spectrum, radius=HPF_RADIUS):
    """Apply circular high-pass filter to spectrum."""
    rows, cols = spectrum.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    
    # Create mask (False in center, True outside)
    mask = (x - ccol)**2 + (y - crow)**2 > radius**2
    
    # Apply mask
    filtered = spectrum.copy()
    filtered[~mask] = 0
    
    return filtered, mask


def process_image(image_path):
    """
    Process image through complete spectral analysis pipeline.
    Returns all intermediate results for visualization.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img_rgb)
    
    # Get log-magnitude spectra
    log_r = get_log_spectrum(r)
    log_g = get_log_spectrum(g)
    log_b = get_log_spectrum(b)
    
    # Apply high-pass filter
    filtered_r, mask = apply_highpass_filter(log_r)
    filtered_g, _ = apply_highpass_filter(log_g)
    filtered_b, _ = apply_highpass_filter(log_b)
    
    # Calculate correlations on filtered spectra
    valid = mask.flatten()
    flat_r = filtered_r.flatten()[valid]
    flat_g = filtered_g.flatten()[valid]
    flat_b = filtered_b.flatten()[valid]
    
    corr_rg, _ = pearsonr(flat_r, flat_g)
    corr_rb, _ = pearsonr(flat_r, flat_b)
    corr_gb, _ = pearsonr(flat_g, flat_b)
    
    # Difference maps
    diff_rg = np.abs(filtered_r - filtered_g)
    diff_rb = np.abs(filtered_r - filtered_b)
    diff_gb = np.abs(filtered_g - filtered_b)
    
    return {
        'image': img_rgb,
        'channels': {'R': r, 'G': g, 'B': b},
        'spectra': {'R': log_r, 'G': log_g, 'B': log_b},
        'filtered': {'R': filtered_r, 'G': filtered_g, 'B': filtered_b},
        'differences': {'RG': diff_rg, 'RB': diff_rb, 'GB': diff_gb},
        'correlations': {'RG': corr_rg, 'RB': corr_rb, 'GB': corr_gb},
        'mask': mask
    }


def create_visualization(real_data, fake_data, output_path):
    """
    Create comprehensive side-by-side visualization.
    """
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(6, 8, figure=fig, hspace=0.35, wspace=0.4)
    
    # Title
    fig.suptitle('Spectral Analysis: Real vs Deepfake Image Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Row 0: Original Images (larger display)
    ax = fig.add_subplot(gs[0, 1:4])
    ax.imshow(real_data['image'])
    ax.set_title('Real Image', fontsize=14, fontweight='bold', color='green')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 4:7])
    ax.imshow(fake_data['image'])
    ax.set_title('Fake Image', fontsize=14, fontweight='bold', color='red')
    ax.axis('off')
    
    # Row 1: RGB Channel Separation
    fig.text(0.02, 0.80, 'RGB\nChannels', fontsize=11, fontweight='bold',
             rotation=90, va='center')
    
    # Real RGB channels
    for i, (channel, color) in enumerate([('R', 'Reds'), ('G', 'Greens'), ('B', 'Blues')]):
        ax = fig.add_subplot(gs[1, i*2])
        ax.imshow(real_data['channels'][channel], cmap=color)
        ax.set_title(f'Real {channel}', fontsize=10)
        ax.axis('off')
    
    # Fake RGB channels
    for i, (channel, color) in enumerate([('R', 'Reds'), ('G', 'Greens'), ('B', 'Blues')]):
        ax = fig.add_subplot(gs[1, i*2+1])
        ax.imshow(fake_data['channels'][channel], cmap=color)
        ax.set_title(f'Fake {channel}', fontsize=10)
        ax.axis('off')
    
    # Row 2: Log-Magnitude Spectra
    fig.text(0.02, 0.65, 'Log-Magnitude\nSpectra', fontsize=11, fontweight='bold',
             rotation=90, va='center')
    
    channels = ['R', 'G', 'B']
    for i, channel in enumerate(channels):
        # Real spectrum
        ax = fig.add_subplot(gs[2, i*2])
        im = ax.imshow(real_data['spectra'][channel], cmap='hot')
        ax.set_title(f'Real {channel} Spectrum', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Fake spectrum
        ax = fig.add_subplot(gs[2, i*2+1])
        im = ax.imshow(fake_data['spectra'][channel], cmap='hot')
        ax.set_title(f'Fake {channel} Spectrum', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 3: High-Pass Filtered Spectra
    fig.text(0.02, 0.50, f'High-Pass\nFiltered\n(R={HPF_RADIUS}px)', fontsize=11, fontweight='bold',
             rotation=90, va='center')
    
    for i, channel in enumerate(channels):
        # Real filtered
        ax = fig.add_subplot(gs[3, i*2])
        im = ax.imshow(real_data['filtered'][channel], cmap='hot')
        ax.set_title(f'Real {channel} Filtered', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Fake filtered
        ax = fig.add_subplot(gs[3, i*2+1])
        im = ax.imshow(fake_data['filtered'][channel], cmap='hot')
        ax.set_title(f'Fake {channel} Filtered', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 4: Difference Maps
    fig.text(0.02, 0.35, 'Difference\nMaps', fontsize=11, fontweight='bold',
             rotation=90, va='center')
    
    diff_pairs = [('RG', 'R-G'), ('RB', 'R-B'), ('GB', 'G-B')]
    for i, (pair, label) in enumerate(diff_pairs):
        # Real differences
        ax = fig.add_subplot(gs[4, i*2])
        im = ax.imshow(real_data['differences'][pair], cmap='viridis')
        ax.set_title(f'Real |{label}|', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Fake differences
        ax = fig.add_subplot(gs[4, i*2+1])
        im = ax.imshow(fake_data['differences'][pair], cmap='viridis')
        ax.set_title(f'Fake |{label}|', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 5: Correlation Statistics and Mask Visualization
    fig.text(0.02, 0.15, 'Statistics\n& Analysis', fontsize=11, fontweight='bold',
             rotation=90, va='center')
    
    # Correlation comparison
    ax = fig.add_subplot(gs[5, 0:3])
    
    corr_types = ['RG', 'RB', 'GB']
    x = np.arange(len(corr_types))
    width = 0.35
    
    real_corrs = [real_data['correlations'][c] for c in corr_types]
    fake_corrs = [fake_data['correlations'][c] for c in corr_types]
    
    bars1 = ax.bar(x - width/2, real_corrs, width, label='Real', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, fake_corrs, width, label='Fake', color='red', alpha=0.7)
    
    ax.set_xlabel('Correlation Type', fontsize=11)
    ax.set_ylabel('Correlation Coefficient', fontsize=11)
    ax.set_title('RGB Channel Correlations Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(corr_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([min(min(real_corrs), min(fake_corrs)) - 0.05, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Statistics table
    ax = fig.add_subplot(gs[5, 4:7])
    ax.axis('off')
    
    stats_text = "SPECTRAL CORRELATION ANALYSIS\n" + "="*45 + "\n\n"
    stats_text += "Real Image:\n"
    for corr_type in corr_types:
        stats_text += f"  ρ_{corr_type} = {real_data['correlations'][corr_type]:.6f}\n"
    
    stats_text += "\nFake Image:\n"
    for corr_type in corr_types:
        stats_text += f"  ρ_{corr_type} = {fake_data['correlations'][corr_type]:.6f}\n"
    
    stats_text += "\nDifference (Real - Fake):\n"
    for corr_type in corr_types:
        diff = real_data['correlations'][corr_type] - fake_data['correlations'][corr_type]
        stats_text += f"  Δρ_{corr_type} = {diff:+.6f}\n"
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # High-pass filter mask visualization
    ax = fig.add_subplot(gs[5, 6:8])
    im = ax.imshow(real_data['mask'], cmap='gray')
    ax.set_title(f'High-Pass Filter Mask\n(Radius={HPF_RADIUS} px)', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Add circle to show radius
    circle = plt.Circle((real_data['mask'].shape[1]//2, real_data['mask'].shape[0]//2), 
                       HPF_RADIUS, color='red', fill=False, linewidth=2)
    ax.add_patch(circle)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize spectral analysis pipeline for deepfake detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--real', '-r', type=str, required=True,
                       help='Path to real image')
    parser.add_argument('--fake', '-f', type=str, required=True,
                       help='Path to fake image')
    parser.add_argument('--output', '-o', type=str, default='spectral_visualization.png',
                       help='Output file path')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SPECTRAL ANALYSIS VISUALIZATION PIPELINE")
    print("="*80)
    print(f"Real image: {args.real}")
    print(f"Fake image: {args.fake}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")
    
    # Process images
    print("Processing real image...")
    real_data = process_image(args.real)
    
    print("Processing fake image...")
    fake_data = process_image(args.fake)
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(real_data, fake_data, args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("CORRELATION COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Channel Pair':<15} {'Real':>12} {'Fake':>12} {'Difference':>12}")
    print("-"*80)
    for corr_type in ['RG', 'RB', 'GB']:
        real_val = real_data['correlations'][corr_type]
        fake_val = fake_data['correlations'][corr_type]
        diff = real_val - fake_val
        print(f"{corr_type:<15} {real_val:>12.6f} {fake_val:>12.6f} {diff:>+12.6f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

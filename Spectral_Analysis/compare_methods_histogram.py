"""
Comparative histogram analysis for different deepfake manipulation methods.
Samples 1000 images from each method and compares RGB correlation distributions.

Usage:
python compare_methods_histogram.py --data_path "D:/FF_data_HQ_vid" --compression c23 --num_samples 1000

Author: Spectral Analysis Pipeline
Date: 2026-02-03
"""
import os
from os.path import join, exists
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import sys

# Import features extraction
sys.path.append(os.path.dirname(__file__))
from features import extract_features

DATASET_PATHS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}


def collect_image_paths(data_path, dataset, compression, num_samples=1000):
    """
    Collect random sample of image paths from dataset.
    :param data_path: root path to data
    :param dataset: dataset name
    :param compression: compression level
    :param num_samples: number of images to sample
    :return: list of image paths
    """
    faces_path = join(data_path, DATASET_PATHS[dataset], compression, 'faces')
    
    if not exists(faces_path):
        print(f"Warning: Faces path does not exist: {faces_path}")
        return []
    
    # Collect all image paths
    all_images = []
    video_folders = [f for f in os.listdir(faces_path) 
                    if os.path.isdir(join(faces_path, f))]
    
    for video_folder in video_folders:
        video_path = join(faces_path, video_folder)
        images = [join(video_path, f) for f in os.listdir(video_path) 
                 if f.endswith('.png')]
        all_images.extend(images)
    
    # Random sample
    if len(all_images) > num_samples:
        all_images = random.sample(all_images, num_samples)
    
    return all_images


def extract_correlations(image_paths, label):
    """
    Extract RGB correlation features from list of images.
    :param image_paths: list of image paths
    :param label: dataset label for progress bar
    :return: dict with correlation arrays
    """
    correlations = {
        'corr_rg': [],
        'corr_rb': [],
        'corr_gb': []
    }
    
    for img_path in tqdm(image_paths, desc=f"Extracting {label}"):
        features = extract_features(img_path)
        if features is not None:
            correlations['corr_rg'].append(features[0])
            correlations['corr_rb'].append(features[1])
            correlations['corr_gb'].append(features[2])
    
    return correlations


def plot_comparison_histograms(real_corr, fake1_corr, fake2_corr, 
                               fake1_name, fake2_name, output_path):
    """
    Plot comparative histograms for RGB correlations.
    :param real_corr: correlations from real images
    :param fake1_corr: correlations from first fake method
    :param fake2_corr: correlations from second fake method
    :param fake1_name: name of first fake method
    :param fake2_name: name of second fake method
    :param output_path: path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('RGB Correlation Distribution Comparison', fontsize=16, fontweight='bold')
    
    correlations = ['corr_rg', 'corr_rb', 'corr_gb']
    titles = ['R-G Correlation', 'R-B Correlation', 'G-B Correlation']
    
    # Row 1: Real vs Fake1
    for i, (corr_type, title) in enumerate(zip(correlations, titles)):
        ax = axes[0, i]
        
        # Plot histograms
        ax.hist(real_corr[corr_type], bins=50, alpha=0.6, label='Real', 
               color='green', density=True, edgecolor='black')
        ax.hist(fake1_corr[corr_type], bins=50, alpha=0.6, label=fake1_name, 
               color='red', density=True, edgecolor='black')
        
        ax.set_xlabel('Correlation Coefficient', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{title}\nReal vs {fake1_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Statistics
        mean_real = np.mean(real_corr[corr_type])
        mean_fake = np.mean(fake1_corr[corr_type])
        separation = abs(mean_real - mean_fake)
        
        ax.text(0.98, 0.97, 
               f'Real μ: {mean_real:.4f}\n{fake1_name} μ: {mean_fake:.4f}\nSep: {separation:.4f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 2: Real vs Fake2
    for i, (corr_type, title) in enumerate(zip(correlations, titles)):
        ax = axes[1, i]
        
        # Plot histograms
        ax.hist(real_corr[corr_type], bins=50, alpha=0.6, label='Real', 
               color='green', density=True, edgecolor='black')
        ax.hist(fake2_corr[corr_type], bins=50, alpha=0.6, label=fake2_name, 
               color='blue', density=True, edgecolor='black')
        
        ax.set_xlabel('Correlation Coefficient', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{title}\nReal vs {fake2_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Statistics
        mean_real = np.mean(real_corr[corr_type])
        mean_fake = np.mean(fake2_corr[corr_type])
        separation = abs(mean_real - mean_fake)
        
        ax.text(0.98, 0.97, 
               f'Real μ: {mean_real:.4f}\n{fake2_name} μ: {mean_fake:.4f}\nSep: {separation:.4f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()


def print_separation_summary(real_corr, fake1_corr, fake2_corr, fake1_name, fake2_name):
    """
    Print summary of separation between real and fake distributions.
    """
    correlations = ['corr_rg', 'corr_rb', 'corr_gb']
    names = ['R-G', 'R-B', 'G-B']
    
    print("\n" + "="*80)
    print("SEPARATION ANALYSIS SUMMARY")
    print("="*80)
    
    total_sep_1 = 0
    total_sep_2 = 0
    
    for corr_type, name in zip(correlations, names):
        mean_real = np.mean(real_corr[corr_type])
        std_real = np.std(real_corr[corr_type])
        
        mean_fake1 = np.mean(fake1_corr[corr_type])
        std_fake1 = np.std(fake1_corr[corr_type])
        sep_1 = abs(mean_real - mean_fake1)
        
        mean_fake2 = np.mean(fake2_corr[corr_type])
        std_fake2 = np.std(fake2_corr[corr_type])
        sep_2 = abs(mean_real - mean_fake2)
        
        total_sep_1 += sep_1
        total_sep_2 += sep_2
        
        print(f"\n{name} Correlation:")
        print(f"  Real:        μ={mean_real:.4f}, σ={std_real:.4f}")
        print(f"  {fake1_name:12s} μ={mean_fake1:.4f}, σ={std_fake1:.4f}, |Δμ|={sep_1:.4f}")
        print(f"  {fake2_name:12s} μ={mean_fake2:.4f}, σ={std_fake2:.4f}, |Δμ|={sep_2:.4f}")
    
    print("\n" + "-"*80)
    print(f"Total Separation (sum of |Δμ| across all correlations):")
    print(f"  Real vs {fake1_name}: {total_sep_1:.4f}")
    print(f"  Real vs {fake2_name}: {total_sep_2:.4f}")
    print("-"*80)
    
    if total_sep_1 > total_sep_2:
        print(f"\n RECOMMENDATION: Use {fake1_name} (better separation: {total_sep_1:.4f} vs {total_sep_2:.4f})")
    else:
        print(f"\n RECOMMENDATION: Use {fake2_name} (better separation: {total_sep_2:.4f} vs {total_sep_1:.4f})")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare RGB correlation histograms for different deepfake methods',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, required=True,
                       help='Root path to FaceForensics++ data')
    parser.add_argument('--compression', '-c', type=str, default='c23',
                       choices=['c0', 'c23', 'c40'],
                       help='Compression level')
    parser.add_argument('--num_samples', '-n', type=int, default=1000,
                       help='Number of samples per method')
    parser.add_argument('--fake1', type=str, default='Deepfakes',
                       choices=list(DATASET_PATHS.keys()),
                       help='First fake method to compare')
    parser.add_argument('--fake2', type=str, default='NeuralTextures',
                       choices=list(DATASET_PATHS.keys()),
                       help='Second fake method to compare')
    parser.add_argument('--output', '-o', type=str, default='comparison_histograms.png',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("\n" + "="*80)
    print("COMPARATIVE HISTOGRAM ANALYSIS")
    print("="*80)
    print(f"Data path: {args.data_path}")
    print(f"Compression: {args.compression}")
    print(f"Samples per method: {args.num_samples}")
    print(f"Comparing: Real vs {args.fake1} vs {args.fake2}")
    print("="*80 + "\n")
    
    # Collect image paths
    print("Collecting image paths...")
    real_paths = collect_image_paths(args.data_path, 'original', args.compression, args.num_samples)
    fake1_paths = collect_image_paths(args.data_path, args.fake1, args.compression, args.num_samples)
    fake2_paths = collect_image_paths(args.data_path, args.fake2, args.compression, args.num_samples)
    
    print(f"Found {len(real_paths)} real images")
    print(f"Found {len(fake1_paths)} {args.fake1} images")
    print(f"Found {len(fake2_paths)} {args.fake2} images\n")
    
    if len(real_paths) == 0 or len(fake1_paths) == 0 or len(fake2_paths) == 0:
        print("ERROR: Not enough images found. Make sure face cropping is complete.")
        return
    
    # Extract correlations
    real_corr = extract_correlations(real_paths, 'Real')
    fake1_corr = extract_correlations(fake1_paths, args.fake1)
    fake2_corr = extract_correlations(fake2_paths, args.fake2)
    
    # Plot comparison
    plot_comparison_histograms(real_corr, fake1_corr, fake2_corr, 
                              args.fake1, args.fake2, args.output)
    
    # Print summary
    print_separation_summary(real_corr, fake1_corr, fake2_corr, args.fake1, args.fake2)


if __name__ == '__main__':
    main()

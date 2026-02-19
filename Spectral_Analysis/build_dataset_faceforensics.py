"""
Build dataset CSV from cropped face images for spectral analysis.
Adapted for FaceForensics++ directory structure with cropped faces.

Usage:
python build_dataset_faceforensics.py --data_path "D:/FF_data_HQ_vid" --compression c23 --method Deepfakes --num_samples 1000
"""
import os
from os.path import join, exists, basename
import csv
import random
import argparse
from tqdm import tqdm
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


def collect_all_images(data_path, dataset, compression):
    """
    Collect all cropped face image paths from dataset.
    :param data_path: root path to data
    :param dataset: dataset name
    :param compression: compression level
    :return: list of image paths
    """
    faces_path = join(data_path, DATASET_PATHS[dataset], compression, 'faces')
    
    if not exists(faces_path):
        print(f"Warning: Faces path does not exist: {faces_path}")
        print(f"Make sure you ran crop_faces.py first!")
        return []
    
    all_images = []
    video_folders = [f for f in os.listdir(faces_path) 
                    if os.path.isdir(join(faces_path, f))]
    
    print(f"Scanning {len(video_folders)} video folders in {dataset}...")
    
    for video_folder in video_folders:
        video_path = join(faces_path, video_folder)
        images = [join(video_path, f) for f in os.listdir(video_path) 
                 if f.endswith('.png')]
        all_images.extend(images)
    
    print(f"Found {len(all_images)} face images in {dataset}")
    return all_images


def sample_images_balanced(image_paths, num_samples, max_per_video=None):
    """
    Sample images with optional constraint on max per video.
    :param image_paths: list of image paths
    :param num_samples: target number of samples
    :param max_per_video: optional limit on images per video folder
    :return: sampled image paths
    """
    if max_per_video is None:
        # Simple random sampling
        if len(image_paths) > num_samples:
            return random.sample(image_paths, num_samples)
        return image_paths
    
    # Stratified sampling: group by video folder
    from collections import defaultdict
    video_groups = defaultdict(list)
    
    for path in image_paths:
        # Extract video folder name (second to last directory)
        video_name = os.path.basename(os.path.dirname(path))
        video_groups[video_name].append(path)
    
    # Sample from each video
    sampled = []
    for video_name, video_images in video_groups.items():
        if len(video_images) > max_per_video:
            sampled.extend(random.sample(video_images, max_per_video))
        else:
            sampled.extend(video_images)
    
    # Final random sampling if we have too many
    if len(sampled) > num_samples:
        sampled = random.sample(sampled, num_samples)
    
    return sampled


def extract_features_to_csv(image_paths, label, method_name, writer, desc):
    """
    Extract features from images and write to CSV.
    :param image_paths: list of image paths
    :param label: 0 for real, 1 for fake
    :param method_name: name of the manipulation method
    :param writer: CSV writer object
    :param desc: description for progress bar
    :return: number of successful extractions
    """
    success_count = 0
    
    for img_path in tqdm(image_paths, desc=desc):
        features = extract_features(img_path)
        if features is not None:
            row = [img_path] + features + [label, method_name]
            writer.writerow(row)
            success_count += 1
    
    return success_count


def build_dataset(data_path, compression, method, num_samples, max_per_video, output_file):
    """
    Build complete dataset CSV.
    """
    print("\n" + "="*80)
    print("BUILDING FACEFORENSICS++ DATASET")
    print("="*80)
    print(f"Data path: {data_path}")
    print(f"Compression: {compression}")
    print(f"Fake method: {method}")
    print(f"Samples per class: {num_samples}")
    if max_per_video:
        print(f"Max per video: {max_per_video}")
    print(f"Output file: {output_file}")
    print("="*80 + "\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Collect images
    print("Step 1: Collecting image paths...")
    real_images = collect_all_images(data_path, 'original', compression)
    fake_images = collect_all_images(data_path, method, compression)
    
    if len(real_images) == 0:
        print(f"\nERROR: No real images found!")
        print(f"Expected location: {join(data_path, DATASET_PATHS['original'], compression, 'faces')}")
        print("Run crop_faces.py first to extract face regions.")
        return
    
    if len(fake_images) == 0:
        print(f"\nERROR: No fake images found for {method}!")
        print(f"Expected location: {join(data_path, DATASET_PATHS[method], compression, 'faces')}")
        print("Run crop_faces.py first to extract face regions.")
        return
    
    # Sample images
    print("\nStep 2: Sampling images...")
    real_sampled = sample_images_balanced(real_images, num_samples, max_per_video)
    fake_sampled = sample_images_balanced(fake_images, num_samples, max_per_video)
    
    print(f"Sampled {len(real_sampled)} real images")
    print(f"Sampled {len(fake_sampled)} fake images")
    
    # Extract features and write CSV
    print("\nStep 3: Extracting features...")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_path', 'corr_rg', 'corr_rb', 'corr_gb',
            'diff_mean', 'diff_max', 'diff_min', 'label', 'method'
        ])
        
        # Process real images
        real_count = extract_features_to_csv(
            real_sampled, 0, 'original', writer, 'Extracting Real'
        )
        
        # Process fake images
        fake_count = extract_features_to_csv(
            fake_sampled, 1, method, writer, f'Extracting {method}'
        )
    
    # Summary
    print("\n" + "="*80)
    print("DATASET BUILD COMPLETE")
    print("="*80)
    print(f"Real images processed:  {real_count}/{len(real_sampled)} ({100*real_count/len(real_sampled):.1f}%)")
    print(f"Fake images processed:  {fake_count}/{len(fake_sampled)} ({100*fake_count/len(fake_sampled):.1f}%)")
    print(f"Total samples:          {real_count + fake_count}")
    print(f"Output saved to:        {output_file}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Build dataset CSV from FaceForensics++ cropped faces',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, required=True,
                       help='Root path to FaceForensics++ data')
    parser.add_argument('--compression', '-c', type=str, default='c23',
                       choices=['c0', 'c23', 'c40'],
                       help='Compression level')
    parser.add_argument('--method', '-m', type=str, default='Deepfakes',
                       choices=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
                       help='Fake manipulation method')
    parser.add_argument('--num_samples', '-n', type=int, default=1000,
                       help='Number of samples per class (real/fake)')
    parser.add_argument('--max_per_video', type=int, default=None,
                       help='Optional: max images per video folder for balanced sampling')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output CSV file path (default: dataset_{method}_{compression}.csv)')
    
    args = parser.parse_args()
    
    # Default output filename
    if args.output is None:
        args.output = f'dataset_{args.method}_{args.compression}.csv'
    
    # Build dataset
    build_dataset(
        args.data_path,
        args.compression,
        args.method,
        args.num_samples,
        args.max_per_video,
        args.output
    )


if __name__ == '__main__':
    main()

"""
Build dataset CSV with image paths and labels.
Outputs a simple CSV with two columns: image_path and label (0=real, 1=fake).

Usage:
python build_dataset_faceforensics.py --data_path "." --method Deepfakes

Author: Spectral Analysis Pipeline
Date: 2026-02-03
"""
import os
from os.path import join, exists
import csv
import argparse

DATASET_PATHS = {
    'original': 'original',
    'Deepfakes': 'Deepfakes',
}


def collect_all_images(data_path, dataset):
    """
    Collect all cropped face image paths from dataset.
    :param data_path: root path to data
    :param dataset: dataset name
    :return: list of image paths
    """
    faces_path = join(data_path, DATASET_PATHS[dataset], 'faces')
    
    if not exists(faces_path):
        print(f"Warning: Faces path does not exist: {faces_path}")
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


def build_dataset(data_path, method, output_file):
    """
    Build simple dataset CSV with image paths and labels.
    """
    print("\n" + "="*80)
    print("BUILDING DATASET")
    print("="*80)
    print(f"Data path: {data_path}")
    print(f"Fake method: {method}")
    print(f"Output file: {output_file}")
    print("="*80 + "\n")
    
    # Collect images
    print("Collecting image paths...")
    real_images = collect_all_images(data_path, 'original')
    fake_images = collect_all_images(data_path, method)
    
    if len(real_images) == 0:
        print(f"\nERROR: No real images found!")
        print(f"Expected location: {join(data_path, DATASET_PATHS['original'], 'faces')}")
        return
    
    if len(fake_images) == 0:
        print(f"\nERROR: No fake images found for {method}!")
        print(f"Expected location: {join(data_path, DATASET_PATHS[method], 'faces')}")
        return
    
    # Write CSV
    print("\nWriting CSV...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label'])
        
        for img_path in real_images:
            writer.writerow([img_path, 0])
        
        for img_path in fake_images:
            writer.writerow([img_path, 1])
    
    # Summary
    print("\n" + "="*80)
    print("DATASET BUILD COMPLETE")
    print("="*80)
    print(f"Real images: {len(real_images)}")
    print(f"Fake images: {len(fake_images)}")
    print(f"Total: {len(real_images) + len(fake_images)}")
    print(f"Output: {output_file}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Build dataset CSV with image paths and labels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, default='.',
                       help='Root path to data')
    parser.add_argument('--method', '-m', type=str, default='Deepfakes',
                       choices=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
                       help='Fake manipulation method')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output CSV file path (default: dataset_{method}.csv)')
    
    args = parser.parse_args()
    
    # Default output filename
    if args.output is None:
        args.output = f'dataset_{args.method}.csv'
    
    # Build dataset
    build_dataset(
        args.data_path,
        args.method,
        args.output
    )


if __name__ == '__main__':
    main()

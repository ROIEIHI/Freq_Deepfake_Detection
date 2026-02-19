"""
Crops face regions from extracted frames using dlib face detector.
Preserves PNG format without compression for spectral analysis.

Usage:
python crop_faces.py --data_path "D:/FF_data_HQ_vid" --dataset Deepfakes --compression c23
python crop_faces.py --data_path "D:/FF_data_HQ_vid" --dataset original --compression c23

Author: Spectral Analysis Pipeline
Date: 2026-02-03
"""
import os
from os.path import join, exists, dirname
import argparse
import cv2
import dlib
from tqdm import tqdm
import numpy as np

DATASET_PATHS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def crop_face_from_image(image_path, output_path, face_detector, scale=1.3):
    """
    Detect and crop face from image, save as PNG.
    :param image_path: path to input image
    :param output_path: path to save cropped face
    :param face_detector: dlib face detector instance
    :param scale: bounding box scale factor
    :return: True if successful, False otherwise
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    height, width = image.shape[:2]
    
    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    
    if len(faces) == 0:
        return False
    
    # Take the largest face (by bounding box area)
    if len(faces) > 1:
        faces = sorted(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()), reverse=True)
    
    face = faces[0]
    
    # Get bounding box
    x, y, size = get_boundingbox(face, width, height, scale=scale)
    
    # Crop face region
    cropped_face = image[y:y+size, x:x+size]
    
    # Ensure output directory exists
    os.makedirs(dirname(output_path), exist_ok=True)
    
    # Save as PNG without compression (compression level 0)
    cv2.imwrite(output_path, cropped_face, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    return True


def crop_dataset_faces(data_path, dataset, compression, scale=1.3):
    """
    Crop faces from all images in a dataset directory.
    :param data_path: root path to FaceForensics++ data
    :param dataset: dataset name (original, Deepfakes, etc.)
    :param compression: compression level (c0, c23, c40)
    :param scale: bounding box scale factor
    """
    # Initialize dlib face detector
    print("Initializing dlib face detector...")
    face_detector = dlib.get_frontal_face_detector()
    
    # Paths
    images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
    faces_path = join(data_path, DATASET_PATHS[dataset], compression, 'faces')
    
    if not exists(images_path):
        print(f"Error: Images path does not exist: {images_path}")
        return
    
    print(f"Processing images from: {images_path}")
    print(f"Saving cropped faces to: {faces_path}")
    
    # Get all video folders
    video_folders = sorted([f for f in os.listdir(images_path) 
                           if os.path.isdir(join(images_path, f))])
    
    stats = {
        'total': 0,
        'success': 0,
        'no_face': 0,
        'error': 0
    }
    
    # Process each video folder
    for video_folder in tqdm(video_folders, desc=f"Processing {dataset}"):
        video_images_path = join(images_path, video_folder)
        video_faces_path = join(faces_path, video_folder)
        
        # Get all PNG images
        image_files = sorted([f for f in os.listdir(video_images_path) 
                             if f.endswith('.png')])
        
        for image_file in image_files:
            stats['total'] += 1
            
            input_path = join(video_images_path, image_file)
            output_path = join(video_faces_path, image_file)
            
            # Skip if already processed
            if exists(output_path):
                stats['success'] += 1
                continue
            
            try:
                success = crop_face_from_image(input_path, output_path, face_detector, scale=scale)
                if success:
                    stats['success'] += 1
                else:
                    stats['no_face'] += 1
            except Exception as e:
                stats['error'] += 1
                # Optionally log errors
                # print(f"Error processing {input_path}: {e}")
    
    # Print statistics
    print("\n" + "="*60)
    print(f"FACE CROPPING COMPLETE - {dataset} / {compression}")
    print("="*60)
    print(f"Total images processed: {stats['total']}")
    print(f"Successfully cropped:   {stats['success']} ({100*stats['success']/stats['total']:.1f}%)")
    print(f"No face detected:       {stats['no_face']} ({100*stats['no_face']/stats['total']:.1f}%)")
    print(f"Errors:                 {stats['error']} ({100*stats['error']/stats['total']:.1f}%)")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Crop faces from FaceForensics++ extracted frames',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, required=True,
                       help='Root path to FaceForensics++ data')
    parser.add_argument('--dataset', '-d', type=str,
                       choices=list(DATASET_PATHS.keys()) + ['all'],
                       default='all',
                       help='Dataset to process')
    parser.add_argument('--compression', '-c', type=str,
                       choices=['c0', 'c23', 'c40'],
                       default='c23',
                       help='Compression level')
    parser.add_argument('--scale', type=float, default=1.3,
                       help='Bounding box scale factor')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            crop_dataset_faces(args.data_path, dataset, args.compression, args.scale)
    else:
        crop_dataset_faces(args.data_path, args.dataset, args.compression, args.scale)

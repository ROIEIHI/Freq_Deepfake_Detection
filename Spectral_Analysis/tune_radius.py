import cv2
import numpy as np
import os
import random
from scipy.stats import pearsonr

# --- CONFIGURATION ---
REAL_DIR = f"D:/FF_data_HQ_vid/original_sequences/youtube/c23/images"
FAKE_DIR = f"D:/FF_data_HQ_vid/manipulated_sequences/Deepfakes/c23/images"
SAMPLE_SIZE = 100  # Check 100 images from each class

def get_log_spectrum(channel):
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)
    return 20 * np.log(np.abs(fshift) + 1e-10)

def test_radius(img, radius):
    # Split
    r, g, b = cv2.split(img)
    log_g, log_b = get_log_spectrum(g), get_log_spectrum(b)
    
    # Mask
    rows, cols = log_g.shape
    crow, ccol = rows//2, cols//2
    y, x = np.ogrid[:rows, :cols]
    mask = (x - ccol)**2 + (y - crow)**2 <= radius**2
    valid = ~mask.flatten()
    
    # Correlate High Freqs
    corr, _ = pearsonr(log_g.flatten()[valid], log_b.flatten()[valid])
    return corr

# 1. Load a Batch of Images
print("Loading sample images...")

# Recursively find image files in subdirectories
def find_image_files(root_dir, max_files):
    image_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(dirpath, filename))
                if len(image_files) >= max_files:
                    return image_files
    return image_files

real_files = find_image_files(REAL_DIR, SAMPLE_SIZE)
fake_files = find_image_files(FAKE_DIR, SAMPLE_SIZE)

print(f"Found {len(real_files)} real images")
print(f"Found {len(fake_files)} fake images")

if len(real_files) == 0:
    print(f"ERROR: No image files found in {REAL_DIR}")
    exit(1)
if len(fake_files) == 0:
    print(f"ERROR: No image files found in {FAKE_DIR}")
    exit(1)

# 2. Check Resolution
print(f"Loading first real image: {real_files[0]}")
sample_img = cv2.imread(real_files[0])
if sample_img is None:
    print(f"ERROR: Failed to load image: {real_files[0]}")
    exit(1)
h, w, _ = sample_img.shape
print(f"DETECTED IMAGE SIZE: {w}x{h}")
print(f"Current Radius (150) covers {(150*2)} pixels diameter.")
if 150*2 > w:
    print("CRITICAL ERROR: Radius is larger than the image! The image was totally blocked.")

# 3. Sweep Radii
radii_to_test = [10, 20, 40, 60, 80, 100, 150]
print(f"\n{'RADIUS':<10} | {'REAL AVG':<10} | {'FAKE AVG':<10} | {'GAP (Diff)':<10}")
print("-" * 50)

best_radius = 0
best_gap = 0

for r in radii_to_test:
    real_scores = []
    fake_scores = []
    
    for f in real_files:
        img = cv2.imread(f)
        if img is not None: real_scores.append(test_radius(img, r))
            
    for f in fake_files:
        img = cv2.imread(f)
        if img is not None: fake_scores.append(test_radius(img, r))
    
    avg_real = np.mean(real_scores)
    avg_fake = np.mean(fake_scores)
    gap = avg_real - avg_fake
    
    print(f"{r:<10} | {avg_real:.4f}     | {avg_fake:.4f}     | {gap:.4f}")
    
    if abs(gap) > abs(best_gap):
        best_gap = gap
        best_radius = r

print(f"\nRecommended Radius: {best_radius} (Gap: {best_gap:.4f})")

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
REAL_DIR = r"D:\FF_data_HQ_vid\original_sequences\youtube\c23\faces"
FAKE_DIR = r"D:\FF_data_HQ_vid\manipulated_sequences\Deepfakes\c23\faces"
LIMIT = 2000  # Number of images to average (to get a clean curve)

def find_image_files(root_dir, max_files):
    image_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(dirpath, filename))
                if len(image_files) >= max_files:
                    return image_files
    return image_files

real_files = find_image_files(REAL_DIR, LIMIT)
fake_files = find_image_files(FAKE_DIR, LIMIT)

def get_azimuthal_average(image_path):
    img = cv2.imread(image_path, 0) # Read as Grayscale (Energy analysis)
    if img is None: return None
    
    # 1. FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    psd = magnitude**2 # Power Spectrum Density

    # 2. Azimuthal Average (Radial Integration)
    h, w = psd.shape
    center_y, center_x = h // 2, w // 2
    
    # Create grid of radii
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.astype(int)

    # Average energy per radius
    # We use bincount to sum energy for each radius r
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-10) # Avoid division by zero
    
    return radial_profile

# --- DATA COLLECTION ---
print(f"Calculating average 1D Spectrum for {LIMIT} images...")

real_profiles = []
fake_profiles = []

# Collect Real
for f in real_files:
    p = get_azimuthal_average(f)
    if p is not None: real_profiles.append(p)

# Collect Fake
for f in fake_files:
    p = get_azimuthal_average(f)
    if p is not None: fake_profiles.append(p)

# --- VISUALIZATION ---
# Truncate to shortest length to match arrays
min_len = min(min([len(p) for p in real_profiles]), min([len(p) for p in fake_profiles]))
real_matrix = np.array([p[:min_len] for p in real_profiles])
fake_matrix = np.array([p[:min_len] for p in fake_profiles])

# Calculate Mean curves
mean_real = np.mean(real_matrix, axis=0)
mean_fake = np.mean(fake_matrix, axis=0)

# Use Log Scale for Y-axis (Energy)
plt.figure(figsize=(10, 6))
plt.plot(np.log(mean_real + 1e-10), color='green', label='Real (Mean)', linewidth=2)
plt.plot(np.log(mean_fake + 1e-10), color='red', label='Fake (Mean)', linewidth=2, linestyle='--')

plt.title("1D Power Spectrum (Azimuthal Average)")
plt.xlabel("Frequency (Radius)")
plt.ylabel("Log Energy")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()
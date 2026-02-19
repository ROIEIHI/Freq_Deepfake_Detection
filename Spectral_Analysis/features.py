import cv2
import numpy as np
from scipy.stats import pearsonr

# --- UPDATED PARAMETER ---
# Based on your "Tuner" script results
HPF_RADIUS = 50  # High-Pass Filter Radius 

def get_log_spectrum(channel):
    """Performs 2D-FFT and returns the Log-Magnitude Spectrum."""
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    return 20 * np.log(magnitude + 1e-10)

def extract_features(image_path):
    try:
        # 1. Load
        img = cv2.imread(image_path)
        if img is None: return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(img_rgb)
    
        # 2. Spectra
        log_r = get_log_spectrum(r)
        log_g = get_log_spectrum(g)
        log_b = get_log_spectrum(b)
    
        # 3. High-Pass Filter (Radius 50)
        rows, cols = log_r.shape
        crow, ccol = rows//2, cols//2
        y, x = np.ogrid[:rows, :cols]
        
        # Mask Center (Low Freq)
        mask = (x - ccol)**2 + (y - crow)**2 <= HPF_RADIUS**2
        valid = ~mask.flatten()
    
        # Flatten
        flat_r = log_r.flatten()[valid]
        flat_g = log_g.flatten()[valid]
        flat_b = log_b.flatten()[valid]
    
        # 4. Features
        # Correlation
        corr_rg, _ = pearsonr(flat_r, flat_g)
        corr_rb, _ = pearsonr(flat_r, flat_b)
        corr_gb, _ = pearsonr(flat_g, flat_b)
        
        # Differences (on valid pixels only)
        diff_map = np.abs(log_r - log_g) 
        diff_valid = diff_map.flatten()[valid]
        
        mean_diff = np.mean(diff_valid)
        max_diff = np.max(diff_valid)
        min_diff = np.min(diff_valid)
    
        return [corr_rg, corr_rb, corr_gb, mean_diff, max_diff, min_diff]
        
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None
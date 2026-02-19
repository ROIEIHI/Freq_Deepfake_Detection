# Deepfake Detection using Xception Neural Network

A deep learning-based system for detecting deepfake images using transfer learning with the Xception architecture. This project implements a binary classifier (Real vs Fake) trained on the FaceForensics++ dataset.

## Project Overview

This project implements a deepfake detection pipeline with the following components:

1. **Dataset Builder** - Collects and organizes face images from FaceForensics++ dataset
2. **Feature Extractor** - Extracts frequency-domain features using FFT analysis
3. **Training Pipeline** - Two-phase transfer learning with Xception (ImageNet pretrained)
4. **Demo Application** - Interactive GUI for real-time deepfake classification

## Architecture

The model uses **Xception** (Extreme Inception) pretrained on ImageNet with a custom classification head:

```
Xception (frozen base) → GlobalAveragePooling2D → Dropout(0.5) → Dense(1, sigmoid)
```

### Training Strategy
- **Phase 1**: Train classification head only (base frozen) - Learning rate: 1e-3
- **Phase 2**: Fine-tune top ~30 layers of Xception - Learning rate: 1e-5

### Data Integrity
- Video-level GroupShuffleSplit ensures no data leakage between train/validation sets
- Images from the same video never appear in both splits

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 80.85% |
| **Precision** | 72.98% |
| **Recall** | 97.47% |
| **F1 Score** | 83.46% |
| **AUC** | 92.76% |

### Confusion Matrix
|  | Predicted Real | Predicted Fake |
|--|----------------|----------------|
| **Actual Real** | 6,686 | 3,680 |
| **Actual Fake** | 258 | 9,938 |

The model achieves **very high recall (97.47%)** for detecting fakes, making it effective at catching deepfake content while maintaining reasonable precision.

## Project Structure

```
Deepfake_Detection/
├── train_xception.py          # Main training script with two-phase learning
├── build_dataset_faceforensics.py  # Dataset CSV builder
├── features.py                # FFT-based feature extraction utilities
├── demo_classifier.py         # Interactive GUI demo application
├── dataset_Deepfakes_c23.csv  # Dataset CSV (image paths & labels)
├── xception_results_*/        # Training outputs (model, plots, reports)
│   ├── best_xception.keras    # Trained model weights
│   ├── confusion_matrix.png   # Confusion matrix visualization
│   ├── metrics_bar_chart.png  # Performance metrics chart
│   └── evaluation_report.txt  # Detailed evaluation report
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA 11.2 and cuDNN (recommended)
- ~8GB GPU memory (e.g., RTX 2080)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ROIEIHI/Freq_Deepfake_Detection.git
cd Freq_Deepfake_Detection
```

2. Create a virtual environment:
```bash
python -m venv env
# On Windows:
.\env\Scripts\activate
# On Linux/Mac:
source env/bin/activate
```

3. Install dependencies:
```bash
pip install tensorflow==2.10.0 pandas scikit-learn matplotlib seaborn pillow
pip install "numpy<2"  # Required for TF 2.10 compatibility
```

### Dataset Setup

The project uses the **FaceForensics++** dataset. Organize face images as:
```
Deepfake_Detection/
├── original/faces/
│   ├── video_001/
│   │   ├── 0000.png
│   │   ├── 0010.png
│   │   └── ...
│   └── ...
└── Deepfakes/faces/
    ├── video_001_002/
    │   ├── 0000.png
    │   └── ...
    └── ...
```

Build the dataset CSV:
```bash
python build_dataset_faceforensics.py --data_path "." --method Deepfakes
```

## Usage

### Training

```bash
# Basic training (default parameters)
python train_xception.py

# Custom parameters
python train_xception.py --batch_size 32 --head_epochs 5 --finetune_epochs 5

# Full options
python train_xception.py \
    --csv dataset_Deepfakes_c23.csv \
    --batch_size 16 \
    --head_epochs 10 \
    --finetune_epochs 10 \
    --val_split 0.2 \
    --seed 42
```

### Demo Application

Launch the interactive GUI to classify individual images:
```bash
python demo_classifier.py --model ./xception_results_*/best_xception.keras
```

The demo provides:
- Drag-and-drop or file browser image loading
- Real-time prediction with confidence scores
- Visual classification results (Real/Fake)

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `dataset_Deepfakes_c23.csv` | Path to dataset CSV |
| `--batch_size` | 16 | Training batch size |
| `--head_epochs` | 10 | Epochs for Phase 1 (head training) |
| `--finetune_epochs` | 10 | Epochs for Phase 2 (fine-tuning) |
| `--val_split` | 0.2 | Validation set proportion |
| `--seed` | 42 | Random seed for reproducibility |
| `--output_dir` | `xception_results` | Output directory prefix |

## Methodology

### Feature Extraction (Frequency Domain)
The `features.py` module provides FFT-based analysis:
- 2D Fast Fourier Transform on RGB channels
- High-pass filtering (radius=50) to focus on high-frequency artifacts
- Cross-channel correlation analysis (R-G, R-B, G-B)
- Spectral difference statistics

### Data Augmentation
- Random horizontal flip
- Random rotation (±7 degrees)
- Xception preprocessing (scales pixels to [-1, 1])

## Training Outputs

After training completes, the results directory contains:
- `best_xception.keras` - Best model checkpoint (by validation AUC)
- `confusion_matrix.png` - Visual confusion matrix
- `metrics_bar_chart.png` - Bar chart of Accuracy, Precision, Recall, F1
- `evaluation_report.txt` - Detailed text report

## Requirements

```
tensorflow==2.10.0
numpy<2
pandas
scikit-learn
matplotlib
seaborn
pillow
```

## License

This project is for educational and research purposes.

## Acknowledgments

- **FaceForensics++** dataset for providing the training data
- **Xception** architecture by François Chollet
- TensorFlow and Keras teams

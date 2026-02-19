"""
Main Pipeline: Build Dataset and Train Models for Deepfake Detection
This script orchestrates the complete workflow:
1. Build dataset from FaceForensics++ cropped faces
2. Train multiple ML models (SVM, XGBoost, Random Forest, etc.)
3. Generate comprehensive evaluation results

Usage:
python main.py
"""
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import our modules
from build_dataset_faceforensics import build_dataset
from train_model_comprehensive import (
    load_and_prepare_data,
    create_output_directory,
    train_models,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_performance_comparison,
    plot_feature_importance,
    save_results_summary
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import pandas as pd
from os.path import join


def main():
    print("\n" + "="*80)
    print("DEEPFAKE DETECTION PIPELINE - SPECTRAL ANALYSIS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Data paths
    DATA_PATH = r"D:\FF_data_HQ_vid"
    COMPRESSION = "c23"
    FAKE_METHOD = "NeuralTextures"
    
    # Dataset parameters
    NUM_SAMPLES = 1000  # Number of samples per class (real/fake)
    MAX_PER_VIDEO = None  # None = no limit, or set a number for balanced sampling
    
    # Output paths
    DATASET_OUTPUT = f"dataset_{FAKE_METHOD}_{COMPRESSION}.csv"
    RESULTS_DIR = f"results_{FAKE_METHOD}"
    
    # Training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    print("CONFIGURATION:")
    print(f"  Data Path: {DATA_PATH}")
    print(f"  Compression: {COMPRESSION}")
    print(f"  Fake Method: {FAKE_METHOD}")
    print(f"  Samples per class: {NUM_SAMPLES}")
    print(f"  Dataset output: {DATASET_OUTPUT}")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Test size: {TEST_SIZE}")
    print()
    
    # ========================================================================
    # STEP 1: BUILD DATASET
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: BUILDING DATASET")
    print("="*80 + "\n")
    
    # Check if dataset already exists
    if os.path.exists(DATASET_OUTPUT):
        user_input = input(f"Dataset '{DATASET_OUTPUT}' already exists. Rebuild? (y/n): ")
        if user_input.lower() != 'y':
            print("Skipping dataset build, using existing file.\n")
        else:
            print("Rebuilding dataset...\n")
            build_dataset(
                data_path=DATA_PATH,
                compression=COMPRESSION,
                method=FAKE_METHOD,
                num_samples=NUM_SAMPLES,
                max_per_video=MAX_PER_VIDEO,
                output_file=DATASET_OUTPUT
            )
    else:
        print("Building dataset from scratch...\n")
        build_dataset(
            data_path=DATA_PATH,
            compression=COMPRESSION,
            method=FAKE_METHOD,
            num_samples=NUM_SAMPLES,
            max_per_video=MAX_PER_VIDEO,
            output_file=DATASET_OUTPUT
        )
    
    # Verify dataset was created
    if not os.path.exists(DATASET_OUTPUT):
        print(f"\nERROR: Dataset file not found: {DATASET_OUTPUT}")
        print("Dataset building failed. Exiting.")
        return
    
    print(f"\nDataset ready: {DATASET_OUTPUT}")
    
    # ========================================================================
    # STEP 2: TRAIN MODELS
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: TRAINING MODELS")
    print("="*80 + "\n")
    
    # Create output directory
    output_path = create_output_directory(RESULTS_DIR)
    print(f"Results will be saved to: {output_path}\n")
    
    # Load and prepare data
    X, y, feature_cols, method_name, groups = load_and_prepare_data(DATASET_OUTPUT)
    
    # Split data by VIDEO ID to prevent data leakage
    print(f"\n{'='*80}")
    print("SPLITTING DATA BY VIDEO ID (preventing leakage)")
    print(f"{'='*80}")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Get video IDs for verification
    train_videos = groups.iloc[train_idx].unique()
    test_videos = groups.iloc[test_idx].unique()
    
    print(f"Train videos: {len(train_videos)}")
    print(f"Test videos: {len(test_videos)}")
    print(f"Train samples: {len(X_train)} (rows)")
    print(f"Test samples: {len(X_test)} (rows)")
    print(f"Total samples: {len(X_train) + len(X_test)}")
    print(f"\nTrain set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"\nTrain label distribution:\n{y_train.value_counts()}")
    print(f"Test label distribution:\n{y_test.value_counts()}")
    print(f"\nOverlap check: {len(set(train_videos) & set(test_videos))} videos in both (should be 0!)")
    
    if len(set(train_videos) & set(test_videos)) > 0:
        print("WARNING: Video leakage detected!")
    else:
        print("No video leakage - train and test are properly separated")
    
    print(f"{'='*80}\n")
    
    # Scale data
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train all models
    results, trained_models = train_models(
        X_train, X_test, y_train, y_test, 
        X_train_scaled, X_test_scaled
    )
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    # Find best model
    best_idx = results_df['Accuracy'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_accuracy = results_df.loc[best_idx, 'Accuracy']
    best_model_data = trained_models[best_model_name]
    
    # ========================================================================
    # STEP 3: RESULTS AND VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: GENERATING RESULTS")
    print("="*80 + "\n")
    
    # Print results
    print("MODEL COMPARISON RESULTS:")
    print("-"*80)
    print(results_df.to_string(index=False))
    
    print(f"\n" + "="*80)
    print(f"BEST MODEL: {best_model_name} (Accuracy: {best_accuracy*100:.2f}%)")
    print("="*80)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, best_model_data['y_pred'], 
                               target_names=['Real', 'Fake']))
    
    # Save models
    print("\nSaving models...")
    joblib.dump(best_model_data['model'], join(output_path, 'best_model.pkl'))
    joblib.dump(scaler, join(output_path, 'scaler.pkl'))
    print(f"Saved best model to {join(output_path, 'best_model.pkl')}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_roc_curves(y_test, trained_models, output_path)
    plot_confusion_matrices(y_test, trained_models, output_path)
    plot_performance_comparison(results_df, output_path)
    plot_feature_importance(best_model_data['model'], feature_cols, output_path)
    print("All visualizations generated")
    
    # Save summary
    print("\nSaving summary...")
    save_results_summary(results_df, best_model_name, best_accuracy, method_name, output_path)
    print("Summary saved")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"Dataset: {DATASET_OUTPUT}")
    print(f"Results directory: {output_path}")
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {best_accuracy*100:.2f}%")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

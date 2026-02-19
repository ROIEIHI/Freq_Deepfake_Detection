"""
Train and evaluate deepfake detection models with comprehensive analysis.
Generates ROC curves, feature importance, and publication-ready visualizations.

Usage:
python train_model_comprehensive.py --dataset dataset_Deepfakes_c23.csv --output_dir results_deepfakes
"""
import os
from os.path import join
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            f1_score, precision_score, recall_score, roc_curve, auc,
                            roc_auc_score)
import xgboost as xgb
import joblib
import time
from datetime import datetime


def create_output_directory(output_dir):
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{output_dir}_{timestamp}"
    os.makedirs(output_path, exist_ok=True)
    return output_path


def load_and_prepare_data(dataset_path):
    """Load dataset and prepare features."""
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    print(f"\n{'='*80}")
    print("DATASET LOADING DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"Raw CSV shape: {df.shape}")
    print(f"Raw CSV columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Drop 'method' column if it exists (string not suitable for ML models)
    if 'method' in df.columns:
        method_name = df['method'].iloc[0]  # Save method name before dropping
        df = df.drop('method', axis=1)
        print(f"Dropped 'method' column. Dataset method: {method_name}")
    else:
        method_name = 'Unknown'
    
    # Extract video ID from image_path to prevent data leakage
    # Path format: .../faces/000/0000.png -> video_id = 000
    df['video_id'] = df['image_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Unique videos: {df['video_id'].nunique()}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Frames per video (mean): {df.groupby('video_id').size().mean():.1f}")
    print(f"{'='*80}\n")
    
    feature_cols = ['corr_rg', 'corr_rb', 'corr_gb', 'diff_mean', 'diff_max', 'diff_min']
    X = df[feature_cols]
    y = df['label']
    groups = df['video_id']  # Group identifier for splitting
    
    print(f"Feature matrix X shape: {X.shape}")
    print(f"Label vector y shape: {y.shape}")
    print(f"Groups vector shape: {groups.shape}\n")
    
    return X, y, feature_cols, method_name, groups


def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train multiple models and return results."""
    models = {
        'XGBoost': (xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        ), False),
        'SVM (RBF)': (SVC(kernel='rbf', C=1.0, probability=True, random_state=42), True),
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), False),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
        'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), True),
        'Decision Tree': (DecisionTreeClassifier(random_state=42), False)
    }
    
    results = []
    trained_models = {}
    
    print("\n" + "="*80)
    print("TRAINING AND EVALUATING MODELS")
    print("="*80)
    
    for name, (model, needs_scaling) in models.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()
        
        if needs_scaling:
            print(f"Training on scaled data: X_train_scaled.shape = {X_train_scaled.shape}")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        else:
            print(f"Training on raw data: X_train.shape = {X_train.shape}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.4f} seconds")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Train Time (s)': train_time
        })
        
        trained_models[name] = {
            'model': model,
            'needs_scaling': needs_scaling,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"Accuracy: {accuracy*100:.2f}% | ROC-AUC: {roc_auc:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
    
    return results, trained_models


def plot_roc_curves(y_test, trained_models, output_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, model_data in trained_models.items():
        if model_data['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, model_data['y_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(output_path, 'roc_curves.png'), dpi=300)
    plt.close()
    print(f"Saved ROC curves to {join(output_path, 'roc_curves.png')}")


def plot_confusion_matrices(y_test, trained_models, output_path):
    """Plot confusion matrices for top 4 models."""
    # Select top 4 by accuracy
    top_models = sorted(trained_models.items(), 
                       key=lambda x: accuracy_score(y_test, x[1]['y_pred']), 
                       reverse=True)[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, model_data) in enumerate(top_models):
        cm = confusion_matrix(y_test, model_data['y_pred'])
        acc = accuracy_score(y_test, model_data['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'],
                   ax=axes[idx], cbar_kws={'label': 'Count'})
        axes[idx].set_title(f'{name}\nAccuracy: {acc*100:.2f}%', fontweight='bold')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(join(output_path, 'confusion_matrices.png'), dpi=300)
    plt.close()
    print(f"Saved confusion matrices to {join(output_path, 'confusion_matrices.png')}")


def plot_performance_comparison(results_df, output_path):
    """Plot comprehensive performance comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        models = results_df['Model'].values
        values = results_df[metric].values * 100
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax.barh(models, values, color=colors)
        
        ax.set_xlabel(f'{metric} (%)', fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 105])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 1, i, f'{val:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(join(output_path, 'performance_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved performance comparison to {join(output_path, 'performance_comparison.png')}")


def plot_feature_importance(best_model, feature_cols, output_path):
    """Plot feature importance for tree-based models."""
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_cols)))
        bars = plt.barh(range(len(feature_cols)), 
                       importances[indices], 
                       color=colors)
        plt.yticks(range(len(feature_cols)), [feature_cols[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Feature Importance Analysis', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importances[indices])):
            plt.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(join(output_path, 'feature_importance.png'), dpi=300)
        plt.close()
        print(f"Saved feature importance to {join(output_path, 'feature_importance.png')}")


def save_results_summary(results_df, best_model_name, best_accuracy, method_name, output_path):
    """Save comprehensive text summary."""
    summary_path = join(output_path, 'results_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEEPFAKE DETECTION - SPECTRAL ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: {method_name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write(f"ACCURACY: {best_accuracy*100:.2f}%\n")
        f.write("="*80 + "\n")
    
    print(f"Saved results summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train deepfake detection models with comprehensive analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Path to dataset CSV file')
    parser.add_argument('--output_dir', '-o', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = create_output_directory(args.output_dir)
    print(f"\nResults will be saved to: {output_path}\n")
    
    # Load data
    X, y, feature_cols, method_name, groups = load_and_prepare_data(args.dataset)
    
    # Split data by VIDEO ID to prevent data leakage
    print(f"\n{'='*80}")
    print("SPLITTING DATA BY VIDEO ID (preventing leakage)")
    print(f"{'='*80}")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
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
    print(f"Train labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"\nTrain label distribution:\n{y_train.value_counts()}")
    print(f"Test label distribution:\n{y_test.value_counts()}")
    print(f"\nOverlap check: {len(set(train_videos) & set(test_videos))} videos in both (should be 0!)")
    
    if len(set(train_videos) & set(test_videos)) > 0:
        print("WARNING: Video leakage detected!")
    else:
        print("No video leakage - train and test are properly separated")
    
    print(f"{'='*80}\n")
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
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
    
    # Print results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    print(f"\n" + "="*80)
    print(f"BEST MODEL: {best_model_name} (Accuracy: {best_accuracy*100:.2f}%)")
    print("="*80)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, best_model_data['y_pred'], 
                               target_names=['Real', 'Fake']))
    
    # Save models
    joblib.dump(best_model_data['model'], join(output_path, 'best_model.pkl'))
    joblib.dump(scaler, join(output_path, 'scaler.pkl'))
    print(f"\nSaved best model to {join(output_path, 'best_model.pkl')}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_roc_curves(y_test, trained_models, output_path)
    plot_confusion_matrices(y_test, trained_models, output_path)
    plot_performance_comparison(results_df, output_path)
    plot_feature_importance(best_model_data['model'], feature_cols, output_path)
    
    # Save summary
    save_results_summary(results_df, best_model_name, best_accuracy, method_name, output_path)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"All results saved to: {output_path}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

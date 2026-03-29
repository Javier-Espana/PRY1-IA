"""
Utility functions for cyberbullying detection project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from typing import Tuple, List
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load cyberbullying dataset.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Pandas DataFrame
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df


def analyze_target_distribution(df: pd.DataFrame, target_column: str = 'cyberbullying_type'):
    """
    Analyze target variable distribution.
    
    Args:
        df: DataFrame
        target_column: Name of target column
    """
    print(f"\n{'='*50}")
    print(f"Target Variable Distribution ({target_column})")
    print(f"{'='*50}")
    
    # Value counts
    distribution = df[target_column].value_counts()
    print(f"\nValue Counts:\n{distribution}")
    
    # Percentages
    percentages = df[target_column].value_counts(normalize=True) * 100
    print(f"\nPercentages:\n{percentages}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    distribution.plot(kind='bar', color='steelblue')
    plt.title('Distribution of Cyberbullying Types', fontsize=14, fontweight='bold')
    plt.xlabel('Cyberbullying Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Distribution plot saved to results/target_distribution.png")


def prepare_data(df: pd.DataFrame, 
                 text_column: str = 'tweet_text',
                 target_column: str = 'cyberbullying_type',
                 test_size: float = 0.2,
                 random_state: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepare data for model training.
    
    Args:
        df: DataFrame with preprocessed tweets
        text_column: Column with tweet text
        target_column: Column with labels
        test_size: Test set proportion
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df[text_column]
    y = df[target_column]
    
    # Convert labels to numeric if needed
    if y.dtype == 'object':
        y_mapping = {label: idx for idx, label in enumerate(y.unique())}
        y = y.map(y_mapping)
        print(f"\nLabel mapping: {y_mapping}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"\nData Split:")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, model_name: str, labels: List[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of model
        labels: Label names
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    filepath = f'results/confusion_matrix_{model_name.replace(" ", "_")}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved for {model_name}")


def plot_model_comparison(results_df: pd.DataFrame):
    """
    Plot model comparison metrics.
    
    Args:
        results_df: DataFrame with model metrics
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        results_df.sort_values(metric, ascending=True).plot(
            x='Model', y=metric, kind='barh', ax=ax, color='steelblue'
        )
        ax.set_title(f'{metric.upper()} Comparison', fontweight='bold')
        ax.set_xlabel(metric.title())
        ax.set_ylabel('')
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Model comparison plot saved to results/model_comparison.png")


def save_metrics_table(results_df: pd.DataFrame):
    """
    Save metrics table as CSV.
    
    Args:
        results_df: DataFrame with metrics
    """
    os.makedirs('results', exist_ok=True)
    filepath = 'results/model_metrics.csv'
    results_df.to_csv(filepath, index=False)
    print(f"✓ Metrics table saved to {filepath}")


def print_results_summary(results_df: pd.DataFrame):
    """Print summary of results."""
    print(f"\n{'='*70}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    print(f"{'='*70}\n")
    
    # Best model for each metric
    print("Best Models by Metric:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, metric]
        print(f"  {metric.upper()}: {best_model} ({best_score:.4f})")

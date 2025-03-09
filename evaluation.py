"""
Evaluation Module for March Madness Prediction
Provides functions for evaluating model performance and calibration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import os


def evaluate_model(model, X, y, model_name="Model"):
    """
    Evaluate a model's performance on a dataset
    
    Parameters:
    -----------
    model : object
        Trained model with predict_proba method
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    model_name : str
        Name of the model for reporting
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Generate predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    brier = brier_score_loss(y, y_pred_proba)
    log_l = log_loss(y, y_pred_proba)
    auc = roc_auc_score(y, y_pred_proba)
    
    # Print results
    print(f"{model_name} Evaluation:")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Log Loss: {log_l:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    
    return {
        'model_name': model_name,
        'brier_score': brier,
        'log_loss': log_l,
        'roc_auc': auc,
        'predictions': y_pred_proba
    }


def evaluate_multiple_models(models, X, y):
    """
    Evaluate multiple models on the same dataset
    
    Parameters:
    -----------
    models : dict
        Dictionary mapping model names to trained models
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing evaluation metrics for each model
    """
    results = []
    
    for name, model in models.items():
        # Evaluate model
        metrics = evaluate_model(model, X, y, model_name=name)
        
        # Store results
        results.append({
            'Model': name,
            'Brier Score': metrics['brier_score'],
            'Log Loss': metrics['log_loss'],
            'ROC AUC': metrics['roc_auc']
        })
    
    # Create DataFrame and sort by Brier Score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Brier Score')
    
    return results_df


def plot_calibration_curve(model, X, y, n_bins=10, model_name="Model", output_dir=None):
    """
    Plot calibration curve for a model
    
    Parameters:
    -----------
    model : object
        Trained model with predict_proba method
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    n_bins : int
        Number of bins for calibration curve
    model_name : str
        Name of the model for plotting
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Generate predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=n_bins)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot perfectly calibrated line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Plot model calibration
    ax.plot(prob_pred, prob_true, 's-', label=f'{model_name}')
    
    # Formatting
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Curve - {model_name}')
    ax.legend(loc='best')
    ax.grid(True)
    
    # Calculate and add Brier score to plot
    brier = brier_score_loss(y, y_pred_proba)
    ax.text(0.05, 0.95, f'Brier Score: {brier:.4f}', 
            transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'calibration_{model_name.replace(" ", "_")}.png'))
    
    plt.tight_layout()
    
    return fig, ax


def plot_multiple_calibration_curves(models, X, y, n_bins=10, output_dir=None):
    """
    Plot calibration curves for multiple models
    
    Parameters:
    -----------
    models : dict
        Dictionary mapping model names to trained models
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    n_bins : int
        Number of bins for calibration curve
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot perfectly calibrated line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # List to store brier scores for legend
    brier_scores = []
    
    # Plot each model
    for name, model in models.items():
        # Generate predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=n_bins)
        
        # Calculate Brier score
        brier = brier_score_loss(y, y_pred_proba)
        brier_scores.append((name, brier))
        
        # Plot model calibration
        ax.plot(prob_pred, prob_true, 's-', label=f'{name} (Brier: {brier:.4f})')
    
    # Formatting
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curves - Model Comparison')
    ax.legend(loc='best')
    ax.grid(True)
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'calibration_comparison.png'))
    
    plt.tight_layout()
    
    return fig, ax


def plot_roc_curve(model, X, y, model_name="Model", output_dir=None):
    """
    Plot ROC curve for a model
    
    Parameters:
    -----------
    model : object
        Trained model with predict_proba method
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    model_name : str
        Name of the model for plotting
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Generate predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    
    # Calculate AUC
    auc = roc_auc_score(y, y_pred_proba)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot random guessing line
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    
    # Plot model ROC curve
    ax.plot(fpr, tpr, label=f'{model_name} (AUC: {auc:.4f})')
    
    # Formatting
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc='best')
    ax.grid(True)
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'roc_{model_name.replace(" ", "_")}.png'))
    
    plt.tight_layout()
    
    return fig, ax


def plot_multiple_roc_curves(models, X, y, output_dir=None):
    """
    Plot ROC curves for multiple models
    
    Parameters:
    -----------
    models : dict
        Dictionary mapping model names to trained models
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot random guessing line
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    
    # Plot each model
    for name, model in models.items():
        # Generate predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        
        # Calculate AUC
        auc = roc_auc_score(y, y_pred_proba)
        
        # Plot model ROC curve
        ax.plot(fpr, tpr, label=f'{name} (AUC: {auc:.4f})')
    
    # Formatting
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Model Comparison')
    ax.legend(loc='best')
    ax.grid(True)
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'roc_comparison.png'))
    
    plt.tight_layout()
    
    return fig, ax


def analyze_prediction_distribution(model, X, y=None, bins=20, model_name="Model", output_dir=None):
    """
    Analyze and plot the distribution of predicted probabilities
    
    Parameters:
    -----------
    model : object
        Trained model with predict_proba method
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray, optional
        Target vector for separating correct/incorrect predictions
    bins : int
        Number of bins for histogram
    model_name : str
        Name of the model for plotting
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Generate predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot overall distribution
    ax.hist(y_pred_proba, bins=bins, alpha=0.5, label='All Predictions')
    
    # If targets provided, separate correct and incorrect predictions
    if y is not None:
        # Generate binary predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Separate correct and incorrect predictions
        correct_idx = y_pred == y
        incorrect_idx = ~correct_idx
        
        # Plot distributions
        ax.hist(y_pred_proba[correct_idx], bins=bins, alpha=0.5, label='Correct Predictions')
        ax.hist(y_pred_proba[incorrect_idx], bins=bins, alpha=0.5, label='Incorrect Predictions')
    
    # Formatting
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Predicted Probabilities - {model_name}')
    ax.legend(loc='best')
    ax.grid(True)
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'dist_{model_name.replace(" ", "_")}.png'))
    
    plt.tight_layout()
    
    return fig, ax


def analyze_feature_importance(model, feature_names, model_name="Model", top_n=20, output_dir=None):
    """
    Analyze and plot feature importance
    
    Parameters:
    -----------
    model : object
        Trained model with feature_importances_ attribute or coef_ attribute
    feature_names : list
        List of feature names
    model_name : str
        Name of the model for plotting
    top_n : int
        Number of top features to show
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Check if model has feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    # Check if model has coef_ attribute (linear models)
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    # For more complex models (e.g., stacked ensembles), try to extract from base estimator
    elif hasattr(model, 'estimator') and hasattr(model.estimator, 'feature_importances_'):
        importance = model.estimator.feature_importances_
    elif hasattr(model, 'estimator') and hasattr(model.estimator, 'coef_'):
        importance = np.abs(model.estimator.coef_[0]) if len(model.estimator.coef_.shape) > 1 else np.abs(model.estimator.coef_)
    # If can't find importance, return None
    else:
        print(f"Could not extract feature importance from {model_name}")
        return None, None
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Select top N features
    if top_n > 0 and len(feature_importance) > top_n:
        feature_importance = feature_importance.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, max(8, len(feature_importance) * 0.3)))
    
    # Plot feature importance
    ax.barh(feature_importance['Feature'], feature_importance['Importance'])
    
    # Formatting
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Feature Importance - {model_name}')
    ax.grid(True, axis='x')
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'importance_{model_name.replace(" ", "_")}.png'))
    
    plt.tight_layout()
    
    return fig, ax, feature_importance


def analyze_seed_performance(predictions, actual, seeds, output_dir=None):
    """
    Analyze model performance by seed matchups
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Predicted probabilities
    actual : numpy.ndarray
        Actual outcomes
    seeds : pandas.DataFrame
        DataFrame containing team seeds for each matchup
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing performance metrics by seed matchup
    """
    # Create a copy of the input data
    data = pd.DataFrame({
        'Prediction': predictions,
        'Actual': actual,
        'Team1Seed': seeds['Team1Seed'],
        'Team2Seed': seeds['Team2Seed']
    })
    
    # Create seed matchup groups
    data['SeedMatchup'] = data.apply(lambda x: f"{int(x['Team1Seed'])} vs {int(x['Team2Seed'])}", axis=1)
    
    # Group by seed matchup
    seed_performance = data.groupby('SeedMatchup').apply(lambda x: pd.Series({
        'Count': len(x),
        'ActualUpsetRate': x['Actual'].mean(),
        'PredictedUpsetRate': x['Prediction'].mean(),
        'Brier': brier_score_loss(x['Actual'], x['Prediction']),
        'LogLoss': log_loss(x['Actual'], x['Prediction']) if len(x) > 1 else 0
    })).reset_index()
    
    # Sort by seed matchup
    seed_performance['Seed1'] = seed_performance['SeedMatchup'].apply(lambda x: int(x.split(' vs ')[0]))
    seed_performance['Seed2'] = seed_performance['SeedMatchup'].apply(lambda x: int(x.split(' vs ')[1]))
    seed_performance = seed_performance.sort_values(['Seed1', 'Seed2'])
    
    # Plot if output directory provided
    if output_dir:
        # Create upset rate comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Select top seed matchups by count
        plot_data = seed_performance[seed_performance['Count'] >= 5].sort_values('Count', ascending=False).head(20)
        
        # Create bar positions
        x = np.arange(len(plot_data))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, plot_data['ActualUpsetRate'], width, label='Actual Upset Rate')
        ax.bar(x + width/2, plot_data['PredictedUpsetRate'], width, label='Predicted Upset Rate')
        
        # Add count labels
        for i, count in enumerate(plot_data['Count']):
            ax.text(i, 0.02, f"n={count}", ha='center', va='bottom')
        
        # Formatting
        ax.set_xlabel('Seed Matchup')
        ax.set_ylabel('Upset Rate')
        ax.set_title('Actual vs. Predicted Upset Rates by Seed Matchup')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data['SeedMatchup'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y')
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'seed_performance.png'))
        
        plt.tight_layout()
    
    return seed_performance


def evaluate_backtesting(predictions_by_year, actual_by_year, output_dir=None):
    """
    Evaluate model performance across multiple years (backtesting)
    
    Parameters:
    -----------
    predictions_by_year : dict
        Dictionary mapping years to predicted probabilities
    actual_by_year : dict
        Dictionary mapping years to actual outcomes
    output_dir : str, optional
        Directory to save plots
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing performance metrics by year
    """
    # Initialize results container
    results = []
    
    # Evaluate each year
    for year in sorted(predictions_by_year.keys()):
        if year in actual_by_year:
            y_pred = predictions_by_year[year]
            y_true = actual_by_year[year]
            
            # Calculate metrics
            brier = brier_score_loss(y_true, y_pred)
            log_l = log_loss(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            
            # Store results
            results.append({
                'Year': year,
                'Games': len(y_true),
                'Brier Score': brier,
                'Log Loss': log_l,
                'ROC AUC': auc
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot Brier Score by year
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(results_df['Year'], results_df['Brier Score'], 'o-', linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('Brier Score')
        ax.set_title('Model Performance by Year (Brier Score)')
        ax.grid(True)
        
        # Add data labels
        for i, row in results_df.iterrows():
            ax.text(row['Year'], row['Brier Score'] + 0.002, f"{row['Brier Score']:.4f}", 
                    ha='center', va='bottom')
        
        plt.savefig(os.path.join(output_dir, 'backtesting_brier.png'))
        
        # Plot all metrics by year
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        ax.plot(results_df['Year'], results_df['Brier Score'], 'o-', linewidth=2, label='Brier Score')
        ax.plot(results_df['Year'], results_df['Log Loss'], 's-', linewidth=2, label='Log Loss')
        ax.plot(results_df['Year'], 1 - results_df['ROC AUC'], '^-', linewidth=2, label='1 - AUC')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Metric Value')
        ax.set_title('Model Performance by Year')
        ax.legend()
        ax.grid(True)
        
        plt.savefig(os.path.join(output_dir, 'backtesting_all_metrics.png'))
    
    return results_df 
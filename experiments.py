"""
Experiments Module for March Madness Prediction
Provides functionality for running experiments and fine-tuning models
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, brier_score_loss, log_loss
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime

# Import the evaluation module
from evaluation import (
    evaluate_model, evaluate_multiple_models, 
    plot_calibration_curve, plot_multiple_calibration_curves,
    plot_roc_curve, plot_multiple_roc_curves,
    analyze_prediction_distribution, analyze_feature_importance
)

# Set up experiment directories
def setup_experiment_dir(base_dir='experiments'):
    """
    Set up directory structure for experiments
    
    Parameters:
    -----------
    base_dir : str
        Base directory for experiments
    
    Returns:
    --------
    str
        Path to the experiment directory
    """
    # Create timestamp for unique experiment ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f'experiment_{timestamp}')
    
    # Create directories
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'results'), exist_ok=True)
    
    return experiment_dir

# Define scoring metrics
def get_scoring_metrics():
    """
    Get scoring metrics for model evaluation
    
    Returns:
    --------
    dict
        Dictionary of scoring metrics
    """
    return {
        'brier_score': make_scorer(brier_score_loss, greater_is_better=False),
        'log_loss': make_scorer(log_loss, greater_is_better=False, needs_proba=True),
        'roc_auc': 'roc_auc'
    }

# Define models and parameter grids for hyperparameter tuning
def get_model_configs():
    """
    Get model configurations for experiments
    
    Returns:
    --------
    dict
        Dictionary mapping model names to (model, param_grid) tuples
    """
    models = {}
    
    # Logistic Regression
    models['LogisticRegression'] = (
        LogisticRegression(max_iter=1000, random_state=42),
        {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'class_weight': [None, 'balanced']
        }
    )
    
    # Random Forest
    models['RandomForest'] = (
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    )
    
    # Gradient Boosting
    models['GradientBoosting'] = (
        GradientBoostingClassifier(random_state=42),
        {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
    )
    
    # XGBoost
    models['XGBoost'] = (
        xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    )
    
    # LightGBM
    models['LightGBM'] = (
        lgb.LGBMClassifier(random_state=42),
        {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, -1],
            'num_leaves': [31, 63, 127],
            'min_child_samples': [20, 40, 60],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    )
    
    # Neural Network
    models['NeuralNetwork'] = (
        MLPClassifier(max_iter=1000, random_state=42),
        {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'solver': ['adam', 'sgd']
        }
    )
    
    return models

def run_grid_search(X_train, y_train, model, param_grid, cv=5, scoring='brier_score', n_jobs=-1):
    """
    Run grid search for hyperparameter tuning
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    model : estimator object
        Base model to tune
    param_grid : dict
        Parameter grid
    cv : int
        Number of cross-validation folds
    scoring : str or dict
        Scoring metric(s)
    n_jobs : int
        Number of parallel jobs
    
    Returns:
    --------
    GridSearchCV
        Fitted grid search object
    """
    # Set up cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Set up grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit=scoring if isinstance(scoring, str) else 'brier_score',
        cv=cv_splitter,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )
    
    # Fit grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"Grid search completed in {elapsed_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search

def run_randomized_search(X_train, y_train, model, param_distributions, n_iter=100, cv=5, scoring='brier_score', n_jobs=-1):
    """
    Run randomized search for hyperparameter tuning
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    model : estimator object
        Base model to tune
    param_distributions : dict
        Parameter distributions
    n_iter : int
        Number of parameter settings to sample
    cv : int
        Number of cross-validation folds
    scoring : str or dict
        Scoring metric(s)
    n_jobs : int
        Number of parallel jobs
    
    Returns:
    --------
    RandomizedSearchCV
        Fitted randomized search object
    """
    # Set up cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Set up randomized search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=scoring if isinstance(scoring, str) else 'brier_score',
        cv=cv_splitter,
        n_jobs=n_jobs,
        random_state=42,
        verbose=1,
        return_train_score=True
    )
    
    # Fit randomized search
    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"Randomized search completed in {elapsed_time:.2f} seconds")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {random_search.best_score_:.4f}")
    
    return random_search

def analyze_search_results(search_results, output_dir=None):
    """
    Analyze and visualize grid search or randomized search results
    
    Parameters:
    -----------
    search_results : GridSearchCV or RandomizedSearchCV
        Search results
    output_dir : str, optional
        Directory to save plots
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing results
    """
    # Get results as DataFrame
    results = pd.DataFrame(search_results.cv_results_)
    
    # Extract parameters
    param_cols = [col for col in results.columns if col.startswith('param_')]
    
    # Create parameter string for plotting
    results['params_str'] = results.apply(
        lambda row: ', '.join(f"{param.split('_')[1]}={row[param]}" 
                             for param in param_cols 
                             if not pd.isna(row[param])),
        axis=1
    )
    
    # Sort by mean test score
    results = results.sort_values('mean_test_score', ascending=False)
    
    # Display top results
    top_results = results[['params_str', 'mean_test_score', 'std_test_score']].head(10)
    print("Top 10 parameter combinations:")
    print(top_results)
    
    # Plot results if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot top N parameter combinations
        n_top = min(10, len(results))
        plt.figure(figsize=(12, 8))
        plt.errorbar(
            range(n_top),
            results['mean_test_score'].head(n_top),
            yerr=results['std_test_score'].head(n_top),
            fmt='o',
            capsize=5
        )
        plt.xticks(range(n_top), results['params_str'].head(n_top), rotation=90)
        plt.xlabel('Parameter Combination')
        plt.ylabel('Mean Test Score')
        plt.title('Top Parameter Combinations')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_parameters.png'))
        
        # If there are numeric parameters, plot their effect on score
        for param in param_cols:
            param_name = param.split('_')[1]
            # Check if parameter is numeric
            try:
                results[param] = pd.to_numeric(results[param])
                numeric = True
            except:
                numeric = False
            
            if numeric:
                plt.figure(figsize=(10, 6))
                plt.scatter(results[param], results['mean_test_score'])
                plt.xlabel(param_name)
                plt.ylabel('Mean Test Score')
                plt.title(f'Effect of {param_name} on Performance')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f'param_effect_{param_name}.png'))
    
    return results

def run_experiment(
    X_train, y_train, X_val, y_val, 
    models_to_use=None,
    search_type='random',
    n_iter=50,
    cv=5,
    experiment_dir=None
):
    """
    Run a complete experiment with multiple models and hyperparameter tuning
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    X_val : pandas.DataFrame
        Validation features
    y_val : pandas.Series
        Validation labels
    models_to_use : list, optional
        List of model names to include in the experiment
    search_type : str
        Type of search ('grid' or 'random')
    n_iter : int
        Number of iterations for randomized search
    cv : int
        Number of cross-validation folds
    experiment_dir : str, optional
        Directory to save experiment results
    
    Returns:
    --------
    dict
        Dictionary containing best models and results
    """
    # Create experiment directory if not provided
    if experiment_dir is None:
        experiment_dir = setup_experiment_dir()
    
    print(f"Running experiment in directory: {experiment_dir}")
    
    # Get all model configurations
    all_models = get_model_configs()
    
    # Filter models if specified
    if models_to_use is not None:
        all_models = {name: config for name, config in all_models.items() if name in models_to_use}
    
    # Get scoring metrics
    scoring = get_scoring_metrics()
    
    # Dictionary to store best models
    best_models = {}
    
    # Run hyperparameter tuning for each model
    for model_name, (model, param_grid) in all_models.items():
        print(f"\n{'='*50}")
        print(f"Tuning {model_name}")
        print(f"{'='*50}")
        
        # Create model-specific directories
        model_dir = os.path.join(experiment_dir, 'models', model_name)
        plot_dir = os.path.join(experiment_dir, 'plots', model_name)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Run search
        if search_type.lower() == 'grid':
            search = run_grid_search(
                X_train, y_train, 
                model, param_grid, 
                cv=cv, 
                scoring=scoring
            )
        else:  # random search
            search = run_randomized_search(
                X_train, y_train, 
                model, param_grid, 
                n_iter=n_iter, 
                cv=cv, 
                scoring=scoring
            )
        
        # Analyze search results
        analyze_search_results(
            search, 
            output_dir=os.path.join(experiment_dir, 'results', f'{model_name}_search')
        )
        
        # Get best model
        best_model = search.best_estimator_
        
        # Evaluate on validation set
        eval_results = evaluate_model(
            best_model, X_val, y_val, 
            model_name=model_name
        )
        
        # Generate plots
        plot_calibration_curve(
            best_model, X_val, y_val, 
            model_name=model_name, 
            output_dir=plot_dir
        )
        
        plot_roc_curve(
            best_model, X_val, y_val, 
            model_name=model_name, 
            output_dir=plot_dir
        )
        
        analyze_prediction_distribution(
            best_model, X_val, y_val, 
            model_name=model_name, 
            output_dir=plot_dir
        )
        
        # Analyze feature importance if available
        try:
            analyze_feature_importance(
                best_model, X_train.columns, 
                model_name=model_name, 
                output_dir=plot_dir
            )
        except:
            print(f"Could not analyze feature importance for {model_name}")
        
        # Save model
        with open(os.path.join(model_dir, f'{model_name}_best.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
        
        # Store model in results
        best_models[model_name] = {
            'model': best_model,
            'params': search.best_params_,
            'cv_score': search.best_score_,
            'val_results': eval_results
        }
    
    # Compare all models
    if len(best_models) > 1:
        models_dict = {name: config['model'] for name, config in best_models.items()}
        
        # Create comparison plots
        plot_multiple_calibration_curves(
            models_dict, X_val, y_val, 
            output_dir=os.path.join(experiment_dir, 'plots')
        )
        
        plot_multiple_roc_curves(
            models_dict, X_val, y_val, 
            output_dir=os.path.join(experiment_dir, 'plots')
        )
        
        # Evaluate all models
        comparison_df = evaluate_multiple_models(models_dict, X_val, y_val)
        
        # Save comparison
        comparison_df.to_csv(os.path.join(experiment_dir, 'results', 'model_comparison.csv'), index=False)
        
        # Display comparison
        print("\nModel Comparison:")
        print(comparison_df)
    
    # Save experiment metadata
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_used': list(all_models.keys()),
        'search_type': search_type,
        'n_iter': n_iter,
        'cv': cv,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'best_model': comparison_df.iloc[0]['Model'] if len(best_models) > 1 else list(best_models.keys())[0]
    }
    
    with open(os.path.join(experiment_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    return best_models

def ensemble_models(models, X_val, y_val, method='weighted_average', weights=None, output_dir=None):
    """
    Create an ensemble of models
    
    Parameters:
    -----------
    models : dict
        Dictionary mapping model names to trained models
    X_val : pandas.DataFrame
        Validation features
    y_val : pandas.Series
        Validation labels
    method : str
        Ensembling method ('simple_average', 'weighted_average', or 'stacking')
    weights : dict, optional
        Dictionary mapping model names to weights (for weighted_average)
    output_dir : str, optional
        Directory to save ensemble results
    
    Returns:
    --------
    dict
        Dictionary containing ensemble model and results
    """
    print(f"\n{'='*50}")
    print(f"Creating Ensemble with method: {method}")
    print(f"{'='*50}")
    
    # Get predictions from all models
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict_proba(X_val)[:, 1]
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # Ensemble prediction based on method
    if method == 'simple_average':
        ensemble_pred = pred_df.mean(axis=1).values
    
    elif method == 'weighted_average':
        # Use equal weights if not provided
        if weights is None:
            # Calculate weights based on validation performance
            # Higher weight for better models (lower Brier score)
            brier_scores = {name: brier_score_loss(y_val, pred) for name, pred in predictions.items()}
            total = sum(1/score for score in brier_scores.values())
            weights = {name: (1/score)/total for name, score in brier_scores.items()}
        
        # Apply weights
        weighted_preds = np.zeros(len(y_val))
        for name, weight in weights.items():
            weighted_preds += weight * predictions[name]
        
        ensemble_pred = weighted_preds
    
    else:  # stacking - not implemented in this simple version
        ensemble_pred = pred_df.mean(axis=1).values
    
    # Calculate ensemble performance
    ensemble_brier = brier_score_loss(y_val, ensemble_pred)
    ensemble_log_loss_val = log_loss(y_val, ensemble_pred)
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    
    print(f"Ensemble Performance:")
    print(f"  Brier Score: {ensemble_brier:.4f}")
    print(f"  Log Loss: {ensemble_log_loss_val:.4f}")
    print(f"  ROC AUC: {ensemble_auc:.4f}")
    
    # Compare to individual models
    individual_briers = {name: brier_score_loss(y_val, pred) for name, pred in predictions.items()}
    best_individual = min(individual_briers.items(), key=lambda x: x[1])
    
    print(f"Best Individual Model: {best_individual[0]} (Brier: {best_individual[1]:.4f})")
    print(f"Ensemble Improvement: {best_individual[1] - ensemble_brier:.4f}")
    
    # Save results if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save ensemble predictions
        ensemble_results = pd.DataFrame({
            'Actual': y_val,
            'Ensemble': ensemble_pred
        })
        
        # Add individual model predictions
        for name, pred in predictions.items():
            ensemble_results[name] = pred
        
        ensemble_results.to_csv(os.path.join(output_dir, 'ensemble_predictions.csv'), index=False)
        
        # Create calibration plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot perfectly calibrated line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        
        # Calculate and plot ensemble calibration
        prob_true, prob_pred = calibration_curve(y_val, ensemble_pred, n_bins=10)
        ax.plot(prob_pred, prob_true, 's-', label=f'Ensemble (Brier: {ensemble_brier:.4f})')
        
        # Plot best individual model calibration
        best_model_name = best_individual[0]
        best_model_pred = predictions[best_model_name]
        prob_true, prob_pred = calibration_curve(y_val, best_model_pred, n_bins=10)
        ax.plot(prob_pred, prob_true, 'o-', label=f'{best_model_name} (Brier: {best_individual[1]:.4f})')
        
        # Formatting
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve - Ensemble vs. Best Individual Model')
        ax.legend(loc='best')
        ax.grid(True)
        
        plt.savefig(os.path.join(output_dir, 'ensemble_calibration.png'))
    
    # Create a simple ensemble model wrapper
    class EnsembleModel:
        def __init__(self, base_models, weights=None, method='weighted_average'):
            self.base_models = base_models
            self.weights = weights
            self.method = method
        
        def predict_proba(self, X):
            # Get predictions from all models
            preds = {}
            for name, model in self.base_models.items():
                preds[name] = model.predict_proba(X)[:, 1]
            
            # Convert to DataFrame
            pred_df = pd.DataFrame(preds)
            
            # Ensemble prediction based on method
            if self.method == 'simple_average':
                ensemble_pred = pred_df.mean(axis=1).values
            
            elif self.method == 'weighted_average' and self.weights is not None:
                # Apply weights
                weighted_preds = np.zeros(len(X))
                for name, weight in self.weights.items():
                    weighted_preds += weight * preds[name]
                
                ensemble_pred = weighted_preds
            
            else:  # Default to simple average
                ensemble_pred = pred_df.mean(axis=1).values
            
            # Return in scikit-learn format (2D array with probabilities for both classes)
            result = np.zeros((len(ensemble_pred), 2))
            result[:, 1] = ensemble_pred
            result[:, 0] = 1 - ensemble_pred
            
            return result
        
        def predict(self, X):
            # Get predicted probabilities and convert to binary predictions
            probs = self.predict_proba(X)[:, 1]
            return (probs >= 0.5).astype(int)
    
    # Create ensemble model
    ensemble_model = EnsembleModel(models, weights=weights, method=method)
    
    # Save ensemble model if output directory provided
    if output_dir:
        with open(os.path.join(output_dir, 'ensemble_model.pkl'), 'wb') as f:
            pickle.dump(ensemble_model, f)
    
    return {
        'model': ensemble_model,
        'method': method,
        'weights': weights,
        'brier_score': ensemble_brier,
        'log_loss': ensemble_log_loss_val,
        'roc_auc': ensemble_auc,
        'improvement': best_individual[1] - ensemble_brier
    }

def feature_selection_experiment(X_train, y_train, X_val, y_val, base_model, feature_step=5, min_features=10, output_dir=None):
    """
    Run feature selection experiment to find optimal feature set
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    X_val : pandas.DataFrame
        Validation features
    y_val : pandas.Series
        Validation labels
    base_model : estimator object
        Base model to use
    feature_step : int
        Number of features to add at each step
    min_features : int
        Minimum number of features to consider
    output_dir : str, optional
        Directory to save results
    
    Returns:
    --------
    dict
        Dictionary containing results
    """
    print(f"\n{'='*50}")
    print(f"Running Feature Selection Experiment")
    print(f"{'='*50}")
    
    # Get feature importance if available
    try:
        # Train base model on all features
        base_model.fit(X_train, y_train)
        
        # Get feature importance
        if hasattr(base_model, 'feature_importances_'):
            importance = base_model.feature_importances_
        elif hasattr(base_model, 'coef_'):
            importance = np.abs(base_model.coef_[0]) if len(base_model.coef_.shape) > 1 else np.abs(base_model.coef_)
        else:
            raise AttributeError("Model doesn't have feature_importances_ or coef_ attribute")
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Print top features
        print("Top 20 Features:")
        print(feature_importance.head(20))
        
        # Run experiment with different feature counts
        results = []
        feature_sets = {}
        
        # Try different feature counts
        max_features = len(X_train.columns)
        feature_counts = list(range(min_features, max_features + feature_step, feature_step))
        if max_features not in feature_counts:
            feature_counts.append(max_features)
        
        for n_features in feature_counts:
            n_features = min(n_features, max_features)
            
            # Get top features
            top_features = feature_importance['Feature'].head(n_features).tolist()
            feature_sets[n_features] = top_features
            
            # Create datasets with selected features
            X_train_selected = X_train[top_features]
            X_val_selected = X_val[top_features]
            
            # Train model
            model = clone(base_model)
            model.fit(X_train_selected, y_train)
            
            # Evaluate model
            y_pred_proba = model.predict_proba(X_val_selected)[:, 1]
            brier = brier_score_loss(y_val, y_pred_proba)
            log_l = log_loss(y_val, y_pred_proba)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # Store results
            results.append({
                'NumFeatures': n_features,
                'Features': top_features,
                'Brier': brier,
                'LogLoss': log_l,
                'AUC': auc
            })
            
            print(f"Features: {n_features}, Brier: {brier:.4f}, Log Loss: {log_l:.4f}, AUC: {auc:.4f}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find optimal feature count
        best_result = results_df.loc[results_df['Brier'].idxmin()]
        optimal_features = best_result['NumFeatures']
        optimal_feature_set = feature_sets[optimal_features]
        
        print(f"\nOptimal Feature Count: {optimal_features}")
        print(f"Optimal Brier Score: {best_result['Brier']:.4f}")
        
        # Plot results if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot metrics vs feature count
            plt.figure(figsize=(12, 8))
            plt.plot(results_df['NumFeatures'], results_df['Brier'], 'o-', label='Brier Score')
            plt.plot(results_df['NumFeatures'], results_df['LogLoss'], 's-', label='Log Loss')
            plt.plot(results_df['NumFeatures'], 1 - results_df['AUC'], '^-', label='1 - AUC')
            
            plt.axvline(x=optimal_features, color='r', linestyle='--', 
                       label=f'Optimal: {optimal_features} features')
            
            plt.xlabel('Number of Features')
            plt.ylabel('Metric Value')
            plt.title('Model Performance vs. Number of Features')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(os.path.join(output_dir, 'feature_selection.png'))
            
            # Save results
            results_df.to_csv(os.path.join(output_dir, 'feature_selection_results.csv'), index=False)
            
            # Save optimal feature set
            with open(os.path.join(output_dir, 'optimal_features.txt'), 'w') as f:
                for feature in optimal_feature_set:
                    f.write(f"{feature}\n")
        
        return {
            'optimal_feature_count': optimal_features,
            'optimal_feature_set': optimal_feature_set,
            'results': results_df,
            'feature_importance': feature_importance
        }
        
    except Exception as e:
        print(f"Feature selection experiment failed: {str(e)}")
        return None

# Main execution function
def main():
    """
    Main function to run experiments
    """
    print("March Madness Prediction Experiments")
    print("====================================")
    
    # Example usage - this would be replaced with actual data loading
    print("Please load your data and call the experiment functions as needed.")
    print("Example usage:")
    print("    experiment_dir = setup_experiment_dir()")
    print("    best_models = run_experiment(X_train, y_train, X_val, y_val, experiment_dir=experiment_dir)")
    print("    ensemble = ensemble_models(best_models, X_val, y_val, output_dir=os.path.join(experiment_dir, 'ensemble'))")

if __name__ == "__main__":
    main() 
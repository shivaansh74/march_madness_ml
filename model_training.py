"""
Model Training Module for March Madness Prediction
Provides functions for training, evaluating, and ensembling models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import joblib
import os


def train_base_models(X, y, cv=5, random_state=42):
    """
    Train a set of base models with cross-validation
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary of trained models
    """
    models = {}
    
    # Define base models
    base_models = {
        'lr': LogisticRegression(random_state=random_state, max_iter=1000),
        'rf': RandomForestClassifier(random_state=random_state),
        'gb': GradientBoostingClassifier(random_state=random_state),
        'xgb': xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
        'lgb': lgb.LGBMClassifier(random_state=random_state),
        'cb': cb.CatBoostClassifier(random_state=random_state, verbose=0)
    }
    
    # Define parameter grids for each model
    param_grids = {
        'lr': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'rf': {
            'n_estimators': [200, 300, 500],
            'max_depth': [5, 8, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gb': {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 8],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        },
        'xgb': {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 8],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'lgb': {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 8, -1],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 0.9, 1.0]
        },
        'cb': {
            'iterations': [200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [3, 5, 8],
            'subsample': [0.8, 0.9, 1.0],
            'l2_leaf_reg': [1, 3, 5, 10]
        }
    }
    
    # Train each model with grid search and calibration
    for name, model in base_models.items():
        print(f"Training {name} with grid search...")
        
        # Define cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=cv_strategy,
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Get best model from grid search
        best_model = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        
        # Apply probability calibration
        calibrated_model = CalibratedClassifierCV(
            best_model,
            method='isotonic',
            cv=cv_strategy
        )
        
        calibrated_model.fit(X, y)
        
        # Store the calibrated model
        models[name] = calibrated_model
        
        # Calculate and print model performance
        y_pred_proba = calibrated_model.predict_proba(X)[:, 1]
        brier = brier_score_loss(y, y_pred_proba)
        log_l = log_loss(y, y_pred_proba)
        
        print(f"{name} - Brier Score: {brier:.4f}, Log Loss: {log_l:.4f}")
    
    return models


def hyperparameter_optimization(X, y, model_type, n_trials=50, cv=5, random_state=42):
    """
    Optimize hyperparameters using Optuna
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    model_type : str
        Type of model to optimize ('xgb', 'lgb', 'cb')
    n_trials : int
        Number of optimization trials
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Best hyperparameters found
    """
    def create_model(trial, model_type):
        if model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': random_state
            }
            return xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
        
        elif model_type == 'lgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': random_state
            }
            return lgb.LGBMClassifier(**params)
        
        elif model_type == 'cb':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'random_strength': trial.suggest_float('random_strength', 0.1, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
                'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
                'random_seed': random_state,
                'verbose': 0
            }
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
            elif params['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
                
            return cb.CatBoostClassifier(**params)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def objective(trial):
        # Create model with current hyperparameters
        model = create_model(trial, model_type)
        
        # Define cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # Perform cross-validation
        scores = []
        for train_idx, val_idx in cv_strategy.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_val_pred = model.predict_proba(X_val)[:, 1]
            
            # Evaluate using Brier score
            score = brier_score_loss(y_val, y_val_pred)
            scores.append(score)
        
        # Return average score across folds
        return np.mean(scores)
    
    # Create Optuna study
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial for {model_type}:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value:.4f}")
    print(f"  Params: {best_trial.params}")
    
    return best_trial.params


def train_stacked_model(X, y, base_models, cv=5, random_state=42):
    """
    Train a stacking ensemble model
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    base_models : dict
        Dictionary of base models to use in the ensemble
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing the trained stacking model and metadata
    """
    print("Training stacked ensemble model...")
    
    # Convert X and y to numpy arrays if they're not already
    X = np.array(X)
    y = np.array(y)
    
    # Define cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Generate out-of-fold predictions for each base model
    meta_features = np.zeros((X.shape[0], len(base_models)))
    
    for i, (name, model) in enumerate(base_models.items()):
        print(f"Generating meta-features for {name}...")
        
        # Create out-of-fold predictions
        oof_preds = np.zeros(X.shape[0])
        
        for train_idx, val_idx in cv_strategy.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # Clone and train the model
            model_clone = joblib.clone(model)
            model_clone.fit(X_train, y_train)
            
            # Predict on validation fold
            val_preds = model_clone.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_preds
        
        # Store out-of-fold predictions as meta-features
        meta_features[:, i] = oof_preds
        
        # Evaluate this model's performance
        brier = brier_score_loss(y, oof_preds)
        log_l = log_loss(y, oof_preds)
        print(f"{name} OOF - Brier Score: {brier:.4f}, Log Loss: {log_l:.4f}")
    
    # Train meta-learner
    meta_learner = LogisticRegression(random_state=random_state, C=0.1, solver='liblinear')
    meta_learner.fit(meta_features, y)
    
    # Get coefficients (importance of each base model)
    model_coefs = {name: coef for name, coef in zip(base_models.keys(), meta_learner.coef_[0])}
    print("Meta-learner coefficients (model importance):")
    for name, coef in sorted(model_coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {coef:.4f}")
    
    # Train full base models on all data
    full_models = {}
    for name, model in base_models.items():
        model_clone = joblib.clone(model)
        model_clone.fit(X, y)
        full_models[name] = model_clone
    
    # Predict with stacked model
    meta_preds = meta_learner.predict_proba(meta_features)[:, 1]
    
    # Evaluate stacked model
    stacked_brier = brier_score_loss(y, meta_preds)
    stacked_log_loss_val = log_loss(y, meta_preds)
    print(f"Stacked Model - Brier Score: {stacked_brier:.4f}, Log Loss: {stacked_log_loss_val:.4f}")
    
    # Return stacked model components
    return {
        'meta_learner': meta_learner,
        'base_models': full_models,
        'model_coefs': model_coefs
    }


def evaluate_models(models, X, y):
    """
    Evaluate multiple models on the same data
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to evaluate
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
        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        brier = brier_score_loss(y, y_pred_proba)
        log_l = log_loss(y, y_pred_proba)
        
        # Store results
        results.append({
            'Model': name,
            'Brier Score': brier,
            'Log Loss': log_l
        })
    
    # Create DataFrame and sort by Brier Score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Brier Score')
    
    return results_df


def calibrate_probabilities(models, X, y, cv=5, random_state=42):
    """
    Apply probability calibration to models
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to calibrate
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary of calibrated models
    """
    calibrated_models = {}
    
    # Define cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    for name, model in models.items():
        print(f"Calibrating {name}...")
        
        # Apply isotonic calibration
        calibrated_model = CalibratedClassifierCV(
            model,
            method='isotonic',
            cv=cv_strategy
        )
        
        calibrated_model.fit(X, y)
        calibrated_models[name] = calibrated_model
        
        # Evaluate calibrated model
        y_pred_proba = calibrated_model.predict_proba(X)[:, 1]
        brier = brier_score_loss(y, y_pred_proba)
        log_l = log_loss(y, y_pred_proba)
        
        print(f"{name} (Calibrated) - Brier Score: {brier:.4f}, Log Loss: {log_l:.4f}")
    
    return calibrated_models


def create_weighted_ensemble(models, X, y, weights=None):
    """
    Create a weighted ensemble of models
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to include in the ensemble
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix used to determine optimal weights if not provided
    y : pandas.Series or numpy.ndarray
        Target vector
    weights : dict, optional
        Dictionary mapping model names to weights
    
    Returns:
    --------
    dict
        Dictionary containing ensemble information
    """
    # If weights not provided, optimize them
    if weights is None:
        # Generate predictions from each model
        preds = {}
        for name, model in models.items():
            preds[name] = model.predict_proba(X)[:, 1]
        
        # Optimize weights using logistic regression
        meta_X = np.column_stack([preds[name] for name in models.keys()])
        meta_model = LogisticRegression(fit_intercept=False)
        meta_model.fit(meta_X, y)
        
        # Extract weights
        weights = {name: coef for name, coef in zip(models.keys(), meta_model.coef_[0])}
        
        # Normalize weights to sum to 1
        weight_sum = sum(abs(w) for w in weights.values())
        weights = {name: abs(w) / weight_sum for name, w in weights.items()}
    
    # Print weights
    print("Ensemble weights:")
    for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {weight:.4f}")
    
    # Create ensemble prediction function
    def predict_proba(X_new):
        # Generate predictions from each model
        preds = {name: model.predict_proba(X_new)[:, 1] for name, model in models.items()}
        
        # Create weighted average
        weighted_preds = np.zeros(X_new.shape[0])
        for name, weight in weights.items():
            weighted_preds += weight * preds[name]
        
        # Return probabilities in the format expected by sklearn
        return np.column_stack([1 - weighted_preds, weighted_preds])
    
    # Test ensemble performance
    ensemble_preds = predict_proba(X)[:, 1]
    ensemble_brier = brier_score_loss(y, ensemble_preds)
    ensemble_log_loss_val = log_loss(y, ensemble_preds)
    
    print(f"Weighted Ensemble - Brier Score: {ensemble_brier:.4f}, Log Loss: {ensemble_log_loss_val:.4f}")
    
    # Return ensemble information
    return {
        'models': models,
        'weights': weights,
        'predict_proba': predict_proba
    }


def save_models(models, filename, output_dir='.'):
    """
    Save trained models to file
    
    Parameters:
    -----------
    models : dict or object
        Models to save
    filename : str
        Filename to save models to
    output_dir : str
        Directory to save models in
    """
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(models, filepath)
    print(f"Saved models to {filepath}")


def load_models(filename, output_dir='.'):
    """
    Load trained models from file
    
    Parameters:
    -----------
    filename : str
        Filename to load models from
    output_dir : str
        Directory to load models from
    
    Returns:
    --------
    dict or object
        Loaded models
    """
    filepath = os.path.join(output_dir, filename)
    models = joblib.load(filepath)
    print(f"Loaded models from {filepath}")
    
    return models 
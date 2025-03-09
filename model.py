#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model training and prediction module for March Madness analysis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
import xgboost as xgb
import lightgbm as lgb
from bayes_opt import BayesianOptimization
import joblib
import os
import pickle
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def train_model(X_train, y_train, model_type="ensemble", optimization="bayesian"):
    """
    Train a prediction model for March Madness outcomes with advanced techniques.
    
    Parameters
    ----------
    X_train : DataFrame
        Feature matrix for training
    y_train : Series
        Target variable (game outcomes)
    model_type : str, optional
        Type of model to train ("random_forest", "gradient_boosting", "xgboost", "lightgbm", "ensemble", "stacking")
    optimization : str, optional
        Optimization method ("grid", "random", "bayesian")
        
    Returns
    -------
    object
        Trained model
    """
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print(f"Training {model_type} model with {optimization} optimization...")
    
    # Select model type and optimization method
    if model_type == "random_forest":
        model = train_random_forest(X_train_scaled, y_train, optimization)
    elif model_type == "gradient_boosting":
        model = train_gradient_boosting(X_train_scaled, y_train, optimization)
    elif model_type == "xgboost":
        model = train_xgboost(X_train_scaled, y_train, optimization)
    elif model_type == "lightgbm":
        model = train_lightgbm(X_train_scaled, y_train, optimization)
    elif model_type == "ensemble":
        model = train_ensemble(X_train_scaled, y_train)
    elif model_type == "stacking":
        model = train_stacking_ensemble(X_train_scaled, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Save model and scaler
    joblib.dump(model, f"models/{model_type}_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    # Save feature names for later use
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)
    
    # Calculate feature importance if the model supports it
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Save feature importances
        feature_importance.to_csv("models/feature_importance.csv", index=False)
        
        print("\nTop 15 most important features:")
        print(feature_importance.head(15))
    
    # Evaluate on training data
    train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"\nTraining accuracy: {train_acc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return model

def train_random_forest(X_train, y_train, optimization="bayesian"):
    """Train an optimized Random Forest model."""
    if optimization == "grid":
        # Grid search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
        
    elif optimization == "random":
        # Random search
        param_distributions = {
            'n_estimators': np.arange(100, 1000, 50),
            'max_depth': np.arange(5, 50, 5),
            'min_samples_split': np.arange(2, 20, 2),
            'min_samples_leaf': np.arange(1, 10),
            'max_features': ['sqrt', 'log2', None]
        }
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
        
    elif optimization == "bayesian":
        # Bayesian optimization
        def rf_objective(n_estimators, max_depth, min_samples_split, min_samples_leaf):
            max_depth = int(max_depth)
            n_estimators = int(n_estimators)
            min_samples_split = int(min_samples_split)
            min_samples_leaf = int(min_samples_leaf)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            
            # Use cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            return cv_scores.mean()
        
        # Define parameter bounds
        pbounds = {
            'n_estimators': (100, 500),
            'max_depth': (5, 30),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 5)
        }
        
        optimizer = BayesianOptimization(f=rf_objective, pbounds=pbounds, random_state=42)
        optimizer.maximize(init_points=10, n_iter=20)
        
        # Get best parameters
        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
    
    else:
        # Default parameters
        model = RandomForestClassifier(
            n_estimators=300, 
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    # Train the model
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train, optimization="bayesian"):
    """Train an optimized Gradient Boosting model."""
    if optimization == "grid":
        # Grid search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
        model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        model = GradientBoostingClassifier(random_state=42, **best_params)
        
    elif optimization == "random":
        # Random search
        param_distributions = {
            'n_estimators': np.arange(100, 500, 50),
            'learning_rate': np.logspace(-3, 0, 10),
            'max_depth': np.arange(3, 10),
            'min_samples_split': np.arange(2, 20, 2),
            'subsample': np.linspace(0.5, 1.0, 6)
        }
        model = GradientBoostingClassifier(random_state=42)
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        model = GradientBoostingClassifier(random_state=42, **best_params)
        
    elif optimization == "bayesian":
        # Bayesian optimization
        def gb_objective(n_estimators, learning_rate, max_depth, min_samples_split, subsample):
            max_depth = int(max_depth)
            n_estimators = int(n_estimators)
            min_samples_split = int(min_samples_split)
            
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                subsample=subsample,
                random_state=42
            )
            
            # Use cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            return cv_scores.mean()
        
        # Define parameter bounds
        pbounds = {
            'n_estimators': (100, 500),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'min_samples_split': (2, 10),
            'subsample': (0.7, 1.0)
        }
        
        optimizer = BayesianOptimization(f=gb_objective, pbounds=pbounds, random_state=42)
        optimizer.maximize(init_points=10, n_iter=20)
        
        # Get best parameters
        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        
        model = GradientBoostingClassifier(random_state=42, **best_params)
    
    else:
        # Default parameters
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
    
    # Train the model
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, optimization="bayesian"):
    """Train an optimized XGBoost model."""
    if optimization == "grid":
        # Grid search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **best_params)
        
    elif optimization == "random":
        # Random search
        param_distributions = {
            'n_estimators': np.arange(100, 1000, 50),
            'learning_rate': np.logspace(-3, 0, 10),
            'max_depth': np.arange(3, 12),
            'min_child_weight': np.arange(1, 10),
            'subsample': np.linspace(0.5, 1.0, 6),
            'colsample_bytree': np.linspace(0.5, 1.0, 6),
            'gamma': np.linspace(0, 1, 6)
        }
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **best_params)
        
    elif optimization == "bayesian":
        # Bayesian optimization
        def xgb_objective(n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma):
            max_depth = int(max_depth)
            n_estimators = int(n_estimators)
            min_child_weight = int(min_child_weight)
            
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            # Use cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            return cv_scores.mean()
        
        # Define parameter bounds
        pbounds = {
            'n_estimators': (100, 500),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'min_child_weight': (1, 10),
            'subsample': (0.7, 1.0),
            'colsample_bytree': (0.7, 1.0),
            'gamma': (0, 1)
        }
        
        optimizer = BayesianOptimization(f=xgb_objective, pbounds=pbounds, random_state=42)
        optimizer.maximize(init_points=10, n_iter=20)
        
        # Get best parameters
        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_child_weight'] = int(best_params['min_child_weight'])
        
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **best_params)
    
    else:
        # Default parameters
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    # Train the model
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, optimization="bayesian"):
    """Train an optimized LightGBM model."""
    if optimization == "grid":
        # Grid search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, -1],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        model = lgb.LGBMClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        model = lgb.LGBMClassifier(random_state=42, **best_params)
        
    elif optimization == "random":
        # Random search
        param_distributions = {
            'n_estimators': np.arange(100, 1000, 50),
            'learning_rate': np.logspace(-3, 0, 10),
            'max_depth': [-1] + list(np.arange(3, 12)),
            'num_leaves': [15, 31, 63, 127, 255],
            'subsample': np.linspace(0.5, 1.0, 6),
            'colsample_bytree': np.linspace(0.5, 1.0, 6)
        }
        model = lgb.LGBMClassifier(random_state=42)
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        model = lgb.LGBMClassifier(random_state=42, **best_params)
        
    elif optimization == "bayesian":
        # Bayesian optimization
        def lgb_objective(n_estimators, learning_rate, num_leaves, subsample, colsample_bytree):
            n_estimators = int(n_estimators)
            num_leaves = int(num_leaves)
            
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42
            )
            
            # Use cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            return cv_scores.mean()
        
        # Define parameter bounds
        pbounds = {
            'n_estimators': (100, 500),
            'learning_rate': (0.01, 0.3),
            'num_leaves': (31, 255),
            'subsample': (0.7, 1.0),
            'colsample_bytree': (0.7, 1.0)
        }
        
        optimizer = BayesianOptimization(f=lgb_objective, pbounds=pbounds, random_state=42)
        optimizer.maximize(init_points=10, n_iter=20)
        
        # Get best parameters
        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['num_leaves'] = int(best_params['num_leaves'])
        
        model = lgb.LGBMClassifier(random_state=42, **best_params)
    
    else:
        # Default parameters
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    # Train the model
    model.fit(X_train, y_train)
    return model

def train_ensemble(X_train, y_train):
    """Train a voting ensemble of multiple models."""
    # Define base models
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
    lgb_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42)
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        voting='soft'  # Use predicted probabilities
    )
    
    # Train the ensemble
    ensemble.fit(X_train, y_train)
    return ensemble

def train_stacking_ensemble(X_train, y_train):
    """Train a stacking ensemble with multiple base models and a meta-learner."""
    # Define base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42))
    ]
    
    # Define meta-learner
    meta_learner = LogisticRegression(random_state=42)
    
    # Create stacking classifier
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba'
    )
    
    # Train the stacking ensemble
    stacking.fit(X_train, y_train)
    return stacking

def calibrate_model_probabilities(model, X_val, y_val):
    """
    Calibrate model probabilities to improve the reliability of predicted probabilities.
    
    Parameters
    ----------
    model : object
        Trained model
    X_val : DataFrame
        Validation features
    y_val : Series
        Validation targets
        
    Returns
    -------
    object
        Calibrated model
    """
    from sklearn.calibration import CalibratedClassifierCV
    
    # Create a calibrated model
    calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
    calibrated_model.fit(X_val, y_val)
    
    return calibrated_model

def make_predictions(model, teams_df, use_calibration=True):
    """
    Generate predictions for all possible matchups in the tournament.
    
    Parameters
    ----------
    model : object
        Trained prediction model
    teams_df : DataFrame
        Team information
    use_calibration : bool, optional
        Whether to use probability calibration
        
    Returns
    -------
    DataFrame
        Predictions for all possible matchups
    """
    from feature_engineering import create_features
    from data_loader import get_sample_data
    
    # Get the latest teams data
    team_ids = teams_df['team_id'].unique()
    
    # Create all possible matchups
    matchups = []
    for i, team1_id in enumerate(team_ids):
        for team2_id in team_ids[i+1:]:
            matchups.append({
                'team1_id': team1_id,
                'team2_id': team2_id
            })
    
    matchups_df = pd.DataFrame(matchups)
    
    # Load scaler
    try:
        _, scaler = load_model()
        if scaler is None:
            raise FileNotFoundError("Scaler not found")
    except:
        print("Warning: No scaler found. Using raw features.")
        scaler = None
    
    # Try to load existing features
    try:
        # This is just a placeholder - in real scenario you would
        # create features for these matchups using actual data
        _, games_df, tourney_results_df = get_sample_data()
        features_df = create_features(teams_df, games_df, tourney_results_df)
        
        # Extract just the feature columns
        feature_cols = [col for col in features_df.columns if col.endswith('_diff')]
        X = features_df[feature_cols]
        
        # Scale features if scaler is available
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        probabilities = model.predict_proba(X_scaled)
        
        # Add predictions to matchups
        predictions = features_df.copy()
        predictions['win_probability'] = probabilities[:, 1]  # Probability of team1 winning
        
        # Create output format
        results = pd.DataFrame({
            'team1_id': predictions['team1_id'],
            'team2_id': predictions['team2_id'],
            'team1_win_prob': predictions['win_probability'],
            'team2_win_prob': 1 - predictions['win_probability']
        })
        
        # Add team names
        results = pd.merge(results, teams_df[['team_id', 'team_name']], 
                        left_on='team1_id', right_on='team_id', how='left')
        results = results.rename(columns={'team_name': 'team1_name'}).drop('team_id', axis=1)
        
        results = pd.merge(results, teams_df[['team_id', 'team_name']], 
                        left_on='team2_id', right_on='team_id', how='left')
        results = results.rename(columns={'team_name': 'team2_name'}).drop('team_id', axis=1)
        
        # Sort by win probability
        results = results.sort_values('team1_win_prob', ascending=False)
        
        # Save results
        os.makedirs("results", exist_ok=True)
        results.to_csv("results/matchup_predictions.csv", index=False)
        
        print(f"Generated predictions for {len(results)} potential matchups.")
        
        return results
    
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return None

def simulate_tournament(teams_df, predictions_df, num_simulations=10000):
    """
    Simulate the entire tournament multiple times to get probabilities of outcomes.
    
    Parameters
    ----------
    teams_df : DataFrame
        Team information with seeds
    predictions_df : DataFrame
        Matchup predictions
    num_simulations : int
        Number of tournament simulations to run
        
    Returns
    -------
    DataFrame
        Tournament outcome probabilities for each team
    """
    # This is a placeholder for tournament simulation logic
    # A full implementation would require tournament bracket structure
    
    print(f"Simulating tournament {num_simulations} times...")
    
    # Initialize results
    results = pd.DataFrame({
        'team_id': teams_df['team_id'],
        'team_name': teams_df['team_name'],
        'round_of_64': 0,
        'round_of_32': 0,
        'sweet_16': 0,
        'elite_8': 0,
        'final_4': 0,
        'championship_game': 0,
        'champion': 0
    })
    
    # Setup tournament structure
    # In a real implementation, you would set up the initial bracket based on seeds
    # For simplicity, we'll just simulate random outcomes based on our prediction model
    
    for _ in tqdm(range(num_simulations), desc="Running simulations"):
        # Example simulation logic (simplified)
        remaining_teams = teams_df['team_id'].tolist()
        
        # For each round
        for round_num, round_name in enumerate(['round_of_64', 'round_of_32', 'sweet_16', 'elite_8', 'final_4', 'championship_game']):
            # Count teams that made it to this round
            for team_id in remaining_teams:
                results.loc[results['team_id'] == team_id, round_name] += 1
            
            # Simulate matchups for this round
            winners = []
            
            # Process each pair of teams
            for i in range(0, len(remaining_teams), 2):
                if i + 1 < len(remaining_teams):
                    team1_id = remaining_teams[i]
                    team2_id = remaining_teams[i + 1]
                    
                    # Find prediction for this matchup
                    matchup = predictions_df[
                        ((predictions_df['team1_id'] == team1_id) & (predictions_df['team2_id'] == team2_id)) |
                        ((predictions_df['team1_id'] == team2_id) & (predictions_df['team2_id'] == team1_id))
                    ]
                    
                    if len(matchup) > 0:
                        # Get win probability
                        if matchup.iloc[0]['team1_id'] == team1_id:
                            win_prob = matchup.iloc[0]['team1_win_prob']
                        else:
                            win_prob = matchup.iloc[0]['team2_win_prob']
                        
                        # Simulate game outcome
                        if np.random.random() < win_prob:
                            winners.append(team1_id)
                        else:
                            winners.append(team2_id)
                    else:
                        # If no prediction available, random winner
                        winners.append(np.random.choice([team1_id, team2_id]))
                else:
                    # Odd number of teams (shouldn't happen in real tournament)
                    winners.append(remaining_teams[i])
            
            # Update remaining teams for next round
            remaining_teams = winners
        
        # Record champion
        if remaining_teams:
            results.loc[results['team_id'] == remaining_teams[0], 'champion'] += 1
    
    # Convert to probabilities
    for col in ['round_of_64', 'round_of_32', 'sweet_16', 'elite_8', 'final_4', 'championship_game', 'champion']:
        results[col] = results[col] / num_simulations
    
    # Sort by championship probability
    results = results.sort_values('champion', ascending=False)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/tournament_simulation.csv", index=False)
    
    return results

def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluate model performance using multiple metrics.
    
    Parameters
    ----------
    model : object
        Trained model
    X_test : DataFrame
        Test features
    y_test : Series
        Test targets
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'log_loss': log_loss(y_test, model.predict_proba(X_test)),
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def load_model(model_path="models/ensemble_model.pkl", scaler_path="models/scaler.pkl"):
    """
    Load a trained model and scaler.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model
    scaler_path : str
        Path to the saved scaler
        
    Returns
    -------
    tuple
        (model, scaler)
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        print(f"Model or scaler file not found. Please train a model first.")
        return None, None

if __name__ == "__main__":
    # Test model training with sample data
    from data_loader import get_sample_data
    from feature_engineering import create_features
    from sklearn.model_selection import train_test_split
    
    print("Testing advanced modeling capabilities...")
    
    # Load sample data
    teams_df, games_df, tourney_results_df = get_sample_data()
    
    # Create features
    features_df = create_features(teams_df, games_df, tourney_results_df)
    
    # Split data
    feature_cols = [col for col in features_df.columns if col.endswith('_diff')]
    X = features_df[feature_cols]
    y = features_df['outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model (choose from random_forest, gradient_boosting, xgboost, lightgbm, ensemble, stacking)
    model = train_model(X_train, y_train, model_type="ensemble", optimization="bayesian")
    
    # Evaluate model
    evaluate_model_performance(model, X_test, y_test)
    
    # Generate predictions
    predictions = make_predictions(model, teams_df)
    
    # Simulate tournament
    tournament_results = simulate_tournament(teams_df, predictions, num_simulations=100)
    
    print("\nModel training and testing complete!") 
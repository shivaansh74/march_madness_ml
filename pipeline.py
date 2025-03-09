"""
March Madness Prediction Pipeline
Main script for end-to-end training and prediction
"""

import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import joblib
import logging

from data_preparation import (
    load_data, clean_and_combine_data, preprocess_seeds,
    create_seed_lookup, create_matchup_data, scale_features,
    create_submission_template, save_submission
)
from feature_engineering import (
    calculate_advanced_team_stats, create_matchup_features,
    calculate_conference_strength, create_conference_features,
    create_historical_matchup_features
)
from model_training import (
    train_base_models, hyperparameter_optimization, 
    train_stacked_model, evaluate_models, calibrate_probabilities,
    create_weighted_ensemble, save_models, load_models
)
from evaluation import (
    evaluate_model, plot_calibration_curve, plot_roc_curve,
    analyze_prediction_distribution, analyze_feature_importance,
    analyze_seed_performance, evaluate_backtesting
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarchMadnessPipeline:
    """Pipeline for March Madness prediction"""
    
    def __init__(self, data_dir="data", output_dir="output", models_dir="models"):
        """
        Initialize the pipeline
        
        Parameters:
        -----------
        data_dir : str
            Directory containing data files
        output_dir : str
            Directory for output files
        models_dir : str
            Directory for model storage
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.models_dir = models_dir
        
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize data containers
        self.raw_data = None
        self.data = None
        self.team_stats = {}
        self.conf_strength = {}
        
        # Initialize model containers
        self.models = {}
        self.ensemble = None
        self.scalers = {}
        
        logger.info(f"Initialized pipeline with data_dir={data_dir}, output_dir={output_dir}, models_dir={models_dir}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess data"""
        logger.info("Loading data...")
        
        # Load raw data
        self.raw_data = load_data(self.data_dir)
        
        # Clean and combine datasets
        self.data = clean_and_combine_data(self.raw_data)
        
        # Preprocess seeds
        if 'TourneySeeds' in self.data:
            self.data['TourneySeeds'] = preprocess_seeds(self.data['TourneySeeds'])
            self.seed_lookup = create_seed_lookup(self.data['TourneySeeds'])
        else:
            self.seed_lookup = {}
            logger.warning("No seed data found!")
        
        # Log data loading status
        for key, df in self.data.items():
            logger.info(f"Loaded {key}: {len(df)} rows")
    
    def engineer_features(self):
        """Engineer features for modeling"""
        logger.info("Engineering features...")
        
        # Calculate team statistics
        if 'RegularSeasonResults' in self.data and 'Teams' in self.data:
            logger.info("Calculating advanced team statistics...")
            self.team_stats = calculate_advanced_team_stats(
                self.data['RegularSeasonResults'],
                self.data['Teams']
            )
        
        # Calculate conference strength
        if 'RegularSeasonResults' in self.data and 'Teams' in self.data:
            logger.info("Calculating conference strength...")
            self.conf_strength = calculate_conference_strength(
                self.data['RegularSeasonResults'],
                self.data['Teams']
            )
        
        # Create matchup data for training
        if 'TourneyResults' in self.data:
            logger.info("Creating matchup data...")
            self.matchup_data = create_matchup_data(
                self.data['TourneyResults'],
                self.seed_lookup
            )
            logger.info(f"Created matchup data with {len(self.matchup_data)} rows")
        else:
            self.matchup_data = None
            logger.warning("No tournament results found for training!")
    
    def create_train_test_split(self, test_years):
        """
        Create train/test split based on years
        
        Parameters:
        -----------
        test_years : list
            List of years to use for testing
        
        Returns:
        --------
        tuple
            Tuple containing X_train, X_test, y_train, y_test
        """
        if self.matchup_data is None:
            logger.error("No matchup data available for training!")
            return None
        
        logger.info(f"Creating train/test split with test years: {test_years}")
        
        # Split data by year
        train_data = self.matchup_data[~self.matchup_data['Season'].isin(test_years)]
        test_data = self.matchup_data[self.matchup_data['Season'].isin(test_years)]
        
        # Process training data
        X_train_raw = train_data.drop(['Result', 'Season', 'Team1', 'Team2'], axis=1)
        y_train = train_data['Result']
        
        # Process test data
        X_test_raw = test_data.drop(['Result', 'Season', 'Team1', 'Team2'], axis=1)
        y_test = test_data['Result']
        
        # Scale features
        X_train, X_test, scaler = scale_features(X_train_raw, X_test_raw)
        self.scalers['main'] = scaler
        
        # Split by gender
        # Men's data
        men_train_idx = X_train_raw['Gender'] == 1
        X_train_men_raw = X_train_raw[men_train_idx].drop('Gender', axis=1)
        y_train_men = y_train[men_train_idx]
        
        men_test_idx = X_test_raw['Gender'] == 1
        X_test_men_raw = X_test_raw[men_test_idx].drop('Gender', axis=1)
        y_test_men = y_test[men_test_idx]
        
        X_train_men, X_test_men, scaler_men = scale_features(X_train_men_raw, X_test_men_raw)
        self.scalers['men'] = scaler_men
        
        # Women's data
        women_train_idx = X_train_raw['Gender'] == 0
        X_train_women_raw = X_train_raw[women_train_idx].drop('Gender', axis=1)
        y_train_women = y_train[women_train_idx]
        
        women_test_idx = X_test_raw['Gender'] == 0
        X_test_women_raw = X_test_raw[women_test_idx].drop('Gender', axis=1)
        y_test_women = y_test[women_test_idx]
        
        X_train_women, X_test_women, scaler_women = scale_features(X_train_women_raw, X_test_women_raw)
        self.scalers['women'] = scaler_women
        
        logger.info(f"Train data: {len(X_train)} rows ({len(X_train_men)} men, {len(X_train_women)} women)")
        logger.info(f"Test data: {len(X_test)} rows ({len(X_test_men)} men, {len(X_test_women)} women)")
        
        # Store feature names
        self.feature_names = X_train_raw.columns
        self.feature_names_men = X_train_men_raw.columns
        self.feature_names_women = X_train_women_raw.columns
        
        # Store train/test data
        self.train_test_data = {
            'all': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X_train_raw': X_train_raw,
                'X_test_raw': X_test_raw
            },
            'men': {
                'X_train': X_train_men,
                'X_test': X_test_men,
                'y_train': y_train_men,
                'y_test': y_test_men,
                'X_train_raw': X_train_men_raw,
                'X_test_raw': X_test_men_raw
            },
            'women': {
                'X_train': X_train_women,
                'X_test': X_test_women,
                'y_train': y_train_women,
                'y_test': y_test_women,
                'X_train_raw': X_train_women_raw,
                'X_test_raw': X_test_women_raw
            }
        }
        
        return self.train_test_data
    
    def train_models(self, optimize=False):
        """
        Train models for prediction
        
        Parameters:
        -----------
        optimize : bool
            Whether to perform hyperparameter optimization
        """
        if not hasattr(self, 'train_test_data'):
            logger.error("No train/test data available! Run create_train_test_split first.")
            return
        
        logger.info("Training models...")
        
        # Get training data
        X_train = self.train_test_data['all']['X_train']
        y_train = self.train_test_data['all']['y_train']
        X_train_men = self.train_test_data['men']['X_train']
        y_train_men = self.train_test_data['men']['y_train']
        X_train_women = self.train_test_data['women']['X_train']
        y_train_women = self.train_test_data['women']['y_train']
        
        # Train base models
        logger.info("Training combined (all data) models...")
        self.models['all'] = train_base_models(X_train, y_train)
        
        logger.info("Training men's models...")
        self.models['men'] = train_base_models(X_train_men, y_train_men)
        
        logger.info("Training women's models...")
        self.models['women'] = train_base_models(X_train_women, y_train_women)
        
        # Hyperparameter optimization for best models
        if optimize:
            logger.info("Performing hyperparameter optimization...")
            
            # Optimize XGBoost for all data
            best_params_xgb = hyperparameter_optimization(X_train, y_train, 'xgb', n_trials=30)
            logger.info(f"Best XGBoost parameters: {best_params_xgb}")
            
            # Optimize LightGBM for all data
            best_params_lgb = hyperparameter_optimization(X_train, y_train, 'lgb', n_trials=30)
            logger.info(f"Best LightGBM parameters: {best_params_lgb}")
        
        # Create stacked models
        logger.info("Training stacked ensemble model for all data...")
        self.models['stacked_all'] = train_stacked_model(X_train, y_train, self.models['all'])
        
        logger.info("Training stacked ensemble model for men's data...")
        self.models['stacked_men'] = train_stacked_model(X_train_men, y_train_men, self.models['men'])
        
        logger.info("Training stacked ensemble model for women's data...")
        self.models['stacked_women'] = train_stacked_model(X_train_women, y_train_women, self.models['women'])
        
        # Create weighted ensemble of all models
        logger.info("Creating weighted ensemble of all models...")
        
        # Combine all base models (all, men's, women's)
        all_models = {}
        for category, models in self.models.items():
            if category.startswith('stacked'):
                # For stacked models, just add the meta learner
                all_models[f"stacked_{category}"] = models['meta_learner']
            else:
                # For base models, add each individual model
                for name, model in models.items():
                    all_models[f"{name}_{category}"] = model
        
        # Create weighted ensemble using all models
        self.ensemble = create_weighted_ensemble(all_models, X_train, y_train)
        
        # Save models
        self.save_models()
    
    def evaluate_models(self):
        """Evaluate models on test data"""
        if not hasattr(self, 'train_test_data') or not self.models:
            logger.error("No train/test data or models available!")
            return
        
        logger.info("Evaluating models...")
        
        # Get test data
        X_test = self.train_test_data['all']['X_test']
        y_test = self.train_test_data['all']['y_test']
        X_test_men = self.train_test_data['men']['X_test']
        y_test_men = self.train_test_data['men']['y_test']
        X_test_women = self.train_test_data['women']['X_test']
        y_test_women = self.train_test_data['women']['y_test']
        
        # Evaluate base models
        logger.info("Evaluating all data models...")
        all_results = evaluate_models(self.models['all'], X_test, y_test)
        logger.info("\n" + str(all_results))
        
        logger.info("Evaluating men's models...")
        men_results = evaluate_models(self.models['men'], X_test_men, y_test_men)
        logger.info("\n" + str(men_results))
        
        logger.info("Evaluating women's models...")
        women_results = evaluate_models(self.models['women'], X_test_women, y_test_women)
        logger.info("\n" + str(women_results))
        
        # Evaluate stacked models
        for category, test_data in [
            ('all', (X_test, y_test)),
            ('men', (X_test_men, y_test_men)),
            ('women', (X_test_women, y_test_women))
        ]:
            logger.info(f"Evaluating stacked model for {category}...")
            stacked_model = self.models[f'stacked_{category}']
            
            # Create a wrapper for the stacked model's predict_proba method
            class StackedModelWrapper:
                def __init__(self, model, base_models):
                    self.meta_learner = model['meta_learner']
                    self.base_models = model['base_models']
                
                def predict_proba(self, X):
                    # Generate predictions from each base model
                    meta_features = np.zeros((X.shape[0], len(self.base_models)))
                    for i, (name, model) in enumerate(self.base_models.items()):
                        meta_features[:, i] = model.predict_proba(X)[:, 1]
                    
                    # Predict with meta-learner
                    return self.meta_learner.predict_proba(meta_features)
            
            wrapper = StackedModelWrapper(stacked_model, stacked_model['base_models'])
            metrics = evaluate_model(wrapper, test_data[0], test_data[1], f"Stacked {category}")
        
        # Evaluate ensemble
        logger.info("Evaluating weighted ensemble...")
        
        # Create a wrapper for the ensemble's predict_proba method
        class EnsembleWrapper:
            def __init__(self, ensemble):
                self.predict_proba = ensemble['predict_proba']
        
        wrapper = EnsembleWrapper(self.ensemble)
        metrics = evaluate_model(wrapper, X_test, y_test, "Weighted Ensemble")
        
        # Generate visualizations
        self._generate_evaluation_visualizations()
        
        return metrics
    
    def _generate_evaluation_visualizations(self):
        """Generate evaluation visualizations"""
        logger.info("Generating evaluation visualizations...")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get test data
        X_test = self.train_test_data['all']['X_test']
        y_test = self.train_test_data['all']['y_test']
        
        # Get raw test data for feature importance
        X_test_raw = self.train_test_data['all']['X_test_raw']
        
        # Get best models from each category
        best_models = {}
        
        # Find best model from all data
        all_results = evaluate_models(self.models['all'], X_test, y_test)
        best_model_name = all_results.iloc[0]['Model']
        best_models['Best All'] = self.models['all'][best_model_name]
        
        # Add stacked model
        class StackedModelWrapper:
            def __init__(self, model):
                self.meta_learner = model['meta_learner']
                self.base_models = model['base_models']
                
                # For feature importance, copy coef_ from meta-learner
                if hasattr(self.meta_learner, 'coef_'):
                    self.coef_ = self.meta_learner.coef_
                
            def predict_proba(self, X):
                # Generate predictions from each base model
                meta_features = np.zeros((X.shape[0], len(self.base_models)))
                for i, (name, model) in enumerate(self.base_models.items()):
                    meta_features[:, i] = model.predict_proba(X)[:, 1]
                
                # Predict with meta-learner
                return self.meta_learner.predict_proba(meta_features)
        
        best_models['Stacked'] = StackedModelWrapper(self.models['stacked_all'])
        
        # Add ensemble
        class EnsembleWrapper:
            def __init__(self, ensemble):
                self.predict_proba = ensemble['predict_proba']
                
                # For feature importance, create a coef_ attribute based on model weights
                weights = ensemble['weights']
                self.coef_ = np.array([[weights[name] for name in weights.keys()]])
                
            def get_feature_importance(self):
                return {name: weight for name, weight in zip(self.models.keys(), self.coef_[0])}
        
        best_models['Ensemble'] = EnsembleWrapper(self.ensemble)
        
        # Generate calibration curves
        logger.info("Generating calibration curves...")
        plot_multiple_calibration_curves(best_models, X_test, y_test, output_dir=viz_dir)
        
        # Generate ROC curves
        logger.info("Generating ROC curves...")
        plot_multiple_roc_curves(best_models, X_test, y_test, output_dir=viz_dir)
        
        # Generate prediction distributions
        logger.info("Generating prediction distributions...")
        for name, model in best_models.items():
            analyze_prediction_distribution(model, X_test, y_test, model_name=name, output_dir=viz_dir)
        
        # Generate feature importance plots
        logger.info("Generating feature importance plots...")
        for name, model in best_models.items():
            if name != 'Ensemble':  # Skip ensemble for feature importance
                try:
                    analyze_feature_importance(model, self.feature_names, model_name=name, output_dir=viz_dir)
                except Exception as e:
                    logger.error(f"Error generating feature importance for {name}: {e}")
        
        # Generate seed performance analysis
        logger.info("Generating seed performance analysis...")
        try:
            # Get predictions from ensemble
            ensemble_preds = self.ensemble['predict_proba'](X_test)[:, 1]
            
            # Get seed information
            seed_data = pd.DataFrame({
                'Team1Seed': X_test_raw['Team1Seed'],
                'Team2Seed': X_test_raw['Team2Seed']
            })
            
            analyze_seed_performance(ensemble_preds, y_test, seed_data, output_dir=viz_dir)
        except Exception as e:
            logger.error(f"Error generating seed performance analysis: {e}")
    
    def generate_predictions(self, year=2025):
        """
        Generate predictions for tournament matchups
        
        Parameters:
        -----------
        year : int
            Year to generate predictions for
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing predictions
        """
        logger.info(f"Generating predictions for {year}...")
        
        # Check if model exists
        if not hasattr(self, 'ensemble'):
            logger.error("No ensemble model available!")
            return
        
        # Create submission template with all possible matchups
        submission_df = create_submission_template(self.data['Teams'], year)
        
        # Process each matchup
        predictions = []
        
        for _, row in submission_df.iterrows():
            id_str = row['ID']
            parts = id_str.split('_')
            
            if len(parts) == 3:
                match_year = int(parts[0])
                team1 = int(parts[1])
                team2 = int(parts[2])
                
                # Determine gender based on team IDs
                men_teams = self.data['Teams'][self.data['Teams']['Gender'] == 'M']['TeamID'].tolist()
                women_teams = self.data['Teams'][self.data['Teams']['Gender'] == 'W']['TeamID'].tolist()
                
                gender = 'M' if team1 in men_teams or team2 in men_teams else 'W'
                
                # Get team statistics
                seasons = sorted(self.team_stats.keys())
                latest_season = seasons[-1] if seasons else None
                
                if not latest_season:
                    # No team stats available
                    prob = 0.5
                else:
                    # Get team stats for the latest season
                    team_stats = self.team_stats.get(latest_season, {})
                    
                    # Get seeds if available
                    seed1 = self.seed_lookup.get((latest_season, gender, team1), 8.0)
                    seed2 = self.seed_lookup.get((latest_season, gender, team2), 8.0)
                    
                    # Create matchup features
                    matchup_features = create_matchup_features(team1, team2, team_stats)
                    
                    # Add seed information
                    matchup_features['Team1Seed'] = seed1
                    matchup_features['Team2Seed'] = seed2
                    matchup_features['SeedDiff'] = seed1 - seed2
                    
                    # Add gender indicator
                    matchup_features['Gender'] = 1 if gender == 'M' else 0
                    
                    # Convert to DataFrame
                    features_df = pd.DataFrame([matchup_features])
                    
                    # Add historical matchup features if available
                    if 'RegularSeasonResults' in self.data:
                        historical_features = create_historical_matchup_features(
                            team1, team2, self.data['RegularSeasonResults']
                        )
                        for k, v in historical_features.items():
                            features_df[k] = v
                    
                    # Add conference features if available
                    if 'Teams' in self.data and self.conf_strength:
                        conf_features = create_conference_features(
                            team1, team2, self.data['Teams'], self.conf_strength
                        )
                        for k, v in conf_features.items():
                            features_df[k] = v
                    
                    # Scale features
                    # Make sure columns match the training data
                    feature_cols = self.feature_names
                    for col in feature_cols:
                        if col not in features_df.columns:
                            features_df[col] = 0  # Add missing columns with default value
                    
                    # Select and order columns to match training data
                    features_df = features_df[feature_cols]
                    
                    # Scale the features
                    scaled_features = self.scalers['main'].transform(features_df)
                    
                    # Get prediction from ensemble
                    prob = self.ensemble['predict_proba'](scaled_features)[0, 1]
                
                # Add prediction to results
                predictions.append({
                    'ID': id_str,
                    'Pred': prob
                })
        
        # Create prediction DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Save predictions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"submission_{year}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        predictions_df.to_csv(filepath, index=False)
        
        logger.info(f"Generated {len(predictions_df)} predictions and saved to {filepath}")
        
        return predictions_df
    
    def save_models(self, filename=None):
        """
        Save models to file
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save models to (default: models_{timestamp}.pkl)
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"models_{timestamp}.pkl"
        
        filepath = os.path.join(self.models_dir, filename)
        
        # Save models, ensemble, and scalers
        model_data = {
            'models': self.models,
            'ensemble': self.ensemble,
            'scalers': self.scalers,
            'feature_names': self.feature_names if hasattr(self, 'feature_names') else None,
            'feature_names_men': self.feature_names_men if hasattr(self, 'feature_names_men') else None,
            'feature_names_women': self.feature_names_women if hasattr(self, 'feature_names_women') else None
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved models to {filepath}")
    
    def load_models(self, filename):
        """
        Load models from file
        
        Parameters:
        -----------
        filename : str
            Filename to load models from
        """
        filepath = os.path.join(self.models_dir, filename)
        
        # Load models
        model_data = joblib.load(filepath)
        
        # Set model attributes
        self.models = model_data['models']
        self.ensemble = model_data['ensemble']
        self.scalers = model_data['scalers']
        
        # Set feature names
        if 'feature_names' in model_data and model_data['feature_names'] is not None:
            self.feature_names = model_data['feature_names']
        
        if 'feature_names_men' in model_data and model_data['feature_names_men'] is not None:
            self.feature_names_men = model_data['feature_names_men']
        
        if 'feature_names_women' in model_data and model_data['feature_names_women'] is not None:
            self.feature_names_women = model_data['feature_names_women']
        
        logger.info(f"Loaded models from {filepath}")


def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description="March Madness Prediction Pipeline")
    
    parser.add_argument('--data-dir', type=str, default="data",
                        help="Directory containing data files")
    parser.add_argument('--output-dir', type=str, default="output",
                        help="Directory for output files")
    parser.add_argument('--models-dir', type=str, default="models",
                        help="Directory for model storage")
    parser.add_argument('--year', type=int, default=2025,
                        help="Year to generate predictions for")
    parser.add_argument('--test-years', type=int, nargs='+', default=[2023, 2024],
                        help="Years to use for testing")
    parser.add_argument('--optimize', action='store_true',
                        help="Perform hyperparameter optimization")
    parser.add_argument('--load-models', type=str,
                        help="Load models from file")
    parser.add_argument('--skip-training', action='store_true',
                        help="Skip model training")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = MarchMadnessPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir
    )
    
    # Load and preprocess data
    pipeline.load_and_preprocess_data()
    
    # Engineer features
    pipeline.engineer_features()
    
    # Create train/test split
    pipeline.create_train_test_split(args.test_years)
    
    # Either load existing models or train new ones
    if args.load_models:
        pipeline.load_models(args.load_models)
    elif not args.skip_training:
        # Train models
        pipeline.train_models(optimize=args.optimize)
    
    # Evaluate models
    pipeline.evaluate_models()
    
    # Generate predictions
    pipeline.generate_predictions(year=args.year)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
March Machine Learning Mania 2025 - NCAA Basketball Prediction
Main script for data processing, model training, and prediction generation
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# More comprehensive LightGBM warning suppression
os.environ['LIGHTGBM_SILENT'] = '1'  # Silence LightGBM output at environment level

# Import tqdm for progress bars
from tqdm import tqdm

import logging
# Set higher level for all loggers to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)

# Also suppress specific warning types
warnings.filterwarnings('ignore', message='No further splits with positive gain')

import joblib
from joblib import Parallel, delayed

class MarchMadnessPredictor:
    def __init__(self, data_dir="data", output_dir="submissions"):
        """
        Initialize the March Madness predictor
        
        Parameters:
        -----------
        data_dir : str
            Directory where the data files are stored
        output_dir : str
            Directory where submissions will be saved
        """
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, 'raw')
        self.output_dir = output_dir
        
        # Check if raw data directory exists
        if not os.path.exists(self.raw_data_dir):
            print(f"Warning: Raw data directory not found at {self.raw_data_dir}")
            print(f"Falling back to {self.data_dir}")
            self.raw_data_dir = self.data_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize model containers
        self.models = {}
        self.scalers = {}
        
        # Initialize data containers
        self.team_data = {}
        self.regular_season_data = None
        self.tourney_data = None
        self.seeds_data = None
        
    def load_data(self):
        """Load and preprocess all data files"""
        print("Loading data files...")
        print(f"Using raw data directory: {self.raw_data_dir}")
        
        # Load men's data
        men_regular_season = pd.read_csv(os.path.join(self.raw_data_dir, "MRegularSeasonCompactResults.csv"))
        men_tourney = pd.read_csv(os.path.join(self.raw_data_dir, "MNCAATourneyCompactResults.csv"))
        men_seeds = pd.read_csv(os.path.join(self.raw_data_dir, "MNCAATourneySeeds.csv"))
        men_team_data = pd.read_csv(os.path.join(self.raw_data_dir, "MTeams.csv"))
        
        # Load women's data
        women_regular_season = pd.read_csv(os.path.join(self.raw_data_dir, "WRegularSeasonCompactResults.csv"))
        women_tourney = pd.read_csv(os.path.join(self.raw_data_dir, "WNCAATourneyCompactResults.csv"))
        women_seeds = pd.read_csv(os.path.join(self.raw_data_dir, "WNCAATourneySeeds.csv"))
        women_team_data = pd.read_csv(os.path.join(self.raw_data_dir, "WTeams.csv"))
        
        # Add gender indicator
        men_regular_season['Gender'] = 'M'
        men_tourney['Gender'] = 'M'
        men_seeds['Gender'] = 'M'
        men_team_data['Gender'] = 'M'
        
        women_regular_season['Gender'] = 'W'
        women_tourney['Gender'] = 'W'
        women_seeds['Gender'] = 'W'
        women_team_data['Gender'] = 'W'
        
        # Combine men's and women's data
        self.regular_season_data = pd.concat([men_regular_season, women_regular_season])
        self.tourney_data = pd.concat([men_tourney, women_tourney])
        self.seeds_data = pd.concat([men_seeds, women_seeds])
        
        # Store team data separately by gender for feature engineering
        self.team_data['M'] = men_team_data
        self.team_data['W'] = women_team_data
        
        print(f"Loaded {len(self.regular_season_data)} regular season games")
        print(f"Loaded {len(self.tourney_data)} tournament games")
        
    def engineer_features(self):
        """Create features for the model"""
        print("Engineering features...")
        
        # Create team statistics by season
        self.team_stats = self._calculate_team_stats()
        
        # Create matchup training data
        self.train_data = self._create_matchup_features()
        
        print(f"Created training dataset with {len(self.train_data)} samples and {self.train_data.shape[1]} features")
        
    def _calculate_team_stats(self):
        """Calculate team statistics by season"""
        stats_by_gender = {}
        
        for gender in ['M', 'W']:
            gender_data = self.regular_season_data[self.regular_season_data['Gender'] == gender]
            
            # Group by season and team
            team_stats_by_season = {}
            
            for season in gender_data['Season'].unique():
                season_data = gender_data[gender_data['Season'] == season]
                
                # Initialize stats for every team
                teams = self.team_data[gender]['TeamID'].unique()
                team_stats = {team: {
                    'Wins': 0, 'Losses': 0, 'ScoredPoints': 0, 'AllowedPoints': 0,
                    'Games': 0, 'HomeWins': 0, 'AwayWins': 0, 'NeutralWins': 0,
                    'HomeGames': 0, 'AwayGames': 0, 'NeutralGames': 0,
                    'WinStreak': 0, 'PointDiff': []
                } for team in teams}
                
                # Process games chronologically
                for idx, game in season_data.sort_values('DayNum').iterrows():
                    wteam, lteam = game['WTeamID'], game['LTeamID']
                    wscore, lscore = game['WScore'], game['LScore']
                    
                    # Update winner stats
                    team_stats[wteam]['Wins'] += 1
                    team_stats[wteam]['ScoredPoints'] += wscore
                    team_stats[wteam]['AllowedPoints'] += lscore
                    team_stats[wteam]['Games'] += 1
                    team_stats[wteam]['WinStreak'] += 1
                    team_stats[wteam]['PointDiff'].append(wscore - lscore)
                    
                    # Update loser stats
                    team_stats[lteam]['Losses'] += 1
                    team_stats[lteam]['ScoredPoints'] += lscore
                    team_stats[lteam]['AllowedPoints'] += wscore
                    team_stats[lteam]['Games'] += 1
                    team_stats[lteam]['WinStreak'] = 0
                    team_stats[lteam]['PointDiff'].append(lscore - wscore)
                    
                    # Update location stats
                    if game['WLoc'] == 'H':
                        team_stats[wteam]['HomeWins'] += 1
                        team_stats[wteam]['HomeGames'] += 1
                        team_stats[lteam]['AwayGames'] += 1
                    elif game['WLoc'] == 'A':
                        team_stats[wteam]['AwayWins'] += 1
                        team_stats[wteam]['AwayGames'] += 1
                        team_stats[lteam]['HomeGames'] += 1
                    else:  # Neutral
                        team_stats[wteam]['NeutralWins'] += 1
                        team_stats[wteam]['NeutralGames'] += 1
                        team_stats[lteam]['NeutralGames'] += 1
                
                # Calculate derived metrics
                for team in teams:
                    stats = team_stats[team]
                    if stats['Games'] > 0:
                        stats['WinPct'] = stats['Wins'] / stats['Games']
                        stats['PointsPerGame'] = stats['ScoredPoints'] / stats['Games']
                        stats['OppPointsPerGame'] = stats['AllowedPoints'] / stats['Games']
                        stats['PointDiffPerGame'] = (stats['ScoredPoints'] - stats['AllowedPoints']) / stats['Games']
                        stats['HomeWinPct'] = stats['HomeWins'] / max(1, stats['HomeGames'])
                        stats['AwayWinPct'] = stats['AwayWins'] / max(1, stats['AwayGames'])
                        stats['NeutralWinPct'] = stats['NeutralWins'] / max(1, stats['NeutralGames'])
                        
                        # Calculate point differential volatility (standard deviation)
                        if len(stats['PointDiff']) > 1:
                            stats['PointDiffStd'] = np.std(stats['PointDiff'])
                        else:
                            stats['PointDiffStd'] = 0
                    else:
                        # Default values for teams with no games
                        stats['WinPct'] = 0
                        stats['PointsPerGame'] = 0
                        stats['OppPointsPerGame'] = 0
                        stats['PointDiffPerGame'] = 0
                        stats['HomeWinPct'] = 0
                        stats['AwayWinPct'] = 0
                        stats['NeutralWinPct'] = 0
                        stats['PointDiffStd'] = 0
                
                # Store stats for this season
                team_stats_by_season[season] = team_stats
                
            stats_by_gender[gender] = team_stats_by_season
            
        return stats_by_gender
    
    def _create_matchup_features(self):
        """Create features for each historical matchup"""
        features = []
        
        # Process tournament games for training data
        for idx, game in self.tourney_data.iterrows():
            season = game['Season']
            gender = game['Gender']
            
            # Get team stats for this season
            team_stats = self.team_stats[gender].get(season, {})
            if not team_stats:
                continue
                
            team1, team2 = game['WTeamID'], game['LTeamID']
            # Sort teams by ID to be consistent with the competition format
            if team1 > team2:
                team1, team2 = team2, team1
                result = 0  # Team 1 (lower ID) lost
            else:
                result = 1  # Team 1 (lower ID) won
                
            # Get team seeds if available
            season_seeds = self.seeds_data[(self.seeds_data['Season'] == season) & 
                                           (self.seeds_data['Gender'] == gender)]
            team1_seed = season_seeds[season_seeds['TeamID'] == team1]['Seed'].values
            team2_seed = season_seeds[season_seeds['TeamID'] == team2]['Seed'].values
            
            team1_seed_num = self._seed_to_number(team1_seed[0]) if len(team1_seed) > 0 else 16
            team2_seed_num = self._seed_to_number(team2_seed[0]) if len(team2_seed) > 0 else 16
            
            # Extract team stats
            team1_stats = team_stats.get(team1, {})
            team2_stats = team_stats.get(team2, {})
            
            if not team1_stats or not team2_stats:
                continue
                
            # Create feature dictionary
            feature_dict = {
                'Season': season,
                'Team1': team1,
                'Team2': team2,
                'SeedDiff': team1_seed_num - team2_seed_num,
                'WinPctDiff': team1_stats.get('WinPct', 0) - team2_stats.get('WinPct', 0),
                'ScoringDiff': team1_stats.get('PointsPerGame', 0) - team2_stats.get('PointsPerGame', 0),
                'DefenseDiff': team1_stats.get('OppPointsPerGame', 0) - team2_stats.get('OppPointsPerGame', 0),
                'Team1WinPct': team1_stats.get('WinPct', 0),
                'Team2WinPct': team2_stats.get('WinPct', 0),
                'Team1PPG': team1_stats.get('PointsPerGame', 0),
                'Team2PPG': team2_stats.get('PointsPerGame', 0),
                'Team1OppPPG': team1_stats.get('OppPointsPerGame', 0),
                'Team2OppPPG': team2_stats.get('OppPointsPerGame', 0),
                'Team1PointDiff': team1_stats.get('PointDiffPerGame', 0),
                'Team2PointDiff': team2_stats.get('PointDiffPerGame', 0),
                'Team1NeutralWinPct': team1_stats.get('NeutralWinPct', 0),
                'Team2NeutralWinPct': team2_stats.get('NeutralWinPct', 0),
                'Team1PointDiffStd': team1_stats.get('PointDiffStd', 0),
                'Team2PointDiffStd': team2_stats.get('PointDiffStd', 0),
                'Gender': 1 if gender == 'M' else 0,
                'Result': result
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _seed_to_number(self, seed_str):
        """Convert seed string (like '1a', '16b') to a numeric value"""
        if not isinstance(seed_str, str):
            return 16  # Default high seed if missing
            
        # Remove letters, handle play-in teams
        seed_num = int(''.join(c for c in seed_str if c.isdigit()))
        return seed_num
        
    def train_models(self):
        """Train multiple models and create an ensemble using parallel processing"""
        print("Training models...")
        
        # Separate features and target
        X = self.train_data.drop(['Result', 'Team1', 'Team2', 'Season'], axis=1, errors='ignore')
        y = self.train_data['Result']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        # Split data by gender for specialized models
        X_men = X[X['Gender'] == 1].drop('Gender', axis=1)
        y_men = y[X['Gender'] == 1]
        X_women = X[X['Gender'] == 0].drop('Gender', axis=1)
        y_women = y[X['Gender'] == 0]
        
        # Scale gender-specific features
        scaler_men = StandardScaler()
        X_men_scaled = scaler_men.fit_transform(X_men)
        self.scalers['men'] = scaler_men
        
        scaler_women = StandardScaler()
        X_women_scaled = scaler_women.fit_transform(X_women)
        self.scalers['women'] = scaler_women
        
        # Define base models with parallel processing where supported
        base_models = {
            'lr': LogisticRegression(C=0.1, solver='liblinear', max_iter=1000, n_jobs=-1),
            'rf': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
        }
        
        # Define tree boosting models with verbosity disabled and parallel processing
        boosting_models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=300, 
                max_depth=5, 
                learning_rate=0.05, 
                random_state=42, 
                use_label_encoder=False, 
                eval_metric='logloss',
                verbosity=0,
                n_jobs=-1),  # Enable parallel processing
            'lgb': lgb.LGBMClassifier(
                n_estimators=300, 
                max_depth=5, 
                learning_rate=0.05, 
                random_state=42,
                verbose=-1,
                force_row_wise=True,
                n_jobs=-1)  # Enable parallel processing
        }
        
        # Train models that need calibration
        for name, model in base_models.items():
            # Combined model (all data)
            print(f"Training {name} on all data...")
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
            calibrated_model.fit(X_scaled, y)
            self.models[f'{name}_all'] = calibrated_model
            
            # Men's model
            print(f"Training {name} on men's data...")
            men_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
            men_model.fit(X_men_scaled, y_men)
            self.models[f'{name}_men'] = men_model
            
            # Women's model
            print(f"Training {name} on women's data...")
            women_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
            women_model.fit(X_women_scaled, y_women)
            self.models[f'{name}_women'] = women_model
        
        # Train tree boosting models (which already output calibrated probabilities)
        for name, model in boosting_models.items():
            # Combined model (all data)
            print(f"Training {name} on all data...")
            model.fit(X_scaled, y)
            self.models[f'{name}_all'] = model
            
            # Men's model
            print(f"Training {name} on men's data...")
            men_model = model.__class__(**model.get_params())
            men_model.fit(X_men_scaled, y_men)
            self.models[f'{name}_men'] = men_model
            
            # Women's model
            print(f"Training {name} on women's data...")
            women_model = model.__class__(**model.get_params())
            women_model.fit(X_women_scaled, y_women)
            self.models[f'{name}_women'] = women_model
        
        print(f"Trained {len(self.models)} models")
        
    def generate_predictions(self, submission_prefix, year=2025, n_jobs=4):
        """
        Generate predictions for all possible matchups using high-performance techniques
        and create separate files for men's and women's basketball
        """
        print(f"Generating predictions for {year}...")
        
        # Get all possible teams
        men_teams = self.team_data['M']['TeamID'].tolist()
        women_teams = self.team_data['W']['TeamID'].tolist()
        
        # --------- MAJOR PERFORMANCE OPTIMIZATION ----------
        # Pre-compute all team features up front to avoid repeated calculations
        print("Pre-computing team features...")
        men_team_features = self._precompute_team_features('M', year)
        women_team_features = self._precompute_team_features('W', year)
        
        # Process men's predictions
        print("Generating men's predictions...")
        men_df = self._generate_gender_predictions(men_teams, men_team_features, 'M', year, n_jobs)
        
        # Save men's predictions
        men_submission_file = f"{submission_prefix}_M.csv"
        men_df.to_csv(os.path.join(self.output_dir, men_submission_file), index=False)
        print(f"Generated {len(men_df)} men's predictions")
        print(f"Saved to {os.path.join(self.output_dir, men_submission_file)}")
        
        # Process women's predictions
        print("Generating women's predictions...")
        women_df = self._generate_gender_predictions(women_teams, women_team_features, 'W', year, n_jobs)
        
        # Save women's predictions
        women_submission_file = f"{submission_prefix}_W.csv"
        women_df.to_csv(os.path.join(self.output_dir, women_submission_file), index=False)
        print(f"Generated {len(women_df)} women's predictions")
        print(f"Saved to {os.path.join(self.output_dir, women_submission_file)}")

    def _precompute_team_features(self, gender, year):
        """Pre-compute features for all teams to avoid redundant calculations"""
        # Get the latest season for this gender
        latest_season = max(self.team_stats[gender].keys())
        team_stats = self.team_stats[gender][latest_season]
        
        # Pre-compute all team features
        team_features = {}
        
        # Get list of all teams
        all_teams = self.team_data[gender]['TeamID'].unique()
        
        # Calculate average stats for teams with missing data
        all_teams_list = list(team_stats.values())
        if all_teams_list:
            avg_winpct = sum(t.get('WinPct', 0) for t in all_teams_list) / len(all_teams_list)
            avg_ppg = sum(t.get('PointsPerGame', 0) for t in all_teams_list) / len(all_teams_list)
            avg_oppppg = sum(t.get('OppPointsPerGame', 0) for t in all_teams_list) / len(all_teams_list)
            avg_neutwinpct = sum(t.get('NeutralWinPct', 0) for t in all_teams_list) / len(all_teams_list)
            avg_pdiff = sum(t.get('PointDiffPerGame', 0) for t in all_teams_list) / len(all_teams_list)
            avg_pdiffstd = sum(t.get('PointDiffStd', 0) for t in all_teams_list) / len(all_teams_list)
        else:
            # Default values if no teams
            avg_winpct = 0.5
            avg_ppg = 70.0
            avg_oppppg = 70.0
            avg_neutwinpct = 0.5
            avg_pdiff = 0.0
            avg_pdiffstd = 10.0
        
        # Pre-compute features for each team
        for team in all_teams:
            team_stat = team_stats.get(team, {})
            
            if not team_stat:
                # Use average values for missing teams
                team_features[team] = {
                    'WinPct': avg_winpct,
                    'PointsPerGame': avg_ppg,
                    'OppPointsPerGame': avg_oppppg,
                    'NeutralWinPct': avg_neutwinpct,
                    'PointDiffPerGame': avg_pdiff,
                    'PointDiffStd': avg_pdiffstd,
                    'Seed': 8  # Default middle seed
                }
            else:
                # Use actual team stats
                team_features[team] = {
                    'WinPct': team_stat.get('WinPct', avg_winpct),
                    'PointsPerGame': team_stat.get('PointsPerGame', avg_ppg),
                    'OppPointsPerGame': team_stat.get('OppPointsPerGame', avg_oppppg),
                    'NeutralWinPct': team_stat.get('NeutralWinPct', avg_neutwinpct),
                    'PointDiffPerGame': team_stat.get('PointDiffPerGame', avg_pdiff),
                    'PointDiffStd': team_stat.get('PointDiffStd', avg_pdiffstd),
                    'Seed': 8  # Default middle seed
                }
        
        return team_features

    def _generate_gender_predictions(self, teams, team_features, gender, year, n_jobs):
        """Generate predictions for a specific gender using vectorized operations with improved probability scaling"""
        # Create all possible matchups
        matchups = []
        for i, team1 in enumerate(teams):
            for team2 in teams[i+1:]:
                if team1 > team2:
                    team1, team2 = team2, team1
                matchups.append((team1, team2))
        
        # Convert gender to numeric
        gender_numeric = 1 if gender == 'M' else 0
        
        # Create large feature matrix for all matchups at once
        print(f"Preparing feature matrix for {len(matchups)} matchups...")
        features = []
        matchup_ids = []
        
        # Batch process creation of feature matrix to avoid memory issues
        batch_size = 10000
        for i in range(0, len(matchups), batch_size):
            batch = matchups[i:i+batch_size]
            
            # Create batch IDs
            batch_ids = [f"{year}_{team1}_{team2}" for team1, team2 in batch]
            matchup_ids.extend(batch_ids)
            
            # Create feature batch
            batch_features = []
            for team1, team2 in batch:
                team1_stats = team_features[team1]
                team2_stats = team_features[team2]
                
                feature_dict = {
                    'SeedDiff': team1_stats['Seed'] - team2_stats['Seed'],
                    'WinPctDiff': team1_stats['WinPct'] - team2_stats['WinPct'],
                    'ScoringDiff': team1_stats['PointsPerGame'] - team2_stats['PointsPerGame'],
                    'DefenseDiff': team1_stats['OppPointsPerGame'] - team2_stats['OppPointsPerGame'],
                    'Team1WinPct': team1_stats['WinPct'],
                    'Team2WinPct': team2_stats['WinPct'],
                    'Team1PPG': team1_stats['PointsPerGame'],
                    'Team2PPG': team2_stats['PointsPerGame'],
                    'Team1OppPPG': team1_stats['OppPointsPerGame'],
                    'Team2OppPPG': team2_stats['OppPointsPerGame'],
                    'Team1PointDiff': team1_stats['PointDiffPerGame'],
                    'Team2PointDiff': team2_stats['PointDiffPerGame'],
                    'Team1NeutralWinPct': team1_stats['NeutralWinPct'],
                    'Team2NeutralWinPct': team2_stats['NeutralWinPct'],
                    'Team1PointDiffStd': team1_stats['PointDiffStd'],
                    'Team2PointDiffStd': team2_stats['PointDiffStd'],
                    'Gender': gender_numeric
                }
                batch_features.append(feature_dict)
            
            features.extend(batch_features)
        
        # Convert to DataFrame
        X = pd.DataFrame(features)
        
        # Ensure ordered columns for all models
        all_feature_names = [
            'SeedDiff', 'WinPctDiff', 'ScoringDiff', 'DefenseDiff',
            'Team1WinPct', 'Team2WinPct', 'Team1PPG', 'Team2PPG',
            'Team1OppPPG', 'Team2OppPPG', 'Team1PointDiff', 'Team2PointDiff',
            'Team1NeutralWinPct', 'Team2NeutralWinPct',
            'Team1PointDiffStd', 'Team2PointDiffStd', 'Gender'
        ]
        X = X[all_feature_names]
        
        # Scale features
        X_all = self.scalers['main'].transform(X)
        
        # Create gender-specific features
        X_gender = X.drop('Gender', axis=1)
        gender_feature_names = [f for f in all_feature_names if f != 'Gender']
        X_gender = X_gender[gender_feature_names]
        gender_key = 'men' if gender == 'M' else 'women'
        X_gender_scaled = self.scalers[gender_key].transform(X_gender)
        
        # Function to predict a batch of matchups with one model
        def predict_batch_with_model(model_name, model, X_batch, is_lgb=False, is_gender_specific=False):
            if is_gender_specific:
                # Use gender-specific data
                if is_lgb:
                    # For LightGBM, use DataFrame with column names
                    X_input = pd.DataFrame(X_batch, columns=gender_feature_names)
                else:
                    X_input = X_batch
            else:
                # Use all data including gender
                if is_lgb:
                    # For LightGBM, use DataFrame with column names
                    X_input = pd.DataFrame(X_batch, columns=all_feature_names)
                else:
                    X_input = X_batch
                
            try:
                # Get probabilities - much faster with batch prediction
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(X_input)[:, 1]
                elif hasattr(model, 'predict'):
                    if isinstance(model, xgb.XGBModel):
                        return model.predict(X_input, output_margin=False)
                    else:
                        return model.predict(X_input)
                else:
                    return np.full(len(X_batch), 0.5)
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                return np.full(len(X_batch), 0.5)
        
        # Process predictions in large batches for better efficiency
        print("Generating predictions with all models...")
        large_batch_size = min(50000, len(X))
        all_probs = np.zeros((len(X), len(self.models)))
        
        # Track progress with tqdm
        with tqdm(total=len(X)) as pbar:
            for batch_start in range(0, len(X), large_batch_size):
                batch_end = min(batch_start + large_batch_size, len(X))
                batch_size = batch_end - batch_start
                
                X_batch = X_all[batch_start:batch_end]
                X_gender_batch = X_gender_scaled[batch_start:batch_end]
                
                # Process each model - in parallel if possible
                model_results = []
                for i, (name, model) in enumerate(self.models.items()):
                    is_lgb = 'lgb' in name
                    is_gender_specific = name.endswith(f'_{gender_key}')
                    
                    if is_gender_specific:
                        # Use gender-specific preprocessed data
                        model_results.append((i, predict_batch_with_model(name, model, X_gender_batch, is_lgb, True)))
                    elif name.endswith('_all'):
                        # Use all data
                        model_results.append((i, predict_batch_with_model(name, model, X_batch, is_lgb, False)))
                    # Skip models for the other gender
                
                # Store results
                for i, probs in model_results:
                    all_probs[batch_start:batch_end, i] = probs
                
                pbar.update(batch_size)
        
        # Compute ensemble (average) prediction
        print("Computing ensemble predictions...")
        # Calculate ensemble by averaging non-zero values for each row
        ensemble_probs = np.zeros(len(X))
        for i in range(len(X)):
            # Count only models that actually made predictions (non-zero)
            valid_preds = all_probs[i, all_probs[i, :] > 0]
            if len(valid_preds) > 0:
                ensemble_probs[i] = np.mean(valid_preds)
            else:
                ensemble_probs[i] = 0.5  # Default if no valid predictions
        
        # After computing the ensemble predictions, apply a calibration to spread out the probabilities
        print("Calibrating ensemble predictions...")
        
        # Option 1: Apply Platt scaling to spread probabilities
        def calibrate_probabilities(probs, temperature=1.5):
            """
            Apply a temperature-based calibration to spread out probabilities
            - temperature > 1 increases spread (more confident predictions)
            - temperature < 1 decreases spread (more conservative predictions)
            """
            # Convert to logits (log-odds)
            logits = np.log(probs / (1 - probs))
            # Apply temperature scaling
            scaled_logits = logits * temperature
            # Convert back to probabilities
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            return scaled_probs
        
        # Apply calibration
        calibrated_probs = calibrate_probabilities(ensemble_probs, temperature=1.5)
        
        # Create submission DataFrame with calibrated probabilities
        submission = pd.DataFrame({
            'ID': matchup_ids,
            'Pred': calibrated_probs
        })
        
        return submission
        
    def save_models(self, filename="models.pkl"):
        """Save trained models to file"""
        with open(os.path.join(self.output_dir, filename), 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers
            }, f)
        
    def load_models(self, filename="models.pkl"):
        """Load trained models from file"""
        with open(os.path.join(self.output_dir, filename), 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.scalers = data['scalers']


if __name__ == "__main__":
    import multiprocessing
    
    # Determine optimal number of jobs based on CPU count
    n_cpu = multiprocessing.cpu_count()
    optimal_jobs = max(1, n_cpu - 1)  # Leave one CPU free for system
    
    print(f"Detected {n_cpu} CPUs, using {optimal_jobs} for parallel processing")
    
    predictor = MarchMadnessPredictor()
    
    # Load and process data
    predictor.load_data()
    predictor.engineer_features()
    
    # Train models
    predictor.train_models()
    
    # Save models for future use
    predictor.save_models()
    
    # Generate predictions with optimized processing
    predictor.generate_predictions("submission_2025", n_jobs=optimal_jobs) 
"""
Data Preparation Module for March Madness Prediction
Handles loading, cleaning, and transforming NCAA basketball data
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
import re


def load_data(data_dir, gender=None):
    """
    Load all relevant data files for March Madness prediction
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
    gender : str, optional
        Filter by gender ('M' for men, 'W' for women, None for both)
    
    Returns:
    --------
    dict
        Dictionary containing DataFrames of loaded data
    """
    data = {}
    
    # Ensure we're using the raw data folder
    raw_data_dir = os.path.join(data_dir, 'raw')
    if not os.path.exists(raw_data_dir):
        print(f"Warning: Raw data directory not found at {raw_data_dir}")
        print(f"Falling back to {data_dir}")
        raw_data_dir = data_dir
    else:
        print(f"Loading data from raw directory: {raw_data_dir}")
    
    # Define file patterns to look for
    file_patterns = [
        "RegularSeasonCompactResults",
        "NCAATourneyCompactResults",
        "NCAATourneySeeds",
        "Teams",
        "TeamCoaches",
        "Conferences",
        "RegularSeasonDetailedResults",
        "NCAATourneyDetailedResults",
        "MasseyOrdinals"
    ]
    
    # Apply gender filter
    if gender:
        gender_prefix = gender
        file_patterns = [f"{gender_prefix}{pattern}" for pattern in file_patterns]
    
    # Load each data file if it exists
    for pattern in file_patterns:
        # Handle both gender-specific and non-gender files
        files = glob.glob(os.path.join(raw_data_dir, f"*{pattern}*.csv"))
        
        for file_path in files:
            file_name = os.path.basename(file_path)
            key = file_name.replace('.csv', '')
            
            try:
                df = pd.read_csv(file_path)
                # Add gender indicator if not already present
                if 'Gender' not in df.columns:
                    # Try to determine gender from filename
                    if file_name.startswith('M'):
                        df['Gender'] = 'M'
                    elif file_name.startswith('W'):
                        df['Gender'] = 'W'
                    else:
                        # If can't determine, don't add
                        pass
                        
                data[key] = df
                print(f"Loaded {key}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return data


def create_processed_directory(data_dir):
    """
    Create a 'processed' subdirectory within the data directory
    
    Parameters:
    -----------
    data_dir : str
        Base data directory
    
    Returns:
    --------
    str
        Path to the processed data directory
    """
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Processed data directory: {processed_dir}")
    return processed_dir


def save_processed_data(data_dict, processed_dir):
    """
    Save processed DataFrames to CSV files in the processed directory
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing processed DataFrames
    processed_dir : str
        Directory to save processed data files
    """
    for name, df in data_dict.items():
        file_path = os.path.join(processed_dir, f"{name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {name} with {len(df)} rows to {file_path}")


def clean_and_combine_data(data, data_dir):
    """
    Clean and combine data from multiple sources
    
    Parameters:
    -----------
    data : dict
        Dictionary containing DataFrames of loaded data
    data_dir : str
        Base data directory for saving processed results
    
    Returns:
    --------
    dict
        Dictionary containing cleaned and combined DataFrames
    """
    cleaned_data = {}
    
    # Combine Men's and Women's regular season results
    reg_season_keys = [k for k in data.keys() if 'RegularSeasonCompactResults' in k]
    if reg_season_keys:
        reg_season_dfs = [data[k] for k in reg_season_keys]
        cleaned_data['RegularSeasonResults'] = pd.concat(reg_season_dfs, ignore_index=True)
        print(f"Combined {len(reg_season_keys)} regular season results datasets: {len(cleaned_data['RegularSeasonResults'])} rows")
    
    # Combine Men's and Women's tournament results
    tourney_keys = [k for k in data.keys() if 'NCAATourneyCompactResults' in k]
    if tourney_keys:
        tourney_dfs = [data[k] for k in tourney_keys]
        cleaned_data['TourneyResults'] = pd.concat(tourney_dfs, ignore_index=True)
        print(f"Combined {len(tourney_keys)} tournament results datasets: {len(cleaned_data['TourneyResults'])} rows")
    
    # Combine Men's and Women's seeds
    seed_keys = [k for k in data.keys() if 'NCAATourneySeeds' in k]
    if seed_keys:
        seed_dfs = [data[k] for k in seed_keys]
        cleaned_data['TourneySeeds'] = pd.concat(seed_dfs, ignore_index=True)
        print(f"Combined {len(seed_keys)} tournament seeds datasets: {len(cleaned_data['TourneySeeds'])} rows")
    
    # Combine Men's and Women's teams
    team_keys = [k for k in data.keys() if 'Teams' in k and len(k) < 10]  # Avoid TeamCoaches
    if team_keys:
        team_dfs = [data[k] for k in team_keys]
        cleaned_data['Teams'] = pd.concat(team_dfs, ignore_index=True)
        print(f"Combined {len(team_keys)} teams datasets: {len(cleaned_data['Teams'])} rows")
    
    # Combine detailed results if available
    detailed_reg_keys = [k for k in data.keys() if 'RegularSeasonDetailedResults' in k]
    if detailed_reg_keys:
        detailed_reg_dfs = [data[k] for k in detailed_reg_keys]
        cleaned_data['RegularSeasonDetailedResults'] = pd.concat(detailed_reg_dfs, ignore_index=True)
        print(f"Combined {len(detailed_reg_keys)} detailed regular season datasets: {len(cleaned_data['RegularSeasonDetailedResults'])} rows")
    
    detailed_tourney_keys = [k for k in data.keys() if 'NCAATourneyDetailedResults' in k]
    if detailed_tourney_keys:
        detailed_tourney_dfs = [data[k] for k in detailed_tourney_keys]
        cleaned_data['TourneyDetailedResults'] = pd.concat(detailed_tourney_dfs, ignore_index=True)
        print(f"Combined {len(detailed_tourney_keys)} detailed tournament datasets: {len(cleaned_data['TourneyDetailedResults'])} rows")
    
    # Add any remaining datasets that weren't combined
    for key, df in data.items():
        if not any(k in key for k in [
            'RegularSeasonCompactResults', 'NCAATourneyCompactResults',
            'NCAATourneySeeds', 'Teams', 'RegularSeasonDetailedResults',
            'NCAATourneyDetailedResults'
        ]):
            cleaned_data[key] = df
    
    # Create processed directory and save cleaned data
    processed_dir = create_processed_directory(data_dir)
    save_processed_data(cleaned_data, processed_dir)
    
    return cleaned_data


def preprocess_seeds(seeds_df):
    """
    Preprocess tournament seeds for modeling
    
    Parameters:
    -----------
    seeds_df : pandas.DataFrame
        DataFrame containing tournament seed information
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with processed seed information
    """
    # Create a copy to avoid modifying original
    seeds = seeds_df.copy()
    
    # Extract numeric seed from seed string
    def extract_seed_number(seed_str):
        if not isinstance(seed_str, str):
            return 16.0  # Default high seed if missing
        
        # Extract number and handle play-in teams
        match = re.search(r'(\d+)([a-z])?', seed_str.lower())
        if match:
            seed_num = int(match.group(1))
            seed_letter = match.group(2)
            
            # Adjust for play-in teams
            if seed_letter:
                if seed_letter == 'a':
                    seed_num -= 0.25
                elif seed_letter == 'b':
                    seed_num -= 0.5
            
            return seed_num
        else:
            return 16.0
    
    # Apply extraction
    seeds['SeedNumber'] = seeds['Seed'].apply(extract_seed_number)
    
    return seeds


def create_seed_lookup(seeds_df):
    """
    Create a lookup dictionary for team seeds by season
    
    Parameters:
    -----------
    seeds_df : pandas.DataFrame
        DataFrame containing tournament seed information
    
    Returns:
    --------
    dict
        Dictionary mapping (Season, Gender, TeamID) to seed number
    """
    seed_lookup = {}
    
    for _, row in seeds_df.iterrows():
        season = row['Season']
        gender = row.get('Gender', 'M')  # Default to 'M' if gender not provided
        team_id = row['TeamID']
        seed_num = row['SeedNumber'] if 'SeedNumber' in row else extract_seed_number(row['Seed'])
        
        seed_lookup[(season, gender, team_id)] = seed_num
    
    return seed_lookup


def extract_seed_number(seed_str):
    """
    Extract numeric seed from seed string
    
    Parameters:
    -----------
    seed_str : str
        Seed string (e.g., '1a', '16b')
    
    Returns:
    --------
    float
        Numeric seed value
    """
    if not isinstance(seed_str, str):
        return 16.0  # Default high seed if missing
    
    # Extract number and handle play-in teams
    match = re.search(r'(\d+)([a-z])?', seed_str.lower())
    if match:
        seed_num = int(match.group(1))
        seed_letter = match.group(2)
        
        # Adjust for play-in teams
        if seed_letter:
            if seed_letter == 'a':
                seed_num -= 0.25
            elif seed_letter == 'b':
                seed_num -= 0.5
        
        return float(seed_num)
    else:
        return 16.0


def create_matchup_data(results_df, seeds_lookup=None):
    """
    Create training data for matchup prediction
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing game results
    seeds_lookup : dict, optional
        Dictionary mapping (Season, Gender, TeamID) to seed number
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with matchup features and results
    """
    matchups = []
    
    for _, game in results_df.iterrows():
        season = game['Season']
        gender = game.get('Gender', 'M')  # Default to 'M' if gender not provided
        
        team1 = game['WTeamID']
        team2 = game['LTeamID']
        
        # Ensure team1 has lower ID than team2 for consistency
        if team1 > team2:
            team1, team2 = team2, team1
            result = 0  # team1 (lower ID) lost
        else:
            result = 1  # team1 (lower ID) won
        
        # Get seeds if available
        seed1 = seeds_lookup.get((season, gender, team1), 8.0) if seeds_lookup else 8.0
        seed2 = seeds_lookup.get((season, gender, team2), 8.0) if seeds_lookup else 8.0
        
        # Create matchup dictionary
        matchup = {
            'Season': season,
            'Gender': 1 if gender == 'M' else 0,  # Binary encoding
            'Team1': team1,
            'Team2': team2,
            'Team1Seed': seed1,
            'Team2Seed': seed2,
            'SeedDiff': seed1 - seed2,
            'Result': result
        }
        
        matchups.append(matchup)
    
    return pd.DataFrame(matchups)


def scale_features(X_train, X_test=None):
    """
    Scale features using StandardScaler
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    X_test : pandas.DataFrame, optional
        Test feature matrix
    
    Returns:
    --------
    tuple
        Tuple containing scaled training features, test features (if provided), and the scaler
    """
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit and transform training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        # Transform test data
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    else:
        return X_train_scaled, scaler


def create_submission_template(teams_data, year=2025):
    """
    Create a template for submission with all possible matchups
    
    Parameters:
    -----------
    teams_data : pandas.DataFrame
        DataFrame containing team information
    year : int
        Year for the submission
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all possible matchups
    """
    submissions = []
    
    # Get men's and women's teams
    men_teams = teams_data[teams_data['Gender'] == 'M']['TeamID'].tolist()
    women_teams = teams_data[teams_data['Gender'] == 'W']['TeamID'].tolist()
    
    # Generate all men's matchups
    for i, team1 in enumerate(men_teams):
        for team2 in men_teams[i+1:]:
            # Ensure team1 has lower ID
            if team1 > team2:
                team1, team2 = team2, team1
            
            # Create matchup ID
            matchup_id = f"{year}_{team1}_{team2}"
            
            submissions.append({
                'ID': matchup_id,
                'Pred': 0.5  # Default prediction
            })
    
    # Generate all women's matchups
    for i, team1 in enumerate(women_teams):
        for team2 in women_teams[i+1:]:
            # Ensure team1 has lower ID
            if team1 > team2:
                team1, team2 = team2, team1
            
            # Create matchup ID
            matchup_id = f"{year}_{team1}_{team2}"
            
            submissions.append({
                'ID': matchup_id,
                'Pred': 0.5  # Default prediction
            })
    
    return pd.DataFrame(submissions)


def extract_teams_from_submission(submission_df):
    """
    Extract team IDs from submission matchups
    
    Parameters:
    -----------
    submission_df : pandas.DataFrame
        DataFrame containing submission IDs
    
    Returns:
    --------
    dict
        Dictionary with team IDs
    """
    teams = set()
    
    for id_str in submission_df['ID']:
        parts = id_str.split('_')
        if len(parts) == 3:
            teams.add(int(parts[1]))
            teams.add(int(parts[2]))
    
    return sorted(list(teams))


def parse_submission_id(id_str):
    """
    Parse submission ID into components
    
    Parameters:
    -----------
    id_str : str
        Submission ID string (e.g., '2025_1101_1102')
    
    Returns:
    --------
    tuple
        Tuple containing (year, team1_id, team2_id)
    """
    parts = id_str.split('_')
    if len(parts) == 3:
        year = int(parts[0])
        team1 = int(parts[1])
        team2 = int(parts[2])
        return year, team1, team2
    else:
        raise ValueError(f"Invalid submission ID format: {id_str}")


def save_submission(predictions, filename, output_dir='.'):
    """
    Save predictions to a submission file
    
    Parameters:
    -----------
    predictions : dict or pandas.DataFrame
        Predictions dictionary or DataFrame
    filename : str
        Filename for the submission
    output_dir : str
        Directory to save the submission
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(predictions, dict):
        predictions_df = pd.DataFrame([
            {'ID': match_id, 'Pred': prob}
            for match_id, prob in predictions.items()
        ])
    else:
        predictions_df = predictions
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    filepath = os.path.join(output_dir, filename)
    predictions_df.to_csv(filepath, index=False)
    
    print(f"Saved {len(predictions_df)} predictions to {filepath}")
    
    # Sample of the predictions
    print("\nSample predictions:")
    print(predictions_df.head())


def main():
    """
    Main function to execute data preparation pipeline when script is run directly.
    """
    print("=== NCAA March Madness Data Preparation ===")
    
    # Define data directory (adjust as needed)
    data_dir = 'data'
    print(f"Base data directory: {data_dir}")
    
    # Load the data from raw folder
    data = load_data(data_dir)
    print(f"Loaded {len(data)} data files")
    
    # Clean and combine the data
    print("\nCleaning and combining data...")
    cleaned_data = clean_and_combine_data(data, data_dir)
    
    # Process seeds if available
    if 'TourneySeeds' in cleaned_data:
        print("\nProcessing tournament seeds...")
        seeds_df = preprocess_seeds(cleaned_data['TourneySeeds'])
        # Save processed seeds
        processed_dir = os.path.join(data_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        seeds_df.to_csv(os.path.join(processed_dir, 'ProcessedSeeds.csv'), index=False)
        print(f"Saved processed seeds with {len(seeds_df)} rows")
        
        # Create seed lookup
        seed_lookup = create_seed_lookup(seeds_df)
        print(f"Created seed lookup with {len(seed_lookup)} entries")
    
    # Create matchup data if results are available
    if 'TourneyResults' in cleaned_data:
        print("\nCreating matchup training data...")
        seed_lookup = create_seed_lookup(cleaned_data['TourneySeeds']) if 'TourneySeeds' in cleaned_data else None
        matchups = create_matchup_data(cleaned_data['TourneyResults'], seed_lookup)
        
        # Save matchup data
        processed_dir = os.path.join(data_dir, 'processed')
        matchups.to_csv(os.path.join(processed_dir, 'MatchupData.csv'), index=False)
        print(f"Saved matchup data with {len(matchups)} rows")
    
    print("\n=== Data Preparation Complete ===")
    print(f"Processed data saved to: {os.path.join(data_dir, 'processed')}")
    
    return cleaned_data


if __name__ == "__main__":
    main() 
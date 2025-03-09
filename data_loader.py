#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loader module for March Madness analysis.
"""

import os
import pandas as pd
import numpy as np
import sys

def load_data(data_dir='data'):
    """
    Load NCAA March Madness data files from the data directory.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
        
    Returns:
    --------
    dict
        Dictionary containing DataFrames for each data file
    """
    # Define the path to the data directory
    base_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)
    
    # Use the raw subdirectory
    raw_data_dir = os.path.join(base_data_dir, 'raw')
    
    # Check if raw directory exists, if not fall back to data_dir
    if not os.path.exists(raw_data_dir):
        print(f"Warning: Raw data directory not found at {raw_data_dir}")
        print(f"Falling back to {base_data_dir}")
        raw_data_dir = base_data_dir
    else:
        print(f"Using raw data directory: {raw_data_dir}")
    
    # List of all expected NCAA data files
    expected_files = [
        # Men's Data
        'MRegularSeasonDetailedResults.csv',
        'MRegularSeasonCompactResults.csv',
        'MNCAATourneyCompactResults.csv',
        'MNCAATourneyDetailedResults.csv',
        'MNCAATourneySeeds.csv',
        'MNCAATourneySlots.csv',
        'MNCAATourneySeedRoundSlots.csv',
        'MSeasons.csv',
        'MTeams.csv',
        'MTeamSpellings.csv',
        'MTeamConferences.csv',
        'MTeamCoaches.csv',
        'MConferenceTourneyGames.csv',
        'MMasseyOrdinals.csv',
        'MGameCities.csv',
        'MSecondaryTourneyCompactResults.csv',
        'MSecondaryTourneyTeams.csv',
        
        # Women's Data
        'WRegularSeasonDetailedResults.csv',
        'WRegularSeasonCompactResults.csv',
        'WNCAATourneyCompactResults.csv',
        'WNCAATourneyDetailedResults.csv',
        'WNCAATourneySeeds.csv',
        'WNCAATourneySlots.csv',
        'WSeasons.csv',
        'WTeams.csv',
        'WTeamSpellings.csv',
        'WTeamConferences.csv',
        'WConferenceTourneyGames.csv',
        'WGameCities.csv',
        'WSecondaryTourneyCompactResults.csv',
        'WSecondaryTourneyTeams.csv',
        
        # General Data
        'Cities.csv',
        'Conferences.csv',
        
        # Submission Files
        'SampleSubmissionStage1.csv',
        'SampleSubmissionStage2.csv',
        'SeedBenchmarkStage1.csv',
    ]
    
    # Check if data directory exists
    if not os.path.exists(raw_data_dir):
        print(f"Error: Data directory not found at {raw_data_dir}")
        return None
    
    # Check which files exist
    available_files = []
    missing_files = []
    
    for filename in expected_files:
        file_path = os.path.join(raw_data_dir, filename)
        if os.path.isfile(file_path):
            available_files.append(filename)
        else:
            missing_files.append(filename)
    
    # Report on available and missing files
    print(f"Found {len(available_files)} out of {len(expected_files)} expected data files.")
    
    if missing_files:
        print(f"Warning: {len(missing_files)} files are missing:")
        for file in missing_files[:5]:  # Show first 5 missing files
            print(f"  - {file}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
    
    # Load available files into DataFrames
    data = {}
    for filename in available_files:
        try:
            file_path = os.path.join(raw_data_dir, filename)
            # Remove file extension for the key
            key = os.path.splitext(filename)[0]
            data[key] = pd.read_csv(file_path)
            print(f"Loaded {filename} with {len(data[key])} rows")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return data

def check_required_files(data):
    """
    Check if the minimum required files for analysis are available
    
    Parameters:
    -----------
    data : dict
        Dictionary of loaded DataFrames
    
    Returns:
    --------
    bool
        True if minimum required files are available, False otherwise
    """
    # Define minimum required files for men's tournament analysis
    required_files = [
        'MRegularSeasonDetailedResults',
        'MNCAATourneyCompactResults',
        'MNCAATourneySeeds',
        'MTeams'
    ]
    
    missing = [req for req in required_files if req not in data]
    
    if missing:
        print("Missing required files for analysis:")
        for file in missing:
            print(f"  - {file}.csv")
        return False
    
    return True

def main():
    """Main function to load data and perform basic checks"""
    print("Loading March Madness data files...")
    data = load_data()
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    if not check_required_files(data):
        print("Warning: Some required files are missing. Analysis may be limited.")
    
    # Example of accessing a specific DataFrame
    if 'MTeams' in data:
        print("\nExample: First 5 teams from MTeams.csv:")
        print(data['MTeams'].head())
    
    # Return data for further processing
    return data

if __name__ == "__main__":
    main() 
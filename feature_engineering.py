#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Engineering Module for March Madness Prediction
Provides functions to create advanced basketball metrics and features
"""

import pandas as pd
import numpy as np
from collections import defaultdict


def calculate_advanced_team_stats(regular_season_data, teams_data):
    """
    Calculate advanced team statistics from regular season data
    
    Parameters:
    -----------
    regular_season_data : pandas.DataFrame
        DataFrame containing regular season game results
    teams_data : pandas.DataFrame
        DataFrame containing team information
        
    Returns:
    --------
    dict
        Dictionary with team stats by season
    """
    # Initialize stats by season and team
    seasons = regular_season_data['Season'].unique()
    teams = teams_data['TeamID'].unique()
    
    team_stats_by_season = {}
    
    for season in seasons:
        print(f"Processing season {season}...")
        season_data = regular_season_data[regular_season_data['Season'] == season]
        
        # Initialize team stats
        team_stats = {team: {
            'Games': 0,
            'Wins': 0,
            'Losses': 0,
            'PointsFor': 0,
            'PointsAgainst': 0,
            'PossessionCount': 0,  # Estimated
            'OffensiveRebounds': 0,
            'DefensiveRebounds': 0,
            'TotalRebounds': 0,
            'Assists': 0,
            'Turnovers': 0,
            'Steals': 0,
            'Blocks': 0,
            'HomeWins': 0,
            'AwayWins': 0,
            'NeutralWins': 0,
            'HomeGames': 0,
            'AwayGames': 0,
            'NeutralGames': 0,
            'WinStreak': 0,
            'LastResults': [],  # Last 10 games (1 for win, 0 for loss)
            'EloRating': 1500,  # Starting Elo rating
            'GameResults': [],  # List of point differentials
            'OpponentStrength': []  # List of opponent win percentages
        } for team in teams}
        
        # Process each game chronologically
        for _, game in season_data.sort_values('DayNum').iterrows():
            wteam, lteam = game['WTeamID'], game['LTeamID']
            wscore, lscore = game['WScore'], game['LScore']
            wloc = game['WLoc']
            
            # Update basic stats for winning team
            team_stats[wteam]['Games'] += 1
            team_stats[wteam]['Wins'] += 1
            team_stats[wteam]['PointsFor'] += wscore
            team_stats[wteam]['PointsAgainst'] += lscore
            team_stats[wteam]['WinStreak'] += 1
            team_stats[wteam]['LastResults'].append(1)
            team_stats[wteam]['LastResults'] = team_stats[wteam]['LastResults'][-10:]  # Keep last 10
            team_stats[wteam]['GameResults'].append(wscore - lscore)
            
            # Update basic stats for losing team
            team_stats[lteam]['Games'] += 1
            team_stats[lteam]['Losses'] += 1
            team_stats[lteam]['PointsFor'] += lscore
            team_stats[lteam]['PointsAgainst'] += wscore
            team_stats[lteam]['WinStreak'] = 0
            team_stats[lteam]['LastResults'].append(0)
            team_stats[lteam]['LastResults'] = team_stats[lteam]['LastResults'][-10:]  # Keep last 10
            team_stats[lteam]['GameResults'].append(lscore - wscore)
            
            # Update opponent strength
            if team_stats[wteam]['Games'] > 1:
                win_pct = (team_stats[wteam]['Wins'] - 1) / (team_stats[wteam]['Games'] - 1)
                team_stats[lteam]['OpponentStrength'].append(win_pct)
            else:
                team_stats[lteam]['OpponentStrength'].append(0.5)  # Default for first game
                
            if team_stats[lteam]['Games'] > 1:
                win_pct = team_stats[lteam]['Wins'] / (team_stats[lteam]['Games'] - 1)
                team_stats[wteam]['OpponentStrength'].append(win_pct)
            else:
                team_stats[wteam]['OpponentStrength'].append(0.5)  # Default for first game
            
            # Update location stats
            if wloc == 'H':
                team_stats[wteam]['HomeWins'] += 1
                team_stats[wteam]['HomeGames'] += 1
                team_stats[lteam]['AwayGames'] += 1
            elif wloc == 'A':
                team_stats[wteam]['AwayWins'] += 1
                team_stats[wteam]['AwayGames'] += 1
                team_stats[lteam]['HomeGames'] += 1
            else:  # Neutral
                team_stats[wteam]['NeutralWins'] += 1
                team_stats[wteam]['NeutralGames'] += 1
                team_stats[lteam]['NeutralGames'] += 1
                
            # Update Elo Ratings
            # Expected win probability based on Elo difference
            elo_diff = team_stats[wteam]['EloRating'] - team_stats[lteam]['EloRating']
            expected_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
            
            # K-factor: importance of the game (higher for later games)
            k_factor = 20 + (game['DayNum'] / 132) * 10  # Scale from 20 to 30 throughout season
            
            # Margin of victory multiplier
            score_diff = wscore - lscore
            mov_multiplier = min(1.5, np.log(abs(score_diff) + 1)) * (2.2 / (elo_diff * 0.001 + 2.2))
            
            # Update Elo ratings
            elo_change = k_factor * mov_multiplier * (1 - expected_win_prob)
            team_stats[wteam]['EloRating'] += elo_change
            team_stats[lteam]['EloRating'] -= elo_change
            
            # Basic box score stats are not available, but could be added if data exists
            # Add estimated possessions based on common formula
            # Possessions ≈ FGA + 0.44*FTA + TO - ORB (using averages when not available)
            avg_possessions = 70  # NCAA average
            team_stats[wteam]['PossessionCount'] += avg_possessions / 2
            team_stats[lteam]['PossessionCount'] += avg_possessions / 2
        
        # Calculate derived metrics for each team
        for team in teams:
            stats = team_stats[team]
            games = stats['Games']
            
            if games > 0:
                # Basic derived stats
                stats['WinPct'] = stats['Wins'] / games
                stats['PointsPerGame'] = stats['PointsFor'] / games
                stats['PointsAllowedPerGame'] = stats['PointsAgainst'] / games
                stats['PointDifferential'] = stats['PointsFor'] - stats['PointsAgainst']
                stats['PointDifferentialPerGame'] = stats['PointDifferential'] / games
                
                # Location stats
                home_games = max(1, stats['HomeGames'])
                away_games = max(1, stats['AwayGames'])
                neutral_games = max(1, stats['NeutralGames'])
                stats['HomeWinPct'] = stats['HomeWins'] / home_games
                stats['AwayWinPct'] = stats['AwayWins'] / away_games
                stats['NeutralWinPct'] = stats['NeutralWins'] / neutral_games
                
                # Recent form (last 10 games)
                if stats['LastResults']:
                    stats['RecentForm'] = sum(stats['LastResults']) / len(stats['LastResults'])
                else:
                    stats['RecentForm'] = 0.5
                
                # Advanced metrics
                # Offensive/Defensive Efficiency (pts per 100 possessions)
                possessions = max(1, stats['PossessionCount'])
                stats['OffensiveEfficiency'] = 100 * stats['PointsFor'] / possessions
                stats['DefensiveEfficiency'] = 100 * stats['PointsAgainst'] / possessions
                stats['NetEfficiency'] = stats['OffensiveEfficiency'] - stats['DefensiveEfficiency']
                
                # Game result volatility (standard deviation of point differentials)
                if len(stats['GameResults']) > 1:
                    stats['PointDiffStd'] = np.std(stats['GameResults'])
                else:
                    stats['PointDiffStd'] = 0
                
                # Strength of Schedule
                if stats['OpponentStrength']:
                    stats['StrengthOfSchedule'] = np.mean(stats['OpponentStrength'])
                else:
                    stats['StrengthOfSchedule'] = 0.5
                
                # Adjusted efficiency (accounting for opponent strength)
                sos_factor = stats['StrengthOfSchedule'] / 0.5  # Normalize around 1.0
                stats['AdjustedOffEff'] = stats['OffensiveEfficiency'] * sos_factor
                stats['AdjustedDefEff'] = stats['DefensiveEfficiency'] / sos_factor
                stats['AdjustedNetEff'] = stats['AdjustedOffEff'] - stats['AdjustedDefEff']
            else:
                # Default values for teams with no games
                default_metrics = [
                    'WinPct', 'PointsPerGame', 'PointsAllowedPerGame', 'PointDifferential',
                    'PointDifferentialPerGame', 'HomeWinPct', 'AwayWinPct', 'NeutralWinPct',
                    'RecentForm', 'OffensiveEfficiency', 'DefensiveEfficiency', 'NetEfficiency',
                    'PointDiffStd', 'StrengthOfSchedule', 'AdjustedOffEff', 'AdjustedDefEff',
                    'AdjustedNetEff'
                ]
                for metric in default_metrics:
                    stats[metric] = 0
        
        # Store stats for this season
        team_stats_by_season[season] = team_stats
    
    return team_stats_by_season


def create_matchup_features(team1, team2, team_stats, seeds=None):
    """
    Create features for a specific matchup between two teams
    
    Parameters:
    -----------
    team1 : int
        TeamID of the first team (lower ID)
    team2 : int
        TeamID of the second team (higher ID)
    team_stats : dict
        Dictionary containing team statistics
    seeds : dict, optional
        Dictionary mapping TeamIDs to seeds
        
    Returns:
    --------
    dict
        Dictionary of matchup features
    """
    # Get team stats
    team1_stats = team_stats.get(team1, {})
    team2_stats = team_stats.get(team2, {})
    
    # Handle missing teams
    if not team1_stats or not team2_stats:
        # Create dummy stats based on averages
        avg_stats = {
            'WinPct': 0.5,
            'PointsPerGame': 70,
            'PointsAllowedPerGame': 70,
            'PointDifferentialPerGame': 0,
            'HomeWinPct': 0.5,
            'AwayWinPct': 0.5,
            'NeutralWinPct': 0.5,
            'RecentForm': 0.5,
            'OffensiveEfficiency': 100,
            'DefensiveEfficiency': 100,
            'NetEfficiency': 0,
            'PointDiffStd': 10,
            'StrengthOfSchedule': 0.5,
            'AdjustedOffEff': 100,
            'AdjustedDefEff': 100,
            'AdjustedNetEff': 0,
            'EloRating': 1500
        }
        
        team1_stats = team1_stats or avg_stats
        team2_stats = team2_stats or avg_stats
    
    # Get seeds if available
    seed1 = seeds.get(team1, 16) if seeds else 8
    seed2 = seeds.get(team2, 16) if seeds else 8
    
    # Create basic comparative features
    matchup_features = {
        'SeedDiff': seed1 - seed2,
        'WinPctDiff': team1_stats.get('WinPct', 0) - team2_stats.get('WinPct', 0),
        'PointsPerGameDiff': team1_stats.get('PointsPerGame', 0) - team2_stats.get('PointsPerGame', 0),
        'PointsAllowedDiff': team1_stats.get('PointsAllowedPerGame', 0) - team2_stats.get('PointsAllowedPerGame', 0),
        'PointDiffPerGameDiff': team1_stats.get('PointDifferentialPerGame', 0) - team2_stats.get('PointDifferentialPerGame', 0),
        'RecentFormDiff': team1_stats.get('RecentForm', 0.5) - team2_stats.get('RecentForm', 0.5),
        'OffensiveEffDiff': team1_stats.get('OffensiveEfficiency', 100) - team2_stats.get('OffensiveEfficiency', 100),
        'DefensiveEffDiff': team1_stats.get('DefensiveEfficiency', 100) - team2_stats.get('DefensiveEfficiency', 100),
        'NetEfficiencyDiff': team1_stats.get('NetEfficiency', 0) - team2_stats.get('NetEfficiency', 0),
        'StrengthOfScheduleDiff': team1_stats.get('StrengthOfSchedule', 0.5) - team2_stats.get('StrengthOfSchedule', 0.5),
        'AdjustedOffEffDiff': team1_stats.get('AdjustedOffEff', 100) - team2_stats.get('AdjustedOffEff', 100),
        'AdjustedDefEffDiff': team1_stats.get('AdjustedDefEff', 100) - team2_stats.get('AdjustedDefEff', 100),
        'AdjustedNetEffDiff': team1_stats.get('AdjustedNetEff', 0) - team2_stats.get('AdjustedNetEff', 0),
        'EloDiff': team1_stats.get('EloRating', 1500) - team2_stats.get('EloRating', 1500),
        
        # Team 1 raw features
        'Team1Seed': seed1,
        'Team1WinPct': team1_stats.get('WinPct', 0),
        'Team1PointsPerGame': team1_stats.get('PointsPerGame', 0),
        'Team1PointsAllowed': team1_stats.get('PointsAllowedPerGame', 0),
        'Team1PointDiff': team1_stats.get('PointDifferentialPerGame', 0),
        'Team1NeutralWinPct': team1_stats.get('NeutralWinPct', 0),
        'Team1RecentForm': team1_stats.get('RecentForm', 0.5),
        'Team1OffensiveEff': team1_stats.get('OffensiveEfficiency', 100),
        'Team1DefensiveEff': team1_stats.get('DefensiveEfficiency', 100),
        'Team1NetEfficiency': team1_stats.get('NetEfficiency', 0),
        'Team1SOS': team1_stats.get('StrengthOfSchedule', 0.5),
        'Team1AdjOffEff': team1_stats.get('AdjustedOffEff', 100),
        'Team1AdjDefEff': team1_stats.get('AdjustedDefEff', 100),
        'Team1AdjNetEff': team1_stats.get('AdjustedNetEff', 0),
        'Team1Elo': team1_stats.get('EloRating', 1500),
        
        # Team 2 raw features
        'Team2Seed': seed2,
        'Team2WinPct': team2_stats.get('WinPct', 0),
        'Team2PointsPerGame': team2_stats.get('PointsPerGame', 0),
        'Team2PointsAllowed': team2_stats.get('PointsAllowedPerGame', 0),
        'Team2PointDiff': team2_stats.get('PointDifferentialPerGame', 0),
        'Team2NeutralWinPct': team2_stats.get('NeutralWinPct', 0),
        'Team2RecentForm': team2_stats.get('RecentForm', 0.5),
        'Team2OffensiveEff': team2_stats.get('OffensiveEfficiency', 100),
        'Team2DefensiveEff': team2_stats.get('DefensiveEfficiency', 100),
        'Team2NetEfficiency': team2_stats.get('NetEfficiency', 0),
        'Team2SOS': team2_stats.get('StrengthOfSchedule', 0.5),
        'Team2AdjOffEff': team2_stats.get('AdjustedOffEff', 100),
        'Team2AdjDefEff': team2_stats.get('AdjustedDefEff', 100),
        'Team2AdjNetEff': team2_stats.get('AdjustedNetEff', 0),
        'Team2Elo': team2_stats.get('EloRating', 1500),
    }
    
    # Add interaction features
    matchup_features['Elo_WinPct_Interaction'] = matchup_features['EloDiff'] * matchup_features['WinPctDiff']
    matchup_features['OffDef_Interaction'] = team1_stats.get('OffensiveEfficiency', 100) - team2_stats.get('DefensiveEfficiency', 100)
    matchup_features['DefOff_Interaction'] = team1_stats.get('DefensiveEfficiency', 100) - team2_stats.get('OffensiveEfficiency', 100)
    
    # Add win probability estimate based on Elo
    elo_diff = matchup_features['EloDiff']
    matchup_features['Elo_WinProb'] = 1 / (1 + 10 ** (-elo_diff / 400))
    
    # Add seed upset potential (higher values mean more upset potential)
    if seed1 > seed2:  # Team 1 (lower ID) is lower seeded (higher numerical seed)
        upset_potential = (team1_stats.get('AdjustedNetEff', 0) - team2_stats.get('AdjustedNetEff', 0)) * (seed1 - seed2)
    else:  # Team 1 is higher seeded
        upset_potential = 0
    matchup_features['UpsetPotential'] = upset_potential
    
    return matchup_features


def calculate_team_momentum(team_results, window=10):
    """
    Calculate team momentum based on recent games
    
    Parameters:
    -----------
    team_results : list
        List of tuples (daynum, is_win, point_diff, opponent_rating)
    window : int
        Number of recent games to consider
    
    Returns:
    --------
    float
        Momentum score (weighted recent performance)
    """
    if not team_results or len(team_results) < 2:
        return 0.0
    
    # Sort by day number
    results = sorted(team_results[-window:], key=lambda x: x[0])
    
    # Calculate weights (more recent games have higher weight)
    weights = np.linspace(0.5, 1.0, len(results))
    
    # Calculate weighted win percentage and point differential
    win_momentum = sum(w * r[1] for w, r in zip(weights, results)) / sum(weights)
    
    # Weight point differential by opponent rating
    point_momentum = sum(w * r[2] * r[3] for w, r in zip(weights, results)) / sum(weights)
    
    # Combine into single momentum metric
    momentum = (win_momentum + 0.01 * point_momentum) / 1.01
    
    return momentum


def encode_seed(seed_str):
    """
    Encode tournament seed as a numeric value
    
    Parameters:
    -----------
    seed_str : str
        Seed string (e.g., '1a', '16b')
    
    Returns:
    --------
    int
        Numeric seed value
    """
    if not isinstance(seed_str, str):
        return 16  # Default high seed
    
    # Extract numeric part and handle play-in teams
    seed_num = int(''.join(c for c in seed_str if c.isdigit()))
    
    # Adjust for play-in indicator
    if 'a' in seed_str.lower():
        seed_num -= 0.25
    elif 'b' in seed_str.lower():
        seed_num -= 0.5
    
    return seed_num


def create_historical_matchup_features(team1, team2, historical_games):
    """
    Create features based on historical matchups between two teams
    
    Parameters:
    -----------
    team1 : int
        TeamID of first team
    team2 : int
        TeamID of second team
    historical_games : pandas.DataFrame
        DataFrame containing historical game results
    
    Returns:
    --------
    dict
        Dictionary of historical matchup features
    """
    # Filter games between these teams
    matchups = historical_games[
        ((historical_games['WTeamID'] == team1) & (historical_games['LTeamID'] == team2)) |
        ((historical_games['WTeamID'] == team2) & (historical_games['LTeamID'] == team1))
    ]
    
    if len(matchups) == 0:
        return {
            'HistoricalMatchups': 0,
            'Team1WinPct': 0.5,
            'AvgPointDiff': 0,
            'LastMatchupDayDiff': 999,
            'RecentMatchupAdvantage': 0
        }
    
    # Calculate features
    total_games = len(matchups)
    team1_wins = len(matchups[(matchups['WTeamID'] == team1)])
    team1_win_pct = team1_wins / total_games if total_games > 0 else 0.5
    
    # Calculate average point differential (from team1's perspective)
    point_diffs = []
    for _, game in matchups.iterrows():
        if game['WTeamID'] == team1:
            point_diffs.append(game['WScore'] - game['LScore'])
        else:
            point_diffs.append(game['LScore'] - game['WScore'])
    
    avg_point_diff = sum(point_diffs) / len(point_diffs) if point_diffs else 0
    
    # Days since last matchup
    last_matchup = matchups.sort_values('DayNum', ascending=False).iloc[0]
    last_matchup_day = last_matchup['DayNum']
    days_since_last = 132 - last_matchup_day  # Assuming 132 days in regular season
    
    # Who won the most recent matchup?
    recent_advantage = 1 if last_matchup['WTeamID'] == team1 else -1
    
    return {
        'HistoricalMatchups': total_games,
        'Team1WinPct': team1_win_pct,
        'AvgPointDiff': avg_point_diff,
        'LastMatchupDayDiff': days_since_last,
        'RecentMatchupAdvantage': recent_advantage
    }


def create_conference_features(team1, team2, teams_data, conferences=None):
    """
    Create features related to team conferences
    
    Parameters:
    -----------
    team1 : int
        TeamID of first team
    team2 : int
        TeamID of second team
    teams_data : pandas.DataFrame
        DataFrame containing team information
    conferences : dict, optional
        Dictionary with conference strength metrics
    
    Returns:
    --------
    dict
        Dictionary of conference-related features
    """
    # Get conference for each team
    team1_conf = teams_data[teams_data['TeamID'] == team1]['ConfAbbrev'].values[0] if len(teams_data[teams_data['TeamID'] == team1]) > 0 else 'Unknown'
    team2_conf = teams_data[teams_data['TeamID'] == team2]['ConfAbbrev'].values[0] if len(teams_data[teams_data['TeamID'] == team2]) > 0 else 'Unknown'
    
    # Same conference indicator
    same_conf = 1 if team1_conf == team2_conf else 0
    
    # Conference strength (if provided)
    if conferences:
        conf1_strength = conferences.get(team1_conf, {'Rating': 0.5}).get('Rating', 0.5)
        conf2_strength = conferences.get(team2_conf, {'Rating': 0.5}).get('Rating', 0.5)
        conf_diff = conf1_strength - conf2_strength
    else:
        conf1_strength = 0.5
        conf2_strength = 0.5
        conf_diff = 0
    
    return {
        'SameConference': same_conf,
        'Team1ConfStrength': conf1_strength,
        'Team2ConfStrength': conf2_strength,
        'ConfStrengthDiff': conf_diff
    }


def calculate_conference_strength(regular_season_data, teams_data):
    """
    Calculate conference strength based on non-conference game results
    
    Parameters:
    -----------
    regular_season_data : pandas.DataFrame
        DataFrame containing regular season game results
    teams_data : pandas.DataFrame
        DataFrame containing team information
    
    Returns:
    --------
    dict
        Dictionary mapping conference abbreviations to strength metrics
    """
    # Create team to conference mapping
    team_conf = {}
    for _, team in teams_data.iterrows():
        team_conf[team['TeamID']] = team['ConfAbbrev']
    
    # Initialize conference stats
    conf_stats = defaultdict(lambda: {
        'NonConfGames': 0,
        'NonConfWins': 0,
        'PointDiffTotal': 0,
        'Games': 0
    })
    
    # Process each game
    for _, game in regular_season_data.iterrows():
        wteam, lteam = game['WTeamID'], game['LTeamID']
        wscore, lscore = game['WScore'], game['LScore']
        
        # Skip if team not found in mapping
        if wteam not in team_conf or lteam not in team_conf:
            continue
            
        wconf = team_conf[wteam]
        lconf = team_conf[lteam]
        
        # Update total games for conferences
        conf_stats[wconf]['Games'] += 1
        conf_stats[lconf]['Games'] += 1
        
        # Process non-conference games
        if wconf != lconf:
            # Update winner conference
            conf_stats[wconf]['NonConfGames'] += 1
            conf_stats[wconf]['NonConfWins'] += 1
            conf_stats[wconf]['PointDiffTotal'] += (wscore - lscore)
            
            # Update loser conference
            conf_stats[lconf]['NonConfGames'] += 1
            conf_stats[lconf]['PointDiffTotal'] -= (wscore - lscore)
    
    # Calculate derived metrics
    for conf, stats in conf_stats.items():
        if stats['NonConfGames'] > 0:
            stats['WinPct'] = stats['NonConfWins'] / stats['NonConfGames']
            stats['AvgPointDiff'] = stats['PointDiffTotal'] / stats['NonConfGames']
            
            # Create overall rating (weighted combination of win pct and point diff)
            win_pct_z = (stats['WinPct'] - 0.5) / 0.15  # Standardize around mean 0.5, SD 0.15
            point_diff_z = stats['AvgPointDiff'] / 5  # Standardize around mean 0, SD 5
            stats['Rating'] = 0.7 * win_pct_z + 0.3 * point_diff_z
        else:
            stats['WinPct'] = 0.5
            stats['AvgPointDiff'] = 0
            stats['Rating'] = 0
    
    # Normalize ratings from 0 to 1
    ratings = [stats['Rating'] for stats in conf_stats.values()]
    if ratings:
        min_rating, max_rating = min(ratings), max(ratings)
        rating_range = max_rating - min_rating
        
        if rating_range > 0:
            for conf, stats in conf_stats.items():
                stats['Rating'] = (stats['Rating'] - min_rating) / rating_range
        
    return dict(conf_stats)


def create_features(teams_df, games_df, tourney_results_df):
    """
    Create features for predicting March Madness tournament outcomes.
    
    Parameters
    ----------
    teams_df : DataFrame
        Teams information
    games_df : DataFrame
        Regular season games data
    tourney_results_df : DataFrame
        Historical tournament results
        
    Returns
    -------
    DataFrame
        Features dataframe with engineered features for each team or matchup
    """
    print("Creating team performance features...")
    
    # Create a features dataframe for each team
    team_features = pd.DataFrame()
    team_features['team_id'] = teams_df['team_id']
    
    # Calculate win percentage for each team
    team_stats = calculate_team_stats(games_df)
    
    # Merge team stats with team_features
    team_features = pd.merge(team_features, team_stats, on='team_id', how='left')
    
    # Create advanced metrics
    team_features = create_advanced_metrics(team_features, games_df)
    
    # Calculate tournament performance metrics
    if 'season' in tourney_results_df.columns:
        tourney_features = calculate_tournament_performance(tourney_results_df)
        team_features = pd.merge(team_features, tourney_features, on='team_id', how='left')
    
    # Fill missing values
    team_features = team_features.fillna(0)
    
    # Create matchup features (for prediction)
    matchup_features = create_matchup_features(team_features, tourney_results_df)
    
    return matchup_features


def calculate_team_stats(games_df):
    """
    Calculate basic team statistics from games data.
    
    Parameters
    ----------
    games_df : DataFrame
        Regular season games data
        
    Returns
    -------
    DataFrame
        Team statistics
    """
    team_stats = pd.DataFrame()
    unique_teams = pd.unique(pd.concat([games_df['team1_id'], games_df['team2_id']]))
    team_stats['team_id'] = unique_teams
    
    # Initialize stats
    team_stats['games_played'] = 0
    team_stats['wins'] = 0
    team_stats['losses'] = 0
    team_stats['points_scored'] = 0
    team_stats['points_allowed'] = 0
    
    # Calculate stats for each team
    for _, game in games_df.iterrows():
        # Team 1 stats
        team1_idx = team_stats[team_stats['team_id'] == game['team1_id']].index
        team_stats.loc[team1_idx, 'games_played'] += 1
        team_stats.loc[team1_idx, 'points_scored'] += game['team1_score']
        team_stats.loc[team1_idx, 'points_allowed'] += game['team2_score']
        if game['team1_score'] > game['team2_score']:
            team_stats.loc[team1_idx, 'wins'] += 1
        else:
            team_stats.loc[team1_idx, 'losses'] += 1
            
        # Team 2 stats
        team2_idx = team_stats[team_stats['team_id'] == game['team2_id']].index
        team_stats.loc[team2_idx, 'games_played'] += 1
        team_stats.loc[team2_idx, 'points_scored'] += game['team2_score']
        team_stats.loc[team2_idx, 'points_allowed'] += game['team1_score']
        if game['team2_score'] > game['team1_score']:
            team_stats.loc[team2_idx, 'wins'] += 1
        else:
            team_stats.loc[team2_idx, 'losses'] += 1
    
    # Calculate derived stats
    team_stats['win_pct'] = team_stats['wins'] / team_stats['games_played']
    team_stats['avg_points_scored'] = team_stats['points_scored'] / team_stats['games_played']
    team_stats['avg_points_allowed'] = team_stats['points_allowed'] / team_stats['games_played']
    team_stats['point_differential'] = team_stats['avg_points_scored'] - team_stats['avg_points_allowed']
    
    return team_stats


def create_advanced_metrics(team_features, games_df):
    """
    Create advanced basketball metrics for each team.
    
    Parameters
    ----------
    team_features : DataFrame
        Basic team features
    games_df : DataFrame
        Regular season games data
        
    Returns
    -------
    DataFrame
        Team features with advanced metrics added
    """
    # Calculate strength of schedule
    sos = calculate_strength_of_schedule(team_features, games_df)
    team_features['strength_of_schedule'] = sos
    
    # Calculate offensive and defensive efficiency
    team_features['offensive_efficiency'] = team_features['avg_points_scored'] * 100 / 70  # per 100 possessions (estimated)
    team_features['defensive_efficiency'] = team_features['avg_points_allowed'] * 100 / 70
    
    # Calculate adjusted metrics based on strength of schedule
    team_features['adj_offensive_efficiency'] = team_features['offensive_efficiency'] * (1 + team_features['strength_of_schedule'] / 100)
    team_features['adj_defensive_efficiency'] = team_features['defensive_efficiency'] / (1 + team_features['strength_of_schedule'] / 100)
    
    # Overall efficiency margin
    team_features['efficiency_margin'] = team_features['adj_offensive_efficiency'] - team_features['adj_defensive_efficiency']
    
    return team_features


def calculate_strength_of_schedule(team_features, games_df):
    """
    Calculate strength of schedule (SOS) for each team.
    
    Parameters
    ----------
    team_features : DataFrame
        Team features
    games_df : DataFrame
        Games data
        
    Returns
    -------
    Series
        Strength of schedule for each team
    """
    # Simplified SOS: average win % of opponents
    sos = pd.Series(index=team_features['team_id'], data=0.0)
    
    for team_id in team_features['team_id']:
        # Get games involving this team
        team1_games = games_df[games_df['team1_id'] == team_id]
        team2_games = games_df[games_df['team2_id'] == team_id]
        
        # Get opponents
        opponents1 = team1_games['team2_id'].values
        opponents2 = team2_games['team1_id'].values
        opponents = np.concatenate([opponents1, opponents2])
        
        # Get opponents' win percentages
        opponents_win_pct = team_features[team_features['team_id'].isin(opponents)]['win_pct'].values
        
        # Calculate SOS
        if len(opponents_win_pct) > 0:
            sos[team_id] = opponents_win_pct.mean()
    
    return sos


def calculate_tournament_performance(tourney_results_df):
    """
    Calculate historical tournament performance metrics.
    
    Parameters
    ----------
    tourney_results_df : DataFrame
        Tournament results data
        
    Returns
    -------
    DataFrame
        Tournament performance metrics for each team
    """
    # Initialize metrics
    tourney_performance = pd.DataFrame()
    unique_teams = pd.unique(pd.concat([tourney_results_df['team1_id'], tourney_results_df['team2_id']]))
    tourney_performance['team_id'] = unique_teams
    
    # Calculate metrics
    tourney_performance['tourney_appearances'] = 0
    tourney_performance['tourney_wins'] = 0
    tourney_performance['final_four_appearances'] = 0
    tourney_performance['championships'] = 0
    
    # Count appearances and wins
    for team_id in tourney_performance['team_id']:
        # Count appearances
        team1_games = tourney_results_df[tourney_results_df['team1_id'] == team_id]
        team2_games = tourney_results_df[tourney_results_df['team2_id'] == team_id]
        
        # Assuming each season has one tournament per team
        if 'season' in tourney_results_df.columns:
            seasons1 = team1_games['season'].unique()
            seasons2 = team2_games['season'].unique()
            all_seasons = np.unique(np.concatenate([seasons1, seasons2]) if len(seasons1) > 0 and len(seasons2) > 0 else 
                                    (seasons1 if len(seasons1) > 0 else seasons2))
            tourney_performance.loc[tourney_performance['team_id'] == team_id, 'tourney_appearances'] = len(all_seasons)
        else:
            # If no season column, just count games as appearances
            appearances = len(team1_games) + len(team2_games)
            tourney_performance.loc[tourney_performance['team_id'] == team_id, 'tourney_appearances'] = appearances
        
        # Count wins
        team1_wins = len(team1_games[team1_games['winner_id'] == team_id])
        team2_wins = len(team2_games[team2_games['winner_id'] == team_id])
        total_wins = team1_wins + team2_wins
        tourney_performance.loc[tourney_performance['team_id'] == team_id, 'tourney_wins'] = total_wins
        
        # Count Final Four appearances (round 5) and championships (round 6)
        if 'round' in tourney_results_df.columns:
            # Final Four appearances
            ff_games1 = team1_games[team1_games['round'] >= 5]
            ff_games2 = team2_games[team2_games['round'] >= 5]
            ff_seasons = np.unique(np.concatenate([ff_games1['season'].unique(), ff_games2['season'].unique()])
                                if len(ff_games1) > 0 and len(ff_games2) > 0 else 
                                (ff_games1['season'].unique() if len(ff_games1) > 0 else ff_games2['season'].unique()))
            tourney_performance.loc[tourney_performance['team_id'] == team_id, 'final_four_appearances'] = len(ff_seasons)
            
            # Championships
            champ_games1 = team1_games[(team1_games['round'] == 6) & (team1_games['winner_id'] == team_id)]
            champ_games2 = team2_games[(team2_games['round'] == 6) & (team2_games['winner_id'] == team_id)]
            championships = len(champ_games1) + len(champ_games2)
            tourney_performance.loc[tourney_performance['team_id'] == team_id, 'championships'] = championships
    
    return tourney_performance


if __name__ == "__main__":
    # Test feature engineering with sample data
    from data_loader import get_sample_data
    
    teams_df, games_df, tourney_results_df = get_sample_data()
    features_df = create_features(teams_df, games_df, tourney_results_df)
    
    print("Feature engineering complete!")
    print(f"Created {features_df.shape[1]} features for {features_df.shape[0]} matchups.")
    print("\nSample features:")
    print(features_df.head()) 
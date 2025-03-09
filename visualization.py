#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module for March Madness Prediction
Provides functions for visualizing tournament data and predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx


def setup_plotting_style():
    """Set up plotting style for consistent visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Define custom colors
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    sns.set_palette(sns.color_palette(colors))


def plot_win_probability_heatmap(teams_data, probabilities, year, gender, output_dir=None):
    """
    Create a heatmap of win probabilities between teams
    
    Parameters:
    -----------
    teams_data : pandas.DataFrame
        DataFrame containing team information
    probabilities : dict
        Dictionary mapping team pairs to win probabilities
    year : int
        Year of the tournament
    gender : str
        'M' for men's tournament, 'W' for women's tournament
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Filter teams by gender
    teams = teams_data[teams_data['Gender'] == gender]
    
    # Sort teams by some metric (e.g., TeamID or team name)
    teams = teams.sort_values('TeamName')
    
    # Create empty probability matrix
    n_teams = len(teams)
    prob_matrix = np.zeros((n_teams, n_teams))
    
    # Fill probability matrix
    for i, team1 in enumerate(teams['TeamID']):
        for j, team2 in enumerate(teams['TeamID']):
            if i != j:
                # Ensure lower TeamID is first
                if team1 < team2:
                    key = f"{year}_{team1}_{team2}"
                    prob = probabilities.get(key, 0.5)
                else:
                    key = f"{year}_{team2}_{team1}"
                    prob = 1.0 - probabilities.get(key, 0.5)
                
                prob_matrix[i, j] = prob
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Create custom colormap (white for 0.5, blue for lower, red for higher)
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        [(0, '#3498db'), (0.5, '#f8f9fa'), (1, '#e74c3c')],
        N=256
    )
    
    # Create heatmap
    sns.heatmap(
        prob_matrix,
        cmap=cmap,
        center=0.5,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Win Probability"},
        ax=ax
    )
    
    # Set labels
    ax.set_xticks(np.arange(n_teams) + 0.5)
    ax.set_yticks(np.arange(n_teams) + 0.5)
    
    # Get team names for labels
    team_names = teams['TeamName'].tolist()
    
    # Set tick labels
    ax.set_xticklabels(team_names, rotation=90)
    ax.set_yticklabels(team_names, rotation=0)
    
    # Add title
    gender_label = "Men's" if gender == 'M' else "Women's"
    ax.set_title(f"{year} {gender_label} Tournament Win Probability Matrix", fontsize=16, pad=20)
    
    # Add axes labels
    ax.set_xlabel('Team 2', fontsize=14, labelpad=10)
    ax.set_ylabel('Team 1', fontsize=14, labelpad=10)
    
    # Add annotation describing the matrix
    ax.text(
        0.5, -0.07,
        "Each cell (i, j) shows the probability that Team i (row) will beat Team j (column)",
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=12
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"winprob_heatmap_{year}_{gender}.png"), dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_team_win_probability_distribution(team_id, teams_data, probabilities, year, gender, output_dir=None):
    """
    Plot the distribution of win probabilities for a specific team
    
    Parameters:
    -----------
    team_id : int
        ID of the team to analyze
    teams_data : pandas.DataFrame
        DataFrame containing team information
    probabilities : dict
        Dictionary mapping team pairs to win probabilities
    year : int
        Year of the tournament
    gender : str
        'M' for men's tournament, 'W' for women's tournament
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Filter teams by gender
    teams = teams_data[teams_data['Gender'] == gender]
    
    # Get team name
    team_name = teams[teams['TeamID'] == team_id]['TeamName'].iloc[0]
    
    # Get all probabilities involving this team
    win_probs = []
    loss_probs = []
    opponent_names = []
    
    for _, opponent in teams.iterrows():
        if opponent['TeamID'] != team_id:
            # Ensure lower TeamID is first
            if team_id < opponent['TeamID']:
                key = f"{year}_{team_id}_{opponent['TeamID']}"
                prob = probabilities.get(key, 0.5)
                win_probs.append(prob)
                loss_probs.append(1.0 - prob)
            else:
                key = f"{year}_{opponent['TeamID']}_{team_id}"
                prob = probabilities.get(key, 0.5)
                win_probs.append(1.0 - prob)
                loss_probs.append(prob)
            
            opponent_names.append(opponent['TeamName'])
    
    # Sort by win probability
    sorted_indices = np.argsort(win_probs)
    win_probs = [win_probs[i] for i in sorted_indices]
    loss_probs = [loss_probs[i] for i in sorted_indices]
    opponent_names = [opponent_names[i] for i in sorted_indices]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Plot probability distribution
    ax1.barh(opponent_names, win_probs, color='#3498db', alpha=0.7)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Win Probability', fontsize=12)
    ax1.set_title(f"Win Probabilities for {team_name}", fontsize=14)
    ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax1.grid(True, axis='x')
    
    # Plot histogram of win probabilities
    ax2.hist(win_probs, bins=10, alpha=0.7, color='#3498db')
    ax2.set_xlabel('Win Probability', fontsize=12)
    ax2.set_ylabel('Number of Matchups', fontsize=12)
    ax2.set_title(f"Distribution of Win Probabilities for {team_name}", fontsize=14)
    ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(np.mean(win_probs), color='red', linestyle='-', alpha=0.7, 
               label=f'Mean: {np.mean(win_probs):.3f}')
    ax2.grid(True)
    ax2.legend()
    
    # Add an explanation
    plt.figtext(
        0.5, 0.01,
        f"This visualization shows predicted win probabilities for {team_name} against all other teams in the {year} {gender} tournament.",
        ha='center',
        fontsize=12,
        wrap=True
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"team_winprob_{team_id}_{year}_{gender}.png"), dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def plot_seed_performance_matrix(tourney_results, seeds_data, output_dir=None):
    """
    Plot historical performance by seed matchup
    
    Parameters:
    -----------
    tourney_results : pandas.DataFrame
        DataFrame containing tournament results
    seeds_data : pandas.DataFrame
        DataFrame containing seed information
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Merge seeds with results
    results = tourney_results.copy()
    
    # Create seed lookup dictionary
    seed_lookup = {}
    for _, row in seeds_data.iterrows():
        season = row['Season']
        gender = row.get('Gender', 'M')  # Default to 'M' if gender not provided
        team_id = row['TeamID']
        seed_num = int(''.join(filter(str.isdigit, row['Seed'])))  # Extract numeric part of seed
        
        seed_lookup[(season, gender, team_id)] = seed_num
    
    # Add seed columns
    results['WSeed'] = results.apply(
        lambda x: seed_lookup.get((x['Season'], x.get('Gender', 'M'), x['WTeamID']), 16),
        axis=1
    )
    results['LSeed'] = results.apply(
        lambda x: seed_lookup.get((x['Season'], x.get('Gender', 'M'), x['LTeamID']), 16),
        axis=1
    )
    
    # Create seed matchup matrix
    seed_matrix = np.zeros((16, 16))
    matchup_counts = np.zeros((16, 16))
    
    for _, game in results.iterrows():
        w_seed = min(game['WSeed'] - 1, 15)  # 0-index seeds
        l_seed = min(game['LSeed'] - 1, 15)  # 0-index seeds
        
        # Increment count of matchup
        matchup_counts[w_seed, l_seed] += 1
        matchup_counts[l_seed, w_seed] += 1
        
        # Increment win for higher seed
        if w_seed < l_seed:
            seed_matrix[w_seed, l_seed] += 1
        else:
            seed_matrix[l_seed, w_seed] += 0  # Lower seed won
    
    # Calculate win probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        win_prob_matrix = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                if matchup_counts[i, j] > 0:
                    if i < j:  # Higher seed vs lower seed
                        win_prob_matrix[i, j] = seed_matrix[i, j] / matchup_counts[i, j]
                    else:  # Lower seed vs higher seed
                        win_prob_matrix[i, j] = 1.0 - (seed_matrix[j, i] / matchup_counts[j, i])
    
    # Create mask for cells with no matchups
    mask = matchup_counts == 0
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        [(0, '#3498db'), (0.5, '#f8f9fa'), (1, '#e74c3c')],
        N=256
    )
    
    # Create count heatmap
    sns.heatmap(
        matchup_counts,
        cmap='viridis',
        annot=True,
        fmt='d',
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Number of Matchups"},
        ax=ax1
    )
    
    # Set labels for count heatmap
    ax1.set_title("Historical Seed Matchup Frequency", fontsize=14)
    ax1.set_xlabel("Seed", fontsize=12)
    ax1.set_ylabel("Seed", fontsize=12)
    
    # Create probability heatmap
    sns.heatmap(
        win_prob_matrix,
        cmap=cmap,
        annot=True,
        fmt='.2f',
        square=True,
        mask=mask,
        center=0.5,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Win Probability (Row vs Column)"},
        ax=ax2
    )
    
    # Set labels for probability heatmap
    ax2.set_title("Historical Seed Matchup Win Probabilities", fontsize=14)
    ax2.set_xlabel("Seed", fontsize=12)
    ax2.set_ylabel("Seed", fontsize=12)
    
    # Set tick labels for both axes
    seed_labels = [str(i+1) for i in range(16)]
    for ax in [ax1, ax2]:
        ax.set_xticklabels(seed_labels)
        ax.set_yticklabels(seed_labels)
    
    # Add annotation describing the matrix
    plt.figtext(
        0.5, 0.01,
        "Left: Number of historical matchups between each seed pairing. Right: Probability of row seed beating column seed.",
        ha='center',
        fontsize=12,
        wrap=True
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "seed_performance_matrix.png"), dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def visualize_team_clusters(team_stats, teams_data, n_clusters=5, method='pca', gender='M', output_dir=None):
    """
    Visualize team clusters based on team statistics
    
    Parameters:
    -----------
    team_stats : dict
        Dictionary containing team statistics
    teams_data : pandas.DataFrame
        DataFrame containing team information
    n_clusters : int
        Number of clusters to create
    method : str
        Dimensionality reduction method ('pca' or 'tsne')
    gender : str
        'M' for men's tournament, 'W' for women's tournament
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Filter teams by gender
    teams = teams_data[teams_data['Gender'] == gender]
    
    # Get latest season
    seasons = sorted(team_stats.keys())
    latest_season = seasons[-1] if seasons else None
    
    if not latest_season:
        return None, None
    
    # Get team stats for the latest season
    season_stats = team_stats[latest_season]
    
    # Create DataFrame of team statistics
    stats_list = []
    for team_id, stats in season_stats.items():
        if team_id in teams['TeamID'].values:
            team_dict = {
                'TeamID': team_id,
                'WinPct': stats.get('WinPct', 0),
                'PointsPerGame': stats.get('PointsPerGame', 0),
                'OppPointsPerGame': stats.get('OppPointsPerGame', 0),
                'PointDiffPerGame': stats.get('PointDiffPerGame', 0),
                'OffensiveEfficiency': stats.get('OffensiveEfficiency', 0),
                'DefensiveEfficiency': stats.get('DefensiveEfficiency', 0),
                'NetEfficiency': stats.get('NetEfficiency', 0)
            }
            stats_list.append(team_dict)
    
    stats_df = pd.DataFrame(stats_list)
    
    # Merge with team names
    stats_df = pd.merge(stats_df, teams[['TeamID', 'TeamName']], on='TeamID')
    
    # Extract features for clustering
    features = [
        'WinPct', 'PointsPerGame', 'OppPointsPerGame', 'PointDiffPerGame', 
        'OffensiveEfficiency', 'DefensiveEfficiency', 'NetEfficiency'
    ]
    X = stats_df[features].values
    
    # Standardize the features
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Perform dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X_std)
        method_name = 'PCA'
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X_std)
        method_name = 't-SNE'
    
    # Perform k-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_std)
    
    # Add cluster and reduced features to DataFrame
    stats_df['Cluster'] = clusters
    stats_df['X1'] = X_reduced[:, 0]
    stats_df['X2'] = X_reduced[:, 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each cluster
    for i in range(n_clusters):
        cluster_data = stats_df[stats_df['Cluster'] == i]
        ax.scatter(
            cluster_data['X1'],
            cluster_data['X2'],
            alpha=0.7,
            s=100,
            label=f'Cluster {i+1}'
        )
        
        # Add team name labels
        for _, row in cluster_data.iterrows():
            ax.text(
                row['X1'],
                row['X2'],
                row['TeamName'],
                fontsize=8,
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
            )
    
    # Calculate cluster statistics
    cluster_stats = stats_df.groupby('Cluster')[features].mean()
    
    # Add cluster descriptions based on statistics
    cluster_descriptions = []
    for i, row in cluster_stats.iterrows():
        if row['OffensiveEfficiency'] > row['DefensiveEfficiency']:
            style = "Offensive"
        else:
            style = "Defensive"
        
        if row['WinPct'] > 0.7:
            strength = "Elite"
        elif row['WinPct'] > 0.6:
            strength = "Strong"
        elif row['WinPct'] > 0.5:
            strength = "Average"
        else:
            strength = "Struggling"
        
        description = f"Cluster {i+1}: {strength} {style} Teams"
        cluster_descriptions.append(description)
    
    # Add cluster descriptions to legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [f"{label} - {desc.split(': ')[1]}" for label, desc in zip(labels, cluster_descriptions)]
    ax.legend(handles, new_labels, loc='upper right', fontsize=10)
    
    # Add title and labels
    gender_label = "Men's" if gender == 'M' else "Women's"
    ax.set_title(f"{gender_label} Basketball Team Clusters ({method_name})", fontsize=16)
    ax.set_xlabel(f"{method_name} Component 1", fontsize=12)
    ax.set_ylabel(f"{method_name} Component 2", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add explanatory text
    plt.figtext(
        0.5, 0.01,
        f"Teams clustered based on statistical performance. Each cluster represents teams with similar playing styles and effectiveness.",
        ha='center',
        fontsize=12,
        wrap=True
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"team_clusters_{gender}_{method}.png"), dpi=300, bbox_inches='tight')
    
    return fig, ax, stats_df


def create_tournament_prediction_visualization(bracket_data, predictions, teams_data, output_dir=None):
    """
    Create a tournament bracket visualization with win probabilities
    
    Parameters:
    -----------
    bracket_data : dict
        Dictionary containing bracket structure
    predictions : dict
        Dictionary mapping team pairs to win probabilities
    teams_data : pandas.DataFrame
        DataFrame containing team information
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes (teams) and edges (games)
    for region in bracket_data['regions']:
        region_name = region['name']
        
        for seed, team_id in region['teams'].items():
            # Get team name
            team_name = teams_data[teams_data['TeamID'] == team_id]['TeamName'].iloc[0]
            node_id = f"{region_name}_{seed}"
            G.add_node(node_id, label=f"{seed}. {team_name}", teamid=team_id)
        
        # Add edges for each round
        for round_num, matchups in region['matchups'].items():
            for matchup in matchups:
                team1_id = f"{region_name}_{matchup[0]}"
                team2_id = f"{region_name}_{matchup[1]}"
                
                # Get team IDs
                team1 = G.nodes[team1_id]['teamid']
                team2 = G.nodes[team2_id]['teamid']
                
                # Get win probability
                if team1 < team2:
                    key = f"{bracket_data['year']}_{team1}_{team2}"
                    prob = predictions.get(key, 0.5)
                else:
                    key = f"{bracket_data['year']}_{team2}_{team1}"
                    prob = 1.0 - predictions.get(key, 0.5)
                
                # Add edge with probability
                G.add_edge(team1_id, f"{region_name}_R{round_num}", weight=prob)
                G.add_edge(team2_id, f"{region_name}_R{round_num}", weight=1-prob)
    
    # Add final four and championship nodes
    G.add_node("Final_Four", label="Final Four")
    G.add_node("Championship", label="Championship")
    G.add_node("Champion", label="Champion")
    
    # Add final four matchups
    for matchup in bracket_data['final_four']:
        region1, region2 = matchup
        
        # Add edges from regional finals to final four
        G.add_edge(f"{region1}_R4", "Final_Four", weight=0.5)
        G.add_edge(f"{region2}_R4", "Final_Four", weight=0.5)
    
    # Add championship game
    G.add_edge("Final_Four", "Championship", weight=0.5)
    G.add_edge("Championship", "Champion", weight=1.0)
    
    # Create layout
    pos = {}
    
    # Position regional nodes
    region_positions = {
        "East": (0, 0),
        "West": (1, 0),
        "South": (0, 1),
        "Midwest": (1, 1)
    }
    
    for region, pos_xy in region_positions.items():
        x_base, y_base = pos_xy
        
        # Position teams in the region
        for seed in range(1, 17):
            pos[f"{region}_{seed}"] = (x_base * 10 + 1, y_base * 10 + seed)
        
        # Position round nodes
        for round_num in range(1, 5):
            pos[f"{region}_R{round_num}"] = (x_base * 10 + round_num + 1, y_base * 10 + 8)
    
    # Position final rounds
    pos["Final_Four"] = (7, 10)
    pos["Championship"] = (9, 10)
    pos["Champion"] = (11, 10)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=8)
    
    # Draw edges with probability-based coloring
    for u, v, data in G.edges(data=True):
        prob = data.get('weight', 0.5)
        
        # Color edges based on probability
        if prob > 0.7:
            color = '#e74c3c'  # Strong favorite
            width = 2.0
        elif prob > 0.6:
            color = '#f39c12'  # Moderate favorite
            width = 1.5
        elif prob > 0.4:
            color = '#2ecc71'  # Toss-up
            width = 1.0
        elif prob > 0.3:
            color = '#3498db'  # Moderate underdog
            width = 1.5
        else:
            color = '#9b59b6'  # Strong underdog
            width = 2.0
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.7, edge_color=color)
        
        # Add probability labels to edges
        edge_labels = {(u, v): f"{prob:.2f}"}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    # Turn off axis
    plt.axis('off')
    
    # Add title
    gender_label = "Men's" if bracket_data['gender'] == 'M' else "Women's"
    plt.title(f"{bracket_data['year']} {gender_label} NCAA Tournament Predictions", fontsize=16)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#e74c3c', lw=2, label='Strong Favorite (>70%)'),
        Line2D([0], [0], color='#f39c12', lw=1.5, label='Moderate Favorite (60-70%)'),
        Line2D([0], [0], color='#2ecc71', lw=1, label='Toss-up (40-60%)'),
        Line2D([0], [0], color='#3498db', lw=1.5, label='Moderate Underdog (30-40%)'),
        Line2D([0], [0], color='#9b59b6', lw=2, label='Strong Underdog (<30%)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"tournament_predictions_{bracket_data['year']}_{bracket_data['gender']}.png"), 
                    dpi=300, bbox_inches='tight')
    
    return fig, ax


def create_team_stats_radar_chart(team_id, team_stats, teams_data, season=None, output_dir=None):
    """
    Create a radar chart of team statistics
    
    Parameters:
    -----------
    team_id : int
        ID of the team to visualize
    team_stats : dict
        Dictionary containing team statistics
    teams_data : pandas.DataFrame
        DataFrame containing team information
    season : int, optional
        Season to visualize (defaults to latest season)
    output_dir : str, optional
        Directory to save the plot
    
    Returns:
    --------
    tuple
        Tuple containing figure and axes objects
    """
    # Get team info
    team_info = teams_data[teams_data['TeamID'] == team_id].iloc[0]
    team_name = team_info['TeamName']
    gender = team_info['Gender']
    
    # If season not provided, use latest
    if season is None:
        seasons = sorted(team_stats.keys())
        season = seasons[-1] if seasons else None
    
    if not season:
        return None, None
    
    # Get team stats for the specified season
    season_stats = team_stats[season]
    
    if team_id not in season_stats:
        return None, None
    
    stats = season_stats[team_id]
    
    # Get all teams of the same gender
    gender_teams = teams_data[teams_data['Gender'] == gender]['TeamID'].tolist()
    
    # Calculate percentiles for each stat
    categories = [
        'WinPct',
        'PointsPerGame',
        'PointDifferentialPerGame',
        'OffensiveEfficiency',
        'DefensiveEfficiency',
        'NeutralWinPct',
        'StrengthOfSchedule'
    ]
    
    category_names = [
        'Win %',
        'Points/Game',
        'Point Diff',
        'Off. Efficiency',
        'Def. Efficiency',
        'Neutral Win %',
        'SOS'
    ]
    
    # Calculate percentiles for each stat
    percentiles = {}
    all_values = {}
    
    for cat in categories:
        all_values[cat] = []
        for t_id in gender_teams:
            if t_id in season_stats:
                value = season_stats[t_id].get(cat, 0)
                all_values[cat].append(value)
        
        # Get team's value and calculate percentile
        team_value = stats.get(cat, 0)
        if all_values[cat]:
            # Special case for DefensiveEfficiency (lower is better)
            if cat == 'DefensiveEfficiency':
                percentile = 1.0 - sum(v <= team_value for v in all_values[cat]) / len(all_values[cat])
            else:
                percentile = sum(v <= team_value for v in all_values[cat]) / len(all_values[cat])
            
            percentiles[cat] = percentile
        else:
            percentiles[cat] = 0.5
    
    # Set up the radar chart
    n_cats = len(categories)
    angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Add values
    values = [percentiles[cat] for cat in categories]
    values += values[:1]  # Close the polygon
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw polygon
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=team_name)
    ax.fill(angles, values, alpha=0.25)
    
    # Draw category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(category_names)
    ax.set_yticklabels([])
    
    # Draw percentile circles
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.set_rlim(0, 1)
    
    # Add gridlines
    ax.grid(True)
    
    # Add title
    gender_label = "Men's" if gender == 'M' else "Women's"
    ax.set_title(f"{team_name} Statistical Profile ({season} {gender_label})", fontsize=15, y=1.08)
    
    # Add explanation of percentiles
    plt.figtext(
        0.5, 0.01,
        f"Each axis represents the percentile rank of {team_name} compared to other {gender_label.lower()} teams.",
        ha='center',
        fontsize=12,
        wrap=True
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"team_radar_{team_id}_{season}.png"), dpi=300, bbox_inches='tight')
    
    return fig, ax


if __name__ == "__main__":
    # Set up plotting style
    setup_plotting_style()
    
    # Example usage (commented out)
    # These functions would typically be imported and used in another script
    pass 
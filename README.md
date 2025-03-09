# 🏀 NCAA March Madness Prediction System

> **Author:** Shivaansh Dhingra  
> **Repository:** [https://github.com/shivaansh74/march_madness_ml](https://github.com/shivaansh74/march_madness_ml)

## Project Overview
This project implements a comprehensive machine learning system for predicting NCAA Basketball Tournament (March Madness) game outcomes. The system processes historical basketball data, engineers relevant features, trains multiple models using ensemble techniques, and generates predictions for all potential tournament matchups for both men's and women's basketball.

**This project was created for the [March Machine Learning Mania 2025 Kaggle Competition](http://kaggle.com/competitions/march-machine-learning-mania-2025/overview).**

The predictions follow the required format for the Kaggle "March Machine Learning Mania" competition, with separate files for men's and women's tournaments, containing matchup ID and win probability for the team with the lower ID number.

## Table of Contents
- [Project Structure](#project-structure)
- [Data](#data)
- [Code Organization](#code-organization)
- [Methodology](#methodology)
- [Features](#features)
- [Models](#models)
- [Performance Optimization](#performance-optimization)
- [Results](#results)
- [Skills Demonstrated](#skills-demonstrated)
- [Installation & Usage](#installation--usage)
- [Future Improvements](#future-improvements)

## Project Structure

```
march_madness/
├── data/
│   └── raw/             # Raw data files
├── models/              # Saved trained models (models.pkl not included due to GitHub size limits)
├── submissions/         # Generated prediction files
├── main.py              # Main entry point for the prediction system
├── data_loader.py       # Data loading utilities
├── data_preparation.py  # Data cleaning and preparation
├── feature_engineering.py # Feature creation and engineering
├── model_training.py    # Model training and hyperparameter tuning
├── evaluation.py        # Model evaluation and performance metrics
├── pipeline.py          # End-to-end ML pipeline
├── visualization.py     # Data and results visualization
├── experiments.py       # Experimental model configurations
├── model.py             # Model class definitions
├── march_madness_analysis.ipynb # Jupyter notebook for analysis
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

**Note:** The trained model file (models.pkl) is not included in the repository due to GitHub file size limitations.

## Data
The system uses historical NCAA basketball data including:

| Data Type | Description | Files |
|-----------|-------------|-------|
| Regular Season Results | Outcomes of regular season games | MRegularSeasonCompactResults.csv, WRegularSeasonCompactResults.csv |
| Tournament Results | Historical NCAA tournament game results | MNCAATourneyCompactResults.csv, WNCAATourneyCompactResults.csv |
| Team Information | Details about each team | MTeams.csv, WTeams.csv |
| Tournament Seeds | Tournament seeding information | MNCAATourneySeeds.csv, WNCAATourneySeeds.csv |
| Conference Data | Team conference affiliations | MTeamConferences.csv, WTeamConferences.csv |
| Detailed Game Stats | Box score statistics | MRegularSeasonDetailedResults.csv, WRegularSeasonDetailedResults.csv |
| Rankings | Massey ranking systems | MMasseyOrdinals.csv, WMasseyOrdinals.csv |

The data preparation process reads raw data from the `data/raw` folder and performs initial cleaning and formatting to prepare it for feature engineering and model training.

## Code Organization

| File | Purpose | Key Functionality |
|------|---------|-------------------|
| `main.py` | Main entry point | Orchestrates the entire prediction workflow |
| `data_loader.py` | Data loading | Loads NCAA basketball data from the raw folder |
| `data_preparation.py` | Data cleaning | Cleans, transforms, and prepares data for analysis |
| `feature_engineering.py` | Feature creation | Calculates advanced team and matchup statistics |
| `model_training.py` | Model training | Trains and tunes various ML models |
| `evaluation.py` | Performance evaluation | Evaluates model performance with metrics and visualizations |
| `pipeline.py` | ML pipeline | Connects data processing, training, and prediction |
| `visualization.py` | Data visualization | Creates visualizations of data and model performance |
| `experiments.py` | Experimentation | Configures and runs model experiments |
| `model.py` | Model definitions | Defines custom model classes and ensembles |

## Methodology
The prediction system follows these steps:

1. **Data Loading**: Imports raw data from CSV files stored in `data/raw`
2. **Feature Engineering**: Calculates team statistics and creates matchup features
3. **Model Training**: Trains multiple models using both men's and women's historical data
4. **Ensemble Creation**: Combines models into an ensemble for more robust predictions
5. **Prediction Generation**: Generates predictions for all possible tournament matchups
6. **Calibration**: Applies temperature scaling to spread out prediction probabilities
7. **Submission**: Creates CSV files in the required format

## Features
The system engineers sophisticated features to capture team performance:

| Feature Category | Examples |
|------------------|----------|
| Basic Performance | Win percentage, Points per game, Points allowed |
| Advanced Metrics | Offensive/Defensive efficiency, Tempo, Effective FG% |
| Matchup Differentials | Seed difference, Scoring differential |
| Home/Away Performance | Home win %, Away win %, Neutral site win % |
| Consistency | Point differential standard deviation |
| Tournament History | Historical seed performance |
| Momentum | Recent game performance metrics |
| Conference Strength | Conference win rates, SOS by conference |

## Models
The system trains multiple models and combines them into an ensemble:

| Model | Implementation | Properties |
|-------|---------------|------------|
| Logistic Regression | sklearn | Linear model with L2 regularization |
| Random Forest | sklearn | Ensemble of 300 decision trees |
| Gradient Boosting | sklearn | Sequential tree building |
| XGBoost | xgboost | Optimized gradient boosting |
| LightGBM | lightgbm | High-performance gradient boosting |
| CatBoost | catboost | Handles categorical features automatically |

For each algorithm, the system trains:
- One model on all data
- One model specialized for men's basketball
- One model specialized for women's basketball

## Performance Optimization
The system incorporates several optimizations for handling the large number of potential matchups:

| Optimization | Implementation | Benefit |
|--------------|----------------|---------|
| Parallel Processing | joblib | Utilizes multiple CPU cores for predictions |
| Vectorized Operations | numpy | Processes predictions in batches |
| Team Feature Caching | Precomputation | Avoids redundant calculations |
| Memory Management | Chunking | Processes data in memory-efficient chunks |
| Progress Tracking | tqdm | Provides visibility into processing status |
| Database Integration | sql | Enables scalable data processing |

## Results
The system generates two prediction files:
- `submissions/submission_2025_M.csv`: Men's tournament predictions
- `submissions/submission_2025_W.csv`: Women's tournament predictions

Each file contains matchup IDs and corresponding win probabilities:

| ID | Pred |
|----|------|
| 2025_1101_1152 | 0.5158 |
| 2025_1101_1153 | 0.4040 |
| 2025_1101_1154 | 0.4764 |
| ... | ... |

Predictions are calibrated using temperature scaling to ensure a realistic distribution of probabilities that reflects the variability in team strengths.

## Skills Demonstrated
This project showcases numerous technical skills:

| Skill Category | Specific Skills |
|----------------|-----------------|
| Programming | Python, Object-oriented design, Parallel processing |
| Data Science | Feature engineering, Model evaluation, Ensemble methods |
| Machine Learning | Classification algorithms, Model calibration, Cross-validation |
| Performance | Vectorization, Memory optimization, Scalable processing |
| Software Engineering | Modular code organization, Error handling, Progress monitoring |
| Statistics | Probability calibration, Bayesian methods, Statistical feature generation |
| Domain Knowledge | Basketball analytics, Tournament prediction, Sports modeling |

## Installation & Usage

### Requirements
- Python 3.9+
- Required packages listed in `requirements.txt`

### Setup
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Place raw data files in the `data/raw` folder

**Important Note:** The trained model file (models.pkl) is not included in the GitHub repository due to size limitations. You will need to train your own models by running the system, or contact the author for access to pre-trained models.

### Running the System
To execute the full prediction pipeline:
```bash
python main.py
```

### Customization
Adjust parameters in `main.py` to:
- Change the number of parallel jobs (`n_jobs` parameter)
- Modify the prediction year (default: 2025)
- Tune the calibration temperature (in `calibrate_probabilities` function)

## Future Improvements
Potential enhancements for the system include:

| Improvement | Benefit |
|-------------|---------|
| Deep learning models | Capture complex non-linear relationships |
| Play-by-play features | More granular feature engineering |
| Bayesian methods | Improved uncertainty quantification |
| Interactive dashboard | Better visualization of predictions |
| Cloud deployment | Scale to larger datasets |
| Automated pipeline | CI/CD for model retraining |

This project demonstrates a comprehensive understanding of machine learning, software engineering, and basketball analytics, combining theory and practice to create an end-to-end prediction system.

## Contact

For questions or feedback about this project, please contact:

- **Name:** Shivaansh Dhingra
- **Email:** dhingrashivaansh@gmail.com
- **GitHub:** [https://github.com/shivaansh74/march_madness_ml](https://github.com/shivaansh74/march_madness_ml)
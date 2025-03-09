## March Machine Learning Mania 2025 - Project Overview

This document provides a comprehensive project overview for the March Machine Learning Mania 2025 Kaggle competition. It outlines the competition's objectives, evaluation metrics, submission format, timeline, prizes, and available datasets. This overview is designed to provide a thorough understanding of the project for participants of all levels, from beginners to experienced Kagglers.

**1. Introduction**

The March Machine Learning Mania competition is an annual event hosted by Kaggle, challenging participants to predict the outcomes of the NCAA Division I Men's and Women's basketball tournaments, also known as March Madness.  For its eleventh iteration in 2025, Kagglers will leverage historical basketball data and machine learning techniques to forecast game results. Unlike casual fans who rely on intuition, participants in this competition will develop data-driven models to predict upsets, analyze probabilities, and ultimately climb the leaderboard by demonstrating superior bracketology skills.  This year's competition combines both Men's and Women's tournaments into a single challenge and requires predictions for all possible matchups, offering a longer prediction window and a more comprehensive forecasting task.

**2. Project Goal**

The primary objective of this project is to build a predictive model that accurately forecasts the outcomes of NCAA Division I Men's and Women's basketball tournament games in 2025.  Specifically, participants are tasked with predicting the probability that in any given hypothetical matchup, the team with the lower Team ID will defeat the team with the higher Team ID. The goal is to minimize prediction error and achieve the highest possible ranking on the competition leaderboard by the end of the tournaments.

**3. Evaluation Metric: Brier Score**

Submissions will be evaluated using the **Brier Score**.  The Brier Score measures the accuracy of probabilistic predictions. In this context, it quantifies the difference between the predicted probability of a team winning and the actual outcome (win or loss).  A lower Brier Score indicates better prediction accuracy.  Mathematically, for each game:

Brier Score = (Probability Predicted - Actual Outcome)^2

Where:

* **Probability Predicted** is the submitted probability that the team with the lower TeamID wins.
* **Actual Outcome** is 1 if the team with the lower TeamID wins, and 0 if the team with the higher TeamID wins.

The overall competition score will be the mean Brier Score across all predicted games in the 2025 tournaments.  Effectively, minimizing the Brier Score is equivalent to minimizing the Mean Squared Error (MSE) in this binary classification problem.

**4. Submission Format**

The submission format has been revised for 2025 to accommodate predictions for all possible matchups and to include both Men's and Women's tournaments in a single file.  Participants are required to submit a CSV file with the following structure:

* **ID**: A unique identifier for each hypothetical matchup, formatted as `Season_TeamID1_TeamID2`.
    * `Season`:  The year 2025.
    * `TeamID1`: The TeamID of the team with the lower ID.
    * `TeamID2`: The TeamID of the team with the higher ID.
    * *Example*: `2025_1101_1102` represents a hypothetical matchup between team 1101 and 1102 in the 2025 season.
* **Pred**: The predicted probability (between 0 and 1) that `TeamID1` (the team with the lower TeamID) will win against `TeamID2`.

**Example Submission File (SampleSubmission.csv):**

```csv
ID,Pred
2025_1101_1102,0.5
2025_1101_1103,0.5
2025_1101_1104,0.5
...
```

**Important Submission Notes:**

* **Combined Men's and Women's Tournaments:**  A single submission file must include predictions for both Men's and Women's hypothetical matchups.  Team IDs for Men's and Women's teams are distinct and do not overlap.
* **All Possible Matchups:** Predictions are required for *every* possible matchup between teams, not just those selected for the actual NCAA tournament. This allows for submissions to be made well in advance of team selections.
* **Probability for Lower TeamID Winning:**  Always predict the probability that the team with the *lower* TeamID wins against the team with the *higher* TeamID.
* **Zero Score Initial Leaderboard:**  Submissions in the correct format will initially score 0.0 on the leaderboard.  The leaderboard will become meaningful only after the 2025 tournaments commence and Kaggle rescores submissions with actual game outcomes.
* **Submission Selection:** Participants can submit multiple times but must manually select two submissions to be counted for final scoring. Automatic selection is not reliable.

**5. Timeline**

* **February 10, 2025 - Start Date:** Competition officially begins, and datasets are released.
* **Week of February 18-21, 2025 - 2025 Tournament Submission File Available:**  The specific submission file format for the 2025 tournament will be made available.
* **March 20, 2025 4PM UTC - Final Submission Deadline:**  The deadline for submitting predictions. Kaggle will release updated data before the deadline to include as much current season data as possible.
* **March 20 - April 8 - Tournament Play and Leaderboard Updates:**  The NCAA tournaments take place. Kaggle will periodically update the leaderboard as game results become available.

**Important Timeline Notes:**

* **Data Updates:** Kaggle will provide data updates closer to the submission deadline, incorporating recent game results and potentially updated rankings. Participants should be prepared to retrain and resubmit their models with the latest data.
* **Leaderboard Refresh:** The leaderboard will be dynamically updated throughout the tournament period as game outcomes are finalized.  This allows participants to track their performance in real-time.

**6. Prizes**

The competition offers substantial monetary prizes for top-performing participants, incentivizing accurate and robust prediction models:

* **1st Place:** $10,000
* **2nd Place:** $8,000
* **3rd Place:** $7,000
* **4th - 8th Place(s):** $5,000 each

**7. Datasets**

A comprehensive set of historical NCAA basketball data is provided, spanning multiple seasons and encompassing both Men's (M prefix) and Women's (W prefix) tournaments.  Some datasets are common to both (no prefix).  The data is organized into five sections, each focusing on different aspects of the game and tournament:

**Data Section 1: The Basics** - Essential for building a fundamental prediction model.

* **MTeams.csv & WTeams.csv:**
    * **Description:**  Identifies college teams in the dataset.
    * **Key Columns:** `TeamID`, `TeamName`, `FirstD1Season` (Men's only), `LastD1Season` (Men's only).
    * **Relevance:**  Fundamental for identifying teams and linking data across files. `TeamID` is the primary key for joining datasets.

* **MSeasons.csv & WSeasons.csv:**
    * **Description:**  Information about each season.
    * **Key Columns:** `Season`, `DayZero`, `RegionW`, `RegionX`, `RegionY`, `RegionZ`.
    * **Relevance:**  Provides context for each season, including the start date (`DayZero`) and region names used in tournament seeding. `Season` is crucial for filtering data by year.

* **MNCAATourneySeeds.csv & WNCAATourneySeeds.csv:**
    * **Description:**  Tournament seeds for each team in each season.
    * **Key Columns:** `Season`, `Seed`, `TeamID`.
    * **Relevance:**  Seeds are a strong indicator of team strength and are vital for understanding tournament structure and potential matchups.

* **MRegularSeasonCompactResults.csv & WRegularSeasonCompactResults.csv:**
    * **Description:**  Compact game results for regular season games.
    * **Key Columns:** `Season`, `DayNum`, `WTeamID` (Winning Team ID), `WScore`, `LTeamID` (Losing Team ID), `LScore`, `WLoc` (Winning Team Location), `NumOT` (Number of Overtime periods).
    * **Relevance:**  Provides historical game outcomes, essential for training models to predict win probabilities based on team matchups and game details. "Compact" results contain basic game information.

* **MNCAATourneyCompactResults.csv & WNCAATourneyCompactResults.csv:**
    * **Description:**  Compact game results for NCAA Tournament games.
    * **Key Columns:**  Same structure as regular season results but specifically for tournament games.
    * **Relevance:**  Tournament games may exhibit different patterns compared to regular season games. This data is crucial for understanding tournament-specific dynamics.

* **SampleSubmissionStage1.csv:**
    * **Description:**  Example submission file demonstrating the required format.
    * **Key Columns:** `ID`, `Pred`.
    * **Relevance:**  Illustrates the submission structure and serves as a starting point for creating submission files.

**Data Section 2: Team Box Scores** - Detailed game statistics at the team level, available from 2003 (Men) and 2010 (Women).

* **MRegularSeasonDetailedResults.csv & WRegularSeasonDetailedResults.csv:**
    * **Description:**  Detailed box scores for regular season games.
    * **Key Columns:**  Includes all columns from Compact Results plus detailed team stats like `FGM`, `FGA`, `FGM3`, `FGA3`, `FTM`, `FTA`, `OR`, `DR`, `Ast`, `TO`, `Stl`, `Blk`, `PF` for both winning (W prefix) and losing (L prefix) teams.
    * **Relevance:**  Provides richer game-level features for more sophisticated models. Allows for analysis of team performance metrics beyond just wins and losses. "Detailed" results contain comprehensive game statistics.

* **MNCAATourneyDetailedResults.csv & WNCAATourneyDetailedResults.csv:**
    * **Description:**  Detailed box scores for NCAA Tournament games.
    * **Key Columns:**  Same structure as detailed regular season results, but for tournament games.
    * **Relevance:**  Detailed statistics specifically for tournament games, potentially revealing different performance characteristics in high-stakes scenarios.

**Data Section 3: Geography** - Location data for games.

* **Cities.csv:**
    * **Description:** Master list of cities where games have been played.
    * **Key Columns:** `CityID`, `City`, `State`.
    * **Relevance:** Provides city information for potential geographic analysis or feature engineering (e.g., travel distance). City IDs are consistent across years.

* **MGameCities.csv & WGameCities.csv:**
    * **Description:**  Links games to the cities where they were played.
    * **Key Columns:** `Season`, `DayNum`, `WTeamID`, `LTeamID`, `CRType` (Game Result Type: Regular, NCAA, Secondary), `CityID`.
    * **Relevance:**  Connects game records to city locations, enabling the use of geographic information in models.

**Data Section 4: Public Rankings** - Weekly team rankings for Men's teams from various ranking systems (since 2003).

* **MMasseyOrdinals.csv:**
    * **Description:**  Weekly ordinal rankings from various systems like Pomeroy, Sagarin, RPI, ESPN, etc. for Men's teams.
    * **Key Columns:** `Season`, `RankingDayNum`, `SystemName`, `TeamID`, `OrdinalRank`.
    * **Relevance:**  Provides external assessments of team strength from established ranking systems. Can be used as features or to validate model predictions. Note: Only for Men's teams.

**Data Section 5: Supplements** - Additional supporting information.

* **MTeamCoaches.csv:**
    * **Description:**  Head coach for each Men's team per season, including coaching change dates.
    * **Key Columns:** `Season`, `TeamID`, `FirstDayNum`, `LastDayNum`, `CoachName`.
    * **Relevance:**  Coach information might be relevant, as coaching strategy can influence team performance. Only for Men's teams.

* **Conferences.csv:**
    * **Description:**  List of Division I conferences.
    * **Key Columns:** `ConfAbbrev`, `Description`.
    * **Relevance:**  Provides conference abbreviations and full names for reference.

* **MTeamConferences.csv & WTeamConferences.csv:**
    * **Description:**  Conference affiliation for each team in each season.
    * **Key Columns:** `Season`, `TeamID`, `ConfAbbrev`.
    * **Relevance:**  Conference strength could be a factor in team performance and prediction accuracy.

* **MConferenceTourneyGames.csv & WConferenceTourneyGames.csv:**
    * **Description:**  Identifies games that were part of conference tournaments.
    * **Key Columns:** `ConfAbbrev`, `Season`, `DayNum`, `WTeamID`, `LTeamID`.
    * **Relevance:**  Conference tournament games may have unique characteristics and can be used to analyze team performance in pre-NCAA tournament scenarios.

* **MSecondaryTourneyTeams.csv & WSecondaryTourneyTeams.csv:**
    * **Description:** Teams participating in secondary post-season tournaments (NIT, etc.).
    * **Key Columns:** `Season`, `SecondaryTourney`, `TeamID`.
    * **Relevance:**  Provides information on teams that were good but didn't make the NCAA tournament, potentially useful for understanding overall team strength distribution.

* **MSecondaryTourneyCompactResults.csv & WSecondaryTourneyCompactResults.csv:**
    * **Description:** Results for secondary post-season tournament games.
    * **Key Columns:** `SecondaryTourney`, `Season`, `DayNum`, `WTeamID`, `LTeamID`, etc. (similar to Compact Results).
    * **Relevance:**  Results from these tournaments might offer additional data points, though less directly related to the NCAA tournament prediction task.

* **MTeamSpellings.csv & WTeamSpellings.csv:**
    * **Description:**  Alternative spellings of team names.
    * **Key Columns:** `TeamNameSpelling`, `TeamID`.
    * **Relevance:**  Useful for mapping external data sources to the competition's TeamIDs, handling variations in team name spellings.

* **MNCAATourneySlots.csv & WNCAATourneySlots.csv:**
    * **Description:** Tournament bracket structure and seeding logic.
    * **Key Columns:** `Season`, `Slot`, `StrongSeed`, `WeakSeed`.
    * **Relevance:**  Understanding how teams are paired in the tournament based on seeds is essential for tournament simulation and bracket generation.

* **MNCAATourneySeedRoundSlots.csv:**
    * **Description:**  Men's tournament bracket structure, mapping seeds to slots and rounds (Men's data only).
    * **Key Columns:** `Seed`, `GameRound`, `GameSlot`, `EarlyDayNum`, `LateDayNum`.
    * **Relevance:**  Provides a structured view of the Men's tournament bracket, helpful for round-by-round analysis and simulation.

**8. Potential Approaches**

Participants can explore a wide range of machine learning techniques and modeling strategies to predict game outcomes. Some potential approaches include:

* **Statistical Models:** Logistic Regression, Elo ratings, Massey ratings, team statistics-based models.
* **Machine Learning Models:** Random Forests, Gradient Boosting Machines (GBM), Neural Networks, Support Vector Machines (SVM).
* **Feature Engineering:**  Creating relevant features from the provided datasets. Examples include:
    * Team performance metrics (win rate, scoring averages, defensive stats, etc.)
    * Team strengths and weaknesses derived from detailed results.
    * Team rankings (Massey Ordinals).
    * Seed differences (for tournament games).
    * Conference strength.
    * Coaching information.
    * Historical matchup performance.
    * Geographic factors (travel distance, home court advantage - though less relevant in the neutral-site tournament).
* **Model Ensembling:** Combining predictions from multiple models to improve robustness and accuracy.
* **Time Series Analysis:**  Analyzing team performance trends over the season.
* **Deep Learning:** Exploring recurrent neural networks (RNNs) or transformers to capture sequential game data.

**9. Conclusion**

The March Machine Learning Mania 2025 competition presents a challenging and engaging opportunity to apply data science and machine learning to the exciting world of college basketball.  By leveraging the comprehensive historical datasets provided and developing innovative prediction models, participants can compete for prizes, recognition, and the coveted title of March Machine Learning Mania champion. This detailed project overview should serve as a strong foundation for embarking on this exciting predictive modeling journey. Good luck, and happy forecasting!``
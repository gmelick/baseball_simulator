# MLB Game Simulator

   A pitch-by-pitch Monte Carlo simulation engine for Major League Baseball games. The simulator uses historical play data from the MLB Stats API, player similarity scoring, and a trained Keras neural network to realistically
    model game outcomes including pitching decisions.

   ---

   ## Table of Contents

   - [Overview](#overview)
   - [How It Works](#how-it-works)
   - [Project Structure](#project-structure)
   - [Pipeline Walkthrough](#pipeline-walkthrough)
     - [Step 1: Data Collection](#step-1-data-collection)
     - [Step 2: Pitcher Hook Data](#step-2-pitcher-hook-data)
     - [Step 3: Similarity Scores](#step-3-similarity-scores)
     - [Step 4: Leash Model Training](#step-4-leash-model-training)
     - [Step 5: Game Simulation](#step-5-game-simulation)
   - [Similarity Methodology](#similarity-methodology)
     - [Batter Similarity](#batter-similarity)
     - [Pitcher Similarity](#pitcher-similarity)
   - [Leash Model](#leash-model)
   - [Simulation Engine](#simulation-engine)
   - [Output Format](#output-format)
   - [Dependencies](#dependencies)
   - [Usage](#usage)

   ---

   ## Overview

   Rather than relying on per-matchup statistics (which are often too sparse to be reliable), this simulator answers the question: *"Given the current pitcher and batter, which historical plate appearances most closely
   resemble this matchup?"*

   It then samples from those similar historical plays -- weighted by similarity -- to simulate each pitch outcome. Pitcher substitutions are governed by a 1D Convolutional Neural Network trained to predict the probability a
   manager pulls a pitcher after any given at-bat.

   Each game is simulated 100 times in parallel, producing a distribution of outcomes including run scoring by inning and detailed pitcher/batter stat lines.

   ---

   ## How It Works

   ```
   MLB Stats API
         |
         v
   Pitch-by-Pitch CSVs  -->  Pitcher Hook CSVs
         |                         |
         v                         v
   Similarity Scores         Leash Model (Keras 1D CNN)
         |                         |
         +----------+--------------+
                    |
                    v
          Game Simulation (100x Monte Carlo)
                    |
                    v
          Output CSVs (Linescore + Box Score)
   ```

   ---

   ## Project Structure

   ```
   baseball_simulator/
   |-- driver.py                          # Entry point; orchestrates the full pipeline
   |-- requirements.txt
   |
   |-- data/
   |   |-- create_season_play_file.py     # Fetches pitch-by-pitch data from MLB Stats API
   |   +-- create_leash_model_data.py     # Extracts per-at-bat pitcher stat lines for hook model
   |
   |-- similarities/
   |   |-- batter_similarity.py           # Computes batter-to-batter similarity matrices
   |   +-- pitcher_similarity.py          # Computes pitcher-to-pitcher similarity matrices
   |
   |-- models/
   |   +-- leash_model.py                 # Trains the pitcher removal prediction model
   |
   +-- simulation/
       |-- simulate_day.py                # Fetches lineups, sets up, and runs game simulations
       +-- game.py                        # Core Game class and multiprocessing simulation logic
   ```

   **Generated data files** (not committed, created at runtime):

   ```
   {YYYY}.csv                   # Pitch-by-pitch data for each season
   {YYYY} Pitcher Hooks.csv     # Per-at-bat pitcher stat lines
   RHP Similarities.csv         # Right-handed pitcher similarity scores
   LHP Similarities.csv         # Left-handed pitcher similarity scores
   RHB Similarities.csv         # Right-handed batter similarity scores
   LHB Similarities.csv         # Left-handed batter similarity scores
   best_model_0.keras            # Trained starter leash model
   best_model_1.keras            # Trained reliever leash model
   {YYYY}/{YYYY_MM_DD}/{YYYY_MM_DD}_{game_pk}.csv  # Simulation output per game
   ```

   ---

   ## Pipeline Walkthrough

   ### Step 1: Data Collection

   **`data/create_season_play_file.py`**

   Fetches granular pitch-by-pitch data from the MLB Stats API (`statsapi.mlb.com`) for every regular season and postseason game back to 2017. Each row in the output CSV represents a single pitch and captures ~100 fields
   including:

   - **Game context:** `game_pk`, `game_date`, `venue`, `home_team`, `away_team`, manager IDs
   - **Pitch mechanics:** `release_speed`, `spin_rate`, `break_vertical`, `pfx_x/z`, `release_pos_x/y/z`, velocity and acceleration vectors
   - **Pitch outcome:** `pitch_type`, `pitch_code`, `balls`, `strikes`, `outs`, `events`
   - **Hit data:** `launch_speed`, `launch_angle`, `spray_angle`, `hit_distance`, `bb_type`
   - **Base state:** runners on base before and after each pitch, stolen base attempts, wild pitches
   - **Fielding:** putouts, assists, errors, defensive positions

   `refresh_plays()` rebuilds all historical data from scratch. `get_plays(start, end)` incrementally adds new games for a date range, skipping any game already present.

   ---

   ### Step 2: Pitcher Hook Data

   **`data/create_leash_model_data.py`**

   Processes the pitch-by-pitch CSVs to build a training dataset for the pitcher removal model. For each at-bat in every game it records the pitcher's **cumulative in-game stat line** at that moment:

   | Column | Description |
   |---|---|
   | `Batters Faced` | Cumulative batters faced |
   | `Outs` | Cumulative outs recorded |
   | `Hits` | Cumulative hits allowed |
   | `Runs` | Cumulative runs allowed |
   | `Strikeouts` | Cumulative strikeouts |
   | `Walks` | Cumulative walks |
   | `Hit By Pitch` | Cumulative HBP |
   | `Home Runs` | Cumulative home runs |
   | `Pitches` | Cumulative pitches thrown |
   | `Strikes` | Cumulative strikes thrown |
   | `Earned Runs` | Cumulative earned runs |
   | `IsInningEnd` | 1 if this at-bat ended the inning |
   | `Starter` | 1 if this pitcher started the game |
   | `Pulled` | **Target variable** -- 1 if the pitcher was removed after this at-bat |

   This produces `{YYYY} Pitcher Hooks.csv` files used to train the leash model.

   ---

   ### Step 3: Similarity Scores

   Similarities are computed over the **trailing 4 seasons** relative to the simulation date, giving the model recent player context without being limited to a single year's sample.

   **Output files:**
   - `RHP Similarities.csv` / `LHP Similarities.csv` -- pitcher pairs with similarity score
   - `RHB Similarities.csv` / `LHB Similarities.csv` -- batter pairs with similarity score

   Each file has columns: `Year_1, ID_1, Year_2, ID_2, Similarity`

   See [Similarity Methodology](#similarity-methodology) for full details.

   ---

   ### Step 4: Leash Model Training

   **`models/leash_model.py`**

   Trains two separate Keras models on the pitcher hook data -- one for starters and one for relievers -- using all data strictly prior to the simulation date to avoid leakage.

   See [Leash Model](#leash-model) for architecture details.

   ---

   ### Step 5: Game Simulation

   **`simulation/simulate_day.py`** + **`simulation/game.py`**

   Fetches the day's starting lineups from `mlb.com`, sets up each game with pre-computed similarity weights, and runs 100 Monte Carlo simulations per game using Python multiprocessing with shared memory.

   See [Simulation Engine](#simulation-engine) for details.

   ---

   ## Similarity Methodology

   All similarities use the **trailing 4 seasons** of pitch data. Players are split by handedness (L/R) and scored independently. Only players with sufficient sample sizes are included (>300 pitches faced for batters, >300
   pitches thrown for pitchers, >50 balls in play).

   Raw rate features are **logit-transformed** (`log(p / (1 - p))`) before distance calculations to linearize statistics that are bounded between 0 and 1.

   Final similarity scores are **min-max normalized** to [0, 1]. During simulation, scores are converted to a `Percentage` column where 1.0 = most similar and 0.0 = least similar.

   ---

   ### Batter Similarity

   **`similarities/batter_similarity.py`**

   Batters are profiled on **25 features**, computed separately for right-handed and left-handed batters:

   **Pitch-level rates** (per total pitches seen):

   | Feature Group | Features |
   |---|---|
   | Batted ball trajectory | `liners_total`, `grounder_total`, `fly_total` |
   | Spray direction | `pull_total`, `center_total`, `oppo_total` |
   | Contact quality | `soft_total`, `medium_total`, `hard_total` |

   **In-play rates** (per ball in play):

   | Feature Group | Features |
   |---|---|
   | Batted ball trajectory | `liners_in_play`, `grounder_in_play`, `fly_in_play` |
   | Spray direction | `pull_in_play`, `center_in_play`, `oppo_in_play` |
   | Contact quality | `soft_in_play`, `medium_in_play`, `hard_in_play` |

   **Plate discipline rates:**
   `swing_rate`, `in_zone_swing_rate`, `chase_rate`, `contact_rate`, `in_zone_contact_rate`, `out_zone_contact_rate`, `strike_rate`

   **Distance metric:** [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) using the empirical covariance matrix of the full batter population. This accounts for natural correlations between features
   (e.g., pull rate and oppo rate are negatively correlated) and normalizes each feature by its population variance.

   ---

   ### Pitcher Similarity

   **`similarities/pitcher_similarity.py`**

   Pitcher similarity is a **50/50 blend** of two independent components:

   #### Component 1: Pitch Arsenal Similarity (50%)

   Compares pitchers based on the physical characteristics of every pitch type in their repertoire, weighted by usage rate.

   - **Pitch features (13 total):** `release_speed`, `end_speed`, `release_pos_x/y/z`, `vx0/vy0/vz0`, `ax/ay/az`, `pfx_x`, `pfx_z`
   - Features are normalized via Yeo-Johnson power transform + StandardScaler
   - Each pitcher is represented as a **probability distribution over pitch types**, with each pitch type characterized by its mean feature vector
   - Two-pitcher distance is computed via [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) (Wasserstein / Optimal Transport), using Mahalanobis distance as the per-pitch-type ground cost

   EMD naturally handles pitchers with different arsenal sizes -- a 4-pitch pitcher versus a 2-pitch pitcher can be compared meaningfully by finding the optimal transport plan between their distributions.

   #### Component 2: Results Similarity (50%)

   The same 25-feature profile used for batter similarity (plate discipline, contact quality, batted ball trajectory/direction) computed from the pitcher's perspective, using Mahalanobis distance.

   ---

   ## Leash Model

   **`models/leash_model.py`** | **`simulation/game.py`**

   A **1D Convolutional Neural Network** that predicts the probability a manager removes a pitcher after any given at-bat, given the pitcher's cumulative in-game performance.

   ### Architecture

   ```
   Input (12, 1)   -- 12 cumulative stat features as a 1D sequence
       |
       +--> Conv1D(64, kernel=3, same) --> BatchNorm --> ReLU
       +--> Conv1D(64, kernel=3, same) --> BatchNorm --> ReLU
       +--> Conv1D(64, kernel=3, same) --> BatchNorm --> ReLU
       |
       +--> GlobalAveragePooling1D
       |
       +--> Dense(1, sigmoid)          -- P(pitcher is pulled)
   ```

   - **Loss:** Binary cross-entropy
   - **Optimizer:** Adam with ReduceLROnPlateau (factor=0.5, patience=20, min=0.0001)
   - **Metric:** Precision at 50% recall
   - **Training:** Up to 20 epochs, batch size 256, 20% validation split, EarlyStopping (patience=50)
   - **Checkpoint:** Best validation-loss model is saved and reloaded after training

   Two separate models are trained:
   - `best_model_0.keras` -- **Starter** removal model
   - `best_model_1.keras` -- **Reliever** removal model

   ### Fast Inference via NumPy

   To avoid TensorFlow/Keras initialization overhead inside the multiprocessing simulation loop, model weights are extracted once into plain NumPy arrays (`extract_model_weights()`). Inference during simulation runs through a
    hand-written **pure NumPy forward pass** (`numpy_forward()` in `game.py`) that reproduces the Conv1D -> BatchNorm -> ReLU -> GAP -> Dense -> Sigmoid pipeline without any framework overhead.

   ---

   ## Simulation Engine

   **`simulation/simulate_day.py`** + **`simulation/game.py`**

   ### Lineup Scraping

   Starting lineups, batting orders, and announced starting pitchers are scraped from `mlb.com/starting-lineups/{date}` using BeautifulSoup. Bench players and full bullpen rosters are fetched from the MLB Stats API live game
   feed. Games where lineups have not yet been posted are retried in a loop.

   ### Pre-computation: Similarity Weighting

   Before any simulation runs, every historical pitch in the plays dataset is assigned per-game-player similarity weights:

   - `P_{pitcher_id}` -- similarity score for each pitcher in today's game vs. every historical pitcher
   - `B_R_{batter_id}` / `B_L_{batter_id}` -- similarity score for each batter vs. every historical batter (separate columns for vs. RHP and vs. LHP)

   Players without sufficient historical data default to a similarity weight of 0.5. The weighted plays DataFrame is saved to a `.feather` file for fast binary I/O before multiprocessing begins.

   ### Situation Index

   Historical plays are bucketed by **game state** using a precomputed integer-keyed dictionary. The state encodes balls, strikes, outs, and base occupancy into a single integer:

   ```
   code = strikes*(3*4*2*2*2) + outs*(4*2*2*2) + balls*(2*2*2) + on1*(2*2) + on2*2 + on3
   ```

   Each key maps to an array of row indices, enabling O(1) retrieval of all matching historical plays for any given count and base configuration.

   ### Pitch Simulation

   For each pitch in a simulated at-bat:

   1. Encode the current game state into the situation code
   2. Retrieve all historical plays matching that code from the situation index
   3. Compute a combined weight for each candidate: `P_{pitcher} x B_{hand}_{batter}`
   4. Filter to plays with weight >= mean weight (discards poor matches)
   5. Sample one play proportionally to the filtered, normalized weights
   6. Apply the outcome: update ball/strike count, outs, runs, baserunner positions, and all stat accumulators

   Baserunner advancement is resolved by matching post-play runner IDs from the sampled historical play to the actual runners currently on base.

   ### Pitcher Hook Decision

   After every at-bat (minimum 3 batters faced, or whenever an inning ends), the appropriate leash model (starter or reliever) is evaluated on the pitcher's current 12-feature cumulative stat vector. The model outputs a
   probability `p`; the pitcher is removed with probability `p`, making hook decisions stochastic rather than deterministic.

   Bullpen replacements are sampled from the team's available relievers, weighted by their historical appearance frequency at the current inning and run-differential context. Once a reliever enters the game he is removed from
    the available pool.

   ### Multiprocessing Architecture

   Each game runs 100 independent simulations distributed across a `multiprocessing.Pool`:

   - **Shared memory:** The plays DataFrame is written into a `shared_memory.SharedMemory` block once and mapped read-only by all worker processes, avoiding repeated serialization
   - **On-disk config:** Game setup (rosters, lineups, model weights) is serialized via `joblib` to `sim_cfg.pkl`; the situation index to `situation_index.pkl`; each worker loads these once on initialization via
   `_init_worker()`
   - **NumPy-only inference:** Model weights arrive in workers as pre-extracted NumPy arrays, enabling the pure-NumPy forward pass without loading Keras per worker

   ---

   ## Output Format

   Results are written to `{YYYY}/{YYYY_MM_DD}/{YYYY_MM_DD}_{game_pk}.csv`. Each of the 100 rows represents one simulation and contains:

   | Field group | Contents |
   |---|---|
   | `T1`--`B9`, `T10+`, `B10+` | Runs per half-inning (top/bottom); extra innings collapsed into one slot |
   | `Home Score`, `Away Score` | Final score |
   | `Home Pitcher 1`--`12` | Pitcher ID + 11 stats: BF, Outs, H, R, K, BB, HBP, HR, Pitches, Strikes, ER |
   | `Away Pitcher 1`--`12` | Same for away pitchers |
   | `Home Batter 1`--`9` | Batter ID + 11 stats: PA, AB, H, 2B, 3B, HR, K, BB, HBP, R, RBI |
   | `Away Batter 1`--`9` | Same for away batters |

   Unused pitcher slots (fewer than 12 pitchers used) are written as empty comma-separated fields.

   ---

   ## Dependencies

   | Package | Purpose |
   |---|---|
   | `numpy` | Array operations, numerical computing |
   | `pandas` | DataFrame processing, CSV/Feather I/O |
   | `scikit-learn` | Yeo-Johnson PowerTransformer, StandardScaler |
   | `scipy` | Mahalanobis distance, covariance matrix inversion |
   | `requests` | MLB Stats API HTTP calls |
   | `beautifulsoup4` | Lineup scraping from mlb.com |
   | `keras` | Leash model definition, training, checkpointing |
   | `POT` | Python Optimal Transport -- Earth Mover's Distance |

   Install all dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   ---

   ## Usage

   ### Simulate today's games

   ```python
   from driver import process_today
   process_today(exclude_games=[])  # optionally pass a list of game PKs to skip
   ```

   ### Simulate a date range

   ```python
   from datetime import date
   from driver import process_date_range
   process_date_range(date(2026, 4, 1), date(2026, 4, 7), exclude_games=[])
   ```

   ### Rebuild all historical data from scratch

   ```python
   # Re-fetch all pitch data (2017 to present)
   from data.create_season_play_file import refresh_plays
   refresh_plays()

   # Rebuild pitcher hook training data from the play files
   from data.create_leash_model_data import refresh_data
   refresh_data()
   ```

   ### Full pipeline for a single date

   When `_process_date()` runs for a given date it performs these steps in order:

   1. Query the MLB schedule API for games scheduled on that date
   2. Fetch any missing pitch-by-pitch data for `date - 1` (yesterday's completed games)
   3. Extract pitcher hook training records for `date - 1`
   4. Compute batter and pitcher similarity matrices over the trailing 4 seasons
   5. Train starter and reliever leash models on all hook data prior to the target date
   6. Scrape starting lineups for the target date from mlb.com
   7. Run 100-simulation Monte Carlo for each scheduled game
   8. Write per-game results CSVs under `{YYYY}/{YYYY_MM_DD}/`

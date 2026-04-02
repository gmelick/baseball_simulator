import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy.linalg import inv
from scipy.spatial.distance import mahalanobis
from ot import emd2

feature_list = ['release_speed', 'end_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'pfx_x', 'pfx_z']
takes = ['*B', 'B', 'C', 'H', 'P']
contact_codes = ['X', 'D', 'E', 'F', 'L']
in_play_codes = ['X', 'D', 'E']

def pitcher_similarities(as_of_date):
    print(f'Calculating Pitcher Similarities as of {as_of_date.strftime("%Y-%m-%d")}')
    df_list = []
    for year in range(as_of_date.year - 3, as_of_date.year + 1):
        if os.path.exists(f'{year}.csv'):
            season_df = pd.read_csv(f'{year}.csv', encoding='cp1252', parse_dates=['game_date'])
            season_df['season'] = year
            df_list.append(season_df)
    df = pd.concat(df_list)
    pitch_similarities = calc_pitch_similarities(df)
    pitch_sims_flattened = {(h, pitch_similarities[h][0][a], pitch_similarities[h][0][b]): pitch_similarities[h][1][a, b] for h in ['R', 'L'] for a in range(len(pitch_similarities[h][0])) for b in range(len(pitch_similarities[h][0]))}
    results_similarities = calc_result_similarities(df)
    results_sims_flattened = {(h, results_similarities[h][0][a], results_similarities[h][0][b]): results_similarities[h][1][a, b] for h in ['R', 'L'] for a in range(len(results_similarities[h][0])) for b in range(len(results_similarities[h][0]))}
    total_similarities = {'R': [], 'L': []}
    max_sim = 0
    for hand in ['R', 'L']:
        pitcher_season_list = list(set(pitch_similarities[hand][0]) & set(results_similarities[hand][0]))
        for i in range(len(pitcher_season_list)):
            pitcher_season = pitcher_season_list[i]
            for j in range(i, len(pitcher_season_list)):
                cmp_pitcher_season = pitcher_season_list[j]
                similarity = pitch_sims_flattened[(hand, pitcher_season, cmp_pitcher_season)] * .5 + results_sims_flattened[(hand, pitcher_season, cmp_pitcher_season)] * .5
                if similarity > max_sim:
                    max_sim = similarity
                total_similarities[hand].append(list(pitcher_season) + list(cmp_pitcher_season) + [pitch_sims_flattened[(hand, pitcher_season, cmp_pitcher_season)] * .5 + results_sims_flattened[(hand, pitcher_season, cmp_pitcher_season)] * .5])
    with open('RHP Similarities.csv', 'w+') as file:
        file.write('Year_1,ID_1,Year_2,ID_2,Similarity\n')
        for entry in total_similarities['R']:
            file.write(f'{entry[1]},{entry[0]},{entry[3]},{entry[2]},{entry[4] / max_sim}\n')
    with open('LHP Similarities.csv', 'w+') as file:
        file.write('Year_1,ID_1,Year_2,ID_2,Similarity\n')
        for entry in total_similarities['L']:
            file.write(f'{entry[1]},{entry[0]},{entry[3]},{entry[2]},{entry[4] / max_sim}\n')

def calc_pitch_similarities(df):
    sim_dict = {}
    for pitch_hand in ['R', 'L']:
        hand_df = df[df.p_throws == pitch_hand].copy()
        pitcher_type_counts = hand_df.groupby(['pitcher', 'season', 'pitch_type'])['game_pk'].count().reset_index()
        pitcher_type_counts = pitcher_type_counts[pitcher_type_counts.game_pk >= 100]
        pitcher_total_pitches = pitcher_type_counts.groupby(['pitcher', 'season'])['game_pk'].sum()
        pitcher_total_pitches = pitcher_total_pitches[pitcher_total_pitches >= 300]
        pt = PowerTransformer(method='yeo-johnson')
        hand_df[feature_list] = pt.fit_transform(hand_df[feature_list])
        scaler = StandardScaler()
        hand_df[feature_list] = scaler.fit_transform(hand_df[feature_list])
        pitch_averages = hand_df.groupby(['pitcher', 'season', 'pitch_type'])[feature_list].mean()
        pitch_averages = pitch_averages.join(pitcher_type_counts.set_index(['pitcher', 'season', 'pitch_type']), how='inner').reset_index()
        pitch_averages = pitch_averages.set_index(['pitcher', 'season']).join(pitcher_total_pitches, how='inner', rsuffix='_count')[feature_list]
        cov = np.cov(pitch_averages.T)
        inv_cov = inv(cov)
        pitcher_list = list(pitcher_total_pitches.index)
        pitcher_arrs = {p: pitch_averages.loc[p].values.astype(float) for p in pitcher_list}
        pitcher_dists = {p: pitcher_type_counts.loc[(pitcher_type_counts.pitcher == p[0]) & (pitcher_type_counts.season == p[1]), 'game_pk'] / pitcher_total_pitches.loc[p] for p in pitcher_list}
        pitcher_sims = np.zeros((len(pitcher_list), len(pitcher_list)))
        for i in range(len(pitcher_list)):
            pitcher = pitcher_list[i]
            pitcher_arr = pitcher_arrs[pitcher]
            if pitcher_arr.ndim <= 1:
                pitcher_arr = np.reshape(pitcher_arr, (1, pitcher_arr.shape[0]))
            pitcher_dist = pitcher_dists[pitcher].values
            for j in range(i + 1, len(pitcher_list)):
                cmp_pitcher = pitcher_list[j]
                cmp_pitcher_arr = pitcher_arrs[cmp_pitcher]
                if cmp_pitcher_arr.ndim <= 1:
                    cmp_pitcher_arr = np.reshape(cmp_pitcher_arr, (1, cmp_pitcher_arr.shape[0]))
                cmp_pitcher_dist = pitcher_dists[cmp_pitcher].values
                pitch_costs = np.zeros((pitcher_arr.shape[0], cmp_pitcher_arr.shape[0]))
                for a in range(len(pitcher_arr)):
                    for b in range(len(cmp_pitcher_arr)):
                        pitch_costs[a, b] = mahalanobis(pitcher_arr[a], cmp_pitcher_arr[b], inv_cov)
                pitcher_sims[i, j] = emd2(pitcher_dist, cmp_pitcher_dist, pitch_costs)
                pitcher_sims[j, i] = pitcher_sims[i, j]
        pitcher_sims[pitcher_sims > 0] = (pitcher_sims[pitcher_sims > 0] - pitcher_sims[pitcher_sims > 0].min()) / (pitcher_sims[pitcher_sims > 0].max() - pitcher_sims[pitcher_sims > 0].min())
        sim_dict[pitch_hand] = (pitcher_list, pitcher_sims)
    return sim_dict

def calc_result_similarities(df):
    sim_dict = {}
    for pitch_hand in ['R', 'L']:
        hand_df = df[df.p_throws == pitch_hand].copy()
        pitch_is_strike = (
                (hand_df["plate_z"] >= hand_df["sz_bot"]) & (hand_df["plate_z"] <= hand_df["sz_top"]) &
                (hand_df["plate_x"] >= -0.71) & (hand_df["plate_x"] <= 0.71)
        )
        swing = ~hand_df["type"].isin(takes)
        contact = hand_df["type"].isin(contact_codes)

        hand_df["pitch_is_strike"] = pitch_is_strike
        hand_df["swing"] = swing
        hand_df["contact"] = contact
        hand_df["swinging_strikes"] = swing & pitch_is_strike
        hand_df["swinging_balls"] = swing & (~pitch_is_strike)
        hand_df["contact_strikes"] = contact & pitch_is_strike
        hand_df["contact_balls"] = contact & (~pitch_is_strike)

        ip = hand_df[hand_df["type"].isin(in_play_codes)].copy()
        ip = ip[ip["launch_angle"].notna() & ip["launch_speed"].notna() & ip["spray_angle"].notna()]

        ip["grounder"] = ip["launch_angle"] < 5
        ip["liner"] = (ip["launch_angle"] >= 5) & (ip["launch_angle"] <= 33)
        ip["fly"] = ip["launch_angle"] > 33

        ip["hard"] = ip["launch_speed"] >= 95
        ip["medium"] = (ip["launch_speed"] < 95) & (ip["launch_speed"] > 75)
        ip["soft"] = ip["launch_speed"] <= 75

        center = (ip["spray_angle"] >= -15) & (ip["spray_angle"] <= 15)
        pull_right = (ip["spray_angle"] > 15) & (ip["stand"] == "L")
        pull_left = (ip["spray_angle"] < -15) & (ip["stand"] == "R")
        oppo_right = (ip["spray_angle"] > 15) & (ip["stand"] == "R")
        oppo_left = (ip["spray_angle"] < -15) & (ip["stand"] == "L")
        ip["center"] = center
        ip["pull"] = pull_right | pull_left
        ip["oppo"] = oppo_right | oppo_left

        agg1 = hand_df.groupby(["pitcher", "season"], sort=False).agg(
            total_pitches=("game_pk", "count"),
            num_strikes=("pitch_is_strike", "sum"),
            num_balls=("pitch_is_strike", lambda s: (~s).sum()),
            total_swings=("swing", "sum"),
            swinging_strikes=("swinging_strikes", "sum"),
            swinging_balls=("swinging_balls", "sum"),
            num_contact=("contact", "sum"),
            num_contact_strikes=("contact_strikes", "sum"),
            num_contact_balls=("contact_balls", "sum"),
        )

        agg2 = ip.groupby(["pitcher", "season"], sort=False).agg(
            num_in_play=("game_pk", "count"),
            num_liners=("liner", "sum"),
            num_grounder=("grounder", "sum"),
            num_fly=("fly", "sum"),
            num_pull=("pull", "sum"),
            num_center=("center", "sum"),
            num_oppo=("oppo", "sum"),
            num_soft=("soft", "sum"),
            num_medium=("medium", "sum"),
            num_hard=("hard", "sum"),
        )
        prof = agg1.join(agg2, how="inner")
        prof = prof[(prof["total_pitches"] > 300) & (prof["num_in_play"] > 50)].copy()
        pitcher_list = list(prof.index)
        for base in ["liners", "grounder", "fly", "pull", "center", "oppo", "soft", "medium", "hard"]:
            prof[f"{base}_total"] = np.divide(prof[f"num_{base}"], prof["total_pitches"])
            prof[f"{base}_in_play"] = np.divide(prof[f"num_{base}"], prof["num_in_play"])
        prof["swing_rate"] = np.divide(prof["total_swings"], prof["total_pitches"])
        prof["in_zone_swing_rate"] = np.divide(prof["swinging_strikes"], prof["num_strikes"])
        prof["chase_rate"] = np.divide(prof["swinging_balls"], prof["num_balls"])
        prof["contact_rate"] = np.divide(prof["num_contact"], prof["total_swings"])
        prof["in_zone_contact_rate"] = np.divide(prof["num_contact_strikes"], prof["swinging_strikes"])
        prof["out_zone_contact_rate"] = np.divide(prof["num_contact_balls"], prof["swinging_balls"])
        prof["strike_rate"] = np.divide(prof["num_strikes"], prof["total_pitches"])
        feat_cols = [
            "liners_total", "grounder_total", "fly_total",
            "pull_total", "center_total", "oppo_total",
            "soft_total", "medium_total", "hard_total",
            "liners_in_play", "grounder_in_play", "fly_in_play",
            "pull_in_play", "center_in_play", "oppo_in_play",
            "soft_in_play", "medium_in_play", "hard_in_play",
            "swing_rate", "in_zone_swing_rate", "chase_rate", "contact_rate",
            "in_zone_contact_rate", "out_zone_contact_rate", "strike_rate",
        ]
        x = prof[feat_cols].to_numpy(dtype=float, copy=False)
        x = np.log(x/(1-x))
        cov = np.cov(x.T)
        inv_cov = inv(cov)
        sim_scores = np.zeros((len(pitcher_list), len(pitcher_list)))
        for i in range(len(pitcher_list)):
            pitcher_stats = x[i]
            for j in range(i+1, len(pitcher_list)):
                cmp_pitcher_stats = x[j]
                sim_scores[i, j] = mahalanobis(pitcher_stats, cmp_pitcher_stats, inv_cov)
                sim_scores[j, i] = sim_scores[i, j]
        sim_scores[sim_scores > 0] = (sim_scores[sim_scores > 0] - sim_scores[sim_scores > 0].min()) / (sim_scores[sim_scores > 0].max() - sim_scores[sim_scores > 0].min())
        sim_dict[pitch_hand] = (pitcher_list, sim_scores)
    return sim_dict

if __name__ == '__main__':
    from datetime import datetime
    pitcher_similarities(datetime(2025, 9, 23))

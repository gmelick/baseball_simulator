import pandas as pd
import numpy as np
import os
from scipy.linalg import inv
from scipy.spatial.distance import mahalanobis

takes = ['*B', 'B', 'C', 'H', 'P']
contact_codes = ['X', 'D', 'E', 'F', 'L']
in_play_codes = ['X', 'D', 'E']

def batter_similarities(as_of_date):
    print(f'Calculating Batter Similarities as of {as_of_date.strftime("%Y-%m-%d")}')
    df_list = []
    for year in range(as_of_date.year - 3, as_of_date.year + 1):
        if os.path.exists(f'{year}.csv'):
            season_df = pd.read_csv(f'{year}.csv', encoding='cp1252', parse_dates=['game_date'])
            season_df['season'] = year
            df_list.append(season_df)
    df = pd.concat(df_list)
    results_similarities = calc_result_similarities(df)

    with open('RHB Similarities.csv', 'w+') as file:
        file.write('Year_1,ID_1,Year_2,ID_2,Similarity\n')
        for i in range(len(results_similarities['R'][0])):
            batter, season = results_similarities['R'][0][i]
            for j in range(i, len(results_similarities['R'][0])):
                cmp_batter, cmp_season = results_similarities['R'][0][j]
                score = results_similarities['R'][1][i, j]
                file.write(f'{season},{batter},{cmp_season},{cmp_batter},{score}\n')

    with open('LHB Similarities.csv', 'w+') as file:
        file.write('Year_1,ID_1,Year_2,ID_2,Similarity\n')
        for i in range(len(results_similarities['L'][0])):
            batter, season = results_similarities['L'][0][i]
            for j in range(i, len(results_similarities['L'][0])):
                cmp_batter, cmp_season = results_similarities['L'][0][j]
                score = results_similarities['L'][1][i, j]
                file.write(f'{season},{batter},{cmp_season},{cmp_batter},{score}\n')

def calc_result_similarities(df):
    sim_dict = {}
    for bat_stand in ['R', 'L']:
        hand_df = df[df.stand == bat_stand].copy()
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

        agg1 = hand_df.groupby(["batter", "season"], sort=False).agg(
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

        agg2 = ip.groupby(["batter", "season"], sort=False).agg(
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
        batter_list = list(prof.index)
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
        sim_scores = np.zeros((len(batter_list), len(batter_list)))
        for i in range(len(batter_list)):
            batter_stats = x[i]
            for j in range(i+1, len(batter_list)):
                cmp_batter_stats = x[j]
                sim_scores[i, j] = mahalanobis(batter_stats, cmp_batter_stats, inv_cov)
                sim_scores[j, i] = sim_scores[i, j]
        sim_scores[sim_scores > 0] = (sim_scores[sim_scores > 0] - sim_scores[sim_scores > 0].min()) / (sim_scores[sim_scores > 0].max() - sim_scores[sim_scores > 0].min())
        sim_dict[bat_stand] = (batter_list, sim_scores)
    return sim_dict

if __name__ == '__main__':
    from datetime import datetime
    batter_similarities(datetime(2025, 9, 23))

import requests
from bs4 import BeautifulSoup
from simulation.game import Game
import numpy as np
import os
import pandas as pd
from datetime import date

game_list_class = 'starting-lineups__matchup'
away_team_class = 'starting-lineups__team-name--away'
home_team_class = 'starting-lineups__team-name--home'
team_name_class = 'starting-lineups__team-name--link'
pitcher_name_class = 'starting-lineups__pitcher--link'
away_lineup_class = 'starting-lineups__team--away'
home_lineup_class = 'starting-lineups__team--home'
lineup_player_class = 'starting-lineups__player'
people_link = 'https://statsapi.mlb.com/api/v1/people'
plays_fields = ['Season', 'game_pk', 'game_date', 'pitcher', 'batter', 'at_bat_number', 'pitch_number', 'inning',
                'balls', 'strikes', 'outs', 'home_score', 'away_score', 'runs_on_pitch',
                'outs_on_pitch', 'earned_runs_on_pitch', 'rbis_on_pitch', 'on_1b', 'on_2b', 'on_3b', '1b_runner_score',
                '2b_runner_score', '3b_runner_score', 'post_on_1b', 'post_on_2b', 'post_on_3b', 'type', 'events']

def simulate_day(cur_date, hook_model_starters, hook_model_relievers, exclude_list):
    base_schedule_url = 'https://statsapi.mlb.com/api/v1/schedule'
    params = {'sportId': 1, 'gameTypes': ['R', 'F', 'D', 'L', 'W'], 'date': cur_date.strftime('%Y-%m-%d')}
    while True:
        try:
            schedule = requests.get(f'{base_schedule_url}', params=params).json()['dates']
            break
        except:
            schedule = None
            continue
    game_pks = []
    for game_date in schedule:
        for game in game_date['games']:
            if 'rescheduleGameDate' in game or 'resumeGameDate' in game or game['status']['statusCode'] == 'CR':
                continue
            if game['gamePk'] not in exclude_list:
                game_pks.append(game['gamePk'])

    plays, pitcher_similarities, rhb_hp_similarities, lhb_hp_similarities, bullpen_df = create_frames(cur_date)

    url = f'https://www.mlb.com/starting-lineups/{cur_date.strftime("%Y-%m-%d")}'
    while True:
        try:
            day_page = BeautifulSoup(requests.get(url, headers={'Accept-Language': 'en-US'}).content, 'html.parser')
            break
        except:
            day_page = None
            continue
    game_list = day_page.find_all('div', {'class': game_list_class})
    while len(game_pks) > 0:
        for game in game_list:
            game_key = int(game['data-gamepk'])
            if game_key not in game_pks:
                continue
            os.makedirs(f'backtests\\{cur_date.year}\\{cur_date.strftime("%Y_%m_%d")}', exist_ok=True)
            print('Getting Lineups')
            away_team = game.find('span', {'class': away_team_class}).find('a', {'class': team_name_class}).text.strip()
            home_team = game.find('span', {'class': home_team_class}).find('a', {'class': team_name_class}).text.strip()
            pitchers = game.find_all('a', {'class': pitcher_name_class})
            if len(pitchers) < 4:
                if cur_date < date.today():
                    game_pks.remove(game_key)
                continue
            away_pitcher_name = pitchers[1].text.strip()
            away_pitcher_link = pitchers[1]['href']
            away_pitcher_id = int(away_pitcher_link[away_pitcher_link.rfind('-') + 1:])
            home_pitcher_name = pitchers[3].text.strip()
            home_pitcher_link = pitchers[3]['href']
            home_pitcher_id = int(home_pitcher_link[home_pitcher_link.rfind('-') + 1:])
            away_lineup_names, away_lineup_ids, away_position = [], [], []
            for player in game.find('ol', {'class': away_lineup_class}).find_all('li', {'class': lineup_player_class}):
                away_lineup_names.append(player.find('a').text.strip())
                away_lineup_ids.append(int(player.find('a')['href'][player.find('a')['href'].rfind('-') + 1:]))
                away_position.append(player.text[player.text.rfind(' ') + 1:])
            home_lineup_names, home_lineup_ids, home_position = [], [], []
            for player in game.find('ol', {'class': home_lineup_class}).find_all('li', {'class': lineup_player_class}):
                home_lineup_names.append(player.find('a').text.strip())
                home_lineup_ids.append(int(player.find('a')['href'][player.find('a')['href'].rfind('-') + 1:]))
                home_position.append(player.text[player.text.rfind(' ') + 1:])
            away_bench = get_bench(game_key, 'away', 'bench', away_lineup_ids)
            home_bench = get_bench(game_key, 'home', 'bench', home_lineup_ids)
            away_bullpen = get_bench(game_key, 'away', 'bullpen', away_pitcher_id)
            home_bullpen = get_bench(game_key, 'home', 'bullpen', home_pitcher_id)
            if len(away_lineup_ids) < 9 or len(home_lineup_ids) < 9:
                continue

            pitcher_list = [home_pitcher_id, away_pitcher_id] + home_bullpen + away_bullpen
            batter_list = home_lineup_ids + away_lineup_ids
            pitcher_hand_map, home_bullpen_appearances, away_bullpen_appearances = setup_game(cur_date, game_key, pitcher_list, batter_list, home_bullpen, away_bullpen, plays, pitcher_similarities, rhb_hp_similarities, lhb_hp_similarities, bullpen_df)

            process_game(cur_date, home_team, away_team, game_key, home_bullpen, away_bullpen, away_pitcher_id,
                         home_pitcher_id, away_lineup_ids, home_lineup_ids, away_bullpen_appearances,
                         home_bullpen_appearances, pitcher_hand_map, hook_model_starters, hook_model_relievers)
            game_pks.remove(game_key)
            os.remove(f'backtests\\{cur_date.year}\\{cur_date.strftime("%Y_%m_%d")}\\{cur_date.strftime("%Y_%m_%d")}_{game_key}.feather')

def simulate_game(cur_date, game_key, home_team, away_team, home_pitcher_id, away_pitcher_id, home_bullpen, away_bullpen, home_lineup_ids, away_lineup_ids, hook_model_starters, hook_model_relievers):
    plays, pitcher_similarities, rhb_hp_similarities, lhb_hp_similarities, bullpen_df = create_frames(cur_date)
    os.makedirs(f'backtests\\{cur_date.year}\\{cur_date.strftime("%Y_%m_%d")}', exist_ok=True)
    pitcher_list = [home_pitcher_id, away_pitcher_id] + home_bullpen + away_bullpen
    batter_list = home_lineup_ids + away_lineup_ids
    pitcher_hand_map, home_bullpen_appearances, away_bullpen_appearances = setup_game(cur_date, game_key, pitcher_list,
                                                                                      batter_list, home_bullpen,
                                                                                      away_bullpen, plays,
                                                                                      pitcher_similarities,
                                                                                      rhb_hp_similarities,
                                                                                      lhb_hp_similarities, bullpen_df)

    process_game(cur_date, home_team, away_team, game_key, home_bullpen, away_bullpen, away_pitcher_id,
                 home_pitcher_id, away_lineup_ids, home_lineup_ids, away_bullpen_appearances,
                 home_bullpen_appearances, pitcher_hand_map, hook_model_starters, hook_model_relievers)
    os.remove(f'backtests\\{cur_date.year}\\{cur_date.strftime("%Y_%m_%d")}\\{cur_date.strftime("%Y_%m_%d")}_{game_key}.feather')
    return 0

def setup_game(cur_date, game_key, pitcher_list, batter_list, home_bullpen, away_bullpen, plays, pitcher_similarities, rhb_hp_similarities, lhb_hp_similarities, bullpen_df):
    pitcher_hand_map = fetch_pitcher_hands(set(pitcher_list))
    batter_hand_map = fetch_batter_hands(set(batter_list))
    similarities = [pitcher_similarities, rhb_hp_similarities, lhb_hp_similarities]
    game_plays = combine_plays_similarities(plays.copy(), pitcher_list, batter_list, batter_hand_map, similarities)
    game_plays.to_feather(f'backtests\\{cur_date.year}\\{cur_date.strftime("%Y_%m_%d")}\\{cur_date.strftime("%Y_%m_%d")}_{game_key}.feather')
    away_bullpen_appearances = bullpen_df[np.isin(bullpen_df.pitcher, away_bullpen)]
    home_bullpen_appearances = bullpen_df[np.isin(bullpen_df.pitcher, home_bullpen)]
    return pitcher_hand_map, home_bullpen_appearances, away_bullpen_appearances

def create_frames(cur_date):
    print('Creating Plays Frame')
    frames = []
    for year in range(cur_date.year - 3, cur_date.year + 1):
        if not os.path.exists(f"{year}.csv"):
            continue
        df = pd.read_csv(f"{year}.csv", encoding="cp1252", parse_dates=["game_date"], low_memory=False).copy()
        df['Season'] = year
        frames.append(df[df["game_date"] < pd.to_datetime(cur_date)])
    plays = pd.concat(frames, ignore_index=True)
    plays['Strike'] = np.isin(plays.type, ['C', 'F', 'L', 'M', 'O', 'S', 'T', 'W'])
    plays['Ball'] = np.isin(plays.type, ['*B', 'B', 'P'])
    plays['Foul'] = np.isin(plays.type, ['F', 'L'])
    plays['Hit'] = np.isin(plays.events, ['single', 'double', 'triple', 'home_run'])
    plays['HBP'] = np.isin(plays.events, ['hit_by_pitch'])
    plays['Double'] = np.isin(plays.events, ['double'])
    plays['Triple'] = np.isin(plays.events, ['triple'])
    plays['Home_Run'] = np.isin(plays.events, ['home_run'])
    plays.drop(columns=['type', 'events'], inplace=True)
    plays['on_1b_na'] = pd.isna(plays['on_1b'])
    plays['on_2b_na'] = pd.isna(plays['on_2b'])
    plays['on_3b_na'] = pd.isna(plays['on_3b'])
    plays['post_on_1b'] = plays['post_on_1b'].fillna(-1)
    plays['post_on_2b'] = plays['post_on_2b'].fillna(-1)
    plays['post_on_3b'] = plays['post_on_3b'].fillna(-1)
    plays['Pitch_concat'] = plays['pitcher'].astype(str) + plays['Season'].astype(str)
    plays['Bat_concat'] = plays['batter'].astype(str) + plays['Season'].astype(str)
    plays['Difference'] = np.minimum(np.absolute(plays.home_score - plays.away_score), 5)

    print('Creating Bullpen Frame')
    appearance_df = plays[(plays.game_date >= pd.to_datetime(date(cur_date.year - 1, cur_date.month, cur_date.day))) & (plays.inning > 1)].copy()
    appearance_df['Entrance_ab_concat'] = appearance_df['pitcher'].astype(str) + appearance_df['game_pk'].astype(str) + appearance_df['at_bat_number'].astype(str).str.zfill(3)
    appearance_df['Entrance_pitch_concat'] = appearance_df['pitcher'].astype(str) + appearance_df['game_pk'].astype(str) + appearance_df['at_bat_number'].astype(str).str.zfill(3) + appearance_df['pitch_number'].astype(str).str.zfill(2)
    appearance_df.set_index('Entrance_ab_concat', inplace=True)
    entrance_df = appearance_df.groupby(['pitcher', 'game_pk'])['at_bat_number'].min().reset_index(drop=False)
    entrance_df['Entrance_ab_concat'] = entrance_df.pitcher.astype(str) + entrance_df.game_pk.astype(str) + entrance_df.at_bat_number.astype(str).str.zfill(3)
    entrance_df.set_index('Entrance_ab_concat', inplace=True)
    pitch_df = appearance_df.groupby(['pitcher', 'game_pk', 'at_bat_number'])['pitch_number'].min().reset_index(drop=False)
    pitch_df['Entrance_pitch_concat'] = pitch_df.pitcher.astype(str) + pitch_df.game_pk.astype(str) + pitch_df.at_bat_number.astype(str).str.zfill(3) + pitch_df.pitch_number.astype(str).str.zfill(2)
    pitch_df.set_index('Entrance_pitch_concat', inplace=True)
    bullpen_df = appearance_df.join(entrance_df, how='inner', rsuffix='_x').set_index('Entrance_pitch_concat').join(pitch_df, how='inner', rsuffix='_y')
    bullpen_df = bullpen_df.groupby(['pitcher', 'inning', 'Difference']).game_pk.count().reset_index()

    print('Loading Similarities')
    lhb_hp_similarities = pd.read_csv('LHB Similarities.csv')
    lhb_hp_similarities['Bat_concat'] = lhb_hp_similarities['ID_2'].astype(str) + lhb_hp_similarities['Year_2'].astype(str)
    lhb_hp_similarities.set_index('Bat_concat', inplace=True)
    rhb_hp_similarities = pd.read_csv('RHB Similarities.csv')
    rhb_hp_similarities['Bat_concat'] = rhb_hp_similarities['ID_2'].astype(str) + rhb_hp_similarities['Year_2'].astype(str)
    rhb_hp_similarities.set_index('Bat_concat', inplace=True)
    lhp_hp_similarities = pd.read_csv('LHP Similarities.csv')
    rhp_hp_similarities = pd.read_csv('RHP Similarities.csv')
    pitcher_similarities = pd.concat([lhp_hp_similarities, rhp_hp_similarities], ignore_index=True)
    standardize_similarities([lhb_hp_similarities, rhb_hp_similarities, pitcher_similarities])
    pitcher_similarities['Pitch_concat'] = pitcher_similarities['ID_2'].astype(str) + pitcher_similarities['Year_2'].astype(str)
    pitcher_similarities = pitcher_similarities.set_index('Pitch_concat')[['ID_1', 'Year_1', 'Percentage']]
    return plays, pitcher_similarities, rhb_hp_similarities, lhb_hp_similarities, bullpen_df

def standardize_similarities(df_list):
    for df in df_list:
        sim_min = df[df['Similarity'] > 0]['Similarity'].min()
        sim_max = df[df['Similarity'] > 0]['Similarity'].max()
        df['Percentage'] = np.minimum(1, 1 - ((df['Similarity'] - sim_min) / (sim_max - sim_min)))

def get_bench(key, team_type, list_type, exclude):
    while True:
        try:
            game_json = requests.get(f'https://statsapi.mlb.com/api/v1.1/game/{key}/feed/live').json()
            break
        except:
            game_json = None
            continue
    if list_type == 'bench':
        list_temp = (game_json['liveData']['boxscore']['teams'][team_type][list_type] +
                     game_json['liveData']['boxscore']['teams'][team_type]['batters'])
        ret_list = []
        for a in list_temp:
            if a not in exclude:
                ret_list.append(a)
    else:
        list_temp = (game_json['liveData']['boxscore']['teams'][team_type][list_type] +
                     game_json['liveData']['boxscore']['teams'][team_type]['pitchers'])
        ret_list = []
        for a in list_temp:
            if a != exclude:
                ret_list.append(a)
    return ret_list

def fetch_pitcher_hands(pitcher_ids):
    while True:
        try:
            response = requests.get(f'{people_link}?personIds={",".join(map(str, pitcher_ids))}').json()
            break
        except:
            response = None
            continue
    return {int(person['id']): person['pitchHand']['code'] for person in response.get('people', [])}

def fetch_batter_hands(batter_ids):
    while True:
        try:
            response = requests.get(f'{people_link}?personIds={",".join(map(str, batter_ids))}').json()
            break
        except:
            response = None
            continue
    return {int(person['id']): person['batSide']['code'] for person in response.get('people', [])}

def combine_plays_similarities(plays, pitcher_list, batter_list, batter_hand_map, similarities):
    pitcher_similarities, rhb_hp_similarities, lhb_hp_similarities = similarities

    print('Creating Pitcher Play Similarities')
    plays.set_index('Pitch_concat', inplace=True)
    pitcher_sims = pitcher_similarities[pitcher_similarities.ID_1.isin(pitcher_list)]
    pitcher_id_years = pitcher_sims.groupby("ID_1")["Year_1"].max().reset_index()
    for pitcher in pitcher_list:
        if pitcher not in pitcher_sims.ID_1.values:
            plays[f'P_{pitcher}'] = .5
        else:
            max_year = pitcher_id_years.loc[pitcher_id_years.ID_1 == pitcher, 'Year_1'].values[0]
            pitcher_sim_year = pitcher_sims[(pitcher_sims.ID_1 == pitcher) & (pitcher_sims.Year_1 == max_year)]
            plays[f'P_{pitcher}'] = plays.join(pitcher_sim_year, how='left')['Percentage'].fillna(.5)

    print('Creating Batter Play Similarities')
    plays.set_index('Bat_concat', inplace=True)
    rhb_sims = rhb_hp_similarities[rhb_hp_similarities.ID_1.isin(batter_list)]
    rhb_id_years = rhb_sims.groupby('ID_1')['Year_1'].max().reset_index()
    lhb_sims = lhb_hp_similarities[lhb_hp_similarities.ID_1.isin(batter_list)]
    lhb_id_years = lhb_sims.groupby('ID_1')['Year_1'].max().reset_index()
    for batter in batter_list:
        batter_hand = batter_hand_map[batter]
        if batter_hand == 'R':
            if batter not in rhb_sims.ID_1.values:
                plays[f'B_R_{batter}'] = .5
                plays[f'B_L_{batter}'] = .5
            else:
                max_year = rhb_id_years.loc[rhb_id_years.ID_1 == batter, 'Year_1'].values[0]
                batter_sim_year = rhb_sims[(rhb_sims.ID_1 == batter) & (rhb_sims.Year_1 == max_year)]
                plays[f'B_R_{batter}'] = plays.join(batter_sim_year, how='left')['Percentage'].fillna(.5)
                plays[f'B_L_{batter}'] = plays[f'B_R_{batter}'].copy()
        elif batter_hand == 'L':
            if batter not in lhb_sims.ID_1.values:
                plays[f'B_R_{batter}'] = .5
                plays[f'B_L_{batter}'] = .5
            else:
                max_year = lhb_id_years.loc[lhb_id_years.ID_1 == batter, 'Year_1'].values[0]
                batter_sim_year = lhb_sims[(lhb_sims.ID_1 == batter) & (lhb_sims.Year_1 == max_year)]
                plays[f'B_R_{batter}'] = plays.join(batter_sim_year, how='left')['Percentage'].fillna(.5)
                plays[f'B_L_{batter}'] = plays[f'B_R_{batter}'].copy()
        else:
            if batter not in lhb_sims.ID_1.values:
                plays[f'B_R_{batter}'] = .5
            else:
                max_year = lhb_id_years.loc[lhb_id_years.ID_1 == batter, 'Year_1'].values[0]
                batter_sim_year = lhb_sims[(lhb_sims.ID_1 == batter) & (lhb_sims.Year_1 == max_year)]
                plays[f'B_R_{batter}'] = plays.join(batter_sim_year, how='left')['Percentage'].fillna(.5)
            if batter not in rhb_sims.ID_1.values:
                plays[f'B_L_{batter}'] = .5
            else:
                max_year = rhb_id_years.loc[rhb_id_years.ID_1 == batter, 'Year_1'].values[0]
                batter_sim_year = rhb_sims[(rhb_sims.ID_1 == batter) & (rhb_sims.Year_1 == max_year)]
                plays[f'B_L_{batter}'] = plays.join(batter_sim_year, how='left')['Percentage'].fillna(.5)
    return plays.reset_index(drop=True)

def process_game(cur_date, home_team, away_team, game_key, home_bullpen, away_bullpen, away_pitcher_id, home_pitcher_id,
                 away_lineup_ids, home_lineup_ids, away_bullpen_appearances, home_bullpen_appearances, pitcher_hand_map, hook_model_starters, hook_model_relievers):
    print(f'Simulating {away_team} @ {home_team} - {cur_date.strftime("%m/%d/%Y")} - {game_key}')
    game = Game(cur_date, home_team, away_team, game_key, away_pitcher_id, home_pitcher_id, away_lineup_ids,
                home_lineup_ids, away_bullpen, home_bullpen, away_bullpen_appearances, home_bullpen_appearances, pitcher_hand_map)
    game.simulate(100, 8, 2, hook_model_starters, hook_model_relievers)

if __name__ == '__main__':
    from keras.models import load_model
    os.chdir('..')
    simulate_day(date(2026, 3, 25), load_model('best_model_0.keras'), load_model('best_model_1.keras'), [])

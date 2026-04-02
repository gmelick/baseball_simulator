from datetime import date, timedelta
import requests
from similarities.batter_similarity import batter_similarities
from similarities.pitcher_similarity import pitcher_similarities
from data.create_season_play_file import get_plays
from data.create_leash_model_data import load_data
from models.leash_model import create_models
from simulation.simulate_day import simulate_day

base_schedule_url = 'https://statsapi.mlb.com/api/v1/schedule'

def process_today(exclude_games):
    _process_date(date.today(), exclude_games)

def process_date_range(start_date, end_date, exclude_games):
    cur_date = start_date
    while cur_date <= end_date:
        _process_date(cur_date, exclude_games)
        cur_date += timedelta(1)

def _process_date(cur_date, exclude_games):
    print(f'Starting Processing for {cur_date.strftime("%Y-%m-%d")}')
    params = {'sportId': 1, 'gameTypes': ['R', 'F', 'D', 'L', 'W', 'C', 'P'], 'date': cur_date.strftime('%Y-%m-%d')}
    while True:
        try:
            schedule = requests.get(f'{base_schedule_url}', params=params).json()['dates']
            break
        except ConnectionError:
            schedule = None
            continue
    game_pks = 0
    for game_date in schedule:
        for game in game_date['games']:
            if 'rescheduleGameDate' in game or 'resumeGameDate' in game or game['status']['statusCode'] == 'CR':
                continue
            game_pks += 1
    get_plays(cur_date - timedelta(1), cur_date - timedelta(1))
    load_data(cur_date - timedelta(1))
    if game_pks == 0:
        return
    batter_similarities(cur_date)
    pitcher_similarities(cur_date)
    hook_model_starters, hook_model_relievers = create_models(cur_date)
    simulate_day(cur_date, hook_model_starters, hook_model_relievers, exclude_games)

if __name__ == '__main__':
    _process_date(date(2026, 3, 26), date.today() - timedelta(1))

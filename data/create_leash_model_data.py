import pandas as pd
import requests
from datetime import date
import os
from requests.exceptions import ReadTimeout, ConnectTimeout

base_url = 'https://statsapi.mlb.com/api/v1.1/game'

def connect(url, params=None):
    while True:
        try:
            if params is None:
                resp = requests.get(url).json()
            else:
                resp = requests.get(url, params=params).json()
            break
        except:
            continue
    return resp

def load_data(cur_date):
    if os.path.exists(f'{cur_date.year}.csv'):
        season_df = pd.read_csv(f'{cur_date.year}.csv', encoding='cp1252', parse_dates=['game_date'])
        season_df = season_df[season_df.game_date == pd.to_datetime(cur_date)]
        if os.path.exists(f'{cur_date.year} Pitcher Hooks.csv'):
            hook_df = pd.read_csv(f'{cur_date.year} Pitcher Hooks.csv', parse_dates=['Date'])
            if pd.to_datetime(cur_date) in hook_df.Date.values:
                return
            file = open(f'{cur_date.year} Pitcher Hooks.csv', 'a+')
        else:
            file = open(f'{cur_date.year} Pitcher Hooks.csv', 'w+')
            file.write('Date,Pitcher,Batters Faced,Outs,Hits,Runs,Strikeouts,Walks,Hit By Pitch,Home Runs,Pitches,'
                       'Strikes,Earned Runs,IsInningEnd,Starter,Pulled\n')
        for game_pk in season_df.game_pk.unique():
            plays = None
            while True:
                try:
                    plays = connect(f'{base_url}/{game_pk}/feed/live')['liveData']['plays']['allPlays']
                    break
                except ReadTimeout:
                    continue
                except ConnectTimeout:
                    continue
            pitcher = None
            stat_line = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            score = 0
            outs = 0
            for play in plays:
                if not play['about']['isTopInning']:
                    continue
                if pitcher is None:
                    stat_line[12] = 1
                    pitcher = play['matchup']['pitcher']['id']
                elif play['matchup']['pitcher']['id'] != pitcher:
                    file.write(f'{cur_date.strftime('%Y-%m-%d')},{pitcher},{",".join([str(x) for x in stat_line])},1\n')
                    stat_line = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    pitcher = play['matchup']['pitcher']['id']
                else:
                    file.write(f'{cur_date.strftime('%Y-%m-%d')},{pitcher},{",".join([str(x) for x in stat_line])},0\n')
                runs_on_play = play['result']['awayScore'] - score
                score += runs_on_play
                stat_line[3] += runs_on_play
                stat_line[0] += 1
                stat_line[1] += play['count']['outs'] - outs
                if play['count']['outs'] == 3:
                    stat_line[10] = 1
                    outs = 0
                else:
                    stat_line[10] = 0
                    outs = play['count']['outs']
                if play['result']['eventType'] in ['single', 'double', 'triple', 'home_run']:
                    stat_line[2] += 1
                    if play['result']['eventType'] == 'home_run':
                        stat_line[7] += 1
                if play['result']['eventType'].startswith('strikeout'):
                    stat_line[4] += 1
                if play['result']['eventType'].endswith('walk'):
                    stat_line[5] += 1
                if play['result']['eventType'] == 'hit_by_pitch':
                    stat_line[6] += 1
                for event in play['playEvents']:
                    if event['isPitch']:
                        if event['details']['isStrike'] or event['details']['isInPlay']:
                            stat_line[8] += 1
                            stat_line[9] += 1
                        else:
                            stat_line[8] += 1
                for runner in play['runners']:
                    if runner['details'].get('earned', False):
                        stat_line[11] += 1
            pitcher = None
            stat_line = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            score = 0
            outs = 0
            for play in plays:
                if play['about']['isTopInning']:
                    continue
                if pitcher is None:
                    stat_line[12] = 1
                    pitcher = play['matchup']['pitcher']['id']
                elif play['matchup']['pitcher']['id'] != pitcher:
                    file.write(f'{cur_date.strftime('%Y-%m-%d')},{pitcher},{",".join([str(x) for x in stat_line])},1\n')
                    stat_line = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    pitcher = play['matchup']['pitcher']['id']
                else:
                    file.write(f'{cur_date.strftime('%Y-%m-%d')},{pitcher},{",".join([str(x) for x in stat_line])},0\n')
                runs_on_play = play['result']['homeScore'] - score
                score += runs_on_play
                stat_line[3] += runs_on_play
                stat_line[0] += 1
                stat_line[1] += play['count']['outs'] - outs
                if play['count']['outs'] == 3:
                    stat_line[10] = 1
                    outs = 0
                else:
                    stat_line[10] = 0
                    outs = play['count']['outs']
                if play['result']['eventType'] in ['single', 'double', 'triple', 'home_run']:
                    stat_line[2] += 1
                    if play['result']['eventType'] == 'home_run':
                        stat_line[7] += 1
                if play['result']['eventType'].startswith('strikeout'):
                    stat_line[4] += 1
                if play['result']['eventType'].endswith('walk'):
                    stat_line[5] += 1
                if play['result']['eventType'] == 'hit_by_pitch':
                    stat_line[6] += 1
                for event in play['playEvents']:
                    if event['isPitch']:
                        if event['details']['isStrike'] or event['details']['isInPlay']:
                            stat_line[8] += 1
                            stat_line[9] += 1
                        else:
                            stat_line[8] += 1
                for runner in play['runners']:
                    if runner.get('earned', False):
                        stat_line[11] += 1
        file.close()

def refresh_data():
    for season in range(2017, date.today().year + 1):
        print(season)
        if os.path.exists(f'{season}.csv'):
            season_df = pd.read_csv(f'{season}.csv', encoding='cp1252')
        else:
            continue
        file = open(f'{season} Pitcher Hooks.csv', 'w+')
        file.write('Date,Pitcher,Batters Faced,Outs,Hits,Runs,Strikeouts,Walks,Hit By Pitch,Home Runs,Pitches,'
                   'Strikes,Earned Runs,IsInningEnd,Starter,Pulled\n')
        file.close()
        for game_pk in season_df.game_pk.unique():
            game_date = season_df[season_df.game_pk == game_pk]['game_date'].max()
            file = open(f'{season} Pitcher Hooks.csv', 'a+')
            plays = None
            while True:
                try:
                    plays = connect(f'{base_url}/{game_pk}/feed/live')['liveData']['plays']['allPlays']
                    break
                except ReadTimeout:
                    continue
                except ConnectTimeout:
                    continue
            pitcher = None
            stat_line = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            score = 0
            outs = 0
            for play in plays:
                if not play['about']['isTopInning']:
                    continue
                if pitcher is None:
                    stat_line[12] = 1
                    pitcher = play['matchup']['pitcher']['id']
                elif play['matchup']['pitcher']['id'] != pitcher:
                    file.write(f'{game_date},{pitcher},{",".join([str(x) for x in stat_line])},1\n')
                    stat_line = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    pitcher = play['matchup']['pitcher']['id']
                else:
                    file.write(f'{game_date},{pitcher},{",".join([str(x) for x in stat_line])},0\n')
                runs_on_play = play['result']['awayScore'] - score
                score += runs_on_play
                stat_line[3] += runs_on_play
                stat_line[0] += 1
                stat_line[1] += play['count']['outs'] - outs
                if play['count']['outs'] == 3:
                    stat_line[10] = 1
                    outs = 0
                else:
                    stat_line[10] = 0
                    outs = play['count']['outs']
                if play['result']['eventType'] in ['single', 'double', 'triple', 'home_run']:
                    stat_line[2] += 1
                    if play['result']['eventType'] == 'home_run':
                        stat_line[7] += 1
                if play['result']['eventType'].startswith('strikeout'):
                    stat_line[4] += 1
                if play['result']['eventType'].endswith('walk'):
                    stat_line[5] += 1
                if play['result']['eventType'] == 'hit_by_pitch':
                    stat_line[6] += 1
                for event in play['playEvents']:
                    if event['isPitch']:
                        if event['details']['isStrike'] or event['details']['isInPlay']:
                            stat_line[8] += 1
                            stat_line[9] += 1
                        else:
                            stat_line[8] += 1
                for runner in play['runners']:
                    if runner.get('earned', False):
                        stat_line[11] += 1
            pitcher = None
            stat_line = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            score = 0
            outs = 0
            for play in plays:
                if play['about']['isTopInning']:
                    continue
                if pitcher is None:
                    stat_line[12] = 1
                    pitcher = play['matchup']['pitcher']['id']
                elif play['matchup']['pitcher']['id'] != pitcher:
                    file.write(f'{game_date},{pitcher},{",".join([str(x) for x in stat_line])},1\n')
                    stat_line = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    pitcher = play['matchup']['pitcher']['id']
                else:
                    file.write(f'{game_date},{pitcher},{",".join([str(x) for x in stat_line])},0\n')
                runs_on_play = play['result']['homeScore'] - score
                score += runs_on_play
                stat_line[3] += runs_on_play
                stat_line[0] += 1
                stat_line[1] += play['count']['outs'] - outs
                if play['count']['outs'] == 3:
                    stat_line[10] = 1
                    outs = 0
                else:
                    stat_line[10] = 0
                    outs = play['count']['outs']
                if play['result']['eventType'] in ['single', 'double', 'triple', 'home_run']:
                    stat_line[2] += 1
                    if play['result']['eventType'] == 'home_run':
                        stat_line[7] += 1
                if play['result']['eventType'].startswith('strikeout'):
                    stat_line[4] += 1
                if play['result']['eventType'].endswith('walk'):
                    stat_line[5] += 1
                if play['result']['eventType'] == 'hit_by_pitch':
                    stat_line[6] += 1
                for event in play['playEvents']:
                    if event['isPitch']:
                        if event['details']['isStrike'] or event['details']['isInPlay']:
                            stat_line[8] += 1
                            stat_line[9] += 1
                        else:
                            stat_line[8] += 1
                for runner in play['runners']:
                    if runner.get('earned', False):
                        stat_line[11] += 1
            file.close()

if __name__ == '__main__':
    os.chdir('..')
    refresh_data()

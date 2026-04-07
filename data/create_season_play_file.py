import requests
import numpy as np
from datetime import datetime
import pandas as pd
import os

def connect(url, params=None):
    resp = None
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

def write_rows(game_pk, season, batter_hand_dict):
    file = open(f'{season}.csv', 'a+')
    game_dict = None
    while True:
        try:
            game_dict = connect(f'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live?hydrate=alignment')
            break
        except requests.exceptions.ConnectTimeout:
            continue
        except requests.exceptions.ReadTimeout:
            continue
    game_date = game_dict['gameData']['datetime']['officialDate']
    home_team_id = game_dict['gameData']['teams']['home']['id']
    home_team = game_dict['gameData']['teams']['home']['abbreviation']
    away_team_id = game_dict['gameData']['teams']['away']['id']
    away_team = game_dict['gameData']['teams']['away']['abbreviation']
    home_manager_id, home_manager_name, away_manager_id, away_manager_name = '' ,'', '', ''
    home_coaches = connect(f'https://statsapi.mlb.com/api/v1/teams/{home_team_id}/coaches?date={game_date}')['roster']
    for coach in home_coaches:
        if coach['jobId'] == 'NTRM':
            home_manager_id = coach['person']['id']
            home_manager_name = coach['person']['fullName']
            break
        if coach['jobId'] == 'MNGR':
            home_manager_id = coach['person']['id']
            home_manager_name = coach['person']['fullName']
    away_coaches = connect(f'https://statsapi.mlb.com/api/v1/teams/{away_team_id}/coaches?date={game_date}')['roster']
    for coach in away_coaches:
        if coach['jobId'] == 'NTRM':
            away_manager_id = coach['person']['id']
            away_manager_name = coach['person']['fullName']
            break
        if coach['jobId'] == 'MNGR':
            away_manager_id = coach['person']['id']
            away_manager_name = coach['person']['fullName']
    venue_id = game_dict['gameData']['venue']['id']
    venue_name = game_dict['gameData']['venue']['name']
    home_score_before, away_score_before = 0, 0
    outs = 0
    prev_half = 'bottom'
    for play in game_dict['liveData']['plays']['allPlays']:
        at_bat_number = play['atBatIndex'] +  1
        inning = play['about']['inning']
        top_bot = play['about']['halfInning']
        if top_bot != prev_half:
            outs = 0
        balls, strikes = 0, 0
        play_description = play['result'].get('description', '').replace(',', '')
        event = ''
        pinch_hitter, pinch_runner, pitcher_sub, defensive_sub = False, False, False, False
        for i in range(len(play['playEvents'])):
            a = 0
            while True:
                if a < len(play['playEvents']) and not play['playEvents'][a]['isPitch']:
                    if play['playEvents'][a]['details'].get('eventType', '') == 'defensive_substitution':
                        defensive_sub = True
                    if play['playEvents'][a]['details'].get('eventType', '') == 'pitching_substitution':
                        pitcher_sub = True
                    if play['playEvents'][a]['details'].get('eventType', '') == 'offensive_substitution':
                        if play['playEvents'][a]['position']['code'] == '11':
                            pinch_hitter = True
                        if play['playEvents'][a]['position']['code'] == '12':
                            pinch_runner = True
                else:
                    break
                a += 1
            if len(play['pitchIndex']) == 0:
                continue
            if not play['playEvents'][i]['isPitch']:
                continue
            pre_play_runner_on_first = play['playEvents'][i]['offense'].get('first', {}).get('id', '')
            pre_play_runner_on_second = play['playEvents'][i]['offense'].get('second', {}).get('id', '')
            pre_play_runner_on_third = play['playEvents'][i]['offense'].get('third', {}).get('id', '')
            sb_attempt_2b, sb_attempt_3b, sb_attempt_home, wild_pitch_passed_ball = False, False, False, False
            sb_success_2b, sb_success_3b, sb_success_home = False, False, False
            a = 1
            while True:
                if i + a < len(play['playEvents']) and not play['playEvents'][i + a]['isPitch']:
                    if play['playEvents'][i + a]['details'].get('eventType', '') in ['stolen_base_2b', 'caught_stealing_2b']:
                        sb_attempt_2b = True
                        if play['playEvents'][i + a]['details'].get('eventType', '') == 'stolen_base_2b':
                            sb_success_2b = True
                    if play['playEvents'][i + a]['details'].get('eventType', '') in ['stolen_base_3b', 'caught_stealing_3b']:
                        sb_attempt_3b = True
                        if play['playEvents'][i + a]['details'].get('eventType', '') == 'stolen_base_3b':
                            sb_success_3b = True
                    if play['playEvents'][i + a]['details'].get('eventType', '') in ['stolen_base_home', 'caught_stealing_home']:
                        sb_attempt_home = True
                        if play['playEvents'][i + a]['details'].get('eventType', '') == 'stolen_base_home':
                            sb_success_home = True
                    if play['playEvents'][i + a]['details'].get('eventType', '') in ['wild_pitch', 'passed_ball']:
                        wild_pitch_passed_ball = True
                else:
                    max_play_index = i + a - 1
                    break
                a += 1
            play_event = play['playEvents'][i]
            if top_bot == 'top':
                bat_score = away_score_before
                field_score = home_score_before
            else:
                bat_score = home_score_before
                field_score = away_score_before
            pitch_number = play_event['pitchNumber']
            pitcher = play_event['defense']['pitcher']['id']
            pitcher_hand = play_event['defense']['pitcher']['pitchHand']['code']
            batter = play_event['offense']['batter']['id']
            bat_side = play_event['offense']['batter']['batSide']['code']
            if batter in batter_hand_dict:
                bat_hand = batter_hand_dict[batter]
            else:
                bat_hand = connect(f"https://statsapi.mlb.com{play_event['offense']['batter']['link']}")['people'][0]['batSide']['code']
                batter_hand_dict[batter] = bat_hand
            catcher = play_event['defense']['catcher']['id']
            first_base = play_event['defense']['first']['id']
            second_base = play_event['defense']['second']['id']
            third_base = play_event['defense']['third']['id']
            shortstop = play_event['defense']['shortstop']['id']
            left_field = play_event['defense']['left']['id']
            center_field = play_event['defense']['center']['id']
            right_field = play_event['defense']['right']['id']
            if play_event['index'] == play['pitchIndex'][-1]:
                event = play['result']['eventType']
            pitch_code = play_event['details']['code']
            pitch_code_description = play_event['details']['description'].replace(',', '')
            pitch_type = play_event['details'].get('type', {'code': ''}).get('code', '')
            pitch_type_description = play_event['details'].get('type', {'description': ''})['description']
            strike_zone_top = play_event['pitchData']['strikeZoneTop']
            strike_zone_bottom = play_event['pitchData']['strikeZoneBottom']
            start_speed = play_event['pitchData'].get('startSpeed', '')
            end_speed = play_event['pitchData'].get('endSpeed', '')
            release_x = play_event['pitchData']['coordinates'].get('x0', '')
            release_y = play_event['pitchData']['coordinates'].get('y0', '')
            release_z = play_event['pitchData']['coordinates'].get('z0', '')
            velocity_x = play_event['pitchData']['coordinates'].get('vX0', '')
            velocity_y = play_event['pitchData']['coordinates'].get('vY0', '')
            velocity_z = play_event['pitchData']['coordinates'].get('vZ0', '')
            acceleration_x = play_event['pitchData']['coordinates'].get('aX', '')
            acceleration_y = play_event['pitchData']['coordinates'].get('aY', '')
            acceleration_z = play_event['pitchData']['coordinates'].get('aZ', '')
            pfx_x = play_event['pitchData']['coordinates'].get('pfxX', '')
            pfx_z = play_event['pitchData']['coordinates'].get('pfxZ', '')
            p_x = play_event['pitchData']['coordinates'].get('pX', '')
            p_z = play_event['pitchData']['coordinates'].get('pZ', '')
            x = play_event['pitchData']['coordinates'].get('x', '')
            y = play_event['pitchData']['coordinates'].get('y', '')
            spin_rate = play_event['pitchData']['breaks'].get('spinRate', '')
            spin_direction = play_event['pitchData']['breaks'].get('spinDirection', '')
            break_angle = play_event['pitchData']['breaks'].get('breakAngle', '')
            break_length = play_event['pitchData']['breaks'].get('breakLength', '')
            break_y = play_event['pitchData']['breaks'].get('breakY', '')
            break_vertical = play_event['pitchData']['breaks'].get('breakVertical', '')
            break_vertical_induced = play_event['pitchData']['breaks'].get('breakVerticalInduced', '')
            break_horizontal = play_event['pitchData']['breaks'].get('breakHorizontal', '')
            zone = play_event['pitchData'].get('zone', '')
            extension = play_event['pitchData'].get('extension', '')
            launch_speed, launch_angle, total_distance, coord_x, coord_y, spray_angle, batted_ball_type, hit_location = '', '', '', '', '', '', '', ''
            if 'hitData' in play_event:
                hit_data = play_event['hitData']
                launch_speed = hit_data.get('launchSpeed', '')
                launch_angle = hit_data.get('launchAngle', '')
                total_distance = hit_data.get('totalDistance', '')
                coord_x = hit_data['coordinates'].get('coordX', '')
                coord_y = hit_data['coordinates'].get('coordY', '')
                if coord_x != '' and coord_y != '':
                    if coord_y == 198.27:
                        if coord_x > 125.42:
                            spray_angle = 90
                        elif coord_x < 125.42:
                            spray_angle = -90
                    else:
                        spray_angle = np.arctan((coord_x - 125.42) / (198.27 - coord_y)) * (180 / np.pi)
                else:
                    spray_angle = ''
                batted_ball_type = hit_data['trajectory']
                hit_location = hit_data.get('location', '')
            runner_on_first_score, runner_on_second_score, runner_on_third_score = False, False, False
            home_score_after, away_score_after = home_score_before, away_score_before
            outs_on_pitch, runs, earned_runs, rbis = 0, 0, 0, 0
            fielded_by, of_assist, fielding_error, dropped_ball = '', '', '', ''
            assist_dict, putout_dict, throwing_error_dict = {}, {}, {}
            assist_tracker, putout_tracker, throwing_error_tracker = 1, 1, 1
            post_play_runner_on_first = pre_play_runner_on_first
            post_play_runner_on_second = pre_play_runner_on_second
            post_play_runner_on_third = pre_play_runner_on_third
            for j in range(len(play['runners'])):
                runner = play['runners'][j]
                if i <= runner['details']['playIndex'] <= max_play_index:
                    for credit in runner.get('credits', []):
                        if fielded_by == "":
                            fielded_by = credit['player']['id']
                        if credit['credit'] == 'f_putout':
                            putout_dict[f'field_putout_{putout_tracker}'] = credit['player']['id']
                            putout_tracker += 1
                        elif credit['credit'] == 'f_assist':
                            assist_dict[f'field_assist_{assist_tracker}'] = credit['player']['id']
                            assist_tracker += 1
                        elif credit['credit'] == 'f_throwing_error':
                            throwing_error_dict[f'throwing_error_{throwing_error_tracker}'] = credit['player']['id']
                            throwing_error_tracker += 1
                        elif credit['credit'] == 'f_assist_of':
                            of_assist = credit['player']['id']
                        elif credit['credit'] == 'f_fielded_ball':
                            fielded_by = credit['player']['id']
                        elif credit['credit'] == 'f_fielding_error':
                            fielding_error = credit['player']['id']
                        elif credit['credit'] == 'f_error_dropped_ball':
                            dropped_ball = credit['player']['id']
                        elif credit['credit'] not in ['f_deflection', 'c_catcher_interf', 'f_touch', 'f_interference']:
                            print(credit['credit'])
                    if runner['movement']['end'] == 'score':
                        if runner["details"]["runner"]["id"] == pre_play_runner_on_first:
                            runner_on_first_score = True
                            if runner['details']['runner']['id'] == post_play_runner_on_first:
                                post_play_runner_on_first = ''
                        if runner["details"]["runner"]["id"] == pre_play_runner_on_second:
                            runner_on_second_score = True
                            if runner['details']['runner']['id'] == post_play_runner_on_second:
                                post_play_runner_on_second = ''
                        if runner["details"]["runner"]["id"] == pre_play_runner_on_third:
                            runner_on_third_score = True
                            if runner['details']['runner']['id'] == post_play_runner_on_third:
                                post_play_runner_on_third = ''
                        runs += 1
                        if runner['details'].get('earned', False):
                            earned_runs += 1
                        if runner.get('rbi', False):
                            rbis += 1
                        if top_bot == 'top':
                            away_score_after += 1
                        else:
                            home_score_after += 1
                    if runner['movement']['end'] == '3B':
                        post_play_runner_on_third = runner['details']['runner']['id']
                        if post_play_runner_on_third == post_play_runner_on_second:
                            post_play_runner_on_second = ''
                        if post_play_runner_on_third == post_play_runner_on_first:
                            post_play_runner_on_first = ''
                    if runner['movement']['end'] == '2B':
                        post_play_runner_on_second = runner['details']['runner']['id']
                        if post_play_runner_on_second == post_play_runner_on_first:
                            post_play_runner_on_first = ''
                    if runner['movement']['end'] == '1B':
                        post_play_runner_on_first = runner['details']['runner']['id']
                    if runner['movement']['isOut']:
                        outs_on_pitch += 1

            file.write(f'{game_pk},{game_date},{venue_id},{venue_name},{home_team_id},{home_team},{away_team_id},'
                       f'{away_team},{home_manager_id},{home_manager_name},{away_manager_id},{away_manager_name},'
                       f'{inning},{top_bot},{at_bat_number},'
                       f'{pitch_number},{pitcher},{pitcher_hand},{batter},{bat_side},{bat_hand},{home_score_before},'
                       f'{away_score_before},{bat_score},{field_score},{balls},{strikes},{outs},{pitch_code},'
                       f'{pitch_code_description},{pitch_type},{pitch_type_description},{event},{strike_zone_top},'
                       f'{strike_zone_bottom},{start_speed},{end_speed},{release_x},{release_y},{release_z},'
                       f'{velocity_x},{velocity_y},{velocity_z},{acceleration_x},{acceleration_y},{acceleration_z},'
                       f'{pfx_x},{pfx_z},{p_x},{p_z},{x},{y},{spin_rate},{spin_direction},{break_angle},{break_length},'
                       f'{break_y},{break_vertical},{break_vertical_induced},{break_horizontal},{zone},{extension},'
                       f'{launch_speed},{launch_angle},{total_distance},{coord_x},{coord_y},{spray_angle},'
                       f'{batted_ball_type},{hit_location},{play_description},{pre_play_runner_on_first},'
                       f'{pre_play_runner_on_second},{pre_play_runner_on_third},{post_play_runner_on_first},'
                       f'{post_play_runner_on_second},{post_play_runner_on_third},{catcher},{first_base},{second_base},'
                       f'{third_base},{shortstop},{left_field},{center_field},{right_field},{runs},{outs_on_pitch},{rbis},'
                       f'{earned_runs},{runner_on_first_score},{runner_on_second_score},{runner_on_third_score},'
                       f'{sb_attempt_2b},{sb_attempt_3b},{sb_attempt_home},{sb_success_2b},'
                       f'{sb_success_3b},{sb_success_home},{wild_pitch_passed_ball},{pinch_hitter},{pinch_runner},'
                       f'{pitcher_sub},{defensive_sub},{fielded_by},{fielding_error},{dropped_ball},{of_assist}')
            for z in range(5):
                if f'field_assist_{z+1}' in assist_dict:
                    file.write(f",{assist_dict[f'field_assist_{z+1}']}")
                else:
                    file.write(",")
            for z in range(3):
                if f'field_putout_{z+1}' in putout_dict:
                    file.write(f",{putout_dict[f'field_putout_{z+1}']}")
                else:
                    file.write(",")
            for z in range(2):
                if f'throwing_error_{z+1}' in throwing_error_dict:
                    file.write(f",{throwing_error_dict[f'throwing_error_{z+1}']}")
                else:
                    file.write(",")
            file.write('\n')
            pinch_hitter = False
            pinch_runner = False
            pitcher_sub = False
            defensive_sub = False
            balls = play_event['count']['balls']
            strikes = play_event['count']['strikes']
            outs = play_event['count']['outs']
            home_score_before = home_score_after
            away_score_before = away_score_after
            prev_half = top_bot
    file.close()
    return batter_hand_dict


base_schedule_url = 'https://statsapi.mlb.com/api/v1/schedule'
def get_plays(start_date, end_date):
    params = {'sportId': 1, 'gameTypes': ['R', 'F', 'D', 'L', 'W', 'C', 'P'], 'startDate': start_date.strftime('%Y-%m-%d'), 'endDate': end_date.strftime('%Y-%m-%d')}
    schedule = connect(f'{base_schedule_url}', params=params)['dates']
    for date in schedule:
        print(f'Adding Plays for {date["date"]}')
        if os.path.exists(f'{date["date"][:4]}.csv'):
            season_df = pd.read_csv(f'{date["date"][:4]}.csv', encoding='cp1252')
        else:
            season_df = None
            with open(f'{date["date"][:4]}.csv', 'w+') as file:
                file.write('game_pk,game_date,venue_id,venue,home_id,home_team,away_id,away_team,home_manager_id,'
                           'home_manager_name,away_manager_id,away_manager_name,inning,inning_topbot,at_bat_number,'
                           'pitch_number,pitcher,p_throws,batter,stand,bat_hand,home_score,away_score,bat_score,'
                           'fld_score,balls,strikes,outs,type,description,pitch_type,pitch_name,events,sz_top,sz_bot,'
                           'release_speed,end_speed,release_pos_x,release_pos_y,release_pos_z,vx0,vy0,vz0,ax,ay,az,'
                           'pfx_x,pfx_z,plate_x,plate_z,x,y,release_spin_rate,spin_axis,break_angle,break_length,'
                           'break_y,break_vertical,break_vertical_induced,break_horizontal,zone,release_extension,'
                           'launch_speed,launch_angle,hit_distance_sc,hc_x,hc_y,spray_angle,bb_type,hit_location,des,'
                           'on_1b,on_2b,on_3b,post_on_1b,post_on_2b,post_on_3b,fielder_2,fielder_3,fielder_4,fielder_5,'
                           'fielder_6,fielder_7,fielder_8,fielder_9,runs_on_pitch,outs_on_pitch,rbis_on_pitch,'
                           'earned_runs_on_pitch,1b_runner_score,2b_runner_score,3b_runner_score,sb_attempt_2b,'
                           'sb_attempt_3b,sb_attempt_home,sb_success_2b,sb_success_3b,sb_success_home,'
                           'passed_ball_wild_pitch,pinch_hitter,pinch_runner,pitcher_sub,defensive_sub,fielded_by,'
                           'fielding_error,dropped_ball,of_assist,field_assist_1,field_assist_2,field_assist_3,'
                           'field_assist_4,field_assist_5,field_putout_1,field_putout_2,field_putout_3,'
                           'throwing_error_1,throwing_error_2\n')
        for game in date['games']:
            if 'rescheduleGameDate' in game or 'resumeGameDate' in game:
                continue
            pk = game['gamePk']
            if season_df is not None and pk in season_df.game_pk.values:
                continue
            write_rows(pk, date["date"][:4], {})

def refresh_plays():
    params = {'sportId': 1, 'gameTypes': ['R', 'F', 'D', 'L', 'W', 'C', 'P']}
    for season in range(2017, datetime.today().year):
        batter_hand_dict = {}
        with open(f'{season}.csv', 'w+') as file:
            file.write('game_pk,game_date,venue_id,venue,home_id,home_team,away_id,away_team,home_manager_id,'
                       'home_manager_name,away_manager_id,away_manager_name,inning,inning_topbot,at_bat_number,'
                       'pitch_number,pitcher,p_throws,batter,stand,bat_hand,home_score,away_score,bat_score,fld_score,'
                       'balls,strikes,outs,type,description,pitch_type,pitch_name,events,sz_top,sz_bot,release_speed,'
                       'end_speed,release_pos_x,release_pos_y,release_pos_z,vx0,vy0,vz0,ax,ay,az,pfx_x,pfx_z,plate_x,'
                       'plate_z,x,y,release_spin_rate,spin_axis,break_angle,break_length,break_y,break_vertical,'
                       'break_vertical_induced,break_horizontal,zone,release_extension,launch_speed,launch_angle,'
                       'hit_distance_sc,hc_x,hc_y,spray_angle,bb_type,hit_location,des,on_1b,on_2b,on_3b,post_on_1b,'
                       'post_on_2b,post_on_3b,fielder_2,fielder_3,fielder_4,fielder_5,fielder_6,fielder_7,fielder_8,'
                       'fielder_9,runs_on_pitch,outs_on_pitch,rbis_on_pitch,earned_runs_on_pitch,1b_runner_score,'
                       '2b_runner_score,3b_runner_score,sb_attempt_2b,sb_attempt_3b,sb_attempt_home,sb_success_2b,'
                       'sb_success_3b,sb_success_home,passed_ball_wild_pitch,pinch_hitter,pinch_runner,pitcher_sub,'
                       'defensive_sub,fielded_by,fielding_error,dropped_ball,of_assist,field_assist_1,field_assist_2,'
                       'field_assist_3,field_assist_4,field_assist_5,field_putout_1,field_putout_2,field_putout_3,'
                       'throwing_error_1,throwing_error_2\n')
        params['season'] = season
        schedule = connect(f'{base_schedule_url}', params)['dates']
        for date in schedule:
            print(f'{date["date"]}')
            for game in date['games']:
                if 'rescheduleGameDate' in game or 'resumeGameDate' in game:
                    continue
                pk = game['gamePk']
                batter_hand_dict = write_rows(pk, season, batter_hand_dict)

if __name__ == '__main__':
    os.chdir('..')
    refresh_plays()

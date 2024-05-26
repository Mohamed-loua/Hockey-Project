import pandas as pd
import numpy as np
import os
import os as path
import json
import requests
import warnings
import copy

warnings.filterwarnings("ignore")

class DataExtractor():
    def __init__(self):
        self.all_games_in_season = None # save the dictionary to access more informations later
        self.all_game_info = None
        self.left_goal_position = np.array([-86,0])
        self.right_goal_position = np.array([86,0])
    
    
    ############################################################################## WE SHOULD USE THE FUNCTION THAT RETURNS A PANDA DATAFRAME
    def get_season_data_for_team(self, year: int, team_id: int) -> dict: 
        season_data = self.get_season_data(year)
        
        team_dict = {}
        for game_id in season_data:
            team_1 = season_data[game_id]['gameData']['teams']['away']['id']
            team_2 = season_data[game_id]['gameData']['teams']['home']['id']
            
            if team_1 == team_id: # could be regrouped
                team_dict[game_id] = season_data[game_id]
                continue
            elif team_2 == team_id:
                team_dict[game_id] = season_data[game_id]
                continue
        return team_dict
    
    
    #function that takes the season to be downloaded and returns a dictionary containing the entirety of the games played during year
    def get_season_data(self, year: int) -> dict:
        with open(f"./hockey/Season{year}{year+1}/season{year}{year+1}.json", 'r') as j:
            entire_season = json.loads(j.read())
        return entire_season
    ############################################################################## WE SHOULD USE THE FUNCTION THAT RETURNS A PANDA DATAFRAME
    
    
    #################################################################################################### ONLY USED IN QUESTION_2.PY
    #Creation of game ID in order to find it in the dictionary
    def build_game_ID(self, game_ID: int, year: int, season_type: int) -> str:
        ID = str(year) + str(season_type).zfill(2) + str(game_ID).zfill(4)
        return ID
    
    #gets the game by looking up its ID in the dictionary which contains all the required data
    def get_game_from_dict(self, year: int, game: int,  season_type: int, entire_season: dict) -> dict :
        ID = self.build_game_ID(game, year, season_type)
        return entire_season[str(ID)]
    
    #get the playoffs games
    def get_game3(self, year: int, type: int, round: int, matchup: int, games: int, entire_season: dict) -> dict :
        game_ID = year*10**6 + 3*10**4 + round*100 + matchup*10 + games
        return entire_season[str(game_ID)]
    
    # gets the specified play (ID) for the game passed as a dictionary
    def get_play_by_ID(self, game : dict, ID : int) -> dict:
        play = game['liveData']['plays']['allPlays'][ID]
        return play
    #################################################################################################### ONLY USED IN QUESTION_2.PY


    #################################################################################################### EXTRACT THE DATASET
    #Get all the play of a season
    def get_season_into_dataframe(self, path_to_file: str) -> pd.DataFrame:
        all_games_in_season = self.get_game_data(path_to_file)
        self.all_games_in_season = all_games_in_season
        df_season = pd.DataFrame()
        
        for game in all_games_in_season:
            game_pk, clean_game = self.clean_single_game_json(all_games_in_season.get(game))
            df_game = self.create_panda_dataframe_for_one_game(game_pk, clean_game)
            df_season = df_season.append(df_game)
            df_season.reset_index(drop=True, inplace=True)
        df_season['about.eventIdx'] = df_season['about.eventIdx'].astype(str).astype(int)
        df_season['coordinates.x'] = df_season['coordinates.x'].astype(str).astype(float)
        df_season['coordinates.y'] = df_season['coordinates.y'].astype(str).astype(float)
        
        return df_season
    
    #Get the dictionary that contains all the regular type play of a season
    def get_season_into_dataframe_more_info(self, all_games_in_season: dict, with_penalty_info: bool = False) -> pd.DataFrame:
        self.all_games_in_season = all_games_in_season
        df_season = pd.DataFrame()
        
        for game in all_games_in_season:   
            try: 
                game_pk, clean_game, penalty_plays_indexes = self.clean_single_game_json_all_event_types(all_games_in_season.get(game))
                
                if with_penalty_info:
                    df_game = self.create_panda_dataframe_for_one_game_more_info(game_pk, clean_game, penalty_plays_indexes)
                else:
                    df_game = self.create_panda_dataframe_for_one_game_more_info(game_pk, clean_game)
                    
                df_season = df_season.append(df_game)
                df_season.reset_index(drop=True, inplace=True)
            except:
                print(f"game {game} has some error")
        df_season['about.eventIdx'] = df_season['about.eventIdx'].astype(str).astype(int)
        df_season['coordinates.x'] = df_season['coordinates.x'].astype(str).astype(float)
        df_season['coordinates.y'] = df_season['coordinates.y'].astype(str).astype(float)
        
        return df_season
    
    #Get the dictionary that contains all the regular type play of a season
    def get_season_into_dataframe_m2(self, all_games_in_season: dict) -> pd.DataFrame:
        self.all_games_in_season = all_games_in_season
        df_season = pd.DataFrame()
        
        for game in all_games_in_season:
            game_pk, clean_game = self.clean_single_game_json(all_games_in_season.get(game))
            df_game = self.create_panda_dataframe_for_one_game(game_pk, clean_game)
            df_season = df_season.append(df_game)
            df_season.reset_index(drop=True, inplace=True)
        df_season['about.eventIdx'] = df_season['about.eventIdx'].astype(str).astype(int)
        df_season['coordinates.x'] = df_season['coordinates.x'].astype(str).astype(float)
        df_season['coordinates.y'] = df_season['coordinates.y'].astype(str).astype(float)
        
        return df_season
    #################################################################################################### EXTRACT THE DATASET
    
    
    # get all shots of one specific team 
    def get_team_shots_from_dataframe(self, df: pd.DataFrame, team_id: int) -> np.array:
        df.rename(columns={
            'team.id': 'teamID', 
            'coordinates.x': 'coordinatesX',
            'coordinates.y': 'coordinatesY'
            }, inplace=True)
        
        df = df[df.teamID == team_id]
        df = df.loc[:, ['coordinatesX', 'coordinatesY']]
        
        df = df.dropna()
        return np.array(df['coordinatesX']), np.array(df['coordinatesY'])
    
    
    # get total time played of one specific team 
    def get_time_played_from_team_season_dataframe(self, df: pd.DataFrame, team_id: int) -> np.array:
        time_played = []
        df.rename(columns={
            'team.id': 'teamID', 
            'coordinates.x': 'coordinatesX',
            'coordinates.y': 'coordinatesY'
            }, inplace=True)
        
        df = df[df.teamID == team_id]
        count_season_games = df['gamePk'].nunique()
        
        for i in range(count_season_games):
            time_played.append(60)
        return np.array(time_played)
    
    
    def get_game_data(self, path_to_file) -> dict:
        file = open(path_to_file, 'r', encoding='utf-8')
        json_str = file.read()
        data_dict = json.loads(json_str)
        file.close()
        return data_dict


    def clean_single_game_json(self, json_dict: dict) -> (str, dict):
        game_pk = json_dict['gamePk']
        live_data = json_dict['liveData']
        shot_ID = 'SHOT'
        goal_ID = 'GOAL'

        play_data = live_data['plays']
        all_plays_data = np.array(play_data['allPlays'])

        #Is empty when a playoff game wasnt played
        if len(all_plays_data) == 0:
            return game_pk, {}
        
        def create_mask(play):
            return True if play['result']['eventTypeId'] == shot_ID or play['result']['eventTypeId'] == goal_ID else False

        vf = np.vectorize(create_mask)
        mask = vf(all_plays_data)

        return game_pk, all_plays_data[mask]
    
    def clean_single_game_json_all_event_types(self, json_dict: dict) -> (str, np.array, np.array):
        game_pk = json_dict['gamePk']
        live_data = json_dict['liveData']

        play_data = live_data['plays']
        all_plays_data = np.array(play_data['allPlays'])
        penalty_plays_indexes = np.array(play_data['penaltyPlays'])

        #Is empty when a playoff game wasnt played
        if len(all_plays_data) == 0:
            return game_pk, {}

        return game_pk, all_plays_data, penalty_plays_indexes
    
    def count(self, row):
        if row['type_of_shot_id'] == 'SHOT':
            return 0
        else:
            return 1

    def count_m2(self, row):
        if row['Is_Goal'] == 0:
            return 0
        else:
            return 1
    
    def compute_distances(self, row):
        coord = np.array([row['coordinates.x'], row['coordinates.y']])
        if row['shooting_on_rink_side'] == 'right':
            return np.linalg.norm(coord - self.right_goal_position)
        else:
            return np.linalg.norm(coord - self.left_goal_position)
        
    def compute_distances_from_last_event(self, row):
        result = None
        if row['last_event_type'] is not None:
            event_coord = np.array([row['coordinates.x'], row['coordinates.y']])
            last_event_coord = np.array([row['last_event_coordinates_x'], row['last_event_coordinates_y']])

            result = np.linalg.norm(event_coord - last_event_coord)
        return result

    def compute_time_from_last_event(self, row):
        result = None
        if row['last_event_type'] is not None:
            event_time = row['about.periodTime']
            last_event_time = row['last_event_time']

            event_time_in_seconds = (int(event_time.split(':')[0]) * 60) + int(event_time.split(':')[1])
            last_event_time_in_seconds = (int(last_event_time.split(':')[0]) * 60) + int(last_event_time.split(':')[1])

            result = int(event_time_in_seconds) - int(last_event_time_in_seconds)
        return result
    
    def compute_angle(self, row):
        coord = np.array([row['coordinates.x'],row['coordinates.y']] )
        if row['shooting_on_rink_side'] == 'right':
            angle = np.arctan2(np.array(row['coordinates.y']), np.array(row['coordinates.x'] - self.right_goal_position[0])) * 180 / np.pi
            if angle < 0:
                return -180 + np.abs(angle)
            else:
                return 180 - np.abs(angle)   
        else:
            return -1 * np.arctan2(np.array(row['coordinates.y']), np.array(row['coordinates.x'] - self.left_goal_position[0])) * 180 / np.pi
        
    def compute_change_in_shot_angle(self, row):
        result = None
        if row['last_event_type'] is not None:
            if row['last_event_type'] == 'SHOT':
                if row['shooting_on_rink_side'] == 'right':
                    last_event_angle = np.arctan2(np.array(row['last_event_coordinates_y']), np.array(row['last_event_coordinates_x'] - self.right_goal_position[0])) * 180 / np.pi
                    if last_event_angle < 0:
                        last_event_angle = -180 + np.abs(last_event_angle)
                    else:
                        last_event_angle = 180 - np.abs(last_event_angle)   
                else:
                    last_event_angle =  -1 * np.arctan2(np.array(row['last_event_coordinates_y']), np.array(row['last_event_coordinates_x'] - self.left_goal_position[0])) * 180 / np.pi
                result = np.abs(row['angle'] - last_event_angle)
            else:
                result = 0
        return result
        
    def compute_speed(self, row):
        result = None
        if row['last_event_type'] is not None:
            distance_from_last_event = row['distance_from_last_event']
            time_from_last_event = row['time_from_last_event']

            result = 0 if time_from_last_event == 0 else distance_from_last_event / time_from_last_event
        return result
    
    def compute_time_since_power_play_started(self, row, game_penalty_array):
        result = None
        row_period = int(row['about.period'])
        row_event_time_in_seconds = (int(row['about.periodTime'].split(':')[0]) * 60) + (int(row['about.periodTime'].split(':')[1]))
        evaluate_penalty = False
        
        for idx, penalty_info in enumerate(game_penalty_array):
            effective_penalty_time = penalty_info['effective_time']
            if effective_penalty_time > 0:
                penalty_period = penalty_info['about']['period']
                penalty_time_in_seconds = (int(penalty_info['about']['periodTime'].split(':')[0]) * 60) + (int(penalty_info['about']['periodTime'].split(':')[1]))
                penalized_team_id = penalty_info['team']['id']
                
                if row_period == penalty_period:
                    if row_event_time_in_seconds > penalty_time_in_seconds:
                        evaluate_penalty = True
                elif row_period == (penalty_period + 1):
                        evaluate_penalty = True
    
                if evaluate_penalty:
                    next_penalty_period = game_penalty_array[idx + 1]['about']['period'] if idx + 1 < game_penalty_array.size else None
                    next_penalty_effective_time = game_penalty_array[idx + 1]['effective_time'] if idx + 1 < game_penalty_array.size else None
                    next_penalized_team_id = game_penalty_array[idx + 1]['team']['id'] if idx + 1 < game_penalty_array.size else None

                    next_penalty_time_in_seconds = None
                    if penalty_period == next_penalty_period:
                        next_penalty_time_in_seconds = (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[0]) * 60) + (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[1])) if idx + 1 < game_penalty_array.size else None
                    elif next_penalty_period == penalty_period + 1:
                        next_penalty_time_in_seconds = (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[0]) * 60) + (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[1])) + (20*60) if idx + 1 < game_penalty_array.size else None

                    if row_period == int(penalty_period):
                        row_event_time_in_seconds = (int(row['about.periodTime'].split(':')[0]) * 60) + (int(row['about.periodTime'].split(':')[1]))
                    elif row_period == int(penalty_period) + 1:
                        row_event_time_in_seconds = (int(row['about.periodTime'].split(':')[0]) * 60) + (int(row['about.periodTime'].split(':')[1])) + (20*60)

                    # if we stack penalties
                    if next_penalty_time_in_seconds is not None and penalty_time_in_seconds + effective_penalty_time > next_penalty_time_in_seconds and penalized_team_id == next_penalized_team_id:
                        if (row_event_time_in_seconds > penalty_time_in_seconds and row_event_time_in_seconds < next_penalty_time_in_seconds) or (row_event_time_in_seconds > next_penalty_time_in_seconds and row_event_time_in_seconds < next_penalty_time_in_seconds + next_penalty_effective_time):
                            return row_event_time_in_seconds - penalty_time_in_seconds
                    else:
                        if row_event_time_in_seconds > penalty_time_in_seconds and row_event_time_in_seconds < penalty_time_in_seconds + effective_penalty_time:
                            return row_event_time_in_seconds - penalty_time_in_seconds
        return result
    
    def compute_number_friendly_skaters_on_the_ice(self, row, game_penalty_array):
        result = None
        row_period = int(row['about.period'])
        row_event_time_in_seconds = (int(row['about.periodTime'].split(':')[0]) * 60) + (int(row['about.periodTime'].split(':')[1]))
        evaluate_penalty = False
        for idx, penalty_info in enumerate(game_penalty_array):
            effective_penalty_time = penalty_info['effective_time']
            if effective_penalty_time > 0:
                penalty_period = penalty_info['about']['period']
                penalty_time_in_seconds = (int(penalty_info['about']['periodTime'].split(':')[0]) * 60) + (int(penalty_info['about']['periodTime'].split(':')[1]))
                penalized_team_id = penalty_info['team']['id']
                
                if row_period == penalty_period:
                    if row_event_time_in_seconds > penalty_time_in_seconds:
                        evaluate_penalty = True
                elif row_period == (penalty_period + 1):
                        evaluate_penalty = True
    
                if evaluate_penalty:
                    next_penalty_period = game_penalty_array[idx + 1]['about']['period'] if idx + 1 < game_penalty_array.size else None
                    next_penalty_effective_time = game_penalty_array[idx + 1]['effective_time'] if idx + 1 < game_penalty_array.size else None
                    next_penalized_team_id = game_penalty_array[idx + 1]['team']['id'] if idx + 1 < game_penalty_array.size else None

                    next_penalty_time_in_seconds = None
                    if penalty_period == next_penalty_period:
                        next_penalty_time_in_seconds = (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[0]) * 60) + (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[1])) if idx + 1 < game_penalty_array.size else None
                    elif next_penalty_period == penalty_period + 1:
                        next_penalty_time_in_seconds = (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[0]) * 60) + (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[1])) + (20*60) if idx + 1 < game_penalty_array.size else None

                    if row_period == penalty_period:
                        row_event_time_in_seconds = (int(row['about.periodTime'].split(':')[0]) * 60) + (int(row['about.periodTime'].split(':')[1]))
                    elif row_period == penalty_period + 1:
                        row_event_time_in_seconds = (int(row['about.periodTime'].split(':')[0]) * 60) + (int(row['about.periodTime'].split(':')[1])) + (20*60)

                    # if we stack penalties
                    if next_penalty_time_in_seconds is not None and penalty_time_in_seconds + effective_penalty_time > next_penalty_time_in_seconds and penalized_team_id == next_penalized_team_id:
                        if (row_event_time_in_seconds > penalty_time_in_seconds and row_event_time_in_seconds < next_penalty_time_in_seconds) or (row_event_time_in_seconds > next_penalty_time_in_seconds and row_event_time_in_seconds < next_penalty_time_in_seconds + next_penalty_effective_time):
                            if penalized_team_id == row['team.id']:
                                return 3
                            else:
                                return 5
                    else:
                        if row_event_time_in_seconds > penalty_time_in_seconds and row_event_time_in_seconds < penalty_time_in_seconds + effective_penalty_time:
                            if penalized_team_id == row['team.id']:
                                return 4
                            else:
                                return 5
        return result
    
    def compute_number_opposing_skaters_on_the_ice(self, row, game_penalty_array):
        result = None
        row_period = int(row['about.period'])
        row_event_time_in_seconds = (int(row['about.periodTime'].split(':')[0]) * 60) + (int(row['about.periodTime'].split(':')[1]))
        evaluate_penalty = False
        for idx, penalty_info in enumerate(game_penalty_array):
            effective_penalty_time = penalty_info['effective_time']
            if effective_penalty_time > 0:
                penalty_period = penalty_info['about']['period']
                penalty_time_in_seconds = (int(penalty_info['about']['periodTime'].split(':')[0]) * 60) + (int(penalty_info['about']['periodTime'].split(':')[1]))
                penalized_team_id = penalty_info['team']['id']
                
                if row_period == penalty_period:
                    if row_event_time_in_seconds > penalty_time_in_seconds:
                        evaluate_penalty = True
                elif row_period == (penalty_period + 1):
                        evaluate_penalty = True
    
                if evaluate_penalty:
                    next_penalty_period = game_penalty_array[idx + 1]['about']['period'] if idx + 1 < game_penalty_array.size else None
                    next_penalty_effective_time = game_penalty_array[idx + 1]['effective_time'] if idx + 1 < game_penalty_array.size else None
                    next_penalized_team_id = game_penalty_array[idx + 1]['team']['id'] if idx + 1 < game_penalty_array.size else None

                    next_penalty_time_in_seconds = None
                    if penalty_period == next_penalty_period:
                        next_penalty_time_in_seconds = (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[0]) * 60) + (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[1])) if idx + 1 < game_penalty_array.size else None
                    elif next_penalty_period == penalty_period + 1:
                        next_penalty_time_in_seconds = (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[0]) * 60) + (int(game_penalty_array[idx + 1]['about']['periodTime'].split(':')[1])) + (20*60) if idx + 1 < game_penalty_array.size else None

                    if row_period == penalty_period:
                        row_event_time_in_seconds = (int(row['about.periodTime'].split(':')[0]) * 60) + (int(row['about.periodTime'].split(':')[1]))
                    elif row_period == penalty_period + 1:
                        row_event_time_in_seconds = (int(row['about.periodTime'].split(':')[0]) * 60) + (int(row['about.periodTime'].split(':')[1])) + (20*60)

                    # if we stack penalties
                    if next_penalty_time_in_seconds is not None and penalty_time_in_seconds + effective_penalty_time > next_penalty_time_in_seconds and penalized_team_id == next_penalized_team_id:
                        if (row_event_time_in_seconds > penalty_time_in_seconds and row_event_time_in_seconds < next_penalty_time_in_seconds) or (row_event_time_in_seconds > next_penalty_time_in_seconds and row_event_time_in_seconds < next_penalty_time_in_seconds + next_penalty_effective_time):
                            if penalized_team_id == row['team.id']:
                                return 5
                            else:
                                return 3
                    else:
                        if row_event_time_in_seconds > penalty_time_in_seconds and row_event_time_in_seconds < penalty_time_in_seconds + effective_penalty_time:
                            if penalized_team_id == row['team.id']:
                                return 5
                            else:
                                return 4
        return result

    #function to add information used to compute the distance
    def distance_helpers(self, row):
        if self.all_games_in_season is not None:
            game = self.all_games_in_season[str(row['ID'])]
        elif self.all_game_info is not None:
            game = self.all_game_info
        else:
            return row
        coord = np.array([row['coordinates.x'],row['coordinates.y']] )
        game_period = game['liveData']['plays']['allPlays'][row['about.eventIdx']]['about']['period']
        team_at_home = game['gameData']['teams']['home']['name']
        # team_away = game['gameData']['teams']['away']['name']
        def away_or_home(row):
            if row['team.name'] == team_at_home:
                return 'home'
            else:
                return 'away'
        
        row['period'] = game_period
        row['away_or_home'] = away_or_home(row)
        try: 
            if game_period == 5:
                leftSide = np.linalg.norm(coord - self.left_goal_position)
                rightSide = np.linalg.norm(coord - self.right_goal_position) 
                if np.minimum(leftSide, rightSide) == rightSide:
                    row['shooting_on_rink_side'] = 'right'
                else :
                    row['shooting_on_rink_side'] = 'left'
            else:
                own_rink_side = game['liveData']['linescore']['periods'][game_period-1][row['away_or_home']]['rinkSide']
                if (own_rink_side == 'right'):
                    row['shooting_on_rink_side'] = 'left'
                else:
                    row['shooting_on_rink_side'] = 'right'
        except KeyError as e:
            row['shooting_on_rink_side'] = None
           
        return row
    
    
    #Added the column about.eventIdx
    def __generate_dataframe_column_names(self)-> list:
        return ['about.period', 'about.periodTime', 'about.eventId','about.eventIdx', 'team.name', 'team.id', 'result.eventTypeId', 'coordinates.x', 'coordinates.y', 'players.0.player.fullName', 'players.1.player.fullName', 'result.secondaryType', 'result.strength.code', 'result.emptyNet']

    def create_panda_dataframe_for_one_game(self, game_pk: str, all_play_data: dict) -> pd.DataFrame:
        if len(all_play_data) == 0:
            return None
        rows_dict = []

        for play_data in all_play_data:
            rows_dict.append(self.__extract_play_data_from_dict(game_pk, play_data))

        df = pd.DataFrame(rows_dict)
        return df
    
    def create_panda_dataframe_for_one_game_more_info(self, game_pk: str, all_play_data: np.array, penalty_plays_indexes: np.array = None) -> pd.DataFrame:
        if len(all_play_data) == 0:
            return None
        rows_dict = []
        previous_event_row = None
        game_penalty_array = None
        
        if penalty_plays_indexes is not None:            
            game_penalty_array = self.extract_penalty_plays_full_data_from_dict(game_pk, all_play_data, penalty_plays_indexes)
        
        for play_data in all_play_data:
            row_dict = self.__extract_play_data_from_dict(game_pk, play_data)
            if penalty_plays_indexes is not None: 
                row_dict['time_since_power_play_started'] = self.compute_time_since_power_play_started(row_dict, game_penalty_array)
                row_dict['num_friendly_skaters'] = self.compute_number_friendly_skaters_on_the_ice(row_dict, game_penalty_array)
                row_dict['num_opposing_skaters'] = self.compute_number_opposing_skaters_on_the_ice(row_dict, game_penalty_array)
            
            if previous_event_row is not None:
                if row_dict['result.eventTypeId'] == 'SHOT' or row_dict['result.eventTypeId'] == 'GOAL':
                    row_dict['last_event_type'] = previous_event_row['result.eventTypeId']
                    row_dict['last_event_coordinates_x'] = previous_event_row['coordinates.x']
                    row_dict['last_event_coordinates_y'] = previous_event_row['coordinates.y']
                    row_dict['last_event_time'] = previous_event_row['about.periodTime']
                    rows_dict.append(row_dict)
            else:
                if row_dict['result.eventTypeId'] == 'SHOT' or row_dict['result.eventTypeId'] == 'GOAL':
                    row_dict['last_event_type'] = None
                    row_dict['last_event_coordinates_x'] = None
                    row_dict['last_event_coordinates_y'] = None
                    row_dict['last_event_time'] = None
                    rows_dict.append(row_dict)
            
            previous_event_row = row_dict

        df = pd.DataFrame(rows_dict)
        df['about.eventIdx'] = df['about.eventIdx'].astype(str).astype(int)
        df['coordinates.x'] = df['coordinates.x'].astype(str).astype(float)
        df['coordinates.y'] = df['coordinates.y'].astype(str).astype(float)
        return df
        
    
    def __extract_play_data_from_dict(self, game_pk: str, full_play_data: dict) -> dict:
        def extract_path_from_column_name(column_name: str) -> list:
            return column_name.split(".")

        def extract_value_from_path(path: list, full_play_data: dict):
            result = full_play_data
            for i in range(len(path)):
                try:
                    if path[i] == str(0):
                        result = result[0]
                    elif path[i] == str(1):
                        result = result[len(result) - 1]
                    else:
                        result = result[path[i]]
                except:
                    return None
            return result

        new_dict = {}

        for column in self.__generate_dataframe_column_names():
            path = extract_path_from_column_name(column)
            value = extract_value_from_path(path, full_play_data)
            new_dict[column] = value
            
        new_dict['ID'] = game_pk # Added the game ID to each play to access informations easily
        new_dict['gamePk'] = game_pk
        return new_dict
    
    def extract_penalty_plays_full_data_from_dict(self, game_pk: str, full_play_data: dict, penalty_plays_indexes) -> np.array:
        penalty_array = np.array([])
        next_penalty_is_penalty_of_same_team_that_overlap = False
        skip_next_index = False
        
        def calculate_time_in_seconds_between_penalty_and_event(penalty_info, next_event_period, next_event_period_time) -> float:
            penalty_period = penalty_info['about']['period']
            penalty_time_in_seconds = (int(penalty_info['about']['periodTime'].split(':')[0]) * 60) + (int(penalty_info['about']['periodTime'].split(':')[1]))
            next_event_time_in_seconds = (int(next_event_period_time.split(':')[0]) * 60) + (int(next_event_period_time.split(':')[1]))
        
            return next_event_time_in_seconds - penalty_time_in_seconds if penalty_period == next_event_period else next_event_time_in_seconds + 1200 - penalty_time_in_seconds
    
        for index in penalty_plays_indexes:
            if skip_next_index:
                skip_next_index = False
            else:
                penalties_info = []
                initial_info = full_play_data[index]
                if int(initial_info['result']['penaltyMinutes']) == 4:
                    initial_info['result']['penaltyMinutes'] = 2
                    duplicate_info = copy.deepcopy(initial_info)

                    initial_info_time_in_seconds = (int(initial_info['about']['periodTime'].split(':')[0]) * 60) + (int(initial_info['about']['periodTime'].split(':')[1]))
                    if initial_info_time_in_seconds > (18*60):
                        duplicate_info['about']['period'] = int(duplicate_info['about']['period']) + 1
                        duplicate_info['about']['periodTime'] = str(int(initial_info['about']['periodTime'].split(':')[0]) + 2 - 20) + ':' + str(int(initial_info['about']['periodTime'].split(':')[1]))
                    penalties_info.append(initial_info)
                    penalties_info.append(duplicate_info)
                else:
                    penalties_info = [full_play_data[index]]

                for penalty_info in penalties_info:
                    penalty_effective_time_in_seconds = 0
                    penalty_minutes_in_seconds = penalty_info['result']['penaltyMinutes'] * 60
                    penalized_team_id = penalty_info['team']['id']

                    # if the penalty is 5 minutes
                    if int(penalty_info['result']['penaltyMinutes']) == 5:
                        # if the next event is also a 5 minutes penalty, then it was a fight, they cancel each other
                        if (index + 1 < len(full_play_data)):
                            next_event = full_play_data[index + 1]
                            if next_event['result']['eventTypeId'] == 'PENALTY' and next_event['result']['penaltyMinutes'] == 5:
                                skip_next_index = True
                                continue

                        penalty_info['effective_time'] = 5*60
                        penalty_array = np.append(penalty_array, penalty_info)

                    elif int(penalty_info['result']['penaltyMinutes']) == 2:
                        i = copy.deepcopy(index) + 1
                        continue_while = True

                        while continue_while and i < len(full_play_data):
                            next_event = full_play_data[i]

                            if ('team' in next_event.keys() and 'about' in next_event.keys()):
                                next_event_type = next_event['result']['eventTypeId']
                                next_event_team_id = next_event['team']['id']
                                next_event_period = next_event['about']['period']
                                next_event_period_time = next_event['about']['periodTime']
                                time_between_penalty_and_event = calculate_time_in_seconds_between_penalty_and_event(penalty_info, next_event_period, next_event_period_time)

                                # if the next event is a penalty at the same time
                                if next_event_type == 'PENALTY' and next_event_team_id != penalized_team_id and penalty_info['about']['period'] == next_event_period and penalty_info['about']['periodTime'] == next_event_period_time:
                                    skip_next_index = True
                                    continue_while = False

                                # if the next event if after the end of the penalty, the penalty has ended
                                elif time_between_penalty_and_event > penalty_minutes_in_seconds:
                                    penalty_effective_time_in_seconds = penalty_minutes_in_seconds
                                    continue_while = False

                                # if the next event is a goal from the opposing team, the penalty has ended
                                elif next_event_type == 'GOAL' and next_event_team_id != penalized_team_id:
                                    if not next_penalty_is_penalty_of_same_team_that_overlap:
                                        penalty_effective_time_in_seconds = time_between_penalty_and_event
                                        continue_while = False
                                    else:
                                        next_penalty_is_penalty_of_same_team_that_overlap = False

                                # if the next event if a penalty from the same team, keep that in memory
                                elif next_event_type == 'PENALTY' and next_event_team_id == penalized_team_id:
                                    next_penalty_is_penalty_of_same_team_that_overlap = True

                                # if the next event if a penalty from the opposing team, the penalty has effectively ended (no more in minority)
                                elif next_event_type == 'PENALTY' and next_event_team_id != penalized_team_id:
                                    if not next_penalty_is_penalty_of_same_team_that_overlap:
                                        penalty_effective_time_in_seconds = time_between_penalty_and_event
                                        continue_while = False
                                    else:
                                        next_penalty_is_penalty_of_same_team_that_overlap = False
                            i += 1

                        penalty_info['effective_time'] = penalty_effective_time_in_seconds
                        penalty_array = np.append(penalty_array, penalty_info)
        return penalty_array

    # get the training data i.e. regular season from year 2015 to 2018
    def get_training_data(self, years: list):
        dicts = dict()
        for year in years:
            all_season_data = self.get_season_data(year)
            dicts[str(year)] = self.get_regular(year, all_season_data)
        return dicts

    # get all regular season games for one year
    def get_regular(self, year: int, all_season_data: dict):
        regular_season_data = dict()
        # select the data for regular's games
        threshold = year * 10 ** 6 + 30000
        for game_id in all_season_data.keys():
            if int(game_id) < threshold:
                regular_season_data[game_id] = all_season_data[game_id]
            else:
                break
        return regular_season_data

    def get_distance_angle_goal_dataframe(self, dicts: dict) -> pd.DataFrame:
        df_tidy = pd.DataFrame()
        for year in dicts:
            print(year)
            # initialize the dataframe, reduce the memory avoid overflow
            df_tidy_one = pd.DataFrame()
            train_one_year = pd.DataFrame()

            # creating dataframe for a specific year
            train_one_year = self.get_season_into_dataframe_m2(dicts[year])
            # create rink side column helps calculating angle and distance
            train_one_year = train_one_year.apply(self.distance_helpers, axis=1)

            # calculate the distance between the shot and rink(considering the goal as a point, i.e. (86,0)or(-86,0))
            train_one_year['distances'] = train_one_year.apply(self.compute_distances, axis=1)

            # angle between two points(vector), considering the rink side.
            train_one_year['angle'] = train_one_year.apply(self.compute_angle, axis=1)

            # Is goal: 1 for goal, 0 for non goal
            train_one_year['IsGoal'] = [train_one_year['result.eventTypeId'] == 'GOAL'][0].astype(int)

            # Empty Net: 1 for True
            train_one_year['Empty_Net'] = [train_one_year['result.emptyNet'] == True][0].astype(int)

            # tidy data with 4 colomnes as requested
            df_tidy_one = pd.DataFrame({'distances': train_one_year['distances'],
                                        'angle': train_one_year['angle'], 'Is_Goal': train_one_year['IsGoal'],
                                        'Empty_Net': train_one_year['Empty_Net']})
            df_tidy = df_tidy.append(df_tidy_one, ignore_index=True)
        return df_tidy
    def get_distance_angle_is_goal_dataframe(self, dicts: dict) -> pd.DataFrame:
        df_tidy = pd.DataFrame()
        for year in dicts:
            print(year)
            # initialize the dataframe, reduce the memory avoid overflow
            df_tidy_one = pd.DataFrame()
            train_one_year = pd.DataFrame()

            # creating dataframe for a specific year
            train_one_year = self.get_season_into_dataframe_m2(dicts[year])
            # create rink side column helps calculating angle and distance
            train_one_year = train_one_year.apply(self.distance_helpers, axis=1)

            # calculate the distance between the shot and rink(considering the goal as a point, i.e. (86,0)or(-86,0))
            train_one_year['distances'] = train_one_year.apply(self.compute_distances, axis=1)

            # angle between two points(vector), considering the rink side.
            train_one_year['angle'] = train_one_year.apply(self.compute_angle, axis=1)

            # Is goal: 1 for goal, 0 for non goal
            train_one_year['is_goal'] = [train_one_year['result.eventTypeId'] == 'GOAL'][0].astype(int)

            # Empty Net: 1 for True
            train_one_year['Empty_Net'] = [train_one_year['result.emptyNet'] == True][0].astype(int)

            # tidy data with 4 colomnes as requested
            df_tidy_one = pd.DataFrame({'distances': train_one_year['distances'],
                                        'angle': train_one_year['angle'], 'is_goal': train_one_year['is_goal'],
                                        'Empty_Net': train_one_year['Empty_Net']})
            df_tidy = df_tidy.append(df_tidy_one, ignore_index=True)
        return df_tidy
    def get_extra_info_dataframe(self, dicts: dict):
        df_complete = pd.DataFrame()
        for year in dicts:
            print(year)
            # initialize the dataframe, reduce the memory avoid overflow
            df_complete_one = pd.DataFrame()
            train_one_year = pd.DataFrame()

            # creating dataframe for a specific year
            train_one_year = self.get_season_into_dataframe_more_info_corrected(dicts[year], False)

            # create rink side column helps calculating angle and distance
            train_one_year = train_one_year.apply(self.distance_helpers, axis=1)

            # calculate the distance between the shot and rink(considering the goal as a point, i.e. (86,0)or(-86,0))
            train_one_year['distances'] = train_one_year.apply(self.compute_distances, axis=1)

            # angle between two points(vector), considering the rink side.
            train_one_year['angle'] = train_one_year.apply(self.compute_angle, axis=1)

            # calculate the distance between the event and the previous event
            train_one_year['distance_from_last_event'] = train_one_year.apply(self.compute_distances_from_last_event,
                                                                              axis=1)

            # calculate the time between the event and the previous event
            train_one_year['time_from_last_event'] = train_one_year.apply(self.compute_time_from_last_event, axis=1)

            # calculate the change in shot angle between the event and the previous event if rebound
            train_one_year['change_in_shot_angle'] = train_one_year.apply(self.compute_change_in_shot_angle, axis=1)

            # calculate the speed as the distance from the previous event divided by the time since the previous event
            train_one_year['speed'] = train_one_year.apply(self.compute_speed, axis=1)

            # Is goal: 1 for goal, 0 for non goal
            train_one_year['IsGoal'] = [train_one_year['result.eventTypeId'] == 'GOAL'][0].astype(int)

            # Empty Net: 1 for True
            train_one_year['Empty_Net'] = [train_one_year['result.emptyNet'] == True][0].astype(int)

            # Is rebound: 1 for True
            train_one_year['is_rebound'] = [train_one_year['last_event_type'] == 'SHOT'][0].astype(int)

            # complete data with 7 columns as requested
            df_complete_one = pd.DataFrame({'game_seconds': train_one_year['about.periodTime'],
                                            'game_period': train_one_year['period'],
                                            'coordinates_x': train_one_year['coordinates.x'],
                                            'coordinates_y': train_one_year['coordinates.y'],
                                            'shot_distance': train_one_year['distances'],
                                            'shot_angle': train_one_year['angle'],
                                            'shot_type': train_one_year['result.secondaryType'],
                                            'last_event_type': train_one_year['last_event_type'],
                                            'last_event_coordinates_x': train_one_year['last_event_coordinates_x'],
                                            'last_event_coordinates_y': train_one_year['last_event_coordinates_y'],
                                            'time_from_last_event': train_one_year['time_from_last_event'],
                                            'distance_from_last_event': train_one_year['distance_from_last_event'],
                                            'is_rebound': train_one_year['is_rebound'],
                                            'change_in_shot_angle': train_one_year['change_in_shot_angle'],
                                            'speed': train_one_year['speed'],
                                            'is_goal': train_one_year['IsGoal']})
            df_complete = df_complete.append(df_complete_one, ignore_index=True)
        return df_complete

    def get_season_into_dataframe_more_info_corrected(self, all_games_in_season: dict,
                                            with_penalty_info: bool = False) -> pd.DataFrame:
        self.all_games_in_season = all_games_in_season
        df_season = pd.DataFrame()

        for game in all_games_in_season:
            try:
                game_pk, clean_game, penalty_plays_indexes = self.clean_single_game_json_all_event_types(
                    all_games_in_season.get(game))

                if with_penalty_info:
                    df_game = self.create_panda_dataframe_for_one_game_more_info(game_pk, clean_game, penalty_plays_indexes)
                else:
                    df_game = self.create_panda_dataframe_for_one_game_more_info(game_pk, clean_game)

                df_season = df_season.append(df_game)
                df_season.reset_index(drop=True, inplace=True)
            except:
                print(f"game {game} has some error")
        df_season['about.eventIdx'] = df_season['about.eventIdx'].astype(str).astype(int)
        df_season['coordinates.x'] = df_season['coordinates.x'].astype(str).astype(float)
        df_season['coordinates.y'] = df_season['coordinates.y'].astype(str).astype(float)

        return df_season

    def compute_all_extra_info_for_game(self, df_game: pd.DataFrame, all_game_info: None) -> pd.DataFrame:
        if all_game_info is not None: 
            self.all_game_info = all_game_info
        
        #create rink side column helps calculating angle and distance
        df_game = df_game.apply(self.distance_helpers, axis=1)

        #calculate the distance between the shot and rink(considering the goal as a point, i.e. (86,0)or(-86,0))
        df_game['distances'] = df_game.apply(self.compute_distances, axis=1)

        #angle between two points(vector), considering the rink side.
        df_game['angle'] = df_game.apply(self.compute_angle, axis=1)

        #calculate the distance between the event and the previous event
        df_game['distance_from_last_event'] = df_game.apply(self.compute_distances_from_last_event, axis=1)

        #calculate the time between the event and the previous event
        df_game['time_from_last_event'] = df_game.apply(self.compute_time_from_last_event, axis=1)

        #calculate the change in shot angle between the event and the previous event if rebound
        df_game['change_in_shot_angle'] = df_game.apply(self.compute_change_in_shot_angle, axis=1)

        #calculate the speed as the distance from the previous event divided by the time since the previous event
        df_game['speed'] = df_game.apply(self.compute_speed, axis=1)

        #Is goal: 1 for goal, 0 for non goal
        df_game['IsGoal'] = [df_game['result.eventTypeId']=='GOAL'][0].astype(int)

        # Empty Net: 1 for True
        df_game['Empty_Net']=[df_game['result.emptyNet']==True][0].astype(int)

        # Is rebound: 1 for True
        df_game['is_rebound']=[df_game['last_event_type']=='SHOT'][0].astype(int)

        #complete data as requested
        df_complete_game = pd.DataFrame({'game_pk': df_game['gamePk'],
                                    'team_name': df_game['team.name'],
                                    'game_seconds': df_game['about.periodTime'],
                                    'game_period': df_game['period'],
                                    'event_idx': df_game['about.eventIdx'],
                                    'team_id': df_game['team.id'],
                                    'coordinates_x': df_game['coordinates.x'],
                                    'coordinates_y': df_game['coordinates.y'],
                                    'shot_distance': df_game['distances'],
                                    'shot_angle': df_game['angle'],
                                    'shot_type': df_game['result.secondaryType'],
                                    'last_event_type': df_game['last_event_type'],
                                    'last_event_coordinates_x': df_game['last_event_coordinates_x'],
                                    'last_event_coordinates_y': df_game['last_event_coordinates_y'],
                                    'time_from_last_event': df_game['time_from_last_event'],
                                    'distance_from_last_event': df_game['distance_from_last_event'],
                                    'time_since_power_play_started': df_game['time_since_power_play_started'],
                                    'num_friendly_skaters': df_game['num_friendly_skaters'],
                                    'num_opposing_skaters': df_game['num_opposing_skaters'],
                                    'is_rebound': df_game['is_rebound'],
                                    'change_in_shot_angle': df_game['change_in_shot_angle'],
                                    'speed': df_game['speed'],
                                     'Is_Goal': df_game['IsGoal']})
        return df_complete_game
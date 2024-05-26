import json
import requests
import numpy as np
import pandas as pd
import logging
import requests
from utils.data_extractor import DataExtractor
from serving_client import ServingClient
from sklearn import preprocessing


logger = logging.getLogger(__name__)


class GameClient:
    def __init__(self, ip = '0.0.0.0', port = '7999'):
        self.data_extractor = DataExtractor()
        self.serving_client = ServingClient(ip, port)
        self.last_processed_event_idx = 0
        self.team_name_away = ''
        self.team_name_home = ''
        self.predicted_goals_for_away_team = 0
        self.predicted_goals_for_home_team = 0
        self.team_away_current_score = 0
        self.team_home_current_score = 0

    def _download_play_by_play_for_game_id(self, game_id: int):
        """
        
        The method will fetch a html page at the follwing url https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/ which contains
        all the information about a hockey game.

        Args:
            game_id: The game id to be fetched

        Returns: A json string which contains information about a specific Hockey game

        """
        try:
            json_play_by_play = requests.get(f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/')
            print(f"The request for game_id {game_id} returned a {json_play_by_play.status_code}")
            if json_play_by_play.status_code != 200:
                print(f'dowload for game_id : {game_id} failed, return code was {json_play_by_play.status_code}')
            return json.loads(json_play_by_play.text), json_play_by_play.status_code
        except Exception as error:
            print(f'Error for game_id {game_id}')
            print(error)
            return '', 404
    
    def _download_play_by_play_for_game_id_diffPatch(self, game_id: int, yyyymmdd_hhmmss: str):
        """
        
        The method will fetch a html page at the follwing url https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/ which contains
        all the information about a hockey game.

        Args:
            game_id: The game id to be fetched

        Returns: A json string which contains information about a specific Hockey game

        """
        try:
            # 'diffPatch?startTimecode='+str(yyyymmdd_hhmmss)
            json_play_by_play = requests.get(f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/diffPatch?startTimecode={yyyymmdd_hhmmss}')
            print(f"The request for game_id {game_id} returned a {json_play_by_play.status_code}")
            if json_play_by_play.status_code != 200:
                print(f'dowload for game_id : {game_id} failed, return code was {json_play_by_play.status_code}')
            return json.loads(json_play_by_play.text), json_play_by_play.status_code
        except Exception as error:
            print(f'Error for game_id {game_id}')
            print(error)
            return '', 404

    def _compute_game_seconds_from_str(self, row):
        return (int(row['game_seconds'].split(':')[0]) * 60) + (int(row['game_seconds'].split(':')[1]))

    def _compute_dataframe(self, play_by_play):
        game_pk, clean_game, penalty_plays_indexes = self.data_extractor.clean_single_game_json_all_event_types(play_by_play)
        df_game = self.data_extractor.create_panda_dataframe_for_one_game_more_info(game_pk, clean_game, penalty_plays_indexes)
        df_game = self.data_extractor.compute_all_extra_info_for_game(df_game, play_by_play)

        df_game['game_seconds_as_num'] = df_game.apply(self._compute_game_seconds_from_str, axis=1)
        le = preprocessing.LabelEncoder()
        le.fit(df_game['shot_type'].unique())
        df_game['shot_types_as_num'] = le.transform(df_game['shot_type'])

        df_game['time_since_power_play_started'] = df_game['time_since_power_play_started'].fillna(0)
        df_game['num_friendly_skaters'] = df_game['num_friendly_skaters'].fillna(5)
        df_game['num_opposing_skaters'] = df_game['num_opposing_skaters'].fillna(5)

        self._get_current_score(df_game)

        df_new_changes = df_game[df_game['event_idx'] > self.last_processed_event_idx].reset_index()
        if not df_new_changes.empty:
            self.last_processed_event_idx = df_new_changes['event_idx'].max()
        return df_new_changes

    def _set_team_name_away_home(self, play:dict) -> list:
        self.team_name_away = play['gameData']['teams']['away']['name']
        self.team_name_home = play['gameData']['teams']['home']['name']

    def _add_predicted_goals(self, df: pd.DataFrame):
        away_df = df[df['team_name'] == self.team_name_away]
        home_df = df[df['team_name'] == self.team_name_home]

        self.predicted_goals_for_away_team += away_df['predicted_goal'].sum()
        self.predicted_goals_for_home_team += home_df['predicted_goal'].sum()

    def _get_current_score(self, df: pd.DataFrame):
        away_df = df[df['team_name'] == self.team_name_away]
        home_df = df[df['team_name'] == self.team_name_home]

        self.team_away_current_score = away_df['Is_Goal'].sum()
        self.team_home_current_score = home_df['Is_Goal'].sum()

    def get_unseen_live_data_for_game_id(self, game_id: str = '2021020329') -> pd.DataFrame:
        play_by_play, status_code = self._download_play_by_play_for_game_id(game_id)
        self._set_team_name_away_home(play_by_play)
        
        if status_code == 200:
            df_new_changes = self._compute_dataframe(play_by_play)
            print(df_new_changes.shape)
            if df_new_changes.shape[0] == 0:
                return df_new_changes, self.team_name_away, self.team_name_home, self.predicted_goals_for_away_team, self.predicted_goals_for_home_team, self.team_away_current_score, self.team_home_current_score
            df_new_changes = self.serving_client.predict(df_new_changes)

            self._add_predicted_goals(df_new_changes)
        else:
            logger.error(f"Unable to contact the NHL API, the return code was: {status_code}")
        return df_new_changes, self.team_name_away, self.team_name_home, self.predicted_goals_for_away_team, self.predicted_goals_for_home_team, self.team_away_current_score, self.team_home_current_score

    def get_unseen_live_data_for_game_id_with_diffpatch(self, game_id: str, yyyymmdd_hhmmss:str) -> pd.DataFrame:
        play_by_play, status_code = self._download_play_by_play_for_game_id_diffPatch(game_id, yyyymmdd_hhmmss)
        self._set_team_name_away_home(play_by_play)

        if status_code == 200:
            df_new_changes = self._compute_dataframe(play_by_play)
            df_new_changes = self.serving_client.predict(df_new_changes)

            self._add_predicted_goals(df_new_changes)
        else:
            logger.error(f"Unable to contact the NHL API, the return code was: {status_code}")
        return df_new_changes, self.team_name_away, self.team_name_home, self.predicted_goals_for_away_team, self.predicted_goals_for_home_team, self.team_away_current_score, self.team_home_current_score

    def get_last_processed_event_idx(self):
        return self.last_processed_event_idx
    
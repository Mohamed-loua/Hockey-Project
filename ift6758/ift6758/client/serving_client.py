import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 7999, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        
        data = []
        X['predicted_goal'] = 0
        X['predicted_proba'] = 0
        try:
            for index, row in X.iterrows():
                game_seconds_as_num = round(row['game_seconds_as_num'], 2)
                game_period = round(row['game_period'], 2)
                coordinates_x = round(row['coordinates_x'], 2)
                coordinates_y = round(row['coordinates_y'], 2)
                shot_types_as_num = round(row['shot_types_as_num'], 2)
                shot_distance = round(row['shot_distance'], 2)
                shot_angle = round(row['shot_angle'], 2)
                is_rebound = round(row['is_rebound'], 2)
                change_in_shot_angle = round(row['change_in_shot_angle'], 2)
                speed = round(row['speed'], 2)
                time_since_power_play_started = round(row['time_since_power_play_started'], 2)
                num_friendly_skaters = round(row['num_friendly_skaters'], 2)
                num_opposing_skaters = round(row['num_opposing_skaters'], 2)

                array_to_predict = [game_seconds_as_num, game_period, coordinates_x, coordinates_y, shot_types_as_num, shot_distance, shot_angle, is_rebound, change_in_shot_angle, speed, time_since_power_play_started, num_friendly_skaters, num_opposing_skaters]
                data.append(array_to_predict)
            
            url = self.base_url + '/predict'
            data = {'Array': data}
            r = requests.post(url, json=data)
            predictions = r.json()['Predictions']

            for index, row in X.iterrows():
                X.loc[index, 'predicted_goal'] = 1 if predictions[index] > 0.5 else 0
                X.loc[index, 'predicted_proba'] = predictions[index]

            return X

        except Exception as e:
            raise e

    def logs(self) -> list:
        """Get server logs"""

        try:
            url = self.base_url + '/logs'
            r = requests.get(url)
            if r.status_code != 200:
                logger.info(f'the return code was {r.status_code}')
            return r.text
        except Exception as e:
            print(e)

    def download_registry_model(self, workspace: str, model: str, version: str, file: str = '') -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        try:
            data = {
                'workspace': workspace,
                'model': model,
                'version': version,
                'file': file
            }

            url = self.base_url + '/download_registry_model'
            r = requests.post(url, json=data)
            if r.status_code != 200:
                logger.info(f'the return code was {r.status_code}')
        except Exception as e:
            print(e)

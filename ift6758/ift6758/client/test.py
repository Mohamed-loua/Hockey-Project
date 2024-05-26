from serving_client import ServingClient
from game_client import GameClient

game_client = GameClient()
serving_client = ServingClient()

serving_client.download_registry_model("ift-6758-team-7", "xgboostwithrandomizedsearchcv", "1.0.2", "XGBoostWithRandomizedSearchCV.json")

game_info = game_client.get_unseen_live_data_for_game_id('2022020510')
df_diff = game_info[0]
print(df_diff.shape[0])
game_info = game_client.get_unseen_live_data_for_game_id('2022020510')
df_diff = game_info[0]
print(df_diff.shape[0])

logs = serving_client.logs()
# df_diff = game_client.get_unseen_live_data_for_game_id_with_diffpatch('2021020329', '20211128_183000' )

# https://statsapi.web.nhl.com/api/v1/game/2021020329/feed/live/
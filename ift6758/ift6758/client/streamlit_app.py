import streamlit as st
import pandas as pd
import numpy as np
from serving_client import ServingClient
from game_client import GameClient
import os

st.title("Hockey Visualization App")
IP = os.environ.get('IP')
PORT = os.environ.get('PORT')
serving_client = ServingClient(IP, PORT)
#global variable
game_client = GameClient(IP, PORT)

def set_game_client(game_id):
    global game_client
    try:
        if st.session_state['game_id'] == game_id:
            game_client = st.session_state['game_client']
        else:
            st.session_state['game_client'] = game_client
            st.session_state['game_id'] = game_id
    except:
        st.session_state['game_client'] = game_client
        st.session_state['game_id'] = game_id

def button_get_model(workspace, model, version, file):
    print('A new model will be dowloaded')
    try:
        serving_client.download_registry_model(workspace, model, version, file)
        return True
    except Exception as e:
        print(e)
    return False

def get_time_left(time_of_event:str) -> str:
    time = time_of_event.split(":")
    minute = int(time[0])
    second = int(time[1])
    if minute == 0 and second == 0:
        return '00:00'
    minutes_left = 20 - minute
    seconds_left = 60 - second
    return str(minutes_left).rjust(2, '0')+':'+str(seconds_left).rjust(2, '0')

def get_score_difference(expected, current):
    difference = expected-current
    if difference >= 0:
        return True, difference
    else:
        return False, difference

def get_game_info_by_game_id(game_id):
    global game_client
    data = game_client.get_unseen_live_data_for_game_id(game_id)
    return data

def set_session_information(game_info):
    df = game_info[0]
    try:
        session_df = st.session_state['dataframe_event']
    except:
        session_df = pd.DataFrame()
    new_df = pd.concat([session_df,df], ignore_index=True)
    st.session_state['dataframe_event'] = new_df
    st.session_state['game_info'] = [game_info[1], game_info[2], game_info[3], game_info[4], game_info[5], game_info[6],
                                     new_df.iloc[-1].game_period,
                                     get_time_left(new_df.iloc[-1].game_seconds)]

    return [new_df, game_info[1], game_info[2], game_info[3], game_info[4], game_info[5], game_info[6]]

def check_if_same_game_id(game_id):
    try:
        if st.session_state['game_id'] != game_id:
            st.session_state['dataframe_event'] = pd.DataFrame()
            return True
        else:
            return False
    except:
        st.session_state['dataframe_event'] = pd.DataFrame()
        return 'error'


with st.sidebar:
    #Default behaviour, will add more complexe behaviour base on our experience but this is for display purpose
    workspace = st.text_input('Workspace', 'ift-6758-team-7', help='Name of the workspace on comet.', placeholder="Workspace")
    model = st.text_input('Model', 'xgboostwithrandomizedsearchcv', help='Name of the model that is stored in the workspace registry.', placeholder="Model")
    version = st.text_input('Version', '1.0.2', help='Version of the model.', placeholder="Version")
    file = st.text_input('File', 'XGBoostWithRandomizedSearchCV.json', help='File name that resides inside the comet model. The name of the'
                                                      ' model doesnt always match the file that is downloaded, use this '
                                                      'field to indicate the proper filename. The exact'
                                                      ' name with extension is required, only json and pickle extension are supported', placeholder="someFile.ext")
    if st.button('Get model', key='get_model_key'):
        model_download_result = button_get_model(workspace,model,version,file)
        if model_download_result:
            st.caption(':heavy_check_mark: :green[Model was fetched]')
        else:
            st.caption(':x: :red[An unexpected behaviour stopped the download]')
    pass

with st.container():
    game_id = st.text_input('Game ID', '2017020001', help='The first 4 digits identify the season of the game (ie. '
                                                          '2017 for the 2017-2018 season). The next 2 digits give the '
                                                          'type of game, where 01 = preseason, 02 = regular season, '
                                                          '03 = playoffs, 04 = all-star. The final 4 digits identify '
                                                          'the specific game number. For playoff games, the 2nd digit '
                                                          'of the specific number gives the round of the playoffs, '
                                                          'the 3rd digit specifies the matchup, and the 4th digit '
                                                          'specifies the game (out of 7).', placeholder="Workspace")
    game_ping = st.button('Ping Game', key='get_game_id')

with st.container():
    if game_ping:
        check_if_same_game_id(game_id)
        set_game_client(game_id)
        game_info = get_game_info_by_game_id(game_id)
        game_info = set_session_information(game_info)

    try:
        game_info = st.session_state['game_info']
        st.header(f"{st.session_state['game_id']}: {game_info[0]} VS {game_info[1]}")
        st.text(f'Period {game_info[6]} - {game_info[7]}')
        team_1_container, team_2_container = st.columns(2)
        with team_1_container:
            info = get_score_difference(game_info[2], game_info[4])
            st.metric(f'{game_info[0]}', f"{game_info[2]} ({game_info[4]})", f'{info[0]}')

        with team_2_container:
            info = get_score_difference(game_info[3], game_info[5])
            st.metric(f'{game_info[1]}', f"{game_info[3]} ({game_info[5]})", f'{info[0]}')
    except Exception as e:
        st.text('There is no game data or prediction yet to be shown')
        print(e)

with st.container():
    st.header(f"Data used for predictions")
    try:
        st.dataframe(st.session_state['dataframe_event'])
    except Exception as e:
        st.text('There is no data yet to be shown')
        print(e)

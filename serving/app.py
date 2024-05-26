"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging

import numpy as np
import requests
from flask import Flask, jsonify, request, abort
from comet_ml.api import API
import sklearn
import pandas as pd
import joblib
import xgboost as xgb
import pickle
from xgboost import DMatrix


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODEL = None
API_KEY = os.environ.get('COMET_API_KEY')

app = Flask(__name__)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    #initialize the logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    #load default model
    global MODEL
    MODEL = get_pickle_model("../default_model/lr_random.pickle")
    pass

@app.route("/test")
def test():
    return 'HelloWorld'

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    file_log = None
    try:
        logging.info('--------------------logs start--------------------')
        file_log = open(LOG_FILE, "r")
        list_of_logs = file_log.readlines()
    except:
        logging.info('--------------------logs end--------------------')
        return print_error('read the logs')
    try:
        logging.info('Printing out the logs')
        response = list_of_logs
        logging.info('--------------------logs end--------------------')
        return jsonify(response)  # response must be json serializable!
    except:
        #return 'Something went wrong while trying to jsonify the logs, please view the log file (/logs)'
        logging.info('--------------------logs end--------------------')
        return print_error('jsonify the logs')

@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: workspace from which to fetch the model(required),
            model: name of the model with the extension if it is for the file otherwise the model name(required),
            version: version of the model(required),
            path: path to the model including the model with the extension(optional)
            file: name of the file with extension if the model name isnt the same as the name in the model (optional)
        }

    """
    global MODEL
    workspace = None
    model = None
    version = None
    path = None #Dont not use path it will be remove
    file = None

    # Get POST json data
    try:
        logging.info('--------------------download_moodel start--------------------')
        logging.info(f'API KEY = {API_KEY}')
        json = request.get_json()
        workspace = json['workspace']
        model = json['model']
        version = json['version']
        try:
            file = json['file']
            if file == '':
                file = None
        except:
            file = None
            logging.info('No file was given ')
        try:
            path = json['path']
            path = path+model
        except:
            path = f'../default_model/{model}'
            logging.info(f'No path was given will use default: {path}')
        app.logger.info(json)
    except:
        logging.info('--------------------download_moodel end--------------------')
        return print_error('read the json body')
    try:
        file_path = Path(path)
        if file_path.is_file():
            #Will load the model from file
            logging.info('A file was found at the path location')

            #If the model was saved in json we use the json method otherwise we assumed it was the pickled option
            try:
                if '.json' in path:
                    logging.info('The model was saved using json')
                    MODEL = get_json_model(path)
                else:
                    logging.info('The model was saved using pickle')
                    MODEL = get_pickle_model(path)
            except:
                logging.info('--------------------download_moodel end--------------------')
                return print_error('load the model')
            logging.info('The model was properly loaded')
            logging.info(f'The new model is now {model}')
        else:
            #Will have to dowload the model from comet
            logging.info(f'There was no file at the following path: {path}')
            logging.info(f'We will try to dowload from comet')

            try:
                api = API(API_KEY)
                api.download_registry_model(workspace, model, version=version, output_path="../default_model/",
                                        expand=True, stage=None)
            except Exception as e:
                logging.info(f'The download from comet didnt happened')
                logging.info('--------------------download_moodel end--------------------')
                logging.info(e)
                return print_error('load the model from comet')
            #Since we have two different extension, we will have to make a check for both
            logging.info(f'Trying to load using the json model')
            path = f'../default_model/{model}.json'
            file_path = Path(path)
            if file_path.is_file():
                try:
                    logging.info(f'a model with json was found')
                    MODEL = get_json_model(path)
                except:
                    logging.info(f'We couldnt open the model file for json')
                    logging.info('--------------------download_moodel end--------------------')
                    return print_error(f'properly open the json model from file : {path}')
            else:
                logging.info(f'Trying to load using the pickle model')
                path = f'../default_model/{model}.pickle'
                file_path = Path(path)
                if file_path.is_file():
                    try:
                        logging.info(f'a model with pickle was found')
                        MODEL = get_pickle_model(path)
                    except:
                        logging.info(f'We couldnt open the model file for pickle')
                        logging.info('--------------------download_moodel end--------------------')
                        return print_error(f'properly open the pickle model from file : {path}')
                else:
                    logging.info(f'Trying to load using the file parameter')
                    if not file == None:
                        path = f'../default_model/{file}'
                        file_path = Path(path)
                        try:
                            if file_path.is_file():
                                if '.json' in file:
                                    logging.info(f'a model with json was found')
                                    MODEL = get_json_model(path)
                                else:
                                    logging.info(f'a model with pickle was found')
                                    MODEL = get_pickle_model(path)
                        except:
                            logging.info(f'There was something that went wrong while trying to read the file')
                            logging.info('--------------------download_moodel end--------------------')
                            return print_error(f'find model using the file parameter')
                    else:
                        logging.info(f'We couldnt open the model file for pickle')
                        logging.info('--------------------download_moodel end--------------------')
                        return print_error(f'find model that was dowloaded from comet')
            logging.info('--------------------download_moodel end--------------------')
            return 'The model was dowloaded'

    except:
        logging.info('--------------------download_moodel end--------------------')
        return print_error('download the model')


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    global MODEL
    try:
        logging.info('Model')
        logging.info(MODEL)
        # Get POST json data
        json = request.get_json()
        app.logger.info(json)

        # Récupération du json contenant les données sous la clé Array
        X = json['Array']
        array = np.array(X)

        response = None
        try:
            prediction = MODEL.predict(array)
            prediction_proba = MODEL.predict_proba(array)
            response = {'Predictions' : prediction.tolist(), 'Prediction_Proba' : prediction_proba.tolist()}
        except:
            logging.info('Proceeding with DMatrix')
            df = pd.DataFrame(data=array, columns=['game_seconds_as_num', 'game_period', 'coordinates_x', 'coordinates_y', 'shot_types_as_num', 'shot_distance', 'shot_angle', 'is_rebound', 'change_in_shot_angle', 'speed', 'time_since_power_play_started', 'num_friendly_skaters', 'num_opposing_skaters'])
            dmatrix = DMatrix(df)
            logging.info(type(dmatrix))
            prediction = MODEL.predict(dmatrix)
            response = {'Predictions' : prediction.tolist()}
        
        app.logger.info(response)
        return jsonify(response)  # response must be json serializable!
    except Exception as e:
        logging.info('Unable to compute the prediction')
        logging.info(e)

def print_error(message:str) -> str:
    """
    The methods will always return a str with the message inside of it. It is best to simply write the action that failed
    Args:
        message: Action that has failed during the execution of the code
    Returns: a string of the following format : Something went wrong while trying to {message}, please view the log file (/logs)

    """
    return f'Something went wrong while trying to {message}, please view the log file (/logs)'

def get_model(path:str):
    if '.json' in path:
        return get_json_model(path)
    else:
        return get_pickle_model(path)
def get_json_model(path:str):
    # f = open(f"{path}{json_model_name}", 'r')
    bst = xgb.Booster()
    bst.load_model(path)
    return bst

def get_pickle_model(path:str):
    model_pickle = pickle.load(open(path, 'rb'))
    return model_pickle

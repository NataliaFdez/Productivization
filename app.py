#from asyncore import file_dispatcher
#from msilib.schema import File

import flask
from flask import request, jsonify
import pickle
import codecs, json
from matplotlib.pyplot import fill
import pandas as pd
import numpy as np

from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA


# MODELO MULTIPLE

def load_model1():
    with open(r'C:\Users\Natalia\OneDrive\Documentos\DataScience_TheBridge\GitHub\_natalia_thebridge_ft_nov21\5N-Productivization\Entrega2\Productivization\finished_model_arima_multiple.model', "rb") as archivo_entrada:
        list_models = pickle.load(archivo_entrada)
        #print(list_models)
    return list_models

list_models = load_model1()


def make_predictions1(list_models,n_periods=2):
    prediccion_model1 = []

    for i in list_models:
        prediccion = i.predict(n_periods=n_periods)
        prediccion_model1.append(list(prediccion))

    #print(prediccion_model1)
    return prediccion_model1



# MODELO UNICO

def load_model2():
    with open(r'C:\Users\Natalia\OneDrive\Documentos\DataScience_TheBridge\GitHub\_natalia_thebridge_ft_nov21\5N-Productivization\Entrega2\Productivization\finished_model_arima.model', "rb") as archivo_entrada:
        modelo_arima = pickle.load(archivo_entrada)
        #print(modelo_arima)
    return modelo_arima

def make_predictions2(modelo_arima,n_periods=2):
    prediccion_model2 = list(modelo_arima.predict(n_periods=n_periods))
    #print(prediccion_model2)
    return prediccion_model2

modelo_arima = load_model2()



app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/',methods=['GET'])
def home():
    return "<h1>MODELO ARIMA</h1><p>Este modelo predice una serie temporal de contaminares en el aire</p>"



@app.route('/api/model/multiple', methods=['GET'])
def predict1():
    modelo1 = load_model1()
    prediccion_model1 = make_predictions1(modelo1)
    
    return jsonify(prediccion_model1)


@app.route('/api/model/unico', methods=['GET'])
def predict2():
    modelo2 = load_model2()
    prediccion_model2 = make_predictions2(modelo2)

    return jsonify(prediccion_model2)


app.run()
import pandas as pd 
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

# VOR Data Read
import pandas as pd 
df_VOR = pd.read_csv('VOR Train Test.csv')
df_VOR = df_VOR.drop('ORDER',axis=1)

Features_VOR = df_VOR.iloc [:,0:-1].values      # Feature Matrix / Independent Variables
Targets_VOR, Values_VOR = pd.factorize(df_VOR["FAULTY PARTS"])      # Target Matrix / Dependent Variable

# MMC Data Read
df_MMC = pd.read_csv('Multimode Train Test.csv')
df_MMC = df_MMC.drop('ORDER',axis=1)

Features_MMC = df_MMC.iloc [:,0:-1].values      # Feature Matrix / Independent Variables
Targets_MMC, Values_MMC = pd.factorize(df_MMC["FAULTY PARTS"])      # Target Matrix / Dependent Variable


app=Flask(__name__)


#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return render_template('Index.html')

@app.route('/VORcheck')
def VORcheck():
    return render_template('VORcheck.html')

@app.route('/MMCcheck')
def MMCcheck():
    return render_template('Multimodecheck.html')

#Prediction function for VOR
def modelnb_VOR(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,19)
    loaded_modelnb_VOR = pickle.load(open("modelnb_VOR.pkl","rb"))
    resultnb_VOR_num = loaded_modelnb_VOR.predict(to_predict)
    resultnb_VOR = Values_VOR[resultnb_VOR_num]
    return resultnb_VOR [0]

def modelknn_VOR(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,19)
    loaded_modelknn_VOR = pickle.load(open("modelknn_VOR.pkl","rb"))
    resultknn_VOR_num = loaded_modelknn_VOR.predict(to_predict)
    resultknn_VOR = Values_VOR[resultknn_VOR_num]
    return resultknn_VOR [0]

def modellr_VOR(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,19)
    loaded_modellr_VOR = pickle.load(open("modellr_VOR.pkl","rb"))
    resultlr_VOR_num = loaded_modellr_VOR.predict(to_predict)
    resultlr_VOR = Values_VOR[resultlr_VOR_num]
    return resultlr_VOR [0]

#Result for VOR
@app.route('/VORresult',methods = ['GET','POST'])
def VORresult():
    if request.method == 'POST':
        to_predict_list_VOR = (request.form.getlist('testresult_VOR', type=int))
        resultnb_VOR = modelnb_VOR(to_predict_list_VOR)
        resultknn_VOR = modelknn_VOR(to_predict_list_VOR)
        resultlr_VOR = modellr_VOR(to_predict_list_VOR)
        return render_template("resultVOR.html",resultnb_VOR=resultnb_VOR,resultknn_VOR = resultknn_VOR, resultlr_VOR=resultlr_VOR)

#Prediction function for MMC
def modellr_MMC(to_predict_list_MMC):
    to_predict_MMC = np.array(to_predict_list_MMC).reshape(1,23)
    loaded_modellr_MMC = pickle.load(open("modellr_mmr.pkl","rb"))
    resultlr_MMC_num = loaded_modellr_MMC.predict(to_predict_MMC)
    resultlr_MMC = Values_MMC[resultlr_MMC_num]
    return resultlr_MMC [0]

def modelsvc_MMC(to_predict_list_MMC):
    to_predict_MMC = np.array(to_predict_list_MMC).reshape(1,23)
    loaded_modelsvc_MMC = pickle.load(open("modelsvc_mmr.pkl","rb"))
    resultsvc_MMC_num = loaded_modelsvc_MMC.predict(to_predict_MMC)
    resultsvc_MMC = Values_MMC[resultsvc_MMC_num]
    return resultsvc_MMC [0]

def modelnb_MMC(to_predict_list_MMC):
    to_predict_MMC = np.array(to_predict_list_MMC).reshape(1,23)
    loaded_modelnb_MMC = pickle.load(open("modelnb_mmr.pkl","rb"))
    resultnb_MMC_num = loaded_modelnb_MMC.predict(to_predict_MMC)
    resultnb_MMC = Values_MMC[resultnb_MMC_num]
    return resultnb_MMC [0]


# Result for MMC
@app.route('/MMCresult',methods = ['POST'])
def MMCresult():
    if request.method == 'POST':
        to_predict_list_MMC = (request.form.getlist('testresult_MMC', type=int))
        resultsvc_MMC = modelsvc_MMC(to_predict_list_MMC)
        resultlr_MMC = modellr_MMC(to_predict_list_MMC)
        resultnb_MMC = modelnb_MMC(to_predict_list_MMC)
        return render_template("resultMMC.html",resultsvc_MMC=resultsvc_MMC, resultlr_MMC=resultlr_MMC, resultnb_MMC=resultnb_MMC)

if __name__ == "__main__":
	app.run(debug=True)

#Pre processing and scoring library
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold,train_test_split,cross_val_score,StratifiedKFold,RepeatedKFold,GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,roc_auc_score, roc_curve,auc

#Model Selection Library
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier


import os
import pandas as pd 
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

# VOR Data Read
import pandas as pd 
df_VOR = pd.read_excel('VOR Train Test.xlsx')
df_VOR = df_VOR.drop('ORDER',axis=1)

Features_VOR = df_VOR.iloc [:,0:-1].values      # Feature Matrix / Independent Variables
Targets_VOR, Values_VOR = pd.factorize(df_VOR["FAULTY PARTS"])      # Target Matrix / Dependent Variable

# MMC Data Read
df_MMC = pd.read_excel('Multimode Train Test.xlsx')
df_MMC = df_MMC.drop('ORDER',axis=1)

Features_MMC = df_MMC.iloc [:,0:-2].values      # Feature Matrix / Independent Variables
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
def modelknn_MMC(to_predict_list_MMC):
    to_predict_MMC = np.array(to_predict_list_MMC).reshape(1,23)
    loaded_modelknn_MMC = pickle.load(open("modelknn_mmr.pkl","rb"))
    resultknn_MMC_num = loaded_modelknn_MMC.predict(to_predict_MMC)
    resultknn_MMC = Values_MMC[resultknn_MMC_num]
    return resultknn_MMC [0]

def modelrf_MMC(to_predict_list_MMC):
    to_predict_MMC = np.array(to_predict_list_MMC).reshape(1,23)
    loaded_modelrf_MMC = pickle.load(open("modelrf_mmr.pkl","rb"))
    resultrf_MMC_num = loaded_modelrf_MMC.predict(to_predict_MMC)
    resultrf_MMC = Values_MMC[resultrf_MMC_num]
    return resultrf_MMC [0]

def modelnb_MMC(to_predict_list_MMC):
    to_predict_MMC = np.array(to_predict_list_MMC).reshape(1,23)
    loaded_modelnb_MMC = pickle.load(open("modelnb_mmr.pkl","rb"))
    resultnb_MMC_num = loaded_modelnb_MMC.predict(to_predict_MMC)
    resultnb_MMC = Values_MMC[resultnb_MMC_num]
    return resultnb_MMC [0]


# Result for MMC
@app.route('/MMCresult',methods = ['GET','POST'])
def MMCresult():
    if request.method == 'POST':
        to_predict_list_MMC = (request.form.getlist('testresult_MMC', type=int))
        resultknn_MMC = modelknn_MMC(to_predict_list_MMC)
        resultrf_MMC = modelrf_MMC(to_predict_list_MMC)
        resultnb_MMC = modelnb_MMC(to_predict_list_MMC)
        return render_template("resultMMC.html",resultknn_MMC=resultknn_MMC, resultrf_MMC=resultrf_MMC, resultnb_MMC=resultnb_MMC)

if __name__ == "__main__":
	app.run(debug=True)

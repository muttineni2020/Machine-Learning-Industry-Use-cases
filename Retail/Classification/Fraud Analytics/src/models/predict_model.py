
import os
import logging
import numpy as np
import pandas as pd

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling imports
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score,precision_recall_curve 
from sklearn.dummy import DummyClassifier
import pickle

import  warnings
warnings.simplefilter('ignore')

def read_data():
    '''
    This method reads data and assign to Data Frame
    '''
    #set the path for raw data
    processed_data_path = os.path.join(os.path.pardir, 'data','processed')
    x_train_file_path = os.path.join(processed_data_path, 'X_train_data.csv')
    x_test_file_path = os.path.join(processed_data_path, 'X_test_data.csv')
    y_train_file_path = os.path.join(processed_data_path, 'Y_train_data.csv')
    y_test_file_path = os.path.join(processed_data_path, 'Y_test_data.csv')
    
    #Read the processed data
    X_train = pd.read_csv(x_train_file_path,index_col=0)
    X_test = pd.read_csv(x_test_file_path,index_col=0)
    Y_train = pd.read_csv(y_train_file_path,index_col=0)
    Y_test = pd.read_csv(y_test_file_path,index_col=0)
    
    return X_train, X_test, Y_train, Y_test

def predict(X_test,Y_test):
    model_file_path = os.path.join(os.path.pardir,'models','xgb_model.pkl')
    model_file_pickle = open(model_file_path, 'rb')
    xgb_model = pickle.load(model_file_pickle)
    predicted_results = xgb_model.predict(X_test)
    return predicted_results


def write_data(df):
    predicted_results_data_path = os.path.join(os.path.pardir,'data','processed')
    write_predicted_results_path = os.path.join(predicted_results_data_path,'predicted_results_X_test.csv')
    df.to_csv(write_predicted_results_path)
    
if __name__=='__main__':
    X_train,X_test,Y_train,Y_test = read_data()
    predicted_results = predict(X_test,Y_test)
    write_data(pd.DataFrame(predicted_results))
    
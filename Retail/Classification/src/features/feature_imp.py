
import os
import logging
import numpy as np
import pandas as pd

# Modeling imports
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score,precision_recall_curve 
from sklearn.dummy import DummyClassifier


import  warnings
warnings.simplefilter('ignore')


def feature_importances(X,Y):
    '''
    Feature importance using Extra tree classifier
    '''
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    feature_imp=model.feature_importances_
    print(feature_imp)

def read_data():
    '''
    This method reads data and assign to Data Frame
    '''
    #set the path for raw data
    processed_data_path = os.path.join(os.path.pardir, 'data','processed')
    fraud_data_no_target_file_path = os.path.join(processed_data_path, 'df_fraud_no_target_data.csv')
    fraud_data_target_file_path = os.path.join(processed_data_path, 'df_fraud_target_data.csv')
    
    #Read the processed data
    df_fraud_x = pd.read_csv(fraud_data_no_target_file_path,index_col=0)
    df_fraud_y = pd.read_csv(fraud_data_target_file_path, index_col=0)
    #print(df_fraud.head(2))
    return df_fraud_x,df_fraud_y
    
if __name__=='__main__':
    X, Y = read_data()
    feature_importances(X,Y)

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

def baseline_model(X_train, X_test, Y_train, Y_test):
    model_dummy = DummyClassifier(strategy='most_frequent',random_state=0)
    model_dummy.fit(X_train,Y_train)
    print('score for baseline model: {0:.2f}'.format(model_dummy.score(X_test,Y_test)))
    # Performance Metrics
    print('accuracy for Baseline model: {0:.2f}'.format(accuracy_score(Y_test,model_dummy.predict(X_test))))

    # Confusion Matrix
    print('Confusion matrix for Baseline model: \n {0}'.format(confusion_matrix(Y_test,model_dummy.predict(X_test))))

    # Precision and Recall scores
    print('Precision for baseline Model: {0:.2f}'.format(precision_score(Y_test, model_dummy.predict(X_test))))
    print('Recall for baseline Model: {0:.2f}'.format(recall_score(Y_test, model_dummy.predict(X_test))))

def random_forest(X_train,X_test,Y_train,Y_test):
    rf = RandomForestClassifier()
    rf_model = rf.fit(X_train, Y_train)

    # Predicting the results
    y_pred = rf_model.predict(X_test)
    
    #Evaluating
    # Evaluating
    model_test_score = rf_model.score(X_test,Y_test)
    conf_matrix = confusion_matrix(Y_test, y_pred)
    print ('Random Forest MODEL TEST SCORE: {0:.5f}'.format(model_test_score))
    print("Random Forest ACCURACY: {0:.2f}".format(accuracy_score(Y_test, y_pred)))
    print("Random Forest ROC-AUC: {0:.2f}".format(roc_auc_score(Y_test, y_pred)))
    print("Random Forest PRECISION: {0:.2f}".format(precision_score(Y_test, y_pred)))
    print("Random Forest RECALL: {0:.2f}".format(recall_score(Y_test, y_pred)))
    print("Random Forest Confusion Matrix:\n",conf_matrix)
    print ('\nRandom Forest True Negatives: ', conf_matrix[0,0])
    print ('Random Forest False Negatives: ', conf_matrix[1,0])
    print ('Random Forest True Positives: ', conf_matrix[1,1])
    print ('Random Forest False Positives: ', conf_matrix[0,1])
    
    return rf_model

def XGBoost(X_train,X_test,Y_train,Y_test):
    xgb = XGBClassifier()

    # Fitting the model
    xgb_model = xgb.fit(X_train, Y_train)

    # Predicting results
    y_pred = xgb_model.predict(X_test)
    
    #Evaluation
    model_train_score = xgb_model.score(X_train,Y_train)
    model_test_score = xgb_model.score(X_test,Y_test)
    conf_matrix = confusion_matrix(Y_test, y_pred)
    print ('XGBoost MODEL TEST SCORE: {0:.5f}'.format(model_train_score))
    print ('XGBoost MODEL TEST SCORE: {0:.5f}'.format(model_test_score))
    print("XGBoost ACCURACY: {0:.2f}".format(accuracy_score(Y_test, y_pred)))
    print("XGBoost ROC-AUC: {0:.2f}".format(roc_auc_score(Y_test, y_pred)))
    print("XGBoost PRECISION: {0:.2f}".format(precision_score(Y_test, y_pred)))
    print("XGBoost RECALL: {0:.2f}".format(recall_score(Y_test, y_pred)))
    print("XGBoost Confusion Matrix:\n",conf_matrix)
    print ('\nXGBoost True Negatives: ', conf_matrix[0,0])
    print ('XGBoost False Negatives: ', conf_matrix[1,0])
    print ('XGBoost True Positives: ', conf_matrix[1,1])
    print ('XGBoost False Positives: ', conf_matrix[0,1])
    return xgb_model

def model_persistance_xgb(xgb):
    #Create file path
    model_file_path = os.path.join(os.path.pardir,'models','xgb_model.pkl')
    
    #Open file to write
    model_file_pickle = open(model_file_path,'wb')
    
    #model persist
    pickle.dump(xgb,model_file_pickle)
    
    model_file_pickle.close()
    

def model_persistance_rf(rf):
    #Create file path
    model_file_path = os.path.join(os.path.pardir,'models','rf_model.pkl')
    
    #Open file to write
    model_file_pickle = open(model_file_path,'wb')
    
    #model persist
    pickle.dump(rf,model_file_pickle)
    
    model_file_pickle.close()

def validation_auc_roc(xgb_model,X_test,Y_test):
    y_predd=xgb_model.predict_proba(X_test)
    p = plot_validation_roc(Y_test,y_predd)
    roc_auc = roc_auc_score(Y_test, y_predd[:,1])
    plt.figure(figsize=(10,5))
    plt.plot(p.FPR,p.TPR, color='orange', label='ROC curve (area = %0.2f)'%roc_auc)
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], color='skyblue', lw=2, linestyle='--',label='Random guess')
    plt.legend(loc="lower right",frameon=False)

def plot_validation_roc(Y_test, y_predd):
    fpr,tpr,thresholds = roc_curve(Y_test,y_predd[:,1])
    return pd.DataFrame({'FPR':fpr,'TPR':tpr,'Threshold':thresholds})

if __name__=='__main__':
    X_train,X_test,Y_train,Y_test = read_data()
    #baseline_model(X_train,X_test,Y_train,Y_test)
    rf=random_forest(X_train,X_test,Y_train,Y_test)
    xgb_model = XGBoost(X_train,X_test,Y_train,Y_test)
    validation_auc_roc(xgb_model,X_test,Y_test)
    model_persistance_rf(rf)
    model_persistance_xgb(xgb_model)
    
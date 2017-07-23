
import os
import logging
import numpy as np
import pandas as pd

def feature_engineering(df_fraud):
    col_list = ['source','browser','sex','age_bin','purchase_bin','country','time_diff','mean_number_of_ip_device_userids','class']
    df_fraud = df_fraud[col_list]
    # Feature Engineering - create dummy variables for categorical features - onehot encoding
    final_fraud = pd.get_dummies(df_fraud,['source','browser','sex','age_bin','purchase_bin','country'])
    
    return final_fraud

def read_data():
    '''
    This method reads raw data and assign to Data Frame
    '''
    #set the path for raw data
    processed_data_path = os.path.join(os.path.pardir, 'data','processed')
    fraud_data_file_path = os.path.join(processed_data_path, 'df_fraud_data.csv')
    
    #Read the processed data
    df_fraud = pd.read_csv(fraud_data_file_path,index_col=0)
    #print(df_fraud.head(2))
    return df_fraud

def write_data(df):
    processed_features_data_path = os.path.join(os.path.pardir,'data','processed')
    write_fraud_data_feature_engineered_path = os.path.join(processed_features_data_path,'df_fraud_feature_engineered_data.csv')
    df.to_csv(write_fraud_data_feature_engineered_path)

if __name__=='__main__':
    df = read_data()
    df = feature_engineering(df)
    write_data(df)
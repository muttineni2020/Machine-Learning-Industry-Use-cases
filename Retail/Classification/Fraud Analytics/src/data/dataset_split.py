
import os
import logging
import numpy as np
import pandas as pd


def process_no_target_dataset(df):
    # Data Set with no class feature
    X = df[ [col for col in df.columns if col != "class"] ]
    return X

def process_target_dataset(df):
    # Data set with only class feature
    Y = df[ [col for col in df.columns if col == "class"] ]#df["class"]
    return Y

def write_data(df,csv_file):
    processed_dataset_data_path = os.path.join(os.path.pardir,'data','processed')
    write_fraud_data_feature_engineered_path = os.path.join(processed_dataset_data_path,csv_file)
    df.to_csv(write_fraud_data_feature_engineered_path)

def read_data():
    '''
    This method reads data and assign to Data Frame
    '''
    #set the path for raw data
    processed_data_path = os.path.join(os.path.pardir, 'data','processed')
    fraud_data_featured_file_path = os.path.join(processed_data_path, 'df_fraud_feature_engineered_data.csv')
    
    #Read the processed data
    df_fraud = pd.read_csv(fraud_data_featured_file_path,index_col=0)
    #print(df_fraud.head(2))
    return df_fraud
    
if __name__=='__main__':
    df = read_data()
    df_X = process_no_target_dataset(df)
    df_Y = process_target_dataset(df)
    write_data(df_X,'df_fraud_no_target_data.csv')
    write_data(df_Y,'df_fraud_target_data.csv')
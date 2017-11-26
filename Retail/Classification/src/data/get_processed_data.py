
import os
import logging
import numpy as np
import pandas as pd

def read_data():
    '''
    This method reads raw data and assign to Data Frame
    '''
    #set the path for raw data
    raw_data_path = os.path.join(os.path.pardir, 'data','raw')
    fraud_data_file_path = os.path.join(raw_data_path, 'fraud_data.csv')
    fraud_ip_country_file_path = os.path.join(raw_data_path, 'IpAddress_to_Country.xlsx')
    
    #Read the data
    df_fraud = pd.read_csv(fraud_data_file_path,index_col=0)
    df_ip_country = pd.read_excel(fraud_ip_country_file_path)
    s = pd.Series(df_ip_country['country'].values, pd.IntervalIndex.from_arrays(df_ip_country['lower_bound_ip_address'], df_ip_country['upper_bound_ip_address']))
    df_fraud['country'] = df_fraud['ip_address'].map(s)
    return df_fraud

def processed_data(df_fraud):
    '''
    This method process all the data manipulations on data set including feature negineering
    '''
    df_fraud.country.fillna('UnKnown',inplace=True)
    df_fraud['signup_time'] = df_fraud.signup_time.apply(pd.to_datetime)#pd.to_datetime(df_fraud.signup_time)
    df_fraud['purchase_time'] = df_fraud.purchase_time.apply(pd.to_datetime)#pd.to_datetime(df_fraud.purchase_time)

    # it is very suspicious if a user signup and then immediately purchase
    df_fraud['time_diff'] = (df_fraud.purchase_time - df_fraud.signup_time).apply(lambda x: x.seconds)
    
    # Count the number of unique user ids associated each device
    df_fraud['userids_per_ipaddress'] = df_fraud.groupby('ip_address')['user_id'].transform('count')

    # Count the number of unique user ids associated each ip address
    df_fraud['userids_per_deviceid'] = df_fraud.groupby('device_id')['user_id'].transform('count')
    
    # Adding age_bin transformation to df_fraud data frame
    df_fraud['age_bin']= pd.qcut(df_fraud.age,3,labels=['Age18-30','Age30-50','Age41+'])

    # Adding purchase_bin transformation to df_fraud data frame
    df_fraud['purchase_bin']=pd.qcut(df_fraud['purchase_value'],4,labels=['low','medium','high','very_high'])
    
    # Add column for the average of the userids_per_deviceid,userids_per_ipaddress
    df_fraud["mean_number_of_ip_device_userids"] = (df_fraud.userids_per_deviceid + df_fraud.userids_per_ipaddress) * 0.5
    
    # day of the week
    df_fraud['signup_time_dow'] = pd.to_datetime(df_fraud['signup_time']).dt.dayofweek
    df_fraud['purchase_time_dow'] = pd.to_datetime(df_fraud['purchase_time']).dt.dayofweek
    
    # week of the year
    df_fraud['signup_time_week'] = pd.to_datetime(df_fraud['signup_time']).dt.week
    df_fraud['purchase_time_week'] = pd.to_datetime(df_fraud['purchase_time']).dt.week
    
    #Hour of the day
    df_fraud['signup_hour_of_day'] = pd.to_datetime(df_fraud['purchase_time']).dt.hour
    df_fraud['purchase_hour_of_day'] = pd.to_datetime(df_fraud['signup_time']).dt.hour
    
    return df_fraud


def write_data(df):
    processed_data_path = os.path.join(os.path.pardir,'data','processed')
    write_fraud_data_path = os.path.join(processed_data_path,'df_fraud_data.csv')
    df.to_csv(write_fraud_data_path)

if __name__=='__main__':
    df=read_data()
    df = processed_data(df)
    write_data(df)
    
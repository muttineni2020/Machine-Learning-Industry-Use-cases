
import os
import logging
import numpy as np
import pandas as pd

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import  warnings
warnings.simplefilter('ignore')


def hist_visualization(df_fraud):
    '''
    Histogram for distribution
    '''
    f, hist_fig=plt.subplots(1,2, figsize=(20,5))

    sns.distplot(df_fraud['purchase_value'], hist=True, kde=True, 
                 bins=int(30), color = 'orange', 
                 hist_kws={'edgecolor':'skyblue'},
                 kde_kws={'linewidth': 2}, ax=hist_fig[0])
    hist_fig[0].set_title('Purchase Histogram - Right skewed 0.67')
    hist_fig[0].set_xlabel('Bins')
    hist_fig[0].set_ylabel('Frequency')

    sns.distplot(df_fraud.age, hist=True, kde=True, 
                 bins=int(30), color = 'green', 
                 hist_kws={'edgecolor':'white'},
                 kde_kws={'linewidth': 2}, ax=hist_fig[1])
    hist_fig[1].set_title('Age : Histogram - Right skewed 0.42')
    hist_fig[1].set_xlabel('Bins')
    hist_fig[1].set_ylabel('Frequency')
    plt.show()

def outliers_visualization(df_fraud):
    '''
    Out liers detection
    '''
    f, sub_fig=plt.subplots(1,3, figsize=(20,5))
    sub_fig[0].scatter(df_fraud['purchase_value'],df_fraud.age,c='c', alpha=0.5)
    sub_fig[0].set_title('Purchase vs Age')
    sub_fig[0].set_xlabel('Purchase Value')
    sub_fig[0].set_ylabel('Age')

    sub_fig[1].boxplot(df_fraud.age)
    sub_fig[1].set_title('Age : Box plot')

    sub_fig[2].boxplot(df_fraud['purchase_value'])
    sub_fig[2].set_title('Purchase Value : Box plot')

    plt.tight_layout
    plt.show()

def other_visualization(df_fraud):
    '''
    Other visualizations - age, sex, class, purchase value, source, browser
    '''
    f, rel_fig=plt.subplots(3,3, figsize=(20,15))
    sns.countplot(x='sex', hue='class', data=df_fraud,ax=rel_fig[0,0])

    sns.catplot(x="class", y="age", kind="box", hue='sex', data=df_fraud, ax=rel_fig[0,1]);

    sns.catplot(x="class", y="purchase_value", kind="box", hue='sex', data=df_fraud, ax=rel_fig[0,2]);

    #sns.catplot(x="class", y="purchase_value", kind="box", hue='sex', data=df_fraud, ax=rel_fig[1,0]);

    sns.scatterplot(x="purchase_value", y="age", hue="class", data=df_fraud, ax=rel_fig[1,0]);

    sns.countplot(x='source', hue='class', data=df_fraud,ax=rel_fig[1,1])

    sns.countplot(x='browser', hue='class', data=df_fraud,ax=rel_fig[1,2])

    sns.factorplot(x="class", y="userids_per_deviceid", data=df_fraud, ax=rel_fig[2,0])

    sns.factorplot(x="class", y="userids_per_ipaddress", data=df_fraud, ax=rel_fig[2,1])

    h=sns.factorplot(x="class", y="time_diff", data=df_fraud, ax=rel_fig[2,2])

    plt.close(2)
    plt.close(3)
    plt.close(4)
    plt.close(5)
    plt.close(6)
    plt.tight_layout()
    plt.show()

def signup_purchase_visualization(df_fraud):
    '''
    signup hour of day, sign up week, purchase hour, purchase week
    '''
    f, time_fig=plt.subplots(2,3, figsize=(20,10))
    sns.countplot(x='signup_hour_of_day', hue='class', data=df_fraud, ax=time_fig[0,0])
    sns.countplot(x='signup_time_dow', hue='class', data=df_fraud, ax=time_fig[0,1])
    sns.countplot(x='signup_time_week', hue='class', data=df_fraud, ax=time_fig[0,2])

    sns.countplot(x='purchase_hour_of_day', hue='class', data=df_fraud, ax=time_fig[1,0])
    sns.countplot(x='purchase_time_dow', hue='class', data=df_fraud, ax=time_fig[1,1])
    sns.countplot(x='purchase_time_week', hue='class', data=df_fraud, ax=time_fig[1,2])

    plt.close(2)
    plt.close(3)
    plt.tight_layout()
    plt.show()

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

if __name__=='__main__':
    df=read_data()
    hist_visualization(df)
    outliers_visualization(df)
    other_visualization(df)
    signup_purchase_visualization(df)
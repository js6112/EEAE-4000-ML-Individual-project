import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import joblib


def get_lstm_data(pd_data, l_seq, scale):
    
    train_pd = pd_data.dropna()
    input_columns = ['fco2_ave_unwtd', 'mld', 'chl', 'mask', 'sss', 'sst']
    output_columns = ['atm_pco2_uatm']
    train_pd = train_pd[input_columns + output_columns]
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_pd.loc[:] = scaler.fit_transform(train_pd.values)
    input_pd = train_pd[input_columns]
    output_pd = train_pd[output_columns]
    
    input_values = []
    for column in input_columns:
        input_values.append(input_pd[column].values.reshape(-1,1))
    input_merge = np.concatenate(input_values, axis=1)
    input_scaled = input_merge 
#     input_scaler = MinMaxScaler(feature_range=(0, 1))
#     input_scaled = input_scaler.fit_transform(input_merge)
    
    n_sample = input_scaled.shape[0]
    m_feature = input_scaled.shape[1]
    
    input_scaled_aug = np.zeros((l_seq-1, m_feature))
    for i in range(np.shape(input_scaled_aug)[0]):
        for j in range(m_feature):
            input_scaled_aug[i,j] = input_scaled[0,j]
    
    input_scaled_aug = np.concatenate((input_scaled_aug, input_scaled), axis=0)
    
    X = np.zeros((n_sample, l_seq, m_feature))
    for i in range(n_sample):
        tmp = 0
        X[i,:,:] = input_scaled_aug[i+tmp:i+tmp+l_seq,:]
        tmp = tmp+1
    
    output_values = output_pd.values.reshape(-1,1)
    output_scaled = output_values
#     output_scaler = MinMaxScaler(feature_range=(0, 1))
#     output_scaled = output_scaler.fit_transform(output_values)
    
    n_sample = output_scaled.shape[0]
    m_feature = 1
    output_scaled_aug = np.zeros((l_seq-1, m_feature))
    for i in range(np.shape(output_scaled_aug)[0]):
        output_scaled_aug[i,:] = output_scaled[0,:]

    output_scaled_aug = np.concatenate((output_scaled_aug, output_scaled), axis=0)

    Y = np.zeros((n_sample, l_seq, m_feature))

    for i in range(n_sample):
        tmp = 0
        Y[i,:,:] = output_scaled_aug[i+tmp:i+tmp+l_seq,:]
        tmp = tmp+1
    return X, Y
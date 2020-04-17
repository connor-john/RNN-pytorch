# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:07:37 2020

@author: Connor
A general implementation for time series with RNNs
"""

# Importing the Libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Hyperparameters -- change params for different datasets
test_size = 12
seq_size = 12
hidden_dim = 100
lr = 0.001
epochs = 100
filename = 'Data/Alcohol_sales.csv'

# Model
class LSTM(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = hidden_dim, output_dim = 1):
        
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        
        self.hidden = (torch.zeros(1,1,hidden_dim), torch.zeros(1,1,hidden_dim))
    
    def forward(self, seq):
        
        lstm_out, self.hidden = self.lstm(seq.view(len(seq),1,-1), self.hidden)
        pred = self.fc1(lstm_out.view(len(seq), -1))
        
        return pred[-1]

# Prepare data function
def prep_data(seq, ss):
    output = []
    l = len(seq)
    
    for i in range(l - ss):
        data = seq[i:i+ss]
        label = seq[i+ss:i+ss+1]
        output.append((data,label))
    
    return output

# Import data
def import_data():
    df = pd.read_csv(filename, index_col = 0, parse_dates = True)
    df = df.dropna()
    df = df['S4248SM144NCEN'].values.astype(float)
    train = df[:-test_size]
    test = df[-test_size:]
    return train, test

# Normalise data
def norm_data(train, scaler):
    train_norm = scaler.fit_transform(train.reshape(-1,1))
    train_norm = torch.FloatTensor(train_norm).view(-1)
    return train_norm

# Main
def main():
    
    # Initialise
    train, test = import_data()
    scaler = MinMaxScaler(feature_range=(-1,1))
    train_norm = norm_data(train, scaler)
    train_data = prep_data(train_norm, seq_size)
    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    # Train
    for i in range(epochs):
        for  seq, y_train in train_data:
        
            optimizer.zero_grad()
        
            model.hidden = (torch.zeros(1,1,model.hidden_dim),torch.zeros(1,1,model.hidden_dim))
        
            y_pred = model(seq)
        
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

        # print result
        print(f'Epoch: {i+1:2} | Loss: {loss.item():10.8f}')

main()     
    
    
    
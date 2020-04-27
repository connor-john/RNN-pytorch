# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:24:00 2020

@author: Connor
A general implementation for NLP text generation with RNNs
"""

# Importing the libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters -- change params for different datasets
lr = 0.001
train_per = 0.9
epochs = 20
batch_size = 100
seq_len = 100
num_hidden = 512
num_layers = 3
drop_prob = 0.5
filename = 'Data/shakespeare.txt'

# RNN model
class CharRNN(nn.Module):
    def __init__(self, all_chars, num_hidden = 256, num_layers = 4, drop_prob = 0.5):
        
        super().__init__()
        
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.all_chars = all_chars
        
        self.decoder = dict(enumerate(all_chars))
        self.encoder = {char: i for i, char in self.decoder.items()}
        
        # Architecture
        self.lstm = nn.LSTM(len(all_chars), num_hidden, num_layers, dropout = drop_prob, batch_first = True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(num_hidden, len(self.all_chars))
        
    def forward(self, x, hidden):
        
        lstm_out, hidden = self.lstm(x, hidden)
        drop_out = self.dropout(lstm_out)
        drop_out = drop_out.contiguous().view(-1, self.num_hidden)
        output = self.fc1(drop_out)
        
        return output, hidden
    
    def hidden_state(self, batch_size):
        
        hidden = (torch.zeros(self.num_layers, batch_size, self.num_hidden).cuda(),
                  torch.zeros(self.num_layers, batch_size, self.num_hidden).cuda())
        
        return hidden
    
# one hot encoding
def one_hot_enc(batch_text, uni_chars):
    one_hot = np.zeros((batch_text.size, uni_chars))
    one_hot = one_hot.astype(np.float32)
    
    one_hot[np.arange(one_hot.shape[0]), batch_text.flatten()] = 1.0
    
    one_hot = one_hot.reshape((*batch_text.shape, uni_chars))
    
    return one_hot

# generate batches for training
def gen_batch(en_text, sample_size = 10, seq_len = 50):
    
    char_len = sample_size * seq_len
    num_batches = int(len(en_text) / char_len)
    
    en_text = en_text[: num_batches * char_len]
    en_text = en_text.reshape((sample_size, -1))
    
    for n in range(0,en_text.shape[-1], seq_len):
        x = en_text[:, n : n + seq_len]
        y = np.zeros_like(x)
        
        try:
            y[:, : -1] = x[:, 1:]
            y[:, -1] = en_text[:, n + seq_len]
        
        except:
            y[:, : -1] = x[:, 1:]
            y[:, -1] = en_text[:, 0]
        
        yield x,y
        
# get the text data
def get_text(filename):
    with open('Data/shakespeare.txt', 'r', encoding = 'utf8') as t:
        text = t.read()
    
    return text

# Initialise text data
def prep_data(text, model):
    encoded_text = np.array([model.encoder[char] for char in text])
    num_char = max(encoded_text) + 1
    train_ind = int(len(encoded_text) * (train_per))
    
    train_data = encoded_text[:train_ind]
    test_data = encoded_text[train_ind:]
    
    return train_data, test_data, num_char

# Prediction 
def predict_next(model, char, hidden = None, k = 1):
    
    encoded_text = model.encoder[char]
    encoded_text = np.array([[encoded_text]])
    encoded_text = one_hot_enc(encoded_text, len(model.all_chars))
    
    inputs = torch.from_numpy(encoded_text)
    inputs = inputs.cuda()
    
    hidden = tuple([state.data for state in hidden])
    
    lstm_out, hidden = model(inputs, hidden)
    
    probs = F.softmax(lstm_out, dim = 1).data
    probs = probs.cpu()
    
    probs, index_pos = probs.topk(k)
    index_pos = index_pos.numpy().squeeze()
    probs = probs.numpy().flatten()
    probs = probs/probs.sum()
    
    char = np.random.choice(index_pos, p = probs)
    
    return model.decoder[char], hidden

# Generate text
def generate_text(model, size, seed = 'The', k = 1):
    
    model.cuda()
    model.eval()
    
    output_chars = [c for c in seed]
    hidden = model.hidden_state(1)
    
    for char in seed:
        char, hidden = predict_next(model, char, hidden, k = k)
    
    output_chars.append(char)
    
    for i in range(size):
        char, hidden = predict_next(model, output_chars[-1], hidden, k = k)
        output_chars.append(char)
    
    return ''.join(output_chars)

def main():
    
    # Initialise
    text = get_text(filename)
    all_char = set(text)
    model = CharRNN(all_chars = all_char, num_hidden = 512, num_layers = 3, drop_prob = 0.5)
    train_data, test_data, num_char = prep_data(text, model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    t = 0
    
    # Train
    model.train()
    model.cuda()
    
    for i in range(epochs):
        
        hidden = model.hidden_state(batch_size)
    
        for x,y in gen_batch(train_data, batch_size, seq_len):
            
            t += 1
            x = one_hot_enc(x, num_char)
            
            inputs = torch.from_numpy(x)
            targets = torch.from_numpy(y)
            inputs = inputs.cuda()
            targets = targets.cuda()
                
            hidden = tuple([state.data for state in hidden])
            
            model.zero_grad()
            
            lstm_out, hidden = model.forward(inputs, hidden)
            loss = criterion(lstm_out, targets.view(batch_size * seq_len).long())
            
            loss.backward()
            
            # Avoid exploding gradient
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)
            
            optimizer.step()
            
            # validation step
            if t % 25 == 0:
                
                val_hidden = model.hidden_state(batch_size)
                val_losses = []
                model.eval()
                
                for x,y in gen_batch(test_data, batch_size, seq_len):
                    
                    x = one_hot_enc(x,num_char)
    
                    inputs = torch.from_numpy(x)
                    targets = torch.from_numpy(y)
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                        
                    val_hidden = tuple([state.data for state in val_hidden])
                    
                    lstm_out, val_hidden = model.forward(inputs, val_hidden)
                    val_loss = criterion(lstm_out, targets.view(batch_size * seq_len).long())
            
                    val_losses.append(val_loss.item())
                
                # Reset to training 
                model.train()
                
                print(f"epoch: {i} | step: {t} | val loss {val_loss.item()}")
    
    # Save model    
    name = 'CharRNN_hidden' + num_hidden + '_layers' + num_layers + '_' + filename + '.net'
    torch.save(model.state_dict(), name)
    
    # Test output
    print(generate_text(model, 1000, seed = 'The ', k = 3))
    
    
    
    
    
    
    
    
    
    
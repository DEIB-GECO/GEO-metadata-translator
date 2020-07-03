import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import preprocessing as k_preproc
from IPython.display import clear_output
import logging
from torch.nn import CrossEntropyLoss
import torch


def tokenize_test_sequences(x,y, tokenizer, MAX_LEN = 600):
    tokenized_x = []
    tokenized_y = []
    for _x,_y in zip(x,y):
        encoded_x = tokenizer.encode(_x)
        encoded_y = tokenizer.encode(_y)
        dim_x = len(encoded_x)
        dim_y = len(encoded_y)
        if(dim_x + dim_y <= MAX_LEN):
            tokenized_x.append(encoded_x)
            tokenized_y.append(encoded_y)
    excluded = len(x) - len(tokenized_x)
    print("{} values were excluded because exceed a MAX LENGTH of: {}".format(excluded,MAX_LEN))
    return tokenized_x, tokenized_y

def generate_test_data(dataframe):
    inputs = dataframe['Input'].values
    targets = dataframe['Output'].values
    corrected_labels = targets
    corrected_inputs = inputs
    return corrected_inputs, corrected_labels

class GPT2Loss(CrossEntropyLoss):
    def __init__(self):
        super(GPT2Loss, self).__init__()

    def forward(self, output, labels):
        # Flatten the tensors (shift-align)
        # Remove last token from output
        output = output[:, : , :-1].contiguous().view(-1, output.size(-1))

        # Remove the first token from labels e do not care for question
        labels = (labels[..., 1:].contiguous()).view(-1)

        return super(GPT2Loss, self).forward(output, labels)

def generate_professor_forcing(dataframe):
    inputs = dataframe['Input'].values
    targets = dataframe['Output'].values
    sequence = []
    for x,y in zip(inputs, targets):
        tmp = x + y
        sequence.append(tmp)
    return sequence

def tokenize_sequences(data, tokenizer, MAX_LEN):
    tokenized_data = []
    for elem in data:
        dim = len(elem)
        encoded = tokenizer.encode(elem)
        dim = len(encoded)
        if(dim <= MAX_LEN):
            tokenized_data.append(encoded)
    excluded = len(data) - len(tokenized_data)
    print("{} values were excluded because exceed a MAX LENGTH of: {}".format(excluded,MAX_LEN))
    return tokenized_data

def pad_sequences(data, value, maxlen):
    if not isinstance(data, (np.ndarray, np.generic)):
            data = np.array(data)
    padded = k_preproc.sequence.pad_sequences(data, padding='post', value=value, maxlen=maxlen)
    return padded.tolist()

def create_tensors(data):
    tensor_list = []
    max_len_tr = 0
    for elem in data:
        ids = torch.tensor(elem).unsqueeze(0)
        tensor_list.append(ids)
    return tensor_list
print("Importing packages...")
from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import numpy as np
import torch
import json
import os
from sklearn.model_selection import train_test_split



EPOCHS = 80
LEARNING_RATE = 2e-4
LEARNING_RATE_STR = "2e-4"
DATASET = "ENCODE"
train_file = "Data/roberta_trainset.csv"
test_file = "Data/roberta_testset.csv"
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
train_labels = [[int(y) for y in x.replace("[","").replace("]","").split(", ")] for x in train.labels]
test_labels = [[int(y) for y in x.replace("[","").replace("]","").split(", ")] for x in test.labels]
train.labels = train_labels
test.labels = test_labels


parameters = {'train_batch_size':10, 
                    'gradient_accumulation_steps':16, 
                    'learning_rate': LEARNING_RATE, 
                    'num_train_epochs': 1, 
                    'max_seq_length': 512, 
                    'output_dir': f'BERT_OUTPUTS/Learning_rate_{LEARNING_RATE_STR}/Output_{DATASET}/current/', 
                    'overwrite_output_dir':True}
    
    
    
val, test = train_test_split(test, test_size = 0.5)
old_validation_loss = 0.0079635
PREVIOUS_WAS_WORSE = False
for epoch in range(70, EPOCHS):

    print("Importing model saved at previous epoch...")
    torch.cuda.empty_cache()
    model = MultiLabelClassificationModel('roberta', f'BERT_OUTPUTS/Learning_rate_{LEARNING_RATE_STR}/Output_{DATASET}/epoch{epoch - 1}',
                                          args = parameters)
    print("Training...")
    model.train_model(train)
    print("Validating for testset...")
    
    
    _, raw_outputs = model.predict(val['text'])
    val_outputs = pd.DataFrame(raw_outputs)
    print(f"Saving results for epoch {epoch}/{EPOCHS}...")
    result, model_outputs, wrong_predictions = model.eval_model(val)
    new_validation_loss = result['eval_loss']
    print(f"VALIDATION LOSS = {new_validation_loss}")
    if(new_validation_loss > old_validation_loss and PREVIOUS_WAS_WORSE):
        break
    elif(new_validation_loss > old_validation_loss):
        PREVIOUS_WAS_WORSE = True
    old_validation_loss = new_validation_loss
    os.system(f"cp -r BERT_OUTPUTS/Learning_rate_{LEARNING_RATE_STR}/Output_{DATASET}/current BERT_OUTPUTS/Learning_rate_{LEARNING_RATE_STR}/Output_{DATASET}/epoch{epoch}")
    
##LAST VAL LOSS = 0.0079635
print(f"Importing model saved at best epoch: {epoch - 2}")
torch.cuda.empty_cache()
model = MultiLabelClassificationModel('roberta', f'BERT_OUTPUTS/Learning_rate_{LEARNING_RATE_STR}/Output_{DATASET}/epoch{epoch - 2}',
                                          args = parameters)
print(f"Testing model for epoch: {epoch - 2}")
_, test_outputs = model.predict(test['text'])               

test_outputs.to_csv(f"Results/RoBERTa_results/Learning_rate_{LEARNING_RATE_STR}/test_outputs_epoch{epoch - 2}.csv")
    

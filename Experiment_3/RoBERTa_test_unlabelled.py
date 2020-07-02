print("Importing packages...")
from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

EPOCHS = 50
LEARNING_RATE = 2e-4
BATCH_SIZE = 10
LR_STRING = "2e-4"

file = pd.read_csv("Data/manually_labelled_gsm_data.csv")


print("Importing model...")
    
model = MultiLabelClassificationModel('roberta', '../Experiment_2/BERT_OUTPUTS/Learning_rate_2e-4/Output_ENCODE/epoch69/', use_cuda = True)


print("Predicting for testset...")
_, test_raw_outputs = model.predict(file['Input'])

test_outputs = pd.DataFrame(test_raw_outputs)
test_outputs.to_csv(f"Results/manually_labelled_gsm_results_RoBERTa_69epochs.csv")


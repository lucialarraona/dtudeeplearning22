
# -*- coding: utf-8 -*-

# HPC FILE -- All together for training /sweep

# Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import Trainer, TrainingArguments

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)


import wandb
import os




print('---------------Libraries import complete-----------------')




# -------------- Data import, already split ---------
path_train = '/zhome/9c/7/174708/project/train.csv'
df_train = pd.read_csv(path_train)

path_valid = '/zhome/9c/7/174708/project/valid.csv'
df_valid = pd.read_csv(path_valid)

path_test = '/zhome/9c/7/174708/project/test.csv'
df_test = pd.read_csv(path_test)

X_train = df_train['headline']
x_valid = df_valid['headline']
X_test = df_test['headline']

y_train = df_train['is_sarcastic']
y_valid = df_valid['is_sarcastic']
y_test = df_test['is_sarcastic']

target_names = list(df_train['is_sarcastic'].unique())



# ---------------- Model Definition / Tokenization / Encoding / Metrics definition ---------------------

# Define model-name (based on hugging-face library)
model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample 
max_length = 200

# Hugging Face has its own tokenizer for the transformer: Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

# Tokenize the dataset, truncate when passed `max_length`, and pad with 0's when less than `max_length`
train_encodings = tokenizer(df_train.headline.values.tolist(), 
      add_special_tokens=True,
      truncation=True,
      max_length=max_length,
      return_token_type_ids=False,
      padding=True,
      return_attention_mask=True,
      return_tensors='pt')

valid_encodings = tokenizer(df_train.headline.values.tolist(), 
      add_special_tokens=True,
      truncation=True,
      max_length=max_length,
      return_token_type_ids=False,
      padding=True,
      return_attention_mask=True,
      return_tensors='pt')


test_encodings = tokenizer(df_test.headline.values.tolist(),
      add_special_tokens=True,
      truncation=True,
      max_length=max_length,
      return_token_type_ids=False,
      padding=True,
      return_attention_mask=True,
      return_tensors='pt')


# Create a new dataset with the tokenized input(headlines) and the labels
class NewsHeadlinesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# Convert our tokenized data into a torch Dataset
train_dataset = NewsHeadlinesDataset(train_encodings, torch.from_numpy(y_train.values))
valid_dataset = NewsHeadlinesDataset(valid_encodings, torch.from_numpy(y_valid.values))
test_dataset = NewsHeadlinesDataset(test_encodings, torch.from_numpy(y_test.values))

# -------- Training with Trainer function from HuggingFace
# Load the model and pass to CUDA
model = BertForSequenceClassification.from_pretrained(model_name,# Use the 12-layer BERT model, with an uncased vocab.
                                                      num_labels = 2, # The number of output labels--2 for binary classification.  si pones num_labels=1 hace MSE LOSS
                                                      output_attentions = False, # Whether the model returns attentions weights.
                                                      output_hidden_states = False ,# Whether the model returns all hidden-states.   
                                                      vocab_size=tokenizer.vocab_size)

# Check device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)

# Define metrics for evaluating the classification model and pass it to the Trainer object

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }




# ----------------- Sweep config and run----------------------

# method
sweep_config = {
    'method': 'random'
}

# hyperparameters to try 
parameters_dict = {
    'epochs': {
        'value': [5,10,15,20]
        },
    'batch_size': {
        'values': [8, 16, 32, 64]
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 5e-5,
        'max': 5e-3
    },
    'weight_decay': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
}


sweep_config['parameters'] = parameters_dict


# start the sweep in Wandb API 
sweep_id = wandb.sweep(sweep_config, 
                        project='dtu_deepl_models', 
                        entity = 'lucialarraona',)



# define training function with the config file parameters as inputs 

def train(config=None): 
  with wandb.init(config=config):
    # set sweep configuration
    config = wandb.config

    # set training arguments
    training_args = TrainingArguments(
        output_dir='./results',
	      report_to='wandb',  # Turn on Weights & Biases logging
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=16,
        logging_dir='./logs',            # directory for storing logs
        metric_for_best_model = 'accuracy',
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        remove_unused_columns=False,

    )

    # define training loop
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,           # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )

# start training loop
    trainer.train()


wandb.agent(sweep_id, train, count=20)



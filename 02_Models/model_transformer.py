

# -*- coding: utf-8 -*-


# Libraries

print('-----------------------MODEL TRANSFORMER TEST 1 HPC on GPU---------------------------')

import torch
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns

from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import Trainer, TrainingArguments
import os
import random
from sklearn.model_selection import train_test_split
import wandb



#wandb.login()

print('Libraries import complete')


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

print('Data import and preprocesing complete')

# ---------------- Bert Base Model

wandb.init(project='dtu_deepl_models', 
           entity='lucialarraona',
           name="bert-base-hpc-test-GPU",
           #tags=["baseline", "low-lr", "1epoch", "test"],
           group='bert')

# Define model-name (based on hugging-face library)
model_name = 'bert-base-uncased'
# max sequence length for each document/sentence sample (headlines are much shorter, we'll change it)
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

valid_encodings = tokenizer(df_valid.headline.values.tolist(), 
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
        item['labels'] = torch.tensor([self.labels[idx]])
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

#Check device and change it to CUDA
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)  


#Define metrics for evaluating the classification model and pass it to the Trainer object

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

training_args = TrainingArguments(
    output_dir='/zhome/9c/7/174708/project/results',          # output directory
    overwrite_output_dir = True,
    num_train_epochs=1,              # total number of training epochs
    evaluation_strategy='epoch',
    save_strategy = 'epoch',
    per_device_train_batch_size=16,   # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    learning_rate = 0.0005,
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='/zhome/9c/7/174708/project/logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    metric_for_best_model = 'accuracy',
                                        
                                        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=400,               # log & save weights each logging_steps
    save_steps=400,
    report_to='wandb'                # report to WANDB to keep track of the metrics :) 
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,           # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

trainer.train() # start the training



# Evaluate the model after training
trainer.evaluate()

def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to('cuda')
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return target_names[probs.argmax()]


# Save the model and tokenizer (for LIT)
model_path = '/zhome/9c/7/174708/project/model_transformer_saved'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)



## Obtain predictions for test dataset 

predictions,labels, metrics = trainer.predict(test_dataset)  

#Finish logging to wandb
wandb.finish() # finish logging to wandb


# Confusion matrix (only saving the elements to print the confusion matrix, go to collab)
matrix = confusion_matrix(labels, predictions.argmax(axis=1))
#plt.figure(figsize = (10,7))
## Confusion matrix with counts
#sns.heatmap(matrix, annot=True,cmap='Blues',fmt='g')
#sns.heatmap(matrix/np.sum(matrix), annot=True, 
#          fmt='.2%', cmap='Oranges')

#plt.xlabel('Predicted class')
#plt.ylabel('True class') 
#plt.savefig('/zhome/9c/7/174708/project/cf.png')

clas_report = classification_report(labels, predictions.argmax(axis=1))

print(clas_report)
print(metrics)

wandb.finish()
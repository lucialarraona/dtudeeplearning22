# -*- coding: utf-8 -*-
"""03_new_lstm_19nov.ipynb
Automatically generated by Colaboratory.

"""


#!pip install torchtext==0.9

#!pip install wandb

#!nvidia-smi


print('-----------------------MODEL LSTM HPC on GPU---------------------------')

import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
from torch.autograd import Variable
from torchtext.vocab import Vectors, GloVe
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import torchmetrics
from torchmetrics.classification import BinaryConfusionMatrix

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


random_seed = 42
torch.manual_seed(random_seed)
#!nvidia-smi

# Commented out IPython magic to ensure Python compatibility.

#!pip install wandb
# #!pip install torchtext

#!pip install torchtext==0.9

from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator, Iterator, ReversibleField
from torchtext.legacy.data import Dataset, Example

import wandb
import os

os.environ["WANDB_API_KEY"] = '64d2528e8d76865170c16bf804e7172bb3780c79'


print('Import libraries: complete')

# Data 
path_train = '/content/drive/MyDrive/DTU/Deep_Learning/FINAL_PROJECT/train.csv'
path_val = '/content/drive/MyDrive/DTU/Deep_Learning/FINAL_PROJECT/valid.csv'
path_test = '/content/drive/MyDrive/DTU/Deep_Learning/FINAL_PROJECT/test.csv'

# Data zuza
path_train = '/content/drive/MyDrive/DL/train.csv'
path_val = '/content/drive/MyDrive/DL/valid.csv'
path_test = '/content/drive/MyDrive/DL/test.csv'

#Data HPC

path_train='/zhome/9c/7/174708/project/train.csv'
path_test='/zhome/9c/7/174708/project/test.csv'
path_val='/zhome/9c/7/174708/project/valid.csv'



df_train = pd.read_csv(path_train)
df_valid=pd.read_csv(path_val)
df_test=pd.read_csv(path_test)


df_train = df_train.rename(columns={'headline':'text', 'is_sarcastic': 'label'})
df_valid = df_valid.rename(columns={'headline':'text', 'is_sarcastic': 'label'})
df_test = df_test.rename(columns={'headline':'text', 'is_sarcastic': 'label'})

print(df_test.shape)


# --------------- Tokenize reviews and create embeddings with GloVe Embeddings ---------------


label_field = LabelField(dtype = torch.float, batch_first = True)
tokenize = lambda x: x.split()
text_field = ReversibleField(sequential=True, tokenize = tokenize, include_lengths=True, batch_first=True, fix_length=200, lower=True, unk_token='<unk>', init_token = '<sos>', eos_token = '<eos>')

print('Tokenize and define text and label field: complete')
# --------------- Create pytorch dataset class -------------


# source : https://gist.github.com/lextoumbourou/8f90313cbc3598ffbabeeaa1741a11c8
# Helped us to use DataFrame as a Data source for building bucket iterators 

class NewsHeadlinesDataset(Dataset):

    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.label if not is_test else None
            text = row.text
            examples.append(Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

 


fields = [('text',text_field), ('label',label_field)]

train_ds, val_ds, test_ds = NewsHeadlinesDataset.splits(fields, train_df=df_train, val_df=df_valid, test_df=df_test)



# Lets look at a random example
print(vars(train_ds[15]))

# Check the type 
print(type(train_ds[15]))


# --------- Create vocabulary ----------

MAX_VOCAB_SIZE = 50000

text_field.build_vocab(train_ds, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors=GloVe(name='6B', dim=300), #customised embedding with GloVe vectors
                 unk_init = torch.Tensor.zero_)

label_field.build_vocab(train_ds)


#No. of unique tokens in text
print("Size of TEXT vocabulary:",len(text_field.vocab))

#No. of unique tokens in label
print("Size of LABEL vocabulary:",len(label_field.vocab))

#Commonly used words
print(text_field.vocab.freqs.most_common(10))  

vocab_size = len(text_field.vocab)
word_embeddings = text_field.vocab.vectors

#itos = text_field.vocab.get_itos()


# ----------- Create iterators (group data in batches) ------------

BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator = BucketIterator.splits(
    (train_ds, val_ds), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = 'cpu')


test_iterator  = BucketIterator(
    test_ds, 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = 'cpu')

for batch in train_iterator:

      text = batch.text[0]
      print(batch.label.size())
      print(text.size())
      break

#for ex in train_iterator.batches:
#    print(ex[0].text)
#    break

for batch in valid_iterator:

      text = batch.text[0]
      print(batch.label.size())
      print(text.size())
      break


for batch in test_iterator:
    text = batch.text[0]
    print(batch.label.size())
    print(text.size())
    break
 

for batch in train_iterator.batches:

  # Let's check batch size.
  print('Batch size: %d\n'% len(batch))
  print('LABEL\tLENGTH\tTEXT'.ljust(10))
  
  # Print each example.
  for example in batch:
    print('%s\t%d\t%s'.ljust(10) % (example.label, len(example.text), example.text))
  print('\n')
  
  # Only look at first batch. Reuse this code in training models.
  break


for batch in valid_iterator.batches:

  # Let's check batch size.
  print('Batch size: %d\n'% len(batch))
  print('LABEL\tLENGTH\tTEXT'.ljust(10))

  # Print each example.
  for example in batch:
    print('%s\t%d\t%s'.ljust(10) % (example.label, len(example.text), example.text))
  print('\n')

  # Only look at first batch. Reuse this code in training models.
  break


# Create batches - needs to be called before each loop.
test_iterator.create_batches() #didnt have this before

test_labels  = []
test_texts = []

for batch in test_iterator.batches:

  # Let's check batch size.
  print('Batch size: %d\n'% len(batch))
  print('LABEL\tLENGTH\tTEXT'.ljust(10))

  # Print each example.
  for example in batch:
    #print('%s\t%d\t%s'.ljust(10) % (example.label, len(example.text), example.text))
    #print('\n')
    test_labels.append(example.label)
    test_texts.append(example.text)


  # Only look at first batch. Reuse this code in training models.
  #break

print(len(test_labels))
print(len(test_texts))




# -------------- Define LSTM Model class (bidirectional LSTM)--------------------

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn1 = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, text, text_lengths):
        
        
        embedded = self.embedding(text)
  
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True).to('cuda')
        
        packed_output, (hidden, cell) = self.rnn1(packed_embedded)
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc2(output))
                
        #hidden = [batch size, hid dim * num directions]
            
        return output


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                bidirectional, dropout, pad_idx):
        
        super(RNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                        dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True).to('cuda')
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
    
        return self.fc(hidden)






# -------------- Functions of accuracy, train and evaluation ---------------------------

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

# training function 
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:

        text, text_lengths = batch.text
        text = text.to('cuda')
        text_lengths = text_lengths.to('cuda')

        optimizer.zero_grad()

        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label.to('cuda'))
        acc = binary_accuracy(predictions, batch.label.to('cuda'))

        #wandb.log({"Training Loss": loss.item()})
        #wandb.log({"Training Accuracy": acc.item()})

        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator,criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            
            text, text_lengths = batch.text
            text = text.to('cuda')
            text_lengths = text_lengths.to('cuda')
            predictions = model(text, text_lengths).squeeze(1)
            
            #compute loss and accuracy
            loss = criterion(predictions, batch.label.to('cuda'))
            acc = binary_accuracy(predictions, batch.label.to('cuda'))

            #wandb.log({"Evaluation Loss": loss.item()})
            #wandb.log({"Evaluation Accuracy": acc.item()})


            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator),


def test(model, iterator, criterion):
    
    metric = BinaryConfusionMatrix().to('cuda')
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()

    #Store test texts, labels and predictions 
    original_texts = []
    text_lengths_list = []
    y_labels = []
    y_preds = []

    # Confusion matrix
    cm = torch.Tensor([[0,0],[0,0]]).to('cuda')

    with torch.no_grad():

        
        for batch in iterator:

            #for data in valid_iter:
            #    (x, x_lengths), y = data.Text, data.Description
            #    orig_text = TEXT.reverse(x.data)
            
            text, text_lengths = batch.text
            orig_text = text_field.reverse(text) # get text from tokens 
            text = text.to('cuda')
            text_lengths = text_lengths.to('cuda')
            #(x, x_lengths) = batch.text
            
            
            #text_obj = batch.dataset.examples
            #test_texts = []
            #for element in text_obj:
            #    test_texts.append(element.text)

            predictions = model(text, text_lengths).squeeze(1)
            rounded_preds = torch.round(torch.sigmoid(predictions)) # 0 and 1 preds for the batch

            
            y_labels.append(batch.label) # append it to list of true labels for full test iterator
            y_preds.append(rounded_preds) # append it to list of predictions for full test iterator
            original_texts.append(orig_text) # append to original_texts list 
            text_lengths_list.append(text_lengths)
            #compute loss and accuracy
            loss = criterion(predictions, batch.label.to('cuda'))
            acc = binary_accuracy(predictions, batch.label.to('cuda'))

            #Confusion matrix calculation
            cm = torch.add(cm, metric(predictions, batch.label.to('cuda')))


            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()   
        
        y_labels = np.array([t.detach().cpu().numpy() for t in y_labels])
        y_preds = np.array([t.detach().cpu().numpy() for t in y_preds])
        text_lengths_list = np.array([t.detach().cpu().numpy() for t in text_lengths_list])

        
            #for batch in iterator.batches:
             #   for example in batch:
             #       test_texts.append(example.text) # append it to list of texts for full test iterator (option2) -  text tokens
        
    return epoch_acc / len(iterator), cm, y_preds, y_labels, original_texts, text_lengths_list






# ---------------- Run 1 model and track it on Wandb -----------------------


print('starting training and validation')
# Create a wandb run to log all your metrics
#run = wandb.init(project='dtu_deepl_models', entity='lucialarraona', group ='lstm', name ='working-model-check-hpc', reinit=True)
run = wandb.init(project='dtu_deepl_models', entity='lucialarraona', group ='FINAL-LSTM-POSTER', name ='bilstm-7', tags= '10 epochs-v2', reinit=True)

config = wandb.config
config.learning_rate = 5e-5
config.batch_size = 32
config.output_size = 1
config.hidden_size = 128
config.embedding_length = 300
config.epochs = 10

INPUT_DIM = len(text_field.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.3
PAD_IDX = text_field.vocab.stoi[text_field.pad_token] # padding



model = RNN(INPUT_DIM,
                      config.embedding_length,
                      config.hidden_size, 
                      config.output_size, 
                        N_LAYERS, 
                        BIDIRECTIONAL, 
                        DROPOUT, 
                        PAD_IDX)


print(model)

#No. of trianable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = text_field.vocab.vectors
print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)

#  to initiaise padded to zeros
model.embedding.weight.data[PAD_IDX] = torch.zeros(config.embedding_length)
print(model.embedding.weight.data)



# Pass model to cuda :) 
model.to('cuda') #CNN to GPU

from torch.optim import Adam

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

wandb.watch(model)




loss=[]
acc=[]
val_acc=[]
val_loss = []

for epoch in range(config.epochs):
    
    train_loss, train_acc = train(model, train_iterator,optimizer,criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator,criterion)
    #test_acc, cm, y_preds, y_labels,test_texts = test(model,test_iterator,criterion)
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Acc: {valid_acc*100:.2f}%')
    
    loss.append(train_loss)
    wandb.log({"Train Loss": train_loss})
            
    acc.append(train_acc)
    wandb.log({"Train Accuracy": train_acc})

    val_loss.append(valid_loss)
    wandb.log({"Val Loss": valid_loss})

    val_acc.append(valid_acc)
    wandb.log({"Val Accuracy": valid_acc})

    

run.finish()




# ----------------- Confusion matrix on test set -----------

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

test_acc, cm, y_preds, y_labels, original_texts,text_lengths_list = test(model,test_iterator,criterion)
print(f'\t Test. Acc: {test_acc*100:.2f}%')
print(cm)

cm = cm.detach().cpu().numpy() #to cpu for plotting the figure with seaborn

plt.figure(figsize = (10,7))
## Confusion matrix with counts
sns.set(font_scale=2.0)
sns.heatmap(cm, annot=True,cmap='Purples',fmt='g')
#sns.heatmap(matrix/np.sum(matrix), annot=True, 
#          fmt='.2%', cmap='Oranges')

plt.xlabel('Predicted class')
plt.ylabel('True class') 
#plt.savefig('/zhome/9c/7/174708/project/cf_bilstm_test8_big.png')



# ---------- Get the binary predictions to compare to bert --------------


#print(rounded_preds) # already rounded predictions
#rounded_preds = np.array([rounded_preds.detach().cpu().numpy() for t in rounded_preds]) # convert to array

# First try 
"""
print('Predictions per batch')
y_preds = np.concatenate(y_preds).tolist()
print(y_preds)
print(len(y_preds))
print('True labels per batch')
print(test_labels)
print(len(test_labels))
print('Headlines per batch')
print(original_texts)
print(len(original_texts))

"""

# Second try

print('Predictions per batch')
y_preds = np.concatenate(y_preds).tolist()
print(y_preds)
print(len(y_preds))


print('True labels per batch (from loop)')
y_labels = np.concatenate(y_labels).tolist()
print(y_labels)
print(len(y_labels))


print('Headlines per batch')
original_texts = np.concatenate(original_texts).tolist()
print(original_texts)
print(len(original_texts))

print('Headline length per batch')
text_lengths_list = np.concatenate(text_lengths_list).tolist()
print(text_lengths_list)
print(len(text_lengths_list))




# Store test set in the loop order along wiht predictions
df_result = pd.DataFrame()
df_result['text'] = original_texts 
df_result['text_length'] = text_lengths_list
df_result['label'] = y_labels  
df_result['predictions'] = y_preds # from the test loop 

df_result.to_csv('/zhome/9c/7/174708/project/df_result9.csv', index=False)




# ----------------- Hyperparameter sweep config and run ----------------------

 
"""
Run a sweep of hyperparameters on our LSTM model to check for potential combinations that lead to better accuracy
"""


"""""
# method
sweep_config = {
    'method': 'random'
}

# hyperparameters to try 
parameters_dict = {
    'epochs': {
        'values': [5,10,15,20,25]
        },
    'batch_size': {
        'values': [8, 16, 32, 64]
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 5e-5,
        'max': 0.01
    },
    'hidden_size': { 
        'values': [128, 256, 512]
    },

    'DROPOUT': {
        'values': [0.2, 0.3, 0.5]

    }


}


sweep_config['parameters'] = parameters_dict


# start the sweep in Wandb API 
sweep_id = wandb.sweep(sweep_config, 
                        project='dtu_deepl_models', 
                        entity = 'lucialarraona')

 # invariant paramenters not in config dict
INPUT_DIM = len(text_field.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
PAD_IDX = text_field.vocab.stoi[text_field.pad_token] # padding
output_size = 1
embedding_length = 300

# modifiy training function to pass the config parameters and pass it to sweep agent
def train_sweep(config=None): 
  with wandb.init(config=config):
    # set sweep configuration
    config = wandb.config


    model = LSTMClassifier(INPUT_DIM,
                            embedding_length, 
                            config.hidden_size, # this one changes
                            output_size,  
                            N_LAYERS, 
                            BIDIRECTIONAL, 
                            config.DROPOUT, # this one changes
                            PAD_IDX)


    print(model)

    #No. of trianable parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    print(f'The model has {count_parameters(model):,} trainable parameters')

    pretrained_embeddings = text_field.vocab.vectors
    print(pretrained_embeddings.shape)
    model.embedding.weight.data.copy_(pretrained_embeddings)

    #  to initiaise padded to zeros
    model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_length)
    print(model.embedding.weight.data)



    # Pass model to cuda :) 
    model.to('cuda') #CNN to GPU

    from torch.optim import Adam

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # learning rate changes

    wandb.watch(model)




    loss=[]
    acc=[]
    val_acc=[]

    for epoch in range(config.epochs):
        
        train_loss, train_acc = train(model, train_iterator,optimizer,criterion)
        valid_acc = evaluate(model, valid_iterator,criterion)
        
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Acc: {valid_acc*100:.2f}%')
        
        loss.append(train_loss)
        acc.append(train_acc)
        val_acc.append(valid_acc)
    

   





# Run the sweep 
wandb.agent(sweep_id, train_sweep, count=20)

"""


# --------------- Predict sarcasm ------------------


""""
import spacy
nlp = spacy.load('en')

def predict_sarcasm(model, sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [text_field.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction =  torch.round(torch.sigmoid(model(tensor, length_tensor)))

        
    return prediction.item()

predict_sarcasm(model, "She gave him a sarcastic smile.")
"""

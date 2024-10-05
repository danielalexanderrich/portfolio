### WIP WIP WIP :)

from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, RobertaForSequenceClassification
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import nltk
import os
import torch
matplotlib.use('qtagg')
plt.style.use('ggplot')

os.chdir('C:\\Users\\Dan\\OneDrive\\Desktop\\data sets')
df = pd.read_csv('Spam_SMS.csv')
df_train, df_test = train_test_split(df, test_size=.2, random_state=100)

train_text = df_train['Message'].values
df_train['label'] = df['Class'].map({'spam': 1, 'ham': 0})
train_labels = df_train['label'].values
test_text = df_test['Message'].values
df_test['label'] = df_test['Class'].map({'spam': 1, 'ham': 0})
test_labels = df_test['label'].values

# need a tokenizer, a custom dataset (with __init__, __len__, __getitem__ methods)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

class SpamClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=100,
            padding='max_length',# ensures uniform length
            truncation=True,# ensures uniform length
            return_token_type_ids=False,
            return_attention_mask=True,# attention mask (in this case) removes all tokens of length 1 -- see below
            return_tensors='pt'
        )

        input_ids = encoded_text['input_ids'].squeeze()
        attention_mask = encoded_text['attention_mask'].squeeze()
        label = torch.tensor(label)

        return{
            'input_ids': input_ids.cpu(), # haven't worked with gpu, need to research
            'attention_mask': attention_mask.cpu(),
            'labels': label.cpu()
        }


train_dataset = SpamClassificationDataset(train_text, train_labels, tokenizer)
test_dataset = SpamClassificationDataset(test_text, test_labels, tokenizer)

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# now just need to research training a pytorch model and train the spam classifier :)

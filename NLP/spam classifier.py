### WIP WIP WIP :)
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,  # enable mixed precision training
    evaluation_strategy='epoch',  # evaluate after each epoch
    save_strategy='epoch',  # save once per epoch
    learning_rate=5e-5,  # default learning rate for RoBERTa
    load_best_model_at_end=True,  # load the best model at the end of training
    metric_for_best_model='accuracy',
    greater_is_better=True
)

# create and train the model on the GPU
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

def predict_message(pred_text):
    # encode the message
    encoded_msg = tokenizer.encode_plus(
        pred_text,
        add_special_tokens=True,
        max_length=100,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # move the input tensor onto the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoded_msg = {k: v.to(device) for k, v in encoded_msg.items()}

    # make the prediction
    with torch.no_grad():
        prediction = model(encoded_msg['input_ids'], encoded_msg['attention_mask'])
        label = prediction.logits.argmax().item()
    if label == 1:
        output = [label, 'spam']
    else:
        output = [label, 'ham']
    return output

predict_message('our new mobile video service is live. just install on your phone to start watching.')

# want to compare to autotokenizer vs robertatokenizer

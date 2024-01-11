import functools
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
# from notebooks import utils
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          CamembertModel)

from datasets import Dataset
import evaluate

nlp = spacy.load('fr_core_news_md')

ID_TO_LABEL = {
    0: 'non-esg',
    1: 'environnemental',
    2: 'social',
    3: 'gouvernance'
}

LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

def load_model(model_path,model_config_path, eval_mode = True):
    # Load the configuration from the JSON file
    config = AutoConfig.from_pretrained(model_config_path)

    # Create the model instance
    model = AutoModelForSequenceClassification.from_config(config)

    # Load the fine-tuned model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    if eval_mode:
        model.eval()

    return model

def get_preds(model, tokenizer, sentence):
    tokenized_sentence = tokenizer(sentence, return_tensors="pt")
    input_ids, attention_mask = tokenized_sentence.input_ids, tokenized_sentence.attention_mask

    out = model(
        input_ids=tokenized_sentence.input_ids,
        attention_mask=tokenized_sentence.attention_mask
    )

    logits = out.logits

    probas = torch.softmax(logits, -1).squeeze()

    pred = torch.argmax(probas)

    return ID_TO_LABEL[pred.item()], probas[pred].item()

def get_preds2(model, tokenizer, sentence):
    tokenized_sentence = tokenizer(sentence, return_tensors="pt")
    input_ids, attention_mask = tokenized_sentence.input_ids, tokenized_sentence.attention_mask

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits

    probas = torch.softmax(logits, -1).squeeze()

    # Create a dictionary to store the probabilities for each label
    label_probas = {label: probas[idx].item() for label, idx in LABEL_TO_ID.items()}

    return label_probas



    
    

def test_model(model, tokenizer, test_df, save_path=None):
    
    test_df['prediction'] = None
    test_df['proba'] = None
    
    for text in tqdm(test_df['text']):
        probas = get_preds2(model, tokenizer, text)
        test_df.loc[test_df['text'] == text, 'prediction'] = max(probas, key=probas.get)
        test_df.loc[test_df['text'] == text, 'proba'] = str(probas)
    
    # if test['label'] exists, drop that column
    test_df.drop(columns=['label'], inplace=True, errors='ignore')
    
    # Calculating metrics
    true_labels = test_df['esg_category']
    predicted_labels = test_df['prediction']

    accuracy = accuracy_score(true_labels, predicted_labels).round(4)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0).round(4)
    # precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    # recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)

    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    
    if save_path:
        test_df.to_csv(save_path, index=False)
        
    print(test_df.columns)
        
    return f1, accuracy
    

def plot_confusion_matrix(true_labels, predicted_labels, labels,save_fig_path = None):
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm,
                annot=True,
                fmt='g',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title('Confusion Matrix')    
    if save_fig_path:
        plt.savefig(save_fig_path)
    




# 
# pl.utils
# 


def cleanse_french_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # perform lemmatisation
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc])
    
    # Remove stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in fr_stop]
    cleaned_text = ' '.join(cleaned_words)

    return cleaned_text


def tokenize_batch(samples, tokenizer):
    text = [sample["text"] for sample in samples]
    labels = torch.tensor([sample["label"] for sample in samples])
    str_labels = [sample["esg_category"] for sample in samples]
    # The tokenizer handles tokenization, padding, truncation and attn mask

    tokens = tokenizer(text, padding="longest", return_tensors="pt", truncation=True)

    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": labels, "str_labels": str_labels, "sentences": text}

def split_esg_dataset(dataset_path, datasize_frac = 1, train_size = 0.6, test_size = 0.2, validation_size = 0.2, random_state = 42,
                      return_dataloader = False, batch_size = 10, tokenizer = None, num_workers_ = 0):
    
    ID_TO_LABEL = {
    0: 'non-esg',
    1: 'environnemental',
    2: 'social',
    3: 'gouvernance'}

    LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
    
    print("Loading dataset...")
    df = pd.read_csv(dataset_path, encoding='utf-8', sep=',')
    df = df.sample(frac=datasize_frac, random_state=42).reset_index(drop=True)

    df['text'].apply(cleanse_french_text)
    df['label'] = df['esg_category'].apply(lambda x: LABEL_TO_ID[x])

    print(f"id to label: {ID_TO_LABEL}\n")
    print(df['esg_category'].value_counts())
    
    print("Splitting dataset...")
    # train, test and validation split
    df_train, df_temp = train_test_split(df, train_size=train_size, stratify=df.label, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp.label, random_state=42) 

    
    if return_dataloader :
        print("Creating dataloaders...")
        if tokenizer is None:
            raise ValueError("Tokenizer is None. Please provide a tokenizer model")
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)
        val_dataset = Dataset.from_pandas(df_val)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers_,
            collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer)
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers_,
            collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer)
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers_,
            collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer)
        )
        print("Done")
        return train_dataloader, val_dataloader, test_dataloader
    
    else:
        return df_train, df_val, df_test




# TORCH MODEL UTIL FUNCTIONS

def predict_single_input(model, tokenizer, input_text, device):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Get the highest probability class
    predicted_class_id = torch.argmax(probabilities, dim=1).item()

    # Convert class id to class name
    predicted_class = ID_TO_LABEL[predicted_class_id]

    return predicted_class, probabilities[0][predicted_class_id].item()


def predict_df(model, tokenizer, df, device):
    df_res = df.copy()
    df_res['predicted_class'] = None
    df_res['probability'] = None
    
    for i, row in df_res.iterrows():
        predicted_class, probability = predict_single_input(model, tokenizer, row['text'], device)
        df_res.at[i, 'predicted_class'] = predicted_class
        df_res.at[i, 'probability'] = probability
    
    return df_res
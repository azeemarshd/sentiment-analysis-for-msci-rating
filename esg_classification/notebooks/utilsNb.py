import functools
import os
import re
import string

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
from pdfminer.high_level import extract_text
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

def load_sd_model(state_dict_path, model_class, checkpoint, num_labels, id2label):
    """ 
    load state_dict model for evaluation purposes
    """
    model = model_class(checkpoint=checkpoint, num_labels=num_labels, id2label=id2label)
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
    model.eval()
    print('\nModel loaded for evaluation')
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
  
    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
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

def predict_single_input(model, tokenizer, input_text, tokenizer_max_len=1024, device='cpu', decimal_places=3):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=tokenizer_max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Create a dictionary of class labels to rounded probabilities
    probas_dict = {ID_TO_LABEL[idx]: round(prob.item(), decimal_places) for idx, prob in enumerate(probabilities[0])}
    predicted_class = max(probas_dict, key=probas_dict.get)
    predicted_class_proba = probas_dict[predicted_class]

    return predicted_class, predicted_class_proba, probas_dict


def print_dict(d):
    return "\n".join(f"{key}: {value}" for key, value in d.items())


def predict_df(model, tokenizer, df, tokenizer_max_len=1024):
    df_res = df.copy()
    df_res['predicted_class'] = None
    df_res['probability'] = None
    df_res['probas_dict'] = None
    
    for i, row in tqdm(df_res.iterrows(), total=len(df_res)):
        cleaned_text = cleanse_french_text(row['text'])
        predicted_class, probability,probas_dict = predict_single_input(model, tokenizer, cleaned_text,tokenizer_max_len = tokenizer_max_len)
        df_res.at[i, 'predicted_class'] = predicted_class
        df_res.at[i, 'probability'] = probability
        df_res.at[i, 'probas_dict'] = print_dict(probas_dict)

    return df_res


# ------------------------------------------------------------
# ------------------ PV PARSER CLASS -------------------------
# ------------------------------------------------------------

class PvParser:
    def __init__(self, pdf_file_path):
        self.pdf_file_path = pdf_file_path
        self.full_text = self.read_pv()
        self.sections_list = self.extract_bold_numbered_sections()
    
    def read_pv(self):
        text = extract_text(self.pdf_file_path)
        cleaned_text = self.clean_large_spaces(text)
        return cleaned_text

    
    def clean_large_spaces(self, text):
        return re.sub(r'\s{5,}', ' ', text)
    
    def extract_section(self, section_number):
        section_number -= 1  # Adjust for 0-based indexing
        if section_number < 0 or section_number >= len(self.sections_list):
            return "Invalid section number"

        start_index = self.full_text.find(self.sections_list[section_number])

        # Check if it's the last section
        if section_number + 1 < len(self.sections_list):
            end_index = self.full_text.find(self.sections_list[section_number + 1])
        else:
            end_index = len(self.full_text)  # Use the end of the document for the last section

        if start_index != -1 and (end_index != -1 or section_number + 1 >= len(self.sections_list)):
            res_text = self.full_text[start_index:end_index].strip()
            return res_text
        return "Section not found"
    
    def extract_bold_numbered_sections(self):
        # Extracts bold numbered sections from the given text
        # pattern = re.compile(r'\n(\d+\.\s+[^\n]+)', re.MULTILINE)
        pattern = re.compile(r'\n(\d+\.\s+[A-Z][^\n]+)', re.MULTILINE)
        matches = pattern.findall(self.full_text)
        return matches
    
    def split_text(self, text, max_length=1024):
        # Split the text into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding the next sentence exceeds the max_length
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                # If the current chunk is not empty, add it to the chunks list
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "  # Start a new chunk

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    
    def pv_to_df(self, chunk_size=None):
        rows = []
        
        if chunk_size is None:
            rows = []
            for i, section_title in enumerate(self.sections_list):
                text = self.extract_section(i + 1)
                # Each section becomes a row with 'Section Title' and 'Text' as columns
                section_number = section_title.split('.')[0]
                row = {'section_number': section_number, 'text': text}
                rows.append(row)
            return pd.DataFrame(rows)
        else:
            for i, section_title in enumerate(self.sections_list):
                text = self.extract_section(i + 1)
                # Split the section text into chunks
                chunks = self.split_text(text, max_length=chunk_size)

                # Create a row for each chunk
                for chunk in chunks:
                    # Extracting the section number from the title, assuming it's the first part before a dot
                    section_number = section_title.split('.')[0]
                    row = {'section_number': section_number, 'text': chunk}
                    rows.append(row)

            # Convert the list of dictionaries into a DataFrame
            df = pd.DataFrame(rows, columns=['section_number', 'text'])
            return df
                

    


import functools
import string

import pandas as pd
import spacy
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset

from transformers import AutoTokenizer

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

nlp = spacy.load('fr_core_news_md')



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

    return {"input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "labels": labels,
            "str_labels": str_labels,
            "sentences": text}



def split_esg_dataset(dataset_path, datasize_frac = 1, train_size = 0.6, test_size = 0.2, validation_size = 0.2, random_state = 42,
                      return_dataloader = False, batch_size = 10, tokenizer_path:str = None, num_workers_ = 0):
    
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
    
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
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
    
    


# 
# MINI BATCHING
# 

# 
# MINI BATCHING UTIL FUNCTIONS
# 
    
def split_text(text, tokenizer, max_length=512, stride=200):
    """
    Splits a long text into smaller chunks of a specified max length, with a stride.
    """
    tokenized_text = tokenizer.tokenize(text)
    length = len(tokenized_text)
    if length <= max_length:
        return [tokenizer.encode(text, add_special_tokens=True)]
    
    chunks = []
    for i in range(0, length, max_length - stride):
        chunk = tokenized_text[i:i + max_length]
        chunks.append(tokenizer.convert_tokens_to_ids(chunk))
    return chunks

def prepare_dataset(df, tokenizer, stride=200):
    new_rows = []
    for _, row in df.iterrows():
        text_chunks = split_text(row['text'], tokenizer, stride=stride)
        for chunk in text_chunks:
            new_rows.append({'text': chunk, 'label': row['label'], 'esg_category': row['esg_category']})
    return pd.DataFrame(new_rows)

def tokenize_batch_bis(samples, tokenizer):
    input_ids = [sample["text"] for sample in samples] # These are already token IDs
    labels = torch.tensor([sample["label"] for sample in samples])
    
    # The input_ids are already tokenized, so we just need to handle padding and attention masks
    max_length = max(len(ids) for ids in input_ids)
    padded_input_ids = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
    attention_mask = [[float(i != tokenizer.pad_token_id) for i in ids] for ids in padded_input_ids]

    return {
        "input_ids": torch.tensor(padded_input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": labels
    }


# 
# metrics
# 

def plot_confusion_matrix(labels, preds, label_names,image_path):
    confusion_norm = confusion_matrix(labels, preds, labels = label_names, normalize="true")
    confusion = confusion_matrix(labels, preds, labels = label_names)
    
    plt.figure(figsize=(15, 15  ))
    sns.heatmap(
        confusion_norm,
        annot=confusion,
        cbar=False,
        fmt="d",
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="viridis"
    )
    
    # save figures
    plt.savefig(image_path)


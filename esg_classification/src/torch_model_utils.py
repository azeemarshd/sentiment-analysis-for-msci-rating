
import functools
import os
import string

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers.data.data_collator import DataCollatorWithPadding

ID_TO_LABEL = {
    0: 'non-esg',
    1: 'environnemental',
    2: 'social',
    3: 'gouvernance'
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
nlp = spacy.load('fr_core_news_md')


def create_directories(output_directory):
    """
    Creates directories images, models/full_model, models/state_dict in output_dir if it does not exist.
    """
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)    
    if not os.path.exists(output_directory + '/images'):
        os.makedirs(output_directory + '/images')
    if not os.path.exists(output_directory + '/models/full_model'):
        os.makedirs(output_directory + '/models/full_model')
    if not os.path.exists(output_directory + '/models/state_dict'):
        os.makedirs(output_directory + '/models/state_dict')
        

# def create_directories(output_directory):
#     """
#     Creates directories images, models/full_model, models/state_dict in output_directory if they do not exist.
#     Also ensures that all intermediate directories in the output_directory path are created.
#     """
    
#     os.makedirs(output_directory, exist_ok=True)

#     # Directories to create within the output_directory
#     sub_directories = [
#         '/images',
#         '/models/full_model',
#         '/models/state_dict'
#     ]

#     # Create each sub_directory if it does not exist
#     for sub_dir in sub_directories:
#         full_path = os.path.join(output_directory, sub_dir)
#         os.makedirs(full_path, exist_ok=True)



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


def df_converter(train_df, val_df, test_df, output="dataloader" , tokenizer = None, batch_size = 10,tokenizer_model_max_length = 512):
    """
     Converts dataframes to datasetdict or datacollator depending on output. 
     This is a wrapper around the DataCollatorWithPadding class to allow us to pass in a tokenizer to be used for tokenizing the data
     
     Args:
     	 train_df: pandas DataFrame with train data
     	 val_df: pandas DataFrame with validation data ( can be any type that supports DataFrame )
     	 test_df: pandas DataFrame with test data ( can be any type that supports DataFrame )
     	 output: str "datasetdict" or "datacollator" or "dataloader". defaults to "dataloader"
     	 tokenizer: str or None if you want to use a tokenizer
     
     Returns: 
     	 a DatasetDict or a datacollator depending on
    """
    
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True,max_length=tokenizer_model_max_length)
    
    # If tokenizer is not provided.
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")
    
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    data = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'valid': val_dataset})
    
    if output == "datasetdict": return data
    
    tokenized_dataset = data.map(tokenize, batched=True)
    tokenized_dataset

    tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    if output == "data_collator": return tokenized_dataset
    
    train_dataloader = DataLoader(
        tokenized_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
        )
    eval_dataloader = DataLoader(
        tokenized_dataset["valid"], shuffle= False, batch_size=batch_size, collate_fn=data_collator
        )
    test_dataloader = DataLoader(
        tokenized_dataset["test"], shuffle = False, batch_size=32, collate_fn=data_collator
        )
    
    if output == "dataloader": return train_dataloader, eval_dataloader, test_dataloader


# --------------------------------------------------------
# ----------------------  TRAINING  ----------------------
# --------------------------------------------------------

def train_model(model, trainloader, valoader , num_epochs=2, accumulate_grad_batches=1, lr_scheduler = 'linear'):
    """
    Trains a given model using the provided trainloader and evaluates it on the validation set using the valoader.
    
    Args:
        model (torch.nn.Module): The model to be trained.
        trainloader (torch.utils.data.DataLoader): The data loader for the training set.
        valoader (torch.utils.data.DataLoader): The data loader for the validation set.
        num_epochs (int, optional): The number of epochs to train the model (default: 2).
        accumulate_grad_batches (int, optional): The number of gradient accumulation steps (default: 1).
        lr_scheduler (str, optional): The learning rate scheduler to use. Can be 'linear' or 'cosine' (default: 'linear').
    
    Returns:
        tuple: A tuple containing the f1 score, accuracy, and a list of average loss per epoch.
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0001)
    
    num_training_steps = num_epochs * len(trainloader)
    
    if lr_scheduler == 'linear':
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
    elif lr_scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps)
    
    
    
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    
    epoch_loss_list = []  # Store average loss per epoch
    
    for epoch in range(num_epochs):
        model.train()
        
        total_loss = 0
        
        
        for step, batch in enumerate(trainloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            
            if (step + 1) % accumulate_grad_batches == 0 or (step + 1) == len(trainloader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if step % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {step}/{len(trainloader)}, Loss: {loss.item()}')

        
        average_loss = total_loss/ len(trainloader)
        epoch_loss_list.append(average_loss)

    model.eval()
    print("Evaluating on validation set...")
    for batch in valoader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        f1_metric.add_batch(predictions=predictions, references=batch["labels"])
        acc_metric.add_batch(predictions=predictions, references=batch["labels"])
        
    
    f1 = f1_metric.compute(average= 'macro')['f1']*100
    acc = acc_metric.compute()['accuracy']*100

    print(f"TRAIN f1 score: {f1}")
    print(f"TRAIN accuracy: {acc}")
    
    return f1, acc, epoch_loss_list




def test_model(model, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    
    all_predictions = []
    all_true_labels = []
        
    model.eval()
    for batch in tqdm(testloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        f1_metric.add_batch(predictions=predictions, references=batch["labels"])
        acc_metric.add_batch(predictions=predictions, references=batch["labels"])
        
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(batch["labels"].cpu().numpy())
        
    
    # Generate classification report
    report = classification_report(all_true_labels, all_predictions, target_names=ID_TO_LABEL.values())
    print("\nClassification Report:\n", report)

    f1 = f1_metric.compute(average= 'macro')['f1']*100
    acc = acc_metric.compute()['accuracy']*100
    print(f"TEST f1 score : {f1}")
    print(f"TEST accuracy: {acc}")
    
    return f1, acc






def plot_epoch_loss(epoch_loss_list, save_path):
    plt.plot(epoch_loss_list)
    plt.title("epoch loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(save_path)
    plt.close()

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



# --------------------------------------------------------
# ----------------------  METRICS  -----------------------
# --------------------------------------------------------

def plot_score(scores, dataloader, score_type="accuracy", save_path=None, dataset_title=" "):
    
    print(f'Mean accuracy of the network on the {len(dataloader.dataset)} test sentences: {torch.mean(torch.tensor(scores)):.1f} % (+/- {torch.std(torch.tensor(scores)):.1f} %)')

    mean = np.mean(scores)
    std = np.std(scores)
    
    # plot scores and fill_between std
    plt.figure()
    plt.plot(scores)
    plt.fill_between(range(len(scores)), mean-std, mean+std, alpha=0.2)
    plt.legend([f'Mean: {mean:.3f} %', f'Std: {std:.3f} %'])
    plt.xlabel('Round')
    plt.ylabel(f'{score_type} ')
    plt.ylim((60, 100))

    plt.title(f'{score_type} on {dataset_title}')
    
    if save_path is not None:
        plt.savefig(save_path)

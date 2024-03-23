import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AdamW, AutoConfig, AutoModel,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          get_scheduler)
from transformers.modeling_outputs import TokenClassifierOutput
from utils import model_utils as utils
import evaluate
from datasets import Dataset, DatasetDict, load_dataset, load_metric

nlp = spacy.load('fr_core_news_md')
ID_TO_LABEL = {
    0: 'non-esg',
    1: 'environnemental',
    2: 'social',
    3: 'gouvernance'
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

CHECKPOINT = 'camembert-base'
# CHECKPOINT = 'flaubert/flaubert_small_cased'
DATASET_PATH = './data/esg_fr_classification.csv'
OUTPUT_MODEL_PATH = './models/test/torch_model1.pt'
TB_LOGS_PATH = './Tensorboard_logs/test/pl_minibatching_model'
BATCH_SIZE = 1
NUM_LABELS = len(ID_TO_LABEL)
DATASET_FRAC = 0.005
ROUNDS = 3
NUM_EPOCHS = 1

TOKENIZER = AutoTokenizer.from_pretrained("camembert-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def df_converter(train_df, val_df, test_df, output="dataloader" , tokenizer = None):
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
        return TOKENIZER(batch["text"], truncation=True,max_length=512)
    
    # If tokenizer is not provided.
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")
    
    tokenizer.model_max_length = 512
    
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
    data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)
    
    if output == "data_collator": return tokenized_dataset
    
    train_dataloader = DataLoader(
        tokenized_dataset["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
        )
    eval_dataloader = DataLoader(
        tokenized_dataset["valid"], shuffle= False, batch_size=BATCH_SIZE, collate_fn=data_collator
        )
    test_dataloader = DataLoader(
        tokenized_dataset["test"], shuffle = False, batch_size=32, collate_fn=data_collator
        )
    
    if output == "dataloader": return train_dataloader, eval_dataloader, test_dataloader

    
# --------------------------------------------------------
# ----------------------  MODEL  -------------------------
# --------------------------------------------------------
class ModelTorch1(nn.Module):
  def __init__(self,checkpoint,num_labels): 
    super(ModelTorch1,self).__init__() 
    self.num_labels = num_labels 

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(
        checkpoint,
        config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True)
        )
    self.dropout = nn.Dropout(0.1) 
    self.classifier = nn.Linear(768,num_labels) # load and initialize weights

  def forward(self, input_ids=None, attention_mask=None,labels=None):
    #Extract outputs from the body
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

    #Add custom layers
    sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

    logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
    
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)

def train_model(model, trainloader, valoader , num_epochs=2):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0001)
    
    num_training_steps = num_epochs * len(trainloader)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)
    
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    
    epoch = 0    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch in tqdm(trainloader, desc = f"training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # progress_bar_train.update(1)
        epoch += 1

    model.eval()
    for batch in tqdm(valoader,desc = f"validating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        f1_metric.add_batch(predictions=predictions, references=batch["labels"])
        acc_metric.add_batch(predictions=predictions, references=batch["labels"])

        # progress_bar_eval.update(1)
        
    print(f1_metric.compute(average= 'macro'))
    print(acc_metric.compute())
    

def test_model(model, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
        
    model.eval()
    for batch in tqdm(testloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        f1_metric.add_batch(predictions=predictions, references=batch["labels"])
        acc_metric.add_batch(predictions=predictions, references=batch["labels"])

    f1 = f1_metric.compute(average= 'macro')['f1']*100
    acc = acc_metric.compute()['accuracy']*100
    print(f"f1 score : {f1}")
    print(f"accuracy: {acc}")
    
    return f1, acc



def plot_score(scores, testloader, score_type="accuracy", save_path=None):
    
    print(f'Mean accuracy of the network on the {len(testloader.dataset)} test images: {torch.mean(torch.tensor(scores)):.1f} % (+/- {torch.std(torch.tensor(scores)):.1f} %)')

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

    plt.title(f'{score_type} of the network on the test images')
    plt.show()
    
    if save_path is not None:
        plt.savefig(save_path)
    

if __name__ == "__main__":
    
    train_df, val_df, test_df = utils.split_esg_dataset(dataset_path=DATASET_PATH,
                                                                            batch_size=BATCH_SIZE,
                                                                            datasize_frac=DATASET_FRAC,
                                                                            return_dataloader=False)
    
    
    print("\n Creating dataloaders...\n")
    trainloader, valoader, testloader = df_converter(train_df, val_df, test_df, output="dataloader", tokenizer=TOKENIZER)
    
    print("\n loading model...\n")
    model=ModelTorch1(checkpoint=CHECKPOINT,num_labels=NUM_LABELS).to(device)
    
    
    
    accuracies = []
    f1_scores = []
    
    
    for _ in tqdm(range(ROUNDS)):
        train_model(model, trainloader, valoader, num_epochs=NUM_EPOCHS)
        f1, acc = test_model(model, testloader)
        accuracies.append(acc)
        f1_scores.append(f1)
    
    print(f"accuracy: {accuracies}")
    print(f"f1 score: {f1_scores}")
        
    plot_score(accuracies, testloader, score_type="accuracy", save_path=None)
    plot_score(f1_scores, testloader, score_type="f1", save_path=None)

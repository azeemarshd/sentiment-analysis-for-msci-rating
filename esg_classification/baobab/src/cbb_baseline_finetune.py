from pprint import pprint
import functools

import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, CamembertForMaskedLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import string
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop

def cleanse_french_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in fr_stop]
    cleaned_text = ' '.join(cleaned_words)

    return cleaned_text



camembert = CamembertForMaskedLM.from_pretrained('camembert-base')
camembert.roberta.embeddings


df = pd.read_csv('~/thesis/data/esg_fr_classification.csv', encoding='utf-8', sep=',')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# rename text_fr column to text and esg_category to label
df = df.rename(columns={'text_fr': 'text'})

df['text'].apply(cleanse_french_text)

# add label id'ing
df['label'] = df['esg_category'].factorize()[0] 

print(df['esg_category'].value_counts())

train_size = 0.6
test_size = 0.2
validation_size = 0.2

# train, test and validation split
df_train, df_temp = train_test_split(df, train_size=train_size, stratify=df.label, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp.label, random_state=42)

print(f"train size: {df_train.shape} \t test size: {df_test.shape} \t validation size: {df_val.shape}")

df_train["len"] = df_train["text"].apply(lambda x: len(x.split()))
ax = df_train["len"].hist(density=True, bins=50)
ax.set_xlabel("Longueur")
ax.set_ylabel("Fréquence")
ax.set_title("Nombre de caractères par phrase")
ax.figure.show()

print(df_train["len"].describe())


def tokenize_batch(samples, tokenizer):
    text = [sample["text"] for sample in samples]
    labels = torch.tensor([sample["label"] for sample in samples])
    str_labels = [sample["esg_category"] for sample in samples]
    # The tokenizer handles tokenization, padding, truncation and attn mask

    tokens = tokenizer(text, padding="longest", return_tensors="pt", truncation=True)

    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": labels, "str_labels": str_labels, "sentences": text}


tokenizer = AutoTokenizer.from_pretrained('camembert-base')


# Convert df to Hugging Face Datasets
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)
val_dataset = Dataset.from_pandas(df_val)

val_dataloader = DataLoader(val_dataset, collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer), batch_size=16)

print(next(iter(val_dataloader))['input_ids'])
print(next(iter(val_dataloader))['attention_mask'])
print(next(iter(val_dataloader))['labels'])
print("\n ---------------------------------\n ")


num_labels = len(df_train["label"].unique())
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=5, 
    shuffle=True, 
    collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer)
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=5, 
    shuffle=False, 
    collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer)
)


class LightningModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, weight_decay, from_scratch=False):
        super().__init__()
        self.save_hyperparameters()
        if from_scratch:
            # Si `from_scratch` est vrai, on charge uniquement la config (nombre de couches, hidden size, etc.) et pas les poids du modèle 
            config = AutoConfig.from_pretrained(
                model_name, num_labels=num_labels
            )
            self.model = AutoModelForSequenceClassification.from_config(config)
        else:
            # Cette méthode permet de télécharger le bon modèle pré-entraîné directement depuis le Hub de HuggingFace sur lequel sont stockés de nombreux modèles
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_labels = self.model.num_labels

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

    def training_step(self, batch):
        out = self.forward(batch)

        logits = out.logits
        # -------- MASKED --------
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.num_labels), batch["labels"].view(-1))

        # ------ END MASKED ------

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_index):
        labels = batch["labels"]
        out = self.forward(batch)

        preds = torch.max(out.logits, -1).indices
        # -------- MASKED --------
        acc = (batch["labels"] == preds).float().mean()
        # ------ END MASKED ------
        self.log("valid/acc", acc)

        f1 = f1_score(batch["labels"].cpu().tolist(), preds.cpu().tolist(), average="macro")
        self.log("valid/f1", f1)

    def predict_step(self, batch, batch_idx):
        """La fonction predict step facilite la prédiction de données. Elle est 
        similaire à `validation_step`, sans le calcul des métriques.
        """
        out = self.forward(batch)

        return torch.max(out.logits, -1).indices

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        


lightning_model = LightningModel("camembert-base", num_labels, lr=3e-5, weight_decay=0.)


model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid/acc", mode="max")

camembert_trainer = pl.Trainer(
    max_epochs=5,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="valid/acc", patience=4, mode="max"),
        model_checkpoint,
    ]
)


camembert_trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader) 

# After model training, save callback details
early_stopping_callback = str(pl.callbacks.EarlyStopping(monitor="valid/acc", patience=4, mode="max"))
model_checkpoint_callback = str(model_checkpoint)

callbacks_str = f"Early Stopping Callback:\n{early_stopping_callback}\n\nModel Checkpoint Callback:\n{model_checkpoint_callback}"

with open('callbacks.txt', 'w') as file:
    file.write(callbacks_str)

print("Callbacks saved to 'callbacks.txt'")


    
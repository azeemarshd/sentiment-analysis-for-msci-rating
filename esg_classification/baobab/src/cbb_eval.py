import pandas as pd
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import string
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop


import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from tqdm.auto import tqdm



def cleanse_french_text(text):
    text = text.lower()

    # Remove stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in fr_stop]
    cleaned_text = ' '.join(cleaned_words)

    return cleaned_text

# Load tokenizer and model
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForSequenceClassification.from_pretrained("camembert-base")

# Load dataset
df = pd.read_csv('~/thesis/data/esg_fr_classification.csv', encoding='utf-8', sep=',')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.rename(columns={'text_fr': 'text'})


df['text'] = df['text'].apply(cleanse_french_text)
df['label'] = df['esg_category'].factorize()[0] 


texts = df['text'].tolist()
labels = df['label'].tolist()


train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)

# Tokenize the test set
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)


# Convert input data into PyTorch tensors
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

dataset = TensorDataset(input_ids, attention_mask)

data_loader = DataLoader(dataset, batch_size=32) 

model = CamembertForSequenceClassification.from_pretrained("camembert-base")

# Predict in batches
model.eval()  # Set the model to evaluation mode
predictions = []
for batch in tqdm(data_loader, desc="Making Predictions"):
    input_ids, attention_mask = batch

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        batch_predictions = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(batch_predictions.cpu().numpy())


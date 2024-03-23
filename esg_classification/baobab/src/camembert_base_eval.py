import string
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import CamembertForSequenceClassification, CamembertTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanse_french_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in fr_stop]
    cleaned_text = ' '.join(cleaned_words)

    return cleaned_text

# Load the tokenizer and model
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForSequenceClassification.from_pretrained("camembert-base")

model.to(device)

# Load your dataset
df = pd.read_csv('~/thesis/data/esg_fr_classification.csv', encoding='utf-8', sep=',')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.rename(columns={'text_fr': 'text'})


df['text'] = df['text'].apply(cleanse_french_text)
df['label'] = df['esg_category'].factorize()[0] 


texts = df['text'].tolist()
labels = df['label'].tolist()


# Split dataset into training and testing
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.3)

# Tokenize the test set
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Create a TensorDataset
dataset = TensorDataset(input_ids, attention_mask)
data_loader = DataLoader(dataset, batch_size=10) 


torch.no_grad()

model.eval()

start = time.time()

# Predict in batches
model.eval()  # Set the model to evaluation mode
predictions = []
for batch in tqdm(data_loader, desc="Making Predictions"):
    input_ids, attention_mask = batch
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)   
    
        
    outputs = model(input_ids, attention_mask=attention_mask)
    batch_predictions = torch.argmax(outputs.logits, dim=-1)
    predictions.extend(batch_predictions.cpu().numpy())

end = time.time()

print(f"Total prediction time: {end - start} seconds")

# save predictions
df['predictions'] = predictions
df.to_csv('~/thesis/data/esg_cb_base_predictions.csv', encoding='utf-8', sep=',', index=False)

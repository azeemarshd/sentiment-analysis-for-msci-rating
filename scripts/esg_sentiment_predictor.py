import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import torch
import os
import argparse
from tqdm import tqdm



class ESGSentimentPredictor:
    def __init__(self):
        self.MODEL_CHECKPOINTS = ["nlptown/bert-base-multilingual-uncased-sentiment",
                     "bardsai/finance-sentiment-fr-base",
                     "cmarkea/distilcamembert-base-sentiment"
                     ]

        self.models = {
            "model_paths":  self.MODEL_CHECKPOINTS,
            "model_names": ["bert", "finance", "camembert"],
            "models": [self.load_model(m)[0] for m in self.MODEL_CHECKPOINTS],
            "tokenizers": [self.load_model(m)[1] for m in self.MODEL_CHECKPOINTS]
        }
        
    def load_model(self,model_name: str):
        """
        Loads a pre-trained model from Hugging Face's model hub.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        return model, tokenizer
    
    def normalize_label(self,label, score):
        """
        Convert labels to a unified numerical scale.
        For 'pos', 'neg', 'neutral', converts based on a predefined mapping.
        Assumes 5-star labels are already numerical.
        """
        label_map = {'positive': 5, 'neutral': 3, 'negative': 1}
        if label in label_map:
            return label_map[label] * score  
        else:
            star_rating = int(label.split()[0])
            return star_rating * score  

    def aggregate_predictions(self,predictions):
        """
        Averages the normalized scores from all models.
        `predictions` is a list of tuples/lists with (label, score) from all models.
        """
        normalized_scores = [self.normalize_label(label, score) for label, score in predictions]
        if normalized_scores:
            return sum(normalized_scores) / len(normalized_scores)
        else:
            return None

    # Function to predict in batches
    def predict_sentiment(self, texts ,model, tokenizer, return_probs = True):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_indices = torch.argmax(predictions, dim=1)
        predicted_classes = [model.config.id2label[idx.item()] for idx in predicted_indices]
        # all_probas = predictions.tolist()
        # print(all_probas)
        probas = predictions.max(dim=1).values.tolist()
        
        if return_probs: return predicted_classes, probas
        else: return predicted_classes
    
    
    def predict_df(self, df):
        
        for i, row in tqdm(df.iterrows(), total=len(df)):
            bert_pred,bert_prob = self.predict_sentiment(row['text'], self.models["models"][0], self.models["tokenizers"][0])
            finance_pred,finance_prob = self.predict_sentiment(row['text'], self.models["models"][1], self.models["tokenizers"][1])
            camembert_pred,camembert_prob = self.predict_sentiment(row['text'], self.models["models"][2], self.models["tokenizers"][2])

            df.at[i, "sentiment_pred"] = round(self.aggregate_predictions(
                                                [(bert_pred[0], bert_prob[0]),
                                                (finance_pred[0], finance_prob[0]),
                                                (camembert_pred[0], camembert_prob[0])]),3)
    
        return df










def predict_dir(input_dir):

    
    # input_dir = args.input_dir
    
    # nyon_2022_input_dir = "./data/csv_data/nyon_2022/"
    # nyon_2023_input_dir = "./data/csv_data/nyon_2023/prediction_results/"
    # vevey_2022_input_dir = "./data/csv_data/vevey_2022/"
    # vevey_2023_input_dir = "./data/csv_data/vevey_2023/"
    
    # list_input_dirs = [nyon_2023_input_dir]
    
    print("Loading models...")
    sentiment_predictor = ESGSentimentPredictor()
    
    # for input_dir in tqdm(list_input_dirs, total=len(list_input_dirs)):
    #     print(f"\n Input folder: {input_dir}...")
        
        # output_dir = input_dir + "prediction_results/"
        # if not os.path.exists(output_dir): os.makedirs(output_dir)
        
    for filename in tqdm(os.listdir(input_dir), desc="Predicting files ...", total=len(os.listdir(input_dir))):
        file_path = os.path.join(input_dir, filename)  # Use os.path.join for better path handling
        if os.path.isfile(file_path):  # Check if it's a file
            df = pd.read_csv(input_dir + filename, encoding='utf-16', sep=',')
            
            df = sentiment_predictor.predict_df(df)
            
            df.to_csv(input_dir + filename, index=False, encoding='utf-16', sep=',')
        else:
            print(f"Skipping directory: {filename}")
            
            
    return 0


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Sentiment prediction directory of PV files')
    parser.add_argument('--input_dir', type=str, default="./data/csv_data/nyon_2023/", help='directory path to the PV files')
    args = parser.parse_args()
    
    
    predict_dir(args.input_dir)
    
    
    print(f"Predictions saved to {args.input_dir}")
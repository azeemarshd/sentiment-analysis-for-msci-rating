from notebooks import utilsNb
# import models

from src import models
from transformers import AutoTokenizer
import transformers
import torch
from tqdm import tqdm
import pandas as pd


def load_sd_model(model_name, model_sd_path):
    """
    Load a pre-trained sentiment analysis model and tokenizer.

    Args:
        - model_name (str): The name of the model to load. Supported model names are:
            - "cb-512": Camembert-base model with a maximum sequence length of 512.
            - "cb-1024": Camembert-base model with a maximum sequence length of 1024.
            - "cbl-512": Camembert-large model with a maximum sequence length of 512.
            - "cbl-1024": Camembert-large model with a maximum sequence length of 1024.
        - model_sd_path (str): The file path to the saved model state dictionary.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.

    Raises:
        KeyError: If an unsupported model name is provided.

    Example:
        model, tokenizer = load_sd_model("cb-512", "/path/to/model_state_dict.pt")
    """
    
    ID_TO_LABEL = {
        0: 'non-esg',
        1: 'environnemental',
        2: 'social',
        3: 'gouvernance'
    }
    LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
    NUM_LABELS = len(ID_TO_LABEL)

    models_meta_inf = {
        "cb-512": {
            "checkpoint": "camembert-base",
            "model_class": models.ModelCB,
            "max_length": 512
        },
        "cb-1024": {
            "checkpoint": "camembert-base",
            "model_class": models.ModelCBLong,
            "max_length": 1024
        },
        "cbl-512": {
            "checkpoint": "camembert/camembert-large",
            "model_class": models.ModelCBL,
            "max_length": 512
        },
        "cbl-1024": {
            "checkpoint": "camembert/camembert-large",
            "model_class": models.ModelCBLlong,
            "max_length": 1024
        }
    }
    transformers.logging.set_verbosity_error()
    checkpoint = models_meta_inf[model_name]["checkpoint"]
    model_class = models_meta_inf[model_name]["model_class"]
    
    TOKENIZER = AutoTokenizer.from_pretrained(checkpoint,)
    model = model_class(checkpoint, NUM_LABELS, id2label=ID_TO_LABEL)
    model.load_state_dict(torch.load(model_sd_path, map_location=torch.device('cpu')))
    
    print(f"Model {model_name} loaded.")
    
    return model, TOKENIZER




# class that combines the four models cb, cb-1024, cbl and cbl-1024
class ESGPredictor:
    def __init__(self, cb_model_path,cb_1024_model_path, cbl_model_path, cbl_1024_model_path):
        self.cb_model, self.tokenizer_base = load_sd_model("cb-512", cb_model_path)
        self.cb_1024_model, _ = load_sd_model("cb-1024", cb_1024_model_path)
        self.cbl_model, self.tokenizer_large = load_sd_model("cbl-512", cbl_model_path)
        self.cbl_1024_model, _ = load_sd_model("cbl-1024", cbl_1024_model_path)
        # self.weights = [[0.2, 0.1,0.7,0.1], # cbl
        #                 [0.7,0.1,0.1,0.8], # cbl-1024
        #                 [0.1,0.8,0.2,0.1]] # cb
        self.weights = [[0.2636232 , 0.24361493, 0.28944688, 0.24629182],
                        [0.23410941, 0.25933202, 0.23502097, 0.23456364],
                        [0.25657445, 0.25933202, 0.248515  , 0.31390135],
                        [0.24569294, 0.23772102, 0.22701716, 0.20524319]]
        self.ID_TO_LABEL = {
            0: 'non-esg',
            1: 'environnemental',
            2: 'social',
            3: 'gouvernance'}
        self.LABEL_TO_ID = {v: k for k, v in self.ID_TO_LABEL.items()}

    def predict_single_input(self,input_text, model, tokenizer, tokenizer_max_len = 512, device = "cpu", decimal_places = 4, return_class_proba = False):
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=tokenizer_max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Create a dictionary of class labels to rounded probabilities
        probas_dict = {self.ID_TO_LABEL[idx]: round(prob.item(), decimal_places) for idx, prob in enumerate(probabilities[0])}
        predicted_class = max(probas_dict, key=probas_dict.get)
        predicted_class_proba = probas_dict[predicted_class]

        if return_class_proba:
            return predicted_class, predicted_class_proba, probas_dict
        else: 
            return predicted_class
    
    def predict(self, input_text):
        # Predict using the three models
        cb_pred = self.predict_single_input(input_text, self.cb_model, self.tokenizer_base, tokenizer_max_len=512)
        cb_1024_pred = self.predict_single_input(input_text, self.cb_model, self.tokenizer_base, tokenizer_max_len=1024)
        cbl_pred = self.predict_single_input(input_text, self.cbl_model, self.tokenizer_large, tokenizer_max_len=512)
        cbl_1024_pred = self.predict_single_input(input_text, self.cbl_1024_model, self.tokenizer_large, tokenizer_max_len=1024)

        pred_dict = {
        "non-esg": 0,
        "environnemental": 0,
        "social": 0,
        "gouvernance": 0
        }
        
        pred_dict[cb_pred] += self.weights[0][list(pred_dict.keys()).index(cb_pred)]
        pred_dict[cb_1024_pred] += self.weights[1][list(pred_dict.keys()).index(cb_1024_pred)]
        pred_dict[cbl_pred] += self.weights[2][list(pred_dict.keys()).index(cbl_pred)]
        pred_dict[cbl_1024_pred] += self.weights[3][list(pred_dict.keys()).index(cbl_1024_pred)]
        
        best_pred = max(pred_dict, key=pred_dict.get)
        
        return best_pred
    
    def predict_csv(self, filepath:str, text_column = "text"):
        
        tqdm.pandas("Predicting...")
        
        df = pd.read_csv(filepath)
        df['esg_predictor'] = df[text_column].progress_apply(self.predict)
        return df
    
    
    def predict_df(self, df,text_column = "text", fname = f" "):
        # df['esg_predictor'] = df[text_column].apply(self.predict)
        tqdm.pandas(desc=f"Predicting {fname}")
        df['esg_predictor'] = df[text_column].progress_apply(self.predict)
        return df
    
    def plot_confusion_matrix(self, y_true, y_pred):
        utilsNb.plot_confusion_matrix(y_true, y_pred, self.ID_TO_LABEL.values())
    
        
        
        
        
        
    
    
    
    
        
        
        
        
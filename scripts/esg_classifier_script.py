from src.ESGPredictor import *
# from esg_classification.ESGPredictor import ESGPredictor
import os
import argparse
import time
from tqdm import tqdm
import pandas as pd


cb_path = "../esg_classification/models/model-cb/run1/models/state_dict/model_cb_sd.pt"
cbl_path = "../esg_classification/models/model-cbl/run1/models/state_dict/cb_large_model1_sd.pt"
cbl_1024_path = "../esg_classification/models/model-cbl-long/run1/models/state_dict/cbl_model_long_sd.pt"


parser = argparse.ArgumentParser(description='Predict directory of PV files')


parser.add_argument('--input_dir', type=str, default="./data/csv_data/nyon_2023/", help='directory path to the PV files')

args = parser.parse_args()



def predict_dir(input_dir):
    
    input_dir = args.input_dir
    
    # nyon_2022_input_dir = "./data/csv_data/nyon_2022/"
    # nyon_2023_input_dir = "./data/csv_data/nyon_2023/"
    # input_dir = "./data/csv_data/vevey_2022/"
    # vevey_2023_input_dir = "./data/csv_data/vevey_2023/"
    # list_input_dirs = [nyon_2023_input_dir, vevey_2022_input_dir, vevey_2023_input_dir]
    

    predictor = ESGPredictor(cb_path, cbl_path, cbl_1024_path)
    
    output_dir = input_dir + "prediction_results/"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # 2. loop through .csv files in folder input_dir
    for filename in tqdm(os.listdir(input_dir), desc="Predicting files", total=len(os.listdir(input_dir))):
        file_path = os.path.join(input_dir, filename)  # Use os.path.join for better path handling
        if os.path.isfile(file_path):  # Check if it's a file
            print(file_path)
            df = pd.read_csv(file_path)
            df_res = predictor.predict_df(df, fname=filename)
            df_res.to_csv(os.path.join(output_dir, filename), index=False, encoding='utf-16', sep=',')
        else:
            print(f"Skipping directory: {filename}")
    
    print(f"Predictions saved to {output_dir}")


def main():
    predict_dir(args.input_dir)
    
if __name__ == "__main__":
    main()
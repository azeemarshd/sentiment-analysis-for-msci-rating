"""
Parser for Morges data regarding decisions. 
"""

import re
import pandas as pd
from pdfminer.high_level import extract_text
import os
from tqdm import tqdm

def read_pv(file_path):
    text = extract_text(file_path)
    
    return text

def extract_decision_text(text):
    bold_pattern = r"\b(décide :)"
    normal_pattern = r"\b(Ainsi délibéré le)\b"
    
    bold_match = re.search(bold_pattern, text)
    normal_match = re.search(normal_pattern, text)
    
    if bold_match and normal_match:
        start_index = bold_match.end()
        end_index = normal_match.start()
        decision_text = text[start_index:end_index].strip()
        return decision_text
    
    return None

def read_files_in_directory(directory):
        unreadable_files = []
        empty_files = []

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                text = read_pv(file_path)
                if not text:
                    empty_files.append(filename)
            except:
                unreadable_files.append(filename)

        return unreadable_files, empty_files
    
    
def create_df_from_files(decision_directory):
    """Takes all the decisions from each file in the given directory and puts them all in a dataframe. Separate column for the file name itself to identify the file origin.

    Args:
        decision_directory (str): The directory containing the decision files.

    Returns:
        pd.DataFrame: A dataframe containing the decision texts and file names.
    """
    decision_texts = []
    file_names = []

    for filename in tqdm(os.listdir(decision_directory), total=len(os.listdir(decision_directory))):
        file_path = os.path.join(decision_directory, filename)
        try:
            text = read_pv(file_path)
            decision_text = extract_decision_text(text)
            if decision_text:
                decision_texts.append(decision_text)
                file_names.append(filename)
            else:
                print(f"No decision text found in {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    df = pd.DataFrame.from_dict({'file_name': file_names, 'text': decision_texts}, )
    return df



def main():
    morges_decision_paths = "../data/raw/morges/decision/"
    
    # file_path = morges_decision_paths + "Décision du préavis N° 7 2.23 - Réponse au postulat du groupe des Vert·e·s Réduire les déchets plastiques à Morges un impératif écologique qui peut se marier avec un gain économique"
    
    # text = read_pv(file_path)
    # decision_text = extract_decision_text(text)
    # print(decision_text)
    
    df = create_df_from_files(morges_decision_paths)
    
    df.to_csv("../data/csv_data/Morges/decision_texts.csv", index=False, header=True)
    
    print(df)
    
    print([len(x) for x in df['decision_text']])
    
    

if __name__ == "__main__":
    main()



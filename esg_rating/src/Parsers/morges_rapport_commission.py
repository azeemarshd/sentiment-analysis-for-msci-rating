
import re
import pandas as pd
from pdfminer.high_level import extract_text
import os
from tqdm import tqdm

import argparse



# parser = argparse.ArgumentParser(description='Predict directory of PV files')
# parser.add_argument('--input_dir', type=str, default="./data/csv_data/nyon_2023/", help='directory path to the PV files')
# args = parser.parse_args()


class MorgesRCParser:
    def __init__(self, pdf_file_path):
        self.pdf_file_path = pdf_file_path
        self.full_text = self.read_pv()
        self.sections_list = self.extract_bold_numbered_sections()
    
    def read_pv(self):
        text = extract_text(self.pdf_file_path)
        cleaned_text = self.clean_large_spaces(text)
        return cleaned_text

    
    def clean_large_spaces(self, text):
        return re.sub(r'\s{5,}', ' ', text)
    
    def extract_section(self, section_number):
        section_number -= 1  # Adjust for 0-based indexing
        if section_number < 0 or section_number >= len(self.sections_list):
            return "Invalid section number"

        start_index = self.full_text.find(self.sections_list[section_number])

        # Check if it's the last section
        if section_number + 1 < len(self.sections_list):
            end_index = self.full_text.find(self.sections_list[section_number + 1])
        else:
            end_index = len(self.full_text)  # Use the end of the document for the last section

        if start_index != -1 and (end_index != -1 or section_number + 1 >= len(self.sections_list)):
            res_text = self.full_text[start_index:end_index].strip()
            return res_text
        return "Section not found"
    
    def extract_bold_numbered_sections(self):
        # Extracts bold numbered sections from the given text
        # pattern = re.compile(r'\n(\d+\.\s+[^\n]+)', re.MULTILINE)
        
        pattern = re.compile(r'\n(\d+\.\s+[A-Z][^\n]+)', re.MULTILINE)

        # pattern = re.compile(r'\n(\d+\s+[A-Z][^\n]+)', re.MULTILINE)
        matches = pattern.findall(self.full_text)
        return matches
    
    
    def split_text(self, text, max_length=1024):
        
        chunks = []
        
        paragraphs = text.split('\n\n\n')
        
        for para in paragraphs:
            if len(para) < max_length:
                chunks.append(para)
                     
            else:
                sentences = re.split(r'(?<=[.!?]) +', para)
                current_chunk = ""
                for sentence in sentences:
                    # Check if adding the next sentence exceeds the max_length
                    if len(current_chunk) + len(sentence) <= max_length:
                        current_chunk += sentence + " "
                    else:
                        # If the current chunk is not empty, add it to the chunks list
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "  # Start a new chunk
        return chunks
        
    def pv_to_df(self, chunk_size=None):
        rows = []
        
        if chunk_size is None:
            rows = []
            for i, section_title in enumerate(self.sections_list):
                text = self.extract_section(i + 1)
                # Each section becomes a row with "section number" and 'text' as columns
                section_number = section_title.split('.')[0]
                row = {'section_number': section_number, 'text': text}
                rows.append(row)
            return pd.DataFrame(rows)
        else:
            for i, section_title in enumerate(self.sections_list):
                text = self.extract_section(i + 1)
                # Split the section text into chunks
                chunks = self.split_text(text, max_length=chunk_size)

                # Create a row for each chunk
                for chunk in chunks:
                    # Extracting the section number from the title, assuming it's the first part before a dot
                    section_number = section_title.split('.')[0]
                    row = {'section_number': section_number, 'text': chunk}
                    rows.append(row)

            # Convert the list of dictionaries into a DataFrame
            df = pd.DataFrame(rows, columns=['section_number', 'text'])
            return df



def main():
    
    morges_rc_path = "../data/raw/morges/rapport_commission"
    test_file = "../data/raw/morges/rapport_commission/N° 13 3.23.pdf"
    
    mpv = MorgesRCParser(test_file)
    print(mpv.sections_list)
    # print(mpv.full_text)
    df = mpv.pv_to_df(chunk_size=512)
    df.to_csv("./test.csv", index=False)

if __name__ == "__main__":
    main()
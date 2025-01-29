import re
import pandas as pd
from pdfminer.high_level import extract_text



class VeveyPVParser:
    def __init__(self, pdf_file_path):
        self.pdf_file_path = pdf_file_path
        self.full_text = self.read_pv()
        self.sections_list = self.extract_bold_numbered_sections()
    

    def read_pv(self):
        text = extract_text(self.pdf_file_path)
        cleaned_text = self.clean_large_spaces(text)

        # Identify the ending marker and include everything up to this point
        end_marker = "AU NOM DU CONSEIL COMMUNAL"
        end_index = cleaned_text.find(end_marker)

        if end_index != -1:
            # Include text up to and including the end marker
            cleaned_text = cleaned_text[:end_index + len(end_marker)]

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
            # Since it's the last section and "AU NOM DU CONSEIL COMMUNAL" is part of self.full_text,
            # ensure the entire content up to this marker is included.
            end_marker = "AU NOM DU CONSEIL COMMUNAL"
            end_index = self.full_text.find(end_marker)
            
            # Assuming you want to include the end marker in the extracted text,
            # adjust the end_index to include it.
            if end_index != -1:
                end_index += len(end_marker)
        
        if start_index != -1 and end_index != -1:
            # Extract the section text, ensuring it includes everything up to the end marker for the last section.
            res_text = self.full_text[start_index:end_index].strip()
            return res_text
        
        return "Section not found"


    
    def extract_bold_numbered_sections(self):
        # Extracts bold numbered sections from the given text
        # pattern = re.compile(r'\n(\d+\.\s+[^\n]+)', re.MULTILINE)
        pattern = re.compile(r'\n(\d+\.\s+[A-Z][^\n]+)', re.MULTILINE)
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
        
    


class NyonPVParser:
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
        


class RollePVParser:
    def __init__(self, pdf_file_path):
        self.pdf_file_path = pdf_file_path
        self.full_text = self.read_pv()
        # self.sections_list = self.extract_bold_numbered_sections()
    

    def read_pv(self):
        text = extract_text(self.pdf_file_path)
        cleaned_text = self.clean_large_spaces(text)
        return cleaned_text
    
    def clean_large_spaces(self, text):
        return re.sub(r'\s{5,}', ' ', text)
    
    
    def split_text(self, text, max_length=1024):
        chunks = []
        paragraphs = text.split('\n\n')
        
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

    def to_df(self, chunk_size=512, output_csv_path=None):
        # Prepare to store chunks in DataFrame
        data = []
        start = 0
        
        while start < len(self.full_text):
            # Ensure not to exceed the bounds of the string
            end = min(start + chunk_size, len(self.full_text))
            if end < len(self.full_text):
                # If end is not the end of the text, try to break at newline or sentence end
                re.split(r'(?<=[.!?]) +', self.full_text[start:end])
                last_possible_end = self.full_text.rfind('\n', start, end + 1)
                if last_possible_end == -1:
                    last_possible_end = self.full_text.rfind('.', start, end + 1)
                    if last_possible_end == -1:
                        last_possible_end = self.full_text.rfind('!', start, end + 1)
                        if last_possible_end == -1:
                            last_possible_end = self.full_text.rfind('?', start, end + 1)
            else:
                last_possible_end = end  # If we're at the end of text, break here
            
            # Check if we found a suitable place to cut the text
            if last_possible_end != -1:
                end = last_possible_end + 1
            
            # Append the current chunk to the list
            data.append(self.full_text[start:end].strip())
            start = end  # Move the start up to the next piece
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['text'])
        
        if output_csv_path is not None:
            df.to_csv(output_csv_path, index=False)
        
        
        return df
        
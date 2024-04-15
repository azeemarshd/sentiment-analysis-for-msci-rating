import Parsers.pv_parser as pv
import pandas as pd
import glob
import os
from tqdm import tqdm



def output_csv(file_paths, filenames, output_path, parser="nyon"):
    """
    Converts PV files to CSV format and saves them to the specified output path.

    Args:
        file_paths (list): List of file paths of the PV files.
        filenames (list): List of filenames corresponding to the PV files.
        output_path (str): Path to the directory where the CSV files will be saved.
        parser (str, optional): Parser type to be used. Defaults to "nyon".

    Returns:
        None
    """
    for file, file_path in tqdm(zip(filenames, file_paths), total=len(file_paths)):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if parser == "nyon":
            p = pv.NyonPVParser(file_path)
            p.pv_to_df(chunk_size=512).to_csv(output_path + file + ".csv", index=False)
        elif parser == "vevey":
            p = pv.VeveyPVParser(file_path)
            p.pv_to_df(chunk_size=512).to_csv(output_path + file[:10] + ".csv", index=False)




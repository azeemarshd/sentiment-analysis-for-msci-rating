"""esg_daily_news dataset language translator

This script translates the dataset to french using deepl api. used ~200K characters overall.

- ESG_daily_news.csv : used Deepl api to translate the dataset to french. used ~200K characters overall.
- gold_standard_corpus: TODO. use another translator as dataset is bigger than deepl api's monthly allowance.

"""


import pandas as pd
from deep_translator import DeeplTranslator
import os

df_daily_news = pd.read_csv('./ESG_daily_news.csv')
df_daily_news.head()

def translate_text(text, src_lang='en', dest_lang='fr'):
    """
     Translate text from one language to another. This is a convenience function for use in DeepL to translate text from one language to another
     
     Args:
     	 text: The text to be translated
     	 src_lang: The source language default is en.
     	 dest_lang: The target language default is fr.
     
     Returns: 
     	 The translated text as a string or None if something went
    """
    api_key = os.getenv('DEEPL_API_KEY')
    # api_key = 'xxx'
    if not api_key:
        raise ValueError("DeepL API key not found. Set the DEEPL_API_KEY environment variable.")
    translated_text = DeeplTranslator(api_key=api_key, source=src_lang, target=dest_lang).translate(text)
    return translated_text

def translate_dataframe(df, columns_to_translate):
    """
     Translate text in a dataframe. This is a convenience function for translating text in a dataframe.
     
     Args:
     	 df: The dataframe to be translated. Should be a pandas dataframe
     	 columns_to_translate: The columns to be translated.
     
     Returns: 
     	 The dataframe with translated text in it. If there are no columns to translate the original dataframe is returned
    """
    for column in columns_to_translate:
        df[column] = df[column].apply(translate_text)
    return df


if __name__ == "__main__":
    
    input_file = './ESG_daily_news.csv' 
    output_file = 'daily_news_fr.csv'  


    df = pd.read_csv(input_file)
    columns_to_translate = ['headline', 'text']  

    # drop row with len 0 (error with deepl api)
    df = df[df['text'].str.len() > 0]

    df_translated = translate_dataframe(df, columns_to_translate) #translate df
    df_translated.to_csv(output_file, index=False)

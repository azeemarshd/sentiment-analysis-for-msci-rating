import pandas as pd
import random


def sample_headlines():
    df = pd.read_csv('./headlines_dataset/esg_headlines.csv',encoding='cp1252', low_memory=False)
    df.drop(['Unnamed: 0', 'mentions_company'], axis=1, inplace=True)

    # environemental
    df_env = df[(df['esg_category'] == 'environmental') & (df['guardian_keywords'].str.len() > 15)]

    # social
    keywords_counter = {}
    df_social = df[ df['esg_category'] == 'social']
    df_social.loc[:, 'guardian_keywords'] = df_social['guardian_keywords'].apply(lambda x: eval(x))

    # guardian keywords counter
    for row in df_social['guardian_keywords']:
        for elt in row:
            if elt in keywords_counter:
                keywords_counter[elt] += 1
            else:
                keywords_counter[elt] = 1
    print(keywords_counter)
    main_keywords = ['employment law', 'discrimination at work','gender pay gap', 'jobs']
    remaining_keywords = [k for k in keywords_counter.keys() if k not in main_keywords]

    # sample only the rows from df_social that contains the main keywords
    df_soc_main = df_social[df_social['guardian_keywords'].apply(lambda x: any(item for item in x if item in main_keywords))]
    print(f"shape of df_soc_main: {df_soc_main.shape}")

    additional_rows = 4500 - df_soc_main.shape[0]
    df_soc_remaining = pd.DataFrame()
    total_counts_remaining_keywords = sum(keywords_counter[k] for k in remaining_keywords)
    for k in remaining_keywords:
        proportion = keywords_counter[k] / total_counts_remaining_keywords
        num_rows_to_sample = round(proportion * additional_rows)
        sampled_df = df_social[df_social['guardian_keywords'].apply(lambda x: any(item for item in x if item == k))].sample(n=num_rows_to_sample, random_state=1)
        df_soc_remaining = pd.concat([df_soc_remaining, sampled_df])

    df_soc = pd.concat([df_soc_main, df_soc_remaining]).reset_index(drop=True)
    df_soc.to_csv('./headlines_dataset/esg_headlines_social.csv',index=False, encoding='utf-8', header=True, sep=',')

    print(f"shape of df_soc: {df_soc.shape}")

    # gov and non-esg
    df_gov = df[ df['esg_category'] == 'governance']
    df_other = df[ df['esg_category'] == 'non-esg']

    dfs = [df_env, df_gov, df_other]    
    # sampling n rows from each df
    for d in dfs:
        print(f"shape of {d['esg_category'].iloc[0]}: {d.shape}")
        d = d.sample(n=4500, random_state=1)
        d.to_csv(f'./headlines_dataset/esg_headlines_{d["esg_category"].iloc[0]}.csv',index=False, encoding='utf-8', header=True, sep=',')

def split_dataset(fname:str, output:str, N:int = 5):
    "split dataset into smaller N sized datasets"
    
    # read the dataset
    df = pd.read_csv(fname,encoding='utf-8', low_memory=False)
    
    # split it smaller same sized N datasets
    dfs = [df[i:i+N] for i in range(0,df.shape[0], N)]
    
    # save each split into a csv file
    for i, d in enumerate(dfs):
        d.to_csv(f'{output}_{i+1}.csv',index=False, encoding='utf-8', header=True, sep=',')
    
def join_csv_files(files:list , output_file:str) -> None : 
    df = pd.concat([pd.read_csv(f, encoding="utf-8") for f in files])
    df.to_csv(output_file, index=False, encoding="utf-8")
    return 0

def is_text_within_limit(text, max_position_embeddings=1024):
    """
    Check if the text is within the specified limit of max_position_embeddings.

    Args:
    text (str): The text to be checked.
    max_position_embeddings (int): The maximum number of position embeddings (default: 1024).

    Returns:
    bool: True if the text is within the limit, False otherwise.
    """
    
    tokens = text.split() # A simple whitespace tokenizer
    l = len(tokens)
    return l, len(tokens) <= max_position_embeddings
    
    
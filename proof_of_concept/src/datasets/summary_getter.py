from duckduckgo_search import DDGS
from newspaper import Article
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os 
import traceback
import argparse

import random
import pyautogui 
import time

def get_article_url(query:str)->str:
    """
     Get link to article. This is a wrapper around DDGS. text () to get the link to the article
     
     Args:
     	query: the query to search for
     
     Returns: 
     	the link to the article or None if not found 
    """
    
    with open("./user_agents.txt", "r") as f:
        user_agents_list = f.readlines()
    
    headers = {
    'dnt': '1',
    'upgrade-insecure-requests': '1',
    'user-agent': random.choice(user_agents_list).strip(),
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'none',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-user': '?0',
    'sec-fetch-dest': 'document',
    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'accept-encoding': 'gzip, deflate, br',
    'cache-control': 'max-age=0',
    'connection': 'keep-alive'
    }

    results = []
    try:
        ddgs = DDGS(headers= headers)
        # with DDGS(headers= headers) as ddgs: #MODIFICATION [HEADERS PRESENT]
        results = [r for r in ddgs.text(query + " - the guardian", max_results=3)]
        if results:
            for result in results:
                link = result['href']
                if "theguardian" in link:
                    return link  # Return the first valid link found
            print(f"No valid Guardian link found for query: {query}")
            return None
        else:
            print(f"No results found for query: {query}")
            return None
    except Exception as e:
        print(f"Error occurred in get_article_url for query '{query}': {e}")
        print(traceback.format_exc())
        print(f"results: {results}")
        return None


def get_article_summary(url: str) -> str:
    """
     Get the summary of an article. This is a wrapper around Article. download () and Article. parse ()
     
     Args:
     	url: URL of the article to retrieve
     
     Returns: 
     	String representation of the article's summary or None if there was an error fetching the article from the url.
    """
    
    if not url:
        return None
    
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        return article.summary
    except Exception as e:
        # print(f"An error occurred while fetching the article summary: {type(e).__name__}, {e}")
        print(f"An error occurred while fetching the article summary")
        return None



def get_article_url_and_summary(data: tuple) -> tuple:
    headline, esg_category = data
    url = get_article_url(headline)
    summary = get_article_summary(url) if url else None
    return (headline, esg_category, summary, url)

def count_empty_rows(df):
    count = 0
    for i in range(len(df)):
        if type(df['text'][i]) != str:
            count += 1
    return count

def get_missed_articles(fname: str, output_file: str, threshold: int = 50, loops_limit: int = 5):
    
    df = pd.read_csv(fname, low_memory=False, encoding='utf-8')
    
    total_missing = count_empty_rows(df)
    print(f"Total missing: {total_missing}")
    loops = 0

    try:
        while (total_missing > threshold) or (loops < loops_limit):
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Fetching missing articles [{total_missing}]"):
                if pd.isnull(row['text']):
                    _, _, summary, url = get_article_url_and_summary((row['headline'], row['esg_category']))
                    df.loc[i, 'text'] = summary
                    df.loc[i, 'url'] = url
            
            total_missing = count_empty_rows(df)
            loops += 1
            df.to_csv(output_file, index=False, header=True)

    except KeyboardInterrupt:
        # Save progress to a temporary file
        temp_output_file = output_file.replace('.csv', '_temp.csv')
        df.to_csv(temp_output_file, index=False, header=True)
        print(f"\nLoop interrupted by user. Progress saved to {temp_output_file}.")

    df.to_csv(output_file, index=False, header=True)



def main(fname,output_file, N = 10, max_requests = 5):
    df_gsc = pd.read_csv(fname, low_memory=False, encoding='utf-8')

    # Create tuples of (headline, esg_category)
    # data_to_process = list(zip(df_gsc['headline'].to_list()[:N], df_gsc['esg_category'].to_list()[:N]))
    data_to_process = list(zip(df_gsc['headline'].to_list(), df_gsc['esg_category'].to_list()))
    final_results = []
    request_count = 0  

    with ThreadPoolExecutor(max_workers= 8 ) as executor:  
        future_to_data = {executor.submit(get_article_url_and_summary, data): data for data in data_to_process}

        for future in tqdm(as_completed(future_to_data), total=len(data_to_process), desc="Fetching summaries"):
            try:
                headline, esg_category, summary, url = future.result()
                final_results.append({'headline': headline, 'esg_category': esg_category, 'text': summary, 'url': url})
                request_count += 1  # Increment the request count
                
                # Move the mouse to a specific position
                # pyautogui.moveTo(random.randint(0,700), random.randint(0,100), duration=random.uniform(0,2))  # Move to (100, 100) over one second
                # Move the mouse relative to its current position
                # pyautogui.moveRel(random.randint(0,700), random.randint(0,100), duration=random.uniform(0,3))  # Move right 50 pixels over half a second
                
                if request_count % 15 == 0:
                    time.sleep(random.randint(1, 4))

                # if request_count % max_requests == 0:  # Pause every max_requests
                #     input(f"Reached {request_count} requests. Please change the VPN server and press Enter to continue...")
                    
            except Exception as e:
                print(f"An error occurred: {type(e).__name__}, {e}")

    final_df = pd.DataFrame(final_results)
    final_df.to_csv(f'{output_file}', index=False, header=True)
    
    


if __name__ == "__main__":
    
    # ------ arguments ------
    # parser = argparse.ArgumentParser(description="Process some articles.")
    # parser.add_argument('fname', type=str, help='The file name containing headlines and categories')
    # parser.add_argument('--N', type=int, default=10, help='Number of headlines to process (default: 10)')
    # parser.add_argument('--max', type=int, default=5, help='Maximum number of requests before pausing (default: 100)')
    # parser.add_argument('--output', type=str, default='./gsc_results/article_summaries.csv', help='Output file name (default: ./gsc_results/article_summaries.csv)')
    
    # args = parser.parse_args()
    
    
    # ------ manual input ------
    # main(args.fname, N=args.N, max_requests=args.max, output_file=args.output)
    
    # ------ Manual input with split dataset ------
    # input_fname_list = [
    #     "./headlines_dataset/esg_headlines_governance.csv",
    #     "./headlines_dataset/esg_headlines_other.csv"
    # ]
    
    # dataset_output_file_list = [
    #     "./headlines_dataset/results/esg_governance/esg_gov_full.csv",
    #     "./headlines_dataset/results/esg_other/esg_other_full.csv"
    # ]
    
    # for i, input_fname in enumerate(input_fname_list):
    #     output_file = dataset_output_file_list[i]
    #     main(input_fname, output_file)
    
    
    # ------ get missed articles ------
    
    fname = "./headlines_dataset/results/esg_other/esg_other_full_v2.csv"
    output_file = "./headlines_dataset/results/esg_other/esg_other_full_v2.csv"

    get_missed_articles(fname, output_file, threshold=200, loops_limit=3)
    
    
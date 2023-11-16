from duckduckgo_search import DDGS
from newspaper import Article
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os 
import traceback
import pprint



def get_article_url(query:str)->str:
    """
     Get link to article. This is a wrapper around DDGS. text () to get the link to the article
     
     Args:
     	query: the query to search for
     
     Returns: 
     	the link to the article or None if not found 
    """
    
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.duckduckgo.com",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache"
    }

    results = []
    try:
        with DDGS(headers= headers) as ddgs:
            results = [r for r in ddgs.text(query + " - the guardian", max_results=5, region='uk-en', )]
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
    # article = Article(url)
    # article.download()
    # article.parse()
    # article.nlp()
    # return article.summary
    
    if not url:
        return None
    
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        return article.summary
    except Exception as e:
        print(f"An error occurred while fetching the article summary: {type(e).__name__}, {e}")
        return None



def get_article_url_and_summary(data: tuple) -> tuple:
    headline, esg_category = data
    url = get_article_url(headline)
    summary = get_article_summary(url) if url else None
    return (headline, esg_category, summary, url)



def main(fname, N = 10, max_requests = 100):
    df_gsc = pd.read_csv(fname, low_memory=False)

    # Create tuples of (headline, esg_category)
    data_to_process = list(zip(df_gsc['headline'].to_list()[:N], df_gsc['esg_category'].to_list()[:N]))
    final_results = []
    request_count = 0  

    with ThreadPoolExecutor(max_workers= os.cpu_count()-2 ) as executor:  
        future_to_data = {executor.submit(get_article_url_and_summary, data): data for data in data_to_process}

        for future in tqdm(as_completed(future_to_data), total=len(data_to_process), desc="Fetching summaries"):
            try:
                headline, esg_category, summary, url = future.result()
                final_results.append({'headline': headline, 'esg_category': esg_category, 'text': summary, 'url': url})
                request_count += 1  # Increment the request count

                if request_count == max_requests:  # Check if the request count has reached 500
                    # Add your conditional statement here
                    
                    # TODO: Change VPN server/ ip rotation/...??
                    print(f"Reached {max_requests} requests")
                    input("vpn server successfully changed. Press Enter to continue...")
                    
            except Exception as e:
                print(f"An error occurred: {type(e).__name__}, {e}")

    final_df = pd.DataFrame(final_results)
    final_df.to_csv(f'./gsc_results/article_summaries_{N}.csv', index=False, header=True)

if __name__ == "__main__":
    fname = './esg_only_headlines.csv'
    
    main(fname, N = 10, max_requests = 4)
    
    
    
    
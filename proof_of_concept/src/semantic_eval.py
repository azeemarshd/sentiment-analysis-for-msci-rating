import pprint
import re

import nltk
import pandas as pd
import tqdm
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util

# Ensure you've downloaded the necessary resources
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('french'))

key_terms = {
    "Pillier Environnemental": {
        "Changement climatique": [
            "émissions de carbone",
            "empreinte carbone",
            "financement de l'impact environnemental",
            "vulnérabilité du changement climatique"
        ],
        "Capital Naturel": [
            "stress hydrique",
            "pénurie d'eau",
            "rareté de l'eau",
            "biodiversité et utilisation des terres",
            "approvisionnement des matières premières"
        ],
        "pollution et déchets":[
            "émissions toxiques et déchets",
            "emballage et déchets",
            "déchets électroniques"
        ],
        "Opportunités Environnementals":[
            "technologies propres",
            "bâtiment écologique",
            "énergie renouvelable"
        ]
    },
    "Pillier Sociale": {
        "capital humain": [
            "gestion du travail",
            "santé et sécurité",
            "développement du capital humain",
            "normes de travail dans la chaîne d'approvisionnement",
        ],
        "Responsabilité du produit":[
            "Sécurité et qualité des produits",
            "Sécurité chimique",
            "Protection financière des consommateurs",
            "Confidentialité et sécurité des données",
            "Investissement responsable",
            "Assurance santé et risque démographique"
        ],
        "Opposition des parties prenantes":[
            "Sourcing controversé",
            "Relations avec la communauté"
        ],
        "Opportunités sociales":[
            "Accès à la communication",
            "Accès à au services financiers",
            "Accès aux soins de santé",
            "Opportunités en nutrition et santé"
        ]
    },
    "Pillier de la gouvernance": {
        "Gouvernance d'entreprise":[
            "Conseil d'administration",
            "Rémunération",
            "Propriété",
            "Comptabilité"
        ],
        "Comportement d'entreprise":[
            "Éthique des affaires",
            "Transparence fiscale"
        ]
    }
}


def combine_subkey_values(dict_input):
    """
    Combine the values of each subkey into a single string
    """
    dict_output = {}
    for key, subdict in dict_input.items():
        new_subdict = {}
        for subkey, values in subdict.items():
            # join the subkey and its list into a single string
            new_subdict[subkey] = f"{subkey}; " + '; '.join(values)
        dict_output[key] = new_subdict
    return dict_output

def get_subkey(dict_input, value_to_find):
    for key, subdict in dict_input.items():
        for subkey, value in subdict.items():
            if value == value_to_find:
                return subkey
    return None


def clean_text(text,remove_stopwords = False):
    """
    Applies some pre-processing on the given text.
    """
    
    # TODO: ADD LEMMATIZATION
    
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    # text = re.sub(r'[^\w\s]', '', text)
    
    if remove_stopwords:
        text = ' '.join([word for word in text.split() if word not in stop_words])

    # # Stemming
    # stemmer = SnowballStemmer(language='french')
    # tokens = [stemmer.stem(token) for token in tokens]

    return text

def sentence_tokenizer(text):
    
    sentences = sent_tokenize(text, language='french')
    return sentences

def paragraph_tokenizer(text, min_length=50, threshold=1000):
    """
    This function should tokenize paragraphs, which are separated by newlines (\n or \n\n).

    Args:
        text (str): text to tokenize
        min_length (int): minimum length of a paragraph to include
        threshold (int): maximum number of characters in a paragraph
    """
    
    raw_paragraphs = re.split('\n{2,}', text)
    paragraphs = []
    
    for raw_paragraph in raw_paragraphs:
        if len(raw_paragraph) <= threshold and len(raw_paragraph) >= min_length:
            paragraphs.append(raw_paragraph)
        elif len(raw_paragraph) > threshold:
            sentences = sentence_tokenizer(raw_paragraph)
            temp_paragraph = ''

            for sentence in sentences:
                if len(temp_paragraph) + len(sentence) > threshold:
                    paragraphs.append(temp_paragraph)
                    temp_paragraph = sentence
                else:
                    temp_paragraph += ' ' + sentence
            
            # If there's any remaining content in temp_paragraph after the loop ends, add it as a paragraph
            if temp_paragraph:
                paragraphs.append(temp_paragraph)
    
    # Remove any empty paragraphs that may have been created
    paragraphs = [paragraph for paragraph in paragraphs if paragraph.strip()]
    
    return paragraphs



def extract_text_from_pdf(pdf_file_path):
    return extract_text(pdf_file_path)

def extract_section(text, start_phrase, end_phrase):
    start_index = text.find(start_phrase)
    end_index = text.find(end_phrase)

    if start_index != -1 and end_index != -1:  # make sure both phrases were found
        res_text = text[start_index:end_index].strip()
        return res_text
    return "Section not found"

def find_top_level_key(key_terms, subkey_to_find):
    for key, subdict in key_terms.items():
        if subkey_to_find in subdict:
            return key
    return None

def get_results_dataframe(results_dict, sort='avg_score'):
    """
    Returns a dataframe of the results dictionary, sorted by the given sort key
    """
    avg_scores = [subdict['avg_score'] for key, subdict in results_dict.items()]
    avg_scores.sort(reverse=True)
    
    ordered_results_dict = {}
    for avg_score in avg_scores:
        for key, subdict in results_dict.items():
            if subdict['avg_score'] == avg_score:
                ordered_results_dict[key] = subdict
                    
    df = pd.DataFrame.from_dict(ordered_results_dict, orient='index')
    df = df.sort_values(by=[sort], ascending=False)
    df = df.reset_index()
    df = df.rename(columns={'index': 'key_term'})
    
    for kt in df['key_term'].unique():
        top_level_key = find_top_level_key(key_terms, kt)
        if top_level_key is not None:
            new_kt = kt + " [" + top_level_key + "]"
            df.loc[df['key_term'] == kt, 'key_term'] = new_kt
    return df


def get_results_dataframe_lsi(results_dict):
    """
    Returns a dataframe of the results dictionary.
    """
    flattened_data = []
    for sentence, subdict in results_dict.items():
        flattened_data.append({
            'index': subdict['index'],
            'text': sentence,
            'key_term': subdict['key_term'],
            'score': subdict['score']
        })

    df = pd.DataFrame(flattened_data)
    df = df.sort_values(by='index')
    df = df.reset_index(drop=True)
    
    for kt in df['key_term'].unique():
        top_level_key = find_top_level_key(key_terms, kt)
        if top_level_key is not None:
            new_kt = kt + " [" + top_level_key + "]"
            df.loc[df['key_term'] == kt, 'key_term'] = new_kt

    return df

def compute_baseline(pdf_file_path):
    """Compute baseline with LSI

    Args:
        pdf_file_path (str): PV file path

    Returns:
        dict: results dictionary
    """
    new_key_terms = combine_subkey_values(key_terms)
    MIN_LENGTH = 150
    MAX_LENGTH = 1024

    # pdf_file_path = "./ccpv230130.pdf"
    pv_full = extract_text_from_pdf(pdf_file_path)
    # pv_clean = clean_text(pv_full)

    # paragraphs = paragraph_tokenizer(pv_clean,min_length=MIN_LENGTH, threshold=MAX_LENGTH)
    paragraphs = [clean_text(paragraph) for paragraph in paragraph_tokenizer(pv_full, min_length=MIN_LENGTH, threshold=MAX_LENGTH)]
    print(f"Length of paragraphs: {len(paragraphs)}")
    
    key_terms_list = [value for key, subdict in new_key_terms.items() for subkey, value in subdict.items()]

    dictionary = corpora.Dictionary([word_tokenize(p) for p in paragraphs + key_terms_list])

    paragraph_bow = [dictionary.doc2bow(word_tokenize(p)) for p in paragraphs]
    
    print(f"Length of paragraph_bow: {len(paragraph_bow)}")
    
    
    
    key_term_bow = [dictionary.doc2bow(word_tokenize(kt)) for kt in key_terms_list]

    model = models.LsiModel(paragraph_bow + key_term_bow, id2word=dictionary, num_topics=100)
    index = similarities.MatrixSimilarity(model[paragraph_bow + key_term_bow])

    # results = {}
    # for i, kt_bow in enumerate(key_term_bow):
    #     sims = index[model[kt_bow]]
    #     sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    #     key_term = get_subkey(new_key_terms, key_terms_list[i])

    #     for doc_position, doc_score in sorted_sims:
    #         if doc_position < len(paragraphs):
    #             paragraph = paragraphs[doc_position]
    #             if paragraph not in results or doc_score > results[paragraph]['score']:
    #                 results[paragraph] = {'key_term': key_term, 'score': doc_score,'index': doc_position}
    
   # Initialize the results dictionary with all paragraphs
    results = {f"{paragraph}_{i}": {'key_term': None, 'score': 0, 'index': i} for i, paragraph in enumerate(paragraphs)}

    # Then later in your code...
    for i, paragraph_bow in enumerate(paragraph_bow):
        sims = index[model[paragraph_bow]]
        sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])

        for kt_position, kt_score in sorted_sims:
            if kt_position >= len(paragraphs):  # we only consider the key terms, which are after the paragraphs in the sims list
                key_term = get_subkey(new_key_terms, key_terms_list[kt_position - len(paragraphs)])  # adjust the index for key_terms_list
                paragraph = f"{paragraphs[i]}_{i}"
                if kt_score > results[paragraph]['score']:
                    results[paragraph] = {'key_term': key_term, 'score': kt_score,'index': i}



    print(f"Length of results: {len(results)}")
    return results




def main():
    print("Preprocessing...")
    
    MIN_LENGTH = 150
    MAX_LENGTH = 1024

    pdf_file_path = "./ccpv230130.pdf"
    pv_full = extract_text_from_pdf(pdf_file_path)

    pv_intro = extract_section(pv_full, start_phrase="SEANCE DU CONSEIL COMMUNAL", end_phrase="RAPPORTS DE COMMISSIONS")
    pv_rapport_de_commissions = extract_section(pv_full, start_phrase="RAPPORTS DE COMMISSIONS", end_phrase="DEPÔT DE PREAVIS")
    pv_depot_de_preavis = extract_section(pv_full, start_phrase="DEPÔT DE PREAVIS", end_phrase="CONSEIL COMMUNAL DE NYON")
    
    pv_clean = [clean_text(paragraph) for paragraph in paragraph_tokenizer(pv_full, min_length=MIN_LENGTH, threshold=MAX_LENGTH)]
    pv_intro_clean = [clean_text(paragraph) for paragraph in paragraph_tokenizer(pv_intro, min_length=MIN_LENGTH, threshold=MAX_LENGTH)]
    pv_rdc_clean = [clean_text(paragraph) for paragraph in paragraph_tokenizer(pv_rapport_de_commissions, min_length=MIN_LENGTH, threshold=MAX_LENGTH)]
    pv_ddp_clean = [clean_text(paragraph) for paragraph in paragraph_tokenizer(pv_depot_de_preavis, min_length=MIN_LENGTH, threshold=MAX_LENGTH)]
    
    # print(f"length of pv_clean with minimum {MIN_LENGTH} and maximum {MAX_LENGTH} chars: {len(pv_clean)}")

    print(f"Length of paragraphs: {len(pv_clean)}")
    
    
    new_key_terms = combine_subkey_values(key_terms)
        
    print("Loading model...")
    model = SentenceTransformer("Sahajtomar/french_semantic")

    key_terms_sample = [value for key, subdict in new_key_terms.items() for subkey, value in subdict.items()]

    # pv_paragraphs = pv_intro_clean + pv_rdc_clean + pv_ddp_clean
    pv_paragraphs = pv_clean.copy()
    
    print("Embedding...")
    embeddings1 = model.encode(key_terms_sample, convert_to_tensor=True)
    embeddings2 = model.encode(pv_paragraphs, convert_to_tensor=True)
    
    print("Calculating similarity...")
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    
    res_avg = []
    for i in range(len(key_terms_sample)):
        tmp_res = 0
        
        for j in range(len(pv_paragraphs)):
            tmp_res += cosine_scores[i][j]
            
        avg_score = tmp_res / len(pv_paragraphs)
        res_avg.append({'key_term': key_terms_sample[i], 'avg_score': avg_score.item()})

    res_max = []
    for i in range(len(key_terms_sample)):
        max_score = max(cosine_scores[i])
        res_max.append({'key_term': key_terms_sample[i], 'max_score': max_score.item()})
        
    results_dict = {get_subkey(new_key_terms, key_term): {'avg_score': res_avg[i]['avg_score'],
                                                          'max_score': res_max[i]['max_score'],
                                                          'paragraphs': []} for i, key_term in enumerate(key_terms_sample)}
    
    
    individual_para_scores = []
    print("Classifying paragraphs...")
    for j in range(len(pv_paragraphs)):
        max_index = cosine_scores[:, j].argmax()
        max_score = cosine_scores[max_index][j].item()
        key_term = key_terms_sample[max_index]
        paragraph = pv_paragraphs[j]
        short_key_term = get_subkey(new_key_terms, key_term)

        # Add the paragraph to the list of paragraphs for the key term
        results_dict[short_key_term]['paragraphs'].append(paragraph)
        
        individual_para_scores.append({'paragraph': paragraph,
                                       'score': max_score,
                                       'key_term': short_key_term})
        
        df = pd.DataFrame(individual_para_scores)
        
        for kt in df['key_term'].unique():
            top_level_key = find_top_level_key(key_terms, kt)
            if top_level_key is not None:
                new_kt = kt + " [" + top_level_key + "]"
                df.loc[df['key_term'] == kt, 'key_term'] = new_kt
        
 
    # return results_dict,pd.DataFrame(individual_para_scores).sort_values(by="score", ascending=False)
    return results_dict,df






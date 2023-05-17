import pprint
import re

import nltk
import pandas as pd
import tqdm
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
    
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    if remove_stopwords:
        text = ' '.join([word for word in text.split() if word not in stop_words])

    # # Stemming
    # stemmer = SnowballStemmer(language='french')
    # tokens = [stemmer.stem(token) for token in tokens]

    return text

def sentence_tokenizer(text):
    
    sentences = sent_tokenize(text, language='french')
    return sentences


def extract_text_from_pdf(pdf_file_path):
    return extract_text(pdf_file_path)

def extract_section(text, start_phrase, end_phrase):
    start_index = text.find(start_phrase)
    end_index = text.find(end_phrase)

    if start_index != -1 and end_index != -1:  # make sure both phrases were found
        res_text = text[start_index:end_index].strip()
        return res_text
    return "Section not found"


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
    
    return df



def main():
    print("Preprocessing...")

    pdf_file_path = "./ccpv230130.pdf"
    pv_full = extract_text_from_pdf(pdf_file_path)

    pv_intro = extract_section(pv_full, start_phrase="SEANCE DU CONSEIL COMMUNAL", end_phrase="RAPPORTS DE COMMISSIONS")
    pv_rapport_de_commissions = extract_section(pv_full, start_phrase="RAPPORTS DE COMMISSIONS", end_phrase="DEPÔT DE PREAVIS")
    pv_depot_de_preavis = extract_section(pv_full, start_phrase="DEPÔT DE PREAVIS", end_phrase="CONSEIL COMMUNAL DE NYON")
    
    with open('./pv/pv_full.txt', 'w', encoding='utf-8') as f:
        f.write(pv_full)

    with open('./pv/pv_intro.txt', 'w', encoding='utf-8') as f:
        f.write(pv_intro)
        
    with open('./pv/pv_rapport_de_commissions.txt', 'w', encoding='utf-8') as f:
        f.write(pv_rapport_de_commissions)

    
    pv_rapport_de_commissions
    pv_clean = [clean_text(sentence) for sentence in sent_tokenize(pv_full, language='french')]
    pv_intro_clean = [clean_text(sentence) for sentence in sent_tokenize(pv_intro, language='french')]
    pv_rdc_clean = [clean_text(sentence) for sentence in sent_tokenize(pv_rapport_de_commissions, language='french')]
    pv_ddp_clean = [clean_text(sentence) for sentence in sent_tokenize(pv_depot_de_preavis, language='french')]
    
    pv_intro_paragraphs = pv_intro.split('\n')
    pv_intro_paragraphs_clean = [clean_text(paragraph) for paragraph in pv_intro_paragraphs]
    pv_rdc_paragraphs = pv_rapport_de_commissions.split('\n')
    pv_rdc_paragraphs_clean = [clean_text(paragraph) for paragraph in pv_rdc_paragraphs]
    pv_ddp_paragraphs = pv_depot_de_preavis.split('\n')
    pv_ddp_paragraphs_clean = [clean_text(paragraph) for paragraph in pv_ddp_paragraphs]
    
    
    new_key_terms = combine_subkey_values(key_terms)
    
    
    
    print("Loading model...")
    model = SentenceTransformer("Sahajtomar/french_semantic")

    key_terms_sample = [value for key, subdict in new_key_terms.items() for subkey, value in subdict.items()]

    pv_sentences = pv_intro_clean + pv_rdc_clean + pv_ddp_clean
    
    print("Embedding...")
    embeddings1 = model.encode(key_terms_sample, convert_to_tensor=True)
    embeddings2 = model.encode(pv_sentences, convert_to_tensor=True)
    
    print("Calculating similarity...")
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    
    
    
    res_avg = []
    for i in range(len(key_terms_sample)):
        tmp_res = 0
        
        for j in range(len(pv_sentences)):
            tmp_res += cosine_scores[i][j]
            
        avg_score = tmp_res / len(pv_sentences)
        res_avg.append({'key_term': key_terms_sample[i], 'avg_score': avg_score.item()})

    res_max = []
    for i in range(len(key_terms_sample)):
        max_score = max(cosine_scores[i])
        res_max.append({'key_term': key_terms_sample[i], 'max_score': max_score.item()})
        
        
    results_dict = {get_subkey(new_key_terms, key_term): {'avg_score': res_avg[i]['avg_score'],
                                                          'max_score': res_max[i]['max_score'],
                                                          'sentences': []} for i, key_term in enumerate(key_terms_sample)}
    
    print("Classifying sentences...")
    for j in range(len(pv_sentences)):
        max_index = cosine_scores[:, j].argmax()
        max_score = cosine_scores[max_index][j]
        key_term = key_terms_sample[max_index]
        sentence = pv_sentences[j]
        short_key_term = get_subkey(new_key_terms, key_term)

        # Add the sentence to the list of sentences for the key term
        results_dict[short_key_term]['sentences'].append(sentence)
    
    return results_dict


# if __name__ == "__main__":
#     main()
  


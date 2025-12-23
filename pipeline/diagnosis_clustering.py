from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import hdbscan
import pandas as pd
import numpy as np
import collections
import re
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

from transformers import AutoModel, AutoTokenizer

import torch

from utils.utils import _get_device

tqdm.pandas()

ITALIAN_STOPWORDS = set(stopwords.words('italian'))

def load_umberto(name):
    if name=='umberto-base-uncased':
        model = AutoModel.from_pretrained("Musixmatch/umberto-wikipedia-uncased-v1")
        tokenizer = AutoTokenizer.from_pretrained("Musixmatch/umberto-wikipedia-uncased-v1")
    else:
        model = AutoModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1")
    return model,tokenizer

def get_umberto_emb(text, model, tokenizer):
    device = _get_device()

    # Tokenize with truncation to avoid sequences > max length
    encoded = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=False
    ).to(device)

    outputs = model(**encoded)              # forward pass
    last_hidden = outputs.last_hidden_state # shape: (1, seq_len, hidden_dim)
    emb = last_hidden.mean(dim=1).squeeze(0)  # mean pooling over tokens

    return emb


def add_bert_embeddings(model, tokenizer, df_testo, text_col):
    
    model.eval()
    model.to(_get_device())
    with torch.no_grad():
        df_testo['Umberto'] = df_testo[text_col].progress_apply(lambda x: get_umberto_emb(x, model, tokenizer))
    umb_dict = {str(key): dict(enumerate(emb.cpu().numpy())) for (key, emb) in dict(enumerate(df_testo['Umberto'])).items()}
    umb_df = pd.DataFrame.from_dict(umb_dict, orient='index')
    column_remapping = {i:'U'+str(i) for i in range(len(umb_df.columns))}
    umb_df.rename(columns=column_remapping, inplace=True)
    df_testo = df_testo[df_testo.columns.drop(list(df_testo.filter(regex='U\d+')))]
    df_testo.drop('level_0', axis=1, inplace=True, errors='ignore')
    return pd.concat([df_testo.reset_index(), umb_df.reset_index()], axis=1)

def my_textrank(df, text_col, threshold=0.8, min_keywords=2):
    txt = df[text_col].str.cat(sep='. ')
    if not txt:
        return ['']
    
    words = re.findall(r"\w+", txt.lower())
    words = [w for w in words if w not in ITALIAN_STOPWORDS]
    if not words:
        return ['']

    total = len(words)
    word_freq = collections.Counter(words)
    word_freq_dict = {k: v/total for k, v in word_freq.items()}

    freq_words = []
    thr = threshold
    while len(freq_words) < min_keywords and thr > 0.001:
        freq_words = [k for k, v in word_freq_dict.items() if v > thr]
        thr /= 2

    return freq_words if freq_words else ['']
    
def get_mapping_bronchio(keywords):
    disease_keywords = [
        {
            'positive': ['bronchiolite'],
            'negative': [],
        },
        {
            'positive' : ['broncospasmo', 'febbre'],
            'negative' : [],
        }
    ]
    for rule in disease_keywords:
        # all positive keywords must appear
        positives_ok = all(
            any(pk in kw for kw in keywords)
            for pk in rule['positive']
        )
        
        # no negative keyword must appear
        negatives_ok = not any(
            nk in kw
            for nk in rule['negative']
            for kw in keywords
        )

        if positives_ok and negatives_ok:
            return True
    return False


def cluster_hdbscan(X_red, df, text_col, min_cluster_size_start, min_cluster_size_end, cluster_col, min_samples=1, cluster_selection_epsilon=0.00):

    unclustered=X_red.index
    n_clusters = 0
    min_cluster_size = min_cluster_size_start
    print(f'Starting clustering min cluster size={min_cluster_size_start} to {min_cluster_size_end}...')
    while min_cluster_size >= min_cluster_size_end:
        distances = pairwise_distances(X_red.loc[unclustered], metric='cosine')
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed', min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)
        cluster_labels = clusterer.fit_predict(distances)
        cluster_labels = [c+n_clusters if c!=-1 else c for c in cluster_labels]
        df.loc[unclustered, cluster_col] = cluster_labels
        unclustered = df[df[cluster_col] == -1].index
        n_clusters = max(cluster_labels)+1
        min_cluster_size //= 2
    print(f'Finished clustering min cluster size={min_cluster_size_start} to {min_cluster_size_end}.')

    ordered_clusters = np.sort(df[cluster_col].unique())
    n_clusters = len(ordered_clusters)

    
    summary = {}
    for j in tqdm(range(n_clusters), desc='Summarizing clusters'):
        summary[ordered_clusters[j]] = my_textrank(df[df[cluster_col]==ordered_clusters[j]], text_col)
   
    df['summary_'+cluster_col] = df[cluster_col].map(lambda x: ';'.join(summary[x]))

    print('Finshed summary of clusters.')

    mapping_clusters_labels = {}
    for k, v in summary.items():
        # `v` is a string or a list of keywords
        label = get_mapping_bronchio(v)  # This returns a list or None
        mapping_clusters_labels[k] = label


    df['label_' + cluster_col] = df[cluster_col].map(mapping_clusters_labels)
    
    print('Finished assignments of labels to clusters.')
    return df



def run_clustering(df, text_col, cluster_col, embedding_model, min_cluster_size_start, min_cluster_size_end, min_samples=1, cluster_selection_epsilon=0.00):
    model, tokenizer = load_umberto(embedding_model)
    df = add_bert_embeddings(model, tokenizer, df, text_col)
    embedding_cols = [c for c in df.columns if re.match(r'U\d+', c)]
    X = df.filter(regex=("U\d.*"))
    pca = PCA(n_components=50)
    X_red = pca.fit_transform(X)
    X_red = pd.DataFrame(X_red, index=X.index)
    df = cluster_hdbscan(X_red, df, text_col, min_cluster_size_start, min_cluster_size_end, cluster_col, min_samples, cluster_selection_epsilon)
    return df

#%%
from rouge_score import rouge_scorer
from bert_score import score as bert_scorer
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os



def semantic_clustering(df_ext,rouge,threshold =0.9):
    df = df_ext.copy()
    similar_clusters = {}
    for idx in rouge.columns:
        similar_ = [i for i, sim in enumerate(rouge[idx]) if sim >= threshold and i != idx]
        if similar_:
            similar_clusters[idx] = similar_
    
    edges = [(k, node) for k in similar_clusters for node in similar_clusters[k]]
    # Create a parent dictionary
    parent = {node: node for edge in edges for node in edge}
    # Function to find the parent of a node
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]
    # Function to union two nodes
    def union(node1, node2):
        parent[find(node1)] = find(node2)
    # Perform union on all edges
    for node1, node2 in edges:
        union(node1, node2)
    # Group nodes by their parent
    groups = {}
    for node in parent:
        root = find(node)
        if root not in groups:
            groups[root] = []
        groups[root].append(node)
    result = {min(value): sorted(set(value) - {min(value)}) for value in groups.values()}
    # Print the result
    for key, value in result.items():
        print(f"{key} {value}")
    inverse_d = {v: k for k, values in result.items() for v in values}
    df['cluster'] = df['cluster'].replace(inverse_d)
    grouped_df = df.groupby('cluster',as_index=False)['caption'].apply(lambda x: ('. ').join(x))
    return df, grouped_df

#%%
'''
captions_path = '../data/frankfurt/cleaned_captions_frankfurt.csv'
captions = pd.read_csv(captions_path)
grouped = captions.groupby('cluster').agg({'caption': '. '.join}).reset_index()
### eliminate outliers ###
grouped = grouped[grouped.cluster != -1]
### compute similarity matrix ###
'''
def compute_sim_matrix_old(df, similarity_func='rouge1'):

    if similarity_func in ['rouge1', 'rougeL', 'bleu']:
        scorer = rouge_scorer.RougeScorer([similarity_func], use_stemmer=True)
    elif similarity_func == 'bert':
        scorer = bert_scorer
    else:
        raise ValueError('Invalid similarity function')
    
    similarity_matrix = pd.DataFrame(index=df['cluster'], columns=df['cluster'])
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            similarity_matrix.iloc[i, j] = scorer.score(grouped.iloc[i].caption,grouped.iloc[j].caption)['rouge1'].precision
    return similarity_matrix

def compute_sim_matrix(df, similarity_func='rouge1'):

    if similarity_func in ['rouge1', 'rougeL', 'bleu']:
        scorer = rouge_scorer.RougeScorer([similarity_func], use_stemmer=True)
    elif similarity_func == 'bert':
        scorer = bert_scorer
    else:
        raise ValueError('Invalid similarity function')
    
    similarity_matrix = pd.DataFrame(index=df['cluster'], columns=df['cluster'])
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            similarity_matrix.iloc[i, j] = scorer.score(df.iloc[i].caption,df.iloc[j].caption)['rouge1'].precision
    return similarity_matrix

# rouge = pd.DataFrame(index=grouped.cluster, columns=grouped.cluster)
# 
# for i in range(len(rouge)):
#     for j in range(len(rouge)):
#             rouge.iloc[i, j] = scorer.score(grouped.iloc[i].caption,grouped.iloc[j].caption)['rouge1'].precision
'''
rouge = compute_sim_matrix(grouped, lambda x,y: sentence_bleu([x.split()],y.split()))
df_clustered_09, grouped_df_09 = semantic_clustering(grouped,rouge, 0.9)
grouped_df_09.to_csv('../data/frankfurt/semantic_clusters_09.csv')
'''




'''
#%%
for i in range(70,95,5):
    h= float(i/100)
    bleah, bleah_grouped = semantic_clustering(df,rouge, h)
    bleah.to_csv(f'../data/src/clustered_df_{str(i)}.csv')
    bleah_grouped.to_csv(f'../data/src/grouped_df_{str(i)}.csv')
    
#%%
import re
with open('../pisa_clustered_df_with_caption.html', 'r') as file:
    html = ''.join(file.readlines())

table_html = re.sub(r'</?div[^>]*>', '', html)
table_html = re.sub(r'</?a[^>]*>', '', table_html)
# Replace <img> tags with the content of their alt attribute
table_html = re.sub(r'<img[^>]*src="([^"]*)"[^>]*>', r'\1', table_html)
df_html = pd.read_html(table_html)[0]
df_html = df_html.rename(columns={'ID':'id'})
def gen_link(x:str):
    return f'<img src="{x}"/>'
df_09, grouped_df_09 = semantic_clustering(df,rouge, 0.9)
df_html = df_html.merge(clusters, on='id', how='inner',suffixes=('_old',''))
df_html = df_html.merge(df_09, on='id', how='inner',suffixes=('','_semantic'))
df_html = df_html.drop(columns=['caption', 'cluster_old'])
df_html['link'] = df_html.link.apply(gen_link)
df_html.to_html('../pisa_clustered_df_with_captions_new.html', escape=False)

#%%

if os.path.exists('../rouge_cooccurence_precision.csv'):\
    rouge = pd.read_csv('../rouge_cooccurence_precision.csv', index_col=0)  
else:
    rouge = pd.DataFrame(index=captions['caption'], columns=captions['caption'])
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    for i in range(len(captions)):
        for j in range(len(captions)):
            rouge.iloc[i, j] = scorer.score(captions.iloc[i].caption,captions.iloc[j].caption)['rouge1'].precision

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#%%
# Assuming you have a pandas dataframe named 'df' with numeric values


numpi = rouge.to_numpy().astype(np.float32)
# %%
'''
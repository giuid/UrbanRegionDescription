#%%
import pandas as pd
import os
import re
def clean_output(decoded : list,captions:list):
    if len(decoded) != len(captions):
        raise ValueError('decoded and captions must have the same length')
    cleaned = []
    for i in range(len(decoded)):
        x = re.sub(r'.*'+captions[i], '', decoded[i])
        x= re.sub(r'\[/INST\]','',x)
        x= re.sub(r'\[INST\]','',x)
        x = re.sub(r'</s>','',x)
        x=re.sub(r'  ',' ',x)
        x=re.sub(r'  ',' ',x)
        x= x.split('###')[-1]
        x = re.sub(r'\n','',x)
        x = re.sub(r'User:',' ',x)
        x = re.sub(r'Output:',' ',x)
        x = re.sub(r'Answer:',' ',x)
        x = re.sub(r'.*:','',x)
        x=re.sub(r'  ',' ',x)
        x=re.sub(r'  ',' ',x)
        cleaned.append(x)

    return cleaned


final_results_path = '../data/frankfurt/summaries/'
files = [x for x in  os.listdir(final_results_path)  if '.csv' in x ]

#%%
#results = {file.replace('.csv',''): pd.read_csv(os.path.join(final_results_path, file)) for file in files}
concatenated_captions = pd.read_csv('../data/pisa/clusters/grouped_df_90.csv', index_col=0)
for file in files:
    df =pd.read_csv(os.path.join(final_results_path, file))
    df['cleaned_output'] = clean_output(df['decoded'].tolist(),df['caption'].tolist())
    if not 'cluster' in df.columns:
        df = df.merge(concatenated_captions, on='caption', how='outer')
        df.to_csv(os.path.join(final_results_path, file), index=False)
        df.to_excel(os.path.join(final_results_path, file.replace('.csv','.xlsx')), index=False)
#merged = df.merge(concatenated_captions, on='caption', how='outer')
# %%
frankfurt_noisy = pd.read_csv('../data/frankfurt/clustered_frankfurt_noisy.csv')
# %%
accorpated_clusters = {3: [7, 8],52: [56, 98], 17: [57],70: [71],6: [80],115: [116]}

for key in accorpated_clusters:
    frankfurt_noisy.loc[frankfurt_noisy.cluster.isin(accorpated_clusters[key]),'cluster'] = key
# %%
#df_grouped = pd.DataFrame(df.groupby('prompt').apply(lambda x: x[['cluster','cleaned_output']].reset_index(drop=True)).unstack()).reset_index()
frankfurt_new_clusters = frankfurt_noisy.copy()
frankfurt_new_clusters.link = frankfurt_new_clusters.link.apply(lambda x: '<img src="'+x+'" width="600" />')
frankfurt_new_clusters = frankfurt_new_clusters.sort_values(by='cluster', ascending=True)
frankfurt_new_clusters = frankfurt_new_clusters.merge(df[['cluster', 'cleaned_output','prompt']], on='cluster', how='outer')
frankfurt_new_clusters[frankfurt_new_clusters.cluster != -1].to_html('../data/frankfurt/frankfurt_new_clusters.html', escape=False)

# %%
df.groupby('prompt').count()
# %%
import pandas as pd
import re
from bs4 import BeautifulSoup

table_html = open('../data/pisa/pisa_clustered_df_with_caption.html').readlines()
table_html = ''.join(table_html)
# Delete all <div>, </div>, <a> and </a> tags
soup = BeautifulSoup(table_html, 'lxml')
images = images = soup.find_all('img')
#%%
final_clusters = pd.read_csv('../data/pisa/final_clusters_semantic.csv')

links = []
for ID in final_clusters.id:
    for link in images:
        if str(ID) in str(link):
            links.append(link)
#%%
final_clusters['html'] = [str(x) for x in links]
#%%
final_clusters.to_html('../data/pisa/final_clusters.html', index=False, escape=False)
# %%

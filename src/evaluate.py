#%%
import os

from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="4"

from rouge_score import rouge_scorer
from bert_score import score as bert_scorer
#from torchtext.data.metrics import bleu_score
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import logging
logging.set_verbosity_error()



def calculate_blue_score(candidate_translation, reference_translations):
    # Tokenize candidate translation and reference translations
    candidate_tokens = nltk.word_tokenize(candidate_translation.lower())
    reference_tokens = [nltk.word_tokenize(reference.lower()) for reference in reference_translations]

    # Calculate individual n-gram precisions for n=1 to 4
    individual_precisions = [nltk.translate.bleu_score.modified_precision(reference_tokens, candidate_tokens, i) for i in range(1, 5)]

    # Calculate the brevity penalty
    brevity_penalty = nltk.translate.bleu_score.brevity_penalty(reference_tokens, candidate_tokens)

    # Calculate the Blue Score
    blue_score = brevity_penalty * nltk.translate.bleu_score.geo_mean(individual_precisions)
    return blue_score

def calculate_scores_old(data,original_column,generated_column):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    df = data.copy()
    scores = [scorer.score(str(x),str(y)) for x,y in tqdm(zip(df[original_column].tolist(),df[generated_column].tolist()),desc='ROUGE',position=1)]
    #bleu_scores = [calculate_blue_score(y,x) for x,y in zip(df.summary.tolist(),df.decoded.tolist())]
    bert_scores = [bert_scorer([str(y)],[str(x)],lang ='en') for x,y in tqdm(zip(df[original_column].tolist(),df[generated_column].tolist()),desc='BERT',position=1)]
    df['rouge1_precision'+'_'+generated_column] = [x['rouge1'].precision for x in scores]
    df['rouge1_recall'+'_'+generated_column] = [x['rouge1'].recall for x in scores]
    df['rouge1_fmeasure'+'_'+generated_column] = [x['rouge1'].fmeasure for x in scores]
    df['rougeL_precision'+'_'+generated_column] = [x['rougeL'].precision for x in scores]
    df['rougeL_recall'+'_'+generated_column] = [x['rougeL'].recall for x in scores]
    df['rougeL_fmeasure'+'_'+generated_column] = [x['rougeL'].fmeasure for x in scores]
    df['bert_precision'+'_'+generated_column] = [float(x[0][0]) for x in bert_scores]
    #df['bleu_score'] = bleu_scores
    df['bert_recall'+'_'+generated_column] = [float(x[1][0]) for x in bert_scores]
    df['bert_fmeasure'+'_'+generated_column] = [float(x[2][0]) for x in bert_scores]
    
    originalsummary = [str(x).split() for x in  df[original_column].tolist()]
    generatedsummary =[str(x).split() for x in   df[generated_column].tolist()]

    berluscores = [sentence_bleu([x],y) for x,y in zip(originalsummary,generatedsummary)]
    df['bleu_score'+'_'+generated_column] = berluscores
    return df

def calculate_scores(data,original_column,generated_column, weights =None):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    df = data.copy()
    scores = []
    bert_scores = []
    bleu_scores = []
    if weights is None:
        weights = [(1./2., 1./2.), (1./3., 1./3., 1./3.),(1./4., 1./4., 1./4., 1./4.)]
    for x,y in tqdm(zip(df[original_column].tolist(),df[generated_column].tolist()),desc='Lines',position=1):
        scores.append(scorer.score(str(x),str(y)))
        bert_scores.append(bert_scorer([str(y)],[str(x)],lang ='en')) 
    #    bleu_scores.append(sentence_bleu([str(y).split()],str(x).split(),weights = weights))
    df['rouge1_precision'] = [x['rouge1'].precision for x in scores]
    df['rouge1_recall'] = [x['rouge1'].recall for x in scores]
    df['rouge1_fmeasure'] = [x['rouge1'].fmeasure for x in scores]
    df['rougeL_precision'] = [x['rougeL'].precision for x in scores]
    df['rougeL_recall'] = [x['rougeL'].recall for x in scores]
    df['rougeL_fmeasure'] = [x['rougeL'].fmeasure for x in scores]
    df['bert_precision'] = [float(x[0][0]) for x in bert_scores]
    
    #df['bleu_score_2-gram'] = [x[0] for x in bleu_scores]
    #df['bleu_score_3-gram'] = [x[1] for x in bleu_scores]
    #df['bleu_score_4-gram'] = [x[2] for x in bleu_scores]

    df['bert_recall'] = [float(x[1][0]) for x in bert_scores]
    df['bert_fmeasure'] = [float(x[2][0]) for x in bert_scores]
    return df

def plot_bar_mean(means :list, bar_labels:list, title = 'Mean Values For Each Dataset',ylabel = 'Mean value',save_path = None): 
    palette = sns.color_palette("colorblind", len(means)) 
    fig, ax = plt.subplots()
    x_pos = list(range(len(bar_labels)))
    for i in range(len(means)):
        plt.bar(x_pos[i], means[i], align='center', label=bar_labels[i], color =palette[i])
    plt.grid()
    # set height of the y-axis
    max_y = 1
    plt.ylim([0, max_y + max_y/10])
    # set axes labels and title
    plt.ylabel(ylabel)
    plt.xticks(x_pos, bar_labels, rotation=30)
    plt.title(title)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()



def generate_plots_for_dfs(final_scores:dict, save_path = None):
    evaluations = ['rouge1','rougeL','bert','bleu']
    metrics = ['precision', 'recall', 'fmeasure']
    for evaluation in evaluations:
        if evaluation == 'bleu':
            means = []
            bar_labels = []
            for key in final_scores:
                print(key)
                column_name = f'{evaluation}_score_{key}'
                means.append(final_scores[key][column_name].mean())
                if key == 'generated_manual_summaries_': name = 'finetuned_t5'
                else:
                    name = key.replace('generated_manual_summaries_','')
                bar_labels.append(name)
            new_save_path = os.path.join(save_path,f'{evaluation}.png') if save_path is not None else None
            plot_bar_mean(means,bar_labels,title = f'{evaluation}',ylabel = f'{evaluation}', save_path = new_save_path)
        else:
            for metric in metrics:
                means = []
                bar_labels = []
                for key in final_scores:
                    print(key)

                    column_name = f'{evaluation}_{metric}_{key}'
                    means.append(final_scores[key][column_name].mean())
                    if key == 'generated_manual_summaries_': name = 'finetuned_t5'
                    else:
                        name = key.replace('generated_manual_summaries_','')
                    bar_labels.append(name)
                    new_save_path = os.path.join(save_path,f'{evaluation}_{metric}.png') if save_path is not None else None
                plot_bar_mean(means,bar_labels,title = f'{evaluation}_{metric}',ylabel = f'{evaluation}_{metric}', save_path = new_save_path)


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
#%%
def evaluate_df(df, manual_df, name:str):#,type:str):
    scores = df.copy()
    #assert type in ['zero_shot','one_shot','few_shot'], 'type must be one of zero_shot, one_shot, few_shot'
    if 'cleaned' not in scores.columns:
        scores['cleaned'] = clean_output(scores.decoded.tolist(),scores.caption.tolist())
    data = df.merge(manual_df, on='cluster', how='inner')
    score = calculate_scores(data,'summary', 'cleaned')
    score.columns = [x+'_'+name if x not in ['cluster'] else x for x in score.columns]
    #scores = scores.merge(score, on='cluster', how='inner')
    return score

cluster_manual = pd.read_csv('../data/pisa/clusters/new_manual_captions.csv', sep =';')
sumaries_path = '../data/pisa/summaries/'
files = [x for x in  os.listdir(sumaries_path)  if '.csv' in x ]
scores = {}
for file in tqdm(files[:], desc='Evaluating files',position=0):
    scores[file] = None
    df = pd.read_csv(os.path.join(sumaries_path, file))
    df['cleaned'] = clean_output(df.decoded.tolist(),df.caption.tolist())
    df['cleaned'] = df['cleaned'].apply(lambda x: ".".join(x.split('.')[:4]))
    evaluation = evaluate_df(df, cluster_manual, file)
    evaluation.to_csv(f'../data/pisa/scores/four_sents_{file}', index = False)
    scores[file] = evaluation
    


    #%%

#%%
'''
cluster_manual = pd.read_csv('../data/pisa/clusters/clusters_manual.tsv', sep='\t')
do_scores = True
if do_scores:
    sumaries_path = '../data/pisa/summaries/'
    files = [x for x in  os.listdir(sumaries_path)  if '.csv' in x ]
    summaries = {}
    for file in files:
        df = pd.read_csv(os.path.join(sumaries_path, file))
        df['cleaned'] = clean_output(df.decoded.tolist(),df.caption.tolist())
        summaries[file] = df
    
    scores = {'zero_shot': None, 'one_shot': None, 'few_shot': None}
    for key in summaries:
        df= summaries[key]
        data = df.merge(cluster_manual, on='cluster', how='inner')
        score = calculate_scores(data,'summary', 'cleaned')
        score.columns = [x+'_'+key if x not in ['cluster'] else x for x in score.columns]
        name = 'zero_shot' if 'zero' in key else 'one_shot' if 'one' in key else 'few_shot'
        if scores[name] is None:
            scores[name] = score
        else:
            scores[name] = scores[name].merge(score, on='cluster', how='inner')
else:
    sumaries_path = '../data/pisa/scores'
    files = [x for x in  os.listdir(sumaries_path)  if '.csv' in x ]
    summaries = {}
    for file in files:
        df = pd.read_csv(os.path.join(sumaries_path, file))
        df['cleaned'] = clean_output(df.decoded.tolist(),df.caption.tolist())
        summaries[file] = df

    scores = {'zero_shot': None, 'one_shot': None, 'few_shot': None}
    for key in summaries:
        df = summaries[key]
        data = df.merge(cluster_manual, on='cluster', how='inner',)
        data.columns = [x+'_'+key if x not in ['cluster'] else x for x in data.columns]
        name = 'zero_shot' if 'zero' in key else 'one_shot' if 'one' in key else 'few_shot'
        if scores[name] is None:
            scores[name] = data
        else:
            scores[name] = scores[name].merge(data, on='cluster', how='inner')
for key in scores:
    scores[key].to_csv(f'../data/pisa/scores/{key}_scores.csv', index = False)
#%%
save_path = '../data/pisa/plots'
evaluations = ['rouge1','rougeL','bert']
metrics = ['precision', 'recall', 'fmeasure']


#%%
evaluations = ['rouge1_precision', 'rouge1_recall', 'rouge1_fmeasure', 'rougeL_precision', 'rougeL_recall', 'rougeL_fmeasure', 'bert_precision', 'bert_recall', 'bert_f1']
for evaluation in evaluations:
    for method in scores:
        means = []
        bar_labels = []
        for key in [x for x in  scores[method].columns if evaluation in x]:
            means.append(scores[method][key].mean())

            name= key.replace('few_shots','').replace('one_shot','').replace('zero_shot','').replace('.csv','').replace(evaluation+'_','').replace('scores','').replace('_',' ')
            bar_labels.append(name)
        new_save_path = os.path.join(save_path,f'{method}_{evaluation}.png') if save_path is not None else None
        print (method)

        plot_bar_mean(means,bar_labels,title = f'{evaluation}',ylabel = f'{evaluation}', save_path = new_save_path)
    
'''
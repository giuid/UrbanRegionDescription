#%%
import os
os.environ['HF_HOME'] = '../llms/data/cache/'
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,4,5"
print('Using ../llms/data/cache')
import pandas as pd
import torch
import transformers
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from tqdm import tqdm
import pandas as pd

hf_token = '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'
login(token=hf_token)
def generate_chat(prompt,caption :str, examples : list =None, mode = 'zero_shot', n_shots = 3):
    assert mode in ['zero_shot','one_shot','few_shots'], 'mode must be one of zero_shot, one_shot, few_shots'
    if mode == 'zero_shot':
        chat = [
            {"role": "user", "content": prompt + caption},
        ]
        return chat
    elif mode == 'one_shot':
        chat = [
            {"role": "user", "content": prompt + examples.caption.iloc[0]},
            {"role": "assistant", "content": examples.summary.iloc[0]},
            {"role": "user", "content": prompt + caption}
        ] 
        return chat
    elif mode == 'few_shots':
        chat = []
        if n_shots > len(examples): n_shots = len(examples)
        for i in range(n_shots):
            chat.append({"role": "user", "content": prompt + examples.caption.iloc[i]})
            chat.append({"role": "assistant", "content": examples.summary.iloc[i]})
        chat.append({"role": "user", "content":  prompt + caption})
        return chat
def select_less_used_gpu():
    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        gpu_memory.append(torch.cuda.memory_allocated(i))
    device = torch.device(f"cuda:{gpu_memory.index(min(gpu_memory))}" if torch.cuda.is_available() else "cpu")
    return device

def generate_summaries_old(model,prompts,captions,tokenizer,manual_captions =None, mode='zero_shot',n_shots=3):
    data = {'prompt':[],'caption':[],'chat':[],'encodeds' :[] , 'decoded':[], 'cluster':[]}
    for prompt in tqdm(prompts,desc='Prompts'):
        #print('Prompt : ',prompt)
        for caption in tqdm(captions,desc='Captions',leave=False,position=1):
           # print('Caption : ',caption)
            chat = generate_chat(prompt,caption,manual_captions,mode=mode,n_shots=n_shots)
            encodeds = tokenizer.apply_chat_template(chat, return_tensors="pt")
            data['prompt'].append(prompt)
            data['caption'].append(caption)
            data['chat'].append(chat)
            data['encodeds'].append(encodeds)
            device = select_less_used_gpu()
            model_inputs = encodeds.to(device)
            try:
                with torch.no_grad():
                    generated_ids = model.generate(model_inputs, max_new_tokens=256, do_sample=True)
                    decoded = tokenizer.batch_decode(generated_ids)
                    data['decoded'].append(decoded[0])
                    del model_inputs
                    del generated_ids
                    del decoded
                    torch.cuda.empty_cache()
            except Exception as e:
                data['decoded'].append(f'error {str(e)}')
                del model_inputs
                torch.cuda.empty_cache()
                continue
    return data

def generate_summaries(model,prompts,captions,clusters,tokenizer,manual_captions =None, mode='zero_shot',n_shots=3):
    data = {'prompt':[],'caption':[],'chat':[],'encodeds' :[] , 'decoded':[], 'cluster':[]}
    for prompt in tqdm(prompts,desc='Prompts'):
        #print('Prompt : ',prompt)
        for caption, cluster in zip(captions,clusters):
           # print('Caption : ',caption)
            chat = generate_chat(prompt,caption,manual_captions,mode=mode,n_shots=n_shots)
            encodeds = tokenizer.apply_chat_template(chat, return_tensors="pt")
            data['prompt'].append(prompt)
            data['caption'].append(caption)
            data['chat'].append(chat)
            data['encodeds'].append(encodeds)
            data['cluster'].append(cluster)
            device = select_less_used_gpu()
            model_inputs = encodeds.to(device)
            try:
                with torch.no_grad():
                    generated_ids = model.generate(model_inputs, max_new_tokens=256, do_sample=True)
                    decoded = tokenizer.batch_decode(generated_ids)
                    data['decoded'].append(decoded[0])
                    del model_inputs
                    del generated_ids
                    del decoded
                    torch.cuda.empty_cache()
            except Exception as e:
                data['decoded'].append(f'error {str(e)}')
                del model_inputs
                torch.cuda.empty_cache()
                continue
    return data


models = ['upstage/SOLAR-10.7B-Instruct-v1.0','meta-llama/Llama-2-13b-chat-hf','meta-llama/Llama-2-7b-chat-hf']
clusters_path ='../data/frankfurt/semantic_clusters_09.csv'
manual_captions_path = '../data/manual_captions.csv'
prompts_path = '../data/prompts.csv'
output_path = '../data/frankfurt/summaries/'
#clusters_path = '../data/pisa/clusters/grouped_df_90.csv'

concatenated_captions = pd.read_csv(clusters_path)
manual_captions = pd.read_csv(manual_captions_path)
prompts = pd.read_csv(prompts_path)
#concatenated_captions = pd.read_csv('./new_concatenated_captions.csv')


#%%
usebnb = False
for model_id in models:
    if model_id == 'upstage/SOLAR-10.7B-Instruct-v1.0':
        name = 'SOLAR-10.7B'
    elif model_id == 'meta-llama/Llama-2-13b-chat-hf':
        name = 'Llama-13B'
    elif model_id == 'meta-llama/Llama-2-7b-chat-hf':
        name = 'Llama-7B'
    print(f'Running {name}')
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map='auto')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    n_shots = 3
    #%%
    prompts_list = prompts.prompt.tolist()
    captions_list = concatenated_captions.caption.tolist()
    clusters_list = concatenated_captions.cluster.tolist()
    print('Generating summaries for one_shot')
    print('saving to ',f'{output_path}one_shot_{name}.csv')
    data_one_shot = generate_summaries(model, prompts_list,captions_list,clusters_list,tokenizer,manual_captions,mode='one_shot')
    df_one_shot = pd.DataFrame(data_one_shot)
    df_one_shot.to_csv(f'{output_path}one_shot_{name}.csv',index=False)
    df_one_shot.to_excel(f'{output_path}one_shot_{name}.xlsx',index=False)
    print('Generating summaries for zero_shot')
    print('saving to ',f'{output_path}zero_shot_{name}.csv')
    data_zero_shot = generate_summaries(model, prompts_list,captions_list,clusters_list,tokenizer,manual_captions,mode='zero_shot')
    df_zero_shot = pd.DataFrame(data_zero_shot)
    df_zero_shot.to_csv(f'{output_path}zero_shot_{name}.csv',index=False)
    df_zero_shot.to_excel(f'{output_path}zero_shot_{name}.xlsx',index=False)
    print('Generating summaries for few_shots')
    print('saving to ',f'{output_path}few_shots_{name}.csv')
    data_few_shots = generate_summaries(model, prompts_list,captions_list,clusters_list,tokenizer,manual_captions,mode='few_shots',n_shots=n_shots)
    df_few_shots = pd.DataFrame(data_few_shots)
    df_few_shots.to_csv(f'{output_path}few_shots_{name}.csv',index=False)
    df_few_shots.to_excel(f'{output_path}few_shots_{name}.xlsx',index=False)
    del model
    torch.cuda.empty_cache()
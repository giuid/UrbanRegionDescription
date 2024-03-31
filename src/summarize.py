#%%
import os
### UNCOMMENT IF YOU WANT TO POINT TO A CUSTOM HUGGINGFACE CACHE FOLDER ###
#os.environ['HF_HOME'] = '../llms/data/cache/'

### UNCOMMENT IF YOU WANT TO SELECT THE GPUs TO USE WHILE SUMMARIZING ###
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,4,5"

import pandas as pd
import torch
import transformers
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from tqdm import tqdm
import pandas as pd

hf_token = 'INSERT_HUGGINGFACE_TOKEN'
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


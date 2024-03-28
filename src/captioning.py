#%%
import os

import torch
import requests
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,5,4"

from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import pandas as pd
from tqdm import tqdm
def download_photos(df, output_folder = './photo/pisa_clustered/',id_column = 'ID', link_column = 'link'):
    output_df = pd.DataFrame()
    if output_folder[-1] != '/':
        output_folder += '/'
    ids = []
    images = []
    for i in tqdm(range(len(df))):
        img_id = df.iloc[i][id_column]
        img_url = df.iloc[i][link_column]
        if not (os.path.isfile(output_folder + str(img_id) + '.jpg')):
            try:
                raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
                raw_image.save(output_folder + str(img_id) + '.jpg')
            except:
                print("img not found")
                raw_image = None


import spacy
#NER = spacy.load("en_core_web_sm")
NER = spacy.load("en_core_web_trf")

def sub_places_with_address(caption_text_list, address):
    cleaned_captions = []
    for caption in caption_text_list:
        cleaned_caption = caption.replace('\\n','')
        spacyed = NER(cleaned_caption)
        for word in spacyed.ents:
            if word.label_ == 'GPE':
                #print(word.text,word.label_)
                if address not in cleaned_caption:
                    cleaned_caption = cleaned_caption.replace(word.text,address)
                else:
                    cleaned_caption = cleaned_caption.replace(word.text,'')##PLACE##'')
        cleaned_captions.append(cleaned_caption)
    return cleaned_captions
def sub_places_with_address_new(caption_text_list, addresses):
    cleaned_captions = []
    for caption, address in tqdm(zip(caption_text_list, addresses)):
        cleaned_caption = caption.replace('\\n','')
        spacyed = NER(cleaned_caption)
        for word in spacyed.ents:
            if word.label_ == 'GPE':
                #print(word.text,word.label_)
                if address not in cleaned_caption:
                    cleaned_caption = cleaned_caption.replace(word.text,address)
                else:
                    cleaned_caption = cleaned_caption.replace(word.text,'')##PLACE##'')
        cleaned_captions.append(cleaned_caption)
    return cleaned_captions


from geopy.geocoders import Nominatim
def extract_address(lat, lon):#, elements =[ 'road', 'town', 'county', 'country']):
    geolocator = Nominatim(user_agent="geolocator")
    location = geolocator.reverse(f'{(lat)}, {str(lon)}',language='en')
    #location.address
    if "city" in location.raw['address'].keys():
        real_address = location.raw['address']['city'] 
        #real_address = location.raw['address']['road']+ ', ' + location.raw['address']['city'] + ', ' + location.raw['address']['country']
    elif "town" in location.raw['address'].keys():
        real_address = location.raw['address']['town'] 
    elif ' village' in location.raw['address'].keys():
        real_address= location.raw['address']['village']
    else: real_address = ''
        #real_address = location.raw['address']['town'] + ',' + location.raw['address']['county'] +',' + location.raw['address']['country']
    return real_address

def render_imgs(path: str = None) -> str:
    """Format the image URLs in <img> tags"""
    return f"""<img src="{path}" width="600" >"""

def execute_captioning_old(data_file, output_file = 'pisa_clustered_df_with_caption.html' , overwrite=False ,photos_path = './photo/pisa_clustered/'):
    if not(os.path.isfile(output_file)) or overwrite:
        ### Load Dataset
        pisa_clustered_df = pd.read_csv(data_file, index_col=0).reset_index(drop=True).drop_duplicates(subset=['ID'])
        ### Remove Outliers
        pisa_clustered_df_without_outliers = pisa_clustered_df[pisa_clustered_df['cluster'] != -1].reset_index(drop=True)
        ### Download Images
        download_photos(pisa_clustered_df_without_outliers.iloc[:])
        ###Load Captioning Model
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="cuda")
        ### Load Images and prepare for captioning
        text = ""
        ids = []    
        images = []
        for id in tqdm(pisa_clustered_df_without_outliers.iloc[:]['ID']):
            if os.path.isfile(photos_path + str(id) + '.jpg'):
                raw_image = Image.open(photos_path + str(id) + '.jpg').convert('RGB')
                ids.append(id)
                images.append(processor(raw_image, text, return_tensors="pt").to("cuda"))
        ### Captioning
        captions = []
        for img_id,inputs in tqdm(zip(ids[:10],images[:10])):
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        ### Merge Captions with Dataset and save to file
        new_df = pd.DataFrame({'ID':ids, 'caption':captions})
        pisa_clustered_df_with_caption = pd.merge(pisa_clustered_df_without_outliers, new_df, on='ID', how='inner')
        pisa_clustered_df_with_caption['link'] = pisa_clustered_df_with_caption.link.apply(render_imgs)
        pisa_clustered_df_with_caption.to_html(output_file, escape=False)
        return pisa_clustered_df_with_caption
    
def execute_captioning_old(data_file, output_file='pisa_clustered_df_with_caption.html', overwrite=False, photos_path='./photo/pisa_clustered/'):
    if not (os.path.isfile(output_file)) or overwrite:
        # Load Dataset
        clustered_df = pd.read_csv(data_file, index_col=0).reset_index(drop=True).drop_duplicates(subset=['ID'])
        # Remove Outliers
        clustered_df_without_outliers = clustered_df[clustered_df['cluster'] != -1].reset_index(drop=True)
        # Download Images
        download_photos(clustered_df_without_outliers.iloc[:], output_folder=photos_path)
        # Load Captioning Model
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="cuda")
        # Load Images and prepare for captioning
        text = ""
        ids = []
        images = []
        for id in tqdm(clustered_df_without_outliers.iloc[:]['ID']):
            if os.path.isfile(photos_path + str(id) + '.jpg'):
                raw_image = Image.open(photos_path + str(id) + '.jpg').convert('RGB')
                ids.append(id)
                images.append(processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16))
        # Captioning
        captions = []
        for img_id, inputs in tqdm(zip(ids[:10], images[:10])):
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        # Merge Captions with Dataset and save to file
        new_df = pd.DataFrame({'ID': ids, 'caption': captions})
        clustered_df_with_caption = pd.merge(clustered_df_without_outliers, new_df, on='ID', how='inner')
        clustered_df_with_caption['link'] = clustered_df_with_caption.link.apply(render_imgs)
        clustered_df_with_caption.to_html(output_file, escape=False)
        return clustered_df_with_caption
    else:
        if os.path.isfile(output_file):
            print(f"File {output_file} already exists")
def execute_captioning(df,output_file,photos_path):
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="cuda")
    text = ""
    ids = []
    images = []
    for id in tqdm(df.iloc[:]['ID']):
        if os.path.isfile(os.path.join(photos_path, str(id) + '.jpg')):
            raw_image = Image.open(os.path.join(photos_path, str(id) + '.jpg')).convert('RGB')
            ids.append(id)
            images.append(processor(raw_image, text, return_tensors="pt").to("cuda"))
    captions = []
    for img_id, inputs in tqdm(zip(ids[:], images[:])):
        out = model.generate(**inputs.to('cuda'))
        caption = processor.decode(out[0], skip_special_tokens=True)
        print(caption)
        captions.append(caption)
    new_df = pd.DataFrame({'ID': ids, 'caption': captions})
    df_with_captions = pd.merge(df, new_df, on='ID', how='inner')
    df_with_captions['link'] = df_with_captions.link.apply(render_imgs)
    df_with_captions.to_html(output_file, escape=False)
    return df_with_captions


def clean_captions_old(df):
    clusters = []
    captions = []
    for cluster in df.cluster.unique():
        sub_captions = df[df.cluster == cluster].caption.tolist()
        lat = df[df.cluster == cluster].iloc[0].lat
        lon = df[df.cluster == cluster].iloc[0].lon
        address = extract_address(lat, lon)
        cleaned_captions = sub_places_with_address(sub_captions, address)
        clusters.append(cluster)
        captions.append(cleaned_captions)
    captions_df = pd.DataFrame({'cluster':clusters, 'caption':captions})
    return captions_df

def clean_captions(df):
    clusters = []
    captions = []
    IDs = []
    for i in tqdm(range(len(df))):
        lat = df.iloc[i].lat
        lon = df.iloc[i].lon
        caption = df.iloc[i].caption
        id_ = df.iloc[i].ID
        address = extract_address(lat, lon)
        cleaned_captions = sub_places_with_address([caption], address)
        captions.append(cleaned_captions)
        IDs.append(id_)
    captions_df = pd.DataFrame({'id':IDs, 'caption':captions})
    return captions_df

def clean_captions_new(df):
    captions= df.caption.tolist()
    IDs = df.ID.tolist()
    lat = df.lat.tolist()
    lon = df.lon.tolist()
    adresses = [extract_address(lat[i], lon[i]) for i in tqdm(range(len(lat)))]
    cleaned_captions = sub_places_with_address_new(captions, adresses)
    captions_df = pd.DataFrame({'id':IDs, 'caption':cleaned_captions})
    return captions_df
#%%
import pandas as pd

do_captioning = False
if do_captioning:
    frankfurt_path = '../data/frankfurt/frankfurt.csv'
    frankfurt = pd.read_csv('../data/frankfurt/frankfurt.csv', index_col=0)

    df_with_captions = execute_captioning(frankfurt, '../data/frankfurt/frankfurt_with_captions.html', photos_path='../data/frankfurt/photos')
df_with_captions = pd.read_html('../data/frankfurt/frankfurt_with_captions.html', index_col=0)[0]
cleaned_captions = clean_captions_new(df_with_captions)
cleaned_captions = cleaned_captions.rename(columns={'id':'ID'})
noisy_clusters = pd.read_csv('../data/frankfurt/clustered_frankfurt_noisy.csv')
cleaned_captions.merge(noisy_clusters[['ID','cluster']], on='ID')

#download_photos(frankfurt, '../data/frankfurt/photos/')
#captions_df['caption'] = captions_df.caption.apply(lambda x: " . ".join(x))
#captions_df.to_csv('concatenated_captions.csv', index =False)


# %% Summarization Phase

# %%

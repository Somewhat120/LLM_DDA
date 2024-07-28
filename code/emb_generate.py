import pandas as pd
import openai
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import argparse


def get_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


def get_bert_token_length(description):
    # Tokenize the description and return the number of tokens
    return len(tokenizer.encode(description, add_special_tokens=True))


def get_biobert_embeddings(text, tokenizer, model):
    # Tokenize text into chunks of 512 tokens
    chunk_size = 512 - 2  # Accounting for special tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunked_tokens = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    all_embeddings = []
    for chunk in chunked_tokens:
        # Add special tokens
        chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        # Convert to tensor
        chunk_tensor = torch.tensor([chunk])
        with torch.no_grad():
            # Generate embeddings
            outputs = model(chunk_tensor)
        # Extract embeddings from the last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings)

    # Aggregate embeddings from all chunks (e.g., by taking the mean)
    aggregated_embeddings = torch.mean(torch.stack(all_embeddings), dim=0)
    return aggregated_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dataset', '--dataset', default=None, type=str)
    parser.add_argument('-method', '--method', default='GPT', type=str, choices=['GPT', 'BioBERT'])
    args = parser.parse_args()
    
    df_drug = pd.read_csv(f'data/{args.dataset}/drug_desc.csv')
    df_dis = pd.read_csv(f'data/{args.dataset}/disease_desc.csv')
    if args.method == 'GPT':
        openai.api_key = open("key.txt","r").read().strip('\n')

        # Apply the function to get embeddings and create a dictionary mapping 'Drug' to embeddings
        drug_to_embeddings = {drug: get_embeddings(description) 
                              for drug, description in zip(df_drug['Drug'], df_drug['Description'])}
        # Save the dictionary as a .pkl file
        pd.to_pickle(drug_to_embeddings, f'feat/{args.dataset}/LLM_drug_emb.pkl')

        disease_to_embedding = {dis: get_embeddings(description) 
                                for dis, description in zip(df_dis['Disease'], df_dis['Description'])}
        pd.to_pickle(drug_to_embeddings, f'feat/{args.dataset}/LLM_disease_emb.pkl')

    elif args.method == 'BioBERT':
        # Load the tokenizer
        model_name = 'dmis-lab/biobert-v1.1'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        df_drug['token_length'] = df_drug['Description'].apply(get_bert_token_length)
        df_drug['embeddings'] = df_drug['Description'].apply(lambda x: get_biobert_embeddings(x, tokenizer, model))
        pd.to_pickle(df_drug, f'feat/{args.dataset}/BERT_drug_emb.pkl')

        df_dis['token_length'] = df_dis['Description'].apply(get_bert_token_length)
        df_dis['embeddings'] = df_dis['Description'].apply(lambda x: get_biobert_embeddings(x, tokenizer, model))
        pd.to_pickle(df_dis, f'feat/{args.dataset}/BERT_disease_emb.pkl')       

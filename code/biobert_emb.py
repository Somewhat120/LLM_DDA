import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# Load the tokenizer
model_name = 'dmis-lab/biobert-v1.1'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


# Function to calculate the number of tokens in a description using BERT tokenizer
def get_bert_token_length(description):
    # Tokenize the description and return the number of tokens
    return len(tokenizer.encode(description, add_special_tokens=True))


# Load the CSV file into a DataFrame
drug_df = pd.read_csv('drug_desc.csv')
# diseases_df = pd.read_csv('disease_desc.csv')

# Apply the function to each description to calculate token length
# diseases_df['token_length'] = diseases_df['Description'].apply(get_bert_token_length)
drug_df['token_length'] = drug_df['Description'].apply(get_bert_token_length)

# Print the DataFrame with the ID and token length columns
# print(diseases_df[diseases_df['token_length'].values > 512])
# print(drug_df[drug_df['token_length'].values > 512])

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


# Generate embeddings for each description
# diseases_df['embeddings'] = diseases_df['Description'].apply(lambda x: get_biobert_embeddings(x, tokenizer, model))
drug_df['embeddings'] = drug_df['Description'].apply(lambda x: get_biobert_embeddings(x, tokenizer, model))
# Save the embeddings to a pickle file
drug_df.to_pickle('drug_embeddings.pkl')
# diseases_df.to_pickle('disease_embeddings.pkl')
# Load the pickle file
loaded_df = pd.read_pickle('drug_embeddings.pkl')
# loaded_df = pd.read_pickle('disease_embeddings.pkl')

# Display the contents
print(loaded_df.head())

# Check the dimensions of the first embedding (example)
if len(loaded_df) > 0 and 'embeddings' in loaded_df.columns:
    first_embedding = loaded_df['embeddings'].iloc[0]
    print("Dimensions of the first embedding:", first_embedding.shape)
else:
    print("No embeddings found or 'embeddings' column is missing.")

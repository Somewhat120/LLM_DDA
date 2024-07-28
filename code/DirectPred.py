import os
import openai
import pandas as pd
import argparse
from tqdm import tqdm

def create_instructional_message(disease_name, omim_id, drug_name, drug_db_id):
    disease_name = str(disease_name)[1:-1].replace("'", "")
    omim_id = str(omim_id)[1:-1].replace("'", "")
    introduction = (
        "The Online Mendelian Inheritance in Man (OMIM) database is a comprehensive, authoritative compendium of human genes and genetic phenotypes. "
        "The DrugBank database is a bioinformatics and cheminformatics resource that combines detailed drug data with comprehensive drug target information. "
        "Our objective is to identify associations between drugs and diseases, where the network's nodes represent both diseases and drugs (including certain chemicals not used as human drugs), and edges indicate the associations between them. It covers associations like arsenic's links to diseases such as prostatic neoplasms and myocardial ischemia."
    )
    query = (
        f"Given this, does the drug {drug_name} with DrugBank ID {drug_db_id}"
        f"have any associations with the diseases in the list '{disease_name}' and OMIM ID {omim_id}?"
    )
    return introduction + " " + query


def query_disease_associations(diseases, drugs, model_id):
    # associated_disease_ids = {drug['DrugBank ID']: [] for drug in drugs}
    if os.path.exists(f'data/{DATASET}/DirectPred.csv'):
        df = pd.read_csv(f'data/{DATASET}/DirectPred.csv')
        exits_drugs = df['DrugBank ID'].unique()
    omim_list = diseases['Disease'].tolist()
    result_list = []
    exits_drugs = []  # Initialize exits_drugs before using it in the loop
    for i in tqdm(range(len(drugs))):
        current_drug = drugs.loc[i, 'Drug']
        if current_drug in exits_drugs:
            continue
        tqdm.write(f"{current_drug}")
        message = create_instructional_message(diseases['Name'].tolist(),
                                               omim_list,
                                               drugs.loc[i, 'Name'],
                                               current_drug
                                               )

        # Make the actual API call (pseudo-code for demonstration purposes)
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=[{"role": "system",
                       "content": "The answer should be a list of OMIM IDs selected from the previous list. "
                                  "Separate the IDs with commas."},
                      {"role": "user", "content": message}],
            temperature=0
        )
        # Properly handle the response
        answer = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip().upper()
        answer = answer.split(",") if answer else []
        answer = [i.strip() for i in answer]

        result_list.extend([[current_drug, omim_id] for omim_id in answer if omim_id in omim_list])
        pd.DataFrame(result_list, columns=['DrugBank ID', 'OMIM ID']).to_csv('F-dataset\Fdataset_result.csv')
        # if answer == "Y":
        #     associated_disease_ids[drug['DrugBank ID']].append(disease['OMIM ID'])
    return pd.DataFrame(result_list, columns=['DrugBank ID', 'OMIM ID'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dataset', '--dataset', default=None, type=str)
    args = parser.parse_args()
    # Load the OpenAI API key
    with open("key.txt", "r") as file:
        openai.api_key = file.read().strip('\n')

    # Define the model to use
    model_id = "gpt-4-0125-preview"

    # Load data
    drugs_df = pd.read_csv(f'data/{args.dataset}/drug.csv')
    diseases_df = pd.read_csv(f'data/{args.dataset}/disease.csv')
    answer = query_disease_associations(diseases_df, drugs_df, model_id)

import os
import openai
import pandas as pd
import argparse


def create_user_message_drug(drug_name, drug_id):
    return (
        f"Generate a single, comprehensive paragraph for the drug '{drug_name}' "
        f"associated with its DrugBank ID '{drug_id}'. "
        "The response should include 10 key pieces of information as follows: "
        "1) detailed description of its chemical structure; "
        "2) its chemical category; "
        "3) its chemical scaffold; "
        "4) any known similar drugs, with examples; "
        "5) detailed description of its pharmacokinetics, including absorption, distribution, metabolism, and excretion; "
        "6) details of its toxicity, with examples; "
        "7) list of any known target proteins; "
        "8) indication of this drug, with specific examples of diseases or symptoms; "
        "9) side effects of this drug, with examples; "
        "10) clinical usage of this drug, with examples. "
        "The information does not need to be current or from a live database. "
        "Ensure the final summary is precise, evidence-based, suitable for a professional pharmacological or chemical audience, and condenses all the points above into a coherent narrative."
    )

def create_user_message_dis_OMIM(disease_name, omim_id):
    return (
        f"Generate a single, cohesive, narrative paragraph for the disease '{disease_name}' associated with OMIM ID '{omim_id}'."
        f"The response should include 10 key information as follows:\n"
        "1) associated genes, proteins, or mutations, with at least 3 examples.\n"
        "2) associated signal pathway, including key molecular or cellular components.\n"
        "3) associated drugs commonly used for treatment, with at least 3 examples and their mechanisms of action.\n"
        "4) any linked comorbidities and complications.\n"
        "5) nature of the disease.\n"
        "6) typical clinical symptoms and signs.\n"
        "7) types of the disease.\n"
        "8) inheritance patterns and any known genetic component, with examples.\n"
        "9) diagnostic criteria and testing methods.\n\n"
        "The information does not need to be current or from a live database. "
        "Ensure the final summary is precise, evidence-based, suitable for a professional medical audience, and condense all the points above into a coherent narrative."
    )


# Read the input CSV and process each disease
with open(input_csv_path, newline='', encoding='utf-8') as infile, \
        open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
    # Set up CSV reader and writer
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=['Name', 'Disease', 'Description'])
    writer.writeheader()

    # Process each row in the input CSV
    for row in reader:
        disease_name = row['Name']
        omim_id = row['Disease']

        conversation = [
            {
                "role": "system",
                "content": "You are an expert in medical research, genetics, and pharmacology."
            },
            {
                "role": "user",
                "content": create_user_message(disease_name, omim_id)
            }
        ]

        # Send request to OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model=model_id,
                messages=conversation,
                max_tokens=450
            )
            # Extract the response content
            description = response['choices'][0]['message']['content']
            # print(description)

        except Exception as e:
            print(f"An error occurred while processing {disease_name}: {e}")
            description = "Error retrieving information"

        # Write the result to the output CSV
        writer.writerow({'Name': disease_name, 'Disease': omim_id, 'Description': description})

print(f"Processing complete. Output saved to {output_csv_path}")


# Disease description generation for R dataset
import csv
import openai

openai.api_key = open("key.txt", "r").read().strip('\n')

# Define the model to use
model_id = "gpt-4-0125-preview"

# File paths
input_csv_path = 'R-dataset\disease.csv'
output_csv_path = 'R-dataset\desease_desc.csv'


# Function to create the user message for the conversation
def create_user_message_dis_MeSH(disease_name, mesh_id):
    return (
        f"Generate a single, cohesive, narrative paragraph for the disease '{disease_name}' in MeSH (Medical Subject Headings) vocabulary have association with MeSH ID '{mesh_id}'."
        f"The response should include 10 key information as follows:\n"
        "1) associated genes, proteins, or mutations, with at least 3 examples.\n"
        "2) associated signal pathway, including key molecular or cellular components.\n"
        "3) associated drugs commonly used for treatment, with at least 3 examples and their mechanisms of action.\n"
        "4) any linked comorbidities and complications.\n"
        "5) nature of the disease.\n"
        "6) typical clinical symptoms and signs.\n"
        "7) types of the disease.\n"
        "8) inheritance patterns and any known genetic component, with examples.\n"
        "9) diagnostic criteria and testing methods.\n\n"
        "The information does not need to be current or from a live database. "
        "Ensure the final summary is precise, evidence-based, suitable for a professional medical audience, and condense all the points above into a coherent narrative."
    )


# Read the input CSV and process each disease


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
    input_csv_path_drug = f'{args.dataset}\drug.csv'
    output_csv_path_drug = f'{args.dataset}\drug_desc.csv'

    with open(input_csv_path, newline='', encoding='utf-8') as infile, \
            open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        # Set up CSV reader and writer
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=['Name', 'Drug', 'Description'])
        writer.writeheader()

        # Process each row in the input CSV
        for row in reader:
            drug_name = row['Name']
            drug_id = row['Drug']
            # SMILES_note = row['SMILES']

            conversation = [
                {
                    "role": "system",
                    "content": "You are an expert in medical research, genetics, chemistry, and pharmacology."
                },
                {
                    "role": "user",
                    "content": create_user_message_drug(drug_name, drug_id)
                }
            ]

            # Send request to OpenAI API
            try:
                response = openai.ChatCompletion.create(
                    model=model_id,
                    messages=conversation,
                    max_tokens=450
                )
                # Extract the response content
                description = response['choices'][0]['message']['content']
                # print(description)

            except Exception as e:
                print(f"An error occurred while processing {drug_name}: {e}")
                description = "Error retrieving information"

            # Write the result to the output CSV
            writer.writerow({'Name': drug_name, 'Drug': drug_id, 'Description': description})

    print(f"Processing complete. Output saved to {output_csv_path}")
    
    input_csv_path_dis = f'{args.dataset}\disease.csv'
    output_csv_path_drug = f'{args.dataset}\disease_desc.csv'
    
    with open(input_csv_path, newline='', encoding='utf-8') as infile, \
            open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        # Set up CSV reader and writer
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=['Name', 'Disease', 'Description'])
        writer.writeheader()

        # Process each row in the input CSV
        for row in reader:
            disease_name = row['Name']
            dis_id = row['Disease']

            conversation = [
                {
                    "role": "system",
                    "content": "You are an expert in medical research, genetics, and pharmacology."
                },
                {
                    "role": "user",
                    "content": 
                    create_user_message_dis_OMIM(disease_name, dis_id) if args.dataset != 'Rdataset' else create_user_message_dis_MeSH(disease_name, dis_id)
                }
            ]

            # Send request to OpenAI API
            try:
                response = openai.ChatCompletion.create(
                    model=model_id,
                    messages=conversation,
                    max_tokens=450
                )
                # Extract the response content
                description = response['choices'][0]['message']['content']
                # print(description)

            except Exception as e:
                print(f"An error occurred while processing {disease_name}: {e}")
                description = "Error retrieving information"

            # Write the result to the output CSV
            writer.writerow({'Name': disease_name, 'Disease': omim_id, 'Description': description})

        print(f"Processing complete. Output saved to {output_csv_path}")
        
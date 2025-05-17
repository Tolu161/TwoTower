#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:12:13 2025

@author: toluojo
"""


# Import necessary libraries
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import requests
import os
import collections
import datasets
import re
import json



# Load datasets from JSON
train_dataset = pd.read_json("train_dataset.json", orient="records", lines=True)
validation_dataset = pd.read_json("validation_dataset.json", orient="records", lines=True)
test_dataset = pd.read_json("test_dataset.json", orient="records", lines=True)


class TripleGenerator:
    def __init__(self):
        self.all_passages = []  # Store all passages for negative sampling
        self.triples = []  # Store generated triples

    def extract_query(self, row):
        return row['query']

    def extract_passages(self, row):
        """Extract passage textsgenefrom the given row, handling dict format."""
        if isinstance(row['passages'], dict):
            passages = row['passages']['passage_text']
            print(f"Extracted {len(passages)} passages: {passages}")  # 🔍 Debug print
            return passages
        else:
            print(f"Unexpected format for 'passages': {type(row['passages'])}")
            return []

    def find_relevant_documents(self, row):
        """Find **all** relevant documents from the passages dict."""
        if isinstance(row['passages'], dict):
            is_selected = row['passages']['is_selected']
            passage_texts = row['passages']['passage_text']
            relevant_docs = [passage_texts[i] for i, selected in enumerate(is_selected) if selected == 1]
            #print(f"Found {len(relevant_docs)} relevant documents: {relevant_docs}")  # 🔍 Debug print
            return relevant_docs
        else:
            #print(f"Unexpected format for 'passages': {type(row['passages'])}")
            return []

    def sample_irrelevant_document(self, relevant_docs):
        """Sample an irrelevant document that is **not** in the relevant docs."""
        if not self.all_passages:
            return None  # No passages available

        #print(f"Sampling irrelevant document (must not be in {relevant_docs})...")  # 🔍 Debug print

        irrelevant_doc = random.choice(self.all_passages)
        while irrelevant_doc in relevant_docs:
            irrelevant_doc = random.choice(self.all_passages)

        #print(f"Selected irrelevant document: {irrelevant_doc}")  # 🔍 Debug print
        return irrelevant_doc

    def generate_triples(self, dataset, dataset_name="Dataset"):
        """Generate **multiple** triples per query."""
        self.triples = []  # Reset triples
        
        print(f"Generating triples for {dataset_name} ({len(dataset)} samples)...")  # 🔍 Debug print

        #print(f"Generating triples for dataset with {len(dataset)} rows...")  # 🔍 Debug print

        for _, row in dataset.iterrows():  # .iterrows() method in pandas allows you to iterate over a DataFrame row by row
            query = self.extract_query(row)
            print(f"Processing query: {query}")  # 🔍 Debug print

            passages = self.extract_passages(row)

            # Add all passages for negative sampling
            self.all_passages.extend(passages)

            # Get **all** relevant documents
            relevant_docs = self.find_relevant_documents(row)

            if not relevant_docs:
                continue  # Skip if no relevant passages

            # Generate a triple for each relevant document
            for relevant_doc in relevant_docs:
                irrelevant_doc = self.sample_irrelevant_document(relevant_docs)

                if irrelevant_doc is None:
                    continue  # Skip if no irrelevant passage

                # Create the triple
                triple = (query, relevant_doc, irrelevant_doc)
                self.triples.append(triple)
                print(f"Generated triple: ({query}, {relevant_doc}, {irrelevant_doc})")  # 🔍 Debug print

        #print(f"Finished generating triples. Total: {len(self.triples)}")  # 🔍 Debug print
        #print(f"Finished generating triples for {dataset_name}. Total: {len(self.triples)} triples.\n")  # 🔍 Debug print
        print(f"Finished generating triples for {dataset_name}. Total: {len(self.triples)} triples.")  # Debug print

        # ✅ Print the first 5 triples
        print("\nFirst 5 triples:")
        for triple in self.triples[:5]:  
            print(triple)  
        print("\n")
        
        
        return self.triples
  
    
# Initialize the TripleGenerator
triple_generator = TripleGenerator()

# Generate the triples for each dataset
train_triples = triple_generator.generate_triples(train_dataset, "Train Dataset")
validation_triples = triple_generator.generate_triples(validation_dataset, "Validation Dataset")
test_triples = triple_generator.generate_triples(test_dataset, "Test Dataset")

  
# Tokenizer functions (as provided)
def create_vocab(corpus: str, min_freq: int = 5) -> dict:
    tokens = re.findall(r'\b\w+\b', corpus.lower())
    word_counts = collections.Counter(tokens)
    filtered = {word: count for word, count in word_counts.items() if count >= min_freq}
    vocab = {word: idx for idx, word in enumerate(sorted(filtered), start=2)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def save_vocab(vocab: dict, path: str):
    with open(path, 'w') as f:
        json.dump(vocab, f)

def load_vocab(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def text_to_ids(text: str, vocab: dict) -> list[int]:
    return [vocab.get(word.lower(), vocab['<UNK>']) for word in re.findall(r'\b\w+\b', text)]

import os
import json

# Check if the file exists
if os.path.exists("vocab.json"):
    # Open and read the file
    with open("vocab.json", "r") as file:
        data = json.load(file)
        print(data)
else:
    print("Error: vocab.json not found in the current directory.")

# Load the vocabulary
vocab = load_vocab("vocab.json")

# Tokenize the triples
def tokenize_triples(triples, vocab):
    tokenized_triples = []
    for triple in triples:
        query, relevant_doc, irrelevant_doc = triple
        tokenized_query = text_to_ids(query, vocab)
        tokenized_relevant_doc = text_to_ids(relevant_doc, vocab)
        tokenized_irrelevant_doc = text_to_ids(irrelevant_doc, vocab)
        tokenized_triples.append((tokenized_query, tokenized_relevant_doc, tokenized_irrelevant_doc))
    return tokenized_triples

train_tokenized_triples = tokenize_triples(train_triples, vocab)
validation_tokenized_triples = tokenize_triples(validation_triples, vocab)
test_tokenized_triples = tokenize_triples(test_triples, vocab)

# Create a reverse mapping from IDs to tokens
id_to_token = {idx: token for token, idx in vocab.items()}

# Example usage
print("Example tokenized triples:")
for i, triple in enumerate(train_tokenized_triples[:5]):
    print(f"Triple {i+1}: {triple}")


# Print final statistics
print(f"Total triples generated: {len(train_triples)} (Train), {len(validation_triples)} (Validation), {len(test_triples)} (Test)")

# Length of vocab - 96546
print(len(vocab))

'''
Triple Generation: The TripleGenerator class generates triples from the datasets.

Tokenization: The tokenize_triples function converts each triple into a tuple of token IDs using the vocabulary.

Vocabulary Mapping: The vocab dictionary maps tokens to IDs, and id_to_token maps IDs back to tokens.

Output: The tokenized triples are stored in train_tokenized_triples, validation_tokenized_triples, and test_tokenized_triples
'''


print(train_tokenized_triples[:5])
print(test_tokenized_triples[:5])
print(validation_tokenized_triples[:5])

# ensuring i have the correct size or amout of tripls : 
    
# Total triples generated: 70874 (Train), 8811 (Validation), 8838 (Test)
    
'''
If total_triples > total_samples, this is expected since each query can have multiple relevant documents.
If total_triples < total_samples, this suggests some queries have no relevant documents, so they were skipped.
If the count seems too high or low, it’s worth checking:
Are irrelevant documents being correctly sampled?
Are all relevant documents detected?
Are any queries missing?
'''
    

'''
Total number of queries : 100,000  
Train: 65,861 samples → Generated: 70,874 triples
Validation: 8,232 samples → Generated: 8,811 triples
Test: 8,233 samples → Generated: 8,838 triples

FORMAT OF TRIPLES : tuple of strings (query, relevant_document, irrelevant_document)
'''






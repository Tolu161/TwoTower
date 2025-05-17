#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:30:55 2025

@author: toluojo
"""


'''

FLOW OF OPERATIONS : 
    
- Generate triples (generate_triples()).
- Tokenize triples (tokenize_triples()).
- Embed tokenized triples (embed_triples()).
- Pad embedded triples (using pad_embeddings())

'''

import random
import pandas as pd
import collections
import re
import json
import numpy as np

# Load datasets from JSON
train_dataset = pd.read_json("train_dataset.json", orient="records", lines=True)
validation_dataset = pd.read_json("validation_dataset.json", orient="records", lines=True)
test_dataset = pd.read_json("test_dataset.json", orient="records", lines=True)

class TripleGenerator:
    def __init__(self):
        self.all_passages = []
        self.triples = []

    def extract_query(self, row):
        return row['query']

    def extract_passages(self, row):
        if isinstance(row['passages'], dict):
            return row['passages']['passage_text']
        return []

    def find_relevant_documents(self, row):
        if isinstance(row['passages'], dict):
            is_selected = row['passages']['is_selected']
            passage_texts = row['passages']['passage_text']
            
            # Separate highly relevant and less relevant passages
            highly_relevant = [passage_texts[i] for i, selected in enumerate(is_selected) if selected == 1]
            less_relevant = [passage_texts[i] for i, selected in enumerate(is_selected) if selected == 0]
        
            
            return highly_relevant, less_relevant  # Return both categories
        return [], []


    def sample_irrelevant_document(self, relevant_docs):
        if not self.all_passages:
            return None
        irrelevant_doc = random.choice(self.all_passages)
        while irrelevant_doc in relevant_docs:
            irrelevant_doc = random.choice(self.all_passages)
        return irrelevant_doc

    def generate_triples(self, dataset, dataset_name):
        self.triples = []
        for _, row in dataset.iterrows():
            query = self.extract_query(row)
            passages = self.extract_passages(row)
            self.all_passages.extend(passages)
            
            # Get both highly relevant and less relevant documents
            highly_relevant_docs, less_relevant_docs = self.find_relevant_documents(row)
            
            print("Highly relevant docs:", highly_relevant_docs)
            print("Less relevant docs:", less_relevant_docs)
        
            
            # Ensure at least one highly relevant passage exists
            if not highly_relevant_docs:
                continue
            
            for highly_relevant_doc in highly_relevant_docs:
                # Sample an 'irrelevant' document (which is actually just less relevant)
                irrelevant_doc = (
                    random.choice(less_relevant_docs) if less_relevant_docs else self.sample_irrelevant_document(highly_relevant_docs)
                )
                if irrelevant_doc is None:
                    continue
                
                # Store the (query, highly relevant doc, less relevant doc)
                self.triples.append((query, highly_relevant_doc, irrelevant_doc))
        return self.triples
    



# INSTANSIATING FUNCTIONS 
# Generate triples
triple_generator = TripleGenerator()
train_triples = triple_generator.generate_triples(train_dataset, "Train Dataset")
validation_triples = triple_generator.generate_triples(validation_dataset, "Validation Dataset")
test_triples = triple_generator.generate_triples(test_dataset, "Test Dataset")

#to save the triples to numpy arrays 
np.save('train_text_triples.npy', train_triples)
np.save('validation_text_triples.npy', validation_triples)
np.save('test_text_triples.npy', test_triples)





# Tokenizer functions
def load_vocab(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def text_to_ids(text: str, vocab: dict) -> list[int]:
    return [vocab.get(word.lower(), vocab['<UNK>']) for word in re.findall(r'\b\w+\b', text)]

# Load vocabulary
vocab = load_vocab("vocab.json")

def tokenize_triples(triples, vocab):
    tokenized_triples = []
    for triple in triples:
        tokenized_query = text_to_ids(triple[0], vocab)
        tokenized_relevant_doc = text_to_ids(triple[1], vocab)
        tokenized_irrelevant_doc = text_to_ids(triple[2], vocab)
        tokenized_triples.append((tokenized_query, tokenized_relevant_doc, tokenized_irrelevant_doc))
    return tokenized_triples



# Load pre-trained ID-to-embedding mappings
id_vectors = np.load('id_vectors.npy', allow_pickle=True).item() # The allow_pickle parameter determines whether the function is allowed to use Python's pickle module to serialize or deserialize data.

def ids_to_embeddings(token_ids: list[int], id_vectors: dict) -> list[np.array]:
    return [id_vectors.get(token_id, np.zeros(300)) for token_id in token_ids]

''' - REMOVING THE PADDING 
Padding with zero vectors to ensure each sequence has the same length, so that they can be processed in parallel - for batch processing in nn


# ENSURE HOMOGENOUS LENGTHS FOR EACH TRIPLE BY ADDING A PADDING 
# ENSURE THAT LIST OF TOKENISED WORDS/EMBEDDINGS ARE SAME LENGTH FOR EACH COMPONENTS 
# to process lengths of sequences in batches as nn requires all sequences in batch to be of the same length 
# here padding with ZERO vectors 
def pad_embeddings(embeddings, max_length, pad_value=np.zeros(300)):  # 300 is the embedding size
    return embeddings + [pad_value] * (max_length - len(embeddings)) if len(embeddings) < max_length else embeddings[:max_length]
'''

# Convert tokenized triples to embeddings, this retruns a list of tuples - each embedding -  a list of NumPy arrays.
def embed_triples(tokenized_triples, id_vectors):
    embedded_triples = []
    for tokenized_query, tokenized_relevant_doc, tokenized_irrelevant_doc in tokenized_triples:
        embedded_query = ids_to_embeddings(tokenized_query, id_vectors)
        embedded_relevant_doc = ids_to_embeddings(tokenized_relevant_doc, id_vectors)
        embedded_irrelevant_doc = ids_to_embeddings(tokenized_irrelevant_doc, id_vectors)
        embedded_triples.append((embedded_query, embedded_relevant_doc, embedded_irrelevant_doc))
    return embedded_triples


# TOKENISE TRIPLES
train_tokenized_triples = tokenize_triples(train_triples, vocab)
validation_tokenized_triples = tokenize_triples(validation_triples, vocab)
test_tokenized_triples = tokenize_triples(test_triples, vocab)

# EMBED THESE TRIPLES 
train_embedded_triples = embed_triples(train_tokenized_triples, id_vectors)
validation_embedded_triples = embed_triples(validation_tokenized_triples, id_vectors)
test_embedded_triples = embed_triples(test_tokenized_triples, id_vectors)
# print(train_embedded_triples[:1]) # optional just to check -DEBUGGING 



#train_embedded_triples = np.array(train_embedded_triples)
#validation_embedded_triples = np.array(validation_embedded_triples)
#test_embedded_triples = np.array(test_embedded_triples)



# send triples as text form - test - to the git repo 


'''
AVERAGE POOLING
It iterates over each tuple in the input list (embedded_triples unpacks the three components of the tuple 
and then calculated the average embedding and appends these to new average triple list 
'''

def average_embeddings(embedded_triples):
    averaged_triples = []
    
    for embedded_query, embedded_relevant_doc, embedded_irrelevant_doc in embedded_triples:
        
        # Calculate the mean embedding for each component
        # np.mean to calculate the means of the embeddings
        avg_query = np.mean(embedded_query, axis=0)
        avg_relevant_doc = np.mean(embedded_relevant_doc, axis=0)
        avg_irrelevant_doc = np.mean(embedded_irrelevant_doc, axis=0)
        
        # Append the averaged embeddings to the list
        averaged_triples.append((avg_query, avg_relevant_doc, avg_irrelevant_doc))
    
    return averaged_triples

# Average the embeddings for the train, validation, and test sets
train_averaged_triples = average_embeddings(train_embedded_triples)
validation_averaged_triples = average_embeddings(validation_embedded_triples)
test_averaged_triples = average_embeddings(test_embedded_triples)

# Save the averaged triples to numpy arrays
np.save('train_averaged_triples.npy', train_averaged_triples)
np.save('validation_averaged_triples.npy', validation_averaged_triples)
np.save('test_averaged_triples.npy', test_averaged_triples)

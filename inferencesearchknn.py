#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:29:31 2025

@author: toluojo
"""

'''
LOAD MODELS AS FUNCTIONS
'''


import torch

def load_query_tower(model_path):
    """
    Load the query tower model from the given path.
    """
    model = torch.load("best_tower_one")
    model.eval()  # Set to evaluation mode
    return model

def load_document_tower(model_path):
    """
    Load the document tower model from the given path.
    """
    model = torch.load("best_tower_two")
    model.eval()  # Set to evaluation mode
    return model

'''
PRECACHE DOCUMENT EMBEDDINGS
'''

import numpy as np

def pre_cache_document_embeddings(document_tower, documents):
    """
    Generate embeddings for all documents using the document tower.
    """
    document_embeddings = []
    for doc in documents:
        doc_tensor = torch.tensor([doc])  # Convert document to tensor
        with torch.no_grad():
            doc_embedding = document_tower(doc_tensor).numpy()  # Get embedding
        document_embeddings.append(doc_embedding)
    return np.array(document_embeddings)  # Convert to numpy array


'''
INFERENCE LOGIC
'''

from sklearn.neighbors import NearestNeighbors

class TwoTowerInference:
    def __init__(self, query_tower, document_tower, document_embeddings, documents):
        """
        Initialize the inference class with the query tower, document tower, pre-cached
        document embeddings, and the list of documents.
        """
        self.query_tower = query_tower
        self.document_tower = document_tower
        self.document_embeddings = document_embeddings
        self.documents = documents
        self.nn = NearestNeighbors(n_neighbors=5, metric='cosine')  # Initialize nearest neighbors
        self.nn.fit(document_embeddings)  # Fit the model to document embeddings

    def encode_query(self, query):
        """
        Tokenize and encode the query using the query tower.
        """
        query_tokens = self.tokenize(query)  # Tokenize the query
        query_tensor = torch.tensor([query_tokens])  # Convert to tensor
        with torch.no_grad():
            query_embedding = self.query_tower(query_tensor).numpy()  # Get embedding
        return query_embedding

    def find_nearest_neighbors(self, query_embedding, k=5):
        """
        Find the k nearest neighbors for the query embedding.
        """
        distances, indices = self.nn.kneighbors([query_embedding])
        return indices[0]  # Return indices of top-k documents

    def search(self, query, k=5):
        """
        Perform a search for the given query and return the top-k results.
        """
        query_embedding = self.encode_query(query)
        nearest_indices = self.find_nearest_neighbors(query_embedding, k)
        results = [self.documents[i] for i in nearest_indices]  # Retrieve documents
        return results

    def tokenize(self, text):
        """
        Tokenize the input text (replace with your actual tokenization logic).
        """
        # Example: Replace with your tokenizer (e.g., from transformers library)
        return text.split()  # Simple whitespace tokenizer for demonstration
    
    
    



'''
PUTTING IT ALL TOGETHER 

'''


# Step 1: Load models
query_tower = load_query_tower('query_tower.pth')
document_tower = load_document_tower('document_tower.pth')

# Step 2: Pre-cache document embeddings
documents = [...]  # List of tokenized documents
document_embeddings = pre_cache_document_embeddings(document_tower, documents)

# Step 3: Initialize inference class
inference_engine = TwoTowerInference(query_tower, document_tower, document_embeddings, documents)

# Step 4: Perform a search
query = "What is the capital of France?"
results = inference_engine.search(query, k=5)

# Print results
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc}")




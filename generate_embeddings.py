#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:28:49 2025

@author: toluojo
"""

# GENERATING THE EMBEDDINGS 


from gensim.models import Word2Vec

# Load your trained Word2Vec model
word2vec_model = Word2Vec.load("word2vec.model")


# Generate Embeddings for Tokenized Data


def generate_embeddings_for_triples(tokenized_triples, id_to_embedding):
    embedded_triples = []
    for triple in tokenized_triples:
        query, relevant_doc, irrelevant_doc = triple
        # Convert token IDs to embeddings
        embedded_query = [id_to_embedding.get(token_id, np.zeros(word2vec_model.vector_size)) for token_id in query]
        embedded_relevant_doc = [id_to_embedding.get(token_id, np.zeros(word2vec_model.vector_size)) for token_id in relevant_doc]
        embedded_irrelevant_doc = [id_to_embedding.get(token_id, np.zeros(word2vec_model.vector_size)) for token_id in irrelevant_doc]
        # Append the embedded triple
        embedded_triples.append((embedded_query, embedded_relevant_doc, embedded_irrelevant_doc))
    return embedded_triples

# Generate embeddings for your tokenized triples
train_embedded_triples = generate_embeddings_for_triples(train_tokenized_triples, id_to_embedding)
validation_embedded_triples = generate_embeddings_for_triples(validation_tokenized_triples, id_to_embedding)
test_embedded_triples = generate_embeddings_for_triples(test_tokenized_triples, id_to_embedding)


# Example: Print the first embedded triple
print("First embedded triple:")
print("Query:", train_embedded_triples[0][0])
print("Relevant Doc:", train_embedded_triples[0][1])
print("Irrelevant Doc:", train_embedded_triples[0][2])
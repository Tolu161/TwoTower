#this will download the data from huggingface to your local machine in tt/ if it doesn't already exist

import requests
import os
import collections
import datasets
import re
import json

script_dir = os.path.dirname(__file__)

def create_vocab(corpus: str, min_freq: int = 5) -> dict:
    # Tokenize and clean
    tokens = re.findall(r'\b\w+\b', corpus.lower())
    
    # Count frequencies
    word_counts = collections.Counter(tokens)
    
    # Filter by frequency
    filtered = {word: count for word, count in word_counts.items() 
                if count >= min_freq}
    
    # Create vocabulary mapping
    vocab = {word: idx for idx, word in enumerate(sorted(filtered), start=2)}
    
    # Add special tokens
    vocab['<PAD>'] = 0  # Padding token
    vocab['<UNK>'] = 1  # Unknown words
    
    return vocab

def save_vocab(vocab: dict, path: str):
    with open(path, 'w') as f:
        json.dump(vocab, f)

def load_vocab(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def get_documents_safe(passages_list: list[dict]) -> list[str]:
    """Safely extract documents with validation"""
    docs = []
    for passage in passages_list:
        for document in passage:
            try:
                docs.append(document)
            except KeyError:
                print(f"Warning: Missing 'document' in {document}")
    return docs

def combine_strings(docs: list) -> str:
    """Safely combine list of strings into a single string"""
    return ' '.join(str(doc) for doc in docs)

def text_to_ids(text: str, vocab: dict) -> list[int]:
    return [vocab.get(word.lower(), vocab['<UNK>']) 
            for word in re.findall(r'\b\w+\b', text)]

def ids_to_text(ids: list[int], vocab: dict) -> str:
    """Convert list of token IDs to their corresponding words"""
    # Handle single integer input
    if isinstance(ids, int):
        ids = [ids]
    
    # Create inverse vocabulary mapping (ID -> word)
    id_to_word = {v: k for k, v in vocab.items()}
    
    # Convert IDs to words, handling unknown IDs
    return ' '.join(id_to_word.get(id, "<UNK>") for id in ids)


if __name__== "__main__":

    # create vocab.json if executing script
    #Check if text8 and download
    if not os.path.exists(os.path.join(script_dir, "text8")):
        print(f"text8 not found, downloading now...")
        r = requests.get("https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8")
        with open("text8", "wb") as f:
            f.write(r.content)
        print(f"text8 downloaded and saved")
    
    #Check if ms_marco parquet files exist and download all if not
    if not os.path.exists(os.path.join(script_dir, "ms_marco_train.parquet")):
        print(f"ms_marco parquets not found, downloading now...")
        ds = datasets.load_dataset("microsoft/ms_marco", "v1.1")
        ds["train"].to_parquet("ms_marco_train.parquet")
        ds["validation"].to_parquet("ms_marco_validation.parquet")
        ds["test"].to_parquet("ms_marco_test.parquet")
        print(f"ms_marco parquets downloaded and saved")
    
    ds_train = datasets.Dataset.from_parquet("ms_marco_train.parquet")
    ds_validation = datasets.Dataset.from_parquet("ms_marco_validation.parquet")
    ds_test = datasets.Dataset.from_parquet("ms_marco_test.parquet")

    #get combined queries
    queries = ds_validation['query'] + ds_train['query'] + ds_test['query']
    queries_texts = [query for query in queries if query]
    queries_corpus = combine_strings(queries_texts)

    #get combined passages
    passages = ds_validation['passages'] + ds_train['passages'] + ds_test['passages']
    passage_texts = [item['passage_text'] for item in passages]
    list_of_docs = get_documents_safe(passage_texts)
    passage_corpus = combine_strings(list_of_docs)     

    #Load text8 corpus
    with open(os.path.join(script_dir, "text8"), "r", encoding='utf-8') as f:
            text8_corpus = f.read()
    
    # combine corpus
    corpus = text8_corpus + ' ' + passage_corpus + ' ' + queries_corpus
    
    vocab = create_vocab(corpus, min_freq=10)
    save_vocab(vocab, "vocab.json")            
    print(f"{len(vocab)} words in vocab")

    example = "wHat Dog is this dog?"

    print(f"Example text: {example}")

    print(f"Example ids: {text_to_ids(example, vocab)}")
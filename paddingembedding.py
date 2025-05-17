#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:28:55 2025

@author: toluojo
"""

# PADDING 
import numpy as np
import torch



data = np.load("train1_embedded_triples.npy", allow_pickle=True)
print(data)



def pad_triples_window(npy_files, max_length, window_start, window_size, pad_subset_ratio=0.1, chunk_size=100):
    """
    Load triples from multiple npy files using memory mapping, extract a window of data in chunks, 
    and apply padding only to a subset.
    
    Parameters:
    - npy_files: Dictionary of file paths {"train": path, "test": path, "validation": path}.
    - max_length: Maximum sequence length for padding.
    - window_start: Start index for extracting a window from the data.
    - window_size: Number of triples to process in this window.
    - pad_subset_ratio: Ratio of the windowed data to apply padding to (default 10%).
    - chunk_size: Number of triples to process at a time to reduce memory usage.
    
    Returns:
    - A generator yielding padded triples in PyTorch tensor format.
    """
    def pad_sequence(seq):
        return np.pad(seq[:max_length], (0, max_length - min(len(seq), max_length)), 'constant')
    
    for key, npy_file in npy_files.items():
        triples = np.load(npy_file, mmap_mode='r', allow_pickle=True)  # Use memory mapping to reduce RAM usage
        
        for chunk_start in range(window_start, window_start + window_size, chunk_size):
            chunk_triples = triples[chunk_start:chunk_start + chunk_size]
            num_to_pad = int(len(chunk_triples) * pad_subset_ratio)
            indices_to_pad = np.random.choice(len(chunk_triples), num_to_pad, replace=False)
            
            for i, (query, relevant_doc, irrelevant_doc) in enumerate(chunk_triples):
                if i in indices_to_pad:  # Only apply padding to selected subset
                    query, relevant_doc, irrelevant_doc = map(pad_sequence, [query, relevant_doc, irrelevant_doc])
                
                yield (
                    torch.tensor(query, dtype=torch.long),
                    torch.tensor(relevant_doc, dtype=torch.long),
                    torch.tensor(irrelevant_doc, dtype=torch.long)
                )

# Example usage
max_length = 256
window_start = 0  # Adjust as needed
window_size = 1000  # Process 1000 triples at a time

npy_files = {
    "train": "train1_embedded_triples.npy",
    "test": "test1_embedded_triples.npy",
    "validation": "validation1_embedded_triples.npy"
}

padded_triples = pad_triples_window(npy_files, max_length, window_start, window_size)

# Save as PyTorch tensor files
for key in npy_files.keys():
    torch.save(list(pad_triples_window({key: npy_files[key]}, max_length, window_start, window_size)), f'{key}_padded_triples.pt')


















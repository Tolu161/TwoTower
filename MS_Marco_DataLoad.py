#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:28:49 2025

@author: toluojo
"""

# Downloading the MS MARCO Dataset 

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from collections import Counter


import nltk 
nltk.download("punkt")

from nltk.tokenize import word_tokenize

# Load MS MARCO dataset AND SPLITTING THE DATA 

from datasets import load_dataset

#ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")

# Load and split the dataset in one go
train_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train[:80%]")
validation_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train[80%:90%]")
test_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train[90%:]")

# Print the sizes of each split
print(f"Train: {len(train_dataset)} samples")
print(f"Validation: {len(validation_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")

df_train = pd.DataFrame(train_dataset)
df_validation = pd.DataFrame(validation_dataset)
df_test = pd.DataFrame(test_dataset)

train = df_train[['query','passages','query_id']]
validate = df_validation[['query','passages','query_id']]
test = df_test[['query','passages','query_id']]

# Save datasets as JSON instead of CSV to preserve structure
train.to_json("train_dataset.json", orient="records", lines=True)
validate.to_json("validation_dataset.json", orient="records", lines=True)
test.to_json("test_dataset.json", orient="records", lines=True)

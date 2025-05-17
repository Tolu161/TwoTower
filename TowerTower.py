#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:18:52 2025

@author: toluojo
"""


'''
-----STEPS TO BUILD TWO TOWER :---------------- 
- Unpack the Tuple: Assuming the  tuple output is (query_embedding, related_doc_embedding, unrelated_doc_embedding), extract them individually.
- Pass Query through TowerOne.
- Pass Both Related and Unrelated Documents through TowerTwo.
- Compute Cosine Similarity Scores.

'''

'''
TWO TOWER MODEL 
'''

# why alpha - it Alpha can control the weighting between the positive (related) and negative (unrelated) loss components.
# Higher α means we prioritise learning from positive pairs more.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import wandb

# End any existing Wandb runs
wandb.finish()

# Initialize Weights & Biases
wandb.init(project="two_tower_model", config={
    "learning_rate": 0.003,
    "epochs": 100,
    "batch_size": 32,  # Ensure this key is present
    "alpha": 0.7,
    "hidden_dim": 300,
    "dropout_rate": 0.2,
    "margin": 1.0,
    "step_size": 5,
    "gamma": 0.1
})

# Print the config to verify
print("Wandb Config:", wandb.config)

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_triples = torch.load('train_padded_triples.pt')
validation_triples = torch.load('validation_padded_triples.pt')
test_triples = torch.load('test_padded_triples.pt')

# Define a fixed sequence length
MAX_SEQ_LEN = 600  # Adjust this based on your data

# Dataset class with padding
class TriplesDataset(Dataset):
    def __init__(self, triples, max_seq_len=MAX_SEQ_LEN):
        self.triples = triples
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        query, related, unrelated = self.triples[idx]

        # Pad or truncate sequences to a fixed length
        def pad_or_truncate(sequence):
            if sequence.shape[0] < self.max_seq_len:
                # Pad with zeros
                padding = torch.zeros((self.max_seq_len - sequence.shape[0], sequence.shape[1]), dtype=torch.float32)
                return torch.cat([sequence, padding], dim=0)
            else:
                # Truncate
                return sequence[:self.max_seq_len, :]

        query = pad_or_truncate(query)
        related = pad_or_truncate(related)
        unrelated = pad_or_truncate(unrelated)

        return query, related, unrelated

# Create DataLoader with fallback for batch_size
BATCH_SIZE = wandb.config.batch_size if hasattr(wandb.config, 'batch_size') else 32
print(f"Using batch size: {BATCH_SIZE}")

train_dataset = TriplesDataset(train_triples)
val_dataset = TriplesDataset(validation_triples)
test_dataset = TriplesDataset(test_triples)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the RNN-based Tower model
class Tower(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300, num_layers=3):
        super(Tower, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        _, hidden_state = self.rnn(x)
        final_hidden_state = hidden_state[-1]  # Extract last layer's hidden state
        prediction = self.fc(final_hidden_state)
        return prediction

# Define Two-Tower Model
class TwoTowerModel(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300, num_layers=1):
        super(TwoTowerModel, self).__init__()
        self.query_tower = Tower(input_dim, hidden_dim, num_layers)  # RNN-based Query Tower
        self.doc_tower = Tower(input_dim, hidden_dim, num_layers)  # RNN-based Document Tower

    def forward(self, query, document):
        query_emb = self.query_tower(query)
        doc_emb = self.doc_tower(document)
        return query_emb, doc_emb

# Correct instantiation
two_tower_model = TwoTowerModel(input_dim=300, hidden_dim=wandb.config.hidden_dim, num_layers=1).to(device)

# Optimizer and scheduler
optimizer = torch.optim.Adam(two_tower_model.parameters(), lr=wandb.config.learning_rate)
scheduler = StepLR(optimizer, step_size=wandb.config.step_size, gamma=wandb.config.gamma)

# Loss function
def triplet_loss_function(query, relevant_document, irrelevant_document, margin):
    relevant_distance = F.cosine_similarity(query, relevant_document)
    irrelevant_distance = F.cosine_similarity(query, irrelevant_document)
    triplet_loss = F.relu(relevant_distance - irrelevant_distance + margin)
    return triplet_loss.mean()

# Training function
def train_epoch(model, dataloader, optimizer, margin):
    model.train()
    total_loss = 0
    
    for query, related, unrelated in tqdm(dataloader, desc="Training", leave=False):
        query, related, unrelated = query.to(device), related.to(device), unrelated.to(device)
        optimizer.zero_grad()
        
        # Forward Pass
        query_emb, related_emb = model(query, related)
        _, unrelated_emb = model(query, unrelated)
        
        # Compute loss
        loss = triplet_loss_function(query_emb, related_emb, unrelated_emb, margin)
        
        # Backpropagation 
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Validation function
def validate_epoch(model, dataloader, margin):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for query, related, unrelated in tqdm(dataloader, desc="Validation", leave=False):
            query, related, unrelated = query.to(device), related.to(device), unrelated.to(device)
            
            # Forward pass
            query_emb, related_emb = model(query, related)
            _, unrelated_emb = model(query, unrelated)
            
            # Compute loss
            loss = triplet_loss_function(query_emb, related_emb, unrelated_emb, margin)
            
            # Accumulate loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Training loop
best_val_loss = float('inf')
for epoch in range(wandb.config.epochs):
    # Train for one epoch
    train_loss = train_epoch(two_tower_model, train_loader, optimizer, wandb.config.margin)
    
    # Validate for one epoch
    val_loss = validate_epoch(two_tower_model, val_loader, wandb.config.margin)
    
    # Log metrics to Wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Validation Loss": val_loss,
        "Learning Rate": scheduler.get_last_lr()[0]
    })
    
    # Learning rate scheduling
    scheduler.step()
    
    # Early stopping and saving best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(two_tower_model.state_dict(), 'best_two_tower_model.pth')
        print(f"New best model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
    else:
        print(f"No improvement in validation loss at epoch {epoch + 1}")

# Testing phase
two_tower_model.load_state_dict(torch.load('best_two_tower_model.pth'))
test_loss = validate_epoch(two_tower_model, test_loader, wandb.config.margin)

# Log test metrics to Wandb
wandb.log({
    "Test Loss": test_loss
})

# Print test results
print(f"Test Loss: {test_loss:.4f}")

# Finish Wandb run
wandb.finish()






















''' - RuntimeError: stack expects each tensor to be equal size, but got [256, 553] at entry 0 and [256, 540] at entry 1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import wandb

# End any existing Wandb runs
wandb.finish()

# Initialize Weights & Biases
wandb.init(project="two_tower_model", config={
    "learning_rate": 0.003,
    "epochs": 100,
    "batch_size": 32,  # Ensure this key is present
    "alpha": 0.7,
    "hidden_dim": 300,
    "dropout_rate": 0.2,
    "margin": 1.0,
    "step_size": 5,
    "gamma": 0.1
})

# Print the config to verify
print("Wandb Config:", wandb.config)

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
#def load_data(file_path):
    #return np.load(file_path, allow_pickle=True)


# Loading the TRIPLES not averaged since rnn -HIDDEN STATE

#train_triples = load_data('train1_em_triples.npy')
#validation_triples = load_data('validation1_averaged_triples.npy')
#test_triples = load_data('test1_averaged_triples.npy')


train_triples = torch.load('train_padded_triples.pt')
print(train_triples)
validation_triples = torch.load('validation_padded_triples.pt')
print('val_triples')
test_triples = torch.load('test_padded_triples.pt')
print(test_triples)


# because RNN not using averaged embedding 
#train_triples = load_data('train1_averaged_triples.npy')
#validation_triples = load_data('validation1_averaged_triples.npy')
#test_triples = load_data('test1_averaged_triples.npy')

# Dataset class
class TriplesDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        query, related, unrelated = self.triples[idx]
        return torch.tensor(query, dtype=torch.float32), torch.tensor(related, dtype=torch.float32), torch.tensor(unrelated, dtype=torch.float32)

# Create DataLoader with fallback for batch_size
BATCH_SIZE = wandb.config.batch_size if hasattr(wandb.config, 'batch_size') else 32
print(f"Using batch size: {BATCH_SIZE}")

train_dataset = TriplesDataset(train_triples)
val_dataset = TriplesDataset(validation_triples)
test_dataset = TriplesDataset(test_triples)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Define the RNN-based Tower model
class Tower(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300, num_layers=3):
        super(Tower, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        _, hidden_state = self.rnn(x)
        final_hidden_state = hidden_state[-1]  # Extract last layer's hidden state
        prediction = self.fc(final_hidden_state)
        return prediction

# Define Two-Tower Model
class TwoTowerModel(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300, num_layers=1):
        super(TwoTowerModel, self).__init__()
        self.query_tower = Tower(input_dim, hidden_dim, num_layers)  # RNN-based Query Tower
        self.doc_tower = Tower(input_dim, hidden_dim, num_layers)  # RNN-based Document Tower

    def forward(self, query, document):
        query_emb = self.query_tower(query)
        doc_emb = self.doc_tower(document)
        return query_emb, doc_emb

# Correct instantiation
two_tower_model = TwoTowerModel(input_dim=300, hidden_dim=wandb.config.hidden_dim, num_layers=1).to(device)

# Optimizer and scheduler
optimizer = torch.optim.Adam(two_tower_model.parameters(), lr=wandb.config.learning_rate)
scheduler = StepLR(optimizer, step_size=wandb.config.step_size, gamma=wandb.config.gamma)

# Loss function
def triplet_loss_function(query, relevant_document, irrelevant_document, margin):
    relevant_distance = F.cosine_similarity(query, relevant_document)
    irrelevant_distance = F.cosine_similarity(query, irrelevant_document)
    triplet_loss = F.relu(relevant_distance - irrelevant_distance + margin)
    return triplet_loss.mean()

# Training function
def train_epoch(model, dataloader, optimizer, margin):
    model.train()
    total_loss = 0
    
    for query, related, unrelated in tqdm(dataloader, desc="Training", leave=False):
        query, related, unrelated = query.to(device), related.to(device), unrelated.to(device)
        optimizer.zero_grad()
        
        # Forward Pass
        query_emb, related_emb = model(query, related)
        _, unrelated_emb = model(query, unrelated)
        
        # Compute loss
        loss = triplet_loss_function(query_emb, related_emb, unrelated_emb, margin)
        
        # Backpropagation 
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Validation function
def validate_epoch(model, dataloader, margin):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for query, related, unrelated in tqdm(dataloader, desc="Validation", leave=False):
            query, related, unrelated = query.to(device), related.to(device), unrelated.to(device)
            
            # Forward pass
            query_emb, related_emb = model(query, related)
            _, unrelated_emb = model(query, unrelated)
            
            # Compute loss
            loss = triplet_loss_function(query_emb, related_emb, unrelated_emb, margin)
            
            # Accumulate loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Training loop
best_val_loss = float('inf')
for epoch in range(wandb.config.epochs):
    # Train for one epoch
    train_loss = train_epoch(two_tower_model, train_loader, optimizer, wandb.config.margin)
    
    # Validate for one epoch
    val_loss = validate_epoch(two_tower_model, val_loader, wandb.config.margin)
    
    # Log metrics to Wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Validation Loss": val_loss,
        "Learning Rate": scheduler.get_last_lr()[0]
    })
    
    # Learning rate scheduling
    scheduler.step()
    
    # Early stopping and saving best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(two_tower_model.state_dict(), 'best_two_tower_model.pth')
        print(f"New best model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
    else:
        print(f"No improvement in validation loss at epoch {epoch + 1}")

# Testing phase
two_tower_model.load_state_dict(torch.load('best_two_tower_model.pth'))
test_loss = validate_epoch(two_tower_model, test_loader, wandb.config.margin)

# Log test metrics to Wandb
wandb.log({
    "Test Loss": test_loss
})

# Print test results
print(f"Test Loss: {test_loss:.4f}")

# Finish Wandb run
wandb.finish()

'''











































''' REDEFINED TWO TOWER MODEL 
# Define the RNN-based Tower model
class Tower(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300, num_layers=3):
        super(Tower, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        _, hidden_state = self.rnn(x)
        final_hidden_state = hidden_state[-1]  # Extract last layer's hidden state
        prediction = self.fc(final_hidden_state)
        return prediction

# Define Two-Tower Model
class TwoTowerModel(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300, num_layers=1):
        super(TwoTowerModel, self).__init__()
        self.query_tower = Tower(input_dim, hidden_dim, num_layers)  # RNN-based Query Tower
        self.doc_tower = Tower(input_dim, hidden_dim, num_layers)  # RNN-based Document Tower

    def forward(self, query, document):
        query_emb = self.query_tower(query)
        doc_emb = self.doc_tower(document)
        return query_emb, doc_emb

# Correct instantiation
two_tower_model = TwoTowerModel(input_dim=300, hidden_dim=wandb.config.hidden_dim, num_layers=1).to(device)

# Optimizer and scheduler
optimizer = torch.optim.Adam(list(query_tower.parameters()) + list(document_tower.parameters()), lr=wandb.config.learning_rate)
scheduler = StepLR(optimizer, step_size=wandb.config.step_size, gamma=wandb.config.gamma)

# Loss function
def triplet_loss_function(query, relevant_document, irrelevant_document, margin):
    relevant_distance = F.cosine_similarity(query, relevant_document)
    irrelevant_distance = F.cosine_similarity(query, irrelevant_document)
    triplet_loss = F.relu(relevant_distance - irrelevant_distance + margin)
    return triplet_loss.mean()

# Training function
def train_epoch(model_one, model_two, dataloader, optimizer, margin):
    model_one.train()
    model_two.train()
    total_loss = 0
    
    for query, related, unrelated in tqdm(dataloader, desc="Training", leave=False):
        query, related, unrelated = query.to(device), related.to(device), unrelated.to(device)
        optimizer.zero_grad()
        
        # Forward Pass
        query_transformed = model_one(query)
        related_transformed = model_two(related)
        unrelated_transformed = model_two(unrelated)
        
        # Compute loss
        loss = triplet_loss_function(query_transformed, related_transformed, unrelated_transformed, margin)
        
        # Backpropagation 
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Validation function
def validate_epoch(model_one, model_two, dataloader, margin):
    model_one.eval()
    model_two.eval()
    total_loss = 0
    
    with torch.no_grad():
        for query, related, unrelated in tqdm(dataloader, desc="Validation", leave=False):
            query, related, unrelated = query.to(device), related.to(device), unrelated.to(device)
            
            # Forward pass
            query_transformed = model_one(query)
            related_transformed = model_two(related)
            unrelated_transformed = model_two(unrelated)
            
            # Compute loss
            loss = triplet_loss_function(query_transformed, related_transformed, unrelated_transformed, margin)
            
            # Accumulate loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Training loop
best_val_loss = float('inf')
for epoch in range(wandb.config.epochs):
    # Train for one epoch
    train_loss = train_epoch(tower_one, tower_two, train_loader, optimizer, wandb.config.margin)
    
    # Validate for one epoch
    val_loss = validate_epoch(tower_one, tower_two, val_loader, wandb.config.margin)
    
    # Log metrics to Wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Validation Loss": val_loss,
        "Learning Rate": scheduler.get_last_lr()[0]
    })
    
    # Learning rate scheduling
    scheduler.step()
    
    # Early stopping and saving best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(tower_one.state_dict(), 'best_tower_one.pth')
        torch.save(tower_two.state_dict(), 'best_tower_two.pth')
        print(f"New best model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
    else:
        print(f"No improvement in validation loss at epoch {epoch + 1}")

# Testing phase
tower_one.load_state_dict(torch.load('best_tower_one.pth'))
tower_two.load_state_dict(torch.load('best_tower_two.pth'))

# Evaluate on the test set
test_loss = validate_epoch(tower_one, tower_two, test_loader, wandb.config.margin)

# Log test metrics to Wandb
wandb.log({
    "Test Loss": test_loss
})

# Print test results
print(f"Test Loss: {test_loss:.4f}")

# Finish Wandb run
wandb.finish()

'''


















'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

# End any existing Wandb runs
wandb.finish()

# Initialize Weights & Biases
wandb.init(project="two_tower_model", config={
    "learning_rate": 0.003,
    "epochs": 100,
    "batch_size": 32,  # Ensure this key is present
    "alpha": 0.7,
    "hidden_dim": 300,
    "dropout_rate": 0.2,
    "margin": 1.0,
    "step_size": 5,
    "gamma": 0.1
})

# Print the config to verify
print("Wandb Config:", wandb.config)

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_data(file_path):
    return np.load(file_path, allow_pickle=True)

train_triples = load_data('train_averaged_triples.npy')
validation_triples = load_data('validation_averaged_triples.npy')
test_triples = load_data('test_averaged_triples.npy')

# Dataset class
class TriplesDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        query, related, unrelated = self.triples[idx]
        return torch.tensor(query, dtype=torch.float32), torch.tensor(related, dtype=torch.float32), torch.tensor(unrelated, dtype=torch.float32)

# Create DataLoader with fallback for batch_size
BATCH_SIZE = wandb.config.batch_size if hasattr(wandb.config, 'batch_size') else 32
print(f"Using batch size: {BATCH_SIZE}")

train_dataset = TriplesDataset(train_triples)
val_dataset = TriplesDataset(validation_triples)
test_dataset = TriplesDataset(test_triples)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the Tower model
class Tower(nn.Module):
    
    def __init__(self, input_dim=300, hidden_dim=300):
        
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x


# Initialize models
tower_one = Tower(input_dim=300, hidden_dim=wandb.config.hidden_dim).to(device)
tower_two = Tower(input_dim=300, hidden_dim=wandb.config.hidden_dim).to(device)


# Define Two-Tower Model
class TwoTowerModel(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300):
        super(TwoTowerModel, self).__init__()
        self.query_tower = Tower(input_dim, hidden_dim)  # Simple Query Tower
        self.doc_tower = Tower(input_dim, hidden_dim)  # Simple Document Tower

    def forward(self, query, document):
        query_emb = self.query_tower(query)
        doc_emb = self.doc_tower(document)
        return query_emb, doc_emb




# Optimizer and scheduler
optimizer = torch.optim.Adam(list(tower_one.parameters()) + list(tower_two.parameters()), lr=wandb.config.learning_rate)
scheduler = StepLR(optimizer, step_size=wandb.config.step_size, gamma=wandb.config.gamma)

# Loss function #  this returns the average distances for related and unrelated documents
def contrastive_loss(query, related, unrelated, alpha, margin):
    score_related = F.cosine_similarity(query, related, dim=1)
    score_unrelated = F.cosine_similarity(query, unrelated, dim=1)
    
    dist_related = 1 - score_related
    dist_unrelated = 1 - score_unrelated
    
    loss_related = alpha * dist_related.pow(2).mean()
    loss_unrelated = (1 - alpha) * F.relu(margin - dist_unrelated).pow(2).mean()
    total_loss = loss_related + loss_unrelated
    
    return total_loss, loss_related, loss_unrelated, dist_related.mean(), dist_unrelated.mean()

# Training function
def train_epoch(model_one, model_two, dataloader, optimizer, alpha, margin):
    model_one.train()
    model_two.train()
    total_loss = 0
    total_loss_related = 0
    total_loss_unrelated = 0
    total_dist_related = 0
    total_dist_unrelated = 0
    
    for query, related, unrelated in tqdm(dataloader, desc="Training", leave=False):
        query, related, unrelated = query.to(device), related.to(device), unrelated.to(device)
        optimizer.zero_grad()
        
        # Forward Pass
        query_transformed = model_one(query)
        related_transformed = model_two(related)
        unrelated_transformed = model_two(unrelated)
        
        # Compute loss and distances
        loss, loss_related, loss_unrelated, dist_related, dist_unrelated = contrastive_loss(
           query_transformed, related_transformed, unrelated_transformed, alpha, margin
       )
        #loss, loss_related, loss_unrelated = contrastive_loss(query_transformed, related_transformed, unrelated_transformed, alpha, margin)
        
        
        # Bacckpropagation 
        loss.backward()
        optimizer.step()
        
        #Accumulate losses and distances 
        total_loss += loss.item()
        total_loss_related += loss_related.item()
        total_loss_unrelated += loss_unrelated.item()
        total_dist_related += dist_related.item()
        total_dist_unrelated += dist_unrelated.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_loss_related = total_loss_related / len(dataloader)
    avg_loss_unrelated = total_loss_unrelated / len(dataloader)
    avg_dist_related = total_dist_related / len(dataloader)
    avg_dist_unrelated = total_dist_unrelated / len(dataloader)
    
    return avg_loss, avg_loss_related, avg_loss_unrelated, avg_dist_related, avg_dist_unrelated

# Validation function
def validate_epoch(model_one, model_two, dataloader, alpha, margin):
    model_one.eval()  # Set model_one to evaluation mode
    model_two.eval()  # Set model_two to evaluation mode
    
    total_loss = 0
    total_loss_related = 0
    total_loss_unrelated = 0
    total_dist_related = 0
    total_dist_unrelated = 0
    
    with torch.no_grad():  # Disable gradient computation for validation
        for query, related, unrelated in tqdm(dataloader, desc="Validation", leave=False):
            # Move data to the correct device (e.g., GPU)
            query, related, unrelated = query.to(device), related.to(device), unrelated.to(device)
            
            # Forward pass through the towers
            query_transformed = model_one(query)
            related_transformed = model_two(related)
            unrelated_transformed = model_two(unrelated)
            
            # Compute the contrastive loss
            loss, loss_related, loss_unrelated, dist_related, dist_unrelated  = contrastive_loss(
                query_transformed, related_transformed, unrelated_transformed, alpha, margin
            )
            
            # Accumulate losses
            total_loss += loss.item()
            total_loss_related += loss_related.item()
            total_loss_unrelated += loss_unrelated.item()
            total_dist_related += dist_related.item()
            total_dist_unrelated += dist_unrelated.item()
    
    # Compute average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_related = total_loss_related / len(dataloader)
    avg_loss_unrelated = total_loss_unrelated / len(dataloader)
    avg_dist_related = total_dist_related / len(dataloader)
    avg_dist_unrelated = total_dist_unrelated / len(dataloader)
    
    return avg_loss, avg_loss_related, avg_loss_unrelated, avg_dist_related, avg_dist_unrelated


# Training loop
best_val_loss = float('inf')
for epoch in range(wandb.config.epochs):
    # Train for one epoch
    train_loss, train_loss_related, train_loss_unrelated, train_dist_related, train_dist_unrelated = train_epoch(
        tower_one, tower_two, train_loader, optimizer, wandb.config.alpha, wandb.config.margin
    )
    
    # Validate for one epoch
    val_loss, val_loss_related, val_loss_unrelated, val_dist_related, val_dist_unrelated = validate_epoch(
        tower_one, tower_two, val_loader, wandb.config.alpha, wandb.config.margin
    )
    
    # Log metrics to Wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Train Loss Related": train_loss_related,
        "Train Loss Unrelated": train_loss_unrelated,
        "Train Distance Related": train_dist_related,
        "Train Distance Unrelated": train_dist_unrelated,
        "Validation Loss": val_loss,
        "Validation Loss Related": val_loss_related,
        "Validation Loss Unrelated": val_loss_unrelated,
        "Validation Distance Related": val_dist_related,
        "Validation Distance Unrelated": val_dist_unrelated,
        "Learning Rate": scheduler.get_last_lr()[0]
    })
    
    # Learning rate scheduling
    scheduler.step()
    
    # Early stopping and saving best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(tower_one.state_dict(), 'best_tower_one.pth')
        torch.save(tower_two.state_dict(), 'best_tower_two.pth')
        print(f"New best model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
    else:
        print(f"No improvement in validation loss at epoch {epoch + 1}")

# Testing phase
tower_one.load_state_dict(torch.load('best_tower_one.pth'))
tower_two.load_state_dict(torch.load('best_tower_two.pth'))

# Evaluate on the test set
test_loss, test_loss_related, test_loss_unrelated, test_dist_related, test_dist_unrelated = validate_epoch(
    tower_one, tower_two, test_loader, wandb.config.alpha, wandb.config.margin
)

# Log test metrics to Wandb
wandb.log({
    "Test Loss": test_loss,
    #"Test Loss Related": test_loss_related,
    #"Test Loss Unrelated": test_loss_unrelated,
    "Test Distance Related": test_dist_related,
    "Test Distance Unrelated": test_dist_unrelated
})

# Print test results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Loss Related: {test_loss_related:.4f}")
print(f"Test Loss Unrelated: {test_loss_unrelated:.4f}")
print(f"Test Distance Related: {test_dist_related:.4f}")
print(f"Test Distance Unrelated: {test_dist_unrelated:.4f}")

# Finish Wandb run
wandb.finish()

'''






'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR




# Initialize Weights & Biases
wandb.init(project="two_tower_model", config={
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32,
    "alpha": 0.7,
    "hidden_dim": 300,
    "dropout_rate": 0.2,
    "margin": 0.5,
    "step_size": 5,
    "gamma": 0.1
})

# print wandb


# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_data(file_path):
    return np.load(file_path, allow_pickle=True)

train_triples = load_data('train1_averaged_triples.npy')
validation_triples = load_data('validation1_averaged_triples.npy')
test_triples = load_data('test1_averaged_triples.npy')

# Dataset class
class TriplesDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        query, related, unrelated = self.triples[idx]
        return torch.tensor(query, dtype=torch.float32), torch.tensor(related, dtype=torch.float32), torch.tensor(unrelated, dtype=torch.float32)

# Create DataLoader
BATCH_SIZE = wandb.config.batch_size
train_dataset = TriplesDataset(train_triples)
val_dataset = TriplesDataset(validation_triples)
test_dataset = TriplesDataset(test_triples)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the Tower model
class Tower(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300, dropout_rate=0.2):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialize models
tower_one = Tower(input_dim=300, hidden_dim=wandb.config.hidden_dim, dropout_rate=wandb.config.dropout_rate).to(device)
tower_two = Tower(input_dim=300, hidden_dim=wandb.config.hidden_dim, dropout_rate=wandb.config.dropout_rate).to(device)

# Optimizer and scheduler
optimizer = torch.optim.Adam(list(tower_one.parameters()) + list(tower_two.parameters()), lr=wandb.config.learning_rate)
scheduler = StepLR(optimizer, step_size=wandb.config.step_size, gamma=wandb.config.gamma)

# Loss function
def contrastive_loss(query, related, unrelated, alpha, margin):
    score_related = F.cosine_similarity(query, related, dim=1)
    score_unrelated = F.cosine_similarity(query, unrelated, dim=1)
    
    dist_related = 1 - score_related
    dist_unrelated = 1 - score_unrelated
    
    loss_related = alpha * dist_related.pow(2).mean()
    loss_unrelated = (1 - alpha) * F.relu(margin - dist_unrelated).pow(2).mean()
    
    return loss_related + loss_unrelated

# Training function
def train_epoch(model_one, model_two, dataloader, optimizer, alpha, margin):
    model_one.train()
    model_two.train()
    total_loss = 0
    for query, related, unrelated in tqdm(dataloader, desc="Training", leave=False):
        query, related, unrelated = query.to(device), related.to(device), unrelated.to(device)
        optimizer.zero_grad()
        query_transformed = model_one(query)
        related_transformed = model_two(related)
        unrelated_transformed = model_two(unrelated)
        loss = contrastive_loss(query_transformed, related_transformed, unrelated_transformed, alpha, margin)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation function
def validate_epoch(model_one, model_two, dataloader, alpha, margin):
    model_one.eval()
    model_two.eval()
    total_loss = 0
    with torch.no_grad():
        for query, related, unrelated in tqdm(dataloader, desc="Validation", leave=False):
            query, related, unrelated = query.to(device), related.to(device), unrelated.to(device)
            query_transformed = model_one(query)
            related_transformed = model_two(related)
            unrelated_transformed = model_two(unrelated)
            loss = contrastive_loss(query_transformed, related_transformed, unrelated_transformed, alpha, margin)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop
best_val_loss = float('inf')
for epoch in range(wandb.config.epochs):
    train_loss = train_epoch(tower_one, tower_two, train_loader, optimizer, wandb.config.alpha, wandb.config.margin)
    val_loss = validate_epoch(tower_one, tower_two, val_loader, wandb.config.alpha, wandb.config.margin)
    
    # Logging
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Validation Loss": val_loss,
        "Learning Rate": scheduler.get_last_lr()[0]
    })
    
    # Learning rate scheduling
    scheduler.step()
    
    # Early stopping and saving best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(tower_one.state_dict(), 'best_tower_one.pth')
        torch.save(tower_two.state_dict(), 'best_tower_two.pth')
        print(f"New best model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
    else:
        print(f"No improvement in validation loss at epoch {epoch + 1}")

# Testing phase
tower_one.load_state_dict(torch.load('best_tower_one.pth'))
tower_two.load_state_dict(torch.load('best_tower_two.pth'))
test_loss = validate_epoch(tower_one, tower_two, test_loader, wandb.config.alpha, wandb.config.margin)
wandb.log({"Test Loss": test_loss})
print(f"Test Loss: {test_loss:.4f}")

wandb.finish()

'''




















'''chatgpt best solution
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# Initialize Weights & Biases
wandb.init(project="two_tower_model", config={
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32,
    "alpha": 0.7
})

# Load data
train_triples = np.load('train1_averaged_triples.npy', allow_pickle=True)
validation_triples = np.load('validation1_averaged_triples.npy', allow_pickle=True)
test_triples = np.load('test1_averaged_triples.npy', allow_pickle=True)

# Convert numpy arrays to PyTorch tensors
def prepare_data(triples):
    return [tuple(torch.tensor(x, dtype=torch.float32) for x in triple) for triple in triples]

train_triples = prepare_data(train_triples)
validation_triples = prepare_data(validation_triples)
test_triples = prepare_data(test_triples)

# Create DataLoader for batch training
BATCH_SIZE = wandb.config.batch_size

def create_dataloader(triples):
    query, related_doc, unrelated_doc = zip(*triples)
    dataset = TensorDataset(torch.stack(query), torch.stack(related_doc), torch.stack(unrelated_doc))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

train_loader = create_dataloader(train_triples)
val_loader = create_dataloader(validation_triples)
test_loader = create_dataloader(test_triples)

# Define the separate towers
class TowerOne(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300):
        super(TowerOne, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TowerTwo(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300):
        super(TowerTwo, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize models, optimizer, and hyperparameters
tower_one = TowerOne()
tower_two = TowerTwo()

LEARNING_RATE = wandb.config.learning_rate
ALPHA = wandb.config.alpha
EPOCHS = wandb.config.epochs
MARGIN = 0.5

optimizer = torch.optim.Adam(list(tower_one.parameters()) + list(tower_two.parameters()), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    tower_one.train()
    tower_two.train()
    train_loss = 0

    # Train in batches
    for query, related_doc, unrelated_doc in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training", leave=False):
        optimizer.zero_grad()
        
        # Forward pass through both towers
        query_transformed = tower_one(query)
        related_transformed = tower_two(related_doc)
        unrelated_transformed = tower_two(unrelated_doc)

        # Compute cosine similarity
        score_related = F.cosine_similarity(query_transformed, related_transformed, dim=1)
        score_unrelated = F.cosine_similarity(query_transformed, unrelated_transformed, dim=1)

        # Convert similarity to distance
        dist_related = 1 - score_related
        dist_unrelated = 1 - score_unrelated

        # Compute contrastive loss with alpha
        loss_related = ALPHA * dist_related.pow(2).mean()
        loss_unrelated = (1 - ALPHA) * F.relu(MARGIN - dist_unrelated).pow(2).mean()
        loss = loss_related + loss_unrelated

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    tower_one.eval()
    tower_two.eval()
    val_loss = 0
    with torch.no_grad():
        for query, related_doc, unrelated_doc in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Validation", leave=False):
            query_transformed = tower_one(query)
            related_transformed = tower_two(related_doc)
            unrelated_transformed = tower_two(unrelated_doc)

            score_related = F.cosine_similarity(query_transformed, related_transformed, dim=1)
            score_unrelated = F.cosine_similarity(query_transformed, unrelated_transformed, dim=1)

            dist_related = 1 - score_related
            dist_unrelated = 1 - score_unrelated

            loss_related = ALPHA * dist_related.pow(2).mean()
            loss_unrelated = (1 - ALPHA) * F.relu(MARGIN - dist_unrelated).pow(2).mean()
            loss = loss_related + loss_unrelated

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # Logging to wandb
    wandb.log({"Epoch": epoch+1, "Train Loss": avg_train_loss, "Validation Loss": avg_val_loss, "Alpha": ALPHA})

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")

# Save model
torch.save(tower_one.state_dict(), 'tower_one.pth')
torch.save(tower_two.state_dict(), 'tower_two.pth')

wandb.finish()

'''
























'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader, TensorDataset

# Initialize Weights & Biases
wandb.init(project="two-tower-training")

# Hyperparameters


learning_rate = 0.001
num_epochs = 10
margin = 0.5






# Load the triples
def load_triples(file_path):
    triples = np.load(file_path, allow_pickle=True)
    return [tuple(torch.tensor(x, dtype=torch.float32) for x in triple) for triple in triples]

train_triples = load_triples('train1_averaged_triples.npy')
validation_triples = load_triples('validation1_averaged_triples.npy')
test_triples = load_triples('test1_averaged_triples.npy')

class TowerOne(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300):
        super(TowerOne, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TowerTwo(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300):
        super(TowerTwo, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Initialize towers
tower_one = TowerOne()
tower_two = TowerTwo()

# Define optimizer
optimizer = torch.optim.Adam(list(tower_one.parameters()) + list(tower_two.parameters()), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_train_loss = 0.0
    progress_bar = tqdm(train_triples, desc=f"Training Epoch {epoch+1}", leave=False)
    for query, related_doc, unrelated_doc in progress_bar:
        optimizer.zero_grad()
        
        # Forward pass
        query_transformed = tower_one(query)
        related_transformed = tower_two(related_doc)
        unrelated_transformed = tower_two(unrelated_doc)
        
        # Compute cosine similarity scores
        score_related = F.cosine_similarity(query_transformed, related_transformed, dim=0)
        score_unrelated = F.cosine_similarity(query_transformed, unrelated_transformed, dim=0)
        
        # Convert similarity to distance
        dist_related = 1 - score_related
        dist_unrelated = 1 - score_unrelated
        
        # Contrastive loss calculation
        loss_related = F.relu(margin - dist_related).pow(2)
        loss_unrelated = dist_unrelated.pow(2)
        loss = (loss_related + loss_unrelated).mean()
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
        # Log to Weights & Biases
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": loss.item(),
            "Related Distance": dist_related.item(),
            "Unrelated Distance": dist_unrelated.item()
        })
    
    avg_train_loss = total_train_loss / len(train_triples)
    
    # Validation phase
    total_val_loss = 0.0
    with torch.no_grad():
        for query, related_doc, unrelated_doc in validation_triples:
            query_transformed = tower_one(query)
            related_transformed = tower_two(related_doc)
            unrelated_transformed = tower_two(unrelated_doc)
            
            score_related = F.cosine_similarity(query_transformed, related_transformed, dim=0)
            score_unrelated = F.cosine_similarity(query_transformed, unrelated_transformed, dim=0)
            
            dist_related = 1 - score_related
            dist_unrelated = 1 - score_unrelated
            
            loss_related = F.relu(margin - dist_related).pow(2)
            loss_unrelated = dist_unrelated.pow(2)
            loss = (loss_related + loss_unrelated).mean()
            
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(validation_triples)
    
    print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
    wandb.log({"Epoch Loss": avg_train_loss, "Validation Loss": avg_val_loss})

# Save models
torch.save(tower_one.state_dict(), 'tower_one.pth')
torch.save(tower_two.state_dict(), 'tower_two.pth')
'''




''' - NOT CONNECTED TO WANDB AND FOR DATA WITH 0 AND 1 RELATIVE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Load the triples generated from the first script
train_triples = np.load('train_averaged_triples.npy', allow_pickle=True)
validation_triples = np.load('validation_averaged_triples.npy', allow_pickle=True)
test_triples = np.load('test_averaged_triples.npy', allow_pickle=True)


# Convert numpy arrays to PyTorch tensors - Float Tensors
train_triples = [tuple(torch.tensor(x, dtype=torch.float32) for x in triple) for triple in train_triples]
validation_triples = [tuple(torch.tensor(x, dtype=torch.float32) for x in triple) for triple in validation_triples]
test_triples = [tuple(torch.tensor(x, dtype=torch.float32) for x in triple) for triple in test_triples]

class TowerOne(nn.Module):
    def __init__(self):
        super(TowerOne, self).__init__()
        self.fc1 = nn.Linear(300, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TowerTwo(nn.Module):
    def __init__(self):
        super(TowerTwo, self).__init__()
        self.fc1 = nn.Linear(300, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize towers
tower_one = TowerOne()
tower_two = TowerTwo()

# Define optimizer
optimizer = torch.optim.Adam(list(tower_one.parameters()) + list(tower_two.parameters()), lr=0.001)

# Training loop
for epoch in range(10):  # Number of epochs
    for query, related_doc, unrelated_doc in train_triples:
        optimizer.zero_grad()
        
        # Pass query through TowerOne
        query_transformed = tower_one(query)
        
        # Pass related and unrelated documents through TowerTwo
        related_transformed = tower_two(related_doc)
        unrelated_transformed = tower_two(unrelated_doc)
        
        # Compute cosine similarity scores
        score_related = F.cosine_similarity(query_transformed, related_transformed, dim=0)
        score_unrelated = F.cosine_similarity(query_transformed, unrelated_transformed, dim=0)
        
        # Convert similarity to distance (for contrastive loss)
        dist_related = 1 - score_related
        dist_unrelated = 1 - score_unrelated
        
        # Define target labels: 1 for similar, 0 for dissimilar
        target = torch.tensor([1.0, 0.0], dtype=torch.float32)
        
        # Define margin
        margin = 0.5
        
        # Compute contrastive loss
        loss_related = (1 - target[0]) * dist_related.pow(2) + target[0] * F.relu(margin - dist_related).pow(2)
        loss_unrelated = (1 - target[1]) * dist_unrelated.pow(2) + target[1] * F.relu(margin - dist_unrelated).pow(2)
        
        # Final loss (mean over batch)
        loss = (loss_related + loss_unrelated).mean()
        
        # Backpropagation
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the model if needed
torch.save(tower_one.state_dict(), 'tower_one.pth')
torch.save(tower_two.state_dict(), 'tower_two.pth')
'''























'''
Epoch 1, Loss: 0.1282159388065338
Epoch 2, Loss: 0.13346928358078003
Epoch 3, Loss: 0.13542117178440094
Epoch 4, Loss: 0.14303472638130188
'''

 


'''
class TowerOne(nn.Module):
    def __init__(self):
        super(TowerOne, self).__init__()
        self.fc = nn.Linear(300, 300)
        self.fc = nn.Linear(300, 300)
        self.fc = nn.Linear(300, 300)
    
    def forward(self, x):
        return self.fc(x)

class TowerTwo(nn.Module):
    def __init__(self):
        super(TowerTwo, self).__init__()
        self.fc = nn.Linear(300, 300)
        self.fc = nn.Linear(300, 300)
        self.fc = nn.Linear(300, 300)
    
    def forward(self, x):
        return self.fc(x)
  

#EMBEDDINGS 

# Simulated embeddings (assumed shape)
query_embedding = torch.rand(1, 5)   # Query embedding
related_doc_embedding = torch.rand(1, 3)  # Related document
unrelated_doc_embedding = torch.rand(1, 3)  # Unrelated document


# Initialize towers
tower_one = TowerOne()
tower_two = TowerTwo()

# Pass query through TowerOne
query_transformed = tower_one(query_embedding)

# Pass related and unrelated documents through TowerTwo
related_transformed = tower_two(related_doc_embedding)
unrelated_transformed = tower_two(unrelated_doc_embedding)

# Compute cosine similarity scores
score_related = F.cosine_similarity(query_transformed, related_transformed, dim=1)
score_unrelated = F.cosine_similarity(query_transformed, unrelated_transformed, dim=1)

# Convert similarity to distance (for contrastive loss)
dist_related = 1 - score_related
dist_unrelated = 1 - score_unrelated

# Define target labels: 1 for similar, 0 for dissimilar
target = torch.tensor([1.0, 0.0])

# Define margin
margin = 0.5

# Compute contrastive loss
loss_related = (1 - target[0]) * dist_related.pow(2) + target[0] * F.relu(margin - dist_related).pow(2)
loss_unrelated = (1 - target[1]) * dist_unrelated.pow(2) + target[1] * F.relu(margin - dist_unrelated).pow(2)

# Final loss (mean over batch)
loss = (loss_related + loss_unrelated).mean()

# Backpropagation
loss.backward()

'''

















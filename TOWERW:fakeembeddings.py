#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:03:15 2025

@author: toluojo
"""

# TOWER WITH NEW GENERATED DATA 




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import wandb

# Initialize Weights & Biases
wandb.init(project="two_tower_model", config={
    "learning_rate": 0.003,
    "epochs": 100,
    "batch_size": 32,
    "alpha": 0.7,
    "hidden_dim": 300,
    "dropout_rate": 0.2,
    "margin": 1.0,
    "step_size": 5,
    "gamma": 0.1
})

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate random embeddings
def generate_embeddings(num_samples, embedding_dim=300):
    return np.random.rand(num_samples, embedding_dim).astype(np.float32)

# Generate triples (query, related, unrelated)
def generate_triples(num_samples):
    triples = []
    for _ in range(num_samples):
        query = generate_embeddings(1)  # Query embedding
        related = generate_embeddings(1)  # Related document embedding
        unrelated = generate_embeddings(1)  # Unrelated document embedding
        triples.append((query, related, unrelated))
    return triples

# Generate train, validation, and test triples
num_train_samples = 1000
num_val_samples = 200
num_test_samples = 200

train_triples = generate_triples(num_train_samples)
validation_triples = generate_triples(num_val_samples)
test_triples = generate_triples(num_test_samples)

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

# Define the RNN-based Tower model
class Tower(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300, num_layers=1):
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

# Instantiate the model
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

































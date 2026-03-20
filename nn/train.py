#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:09:51 2026

@author: righley
"""
"""
Training Script for Chess Evaluation Neural Network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time

from model import ChessEvaluationNet
from dataset import ChessDataset


def train_model(
    data_path="../data/positions.npz",
    save_path="weights.pth",
    epochs=20,
    batch_size=512,
    learning_rate=0.001,
    val_split=0.1,
    device=None
):
    """
    Train the chess evaluation neural network.
    
    Args:
        data_path: Path to the .npz file with positions and labels
        save_path: Where to save the trained model weights
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        val_split: Fraction of data to use for validation
        device: 'cuda' or 'cpu' (auto-detected if None)
    """
    
    # Detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = ChessDataset(data_path)
    
    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples:   {train_size:,}")
    print(f"Validation samples: {val_size:,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )
    
    # Create model
    print("\nCreating model...")
    model = ChessEvaluationNet()
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduce LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_positions, batch_labels in train_loader:
            # Move to device
            batch_positions = batch_positions.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_positions)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_positions, batch_labels in val_loader:
                batch_positions = batch_positions.to(device)
                batch_labels = batch_labels.to(device).unsqueeze(1)
                
                outputs = model(batch_positions)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            saved_marker = " (saved)"
        else:
            saved_marker = ""
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s{saved_marker}")
    
    print("="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    
    return model, history


def test_model(model_path="weights.pth", data_path="../data/positions.npz", device=None):
    """
    Test the trained model on some sample positions.
    """
    import chess
    from dataset import board_to_tensor
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = ChessEvaluationNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("\n" + "="*60)
    print("Testing trained model on sample positions")
    print("="*60)
    
    # Test position 1: Starting position (should be ~0, equal)
    board = chess.Board()
    tensor = torch.tensor(board_to_tensor(board)).to(device)
    score = model.evaluate(tensor)
    print(f"\n1. Starting position")
    print(f"   Expected: ~0.0 (equal)")
    print(f"   Model:    {score:+.3f}")
    
    # Test position 2: White up a queen
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    tensor = torch.tensor(board_to_tensor(board)).to(device)
    score = model.evaluate(tensor)
    print(f"\n2. White up a queen")
    print(f"   Expected: positive (White winning)")
    print(f"   Model:    {score:+.3f}")
    
    # Test position 3: Black up a queen
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    tensor = torch.tensor(board_to_tensor(board)).to(device)
    score = model.evaluate(tensor)
    print(f"\n3. Black up a queen")
    print(f"   Expected: negative (Black winning)")
    print(f"   Model:    {score:+.3f}")
    
    # Test position 4: White up a rook
    board = chess.Board("rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1")
    tensor = torch.tensor(board_to_tensor(board)).to(device)
    score = model.evaluate(tensor)
    print(f"\n4. White up a rook")
    print(f"   Expected: positive (White better)")
    print(f"   Model:    {score:+.3f}")


if __name__ == "__main__":
    # Train the model
    model, history = train_model(
        data_path="../data/positions.npz",
        save_path="weights.pth",
        epochs=20,
        batch_size=512,
        learning_rate=0.001
    )
    
    # Test on sample positions
    test_model()
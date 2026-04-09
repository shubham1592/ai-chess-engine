#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 01:03:03 2026

@author: righley
"""

"""
Training Script for Chess Evaluation Neural Network.
Uses Stockfish-labeled data for better accuracy.
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
    data_path="../data/positions_stockfish.npz",
    save_path="weights_stockfish.pth",
    epochs=30,
    batch_size=512,
    learning_rate=0.001,
    val_split=0.1,
    device=None
):
    """
    Train the chess evaluation neural network with Stockfish labels.
    """
    
    # Detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
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
        num_workers=4,
        pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=(device == 'cuda')
    )
    
    # Create model
    print("\nCreating model...")
    model = ChessEvaluationNet()
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
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
    print("Starting training with Stockfish labels...")
    print("="*60)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_positions, batch_labels in train_loader:
            batch_positions = batch_positions.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_positions)
            loss = criterion(outputs, batch_labels)
            
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
            saved_marker = " ✓ saved"
        else:
            saved_marker = ""
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s{saved_marker}")
    
    print("="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    
    return model, history


def test_model(model_path="weights_stockfish.pth", device=None):
    """
    Test the trained model on sample positions.
    """
    import chess
    from dataset import board_to_tensor
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = ChessEvaluationNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model = model.to(device)
    model.eval()
    
    print("\n" + "="*60)
    print("Testing trained model on sample positions")
    print("="*60)
    
    test_positions = [
        ("Starting position", 
         chess.Board(), 
         "~0.0 (equal)"),
        
        ("White up a Queen", 
         chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
         "highly positive (White winning)"),
        
        ("Black up a Queen",
         chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"),
         "highly negative (Black winning)"),
        
        ("White up a Rook",
         chess.Board("rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1"),
         "positive (White better)"),
        
        ("Black up a Rook",
         chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1"),
         "negative (Black better)"),
        
        ("White up a Knight",
         chess.Board("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
         "positive (White better)"),
        
        ("Sicilian Defense position",
         chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
         "~0.0 (equal opening)"),
    ]
    
    print("\n{:<25} {:>10} {:>25}".format("Position", "Model", "Expected"))
    print("-"*60)
    
    for name, board, expected in test_positions:
        tensor = torch.tensor(board_to_tensor(board)).to(device)
        score = model.evaluate(tensor)
        print(f"{name:<25} {score:>+10.3f} {expected:>25}")
    
    print("\n" + "="*60)
    print("Score interpretation:")
    print("  +1.0 = White is winning (checkmate)")
    print("  +0.5 = White has strong advantage")
    print("   0.0 = Equal position")
    print("  -0.5 = Black has strong advantage")
    print("  -1.0 = Black is winning (checkmate)")
    print("="*60)


if __name__ == "__main__":
    # Train the model
    model, history = train_model(
        data_path="../data/positions_stockfish.npz",
        save_path="weights_stockfish.pth",
        epochs=30,
        batch_size=512,
        learning_rate=0.001
    )
    
    # Test on sample positions
    test_model()
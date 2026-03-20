#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:05:40 2026

@author: righley
"""

"""
Neural Network Model for Chess Position Evaluation.
Simple MLP architecture that takes board tensor and outputs evaluation score.
"""

import torch
import torch.nn as nn


class ChessEvaluationNet(nn.Module):
    """
    Multi-Layer Perceptron for chess position evaluation.
    
    Architecture:
        Input:  773 neurons (board representation)
        Hidden: 512 -> 256 -> 128 neurons with ReLU
        Output: 1 neuron with Tanh (score between -1 and +1)
    """
    
    def __init__(self, input_size=773, hidden_sizes=[512, 256, 128], dropout=0.2):
        """
        Args:
            input_size: Size of input tensor (773 for our encoding)
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability for regularization
        """
        super(ChessEvaluationNet, self).__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Tanh())  # Output between -1 and +1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, 773)
        
        Returns:
            Tensor of shape (batch_size, 1) with values in [-1, +1]
        """
        return self.network(x)
    
    def evaluate(self, x):
        """
        Evaluate a single position (no batch dimension).
        
        Args:
            x: Tensor of shape (773,)
        
        Returns:
            Float evaluation score in [-1, +1]
        """
        self.eval()
        with torch.no_grad():
            x = x.unsqueeze(0)  # Add batch dimension
            output = self.forward(x)
            return output.item()


def count_parameters(model):
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    print("="*50)
    print("Chess Evaluation Neural Network")
    print("="*50)
    
    # Create model
    model = ChessEvaluationNet()
    
    # Print architecture
    print("\nArchitecture:")
    print(model)
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {num_params:,}")
    
    # Test forward pass with random input
    print("\nTesting forward pass...")
    batch_size = 32
    dummy_input = torch.randn(batch_size, 773)
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test single position evaluation
    print("\nTesting single position evaluation...")
    single_input = torch.randn(773)
    score = model.evaluate(single_input)
    print(f"Single position score: {score:.4f}")
    
    print("\nModel test passed!")
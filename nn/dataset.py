"""
Neural Network Model for Chess Position Evaluation

This defines a simple feedforward network (MLP) that takes a chess position
as input and outputs an evaluation score. The architecture is:

    Input (773) → Hidden (512) → Hidden (256) → Hidden (128) → Output (1)

Each hidden layer uses BatchNorm, ReLU activation, and Dropout for regularization.
The output uses Tanh to squash the score to [-1, +1] range.
"""

import torch
import torch.nn as nn


class ChessEvaluationNet(nn.Module):
    """
    Multi-Layer Perceptron for evaluating chess positions.
    
    Takes a 773-dimensional binary vector representing the board state
    and outputs a single score between -1 (Black winning) and +1 (White winning).
    
    Why this architecture:
    - MLP is simpler and faster than CNN for this task
    - 3 hidden layers is enough to learn complex patterns
    - Decreasing layer sizes (512→256→128) compress information gradually
    - Dropout prevents overfitting to training positions
    - BatchNorm makes training more stable
    """
    
    def __init__(self, input_size=773, hidden_sizes=[512, 256, 128], dropout=0.2):
        """
        Build the network layers.
        
        Args:
            input_size: 773 for our board encoding (12*64 pieces + 5 extra features)
            hidden_sizes: Neurons in each hidden layer
            dropout: Probability of dropping neurons during training
        """
        super(ChessEvaluationNet, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Final output layer: single neuron with tanh activation
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Run a batch of positions through the network.
        
        Args:
            x: Tensor of shape (batch_size, 773)
        
        Returns:
            Tensor of shape (batch_size, 1) with scores in [-1, +1]
        """
        return self.network(x)
    
    def evaluate(self, x):
        """
        Evaluate a single position (convenience method for inference).
        
        Args:
            x: Tensor of shape (773,) representing one position
        
        Returns:
            Float score between -1 and +1
        """
        self.eval()
        with torch.no_grad():
            x = x.unsqueeze(0)  # Add batch dimension: (773,) → (1, 773)
            output = self.forward(x)
            return output.item()


def count_parameters(model):
    """Count trainable parameters in the model."""
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
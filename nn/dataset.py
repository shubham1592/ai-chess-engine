"""
Dataset utilities for chess neural network.

This module handles converting chess positions into tensors that the neural
network can process. Each position becomes a 773-dimensional binary vector.

Board Encoding (773 features):
    - 768 features: 12 piece types × 64 squares (one-hot encoding)
    - 1 feature: whose turn it is (1=White, 0=Black)
    - 4 features: castling rights (white kingside, white queenside, 
                                   black kingside, black queenside)
"""

import chess
import numpy as np
import torch
from torch.utils.data import Dataset


def board_to_tensor(board):
    """
    Convert a chess position into a 773-dimensional tensor.
    
    This encoding allows the neural network to see:
    - Where each piece is located
    - Whose turn it is
    - What castling options remain
    
    Args:
        board: A chess.Board object
    
    Returns:
        numpy array of shape (773,) with float32 values (0.0 or 1.0)
    """
    tensor = np.zeros(773, dtype=np.float32)
    
    # Encode piece positions
    # We go through each piece type and color, marking which squares have that piece
    piece_order = [
        (chess.PAWN, chess.WHITE),    # Indices 0-63
        (chess.KNIGHT, chess.WHITE),  # Indices 64-127
        (chess.BISHOP, chess.WHITE),  # Indices 128-191
        (chess.ROOK, chess.WHITE),    # Indices 192-255
        (chess.QUEEN, chess.WHITE),   # Indices 256-319
        (chess.KING, chess.WHITE),    # Indices 320-383
        (chess.PAWN, chess.BLACK),    # Indices 384-447
        (chess.KNIGHT, chess.BLACK),  # Indices 448-511
        (chess.BISHOP, chess.BLACK),  # Indices 512-575
        (chess.ROOK, chess.BLACK),    # Indices 576-639
        (chess.QUEEN, chess.BLACK),   # Indices 640-703
        (chess.KING, chess.BLACK),    # Indices 704-767
    ]
    
    for i, (piece_type, color) in enumerate(piece_order):
        for square in board.pieces(piece_type, color):
            tensor[i * 64 + square] = 1.0
    
    # Encode side to move
    tensor[768] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Encode castling rights
    tensor[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[772] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    return tensor


def result_to_label(result, board_turn_was_white):
    """
    Convert a game result string to a numerical label.
    
    Labels are always from White's perspective:
        +1 = White wins
         0 = Draw
        -1 = Black wins
    
    Args:
        result: Game result string ("1-0", "0-1", or "1/2-1/2")
        board_turn_was_white: Unused, kept for compatibility
    
    Returns:
        Float label: +1.0, 0.0, or -1.0
    """
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    else:
        return 0.0


class ChessDataset(Dataset):
    """
    PyTorch Dataset for loading chess positions and their evaluations.
    
    Expects a .npz file with:
        - 'positions': numpy array of shape (N, 773)
        - 'labels': numpy array of shape (N,)
    """
    
    def __init__(self, positions_file):
        """
        Load the dataset from a numpy file.
        
        Args:
            positions_file: Path to .npz file with positions and labels
        """
        data = np.load(positions_file)
        self.positions = torch.tensor(data['positions'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.float32)
        
        print(f"Loaded {len(self.positions)} positions from {positions_file}")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.positions[idx], self.labels[idx]


# Quick test
if __name__ == "__main__":
    # Test board_to_tensor with starting position
    board = chess.Board()
    tensor = board_to_tensor(board)
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Non-zero elements: {np.count_nonzero(tensor)}")
    print(f"Side to move (White=1): {tensor[768]}")
    print(f"Castling rights: {tensor[769:773]}")
    
    # Verify some pieces
    print(f"\nWhite pawns (indices 8-15 should be 1):")
    print(f"  {tensor[8:16]}")
    
    print(f"\nWhite king on e1 (index 320+4=324 should be 1):")
    print(f"  Index 324 = {tensor[324]}")
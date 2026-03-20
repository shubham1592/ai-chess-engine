#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:05:40 2026

@author: righley
"""

"""
Dataset utilities for chess neural network.
Handles board encoding and data loading.
"""

import chess
import numpy as np
import torch
from torch.utils.data import Dataset


def board_to_tensor(board):
    """
    Convert a chess.Board to a 773-dimensional tensor.
    
    Encoding:
        - Indices 0-63:    White Pawns
        - Indices 64-127:  White Knights
        - Indices 128-191: White Bishops
        - Indices 192-255: White Rooks
        - Indices 256-319: White Queens
        - Indices 320-383: White King
        - Indices 384-447: Black Pawns
        - Indices 448-511: Black Knights
        - Indices 512-575: Black Bishops
        - Indices 576-639: Black Rooks
        - Indices 640-703: Black Queens
        - Indices 704-767: Black King
        - Index 768:       Side to move (1=White, 0=Black)
        - Indices 769-772: Castling rights (WK, WQ, BK, BQ)
    
    Args:
        board: chess.Board object
    
    Returns:
        numpy array of shape (773,) with float32 values
    """
    tensor = np.zeros(773, dtype=np.float32)
    
    # Piece placement (12 piece types x 64 squares = 768 values)
    piece_order = [
        (chess.PAWN, chess.WHITE),    # 0-63
        (chess.KNIGHT, chess.WHITE),  # 64-127
        (chess.BISHOP, chess.WHITE),  # 128-191
        (chess.ROOK, chess.WHITE),    # 192-255
        (chess.QUEEN, chess.WHITE),   # 256-319
        (chess.KING, chess.WHITE),    # 320-383
        (chess.PAWN, chess.BLACK),    # 384-447
        (chess.KNIGHT, chess.BLACK),  # 448-511
        (chess.BISHOP, chess.BLACK),  # 512-575
        (chess.ROOK, chess.BLACK),    # 576-639
        (chess.QUEEN, chess.BLACK),   # 640-703
        (chess.KING, chess.BLACK),    # 704-767
    ]
    
    for i, (piece_type, color) in enumerate(piece_order):
        for square in board.pieces(piece_type, color):
            tensor[i * 64 + square] = 1.0
    
    # Side to move
    tensor[768] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Castling rights
    tensor[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[772] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    return tensor


def result_to_label(result, board_turn_was_white):
    """
    Convert game result string to a label.
    
    Labels are from White's perspective:
        White wins: +1
        Draw:        0
        Black wins: -1
    
    Args:
        result: String like "1-0", "0-1", or "1/2-1/2"
        board_turn_was_white: Whose turn it was (not used here, but kept for flexibility)
    
    Returns:
        float: +1, 0, or -1
    """
    if result == "1-0":
        return 1.0   # White wins
    elif result == "0-1":
        return -1.0  # Black wins
    else:
        return 0.0   # Draw


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess positions.
    
    Loads pre-processed numpy arrays of positions and labels.
    """
    
    def __init__(self, positions_file):
        """
        Args:
            positions_file: Path to .npz file containing 'positions' and 'labels'
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
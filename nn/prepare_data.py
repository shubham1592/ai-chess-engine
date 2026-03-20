#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:05:40 2026

@author: righley
"""

"""
PGN Parser and Data Preparation Script.
Extracts positions from chess games and saves them for neural network training.
"""

import chess
import chess.pgn
import numpy as np
import os
import random
from tqdm import tqdm

# Import from our dataset module
from dataset import board_to_tensor, result_to_label


def parse_pgn_file(pgn_path, max_games=None, positions_per_game=8, skip_first_n_moves=5):
    """
    Parse a PGN file and extract positions with their game results.
    
    Args:
        pgn_path: Path to the PGN file
        max_games: Maximum number of games to process (None = all)
        positions_per_game: How many random positions to sample from each game
        skip_first_n_moves: Skip opening moves (first N moves by each side)
    
    Returns:
        positions: numpy array of shape (N, 773)
        labels: numpy array of shape (N,)
    """
    positions_list = []
    labels_list = []
    
    games_processed = 0
    games_skipped = 0
    
    print(f"Opening {pgn_path}...")
    file_size = os.path.getsize(pgn_path)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
        
        # Create progress bar
        pbar = tqdm(total=max_games if max_games else None, desc="Processing games")
        
        while True:
            # Check if we've reached max games
            if max_games and games_processed >= max_games:
                break
            
            # Read next game
            try:
                game = chess.pgn.read_game(pgn_file)
            except Exception as e:
                print(f"Error reading game: {e}")
                continue
            
            if game is None:
                break  # End of file
            
            # Get game result
            result = game.headers.get("Result", "*")
            
            # Skip games without clear result
            if result not in ["1-0", "0-1", "1/2-1/2"]:
                games_skipped += 1
                continue
            
            # Get label for this game
            label = result_to_label(result, True)
            
            # Play through the game and collect positions
            board = game.board()
            game_positions = []
            move_count = 0
            
            for move in game.mainline_moves():
                board.push(move)
                move_count += 1
                
                # Skip early opening moves
                if move_count <= skip_first_n_moves * 2:
                    continue
                
                # Skip positions where game is over
                if board.is_game_over():
                    continue
                
                # Store this position
                game_positions.append(board.copy())
            
            # Sample random positions from this game
            if len(game_positions) >= positions_per_game:
                sampled = random.sample(game_positions, positions_per_game)
            else:
                sampled = game_positions  # Take all if fewer available
            
            # Convert positions to tensors
            for pos in sampled:
                tensor = board_to_tensor(pos)
                positions_list.append(tensor)
                labels_list.append(label)
            
            games_processed += 1
            pbar.update(1)
            
            # Print progress every 5000 games
            if games_processed % 5000 == 0:
                pbar.set_postfix({
                    'positions': len(positions_list),
                    'skipped': games_skipped
                })
        
        pbar.close()
    
    print(f"\nDone!")
    print(f"Games processed: {games_processed}")
    print(f"Games skipped (no result): {games_skipped}")
    print(f"Total positions extracted: {len(positions_list)}")
    
    # Convert to numpy arrays
    positions = np.array(positions_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    
    return positions, labels


def save_dataset(positions, labels, output_path):
    """Save positions and labels to a compressed numpy file."""
    np.savez_compressed(output_path, positions=positions, labels=labels)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to {output_path} ({file_size:.1f} MB)")


def main():
    # Configuration
    PGN_FILE = "../data/games.pgn"       # Path to your PGN file
    OUTPUT_FILE = "../data/positions.npz" # Where to save processed data
    MAX_GAMES = 50000                     # Process 50k games (adjust as needed)
    POSITIONS_PER_GAME = 8                # Sample 8 positions per game
    SKIP_OPENING_MOVES = 5                # Skip first 5 moves each side
    
    # Check if PGN file exists
    if not os.path.exists(PGN_FILE):
        print(f"ERROR: PGN file not found at {PGN_FILE}")
        print("Make sure you have downloaded and extracted the Lichess Elite Database")
        print("and placed the .pgn file in the data/ folder.")
        return
    
    print("="*60)
    print("Chess Neural Network - Data Preparation")
    print("="*60)
    print(f"PGN file: {PGN_FILE}")
    print(f"Max games: {MAX_GAMES}")
    print(f"Positions per game: {POSITIONS_PER_GAME}")
    print(f"Expected total positions: ~{MAX_GAMES * POSITIONS_PER_GAME:,}")
    print("="*60)
    
    # Parse PGN and extract positions
    positions, labels = parse_pgn_file(
        PGN_FILE,
        max_games=MAX_GAMES,
        positions_per_game=POSITIONS_PER_GAME,
        skip_first_n_moves=SKIP_OPENING_MOVES
    )
    
    # Print label distribution
    white_wins = np.sum(labels == 1.0)
    black_wins = np.sum(labels == -1.0)
    draws = np.sum(labels == 0.0)
    total = len(labels)
    
    print(f"\nLabel distribution:")
    print(f"  White wins: {white_wins:,} ({100*white_wins/total:.1f}%)")
    print(f"  Draws:      {draws:,} ({100*draws/total:.1f}%)")
    print(f"  Black wins: {black_wins:,} ({100*black_wins/total:.1f}%)")
    
    # Shuffle the data
    print("\nShuffling data...")
    indices = np.random.permutation(len(positions))
    positions = positions[indices]
    labels = labels[indices]
    
    # Save to file
    save_dataset(positions, labels, OUTPUT_FILE)
    
    print("\nData preparation complete!")
    print(f"You can now use {OUTPUT_FILE} for training.")


if __name__ == "__main__":
    main()
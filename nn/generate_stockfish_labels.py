#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 00:35:13 2026

@author: righley
"""

"""
Generate Stockfish Labels for Neural Network Training.
Replaces game outcome labels with accurate position evaluations.
"""

import chess
import chess.engine
import numpy as np
import os
import time
from tqdm import tqdm

# Import board conversion
from dataset import board_to_tensor


# ============================================================
# CONFIGURATION - ADJUST THESE
# ============================================================

STOCKFISH_PATH = "/home/righley/stockfish/stockfish-ubuntu-x86-64-avx2"
PGN_FILE = "../data/games.pgn"
OUTPUT_FILE = "../data/positions_stockfish.npz"

# Stockfish settings
DEPTH = 8                    # Search depth (8 is good balance of speed/accuracy)
TIME_LIMIT = 0.05            # Max time per position in seconds (50ms)

# Data settings
MAX_GAMES = 30000            # Number of games to process
POSITIONS_PER_GAME = 10      # Positions to sample per game
SKIP_OPENING_MOVES = 6       # Skip first N moves (opening book territory)

# ============================================================


def stockfish_evaluate(engine, board, depth=DEPTH, time_limit=TIME_LIMIT):
    """
    Get Stockfish evaluation for a position.
    
    Returns:
        Score in range [-1, 1] (normalized from centipawns)
        +1 = White winning, -1 = Black winning
    """
    try:
        # Get evaluation
        info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))
        score = info["score"].white()
        
        # Handle checkmate scores
        if score.is_mate():
            mate_in = score.mate()
            if mate_in > 0:
                return 1.0   # White mates
            else:
                return -1.0  # Black mates
        
        # Convert centipawns to [-1, 1] range
        # Using tanh-like scaling: ±400cp maps to roughly ±0.8
        cp = score.score()
        normalized = np.tanh(cp / 400.0)
        
        return float(normalized)
    
    except Exception as e:
        print(f"Error evaluating position: {e}")
        return 0.0


def process_games_with_stockfish():
    """
    Process PGN file and generate Stockfish-labeled dataset.
    """
    import chess.pgn
    
    print("="*60)
    print("Stockfish Label Generator")
    print("="*60)
    print(f"Stockfish path: {STOCKFISH_PATH}")
    print(f"PGN file: {PGN_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Depth: {DEPTH}")
    print(f"Max games: {MAX_GAMES}")
    print(f"Positions per game: {POSITIONS_PER_GAME}")
    print(f"Expected positions: ~{MAX_GAMES * POSITIONS_PER_GAME:,}")
    print("="*60)
    
    # Check files exist
    if not os.path.exists(STOCKFISH_PATH):
        print(f"ERROR: Stockfish not found at {STOCKFISH_PATH}")
        return
    
    if not os.path.exists(PGN_FILE):
        print(f"ERROR: PGN file not found at {PGN_FILE}")
        return
    
    # Start Stockfish engine
    print("\nStarting Stockfish engine...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    # Configure engine for speed
    engine.configure({"Threads": 4, "Hash": 256})
    
    print("Stockfish ready!\n")
    
    positions_list = []
    labels_list = []
    
    games_processed = 0
    positions_evaluated = 0
    start_time = time.time()
    
    with open(PGN_FILE, 'r', encoding='utf-8', errors='ignore') as pgn_file:
        pbar = tqdm(total=MAX_GAMES, desc="Processing games")
        
        while games_processed < MAX_GAMES:
            # Read next game
            try:
                game = chess.pgn.read_game(pgn_file)
            except Exception as e:
                continue
            
            if game is None:
                break  # End of file
            
            # Skip games without clear result
            result = game.headers.get("Result", "*")
            if result not in ["1-0", "0-1", "1/2-1/2"]:
                continue
            
            # Play through game and collect positions
            board = game.board()
            game_positions = []
            move_count = 0
            
            for move in game.mainline_moves():
                board.push(move)
                move_count += 1
                
                # Skip opening moves
                if move_count <= SKIP_OPENING_MOVES * 2:
                    continue
                
                # Skip if game is over
                if board.is_game_over():
                    continue
                
                game_positions.append(board.copy())
            
            # Sample positions from this game
            if len(game_positions) >= POSITIONS_PER_GAME:
                import random
                sampled = random.sample(game_positions, POSITIONS_PER_GAME)
            else:
                sampled = game_positions
            
            # Evaluate each position with Stockfish
            for pos in sampled:
                tensor = board_to_tensor(pos)
                label = stockfish_evaluate(engine, pos)
                
                positions_list.append(tensor)
                labels_list.append(label)
                positions_evaluated += 1
            
            games_processed += 1
            pbar.update(1)
            
            # Progress update every 1000 games
            if games_processed % 1000 == 0:
                elapsed = time.time() - start_time
                rate = positions_evaluated / elapsed
                pbar.set_postfix({
                    'positions': positions_evaluated,
                    'rate': f'{rate:.1f}/s'
                })
        
        pbar.close()
    
    # Close Stockfish
    engine.quit()
    
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Games processed: {games_processed:,}")
    print(f"Positions evaluated: {positions_evaluated:,}")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    print(f"Average rate: {positions_evaluated/elapsed:.1f} positions/second")
    
    # Convert to numpy arrays
    positions = np.array(positions_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    
    # Print label distribution
    print(f"\nLabel distribution:")
    print(f"  Min: {labels.min():.3f}")
    print(f"  Max: {labels.max():.3f}")
    print(f"  Mean: {labels.mean():.3f}")
    print(f"  Std: {labels.std():.3f}")
    
    # Count rough categories
    white_winning = np.sum(labels > 0.3)
    black_winning = np.sum(labels < -0.3)
    equal = np.sum((labels >= -0.3) & (labels <= 0.3))
    
    print(f"\nPosition breakdown:")
    print(f"  White winning (>0.3):  {white_winning:,} ({100*white_winning/len(labels):.1f}%)")
    print(f"  Equal (-0.3 to 0.3):   {equal:,} ({100*equal/len(labels):.1f}%)")
    print(f"  Black winning (<-0.3): {black_winning:,} ({100*black_winning/len(labels):.1f}%)")
    
    # Shuffle
    print("\nShuffling data...")
    indices = np.random.permutation(len(positions))
    positions = positions[indices]
    labels = labels[indices]
    
    # Save
    np.savez_compressed(OUTPUT_FILE, positions=positions, labels=labels)
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nSaved to {OUTPUT_FILE} ({file_size:.1f} MB)")
    
    print("\n" + "="*60)
    print("Done! You can now retrain the neural network with:")
    print("  python train.py")
    print("="*60)


if __name__ == "__main__":
    process_games_with_stockfish()
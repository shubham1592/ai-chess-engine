"""
Compare Heuristic Engine vs Neural Network Engine.
Plays games between the two engines to see which performs better.
"""

import chess
import time
from engine import ChessEngine
from nn_engine import NNEngine


def play_game(white_engine, black_engine, max_moves=100, verbose=True):
    """
    Play a game between two engines.
    
    Returns:
        result: "1-0", "0-1", or "1/2-1/2"
        moves: List of moves played
    """
    board = chess.Board()
    moves = []
    
    while not board.is_game_over() and len(moves) < max_moves * 2:
        if board.turn == chess.WHITE:
            move = white_engine.get_best_move(board)
        else:
            move = black_engine.get_best_move(board)
        
        if move is None:
            break
        
        if verbose:
            move_num = len(moves) // 2 + 1
            if board.turn == chess.WHITE:
                print(f"{move_num}. {board.san(move)}", end=" ")
            else:
                print(f"{board.san(move)}")
        
        moves.append(move)
        board.push(move)
    
    if verbose:
        print()
    
    if board.is_checkmate():
        result = "0-1" if board.turn == chess.WHITE else "1-0"
    elif board.is_game_over():
        result = "1/2-1/2"
    else:
        result = "1/2-1/2"  # Max moves reached
    
    return result, moves, board


def run_match(num_games=10, depth=3, time_limit=5.0):
    """
    Run a match between heuristic and NN engines.
    Each engine plays both colors.
    """
    print("="*60)
    print("ENGINE MATCH: Heuristic vs Neural Network")
    print("="*60)
    print(f"Games: {num_games}")
    print(f"Depth: {depth}")
    print(f"Time limit: {time_limit}s per move")
    print("="*60)
    
    # Create engines
    print("\nInitializing engines...")
    heuristic_engine = ChessEngine(max_depth=depth, time_limit=time_limit)
    nn_engine = NNEngine(weights_path="nn/weights.pth", max_depth=depth, time_limit=time_limit, device='cpu')
    
    # Track results
    heuristic_wins = 0
    nn_wins = 0
    draws = 0
    
    results = []
    
    for game_num in range(1, num_games + 1):
        # Alternate colors
        if game_num % 2 == 1:
            white_name = "Heuristic"
            black_name = "Neural Net"
            white_engine = heuristic_engine
            black_engine = nn_engine
        else:
            white_name = "Neural Net"
            black_name = "Heuristic"
            white_engine = nn_engine
            black_engine = heuristic_engine
        
        print(f"\n{'='*60}")
        print(f"GAME {game_num}: {white_name} (White) vs {black_name} (Black)")
        print("="*60)
        
        start_time = time.time()
        result, moves, final_board = play_game(
            white_engine, black_engine, 
            max_moves=80, 
            verbose=True
        )
        elapsed = time.time() - start_time
        
        # Determine winner
        if result == "1-0":
            winner = white_name
            if white_name == "Heuristic":
                heuristic_wins += 1
            else:
                nn_wins += 1
        elif result == "0-1":
            winner = black_name
            if black_name == "Heuristic":
                heuristic_wins += 1
            else:
                nn_wins += 1
        else:
            winner = "Draw"
            draws += 1
        
        results.append({
            'game': game_num,
            'white': white_name,
            'black': black_name,
            'result': result,
            'winner': winner,
            'moves': len(moves),
            'time': elapsed
        })
        
        print(f"\nResult: {result} - {winner}")
        print(f"Moves: {len(moves)}, Time: {elapsed:.1f}s")
        
        # Show current standings
        print(f"\nStandings: Heuristic {heuristic_wins} - {draws} - {nn_wins} Neural Net")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nHeuristic Engine: {heuristic_wins} wins")
    print(f"Neural Net Engine: {nn_wins} wins")
    print(f"Draws: {draws}")
    
    total = heuristic_wins + nn_wins + draws
    h_score = heuristic_wins + (draws * 0.5)
    nn_score = nn_wins + (draws * 0.5)
    
    print(f"\nScore: Heuristic {h_score}/{total} vs Neural Net {nn_score}/{total}")
    
    if heuristic_wins > nn_wins:
        print("\n🏆 WINNER: Heuristic Engine")
    elif nn_wins > heuristic_wins:
        print("\n🏆 WINNER: Neural Network Engine")
    else:
        print("\n🤝 MATCH DRAWN")
    
    # Game-by-game summary
    print("\n" + "-"*60)
    print("Game-by-Game Results:")
    print("-"*60)
    for r in results:
        print(f"Game {r['game']}: {r['white']} vs {r['black']} = {r['result']} ({r['moves']} moves)")
    
    return results


def quick_test():
    """Quick test with just 2 games."""
    print("Running quick test (2 games)...\n")
    run_match(num_games=2, depth=3, time_limit=3.0)


if __name__ == "__main__":
    # Run a 10-game match
    # Change num_games to 2 for a quick test
    run_match(num_games=6, depth=3, time_limit=5.0)
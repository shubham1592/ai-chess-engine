"""
Chess Engine - Main Entry Point
Foundations of AI Project

This module provides command-line interface for testing the chess engine.
For the GUI, run: streamlit run app.py
"""

import
from engine import ChessEngine, RandomEngine, GreedyEngine, play_game
from evaluation import evaluate_board, get_evaluation_breakdown


def print_board(board):
    """Print the board with coordinates."""
    print("\n  a b c d e f g h")
    print(" +-----------------+")
    for rank in range(7, -1, -1):
        print(f"{rank + 1}| ", end="")
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece:
                print(f"{piece.symbol()} ", end="")
            else:
                print(". ", end="")
        print(f"|{rank + 1}")
    print(" +-----------------+")
    print("  a b c d e f g h\n")


def play_vs_engine():
    """Play against the chess engine in console."""
    print("\n" + "="*50)
    print("   CHESS ENGINE - Minimax + Alpha-Beta Pruning")
    print("="*50)
    
    engine = ChessEngine(max_depth=4, time_limit=5.0)
    board = chess.Board()
    
    # Choose color
    color = input("\nPlay as White or Black? (w/b): ").lower()
    player_is_white = (color != 'b')
    
    print(f"\nYou are playing as {'White' if player_is_white else 'Black'}")
    print("Enter moves in algebraic notation (e.g., e4, Nf3, O-O)")
    print("Type 'quit' to exit, 'undo' to take back a move\n")
    
    while not board.is_game_over():
        print_board(board)
        
        # Show evaluation
        eval_score = evaluate_board(board)
        eval_display = eval_score / 100
        print(f"Evaluation: {eval_display:+.2f} ", end="")
        if eval_score > 50:
            print("(White is better)")
        elif eval_score < -50:
            print("(Black is better)")
        else:
            print("(Equal)")
        
        is_player_turn = (board.turn == chess.WHITE) == player_is_white
        
        if is_player_turn:
            # Player's turn
            print(f"\nYour turn ({'White' if board.turn else 'Black'})")
            print("Legal moves:", " ".join(board.san(m) for m in board.legal_moves))
            
            while True:
                move_str = input("Enter move: ").strip()
                
                if move_str.lower() == 'quit':
                    print("Thanks for playing!")
                    return
                
                if move_str.lower() == 'undo':
                    if len(board.move_stack) >= 2:
                        board.pop()
                        board.pop()
                        print("Undid last two moves.")
                        break
                    else:
                        print("No moves to undo.")
                        continue
                
                try:
                    move = board.parse_san(move_str)
                    board.push(move)
                    break
                except ValueError:
                    print("Invalid move! Try again.")
        
        else:
            # Engine's turn
            print(f"\nEngine thinking...")
            move, score, depth, stats = engine.iterative_deepening(board)
            
            print(f"Engine plays: {board.san(move)}")
            print(f"  Depth: {depth}, Nodes: {stats['nodes']:,}, Time: {stats['time']:.2f}s")
            
            board.push(move)
    
    # Game over
    print_board(board)
    print("\n" + "="*40)
    print("GAME OVER!")
    
    if board.is_checkmate():
        winner = "Black" if board.turn else "White"
        print(f"Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate - Draw!")
    elif board.is_insufficient_material():
        print("Insufficient material - Draw!")
    else:
        print(f"Result: {board.result()}")


def test_engine():
    """Test engine vs random/greedy opponents."""
    print("\n" + "="*50)
    print("   ENGINE TEST - Playing against baselines")
    print("="*50)
    
    engine = ChessEngine(max_depth=3, time_limit=2.0)
    
    # Test vs Random
    print("\n1. Testing vs Random Engine (10 games)...")
    wins, losses, draws = 0, 0, 0
    
    for i in range(10):
        random_engine = RandomEngine()
        
        if i % 2 == 0:
            result, _, _ = play_game(engine, random_engine, verbose=False)
        else:
            result, _, _ = play_game(random_engine, engine, verbose=False)
            # Flip result
            if result == "1-0":
                result = "0-1"
            elif result == "0-1":
                result = "1-0"
        
        if result == "1-0" and i % 2 == 0:
            wins += 1
        elif result == "0-1" and i % 2 == 1:
            wins += 1
        elif result == "1/2-1/2":
            draws += 1
        else:
            losses += 1
        
        print(f"  Game {i+1}: {result}")
    
    print(f"\nResults vs Random: {wins}W - {draws}D - {losses}L")
    
    # Test vs Greedy
    print("\n2. Testing vs Greedy Engine (10 games)...")
    wins, losses, draws = 0, 0, 0
    
    for i in range(10):
        greedy = GreedyEngine()
        
        if i % 2 == 0:
            result, _, _ = play_game(engine, greedy, verbose=False)
        else:
            result, _, _ = play_game(greedy, engine, verbose=False)
        
        if (result == "1-0" and i % 2 == 0) or (result == "0-1" and i % 2 == 1):
            wins += 1
        elif result == "1/2-1/2":
            draws += 1
        else:
            losses += 1
        
        print(f"  Game {i+1}: {result}")
    
    print(f"\nResults vs Greedy: {wins}W - {draws}D - {losses}L")


def analyze_position():
    """Analyze a position from FEN."""
    print("\n" + "="*50)
    print("   POSITION ANALYZER")
    print("="*50)
    
    engine = ChessEngine(max_depth=5, time_limit=10.0)
    
    print("\nEnter FEN (or press Enter for starting position):")
    fen = input().strip()
    
    if not fen:
        fen = chess.STARTING_FEN
    
    try:
        board = chess.Board(fen)
    except ValueError:
        print("Invalid FEN!")
        return
    
    print_board(board)
    
    print("\nAnalyzing position...")
    analysis = engine.analyze_position(board)
    
    print(f"\nAnalysis Results:")
    print(f"  Best Move: {board.san(analysis['best_move'])}")
    print(f"  Evaluation: {analysis['score']/100:+.2f}")
    print(f"  Depth: {analysis['depth']}")
    print(f"  Nodes searched: {analysis['nodes']:,}")
    print(f"  Time: {analysis['time']:.2f}s")
    print(f"  Nodes/sec: {analysis['nps']:,}")
    
    print(f"\nTop 5 moves:")
    for move, score in analysis['top_moves']:
        print(f"  {board.san(move)}: {score/100:+.2f}")
    
    print(f"\nEvaluation Breakdown:")
    breakdown = get_evaluation_breakdown(board)
    print(f"  Material:      {breakdown['material']/100:+.2f}")
    print(f"  Positioning:   {breakdown['piece_position']/100:+.2f}")
    print(f"  Pawn Structure:{breakdown['pawn_structure']/100:+.2f}")
    print(f"  King Safety:   {breakdown['king_safety']/100:+.2f}")
    print(f"  Mobility:      {breakdown['mobility']/100:+.2f}")
    print(f"  Center:        {break12.60down['center_control']/100:+.2f}")
    print(f"  Rooks:         {breakdown['rook_placement']/100:+.2f}")
    print(f"  Bishop Pair:   {breakdown['bishop_pair']/100:+.2f}")
    print(f"  TOTAL:         {breakdown['total']/100:+.2f}")


def main():
    """Main menu."""
    while True:
        print("\n" + "="*50)
        print("   CHESS ENGINE - Foundations of AI Project")
        print("="*50)
        print("\n1. Play against the engine")
        print("2. Test engine vs baselines")
        print("3. Analyze a position")
        print("4. Run Streamlit GUI")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            play_vs_engine()
        elif choice == '2':
            test_engine()
        elif choice == '3':
            analyze_position()
        elif choice == '4':
            print("\nTo run the GUI, execute in terminal:")
            print("  streamlit run app.py")
            print("\nThis will open the web interface in your browser.")
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("Invalid option!")


if __name__ == "__main__":
    main()
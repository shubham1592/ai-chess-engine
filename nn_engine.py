"""
Neural Network Chess Engine

This engine uses a trained neural network to evaluate positions instead of
handcrafted heuristics. The NN was trained on ~300k positions labeled by
Stockfish, so it learned evaluation patterns from a strong engine.

We combine the NN score with simple material counting (70/30 split) to
prevent tactical blunders while keeping the NN's positional understanding.
"""

import chess
import torch
import time
import os

# Add parent directory to path so we can import from nn/
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.model import ChessEvaluationNet
from nn.dataset import board_to_tensor
from move_ordering import order_moves, get_quiescence_moves


class NNEngine:
    """
    Chess engine that uses a neural network for position evaluation.
    
    The NN outputs a score between -1 (Black winning) and +1 (White winning).
    We scale this to centipawns and mix it with material counting for safety.
    """
    
    def __init__(self, weights_path="nn/weights_stockfish.pth", max_depth=3, time_limit=10.0, device=None):
        """
        Load the trained neural network and set up search parameters.
        
        Args:
            weights_path: Path to the saved model weights
            max_depth: How many moves ahead to search (lower than heuristic engine
                       because NN inference is slower)
            time_limit: Max seconds per move
            device: 'cuda' or 'cpu' (defaults to cpu for stability)
        """
        # Use CPU by default to avoid CUDA issues
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        
        # Load the trained model
        self.model = ChessEvaluationNet()
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode (disables dropout)
        
        self.max_depth = max_depth
        self.time_limit = time_limit
        
        # Search statistics
        self.nodes_searched = 0
        self.nn_calls = 0
        self.start_time = None
        self.time_exceeded = False
        
        # Cache evaluated positions to avoid redundant NN calls
        self.eval_cache = {}
        
        print(f"NN Engine initialized on {self.device}")
    
    def reset_stats(self):
        """Reset search statistics."""
        self.nodes_searched = 0
        self.nn_calls = 0
        self.time_exceeded = False
        self.eval_cache.clear()
    
    def nn_evaluate(self, board):
        """
        Evaluate a position using neural network + material counting.
        
        We use a 70/30 split between NN and material. This hybrid approach
        works better than pure NN because:
        - NN provides positional understanding learned from Stockfish
        - Material counting prevents obvious blunders (taking free pieces)
        
        Returns:
            Score in centipawns (positive = White better)
        """
        # Check cache first to avoid redundant NN calls
        fen = board.fen()
        if fen in self.eval_cache:
            return self.eval_cache[fen]
        
        self.nn_calls += 1
        
        # Convert board to tensor format the NN expects
        tensor = torch.tensor(board_to_tensor(board), dtype=torch.float32)
        tensor = tensor.to(self.device)
        
        # Get NN evaluation (returns value between -1 and +1)
        with torch.no_grad():
            nn_score = self.model.evaluate(tensor)
        
        # Scale NN output to centipawns (roughly -1000 to +1000)
        nn_score_cp = nn_score * 1000
        
        # Count material as a safety check
        material_score = self._count_material(board)
        
        # Combine: 70% NN judgment + 30% material safety
        combined_score = (0.7 * nn_score_cp) + (0.3 * material_score)
        
        # Cache for later
        self.eval_cache[fen] = combined_score
        
        return combined_score
    
    def _count_material(self, board):
        """
        Simple material counting in centipawns.
        
        This is much simpler than our full heuristic evaluation but catches
        obvious material imbalances that the NN might miss.
        """
        import chess
        
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        score = 0
        for piece_type in piece_values:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            score += piece_values[piece_type] * (white_count - black_count)
        
        return score
    
    def quiescence_search(self, board, alpha, beta, depth=0):
        """
        Quiescence search using NN evaluation.
        """
        self.nodes_searched += 1
        
        if time.time() - self.start_time > self.time_limit:
            self.time_exceeded = True
            return self.nn_evaluate(board)
        
        stand_pat = self.nn_evaluate(board)
        
        if board.turn == chess.WHITE:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat
        
        if depth > 4:
            return stand_pat
        
        tactical_moves = get_quiescence_moves(board)
        
        if board.turn == chess.WHITE:
            for move in tactical_moves:
                board.push(move)
                score = self.quiescence_search(board, alpha, beta, depth + 1)
                board.pop()
                
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            return alpha
        else:
            for move in tactical_moves:
                board.push(move)
                score = self.quiescence_search(board, alpha, beta, depth + 1)
                board.pop()
                
                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score
            return beta
    
    def minimax(self, board, depth, alpha, beta, maximizing):
        """
        Minimax with alpha-beta pruning using NN evaluation.
        """
        self.nodes_searched += 1
        
        if time.time() - self.start_time > self.time_limit:
            self.time_exceeded = True
            return self.nn_evaluate(board), None
        
        # Terminal conditions
        if board.is_game_over():
            if board.is_checkmate():
                if board.turn == chess.WHITE:
                    return -99999 - depth, None
                else:
                    return 99999 + depth, None
            return 0, None
        
        # Leaf node
        if depth == 0:
            return self.quiescence_search(board, alpha, beta), None
        
        moves = order_moves(board)
        
        if not moves:
            return self.nn_evaluate(board), None
        
        best_move = moves[0]
        
        if maximizing:
            max_eval = float('-inf')
            
            for move in moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if self.time_exceeded:
                    return max_eval, best_move
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            
            for move in moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if self.time_exceeded:
                    return min_eval, best_move
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    def iterative_deepening(self, board):
        """Search with iterative deepening."""
        self.reset_stats()
        self.start_time = time.time()
        
        best_move = None
        best_score = 0
        depth_reached = 0
        
        maximizing = (board.turn == chess.WHITE)
        
        for depth in range(1, self.max_depth + 1):
            if self.time_exceeded:
                break
            
            score, move = self.minimax(
                board, depth, float('-inf'), float('inf'), maximizing
            )
            
            if not self.time_exceeded and move:
                best_move = move
                best_score = score
                depth_reached = depth
            
            if abs(score) > 90000:
                break
        
        elapsed = time.time() - self.start_time
        
        stats = {
            'depth': depth_reached,
            'nodes': self.nodes_searched,
            'nn_calls': self.nn_calls,
            'time': elapsed,
            'nps': int(self.nodes_searched / elapsed) if elapsed > 0 else 0
        }
        
        return best_move, best_score, stats
    
    def get_best_move(self, board):
        """Get best move for current position."""
        move, score, stats = self.iterative_deepening(board)
        return move
    
    def evaluate_position(self, board):
        """Get NN evaluation for a position (for display)."""
        return self.nn_evaluate(board)


# Test the engine
if __name__ == "__main__":
    print("="*50)
    print("Neural Network Chess Engine Test")
    print("="*50)
    
    # Create engine (using new Stockfish-trained weights)
    engine = NNEngine(weights_path="nn/weights_stockfish.pth", max_depth=3, device='cpu')
    
    # Test on starting position
    board = chess.Board()
    
    print("\nFinding best move for starting position...")
    move, score, stats = engine.iterative_deepening(board)
    
    print(f"\nBest move: {board.san(move)}")
    print(f"Score: {score/100:+.2f} pawns")
    print(f"Depth: {stats['depth']}")
    print(f"Nodes: {stats['nodes']:,}")
    print(f"NN calls: {stats['nn_calls']:,}")
    print(f"Time: {stats['time']:.2f}s")
    print(f"Nodes/sec: {stats['nps']:,}")
    
    # Play a few moves
    print("\n" + "="*50)
    print("Playing first 6 moves...")
    print("="*50)
    
    board = chess.Board()
    for i in range(6):
        move, score, stats = engine.iterative_deepening(board)
        print(f"\nMove {i+1}: {board.san(move)} (eval: {score/100:+.2f})")
        board.push(move)
    
    print("\nFinal position:")
    print(board)
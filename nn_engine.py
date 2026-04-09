"""
Neural Network Chess Engine.
Uses trained neural network for position evaluation with Minimax search.
"""

import chess
import torch
import time
import os

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.model import ChessEvaluationNet
from nn.dataset import board_to_tensor
from move_ordering import order_moves, get_quiescence_moves


class NNEngine:
    """Chess engine using neural network evaluation."""
    
    def __init__(self, weights_path="nn/weights_stockfish.pth", max_depth=3, time_limit=10.0, device=None):
        """
        Args:
            weights_path: Path to trained model weights
            max_depth: Maximum search depth
            time_limit: Time limit per move in seconds
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        # Set device (default to CPU for stability)
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        
        # Load model
        self.model = ChessEvaluationNet()
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.max_depth = max_depth
        self.time_limit = time_limit
        
        # Stats
        self.nodes_searched = 0
        self.nn_calls = 0
        self.start_time = None
        self.time_exceeded = False
        
        # Cache for evaluated positions
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
        Evaluate position using neural network + material bonus.
        Returns score in centipawns.
        """
        # Check cache first
        fen = board.fen()
        if fen in self.eval_cache:
            return self.eval_cache[fen]
        
        self.nn_calls += 1
        
        # Convert board to tensor
        tensor = torch.tensor(board_to_tensor(board), dtype=torch.float32)
        tensor = tensor.to(self.device)
        
        # Get NN evaluation
        with torch.no_grad():
            nn_score = self.model.evaluate(tensor)
        
        # Scale NN score from [-1, 1] to centipawns [-1000, 1000]
        nn_score_cp = nn_score * 1000
        
        # Add material counting (helps with tactics)
        material_score = self._count_material(board)
        
        # Combine: 70% NN + 30% material
        combined_score = (0.7 * nn_score_cp) + (0.3 * material_score)
        
        # Cache result
        self.eval_cache[fen] = combined_score
        
        return combined_score
    
    def _count_material(self, board):
        """Simple material counting in centipawns."""
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
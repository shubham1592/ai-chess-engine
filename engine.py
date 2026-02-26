"""
Chess Engine with Minimax + Alpha-Beta Pruning + Iterative Deepening
"""

import chess
import time
from evaluation import evaluate_board, PIECE_VALUES
from move_ordering import order_moves, get_quiescence_moves


class ChessEngine:
    """Chess engine using Minimax with Alpha-Beta pruning."""
    
    def __init__(self, max_depth=4, time_limit=10.0):
        self.max_depth = max_depth
        self.time_limit = time_limit  # seconds
        self.nodes_searched = 0
        self.start_time = None
        self.time_exceeded = False
        
        # Transposition table for caching positions
        self.transposition_table = {}
        
        # Stats for analysis
        self.stats = {
            'nodes': 0,
            'tt_hits': 0,
            'cutoffs': 0,
            'depth_reached': 0
        }
    
    def reset_stats(self):
        """Reset search statistics."""
        self.stats = {
            'nodes': 0,
            'tt_hits': 0,
            'cutoffs': 0,
            'depth_reached': 0
        }
        self.nodes_searched = 0
        self.time_exceeded = False
    
    def get_board_hash(self, board):
        """Get hash of board position for transposition table."""
        return board.fen()
    
    def quiescence_search(self, board, alpha, beta, depth=0):
        """
        Quiescence search to avoid horizon effect.
        Only searches tactical moves (captures, checks) until position is quiet.
        """
        self.nodes_searched += 1
        self.stats['nodes'] += 1
        
        # Check time
        if time.time() - self.start_time > self.time_limit:
            self.time_exceeded = True
            return evaluate_board(board)
        
        # Stand-pat score
        stand_pat = evaluate_board(board)
        
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
        
        # Limit quiescence depth
        if depth > 6:
            return stand_pat
        
        # Search only tactical moves
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
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: chess.Board object
            depth: remaining depth to search
            alpha: best score for maximizer
            beta: best score for minimizer
            maximizing: True if maximizing player (White)
        
        Returns:
            (score, best_move) tuple
        """
        self.nodes_searched += 1
        self.stats['nodes'] += 1
        
        # Check time limit
        if time.time() - self.start_time > self.time_limit:
            self.time_exceeded = True
            return evaluate_board(board), None
        
        # Check transposition table
        board_hash = self.get_board_hash(board)
        if board_hash in self.transposition_table:
            tt_entry = self.transposition_table[board_hash]
            if tt_entry['depth'] >= depth:
                self.stats['tt_hits'] += 1
                return tt_entry['score'], tt_entry['best_move']
        
        # Terminal conditions
        if board.is_game_over():
            if board.is_checkmate():
                # Add depth bonus to prefer faster checkmates
                if board.turn == chess.WHITE:
                    return -99999 - depth, None  # Black wins
                else:
                    return 99999 + depth, None   # White wins
            return 0, None  # Draw
        
        # Leaf node - use quiescence search
        if depth == 0:
            return self.quiescence_search(board, alpha, beta), None
        
        # Get ordered moves
        tt_move = None
        if board_hash in self.transposition_table:
            tt_move = self.transposition_table[board_hash].get('best_move')
        
        moves = order_moves(board, tt_move=tt_move)
        
        if not moves:
            return evaluate_board(board), None
        
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
                    self.stats['cutoffs'] += 1
                    break  # Beta cutoff
            
            # Store in transposition table
            self.transposition_table[board_hash] = {
                'score': max_eval,
                'depth': depth,
                'best_move': best_move
            }
            
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
                    self.stats['cutoffs'] += 1
                    break  # Alpha cutoff
            
            # Store in transposition table
            self.transposition_table[board_hash] = {
                'score': min_eval,
                'depth': depth,
                'best_move': best_move
            }
            
            return min_eval, best_move
    
    def iterative_deepening(self, board):
        """
        Iterative deepening search.
        Searches progressively deeper until time limit.
        
        Returns:
            (best_move, score, depth_reached, stats)
        """
        self.reset_stats()
        self.start_time = time.time()
        
        best_move = None
        best_score = 0
        
        maximizing = (board.turn == chess.WHITE)
        
        for depth in range(1, self.max_depth + 1):
            if self.time_exceeded:
                break
            
            score, move = self.minimax(
                board, 
                depth, 
                float('-inf'), 
                float('inf'), 
                maximizing
            )
            
            if not self.time_exceeded and move:
                best_move = move
                best_score = score
                self.stats['depth_reached'] = depth
            
            # Early exit if we found checkmate
            if abs(score) > 90000:
                break
        
        elapsed = time.time() - self.start_time
        self.stats['time'] = elapsed
        self.stats['nps'] = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
        
        return best_move, best_score, self.stats['depth_reached'], self.stats
    
    def get_best_move(self, board):
        """
        Get the best move for the current position.
        Main interface method.
        
        Returns:
            chess.Move object
        """
        move, score, depth, stats = self.iterative_deepening(board)
        return move
    
    def analyze_position(self, board, depth=None):
        """
        Analyze a position and return detailed information.
        
        Returns:
            dict with analysis results
        """
        if depth:
            old_depth = self.max_depth
            self.max_depth = depth
        
        move, score, depth_reached, stats = self.iterative_deepening(board)
        
        if depth:
            self.max_depth = old_depth
        
        # Get top moves
        top_moves = self.get_top_moves(board, n=5)
        
        return {
            'best_move': move,
            'score': score,
            'depth': depth_reached,
            'nodes': stats['nodes'],
            'time': stats.get('time', 0),
            'nps': stats.get('nps', 0),
            'tt_hits': stats['tt_hits'],
            'cutoffs': stats['cutoffs'],
            'top_moves': top_moves
        }
    
    def get_top_moves(self, board, n=5):
        """Get top n moves with their evaluations."""
        moves_with_scores = []
        maximizing = (board.turn == chess.WHITE)
        
        for move in board.legal_moves:
            board.push(move)
            # Quick evaluation (depth 1)
            score = evaluate_board(board)
            board.pop()
            moves_with_scores.append((move, score))
        
        # Sort by score
        moves_with_scores.sort(
            key=lambda x: x[1], 
            reverse=maximizing
        )
        
        return moves_with_scores[:n]
    
    def clear_transposition_table(self):
        """Clear the transposition table."""
        self.transposition_table.clear()


class RandomEngine:
    """Simple random move engine for testing."""
    
    def __init__(self):
        import random
        self.random = random
    
    def get_best_move(self, board):
        moves = list(board.legal_moves)
        if moves:
            return self.random.choice(moves)
        return None


class GreedyEngine:
    """Greedy engine that captures highest value pieces."""
    
    def get_best_move(self, board):
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            score = evaluate_board(board)
            board.pop()
            
            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        
        return best_move


def play_game(white_engine, black_engine, max_moves=200, verbose=True):
    """
    Play a game between two engines.
    
    Returns:
        (result, board, move_list)
    """
    board = chess.Board()
    move_list = []
    
    while not board.is_game_over() and len(move_list) < max_moves:
        if board.turn == chess.WHITE:
            move = white_engine.get_best_move(board)
        else:
            move = black_engine.get_best_move(board)
        
        if move is None:
            break
        
        if verbose:
            print(f"Move {len(move_list) + 1}: {board.san(move)}")
        
        move_list.append(move)
        board.push(move)
    
    result = board.result()
    return result, board, move_list
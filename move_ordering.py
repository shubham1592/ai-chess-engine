"""
Move ordering for alpha-beta pruning optimization.
Orders moves: Checks > Captures (MVV-LVA) > Threats > Quiet moves
"""

import chess
from evaluation import PIECE_VALUES


def score_move(board, move):
    """
    Score a move for ordering purposes.
    Higher scores = moves searched first.
    """
    score = 0
    
    # 1. Checks get highest priority (10000+)
    board.push(move)
    is_check = board.is_check()
    board.pop()
    
    if is_check:
        score += 10000
    
    # 2. Captures scored by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if victim:
            victim_value = PIECE_VALUES.get(victim.piece_type, 0)
            attacker_value = PIECE_VALUES.get(attacker.piece_type, 0)
            # MVV-LVA: prioritize capturing high-value pieces with low-value pieces
            score += 5000 + (victim_value * 10 - attacker_value)
        
        # En passant
        if board.is_en_passant(move):
            score += 5000 + PIECE_VALUES[chess.PAWN] * 10
    
    # 3. Promotions
    if move.promotion:
        score += 4000 + PIECE_VALUES.get(move.promotion, 0)
    
    # 4. Threats (attacks on enemy pieces after the move)
    if not board.is_capture(move) and not is_check:
        piece = board.piece_at(move.from_square)
        if piece:
            # Check if move attacks any enemy pieces
            board.push(move)
            for sq in chess.SQUARES:
                target = board.piece_at(sq)
                if target and target.color != piece.color:
                    if board.is_attacked_by(piece.color, sq):
                        score += 100 + PIECE_VALUES.get(target.piece_type, 0) // 10
            board.pop()
    
    # 5. Pawn moves toward promotion
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(move.to_square)
        if piece.color == chess.WHITE:
            score += to_rank * 5  # Closer to 8th rank = higher score
        else:
            score += (7 - to_rank) * 5  # Closer to 1st rank = higher score
    
    # 6. Castling bonus
    if board.is_castling(move):
        score += 500
    
    # 7. Central moves (d4, d5, e4, e5 target squares)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    if to_file in [3, 4] and to_rank in [3, 4]:
        score += 50
    
    return score


def order_moves(board, moves=None, tt_move=None):
    """
    Order moves for more efficient alpha-beta pruning.
    
    Args:
        board: chess.Board object
        moves: list of moves to order (defaults to all legal moves)
        tt_move: best move from transposition table (searched first)
    
    Returns:
        Sorted list of moves (best first)
    """
    if moves is None:
        moves = list(board.legal_moves)
    
    scored_moves = []
    for move in moves:
        score = score_move(board, move)
        
        # Transposition table move gets highest priority
        if tt_move and move == tt_move:
            score += 100000
        
        scored_moves.append((score, move))
    
    # Sort by score descending (highest first)
    scored_moves.sort(key=lambda x: x[0], reverse=True)
    
    return [move for score, move in scored_moves]


def is_tactical_move(board, move):
    """Check if a move is tactical (capture, check, or promotion)."""
    if board.is_capture(move):
        return True
    
    if move.promotion:
        return True
    
    # Check if move gives check
    board.push(move)
    is_check = board.is_check()
    board.pop()
    
    return is_check


def get_quiescence_moves(board):
    """
    Get moves for quiescence search (only tactical moves).
    Used to avoid horizon effect.
    """
    tactical_moves = []
    
    for move in board.legal_moves:
        if is_tactical_move(board, move):
            tactical_moves.append(move)
    
    return order_moves(board, tactical_moves)


def get_move_category(board, move):
    """Get human-readable category of a move for debugging."""
    categories = []
    
    board.push(move)
    if board.is_check():
        categories.append("Check")
    board.pop()
    
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        if victim:
            piece_names = {
                chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
                chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"
            }
            categories.append(f"Capture {piece_names.get(victim.piece_type, '?')}")
        elif board.is_en_passant(move):
            categories.append("En Passant")
    
    if move.promotion:
        promo_names = {chess.QUEEN: "Q", chess.ROOK: "R", chess.BISHOP: "B", chess.KNIGHT: "N"}
        categories.append(f"Promote to {promo_names.get(move.promotion, '?')}")
    
    if board.is_castling(move):
        if chess.square_file(move.to_square) == 6:
            categories.append("Kingside Castle")
        else:
            categories.append("Queenside Castle")
    
    if not categories:
        categories.append("Quiet")
    
    return ", ".join(categories)
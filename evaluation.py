"""
Heuristic-based board evaluation for chess engine.
Evaluates positions based on material, piece positioning, and strategic factors.
"""

import chess

# Piece values (centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-Square Tables (PST) - bonus/penalty based on piece position
# Values are from White's perspective, flipped for Black

PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_MIDDLEGAME_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

KING_ENDGAME_TABLE = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

PST = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_MIDDLEGAME_TABLE
}


def get_piece_square_value(piece_type, square, is_white, is_endgame=False):
    """Get positional bonus for a piece on a given square."""
    if piece_type == chess.KING and is_endgame:
        table = KING_ENDGAME_TABLE
    else:
        table = PST[piece_type]
    
    if is_white:
        return table[63 - square]  # Flip for white
    else:
        return table[square]


def is_endgame(board):
    """Determine if we're in endgame phase."""
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    minor_pieces = (len(board.pieces(chess.KNIGHT, chess.WHITE)) + 
                    len(board.pieces(chess.BISHOP, chess.WHITE)) +
                    len(board.pieces(chess.KNIGHT, chess.BLACK)) + 
                    len(board.pieces(chess.BISHOP, chess.BLACK)))
    
    return queens == 0 or (queens == 2 and minor_pieces <= 2)


def evaluate_material(board):
    """Calculate material balance."""
    score = 0
    for piece_type in PIECE_VALUES:
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        score += PIECE_VALUES[piece_type] * (white_pieces - black_pieces)
    return score


def evaluate_piece_positions(board, endgame):
    """Evaluate piece positioning using piece-square tables."""
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = get_piece_square_value(piece.piece_type, square, piece.color, endgame)
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    return score


def evaluate_pawn_structure(board):
    """Evaluate pawn structure (doubled, isolated, passed pawns)."""
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        multiplier = 1 if color == chess.WHITE else -1
        
        files_with_pawns = [chess.square_file(sq) for sq in pawns]
        
        # Doubled pawns penalty
        for f in range(8):
            pawns_on_file = files_with_pawns.count(f)
            if pawns_on_file > 1:
                score -= 20 * (pawns_on_file - 1) * multiplier
        
        # Isolated pawns penalty
        for pawn_sq in pawns:
            f = chess.square_file(pawn_sq)
            has_neighbor = False
            if f > 0 and (f - 1) in files_with_pawns:
                has_neighbor = True
            if f < 7 and (f + 1) in files_with_pawns:
                has_neighbor = True
            if not has_neighbor:
                score -= 15 * multiplier
        
        # Passed pawns bonus
        for pawn_sq in pawns:
            f = chess.square_file(pawn_sq)
            r = chess.square_rank(pawn_sq)
            is_passed = True
            
            enemy_pawns = board.pieces(chess.PAWN, not color)
            for enemy_sq in enemy_pawns:
                ef = chess.square_file(enemy_sq)
                er = chess.square_rank(enemy_sq)
                if abs(ef - f) <= 1:
                    if color == chess.WHITE and er > r:
                        is_passed = False
                        break
                    elif color == chess.BLACK and er < r:
                        is_passed = False
                        break
            
            if is_passed:
                # Bonus increases as pawn advances
                if color == chess.WHITE:
                    score += 20 + (r * 10) * multiplier
                else:
                    score += 20 + ((7 - r) * 10) * multiplier
    
    return score


def evaluate_king_safety(board, endgame):
    """Evaluate king safety (castling, pawn shield)."""
    if endgame:
        return 0  # King safety less important in endgame
    
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        multiplier = 1 if color == chess.WHITE else -1
        king_sq = board.king(color)
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        
        # Bonus for castled king (king on g or c file)
        if color == chess.WHITE:
            if king_file in [6, 2] and king_rank == 0:
                score += 30 * multiplier
        else:
            if king_file in [6, 2] and king_rank == 7:
                score += 30 * multiplier
        
        # Pawn shield bonus
        shield_squares = []
        if color == chess.WHITE and king_rank == 0:
            for df in [-1, 0, 1]:
                f = king_file + df
                if 0 <= f <= 7:
                    shield_squares.append(chess.square(f, 1))
                    shield_squares.append(chess.square(f, 2))
        elif color == chess.BLACK and king_rank == 7:
            for df in [-1, 0, 1]:
                f = king_file + df
                if 0 <= f <= 7:
                    shield_squares.append(chess.square(f, 6))
                    shield_squares.append(chess.square(f, 5))
        
        for sq in shield_squares:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                score += 10 * multiplier
    
    return score


def evaluate_mobility(board):
    """Evaluate piece mobility (number of legal moves)."""
    # Save current turn
    original_turn = board.turn
    
    board.turn = chess.WHITE
    white_mobility = len(list(board.legal_moves))
    
    board.turn = chess.BLACK
    black_mobility = len(list(board.legal_moves))
    
    # Restore turn
    board.turn = original_turn
    
    return (white_mobility - black_mobility) * 5


def evaluate_center_control(board):
    """Evaluate control of center squares."""
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    extended_center = [chess.C3, chess.C4, chess.C5, chess.C6,
                       chess.D3, chess.D6, chess.E3, chess.E6,
                       chess.F3, chess.F4, chess.F5, chess.F6]
    
    score = 0
    
    for sq in center_squares:
        piece = board.piece_at(sq)
        if piece:
            if piece.color == chess.WHITE:
                score += 10
            else:
                score -= 10
        
        # Control of center by attacks
        if board.is_attacked_by(chess.WHITE, sq):
            score += 5
        if board.is_attacked_by(chess.BLACK, sq):
            score -= 5
    
    for sq in extended_center:
        if board.is_attacked_by(chess.WHITE, sq):
            score += 2
        if board.is_attacked_by(chess.BLACK, sq):
            score -= 2
    
    return score


def evaluate_rook_placement(board):
    """Evaluate rook placement (open files, 7th rank)."""
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        multiplier = 1 if color == chess.WHITE else -1
        rooks = board.pieces(chess.ROOK, color)
        
        for rook_sq in rooks:
            f = chess.square_file(rook_sq)
            r = chess.square_rank(rook_sq)
            
            # Rook on 7th rank bonus
            if (color == chess.WHITE and r == 6) or (color == chess.BLACK and r == 1):
                score += 20 * multiplier
            
            # Open/semi-open file bonus
            white_pawns_on_file = any(chess.square_file(sq) == f 
                                      for sq in board.pieces(chess.PAWN, chess.WHITE))
            black_pawns_on_file = any(chess.square_file(sq) == f 
                                      for sq in board.pieces(chess.PAWN, chess.BLACK))
            
            if not white_pawns_on_file and not black_pawns_on_file:
                score += 15 * multiplier  # Open file
            elif (color == chess.WHITE and not white_pawns_on_file) or \
                 (color == chess.BLACK and not black_pawns_on_file):
                score += 10 * multiplier  # Semi-open file
        
        # Connected rooks bonus
        rook_list = list(rooks)
        if len(rook_list) == 2:
            r1, r2 = rook_list
            r1_rank, r1_file = chess.square_rank(r1), chess.square_file(r1)
            r2_rank, r2_file = chess.square_rank(r2), chess.square_file(r2)
            
            connected = False
            # Same rank - check if path is clear
            if r1_rank == r2_rank:
                min_file = min(r1_file, r2_file)
                max_file = max(r1_file, r2_file)
                clear = True
                for f in range(min_file + 1, max_file):
                    if board.piece_at(chess.square(f, r1_rank)) is not None:
                        clear = False
                        break
                if clear:
                    connected = True
            # Same file - check if path is clear
            elif r1_file == r2_file:
                min_rank = min(r1_rank, r2_rank)
                max_rank = max(r1_rank, r2_rank)
                clear = True
                for r in range(min_rank + 1, max_rank):
                    if board.piece_at(chess.square(r1_file, r)) is not None:
                        clear = False
                        break
                if clear:
                    connected = True
            
            if connected:
                score += 10 * multiplier
    
    return score


def evaluate_bishop_pair(board):
    """Bonus for having the bishop pair."""
    score = 0
    
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 30
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 30
    
    return score


def evaluate_board(board):
    """
    Main evaluation function combining all heuristics.
    Returns score in centipawns from White's perspective.
    Positive = White advantage, Negative = Black advantage.
    """
    # Check for game over conditions
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -99999  # Black wins
        else:
            return 99999   # White wins
    
    if board.is_stalemate() or board.is_insufficient_material() or \
       board.is_fifty_moves() or board.is_repetition():
        return 0  # Draw
    
    endgame = is_endgame(board)
    
    score = 0
    score += evaluate_material(board)
    score += evaluate_piece_positions(board, endgame)
    score += evaluate_pawn_structure(board)
    score += evaluate_king_safety(board, endgame)
    score += evaluate_mobility(board)
    score += evaluate_center_control(board)
    score += evaluate_rook_placement(board)
    score += evaluate_bishop_pair(board)
    
    return score


def get_evaluation_breakdown(board):
    """Get detailed breakdown of evaluation for display."""
    endgame = is_endgame(board)
    
    return {
        'material': evaluate_material(board),
        'piece_position': evaluate_piece_positions(board, endgame),
        'pawn_structure': evaluate_pawn_structure(board),
        'king_safety': evaluate_king_safety(board, endgame),
        'mobility': evaluate_mobility(board),
        'center_control': evaluate_center_control(board),
        'rook_placement': evaluate_rook_placement(board),
        'bishop_pair': evaluate_bishop_pair(board),
        'total': evaluate_board(board),
        'is_endgame': endgame
    }
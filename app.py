"""
Streamlit Frontend for Chess Engine
Play against the AI with evaluation bar and analysis
"""

import streamlit as st
import chess
import chess.svg
from engine import ChessEngine
from evaluation import evaluate_board, get_evaluation_breakdown

# Page config
st.set_page_config(
    page_title="Chess Engine - Minimax AI",
    page_icon="♟️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    .eval-bar { 
        width: 30px; 
        height: 400px; 
        border: 2px solid #333;
        border-radius: 4px;
        position: relative;
        background: linear-gradient(to top, #1a1a1a 50%, #f0f0f0 50%);
    }
    .eval-marker {
        position: absolute;
        width: 100%;
        height: 4px;
        background: #ff4444;
        transition: top 0.3s ease;
    }
    .move-history {
        font-family: monospace;
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        background: #f5f5f5;
        border-radius: 8px;
    }
    .stats-box {
        background: #e8f4e8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()
if 'engine' not in st.session_state:
    st.session_state.engine = ChessEngine(max_depth=4, time_limit=5.0)
if 'move_history' not in st.session_state:
    st.session_state.move_history = []
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []
if 'player_color' not in st.session_state:
    st.session_state.player_color = chess.WHITE
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'last_engine_stats' not in st.session_state:
    st.session_state.last_engine_stats = None


def reset_game():
    """Reset the game to initial state."""
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.evaluation_history = []
    st.session_state.game_started = False
    st.session_state.last_engine_stats = None
    st.session_state.engine.clear_transposition_table()


def make_move(move):
    """Make a move on the board."""
    san = st.session_state.board.san(move)
    st.session_state.board.push(move)
    st.session_state.move_history.append(san)
    
    # Record evaluation
    eval_score = evaluate_board(st.session_state.board)
    st.session_state.evaluation_history.append(eval_score)


def engine_move():
    """Make the engine's move."""
    if st.session_state.board.is_game_over():
        return
    
    with st.spinner("Engine thinking..."):
        move, score, depth, stats = st.session_state.engine.iterative_deepening(
            st.session_state.board
        )
        st.session_state.last_engine_stats = stats
    
    if move:
        make_move(move)


def get_eval_percentage(score):
    """Convert centipawn score to percentage for eval bar."""
    # Clamp score between -1000 and 1000 for display
    clamped = max(-1000, min(1000, score))
    # Convert to 0-100 percentage (50 = equal)
    percentage = 50 + (clamped / 20)
    return max(0, min(100, percentage))


def render_board_svg(board, size=400):
    """Render the chess board as SVG."""
    # Flip board if player is black
    flipped = st.session_state.player_color == chess.BLACK
    
    # Highlight last move
    lastmove = None
    if board.move_stack:
        lastmove = board.peek()
    
    # Check square
    check = None
    if board.is_check():
        check = board.king(board.turn)
    
    svg = chess.svg.board(
        board,
        size=size,
        flipped=flipped,
        lastmove=lastmove,
        check=check,
        colors={
            'square light': '#f0d9b5',
            'square dark': '#b58863',
            'square light lastmove': '#cdd16a',
            'square dark lastmove': '#aaa23a'
        }
    )
    return svg


# Main layout
st.title("♟️ Chess Engine - Minimax with Alpha-Beta Pruning")
st.markdown("Play against an AI using adversarial search algorithms!")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Engine depth
    depth = st.slider("Search Depth", 1, 6, 4, 
                      help="Higher = stronger but slower")
    st.session_state.engine.max_depth = depth
    
    # Time limit
    time_limit = st.slider("Time Limit (seconds)", 1, 30, 5)
    st.session_state.engine.time_limit = time_limit
    
    st.divider()
    
    # Color selection
    color = st.radio("Play as", ["White", "Black"])
    new_color = chess.WHITE if color == "White" else chess.BLACK
    
    if new_color != st.session_state.player_color:
        st.session_state.player_color = new_color
        reset_game()
    
    st.divider()
    
    # New game button
    if st.button("🔄 New Game", use_container_width=True):
        reset_game()
        st.rerun()
    
    st.divider()
    
    # Evaluation breakdown
    st.header("📊 Position Analysis")
    breakdown = get_evaluation_breakdown(st.session_state.board)
    
    st.metric("Total Evaluation", f"{breakdown['total'] / 100:.2f}")
    
    with st.expander("Detailed Breakdown"):
        st.write(f"Material: {breakdown['material'] / 100:.2f}")
        st.write(f"Piece Position: {breakdown['piece_position'] / 100:.2f}")
        st.write(f"Pawn Structure: {breakdown['pawn_structure'] / 100:.2f}")
        st.write(f"King Safety: {breakdown['king_safety'] / 100:.2f}")
        st.write(f"Mobility: {breakdown['mobility'] / 100:.2f}")
        st.write(f"Center Control: {breakdown['center_control'] / 100:.2f}")
        st.write(f"Rook Placement: {breakdown['rook_placement'] / 100:.2f}")
        st.write(f"Bishop Pair: {breakdown['bishop_pair'] / 100:.2f}")
        st.write(f"Phase: {'Endgame' if breakdown['is_endgame'] else 'Middlegame'}")

# Main game area
col1, col2, col3 = st.columns([1, 2, 1])

# Evaluation bar (left)
with col1:
    st.subheader("Evaluation")
    
    eval_score = evaluate_board(st.session_state.board)
    eval_pct = get_eval_percentage(eval_score)
    
    # Create eval bar using progress bar
    white_pct = eval_pct
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="width: 40px; height: 300px; border: 2px solid #333; border-radius: 4px; 
                    background: linear-gradient(to top, #1a1a1a {100-white_pct}%, #f0f0f0 {100-white_pct}%);
                    position: relative;">
        </div>
        <div>
            <strong>{eval_score/100:+.2f}</strong><br>
            <small>{'White' if eval_score > 0 else 'Black' if eval_score < 0 else 'Equal'} 
            {'advantage' if eval_score != 0 else ''}</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Engine stats
    if st.session_state.last_engine_stats:
        stats = st.session_state.last_engine_stats
        st.markdown("---")
        st.caption("Last Search Stats")
        st.write(f"Depth: {stats.get('depth_reached', 0)}")
        st.write(f"Nodes: {stats.get('nodes', 0):,}")
        st.write(f"Time: {stats.get('time', 0):.2f}s")
        st.write(f"NPS: {stats.get('nps', 0):,}")

# Chess board (center)
with col2:
    board = st.session_state.board
    
    # Display game status
    if board.is_game_over():
        result = board.result()
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            st.success(f"🏆 Checkmate! {winner} wins!")
        elif board.is_stalemate():
            st.info("🤝 Stalemate - Draw!")
        elif board.is_insufficient_material():
            st.info("🤝 Insufficient material - Draw!")
        elif board.is_fifty_moves():
            st.info("🤝 Fifty-move rule - Draw!")
        elif board.is_repetition():
            st.info("🤝 Threefold repetition - Draw!")
    elif board.is_check():
        st.warning("⚠️ Check!")
    
    # Render board
    svg = render_board_svg(board, size=500)
    st.markdown(f'<div style="display: flex; justify-content: center;">{svg}</div>', 
                unsafe_allow_html=True)
    
    # Move input
    st.markdown("---")
    
    is_player_turn = (board.turn == st.session_state.player_color)
    
    if not board.is_game_over():
        if is_player_turn:
            st.markdown("**Your turn!** Enter your move:")
            
            col_a, col_b = st.columns([3, 1])
            
            with col_a:
                # Get legal moves for autocomplete
                legal_moves = [board.san(m) for m in board.legal_moves]
                move_input = st.selectbox(
                    "Select move",
                    options=[""] + sorted(legal_moves),
                    key="move_select",
                    label_visibility="collapsed"
                )
            
            with col_b:
                if st.button("Make Move", type="primary", use_container_width=True):
                    if move_input:
                        try:
                            move = board.parse_san(move_input)
                            make_move(move)
                            st.rerun()
                        except ValueError:
                            st.error("Invalid move!")
        else:
            st.markdown("**Engine's turn...**")
            if st.button("Let Engine Play", type="primary"):
                engine_move()
                st.rerun()

# Move history (right)
with col3:
    st.subheader("Move History")
    
    if st.session_state.move_history:
        moves_text = ""
        for i, move in enumerate(st.session_state.move_history):
            if i % 2 == 0:
                moves_text += f"{i//2 + 1}. {move} "
            else:
                moves_text += f"{move}\n"
        
        st.text_area("Moves", moves_text, height=300, disabled=True,
                     label_visibility="collapsed")
        
        # Undo button
        if st.button("↩️ Undo Last Move"):
            if st.session_state.board.move_stack:
                st.session_state.board.pop()
                st.session_state.move_history.pop()
                if st.session_state.evaluation_history:
                    st.session_state.evaluation_history.pop()
                st.rerun()
    else:
        st.info("No moves yet. Start playing!")
    
    st.divider()
    
    # FEN display
    st.subheader("Position (FEN)")
    st.code(board.fen(), language=None)
    
    # Load FEN
    with st.expander("Load Position"):
        fen_input = st.text_input("Enter FEN")
        if st.button("Load FEN"):
            try:
                new_board = chess.Board(fen_input)
                st.session_state.board = new_board
                st.session_state.move_history = []
                st.session_state.evaluation_history = []
                st.rerun()
            except ValueError:
                st.error("Invalid FEN!")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Chess Engine using Minimax + Alpha-Beta Pruning | Built for Foundations of AI</p>
    <p>Features: Iterative Deepening • Move Ordering • Quiescence Search • Transposition Tables</p>
</div>
""", unsafe_allow_html=True)

# Auto-play engine if it's engine's turn and game just started
if not board.is_game_over() and not is_player_turn and st.session_state.game_started:
    engine_move()
    st.rerun()

# Mark game as started after first render
if not st.session_state.game_started:
    st.session_state.game_started = True
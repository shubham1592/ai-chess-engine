"""
Streamlit Frontend for Chess Engine
Play against Heuristic AI or Neural Network AI
"""

import streamlit as st
import chess
import chess.svg
from engine import ChessEngine
from nn_engine import NNEngine
from evaluation import evaluate_board, get_evaluation_breakdown

# Page config
st.set_page_config(
    page_title="Chess Engine - AI Project",
    page_icon="♟️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    .move-history {
        font-family: monospace;
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        background: #f5f5f5;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()
if 'move_history' not in st.session_state:
    st.session_state.move_history = []
if 'player_color' not in st.session_state:
    st.session_state.player_color = chess.WHITE
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'last_engine_stats' not in st.session_state:
    st.session_state.last_engine_stats = None
if 'engine_type' not in st.session_state:
    st.session_state.engine_type = "heuristic"
if 'engine' not in st.session_state:
    st.session_state.engine = None


def get_engine():
    """Get or create the appropriate engine."""
    if st.session_state.engine is None:
        if st.session_state.engine_type == "heuristic":
            st.session_state.engine = ChessEngine(max_depth=4, time_limit=5.0)
        else:
            st.session_state.engine = NNEngine(
                weights_path="nn/weights.pth", 
                max_depth=3, 
                time_limit=5.0,
                device='cpu'
            )
    return st.session_state.engine


def reset_game():
    """Reset the game to initial state."""
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.game_started = False
    st.session_state.last_engine_stats = None
    if hasattr(st.session_state.engine, 'clear_transposition_table'):
        st.session_state.engine.clear_transposition_table()


def switch_engine(new_type):
    """Switch to a different engine type."""
    if st.session_state.engine_type != new_type:
        st.session_state.engine_type = new_type
        st.session_state.engine = None  # Force recreation
        reset_game()


def make_move(move):
    """Make a move on the board."""
    san = st.session_state.board.san(move)
    st.session_state.board.push(move)
    st.session_state.move_history.append(san)


def engine_move():
    """Make the engine's move."""
    if st.session_state.board.is_game_over():
        return
    
    engine = get_engine()
    
    with st.spinner("Engine thinking..."):
        if st.session_state.engine_type == "heuristic":
            move, score, depth, stats = engine.iterative_deepening(st.session_state.board)
            st.session_state.last_engine_stats = stats
        else:
            move, score, stats = engine.iterative_deepening(st.session_state.board)
            st.session_state.last_engine_stats = stats
    
    if move:
        make_move(move)


def get_eval_percentage(score):
    """Convert centipawn score to percentage for eval bar."""
    clamped = max(-1000, min(1000, score))
    percentage = 50 + (clamped / 20)
    return max(0, min(100, percentage))


def get_current_eval():
    """Get evaluation from current engine."""
    if st.session_state.engine_type == "heuristic":
        return evaluate_board(st.session_state.board)
    else:
        engine = get_engine()
        return engine.evaluate_position(st.session_state.board)


def render_board_svg(board, size=400):
    """Render the chess board as SVG."""
    flipped = st.session_state.player_color == chess.BLACK
    
    lastmove = None
    if board.move_stack:
        lastmove = board.peek()
    
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
st.title("♟️ Chess Engine — Foundations of AI")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Engine selection
    st.subheader("Select Engine")
    engine_choice = st.radio(
        "Play against:",
        ["Heuristic Engine (Phase 1)", "Neural Network Engine (Phase 2)"],
        index=0 if st.session_state.engine_type == "heuristic" else 1
    )
    
    new_type = "heuristic" if "Heuristic" in engine_choice else "neural"
    if new_type != st.session_state.engine_type:
        switch_engine(new_type)
        st.rerun()
    
    # Engine info box
    if st.session_state.engine_type == "heuristic":
        st.info("**Heuristic Engine**\n\nUses handcrafted evaluation:\n- Material counting\n- Piece-square tables\n- Pawn structure\n- King safety\n- Mobility")
    else:
        st.info("**Neural Network Engine**\n\nUses learned evaluation:\n- Trained on 400k positions\n- 3-layer MLP (512→256→128)\n- Learned from game outcomes")
    
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
    
    # Evaluation breakdown (only for heuristic)
    st.header("📊 Position Analysis")
    
    eval_score = get_current_eval()
    st.metric("Evaluation", f"{eval_score / 100:+.2f}")
    
    if st.session_state.engine_type == "heuristic":
        with st.expander("Detailed Breakdown"):
            breakdown = get_evaluation_breakdown(st.session_state.board)
            st.write(f"Material: {breakdown['material'] / 100:+.2f}")
            st.write(f"Piece Position: {breakdown['piece_position'] / 100:+.2f}")
            st.write(f"Pawn Structure: {breakdown['pawn_structure'] / 100:+.2f}")
            st.write(f"King Safety: {breakdown['king_safety'] / 100:+.2f}")
            st.write(f"Mobility: {breakdown['mobility'] / 100:+.2f}")
            st.write(f"Center Control: {breakdown['center_control'] / 100:+.2f}")
            st.write(f"Rook Placement: {breakdown['rook_placement'] / 100:+.2f}")
            st.write(f"Bishop Pair: {breakdown['bishop_pair'] / 100:+.2f}")

# Main game area
col1, col2, col3 = st.columns([1, 2, 1])

# Evaluation bar (left)
with col1:
    st.subheader("Evaluation")
    
    eval_score = get_current_eval()
    eval_pct = get_eval_percentage(eval_score)
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
        st.write(f"Depth: {stats.get('depth_reached', stats.get('depth', 0))}")
        st.write(f"Nodes: {stats.get('nodes', 0):,}")
        if 'nn_calls' in stats:
            st.write(f"NN Calls: {stats.get('nn_calls', 0):,}")
        st.write(f"Time: {stats.get('time', 0):.2f}s")

# Chess board (center)
with col2:
    board = st.session_state.board
    
    # Engine type indicator
    engine_name = "Heuristic" if st.session_state.engine_type == "heuristic" else "Neural Network"
    st.caption(f"Playing against: **{engine_name} Engine**")
    
    # Display game status
    if board.is_game_over():
        result = board.result()
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            st.success(f"🏆 Checkmate! {winner} wins!")
        elif board.is_stalemate():
            st.info("🤝 Stalemate — Draw!")
        elif board.is_insufficient_material():
            st.info("🤝 Insufficient material — Draw!")
        elif board.is_fifty_moves():
            st.info("🤝 Fifty-move rule — Draw!")
        elif board.is_repetition():
            st.info("🤝 Threefold repetition — Draw!")
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
            st.markdown("**Your turn!** Select your move:")
            
            col_a, col_b = st.columns([3, 1])
            
            with col_a:
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
                st.rerun()
            except ValueError:
                st.error("Invalid FEN!")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>Chess Engine — Foundations of AI Project</strong></p>
    <p>Phase 1: Minimax + Alpha-Beta + Heuristics | Phase 2: Neural Network Evaluation</p>
</div>
""", unsafe_allow_html=True)

# Auto-play engine if it's engine's turn and game just started
if not board.is_game_over() and not is_player_turn and st.session_state.game_started:
    engine_move()
    st.rerun()

if not st.session_state.game_started:
    st.session_state.game_started = True
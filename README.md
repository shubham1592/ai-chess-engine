# Chess Engine — Foundations of AI Project

A chess-playing AI that implements two different evaluation approaches: a classical heuristic-based engine and a neural network engine trained on Stockfish evaluations.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [How to Use](#how-to-use)
   - [Playing via Web Interface](#option-1-play-via-web-interface-recommended)
   - [Playing via Terminal](#option-2-play-via-terminal)
   - [Watching Engines Compete](#option-3-watch-engines-compete-against-each-other)
5. [How It Works](#how-it-works)
6. [Component Connections](#component-connections)
7. [Results](#results)
8. [Authors](#authors)

---

## Project Overview

This project contains two chess engines:

| Engine | Evaluation Method | Strength |
|--------|-------------------|----------|
| **Heuristic Engine** | Hand-crafted rules (material, position, pawn structure) | Stronger |
| **Neural Network Engine** | Trained MLP + material counting (70/30 hybrid) | Competitive |

Both engines use the same search algorithm (Minimax with Alpha-Beta pruning) but differ in how they evaluate positions.

---

## Project Structure

```
chess_engine/
│
├── engine.py              # Heuristic engine (Minimax + Alpha-Beta search)
├── evaluation.py          # Hand-crafted evaluation functions
├── move_ordering.py       # Move prioritization for efficient search
├── nn_engine.py           # Neural network engine (hybrid evaluation)
├── compare_engines.py     # Runs matches between both engines
├── app.py                 # Streamlit web interface
├── main.py                # Command-line interface
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── PROJECT_PLAN.md        # Original project planning document
│
├── data/
│   ├── games.pgn                # Raw chess games (Lichess Elite Database)
│   └── positions_stockfish.npz  # Training data with Stockfish labels
│
└── nn/
    ├── dataset.py              # Board encoding (position → 773-dim tensor)
    ├── model.py                # Neural network architecture (MLP)
    ├── train_stockfish.py      # Training script
    ├── generate_stockfish_labels.py  # Creates Stockfish-labeled dataset
    └── weights_stockfish.pth   # Trained model weights
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/shubham1592/ai-chess-engine.git
cd ai-chess-engine
```

### Step 2: Install Python Dependencies

Make sure you have Python 3.10 or higher installed.

```bash
pip install -r requirements.txt
```

This installs:
- `python-chess` — Chess rules and move generation
- `torch` — Neural network framework (PyTorch)
- `streamlit` — Web interface
- `numpy` — Numerical operations
- `tqdm` — Progress bars

### Step 3: Verify Installation

```bash
python -c "import chess; import torch; import streamlit; print('All dependencies installed!')"
```

---

## How to Use

### Option 1: Play via Web Interface (Recommended)

The web interface is the easiest way to play against the engine.

#### Step 1: Start the Streamlit Server

Open a terminal and run:

```bash
streamlit run app.py
```

#### Step 2: Open in Browser

The command will print a URL like:

```
Local URL: http://localhost:8501
```

Open this URL in your web browser (it usually opens automatically).

#### Step 3: Configure Your Game

In the **sidebar** on the left, you can:

1. **Select Engine**: Choose between:
   - "Heuristic Engine (Phase 1)" — Classical evaluation
   - "Neural Network Engine (Phase 2)" — ML-based evaluation

2. **Search Depth**: Use the slider to set how many moves ahead the engine thinks (1-5). Higher = stronger but slower.

3. **Play as**: Choose White or Black.

4. **New Game**: Click to reset the board.

#### Step 4: Make Moves

1. On your turn, select a move from the dropdown menu
2. Click "Make Move"
3. Click "Let Engine Play" when it's the engine's turn
4. The evaluation bar on the left shows who's winning

#### Step 5: Stop the Server

Press `Ctrl+C` in the terminal to stop the server.

---

### Option 2: Play via Terminal

If you prefer a text-based interface:

#### Step 1: Start the Program

```bash
python main.py
```

#### Step 2: Select Option 1

You'll see a menu:

```
==================================================
   CHESS ENGINE - Foundations of AI Project
==================================================

1. Play against the engine
2. Test engine vs baselines
3. Analyze a position
4. Run Streamlit GUI
5. Exit

Select option (1-5):
```

Type `1` and press Enter.

#### Step 3: Choose Your Color

```
Play as White or Black? (w/b):
```

Type `w` for White or `b` for Black.

#### Step 4: Play the Game

The board is displayed in ASCII:

```
  a b c d e f g h
 +-----------------+
8| r n b q k b n r |8
7| p p p p p p p p |7
6| . . . . . . . . |6
5| . . . . . . . . |5
4| . . . . . . . . |4
3| . . . . . . . . |3
2| P P P P P P P P |2
1| R N B Q K B N R |1
 +-----------------+
  a b c d e f g h

Evaluation: +0.32 (White is slightly better)

Your turn (White)
Legal moves: Na3 Nc3 Nf3 Nh3 a3 a4 b3 b4 c3 c4 d3 d4 e3 e4 f3 f4 g3 g4 h3 h4
Enter move:
```

Enter moves in standard algebraic notation:
- `e4` — Move pawn to e4
- `Nf3` — Move knight to f3
- `O-O` — Castle kingside
- `O-O-O` — Castle queenside
- `Bxc6` — Bishop captures on c6
- `e8=Q` — Pawn promotes to queen

Type `quit` to exit or `undo` to take back a move.

---

### Option 3: Watch Engines Compete Against Each Other

You can run an automated match between the heuristic engine and neural network engine.

#### Step 1: Run the Comparison Script

```bash
python compare_engines.py
```

#### Step 2: Watch the Match

The script will play 6 games (each engine plays 3 games as White):

```
============================================================
ENGINE MATCH: Heuristic vs Neural Network
============================================================
Games: 6
Depth: 3
Time limit: 5.0s per move
============================================================

GAME 1: Heuristic (White) vs Neural Net (Black)
============================================================
1. e4 Nh6
2. Qh5 d5
...
Result: 1-0 - Heuristic
Moves: 61, Time: 66.2s

Standings: Heuristic 1 - 0 - 0 Neural Net
```

#### Step 3: View Final Results

After all games:

```
============================================================
FINAL RESULTS
============================================================

Heuristic Engine: 5 wins
Neural Net Engine: 0 wins
Draws: 1

Score: Heuristic 5.5/6 vs Neural Net 0.5/6

🏆 WINNER: Heuristic Engine
```

#### Customize the Match

Edit `compare_engines.py` to change:

```python
run_match(num_games=6, depth=3, time_limit=5.0)
```

- `num_games`: Number of games to play
- `depth`: Search depth for both engines
- `time_limit`: Maximum seconds per move

---

## How It Works

### Search Algorithm (Both Engines)

Both engines use the same search:

1. **Minimax**: Explores all possible moves, assuming both players play optimally
2. **Alpha-Beta Pruning**: Skips branches that can't affect the result (huge speedup)
3. **Iterative Deepening**: Searches depth 1, then 2, then 3... until time runs out
4. **Move Ordering**: Searches best-looking moves first (checks → captures → threats)
5. **Quiescence Search**: Extends search in tactical positions to avoid blunders

### Heuristic Engine Evaluation

Scores positions based on hand-crafted rules:

| Factor | Description |
|--------|-------------|
| Material | Piece values (Q=9, R=5, B=3.3, N=3.2, P=1) |
| Piece-Square Tables | Bonus for good piece placement (knights in center, etc.) |
| Pawn Structure | Penalties for doubled/isolated pawns, bonus for passed pawns |
| King Safety | Bonus for castling and pawn shield |
| Mobility | More legal moves = better |
| Rook Placement | Bonus for open files and 7th rank |

### Neural Network Engine Evaluation

Uses a trained neural network:

1. **Input**: Board converted to 773 binary features
2. **Network**: 3-layer MLP (512 → 256 → 128 neurons)
3. **Output**: Score between -1 (Black winning) and +1 (White winning)
4. **Hybrid**: Final score = 70% NN + 30% material counting

The NN was trained on 300,000 positions labeled by Stockfish.

---

## Component Connections

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│                                                             │
│     app.py (Streamlit)         main.py (Console)           │
│           │                          │                      │
└───────────┼──────────────────────────┼──────────────────────┘
            │                          │
            ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      ENGINE LAYER                           │
│                                                             │
│     engine.py                    nn_engine.py               │
│   (Heuristic Engine)          (Neural Net Engine)           │
│          │                           │                      │
│          ▼                           ▼                      │
│   evaluation.py              nn/model.py                    │
│   (Hand-crafted             (Trained MLP)                   │
│    heuristics)                       │                      │
│          │                           │                      │
│          └───────────┬───────────────┘                      │
│                      ▼                                      │
│              move_ordering.py                               │
│           (Move prioritization)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                    CHESS LIBRARY                            │
│                                                             │
│                    python-chess                             │
│        (Board state, move generation, game rules)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User makes a move** → Interface passes it to the engine
2. **Engine searches** → Uses Minimax to explore possible moves
3. **Positions evaluated** → Heuristic or NN scores each position
4. **Best move found** → Returned to interface
5. **Board updated** → User sees the engine's response

---

## Results

### Engine Match (6 Games)

| Configuration | Wins | Draws | Losses | Score |
|---------------|------|-------|--------|-------|
| NN (game outcome labels) | 0 | 0 | 6 | 0.0/6 |
| NN (Stockfish labels) | 0 | 0 | 6 | 0.0/6 |
| NN + Material hybrid (70/30) | 0 | 3 | 3 | **1.5/6** |

### Game-by-Game Results (Final Configuration)

| Game | White | Black | Result | Moves |
|------|-------|-------|--------|-------|
| 1 | Heuristic | Neural Net | Draw | 160 |
| 2 | Neural Net | Heuristic | 0-1 | 48 |
| 3 | Heuristic | Neural Net | Draw | 160 |
| 4 | Neural Net | Heuristic | 0-1 | 54 |
| 5 | Heuristic | Neural Net | Draw | 160 |
| 6 | Neural Net | Heuristic | 0-1 | 54 |

**Key Observation:** The NN holds draws as Black (defensive play) but loses as White (weaker opening play).

### Key Findings

1. **Label quality matters**: Stockfish labels reduced validation loss from 0.73 to 0.09
2. **Hybrid evaluation helps**: Combining NN with material counting achieved 3 draws
3. **Heuristics still win**: With limited data (300k positions), hand-tuned rules outperform learned evaluation
4. **NN defends better than attacks**: Draws all games as Black, loses all as White

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'chess'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "Streamlit: command not found"

Make sure Streamlit is installed and in your PATH:
```bash
pip install streamlit
python -m streamlit run app.py
```

### "CUDA error" when running NN engine

The NN engine defaults to CPU. If you see CUDA errors, the code already handles this by using `device='cpu'`.

### Engine plays slowly

Reduce the search depth in the sidebar (web) or edit `max_depth` in the code.

---

## Authors

Shubham Kumar and Amrita Singh
Foundations of Artificial Intelligence  
Northeastern University  
Spring 2026

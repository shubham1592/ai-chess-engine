# Chess Engine Project: Complete Documentation
## Foundations of Artificial Intelligence

**Authors:** Shubham Kumar & Partner  
**Course:** Foundations of AI (Professor Rose Sloan)  
**University:** Northeastern University  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1: Heuristic Engine](#2-phase-1-heuristic-engine)
3. [Phase 2: Neural Network Engine](#3-phase-2-neural-network-engine)
4. [Phase 2 Improvements](#4-phase-2-improvements)
5. [Experimental Results](#5-experimental-results)
6. [Key Findings](#6-key-findings)
7. [Technical Implementation](#7-technical-implementation)
8. [Future Scope](#8-future-scope)
9. [Conclusion](#9-conclusion)

---

## 1. Project Overview

### 1.1 Objective

This project implements and compares two chess-playing AI agents:

1. **Heuristic Engine (Phase 1):** Classical AI using handcrafted evaluation functions with Minimax search
2. **Neural Network Engine (Phase 2):** Machine learning approach using a trained neural network for position evaluation

Both engines share the same search algorithm (Minimax with Alpha-Beta pruning) but differ fundamentally in how they evaluate chess positions. The project explores whether a neural network trained on expert-labeled positions can learn to evaluate chess positions as well as hand-tuned heuristics.

### 1.2 Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Chess Library | python-chess |
| ML Framework | PyTorch |
| Evaluation Engine | Stockfish 17 (depth 8) |
| Training Data | Lichess Elite Database (Dec 2022) |
| Frontend | Streamlit |
| Hardware | Ryzen 7 4000, NVIDIA RTX 3050, 8GB RAM |

### 1.3 Project Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | Weeks 1–4 | Heuristic engine with Minimax + Alpha-Beta |
| Phase 2a | Week 5 | NN trained on game outcomes (failed) |
| Phase 2b | Week 6 | NN retrained on Stockfish labels (improved) |
| Phase 2c | Weeks 7–8 | Hybrid evaluation experiments and final testing |

---

## 2. Phase 1: Heuristic Engine

### 2.1 Search Algorithm

We implemented Minimax with Alpha-Beta Pruning as the core adversarial search algorithm. The Minimax algorithm explores the game tree assuming both players play optimally — the maximizing player (White) tries to maximize the evaluation score while the minimizing player (Black) tries to minimize it. Alpha-Beta pruning eliminates branches that cannot affect the final decision, reducing the effective branching factor from ~35 to roughly its square root in the best case.

On top of this, we added several optimizations. Iterative deepening searches progressively deeper (depth 1, then 2, then 3, etc.) and returns the best move found when time expires. This guarantees we always have a move ready and results from shallower depths improve move ordering at deeper levels. Move ordering prioritizes checks, then captures ordered by MVV-LVA (Most Valuable Victim – Least Valuable Attacker), then threats, and finally quiet moves. Better move ordering leads to more Alpha-Beta cutoffs, which means faster search. Quiescence search extends the evaluation at leaf nodes by continuing to search capture sequences until the position is "quiet," preventing the horizon effect where the engine misses a tactic just beyond its search depth. Finally, the transposition table caches evaluated positions so we never evaluate the same position twice when it is reached via different move orders.

### 2.2 Heuristic Evaluation Function

The heuristic evaluation function assigns a numerical score to any chess position. We built it from multiple factors, each capturing a different aspect of chess knowledge:

**Material balance** forms the foundation. We assign standard piece values: Pawn = 100, Knight = 320, Bishop = 330, Rook = 500, Queen = 900. The total material difference between White and Black gives a basic assessment of who is ahead.

**Piece-square tables** add positional awareness. Each piece type has a 64-square table of bonuses and penalties. For example, knights get bonuses for being centralized and penalties for being on the edge; pawns get bonuses for advancing; kings get bonuses for staying safely castled in the middlegame but centralizing in the endgame.

**Pawn structure** evaluation penalizes doubled pawns (two pawns on the same file) and isolated pawns (no friendly pawns on adjacent files), while rewarding passed pawns (no opposing pawns blocking their advance).

**King safety** gives bonuses for having castled and for maintaining a pawn shield in front of the king.

**Mobility** counts the number of legal moves available. More options generally means a better position.

**Center control** rewards occupying or attacking the four central squares (d4, d5, e4, e5), which is a fundamental chess principle.

**Rook placement** gives bonuses for rooks on open files (no pawns blocking) and on the 7th rank (where they attack the opponent's pawns).

**Bishop pair** adds a bonus when a player still has both bishops, since together they cover all 64 squares and become stronger in open positions.

---

## 3. Phase 2: Neural Network Engine

### 3.1 Motivation

While the heuristic engine plays solid chess, its evaluation is limited to what we explicitly program. A neural network can potentially learn patterns we did not think to encode — subtle positional concepts, piece coordination, long-term structural features, and other nuances of chess evaluation.

### 3.2 Architecture

We use a fully connected neural network (MLP) with the following structure:

- **Input layer:** 773 features (12 piece types × 64 squares = 768 binary features for piece placement, plus 1 feature for side to move, plus 4 for castling rights)
- **Hidden layer 1:** 512 neurons with ReLU activation and batch normalization
- **Hidden layer 2:** 256 neurons with ReLU activation and batch normalization
- **Hidden layer 3:** 128 neurons with ReLU activation and batch normalization
- **Output layer:** 1 neuron with tanh activation (output in range -1 to +1)

Dropout (0.3) is applied between layers for regularization.

### 3.3 Training Data

We extracted positions from the Lichess Elite Database (December 2022), which contains games from high-rated players (2400+ Elo). From these games, we sampled positions at regular intervals to get a diverse set of board states, yielding approximately 298,000 training positions.

### 3.4 First Attempt: Game Outcome Labels

Our first approach labeled each position with the eventual game result: +1 for White win, -1 for Black win, 0 for draw. This is simple but noisy — a position where White is completely winning can still be labeled -1 if White blunders later. The model trained with these labels achieved a validation loss of 0.73 and could not distinguish basic material advantages. In head-to-head testing, it scored 0/6 against the heuristic engine, losing every game.

### 3.5 Second Attempt: Stockfish Labels

We then used Stockfish 17 at depth 8 to evaluate each of the 298,000 positions individually. The centipawn score from Stockfish was converted to the -1 to +1 range using the formula tanh(centipawns / 400). This gives a smooth, position-specific label: a position where White is up a pawn gets ~+0.25, up a piece gets ~+0.65, and completely winning gets close to +1.0.

This change was dramatic. Validation loss dropped from 0.73 to 0.09 — an 8x improvement. The network could now correctly identify material advantages, positional imbalances, and relative piece activity. However, it still scored 0/6 against the heuristic engine because it would occasionally misjudge tactical positions.

---

## 4. Phase 2 Improvements

### 4.1 Hybrid Evaluation (70/30)

To address the NN's tactical blind spots, we combined it with a simple material counting function. The final evaluation is:

```
evaluation = 0.7 × NN_score + 0.3 × material_score
```

The NN score is converted from the tanh range (-1, +1) to centipawns by multiplying by 400. The material score uses the same piece values as the heuristic engine. This hybrid approach lets the NN contribute its learned positional understanding while the material component prevents obvious tactical blunders like hanging pieces.

### 4.2 Experiment: Adding Piece-Square Tables (60/20/20)

We also tried incorporating piece-square tables into the hybrid:

```
evaluation = 0.6 × NN_score + 0.2 × material_score + 0.2 × PST_score
```

The idea was that PST bonuses would improve the NN's opening play (encouraging central pawn moves, proper knight development, etc.). However, this actually hurt performance — the extra heuristic signal added noise that conflicted with the NN's learned patterns. We reverted to the 70/30 configuration.

### 4.3 Experiment: 50/50 Hybrid

We also tested a 50/50 split between NN and material. This gave more weight to tactical safety but reduced the NN's influence too much, and the results were worse than 70/30. The 70/30 ratio remained our best configuration.

---

## 5. Experimental Results

### 5.1 Training Metrics

| Metric | Game Outcome Labels | Stockfish Labels |
|--------|-------------------|------------------|
| Validation Loss | 0.73 | 0.09 |
| Training Positions | ~298,000 | ~298,000 |
| Epochs | 50 | 50 |
| Material Recognition | Failed | Passed |

### 5.2 Engine Match Progression

We ran all matches as 6-game series at depth 3 with a 5-second time limit per move, alternating colors.

| Configuration | Heuristic Wins | NN Wins | Draws | NN Score |
|---|---|---|---|---|
| NN with game outcome labels | 6 | 0 | 0 | 0.0/6 |
| NN with Stockfish labels (pure) | 6 | 0 | 0 | 0.0/6 |
| NN + Material hybrid (70/30) — Run 1 | 4 | 0 | 2 | 1.0/6 |
| NN + Material + PST (60/20/20) | 6 | 0 | 0 | 0.0/6 |
| NN + Material hybrid (70/30) — Run 2 | 3 | 0 | 3 | 1.5/6 |

### 5.3 Latest Match Results (Final Run)

| Game | White | Black | Result | Moves |
|------|-------|-------|--------|-------|
| 1 | Heuristic | Neural Net | 1/2–1/2 | 160 |
| 2 | Neural Net | Heuristic | 0–1 | 48 |
| 3 | Heuristic | Neural Net | 1/2–1/2 | 160 |
| 4 | Neural Net | Heuristic | 0–1 | 54 |
| 5 | Heuristic | Neural Net | 1/2–1/2 | 160 |
| 6 | Neural Net | Heuristic | 0–1 | 54 |

**Final score: Heuristic 4.5/6 vs Neural Net 1.5/6**

A clear pattern emerges: the NN engine draws all three games as Black but loses all three as White. As Black, it can defend reactively and hold positions for 160 moves. As White, it plays passive openings (1. e3) and fails to create initiative, allowing the heuristic engine to take over.

---

## 6. Key Findings

### 6.1 Label Quality Is the Single Biggest Factor

The most impactful change in the entire project was switching from game outcome labels to Stockfish position evaluations. Validation loss improved 8x (0.73 → 0.09), and the network went from being unable to tell which side has more material to correctly evaluating complex positions. This makes intuitive sense: game outcomes are noisy (a winning position can still lead to a loss if the player blunders later), while Stockfish labels give clean, position-specific ground truth.

### 6.2 Hybrid Evaluation Is Essential for Limited-Data NNs

A pure neural network evaluation, even with good training, occasionally misjudges tactical positions. Adding a 30% material counting component acts as a safety net that prevents the most obvious blunders. This is similar to how production engines like Stockfish NNUE combine learned and handcrafted features.

### 6.3 Simpler Hybrids Outperform Complex Ones

Adding piece-square tables to the hybrid (60/20/20) hurt performance. The NN had already learned some positional patterns from Stockfish labels, and the additional PST signal created conflicting evaluations. The simpler 70/30 NN + material approach was consistently the best.

### 6.4 Explicit Knowledge Wins With Limited Data

The heuristic engine encodes decades of chess theory — material values, positional principles, structural patterns — all hand-tuned by chess experts. Our NN learned from ~300,000 positions, which is tiny compared to the billions of positions used by production engines. With limited training data, explicit domain knowledge still wins.

### 6.5 The NN Defends Better Than It Attacks

The NN draws all games as Black but loses all games as White. Defensive play is reactive — respond to the opponent's threats, maintain material balance, avoid blunders. Attacking play requires initiative — choosing ambitious openings, creating imbalances, exploiting weaknesses. The NN learned to evaluate positions accurately enough to defend, but not well enough to generate winning chances.

---

## 7. Technical Implementation

### 7.1 Project Files

**engine.py** — The heuristic engine. Contains the Minimax search with Alpha-Beta pruning, iterative deepening loop, quiescence search, and transposition table. Calls `evaluation.py` for position scoring and `move_ordering.py` for ordering moves at each node.

**evaluation.py** — The handcrafted evaluation function. Computes material balance, piece-square table bonuses, pawn structure, king safety, mobility, center control, rook placement, and bishop pair bonuses. Returns a centipawn score from White's perspective.

**move_ordering.py** — Orders moves to maximize Alpha-Beta pruning efficiency. Priority: checks first, then captures sorted by MVV-LVA, then threatening moves, then quiet moves.

**nn_engine.py** — The neural network engine. Uses the same Minimax search as the heuristic engine but replaces the evaluation function with a hybrid of 70% NN output and 30% material counting. Loads the trained model from `nn/chess_nn.pth`.

**nn/model.py** — Defines the neural network architecture (773 → 512 → 256 → 128 → 1 MLP with batch normalization and dropout).

**nn/prepare_data.py** — Extracts positions from PGN files and labels them using Stockfish 17 at depth 8. Converts board positions to 773-dimensional binary feature vectors. Saves the dataset as a NumPy archive.

**nn/train.py** — Training script. Uses MSE loss, Adam optimizer with learning rate scheduling (ReduceLROnPlateau), 70/30 train/validation split, and trains for 50 epochs.

**app.py** — Streamlit web frontend. Provides a visual chessboard, move input, evaluation bar, engine selection (heuristic or NN), and move history.

**main.py** — Console interface for playing against either engine in the terminal.

**compare_engines.py** — Runs a head-to-head match between the two engines over a configurable number of games, alternating colors, and reports detailed results.

### 7.2 Data Pipeline

The data pipeline proceeds as follows: we start with raw games from the Lichess Elite Database (December 2022 PGN file), then `prepare_data.py` extracts positions at regular intervals from each game and labels each position using Stockfish 17 at depth 8. The centipawn scores are converted to the range (-1, +1) using tanh(cp/400). The resulting ~298,000 labeled positions are saved as a NumPy archive. The training script `train.py` loads this data, splits it 70/30 into training and validation sets, and trains the MLP for 50 epochs.

### 7.3 Known Issues and Resolutions

During development we encountered and resolved several technical issues. The `chess.between()` function in python-chess returns a `SquareSet` that cannot be directly iterated in some contexts — we fixed this by converting it appropriately. The `torch.load()` function requires `weights_only=True` and `map_location='cpu'` for safe loading without GPU. The `ReduceLROnPlateau` scheduler's `verbose=True` parameter was deprecated — we removed it. Large data files (`games.pgn`, `positions.npz`) needed to be excluded from Git via `.gitignore`, and we had to use forced resets when commits accidentally included oversized files.

---

## 8. Future Scope

Several directions could improve the neural network engine:

**More training data.** Our 298,000 positions are a tiny fraction of what production engines use. Training on millions of positions would give the NN more patterns to learn from and likely improve both its tactical and positional understanding.

**Better architecture.** A convolutional neural network (CNN) could capture spatial patterns on the board more naturally than our fully connected MLP. Transformer architectures could model long-range piece interactions.

**NNUE-style efficient updates.** Stockfish's NNUE architecture uses incrementally updatable networks that only recompute the parts of the network affected by each move, making evaluation much faster and allowing deeper search.

**Opening book.** A simple opening book would bypass the NN's weakest phase (the opening) and let it play from positions where its evaluation is more reliable.

**Self-play training.** Rather than learning from static Stockfish labels, the engine could improve through self-play reinforcement learning, similar to AlphaZero's approach.

**Endgame tablebases.** For positions with few pieces, tablebases provide perfect play and would eliminate any evaluation errors in simplified endgames.

---

## 9. Conclusion

We built two chess engines that share a search algorithm but differ in evaluation philosophy. The heuristic engine, encoding decades of human chess knowledge, is the stronger player — winning 3 and drawing 3 out of 6 games against the neural network engine. However, the NN engine's journey from scoring 0/6 (losing every game quickly) to 1.5/6 (drawing three competitive 160-move games) demonstrates the power of systematic experimentation.

The three most impactful decisions we made were switching to Stockfish position labels (8x training improvement), adopting a hybrid evaluation (first draws achieved), and keeping the hybrid simple (reverting PST additions that hurt performance). These findings align with broader lessons in machine learning: data quality matters more than model complexity, and combining learned and engineered features often outperforms either alone.

The heuristic engine's victory is not surprising — it has an enormous knowledge advantage. But the fact that a relatively simple neural network, trained on 300,000 positions on consumer hardware in under two hours, can hold the heuristic engine to draws in half its games suggests that with more data and compute, the gap would narrow significantly.

---

**GitHub Repository:** [github.com/shubham1592/ai-chess-engine](https://github.com/shubham1592/ai-chess-engine)

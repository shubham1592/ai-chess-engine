# Chess Engine Project Plan

## Project Overview

This project implements a high-performance chess engine utilizing a hybrid architecture that combines classical game-tree search with modern neural network-based evaluation. The system is designed to leverage the tactical precision of adversarial search algorithms while incorporating strategic positional understanding learned from expert-level human gameplay.

---

## Technical Architecture

### Phase 1: Classical Search Implementation

The core engine utilizes a Minimax algorithm enhanced by Alpha-Beta pruning and iterative deepening to manage computational complexity.

* **Search Depth**: Target depth of 4-5 plies with quiescence extensions
* **Heuristic Evaluation**: Board states are assessed using hand-crafted weights for material count (Pawn=100cp, Knight=320cp, Bishop=330cp, Rook=500cp, Queen=900cp) and positional factors such as king safety, pawn structure, and piece mobility
* **Optimization**: Efficiency is improved through move ordering, prioritizing checks, captures, and threats to maximize pruning
* **Transposition Table**: Caches evaluated positions to avoid redundant computation

### Phase 2: Neural Network Integration

A multi-layer perceptron replaces traditional hand-crafted evaluation functions.

* **Architecture**: 3-layer MLP (773 → 512 → 256 → 128 → 1) with BatchNorm, ReLU, and Dropout
* **Input Representation**: Board states are converted into 773-dimensional binary tensors encoding piece positions, side to move, and castling rights
* **Training Data**: The model is trained on ~300,000 positions from the Lichess Elite Database, labeled by Stockfish 17 at depth 8
* **Output**: The network produces a scalar evaluation score between -1 (Black winning) and +1 (White winning)

### Phase 3: Hybrid System Integration

The final engine integrates the trained PyTorch model as the primary evaluation function within the Alpha-Beta search framework. The hybrid evaluation combines:

* **70% Neural Network**: Provides learned positional understanding from Stockfish evaluations
* **30% Material Counting**: Prevents tactical blunders and ensures basic material awareness

---

## Project Deliverables

### Development Timeline

* **Month 1 (February)**: 
  - Week 1: Environment setup, baseline random and greedy movers
  - Week 2: Minimax algorithm with Alpha-Beta pruning
  - Week 3: Heuristic evaluation function (material, PST, pawn structure)
  - Week 4: Move ordering, iterative deepening, quiescence search

* **Month 2 (March)**: 
  - Week 5: Data pipeline setup, PGN parsing, position extraction
  - Week 6: Neural network architecture, initial training on game outcomes
  - Week 7: Stockfish label generation, improved training, hybrid integration
  - Week 8: Engine comparison, final testing, documentation

### Final Outputs

* **Playable Web Interface**: An interactive UI built with Streamlit featuring real-time evaluation visualization
* **Console Interface**: Command-line interface for playing and testing
* **Comparative Analysis**: Documentation and performance metrics comparing the neural network engine against the heuristic baseline
* **Source Code**: Full implementation in Python using python-chess for state management and PyTorch for deep learning

---

## Tools and Technologies

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Game Logic | python-chess |
| Deep Learning | PyTorch 2.10 |
| Position Evaluation | Stockfish 17 |
| Frontend | Streamlit |
| Database | Lichess Elite Database (PGN) |
| Hardware | Ryzen 7 4000, RTX 3050, 8GB RAM |

---

## Expected Results

| Configuration | Expected Outcome |
|---------------|------------------|
| Heuristic Engine vs Random | Heuristic wins consistently |
| Heuristic Engine vs Greedy | Heuristic wins most games |
| NN Engine vs Heuristic | Heuristic expected to win, NN competitive |

The heuristic engine is expected to outperform the neural network given limited training data (~300k positions vs billions in production engines), but the NN should demonstrate learned evaluation capabilities and achieve competitive games.

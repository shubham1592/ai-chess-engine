# Hybrid AI Chess Engine

## Project Overview
This project implements a high-performance chess engine utilizing a hybrid architecture that combines classical game-tree search with modern neural network-based evaluation. The system is designed to leverage the tactical precision of adversarial search algorithms while incorporating strategic positional understanding learned from expert-level human gameplay.

## Technical Architecture

### Phase 1: Classical Search Implementation
The core engine utilizes a Minimax algorithm enhanced by Alpha-Beta pruning and iterative deepening to manage computational complexity.
* **Search Depth**: Target depth of 5 to 6 plies.
* **Heuristic Evaluation**: Initial board states are assessed using hand-crafted weights for material count (Pawn=1, Knight=3, Rook=5) and positional factors such as king safety and piece connectivity.
* **Optimization**: Efficiency is improved through move ordering, prioritizing checks, captures, and threats to maximize pruning.



### Phase 2: Neural Network Integration
A recurrent neural network replaces traditional hand-crafted evaluation functions.
* **Input Representation**: Board states are converted into bitboard representations or tensors.
* **Training Data**: The model is trained on over 10,000 grandmaster games from the Lichess Elite Database (KingBase PGN format).
* **Output**: The network produces a scalar evaluation score representing positional strength.

### Phase 3: Hybrid System Integration
The final engine integrates the trained PyTorch model as the primary evaluation function within the Alpha-Beta search framework. To maintain performance, the engine utilizes the classical heuristic for preliminary rapid evaluation and reserves the neural network for critical positions.

## Project Deliverables

### Development Timeline
* **Month 1 (February)**: Establishment of environment, baseline implementation of Minimax with Alpha-Beta pruning, and integration of classical heuristics.
* **Month 2 (March)**: Development of the data pipeline, training the neural network on the KingBase dataset, and full integration of the hybrid system.

### Final Outputs
* **Playable Web Interface**: An interactive UI built with Streamlit featuring real-time win probability visualization.
* **Comparative Analysis**: Documentation and performance metrics comparing the Week 8 hybrid engine against the Week 3 heuristic baseline.
* **Source Code**: Full implementation in Python using python-chess for state management and PyTorch for deep learning.

## Tools and Technologies
* **Language**: Python
* **Game Logic**: python-chess
* **Deep Learning**: PyTorch
* **Frontend**: Streamlit
* **Database**: KingBase (Lichess Elite PGN)
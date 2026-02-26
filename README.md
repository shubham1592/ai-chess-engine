# Hybrid AI Chess Engine

## Project Overview
[cite_start]This project implements a high-performance chess engine utilizing a hybrid architecture that combines classical game-tree search with modern neural network-based evaluation[cite: 4]. [cite_start]The system is designed to leverage the tactical precision of adversarial search algorithms while incorporating strategic positional understanding learned from expert-level human gameplay[cite: 5].

## Technical Architecture

### Phase 1: Classical Search Implementation
[cite_start]The core engine utilizes a Minimax algorithm enhanced by Alpha-Beta pruning and iterative deepening to manage computational complexity[cite: 7].
* [cite_start]**Search Depth**: Target depth of 5 to 6 plies[cite: 8].
* [cite_start]**Heuristic Evaluation**: Initial board states are assessed using hand-crafted weights for material count (Pawn=1, Knight=3, Rook=5) and positional factors such as king safety and piece connectivity[cite: 8, 33, 34].
* [cite_start]**Optimization**: Efficiency is improved through move ordering, prioritizing checks, captures, and threats to maximize pruning[cite: 9, 39].

### Phase 2: Neural Network Integration
[cite_start]A recurrent neural network replaces traditional hand-crafted evaluation functions[cite: 10].
* [cite_start]**Input Representation**: Board states are converted into bitboard representations or tensors[cite: 11, 53].
* [cite_start]**Training Data**: The model is trained on over 10,000 grandmaster games from the Lichess Elite Database (KingBase PGN format)[cite: 12, 52].
* [cite_start]**Output**: The network produces a scalar evaluation score representing positional strength[cite: 11, 58].

### Phase 3: Hybrid System Integration
[cite_start]The final engine integrates the trained PyTorch model as the primary evaluation function within the Alpha-Beta search framework[cite: 13]. [cite_start]To maintain performance, the engine utilizes the classical heuristic for preliminary rapid evaluation and reserves the neural network for critical positions[cite: 14, 63].

## Project Deliverables

### Development Timeline
* [cite_start]**Month 1 (February)**: Establishment of environment, baseline implementation of Minimax with Alpha-Beta pruning, and integration of classical heuristics[cite: 24, 25, 29, 32].
* [cite_start]**Month 2 (March)**: Development of the data pipeline, training the neural network on the KingBase dataset, and full integration of the hybrid system[cite: 49, 51, 61].

### Final Outputs
* [cite_start]**Playable Web Interface**: An interactive UI built with Streamlit featuring real-time win probability visualization[cite: 17, 21, 64].
* [cite_start]**Comparative Analysis**: Documentation and performance metrics comparing the Week 8 hybrid engine against the Week 3 heuristic baseline[cite: 19, 21, 64].
* [cite_start]**Source Code**: Full implementation in Python using python-chess for state management and PyTorch for deep learning[cite: 16, 17].

## Tools and Technologies
* [cite_start]**Language**: Python[cite: 16, 40].
* [cite_start]**Game Logic**: python-chess[cite: 17, 40].
* [cite_start]**Deep Learning**: PyTorch[cite: 17].
* [cite_start]**Frontend**: Streamlit[cite: 17, 40].
* [cite_start]**Database**: KingBase (Lichess Elite PGN)[cite: 17, 48].
# Zero-Day Attack Detection with Reinforcement Learning

This repository contains implementations of Q-Learning and Deep Q-Network (DQN) approaches for network intrusion detection, with a special focus on detecting previously unseen "zero-day" attacks.

![Network Security](https://img.shields.io/badge/Network-Security-blue)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement-Learning-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

## Research Overview

This project investigates how reinforcement learning techniques can be applied to detect network anomalies, particularly zero-day attacks. Two distinct approaches are implemented and compared:

1. **Traditional Q-Learning**: A tabular reinforcement learning approach with state discretization
2. **Deep Q-Networks (DQN)**: A deep learning-based approach for handling continuous state spaces

The research specifically addresses the challenge of detecting attack types ("Fuzzers" and "Reconnaissance") that are deliberately excluded from the training data but present in the testing data, simulating zero-day attack scenarios.

## Dataset

The project uses the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) which contains a mixture of normal network traffic and various attack types. The data preparation process involves:

1. Removing "Fuzzers" and "Reconnaissance" attack categories from the training set
2. Creating a specialized zero-day evaluation dataset containing only these attack types plus normal traffic
3. Applying preprocessing techniques including normalization, feature selection, and class balancing

## Key Features

### Common to Both Approaches
- Data preprocessing pipeline with feature selection, scaling, and class balancing
- State representation using clustering for Q-learning and raw features for DQN
- Systematic evaluation of 5 hyperparameter schemes per approach
- Zero-day detection performance metrics

### Q-Learning Implementation
- Tabular Q-learning with state discretization via clustering
- Epsilon-greedy exploration strategy with decay
- Custom reward function for anomaly detection

### DQN Implementation
- Neural network-based Q-function approximation
- Experience replay for stable learning
- Target network for reducing overestimation bias
- Advanced exploration strategies
- Class-weighted loss function to prioritize attack detection

## Results and Findings

The repository includes visualization tools to compare the performance of both approaches:
- Confusion matrices for all schemes
- ROC and precision-recall curves
- Training vs. test accuracy plots to detect overfitting
- Comparative analysis of detection rates for known vs. zero-day attacks

Key metrics tracked:
- Detection rate (recall)
- False positive rate
- Precision
- F1 score
- Area Under ROC Curve (AUC)

## Usage

### Prerequisites
pandas numpy scikit-learn matplotlib seaborn torch (for DQN) imbalanced-learn joblib

### Running the Notebooks
1. Prepare the UNSW-NB15 dataset files (`UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`)
2. Run the preprocessing cells in either notebook to generate the processed datasets
3. Execute the model training and evaluation cells
4. View the generated visualizations and performance metrics

## Methodology Overview

Both approaches follow this general workflow:
1. Dataset preparation and preprocessing 
2. Environment creation for reinforcement learning
3. Agent training with different hyperparameter schemes
4. Evaluation on regular test data and zero-day evaluation data
5. Performance comparison and visualization

## Repository Structure

- **dqn pake schema overfitt (2) (1).ipynb**: Implementation of the Deep Q-Network approach
- **qlearning 5 skema.ipynb**: Implementation of the traditional Q-Learning approach
- **README.md**: This documentation file
- **requirements.txt**: List of required Python packages

## Hyperparameter Optimization

Both implementations evaluate 5 different hyperparameter schemes:

### Q-Learning Schemes
- Learning rate (alpha)
- Discount factor (gamma)
- Exploration rate (epsilon) and its decay parameters

### DQN Schemes
- Learning rate
- Epsilon decay strategies
- Batch size
- Replay buffer size

## Citation

If you use this code for academic research, please cite:
@misc{ZeroDayAttackRL, author = {Your Name}, title = {Zero-Day Attack Detection with Reinforcement Learning}, year = {2025}, publisher = {GitHub}, url = {https://github.com/yourusername/zero-day-attack-detection} }


## License

This project is licensed under the MIT License - see the LICENSE file for details.

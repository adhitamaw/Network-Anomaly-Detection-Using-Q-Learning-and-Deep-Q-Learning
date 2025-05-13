# Network Intrusion Detection with Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-green)](https://scikit-learn.org/)

## Zero-Day Attack Detection Research using Q-Learning and Deep Q-Network

This repository contains the implementation and comparison of two reinforcement learning algorithms for network intrusion detection, with a specific focus on the ability to detect zero-day attacks.

## 📋 Research Description

This research develops and compares two reinforcement learning approaches for network intrusion detection:
- **Q-Learning**: A simple yet efficient tabular algorithm
- **Deep Q-Network (DQN)**: A more complex neural network-based algorithm

Both algorithms are trained to detect normal attacks and evaluated on their ability to detect zero-day attacks that were not seen during training. We use the UNSW-NB15 dataset with Fuzzers and Reconnaissance attack categories as a simulation of zero-day attacks.

## 🎯 Research Objectives

1. Develop a reinforcement learning-based network intrusion detection model
2. Evaluate and compare the performance of Q-Learning and DQN in detecting:
   - Recognized standard attacks
   - Previously unseen zero-day attacks
3. Optimize parameters to achieve the best detection performance
4. Analyze the generalization capabilities of both algorithms

## 📊 Dataset

This research uses the **UNSW-NB15** dataset containing normal network traffic data and various types of attacks. The dataset is divided into:
- **Training data**: Common attacks (without Fuzzers and Reconnaissance)
- **Testing data**: All types of attacks
- **Zero-day evaluation data**: Only Fuzzers and Reconnaissance attacks + normal traffic

## 📈 Research Results

### Q-Learning
- Best model: **Scheme 1** (alpha=0.01, gamma=0.8, epsilon=1.0, epsilon_decay=0.8, epsilon_min=0.05)
- Standard test data accuracy: 92.43%
- Zero-day data accuracy: 88.91%
- Zero-day data detection rate (recall): 69.29%

### Deep Q-Network (DQN)
- Best model: **Scheme 1** (learning_rate=0.0001, epsilon_decay=0.93, batch_size=32, buffer_size=50000)
- Standard test data accuracy: 99.76%
- Zero-day data accuracy: 99.58%
- Zero-day data detection rate (recall): 99.34%

### Comparison
DQN shows significantly better performance than Q-Learning, especially in detecting zero-day attacks with a detection rate difference of about 30%. However, Q-Learning has advantages in terms of computational efficiency and lower memory requirements.

## 🔬 Methodology

Both algorithms use the same techniques for data preprocessing:
1. Feature cleaning and normalization
2. Handling missing values
3. Feature selection
4. Class balancing using SMOTE+Tomek

The main differences are in the model architecture:
- Q-Learning: Tabular approach with state discretization using K-Means clustering
- DQN: Neural network with 2 hidden layers (64 and 32 neurons)

## 🙏 Acknowledgements

- UNSW-NB15 Dataset: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

## 📧 Contact

Please contact the author for questions or research collaborations.

---

© 2025. All rights reserved.

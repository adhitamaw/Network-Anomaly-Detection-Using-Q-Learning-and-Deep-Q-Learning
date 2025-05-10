# Zero-Day Attack Detection with Reinforcement Learning

This repository implements and compares traditional Q-Learning and Deep Q-Network (DQN) approaches for network intrusion detection, with a special focus on detecting previously unseen "zero-day" attacks.

![Network Security](https://img.shields.io/badge/Network-Security-blue)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement-Learning-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

## Research Overview

This project investigates how reinforcement learning techniques can be applied to detect network anomalies, particularly zero-day attacks. Two distinct approaches are implemented and compared:

1. **Traditional Q-Learning**: A tabular reinforcement learning approach with state discretization via clustering
2. **Deep Q-Networks (DQN)**: A deep learning-based approach for handling continuous state spaces

The research specifically addresses the challenge of detecting attack types ("Fuzzers" and "Reconnaissance") that are deliberately excluded from the training data but present in the testing data, simulating zero-day attack scenarios.

## Dataset

The project uses the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) which contains a mixture of normal network traffic and various attack types. The data preparation process involves:

1. Removing "Fuzzers" and "Reconnaissance" attack categories from the training set
2. Creating a specialized zero-day evaluation dataset containing only these attack types plus normal traffic
3. Applying preprocessing techniques including normalization, feature selection, and class balancing

## Implementation Details

### Data Preprocessing
- Handling missing values and duplicates
- Feature encoding for categorical variables
- Class imbalance handling with SMOTE, Random Undersampling, and SMOTETomek
- Feature selection with SelectKBest and removal of highly correlated features
- Creation of separate datasets for regular testing and zero-day evaluation

### Q-Learning Implementation
- Environment representation with state discretization via clustering (MiniBatchKMeans)
- Simple reward structure for correct/incorrect classifications
- Q-table initialization and update with standard Q-Learning formula
- Epsilon-greedy policy with decay for balancing exploration and exploitation
- Systematic evaluation of 5 hyperparameter schemes

### DQN Implementation
- Neural network architecture for Q-function approximation
- Experience replay buffer for improved stability
- Target network to reduce overestimation bias
- Enhanced reward structure to prioritize attack detection
- Class-weighted loss function
- Gradient clipping to prevent exploding gradients
- Automatic model checkpoint and early stopping

## Hyperparameter Schemes

Each approach evaluates 5 different hyperparameter schemes:

### Q-Learning Schemes
Scheme 1: {'alpha': 0.01, 'gamma': 0.8, 'epsilon': 1.0, 'epsilon_decay': 0.8, 'epsilon_min': 0.05} Scheme 2: {'alpha': 0.1, 'gamma': 0.95, 'epsilon': 1.0, 'epsilon_decay': 0.95, 'epsilon_min': 0.2} Scheme 3: {'alpha': 0.03, 'gamma': 0.99, 'epsilon': 0.8, 'epsilon_decay': 0.99, 'epsilon_min': 0.3} Scheme 4: {'alpha': 0.5, 'gamma': 0.9, 'epsilon': 0.7, 'epsilon_decay': 0.9, 'epsilon_min': 0.1} Scheme 5: {'alpha': 0.2, 'gamma': 0.7, 'epsilon': 0.9, 'epsilon_decay': 0.85, 'epsilon_min': 0.01}

### DQN Schemes
Scheme 1: {'learning_rate': 0.0001, 'epsilon_decay': 0.93, 'batch_size': 32, 'buffer_size': 50000} Scheme 2: {'learning_rate': 0.0003, 'epsilon_decay': 0.95, 'batch_size': 64, 'buffer_size': 50000} Scheme 3: {'learning_rate': 0.0005, 'epsilon_decay': 0.97, 'batch_size': 128, 'buffer_size': 50000} Scheme 4: {'learning_rate': 0.001, 'epsilon_decay': 0.98, 'batch_size': 256, 'buffer_size': 50000} Scheme 5: {'learning_rate': 0.003, 'epsilon_decay': 0.99, 'batch_size': 512, 'buffer_size': 50000}


## Evaluation Metrics

Both approaches are evaluated using comprehensive metrics, focusing on their ability to detect zero-day attacks:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of true positives among positive predictions
- **Recall (Detection Rate)**: Proportion of true positives identified correctly
- **F1-Score**: Harmonic mean of precision and recall
- **False Positive Rate (FPR)**: Proportion of false positives among negative instances
- **Area Under ROC Curve (AUC)**: Classification performance across thresholds

## Results and Visualizations

The project includes extensive visualizations to analyze model performance:

1. **Confusion Matrices**: Comparing prediction performance across schemes for both test and zero-day scenarios
2. **Training vs. Test Accuracy**: Monitoring overfitting during training
3. **ROC and Precision-Recall Curves**: Evaluating classification performance across thresholds
4. **Hyperparameter Comparison**: Analyzing the impact of different parameter settings
5. **Q-Learning vs. DQN Comparison**: Contrasting both approaches on key metrics

## Repository Structure

- **dqn pake schema overfitt (2) (1).ipynb**: Implementation of the Deep Q-Network approach
- **qlearning 5 skema.ipynb**: Implementation of the traditional Q-Learning approach
- **README.md**: This documentation file
- **requirements.txt**: List of required Python packages

## Usage

### Prerequisites
pandas numpy scikit-learn matplotlib seaborn torch tqdm joblib imbalanced-learn


### Running the Notebooks

1. Download the UNSW-NB15 dataset files (`UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`)
2. Place the dataset files in the same directory as the notebooks
3. Run the preprocessing cells in either notebook to generate the processed datasets
4. Execute the model training and evaluation cells
5. View the generated visualizations and performance metrics

## Key Findings

- Both Q-Learning and DQN approaches can detect zero-day attacks with varying levels of success
- The addition of class-weighted loss functions and enhanced reward structures significantly improves zero-day detection
- State representation via clustering provides effective discretization for Q-Learning
- Hyperparameters significantly impact the model's ability to generalize to unseen attack types
- Trade-offs exist between detection rate and false positive rate across different schemes

## Citation

If you use this code for academic research, please cite:
@misc{ZeroDayAttackRL, author = {Your Name}, title = {Zero-Day Attack Detection with Reinforcement Learning}, year = {2023}, publisher = {GitHub}, journal = {GitHub repository}, howpublished = {\url{https://github.com/yourusername/zero-day-attack-detection}} }


## License

This project is licensed under the MIT License - see the LICENSE file for details.

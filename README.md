# Marketing Analytics A/B Testing

This repository contains an implementation of a multi-armed bandit problem with different algorithms designed for marketing analytics and A/B testing experiments. It simulates multiple bandit arms, each representing different variants of a marketing campaign, to determine the optimal strategy for maximizing the reward (e.g., conversions, engagement) over time. The following algorithms are implemented:

- **Epsilon-Greedy**
- **Thompson Sampling**

The results of the experiments are visualized using `matplotlib`, showing the comparison between different strategies and their effectiveness over time.
The results of the experiments are reported in a jupyter notebook file and csv files.

## Design of Experiment

### Bandit Options
- **Advertisement Rewards (Bandit_Reward):** [1, 2, 3, 4]
- **Number of Trials:** 20,000

### Goals
1. Implement a `Bandit` class as an abstract base class.
2. Create `EpsilonGreedy` and `ThompsonSampling` classes that inherit from `Bandit`.
3. Perform experiments to evaluate each algorithm.

## Implementation Details

### Classes
1. **Bandit Class**
   - Abstract base class defining required methods: `pull`, `update`, `experiment`, and `report`.
   - Serves as a blueprint for bandit algorithms.

2. **EpsilonGreedy Class**
   - Implements an epsilon-decay strategy where ε decays as 1/t.
   - Balances exploration and exploitation during trials.

3. **ThompsonSampling Class**
   - Uses Bayesian principles with known precision to sample rewards.
   - Dynamically adjusts exploration based on posterior distributions.

### Experiment Requirements
- Visualize learning processes for each algorithm.
- Compare cumulative rewards and regrets for Epsilon-Greedy and Thompson Sampling.
- Store results in a CSV file with columns: `{Bandit, Reward, Algorithm}`.
- Print:
  - Cumulative Reward
  - Cumulative Regret

## Installation

### Prerequisites
Ensure the following tools are installed on your system:
- **Python 3.8+**
- **Required Libraries:** Install via `pip`:
  ```bash
  pip install -r requirements.txt

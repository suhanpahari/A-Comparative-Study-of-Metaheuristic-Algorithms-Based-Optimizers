# A Comparative Study of Metaheuristic Algorithm-Based Optimizers

In this research project, I aim to find the most effective metaheuristic model for capturing non-linear patterns in time series analysis. To achieve this, I evaluated and compared five distinct optimization algorithms:

1. **Dung Beetle Algorithm (DBO)-based Optimizer**
2. **Genetic Algorithm (GA)-based Optimizer**
3. **Particle Swarm Optimization (PSO)-based Optimizer**
4. **Gravitational Search Algorithm (GSA)-based Optimizer**
5. **Red Deer Algorithm (RDA)-based Optimizer**

## Project Overview

Using these algorithms, I developed a hybrid model that combines features from each approach, creating a more robust version. The mathematical formulation behind my custom algorithm captures each optimizer's strengths, resulting in superior performance in time series forecasting tasks compared to the individual algorithms.

While my hybrid algorithm outperformed the standalone algorithms mentioned above, an approach called **DBO-CNN** yielded even better results in certain scenarios.

## Forecasting and Comparison Results

Here, I showcase my forecasting results and performance comparisons, highlighting the strengths and weaknesses of each algorithm. By evaluating metrics like Mean Squared Error (MSE) and R² Score, this project provides insights into which models perform best for specific types of time series patterns.

### Key Components

- **Mathematical Details**: An in-depth explanation of the hybrid algorithm's mathematical formulation.
- **Evaluation Metrics**: Comparison of forecasting performance using standard metrics such as MSE, MAE, and R² Score.
- **Algorithmic Breakdown**: Brief summaries of the individual metaheuristic algorithms and their unique properties.
- **Results Visualization**: Graphical representation of the forecasting results for each algorithm and the hybrid model.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tensorflow`, `keras`, and any additional libraries listed in `requirements.txt`.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/metaheuristic-optimizers-study.git
   cd metaheuristic-optimizers-study

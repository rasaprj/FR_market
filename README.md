# Developing Hierarchical Control Strategy via Model Predictive Control and Reinforcement Learning for Battery Frequency Regulation Market

### MSc Thesis – Imperial College London

---

## 📖 Project Overview

This repository supports the MSc thesis:
**“Developing Hierarchical Control Strategy via Model Predictive Control and Reinforcement Learning for Battery Frequency Regulation Market”**

The project develops and evaluates a **multi-stage hierarchical control framework** for battery energy storage systems (BESS) participating in frequency regulation (FR) markets. The framework integrates **Model Predictive Control (MPC)** and **Reinforcement Learning (RL)** to balance short-term revenue maximization with long-term degradation minimization.

---

## ⚙️ Methodology Summary

The workflow follows several stages:

1. **Low-Fidelity MPC Simulation:** Generates state–action pairs to form the dataset for supervised learning.
2. **Supervised Learning Pretraining:** An actor–critic deep neural network is initialized using MPC-generated data.

3. **Reinforcement Learning Pretraining:** The network is refined in a high-fidelity, degradation-aware environment.

4. **Anchor-Based Adaptation:** A degradation-unaware RL policy is trained with setpoint tracking capability.

5. **Hierarchical MPC–RL Integration:** The trained RL agent is embedded into a hierarchical control scheme, where MPC provides high-level constraints and RL executes fast revenue-driven actions.

*The overall workflow is illustrated in Figure 1.*

---

## Repository Structure

```
├── MPC/              # Low-fidelity MPC to generate state–action pairs
├── DNN/              # Supervised learning (SL) for pretraining actor–critic network
├── RL_Pretrain/      # Pretraining of degradation-aware RL agent
├── RL_Pretrain_8dim/ # Pretraining of degradation-unaware RL + setpoint tracking
├── MPC-RL/           # Hierarchical control scheme (MPC + RL in high-fidelity env)
└── Data/             # PJM dataset (FR signals, FR prices, DAM energy prices)
---

## Requirements

* Julia ≥ 1.9
* [JuMP.jl](https://jump.dev/) (optimization modeling)
* [Ipopt](https://coin-or.github.io/Ipopt/) or [Gurobi](https://www.gurobi.com/) (solvers)
* [Flux.jl](https://fluxml.ai/) (neural networks)
* [CSV.jl](https://csv.juliadata.org/stable/) and [DataFrames.jl](https://dataframes.juliadata.org/stable/) (data handling)



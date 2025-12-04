#  Hierarchical Control  via MPC-RL for for Multi-Timescale Battery Systems

---

## ⚙️ Methodology Summary

The workflow follows several stages:

1. **Low-Fidelity MPC Simulation:** Generates state–action pairs to form the dataset for supervised learning.
2. 
3. **Behavior Cloning:** An actor–critic deep neural network is initialized using MPC-generated data.

4. **Reinforcement Learning Pretraining:** The network is refined in a high-fidelity, degradation-aware environment.

5. **Anchor-Based Adaptation:** A degradation-unaware RL policy is trained with setpoint tracking capability.

6. **Hierarchical MPC–RL Integration:** The trained RL agent is embedded into a hierarchical control scheme, where MPC provides high-level constraints and RL executes fast revenue-driven actions.
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


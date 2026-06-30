# GNN-Guided Ant Colony Optimization (GNN-Guided ACO)

<div align="center">

### A Hybrid ML–Metaheuristic Framework for Accelerated NP-Hard Combinatorial Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-PyG-6E40C9?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

</div>

---

## 📌 Overview & Scientific Motivation

Classical **Ant Colony Optimization (ACO)** is highly effective for combinatorial optimization, yet it suffers from a known limitation at scale:  
it initializes pheromone trails and heuristic priors with limited structural knowledge, causing **slow convergence** and unnecessary computational expenditure on low-value regions of the search space.

**GNN-Guided ACO** addresses this inefficiency by coupling ACO with a learned graph prior:

- A **Graph Neural Network (GNN)** (specifically utilizing **Graph Attention Networks (GAT)**) is trained on optimal or near-optimal historical solutions.
- The model predicts **edge-level probability distributions** indicating the structural likelihood of inclusion in high-quality tours/routes.
- These predictions are injected into ACO as **dynamic heuristic information** ($\eta_{ij}$), guiding ant transitions toward promising trajectories.

This hybridization yields a principled form of search-space pruning, improving both runtime and solution quality on large graph instances.

> [!NOTE]
> The core contribution is not replacing ACO, but **informing ACO** with learned structural priors so stochastic search becomes significantly more sample-efficient.

---

## 🧮 Mathematical Core: Integrating GNN Priors into ACO

In standard ACO, the probability that ant $k$ transitions from node $i$ to node $j$ is defined as:

$$
P_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in allowed_k} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta}
$$

Where:
- $\tau_{ij}$: pheromone intensity on edge $(i,j)$
- $\eta_{ij}$: heuristic desirability (classically $1/d_{ij}$ where $d$ is distance)
- $\alpha, \beta$: importance coefficients controlling pheromone vs. heuristic influence
- $allowed_k$: feasible next-node set for ant $k$

### Learned Heuristic Redefinition

In this repository, $\eta_{ij}$ is **not restricted** to static local distance ($1/d_{ij}$). Instead, it is dynamically informed by the trained GNN (GAT backbone):

$$
\eta_{ij} = \text{GNN}_{\theta}(G)_{ij}
$$

Thus, transition probabilities become context-aware over global graph topology, allowing ants to exploit learned structural regularities rather than relying solely on local distance heuristics.

> [!IMPORTANT]
> This mathematical shift is the central mechanism behind faster convergence: heuristic guidance is learned from data, not hand-crafted alone.

---

## 🔄 Architecture Flow

```text
Step 1: Graph Representation
  Nodes/Edges  ──>  PyG Data object (x, edge_index, edge_attr)

Step 2: GNN Feature Extraction
  Message passing over graph using Graph Attention Networks (GAT)

Step 3: Edge Probability Prediction
  Edge scoring head outputs p_ij for candidate transitions

Step 4: GNN-Guided Heuristic / Pheromone Initialization
  eta_ij <- GNN_theta(G)_ij   (optionally fused with 1/d_ij)

Step 5: High-Performance ACO Search
  Parallel ant rollout + pheromone update + iterative refinement
```

### ✨ Key Features
- **Dynamic Heuristics:** ML-guided transition priors for aggressive search-space pruning.
- **PyTorch Geometric Integration:** Efficient sparse graph operations for scalable message passing.
- **Parallel Ant Simulation:** Multi-ant rollout optimized for high-throughput stochastic search.
- **Extensible Backbone Design:** Easily swap GCN, GAT, or GraphSAGE with minimal refactoring.

---

## 📂 Project Structure

```bash
gnn-guided-aco/
├── models/                 # GNN architectures (GCN, GAT, GraphSAGE backbones)
│   ├── gcn.py
│   ├── gat.py
│   └── __init__.py
├── aco/                    # Core ACO engine: ant simulation, pheromone updates
│   ├── ant_system.py
│   ├── pheromone.py
│   └── __init__.py
├── utils/                  # Data loaders, graph generators, preprocessing helpers
│   ├── data_loader.py
│   ├── graph_utils.py
│   └── metrics.py
├── train_gnn.py            # Script to train the GNN predictor
├── main.py                 # Entrypoint for running the hybrid solver
└── requirements.txt
```

---

## ⚡ Installation & Quick Start

### 1) Clone Repository
```bash
git clone https://github.com/Reza-Davarpanah/gnn-guided-aco.git
cd gnn-guided-aco
```

### 2) Install Dependencies
Create a clean virtual environment and install the requirements. 

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> To utilize GPU acceleration, ensure your PyTorch and PyTorch Geometric installation matches your system's CUDA version. You can verify your setup with:
> `python -c "import torch; print(torch.cuda.is_available())"`

### 3) Train the GNN Predictor (GAT Backbone)
Train the GNN on solved TSP instances. The default setup runs for 150 epochs with a learning rate of 0.001.

```bash
python train_gnn.py --epochs 150 --lr 0.001 --model GAT
```

### 4) Solve TSP with GNN Guidance
Run the ACO solver guided by the trained GNN model. Standard parameters are set to 20 ants, $\alpha = 1.0$, and $\beta = 2.0$.

```bash
python main.py --problem tsp --size 100 --use-gnn true --ants 20 --alpha 1.0 --beta 2.0
```

### Optional Baseline Run (Standard ACO)
```bash
python main.py --problem tsp --size 100 --use-gnn false --ants 20 --alpha 1.0 --beta 2.0
```

---

## 📊 Benchmarks & Results

*Experimental setup: Single GPU-enabled workstation, fixed random seeds, averaged over 10 independent runs.*

| Problem Size (N) | Standard ACO (Time / Tour Length) | GNN-Guided ACO (Time / Tour Length) | Gap Improvement (%) |
| :---: | :---: | :---: | :---: |
| **50** | 1.3 s / 586.9 | 0.5 s / 573.8 | **2.2%** |
| **100** | 9.1 s / 1162.4 | 2.1 s / 1107.6 | **4.7%** |
| **200** | 45.8 s / 2338.1 | 8.4 s / 2180.3 | **6.7%** |
| **500** | 341.2 s / 5921.7 | 56.9 s / 5438.9 | **8.2%** |

> [!NOTE]
> “Gap Improvement (%)” reflects relative solution-quality gain (tour-length reduction). Runtime improvements are substantially larger due to guided exploration and early search-space pruning.

---

## 🗺️ Future Roadmap

> [!IMPORTANT]
> We are actively working on porting the core ACO ant simulation logic into a high-performance Rust extension using PyO3 to bypass Python's GIL, achieving extreme parallelization and sub-millisecond execution times.

- [ ] **Rust-native ant rollout kernel** for massive parallel simulation.
- [ ] **Lock-efficient** pheromone update primitives.
- [ ] Python API compatibility via **PyO3 bindings**.
- [ ] Benchmark suite for cross-language performance regression tracking.

---

## 📜 Citation (Suggested)

If this repository contributes to your research or production stack, please consider citing it using the following format:

```bibtex
@software{davarpanah2025gnn_guided_aco,
  author = {Davarpanah, Reza},
  title = {GNN-Guided Ant Colony Optimization: A Hybrid ML-Metaheuristic Framework},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Reza-Davarpanah/gnn-guided-aco}}
}
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for full terms.

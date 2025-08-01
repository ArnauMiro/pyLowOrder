# Dimensional Notation Convention

This document defines the dimensional notation used across the GNN-related modules of `pyLowOrder`. These symbolic conventions describe tensor shapes and dimensions in a consistent, compact manner, both in code and documentation.

> **Note**: These conventions are specific to the GNN modules under `pyLOM.NN` and **do not apply to other architectures** in `pyLowOrder`, such as KAN or MLP, which may use different formats.

---

## 📐 Symbol Definitions

| Symbol | Meaning                                  | Example Shape           |
|--------|-------------------------------------------|-------------------------|
| `B`    | Batch size (number of operating conditions) | `[B, D]`                |
| `N`    | Number of nodes in a graph or subgraph    | `[N, F]`                |
| `F`    | Node feature dimension                    | `[N, F]`                |
| `D`    | Global input feature dimension (e.g., Mach, AoA) | `[B, D]`          |
| `E`    | Number of edges                           | `[2, E]`                |
| `A`    | Edge feature dimension                    | `[E, A]`                |
| `O`    | Output feature dimension (e.g., pressure) | `[B * N, O]`            |
| `S`    | Number of seed nodes                      | `[S]`, `[B * S]`        |
| `L`    | Number of message-passing layers          | Scalar                  |

---

## 🧠 Typical Tensor Shapes

| Component              | Shape                    | Description |
|------------------------|--------------------------|-------------|
| Global inputs          | `[B, D]`                 | `B` parameter sets (e.g., Mach, AoA) |
| Node features          | `[N, F]` or `[B * N, F]` | Per-node physical features |
| Edge features          | `[E, A]` or `[B * E, A]` | Features for each connection |
| Edge index (COO)       | `[2, E]` or `[2, B * E]` | Source/target indices |
| Predictions            | `[B * N, O]`             | One output per node per sample |
| Targets (optional)     | `[B * N, O]`             | Ground truth values |
| Seed nodes (optional)  | `[B * S]`                | Used for loss masking |

---

## 📌 Notes

- Batched graphs are constructed as **disconnected subgraphs** within a larger unified graph.
- All tensor operations (e.g., forward pass, loss computation) are fully vectorized over `B`, not looped.
- The dimension `B` is orthogonal to the spatial dimensions `N` and `E` — it indexes parameter conditions.

---

## 📎 See Also

- [`GraphPreparer`](../utils/preparation.py): Builds batched graphs.
- [`Graph`](../graph.py): Custom graph class with feature dictionaries.
- [`GNS`](../architectures/gns.py): GNN model using these conventions.


# GNS Tensor Notation

This notation is used in GNS-related documentation and scripts.

## Symbols

| Symbol | Meaning |
|---|---|
| `B` | batch size (number of parameter samples) |
| `N` | number of graph nodes |
| `E` | number of graph edges |
| `D` | global input feature dimension |
| `F` | node feature dimension |
| `A` | edge feature dimension |
| `O` | output feature dimension |

## Typical shapes

| Tensor | Shape |
|---|---|
| global inputs | `[B, D]` |
| node features | `[N, F]` |
| edge features | `[E, A]` |
| edge index (COO) | `[2, E]` |
| predictions (flattened) | `[B * N, O]` |
| targets (flattened) | `[B * N, O]` |

## Notes

- GNS uses message passing on a fixed graph.
- Batched conditions are typically represented by repeating/injecting global inputs over graph nodes.
- Some scripts may keep outputs as `[B, N, O]` before flattening for metrics.

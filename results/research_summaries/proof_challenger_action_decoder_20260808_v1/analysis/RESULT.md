# Proof challenger action decoder

The graph proof proposes a complete answer set; a pairwise action
model must predict positive utility over KEEP before it is applied.

## Train-only nested result

| policy | pooled F1 | delta |
|---|---:|---:|
| retained staged pipeline | 0.481128 | -- |
| reordered historical frozen scorer | 0.481128 | +0.000000 |
| challenger-specific nested OOF | 0.485320 | +0.004193 |

Train promotion gate passed: **False**

Edits: 3 changed; 2 helped; 0 harmed.

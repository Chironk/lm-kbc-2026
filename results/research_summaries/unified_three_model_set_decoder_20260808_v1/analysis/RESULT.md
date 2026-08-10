# Unified three-model set decoder

All coherent Qwen, Gemma, and Ministral generation sets are peer
hypotheses with KEEP in one row-balanced learned decoder.

| policy | pooled F1 | delta |
|---|---:|---:|
| retained staged reconstruction | 0.481128 | -- |
| unified nested OOF decoder | 0.487193 | +0.006066 |
| coherent set oracle | 0.738689 | +0.257561 |

Promotion gate passed: **False**

Inventory: 3519 alternatives on 442 rows.
OOF edits: 43 changed; 8 helped; 7 harmed; 28 neutral.

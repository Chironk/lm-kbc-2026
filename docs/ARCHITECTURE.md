# Final architecture and code map

## Supported path

The supported system is a single frozen pipeline, not a sequence of manual
post-hoc edits. The shell launcher delegates task planning and generation to
`end_to_end_pipeline.py`, then delegates graph construction, decoding,
validation, and packaging to `final_submission_pipeline.py`.

1. **Plan.** Strip labels from the selected split, bind every task to pinned
   model/config/prompt hashes, and create resumable route task files.
2. **Generate.** Run Qwen, Gemma, and Ministral independently. Qwen provides
   the retained incumbent and raw samples; Gemma adds an independent proposal;
   Ministral contributes a three-sample route and a ten-sample SyntheticCoT
   route.
3. **Build the evidence graph.** Normalize answer surfaces into candidate
   components while preserving complete generation events. Directed
   `supports` edges connect an event to every component in that event's
   complete answer set. Event attributes retain model family, route, and
   sample identity.
4. **Decode.** Apply the frozen cardinality, numeric, route, component, and
   Ministral support stages to produce an incumbent answer set.
5. **Symbolic correction.** For entity-valued relations, compare complete
   challenger sets using independent-family support, cross-family set
   compatibility, and output-structure safeguards. For `hasCapacity`, use the
   scalar-numeric proof: valid singleton values are grouped under the official
   5% tolerance and a replacement requires a strict family-evidence advantage.
   Invalid graph evidence fails closed to the incumbent.
6. **Package.** Validate key coverage and artifact contracts, then create a zip
   with one root-level `predictions.jsonl`.

## Model portfolio

| Family | Pinned checkpoint | Parameters | Main role |
|---|---|---:|---|
| Qwen | `Qwen/Qwen3.5-9B` | 9,409,813,744 | primary proposals and incumbent |
| Gemma | `google/gemma-3-12b-it` | 12,187,325,040 | independent proposal route |
| Ministral | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 8,918,026,240 | heterogeneous proposal and support routes |
| **Total** | | **30,515,165,024** | under the 32B cap |

Exact revisions and parameter counts are in `configs/final/` and
`artifacts/frozen/MANIFEST.json`.

## Production entrypoints

- `run_final_submission_pipeline.sh`: only user-facing final launcher.
- `final_submission_pipeline.py`: final graph schema, decoder order, scoring,
  and package contract.
- `end_to_end_pipeline.py`: label-free task plan and route generation.
- `run_submission.py`: retained production Qwen route.
- `run_agent.py`: isolated Gemma/Ministral route execution.

The tested graph transforms, feature definitions, and decoder stages imported
by the final pipeline live under `experiments/heterogeneous_agents/components/`.
They are implementation dependencies, not separate recommended pipelines.
Optional post-hoc visualization code is isolated under `analysis/`.

## Frozen decoder artifacts

`artifacts/frozen/MANIFEST.json` is the root of trust. It binds the five small
decoder artifacts by SHA-256 and records the model parameter contract. It
contains no split-specific predictions and no test labels.

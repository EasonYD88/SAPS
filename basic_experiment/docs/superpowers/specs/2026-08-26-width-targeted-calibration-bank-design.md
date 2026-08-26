# Width-Targeted Calibration Schedule Bank Design

## Objective

Build a deterministic, development-only schedule bank that supplies at least 20 valid direct-response probes for every `(epsilon, width)` cell required by the locked `d=12`, `q=4` smoke experiment. The bank preserves modular support sizes 3--5, the existing instance count, rounds, and balanced batch capacities.

## Isolation and information-flow contract

The bank may consume only development reward instances, their candidate supports, fixed experiment structure, and trajectories drawn from a dedicated `width_calibration_bank` RNG namespace. It may not consume held-out instances, held-out gain evaluations, method scores, method rankings, or method performance. Bank schedules are calibration instruments, not experimental methods.

The manifest records these forbidden flows explicitly:

- `held-out evaluation -> calibration`
- `method results -> calibration`
- `calibration probes -> held-out gain estimates`

All three must be marked `forbidden`, with validation rejecting any non-development reward-instance ID in the bank.

## Schedule construction

For a support and a requested target width, construct a balanced schedule using one of the locked round counts whose final occupied batch contains exactly that many support positions. Assignment of support positions and capacity-filling positions is deterministic under the dedicated RNG namespace. Candidate schedules are deduplicated by canonical batches and receive SHA-256 schedule IDs.

The bank targets widths 1--5 independently for each configured epsilon. It requests 24 probes per cell by default, may generate up to 32 to replace invalid probes, and accepts a cell only when `valid_count >= 20`. Invalid/empty-rank probes remain recorded but cannot be borrowed across widths or epsilons. If any cell is short, calibration status is `inconclusive`; no interpolation or held-out fallback is permitted.

## Frozen artifacts

Calibration emits:

- `calibration_schedule_bank.jsonl`: one canonical schedule record per probe assignment.
- `calibration_coverage.csv`: requested, attempted, valid, and invalid counts per `(epsilon, width)`.
- `calibration_manifest.json`: grids, capacities, rounds, budgets, RNG seed/namespace, development reward-instance IDs, schedule IDs, information-flow contract, resource totals, artifact hashes, and status.
- `experiment_manifest.json`: the existing frozen width weights plus the SHA-256 of the calibration manifest and schedule bank.

The calibration manifest separately reports model calls, terminal-label calls, and wall time. Terminal-label calls include every development trajectory used for calibration geometry or direct-response probing and never count held-out adaptation or gain labels.

## Evaluation DAG

Stage A creates and freezes the calibration artifacts. Stage B shards held-out reward instances and reads only the frozen Stage A manifest/weights. Stage C merges and analyzes Stage B. Any failed or inconclusive Stage A prevents Stage B through `afterok` dependencies and loader validation.

## Tests

Tests cover exact widths 1--5 under balanced capacities, deterministic IDs and RNG isolation, development-only records, cell-level coverage without cross-cell borrowing, inconclusive behavior, artifact SHA-256 integrity, forbidden-flow declarations, and separate resource accounting.

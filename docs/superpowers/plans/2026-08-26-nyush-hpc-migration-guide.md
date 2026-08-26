# NYU Shanghai HPC Migration Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a tested SAPS release snapshot and a short, evidence-driven guide that a Codex session on NYU Shanghai HPC can use to adapt and validate the project safely.

**Architecture:** Preserve the existing NYU Torch scripts as provenance and make the target-side Codex discover NYUSH scheduler, storage, module, and network contracts before creating `basic_experiment/slurm/nyush/` variants. Move code through Git at an immutable annotated tag; rebuild Python on NYUSH and validate in a ladder from unit tests to one smoke job before any full DAG.

**Tech Stack:** Markdown, Git, Python 3.11+, pytest, setuptools, Bash, SLURM-compatible discovery commands

## Global Constraints

- The target-side Codex must discover actual NYUSH settings and must not invent private account, partition, QOS, module, or filesystem names.
- SAPS is CPU-only and requires Python 3.11 or newer.
- Preserve `basic_experiment/slurm/*.sbatch`; NYUSH variants belong under `basic_experiment/slurm/nyush/`.
- Rebuild the Python environment on NYUSH; never copy the Torch Conda environment or virtual environment.
- Do not copy or commit tokens, credentials, caches, virtual environments, or unrelated scratch data.
- Stop at the first failed validation gate; do not submit a production experiment during migration.
- Use the immutable release tag `nyush-migration-2026-08-26` for the target checkout.

---

### Task 1: Freeze and commit the current calibration workflow

**Files:**
- Modify: `basic_experiment/README.md`
- Modify: `basic_experiment/configs/phase1_main.yaml`
- Modify: `basic_experiment/configs/screening.yaml`
- Modify: `basic_experiment/configs/smoke.yaml`
- Modify: `basic_experiment/configs/smoke_calibration.yaml`
- Modify: `basic_experiment/slurm/run_calibration.sbatch`
- Modify: `basic_experiment/src/feedback_frontier/cli.py`
- Modify: `basic_experiment/src/feedback_frontier/config.py`
- Modify: `basic_experiment/src/feedback_frontier/runners/calibration_bank.py`
- Modify: `basic_experiment/src/feedback_frontier/runners/evaluate_planner.py`
- Test: `basic_experiment/tests/test_calibration_bank.py`
- Test: `basic_experiment/tests/test_runner_analysis.py`
- Test: `basic_experiment/tests/test_smoke_contract.py`
- Exclude: `basic_experiment/logs/saps-calibrate-16426540.*`
- Exclude: `basic_experiment/logs/saps-frozen-16426554_*`
- Exclude: `basic_experiment/outputs/phase1-smoke-calibration-bank-v1/`

**Interfaces:**
- Consumes: the already-written calibration command, deterministic `calibration_rng_seed`, frozen calibration artifacts, and their tests.
- Produces: one reviewed commit containing the current 13 source/config/test/documentation modifications and no newly generated runtime artifacts.

- [ ] **Step 1: Confirm no relevant SLURM job is still mutating the workspace**

Run on Torch:

```bash
squeue --me -o "%.18i %.40j %.8T %.10M %.20R"
```

Expected: no active `saps-calibrate`, `saps-frozen`, or dependent merge job. If one is active, stop this task until it reaches a terminal state and inspect it with `sacct`.

- [ ] **Step 2: Review the exact source boundary**

Run:

```bash
git diff --stat
git diff --check
git status --short
```

Expected: the 13 listed tracked files are modified. Generated logs and `phase1-smoke-calibration-bank-v1/` may remain untracked, but no credential, environment, cache, or unrelated file may enter the commit.

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
cd /scratch/yd2915/SAPS/basic_experiment
python -m pytest -q
```

Expected: exit code 0 and all collected tests pass. The observed pre-plan baseline is 103 passing tests; use the actual collected count if additional intentional tests appeared before execution.

- [ ] **Step 4: Validate every shipped configuration and shell script**

Run:

```bash
cd /scratch/yd2915/SAPS/basic_experiment
for config in configs/phase1_main.yaml configs/screening.yaml configs/smoke.yaml configs/smoke_calibration.yaml; do
  feedback-frontier validate-config --config "$config"
done
for script in slurm/*.sbatch; do
  bash -n "$script"
done
```

Expected: four `valid:` messages, then no output from `bash -n`, with overall exit code 0.

- [ ] **Step 5: Stage only the reviewed feature files**

Run from `/scratch/yd2915/SAPS`:

```bash
git add \
  basic_experiment/README.md \
  basic_experiment/configs/phase1_main.yaml \
  basic_experiment/configs/screening.yaml \
  basic_experiment/configs/smoke.yaml \
  basic_experiment/configs/smoke_calibration.yaml \
  basic_experiment/slurm/run_calibration.sbatch \
  basic_experiment/src/feedback_frontier/cli.py \
  basic_experiment/src/feedback_frontier/config.py \
  basic_experiment/src/feedback_frontier/runners/calibration_bank.py \
  basic_experiment/src/feedback_frontier/runners/evaluate_planner.py \
  basic_experiment/tests/test_calibration_bank.py \
  basic_experiment/tests/test_runner_analysis.py \
  basic_experiment/tests/test_smoke_contract.py
git diff --cached --name-status
```

Expected: exactly the 13 listed files and no `logs/` or `outputs/` path.

- [ ] **Step 6: Commit the calibration workflow**

Run:

```bash
git commit -m "feat: add frozen width calibration workflow"
```

Expected: one commit containing 13 files, followed by a working tree whose remaining changes are only generated runtime artifacts.

---

### Task 2: Write the target-side NYUSH migration guide

**Files:**
- Create: `docs/NYUSH_HPC_MIGRATION.md`
- Reference: `docs/superpowers/specs/2026-08-26-nyush-hpc-migration-guide-design.md`
- Reference: `basic_experiment/README.md`
- Reference: `basic_experiment/pyproject.toml`
- Reference: `basic_experiment/slurm/*.sbatch`

**Interfaces:**
- Consumes: immutable tag name `nyush-migration-2026-08-26`, Python requirement `>=3.11`, the five Torch SLURM scripts, and the existing smoke/calibration DAG.
- Produces: `docs/NYUSH_HPC_MIGRATION.md`, an instruction document that a target-side Codex can execute without assuming NYUSH-private configuration.

- [ ] **Step 1: Write the guide's scope and stop rules**

The opening must state all of the following:

```markdown
# Migrating SAPS to NYU Shanghai HPC

This guide is for a Codex session running inside an authenticated NYU Shanghai
HPC environment. It must inspect the real cluster configuration before editing
or submitting jobs. Do not copy the Torch Python environment, guess scheduler
settings, expose credentials, or run a production experiment during migration.

Stop and ask the user or NYUSH IT if the scheduler, allocation, storage policy,
Python installation path, or compute-node network policy cannot be established
from read-only commands and authenticated NYUSH documentation.
```

- [ ] **Step 2: Add immutable checkout commands**

The guide must use the annotated tag rather than a moving branch:

```bash
git clone https://github.com/EasonYD88/SAPS.git
cd SAPS
git fetch --tags --force
git checkout --detach nyush-migration-2026-08-26
git rev-parse HEAD
git status --short --branch
```

Expected target state: detached at the release tag with no working-tree changes. The target-side handoff record stores the printed SHA.

- [ ] **Step 3: Add a read-only environment discovery block**

The guide must tell target Codex to capture the output of:

```bash
hostname -f
uname -a
id
pwd
command -v sbatch || true
command -v srun || true
command -v sinfo || true
command -v module || true
command -v python3 || true
sbatch --version 2>/dev/null || true
sinfo -o "%P %a %l %D %c %m %G" 2>/dev/null || true
sacctmgr -n -P show assoc user="$USER" format=Account,User,Partition,QOS 2>/dev/null || true
df -h .
quota -s 2>/dev/null || true
module avail python 2>&1 | sed -n '1,120p'
python3 --version 2>/dev/null || true
```

The surrounding text must require comparison with authenticated NYUSH HPC documentation. A missing or permission-denied command is evidence to investigate, not permission to invent a value.

- [ ] **Step 4: Add the fresh-environment procedure**

The guide must say to choose a documented Python 3.11+ module, Conda environment, or venv. For a supported `python3`, provide this concrete venv path:

```bash
cd SAPS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './basic_experiment[dev]'
python --version
python -m pip freeze
cd basic_experiment
python -m pytest -q
feedback-frontier validate-config --config configs/smoke.yaml
```

Expected: Python 3.11 or newer, all tests pass, and the smoke configuration prints `valid:`. If compute nodes cannot reach package indexes, the guide must require the NYUSH-documented wheel/cache workflow rather than adding ad hoc proxy settings.

- [ ] **Step 5: Add the NYUSH SLURM adaptation contract**

The guide must preserve Torch scripts and create copies:

```bash
cd SAPS
mkdir -p basic_experiment/slurm/nyush
for script in basic_experiment/slurm/*.sbatch; do
  cp "$script" "basic_experiment/slurm/nyush/$(basename "$script")"
done
```

It must tell target Codex to edit only the copies and replace Torch-specific account, partition, QOS, module, resource, and notification directives using target evidence. Every NYUSH copy must replace the absolute Torch project root with:

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
```

After editing, validate all copies:

```bash
for script in basic_experiment/slurm/nyush/*.sbatch; do
  bash -n "$script"
done
```

Expected: no `bash -n` output and exit code 0. Target Codex must show a diff and explain each cluster-specific replacement before submission.

- [ ] **Step 6: Add the smoke-job validation ladder**

The guide must require a deliberately short wall time and the smallest smoke configuration supported by NYUSH. Submit only after unit/config/shell validation:

```bash
cd SAPS/basic_experiment
mkdir -p logs
JOB_ID=$(sbatch --parsable slurm/nyush/run_experiment.sbatch \
  configs/smoke.yaml nyush-smoke-v1)
echo "$JOB_ID"
squeue -j "$JOB_ID" -o "%i %T %M %D %R"
```

After terminal state:

```bash
sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P
test -f outputs/nyush-smoke-v1/experiment_manifest.json
test -f outputs/nyush-smoke-v1/gate_report.json
python -c 'import json; p=json.load(open("outputs/nyush-smoke-v1/experiment_manifest.json")); print(p["status"], p.get("artifact_sha256", {}))'
```

Expected: scheduler state `COMPLETED`, exit code `0:0`, manifest status `complete`, and non-empty artifact hashes. The guide must tell target Codex to stop and diagnose logs on any mismatch.

- [ ] **Step 7: Add the production DAG gate and handoff record**

The guide must state that the calibration → frozen array → merge/analyze commands in `basic_experiment/README.md` are allowed only after the smoke gate. It must require a final report with:

```text
Git tag and commit SHA
NYUSH hostname and filesystem root
Python and dependency versions
Scheduler account, partition/queue, QOS, CPU, memory, and wall time
Smoke job ID, terminal state, stdout, and stderr paths
Manifest status and artifact hashes
NYUSH-specific files changed
Location and retention policy for new outputs
```

The data-policy section must say that small tracked fixtures arrive through Git, while new or large results use the transfer method documented by NYUSH. Tokens, credential files, `.venv`, caches, and unrelated scratch data never enter Git.

- [ ] **Step 8: Add explicit failure handling and completion criteria**

The guide must include a failure table with these exact decisions:

```text
Missing scheduler or allocation -> stop and request account/access details.
Python older than 3.11 -> select a documented module or approved environment; do not lower the project requirement.
No package-index access -> use the NYUSH-documented wheel/cache workflow; do not add an ad hoc proxy.
Scheduler rejection -> inspect the scheduler reason and change only the rejected resource contract.
Test, manifest, or hash mismatch -> stop before the full DAG and preserve logs for diagnosis.
```

The completion section must require all of the following: checkout resolves to
`nyush-migration-2026-08-26`; the full test suite and configuration validation
pass; one compute-node smoke job finishes with exit code `0:0`; expected
manifests and hashes validate; and the handoff record contains enough
information to reproduce the run.

---

### Task 3: Validate and commit the migration guide

**Files:**
- Create: `docs/NYUSH_HPC_MIGRATION.md`
- Verify: `docs/superpowers/specs/2026-08-26-nyush-hpc-migration-guide-design.md`

**Interfaces:**
- Consumes: the complete guide from Task 2 and the approved design spec.
- Produces: a reviewed documentation commit with valid shell examples, no placeholders, and full design coverage.

- [ ] **Step 1: Scan for placeholders and accidental secrets**

Run:

```bash
rg -n '\b(TBD|TODO|FIXME|XXX)\b|<[^>]+>' docs/NYUSH_HPC_MIGRATION.md
rg -n -i '(token|password|secret)[[:space:]]*[:=][[:space:]]*[^[:space:]]+' docs/NYUSH_HPC_MIGRATION.md
```

Expected: no output. Explanatory prose may use the words `token`, `password`, or `secret`, but no value assignment may appear.

- [ ] **Step 2: Syntax-check every executable shell block**

Copy each complete Bash block from the guide into a temporary file created with `mktemp`, replace no cluster values, and run:

```bash
bash -n /tmp/nyush-migration-guide-block.sh
```

Expected: no output and exit code 0 for every complete block. Do not execute scheduler commands during this syntax check.

- [ ] **Step 3: Check formatting and design coverage**

Run:

```bash
git diff --check -- docs/NYUSH_HPC_MIGRATION.md
rg -n '^## ' docs/NYUSH_HPC_MIGRATION.md
```

Expected headings: scope/stop conditions, immutable checkout, discovery, environment rebuild, SLURM adaptation, validation ladder, failure handling, data policy, and completion report.

- [ ] **Step 4: Commit only the guide**

Run:

```bash
git add docs/NYUSH_HPC_MIGRATION.md
git diff --cached --name-status
git commit -m "docs: add NYUSH HPC migration guide"
```

Expected: the staged list contains only `docs/NYUSH_HPC_MIGRATION.md` before commit.

---

### Task 4: Verify, tag, and publish the migration snapshot

**Files:**
- Verify: `basic_experiment/`
- Verify: `docs/NYUSH_HPC_MIGRATION.md`
- Verify: `docs/superpowers/specs/2026-08-26-nyush-hpc-migration-guide-design.md`
- Verify: `docs/superpowers/plans/2026-08-26-nyush-hpc-migration-guide.md`

**Interfaces:**
- Consumes: committed calibration workflow, design, plan, and migration guide.
- Produces: `main` and annotated tag `nyush-migration-2026-08-26` on GitHub, plus a target-side handoff containing the tag and resolved SHA.

- [ ] **Step 1: Run final source verification**

Run:

```bash
cd /scratch/yd2915/SAPS/basic_experiment
python -m pytest -q
for config in configs/phase1_main.yaml configs/screening.yaml configs/smoke.yaml configs/smoke_calibration.yaml; do
  feedback-frontier validate-config --config "$config"
done
for script in slurm/*.sbatch; do
  bash -n "$script"
done
```

Expected: tests pass, four configurations print `valid:`, and every script passes `bash -n`.

- [ ] **Step 2: Confirm the release excludes runtime artifacts**

Run from `/scratch/yd2915/SAPS`:

```bash
git diff --cached --quiet
git diff --quiet
git status --short
```

Expected: no staged or unstaged tracked changes. Untracked calibration logs and outputs may remain locally; they must not be part of any release commit.

- [ ] **Step 3: Create the immutable annotated tag**

Run:

```bash
git tag -a nyush-migration-2026-08-26 -m "NYUSH HPC migration baseline"
git rev-list -n 1 nyush-migration-2026-08-26
git log -1 --format='%H %s' nyush-migration-2026-08-26
```

Expected: both commands resolve to the same release commit.

- [ ] **Step 4: Push branch and tag**

Run in the user's authenticated terminal if the agent cannot access the in-memory Git credential cache:

```bash
git push origin main
git push origin nyush-migration-2026-08-26
```

Expected: Git reports updates for `main` and a new tag.

- [ ] **Step 5: Verify remote refs and hand off to NYUSH**

Run:

```bash
git ls-remote origin refs/heads/main refs/tags/nyush-migration-2026-08-26 'refs/tags/nyush-migration-2026-08-26^{}'
git rev-list -n 1 nyush-migration-2026-08-26
```

Expected: the peeled annotated-tag SHA equals the local `git rev-list` SHA, and `main` contains that commit. Send the NYUSH Codex the repository URL, tag name, resolved SHA, and `docs/NYUSH_HPC_MIGRATION.md` as its starting instruction.

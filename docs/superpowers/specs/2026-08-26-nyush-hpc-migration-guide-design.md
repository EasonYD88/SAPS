# NYU Shanghai HPC Migration Guide Design

**Date:** 2026-08-26

## Goal

Create `docs/NYUSH_HPC_MIGRATION.md`, a short operational handoff for a Codex
session running on the NYU Shanghai HPC cluster. The target-side Codex must
discover the cluster's actual scheduler, accounts, partitions, storage,
modules, and network constraints before changing or submitting anything.

The migration baseline must include the current uncommitted SAPS work after it
has passed the full test suite, been committed, and been pushed to GitHub.

## Constraints

- The public NYU Shanghai HPC page links to detailed documentation that
  requires authenticated access. Source-side instructions must not invent
  internal cluster settings.
- SAPS is a CPU-only Python project requiring Python 3.11 or newer.
- The current five SLURM scripts contain Torch-specific account, partition,
  module, email, and absolute-path settings.
- Python environments are not portable between clusters and must be rebuilt.
- Existing Torch scripts must remain available as provenance; NYUSH variants
  belong under `basic_experiment/slurm/nyush/`.
- Tokens, credentials, virtual environments, caches, and unrelated scratch
  data must not be copied or committed.

## Considered Approaches

### 1. Target-side discovery guide

Commit a guide that tells the NYUSH Codex how to inspect its authenticated
environment, rebuild dependencies, create NYUSH-specific SLURM scripts, and
validate the port. This is the selected approach because it is small, auditable,
and robust to inaccessible or changing cluster details.

### 2. Parameterize every SLURM script before migration

Replace all Torch settings with environment-variable contracts now. This would
reduce duplication but expand the source-side change before NYUSH requirements
can be verified. It is deferred until both clusters' contracts are known.

### 3. Containerize the project

Build an Apptainer or Singularity image for reproducibility. This is unnecessary
for the current CPU-only dependency set and depends on an unverified NYUSH
container runtime.

## Guide Structure

### 1. Source release gate

The source-side operator must:

1. Review the current working tree and exclude generated or sensitive files.
2. Run the complete test suite.
3. Commit and push the current work plus the migration guide.
4. Record the exact commit SHA that NYUSH must clone.

The guide must not recommend copying an uncommitted working directory as the
primary migration path.

### 2. Read-only target discovery

Before editing files, the NYUSH Codex must gather and record:

- hostname, operating system, shell, and repository path;
- scheduler availability and version;
- permitted accounts, partitions or queues, QOS, CPU, memory, array, and
  wall-time limits;
- home, project, scratch, archive, quota, backup, and purge policies;
- available Python, Conda, module, and container tooling;
- compute-node outbound-network restrictions; and
- the authenticated NYUSH HPC documentation relevant to these findings.

Discovery commands must be read-only. If an essential value cannot be
determined, the Codex must stop and ask the user or NYUSH IT rather than guess.

### 3. Immutable checkout and environment rebuild

The target-side workflow must clone the GitHub repository, check out the
recorded commit, and create a fresh Python environment in an administrator-
approved location. It then installs `basic_experiment` with development
dependencies and records Python and package versions.

Copying a Torch Conda environment or virtual environment is explicitly
forbidden because paths and compiled dependencies may not be portable.

### 4. NYUSH scheduler adaptation

The NYUSH Codex must preserve `basic_experiment/slurm/*.sbatch` and create
adapted scripts in `basic_experiment/slurm/nyush/`. It may change only settings
supported by target-side evidence, including:

- account, partition or queue, and QOS;
- CPU, memory, wall time, and array concurrency;
- project root and output paths;
- module loading or environment activation; and
- notification directives supported by the scheduler.

The target scripts must use a repository-relative or explicitly configured
project root instead of `/scratch/yd2915/SAPS/basic_experiment`.

### 5. Validation ladder

Validation must stop at the first failing gate:

1. Install/import check and complete `pytest` suite.
2. Configuration validation for the smoke configuration.
3. Minimal CLI check without a production experiment.
4. One short smoke scheduler job.
5. Audit scheduler state, exit code, stdout, stderr, manifest, and artifact
   hashes.
6. Only after those pass, run the calibration-to-array-to-merge/analyze DAG.

The guide must provide diagnostic commands but must not submit production jobs
automatically.

### 6. Handoff record and data policy

The target-side Codex must report the commit SHA, Python and dependency
versions, scheduler account and partition, job IDs, logs, and output paths.
Existing small tracked artifacts arrive through Git. New or large experiment
outputs should use the transfer mechanism recommended by NYUSH rather than be
silently added to Git.

## Error Handling

- Missing scheduler or allocation: stop and request account/access details.
- Unsupported Python version: select a documented module or create an approved
  environment; do not lower the project's Python requirement silently.
- No compute-node network: build the environment on an allowed node or use the
  cluster's documented wheel/cache workflow.
- Scheduler rejection: inspect the scheduler's reason and change only the
  rejected resource contract.
- Test, manifest, or hash mismatch: stop before any full experiment and preserve
  logs for diagnosis.

## Completion Criteria

Migration is complete only when:

- NYUSH is running the recorded Git commit;
- the full test suite passes in the rebuilt environment;
- configuration validation succeeds;
- a smoke job finishes successfully on a compute node;
- expected manifests and hashes validate; and
- the handoff record contains enough information to reproduce the run.

## Out of Scope

- Installing or requesting an NYUSH account on the user's behalf.
- Guessing private NYUSH account, partition, or filesystem names.
- Running the full production experiment during initial migration.
- Containerization or a cross-cluster SLURM abstraction layer.

## Reference

- [NYU Shanghai High Performance Computing](https://shanghai.nyu.edu/page/high-performance-computing)

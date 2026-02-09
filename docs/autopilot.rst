Nonstop CI Autopilot
====================

StellCoilBench includes a **nonstop CI autopilot** that automatically proposes,
runs, and records coil optimisation cases in a continuous loop.  The system is
designed to explore the parameter space without human intervention while hard
guardrails prevent runaway failures.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

The autopilot loop has three phases that repeat indefinitely:

1. **Propose** — ``tools/propose_batch.py`` generates 8 new cases
   (a mix of mutations of the best-so-far results and random explorations).

2. **Run** — the CI workflow picks up the pending cases and runs each one
   via ``stellcoilbench run-ci-case``.

3. **Record** — results are committed to ``cases/done/<case_id>/summary.json``
   and become input for the next proposal round.

A **batch barrier** ensures only one batch of 8 is in-flight at a time:
the proposer refuses to write new cases while ``cases/pending/`` is non-empty.

.. code-block:: text

   ┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
   │   propose    │ ───► │  cases/pending/   │ ───► │   run-ci-case   │
   │  batch (8)   │      │  *.json           │      │  (up to 8 ∥)    │
   └─────────────┘      └──────────────────┘      └────────┬────────┘
         ▲                                                  │
         │               ┌──────────────────┐               │
         └────────────── │  cases/done/      │ ◄────────────┘
                         │  */summary.json   │
                         └──────────────────┘


CI Scheduling Model
-------------------

The autopilot and benchmark pipeline share a single workflow file
(``update-db-self-hosted.yml``) but are triggered independently and do not
interfere with each other.

**Trigger → job mapping:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Trigger
     - Jobs that run
   * - ``push`` to main (human code / case changes)
     - **Benchmark:** ``determine-cases`` → ``run-cases`` → ``update-leaderboard``.
       **Autopilot:** ``run-autopilot-cases`` → ``propose-autopilot-batch``.
   * - ``push`` (autopilot proposer adds ``cases/pending/``)
     - **Autopilot:** ``run-autopilot-cases`` → ``propose-autopilot-batch``.
       Benchmark jobs run ``determine-cases`` but find no new ``.yaml`` cases.
       This is the **nonstop loop** trigger.
   * - ``push`` (autopilot runner adds ``cases/done/`` only)
     - **Nothing** — excluded by ``paths-ignore``.
   * - ``schedule`` (cron, every 15 min) or ``workflow_dispatch``
     - **Autopilot only:** ``run-autopilot-cases`` → ``propose-autopilot-batch``.
       Benchmark jobs are skipped (filtered by ``event_name == 'push'``).
       Cron acts as a safety net if the push-based loop stalls.

The proposer (``propose-autopilot-batch``) runs on **every** event type —
push, schedule, and dispatch.  This is safe because it has multiple built-in
guards:

- ``PAUSE_AUTORUN`` file check
- Batch barrier (refuses to propose while ``cases/pending/`` is non-empty)
- Guardrails (failure-rate and repeated-failure checks)

Running on push solves the **bootstrap problem**: the first push after merging
the autopilot code immediately proposes 8 cases, without waiting for a cron
tick.

**Why autopilot commits don't cascade:**

When the autopilot runner commits results to ``cases/done/`` or the proposer
commits new cases to ``cases/pending/``, those pushes only touch paths listed
in the workflow's ``paths-ignore``.  GitHub skips the workflow entirely for
such pushes, so the benchmark pipeline (``determine-cases``, ``run-cases``,
``update-leaderboard``) never re-runs on autopilot commits.

**Nonstop loop (self-sustaining):**

Once bootstrapped, the autopilot runs continuously without waiting for cron:

1. ``propose-autopilot-batch`` proposes 8 cases, commits to ``cases/pending/``,
   pushes to main.
2. That push triggers a new workflow run (``cases/pending/**`` is **not** in
   ``paths-ignore`` for the self-hosted workflow).
3. ``run-autopilot-cases`` finds the 8 pending cases, runs them in parallel,
   commits results to ``cases/done/``, deletes the pending files, pushes.
4. ``propose-autopilot-batch`` in the same run proposes 8 more, pushes.
5. Goto step 2.

The runner's push (touching ``cases/done/`` and ``cases/pending/``) may also
trigger a queued run.  This is harmless: the batch barrier ensures the proposer
never double-proposes, and extra runs where pending is empty are quick no-ops.

The cron schedule (every 15 minutes) acts as a **safety net** in case the loop
stalls (e.g. due to a transient GitHub outage that drops a push event).

**Kickstarting the loop:**

The loop starts automatically on the first ``git push`` that includes the
autopilot code.  The ``propose-autopilot-batch`` job runs, proposes 8 cases,
and pushes — which immediately triggers a new run that executes them.

You can also trigger manually: **GitHub → Actions → "StellCoilBench CI
(Self-Hosted)" → "Run workflow"** (the ``workflow_dispatch`` button).

**Concurrency:**

All triggers share a single concurrency group
(``stellcoilbench-selfhosted-<ref>``) with ``cancel-in-progress: false``.
If a cron-triggered run arrives while a push-triggered benchmark is still
running, it queues behind it rather than cancelling it.


Directory Layout
----------------

::

   cases/
     pending/              # proposer writes new cases here (8 at a time)
     done/                 # CI writes completed results here
       <case_id>/
         summary.json      # metrics, timing, success/failure, original config
         case.yaml         # the case config used (for traceability)
         coils.json        # optimised coil geometry (if successful)
   policy/
     proposer_policy.yaml  # thresholds, budgets, mutation ranges, guardrails
   tools/
     propose_batch.py      # non-LLM batch proposer
     build_context.py      # builds context payload from recent results
   PAUSE_AUTORUN           # emergency stop: create this file to halt proposals


CI Case JSON Format
-------------------

Each pending case is a JSON file in ``cases/pending/<case_id>.json``.
It wraps the standard ``case.yaml`` config inside a ``case_config`` key and
adds resource limits, lineage, and tags:

.. code-block:: json

   {
     "case_id": "2026-02-08_123045_84721",
     "parent_ids": ["2026-02-07_091500_12345"],
     "tags": ["exploit", "ncoils=4"],
     "resource": {
       "max_total_iterations": 5000,
       "timeout_minutes": 120
     },
     "case_config": {
       "description": "Mutated from parent ...",
       "surface_params": {
         "surface": "input.LandremanPaul2021_QA",
         "range": "half period"
       },
       "coils_params": { "ncoils": 4, "order": 8 },
       "optimizer_params": {
         "algorithm": "L-BFGS-B",
         "max_iterations": 5000
       },
       "coil_objective_terms": {
         "total_length": "l2_threshold",
         "length_threshold": 24.0,
         "length_weight": 0.05,
         "coil_curvature": "lp_threshold",
         "coil_curvature_p": 2,
         "linking_number": ""
       }
     },
     "random_seed": 42
   }

Hard cap: ``max_total_iterations`` and ``optimizer_params.max_iterations`` must
not exceed **10 000** (enforced by both the validator and the runner).


Summary Output Format
---------------------

After a case completes (success or failure), the runner writes
``cases/done/<case_id>/summary.json``:

.. code-block:: json

   {
     "case_id": "2026-02-08_123045_84721",
     "success": true,
     "total_score": 0.00123,
     "iterations_used": 4200,
     "walltime_sec": 1832.5,
     "failure_reason": "",
     "failure_class": "",
     "metrics": { "final_squared_flux": 0.00123, "..." : "..." },
     "case_config": { "..." : "..." }
   }


How the Proposer Decides What to Run Next
------------------------------------------

The core decision logic lives in two files:

- ``tools/build_context.py`` — reads all completed results and computes a
  context payload.
- ``tools/propose_batch.py`` — consumes that context and emits the next batch
  of cases.

Below is the full decision sequence, executed every time the proposer runs
(every 15 minutes via cron, or on manual dispatch).

**Step 1 — Pre-flight checks** (``propose_batch.main()``)

Before any analysis, the proposer checks two hard gates:

1. Does ``PAUSE_AUTORUN`` exist in the repo root?  If yes → exit immediately
   (code 0, no error).
2. Are there any ``*.json`` files still in ``cases/pending/``?  If yes → exit
   immediately.  This is the **batch barrier**: it ensures the previous batch
   finishes before a new one is proposed.

**Step 2 — Build context** (``build_context.build_context()``)

The proposer loads every ``cases/done/*/summary.json`` (up to the most recent
200), sorted newest-first.  From these it computes:

- **Failure statistics** — over a sliding window (default 30 most recent runs):
  fail count, fail rate, per-reason counts, per-class counts, most common
  failure reason and its count.
- **Top parents** — the *K* best successful results (lowest ``total_score``,
  which is the squared-flux error), each carrying its full ``case_config`` so
  the proposer can mutate it.  Default *K* = 10.
- **Recent config hashes** — SHA-256 truncated to 16 hex chars of the last 50
  configs, used to reject duplicates (novelty filter).
- **Surface exploration counts** — how many runs have been completed per plasma
  surface, for coverage awareness.

**Step 3 — Guardrail check** (``propose_batch.check_guardrails()``)

Three guardrails are evaluated in order.  If *any* fires, the proposer halts:

1. **Failure rate** — if ``fail_rate > max_fail_rate`` (default 0.6), stop.
2. **Repeated failure reason** — if the single most common failure reason
   appears more than ``max_common_failure_count`` (default 12) times in the
   window, stop.
3. **Critical failure class** — if any failure class listed in
   ``critical_failure_classes`` (e.g. ``nan_in_objective``, ``timeout``)
   appears ≥ ``max_critical_class_count`` (default 10) times, stop.

When a guardrail fires, the proposer optionally writes ``PAUSE_AUTORUN``
(controlled by ``cooldown.write_pause_file``) and exits with code 0.

**Step 4 — Safe mode detection** (``propose_batch.is_safe_mode()``)

If the failure rate exceeds ``safe_mode.threshold`` (default 0.35) but is below
the hard guardrail cutoff, the proposer enters **safe mode**.  In safe mode:

- Mutation uses a smaller ``weight_sigma`` (less aggressive jitter).
- Exploration is limited to a lower iteration cap.
- Surface selection is restricted to ``safe_mode.preferred_surfaces`` (simpler
  surfaces that are less likely to fail).

**Step 5 — Batch composition** (``propose_batch.propose_batch()``)

The batch is split according to ``exploit_fraction`` (default 0.5):

- **Exploit slots** = ``floor(exploit_fraction * batch_size)``  → e.g. 4 of 8
- **Explore slots** = ``batch_size - exploit_slots``              → e.g. 4 of 8

If there are no parents yet (first-ever run), all exploit slots are silently
converted to explore slots.

*Exploit (mutation) — fills the exploit slots:*

.. code-block:: text

   repeat (up to 5× exploit_count attempts):
     1. Pick a random parent from top_parents
     2. Deep-copy the parent's case_config
     3. For each *_weight key: multiply by exp(N(0, σ_w²)), clamp to [w_min, w_max]
     4. For each *_threshold key: multiply by exp(N(0, σ_t²)), clamp to [t_min, t_max]
     5. Jitter max_iterations by ±10% (clamped to [100, 10000])
     6. Assign new case_id, random_seed, parent_ids = [parent.case_id]
     7. Hash the new config → reject if hash matches any recent or in-batch hash
     8. Validate via validate_ci_case() → reject if invalid
     9. Accept into batch

The key insight: mutation keeps the parent's surface, algorithm, ncoils, order,
and objective term types intact — it only jitters the *numeric* weights and
thresholds.  This is a local search around configurations that already worked.

*Explore (random) — fills the explore slots:*

.. code-block:: text

   repeat (up to 5× explore_count attempts):
     1. Pick a random surface from policy.exploration.surfaces
     2. Pick a random algorithm from policy.exploration.algorithms
     3. Pick random ncoils and order from policy choice lists
     4. Sample max_iterations uniformly from policy range
     5. Sample each weight from a log-uniform distribution over its range
     6. Sample each threshold from a log-uniform distribution over its range
     7. Assemble a complete case_config from scratch
     8. Hash → reject if duplicate
     9. Validate → reject if invalid
    10. Accept into batch

Explore is a global search across the full parameter space, designed to discover
entirely new promising regions.

*Backfill:*

If after both phases the batch is still short (e.g. too many duplicates or
validation failures), additional explore cases are generated without the novelty
filter until the batch is full.

**Step 6 — Write pending cases**

Each accepted case is written as ``cases/pending/<case_id>.json``.  The CI
commit step then pushes these to ``main``, which triggers the runner workflow
to execute them.

**Step 7 — Runner execution** (``stellcoilbench run-ci-case``)

The CI workflow (in ``update-db-self-hosted.yml``) picks up all
``cases/pending/*.json`` files and launches up to 8 in parallel:

1. Validate the JSON against resource caps.
2. Write a ``case.yaml`` from the embedded ``case_config``.
3. Run ``optimize_coils()`` with a hard iteration cap of 10,000.
4. Write ``cases/done/<case_id>/summary.json`` with metrics, timing, and the
   original config (always — even on failure).
5. Remove the pending JSON file.
6. Commit all results and push to ``main``.

That push may trigger another workflow run, but since ``cases/pending/`` is
now empty the runner will find nothing and the proposer (on the next cron tick)
will start from Step 2 with the newly recorded results.

.. _decision-flowchart:

**Decision flowchart (summary)**

.. code-block:: text

   propose_batch.py invoked (cron / dispatch)
     │
     ├── PAUSE_AUTORUN exists?  ──yes──►  exit 0
     │
     ├── cases/pending/ non-empty?  ──yes──►  exit 0  (batch barrier)
     │
     ├── build context from cases/done/*/summary.json
     │     └── failure_stats, top_parents, config_hashes, surface_counts
     │
     ├── check guardrails
     │     ├── fail_rate > 0.6?       ──yes──►  halt  (optionally write PAUSE_AUTORUN)
     │     ├── reason repeated > 12?  ──yes──►  halt
     │     └── critical class ≥ 10?   ──yes──►  halt
     │
     ├── safe mode?  (fail_rate > 0.35 but < 0.6)
     │     └── yes → reduced sigma, lower iter cap, simpler surfaces
     │
     ├── compose batch of 8:
     │     ├── 4 × exploit  (mutate random parent from top-10)
     │     │     └── jitter weights/thresholds, novelty check, validate
     │     ├── 4 × explore  (random from full parameter space)
     │     │     └── sample all params, novelty check, validate
     │     └── backfill with explore if short
     │
     └── write cases/pending/<case_id>.json  ×8


Policy File
-----------

``policy/proposer_policy.yaml`` controls every tunable aspect of the autopilot.

**Batch settings**

.. code-block:: yaml

   batch_size: 8
   exploit_fraction: 0.5     # half mutations, half explorations
   top_k_parents: 10         # how many top results to mutate from

**Mutation parameters** (exploit)

.. code-block:: yaml

   mutation:
     weight_sigma: 0.15      # lognormal std-dev for weight jitter
     threshold_sigma: 0.10   # lognormal std-dev for threshold jitter
     weight_min: 1.0e-6
     weight_max: 10000.0

The mutation operator takes a successful parent config and applies
multiplicative lognormal noise to its weights and thresholds:

.. math::

   w' = \mathrm{clamp}\bigl(w \cdot e^{\mathcal{N}(0,\,\sigma^2)},\;
        [w_\min,\, w_\max]\bigr)

**Exploration parameters**

.. code-block:: yaml

   exploration:
     length_weight_range: [0.005, 0.10]
     curvature_weight_range: [0.02, 0.30]
     surfaces: ["input.LandremanPaul2021_QA", "input.circular_tokamak", ...]
     algorithms: ["L-BFGS-B", "augmented_lagrangian"]

Exploration samples all parameters independently from log-uniform
distributions within the configured ranges.

**Guardrails**

.. code-block:: yaml

   guardrails:
     sliding_window: 30
     max_fail_rate: 0.6
     max_common_failure_count: 12
     critical_failure_classes: ["vmec_nonconverged", "nan_in_objective", "timeout"]

The proposer **stops** if any guardrail fires and (optionally) writes
a ``PAUSE_AUTORUN`` file.

**Safe mode**

When the failure rate is elevated (above ``safe_mode.threshold``, default 0.35)
but below the hard cutoff, the proposer switches to safe mode:

- Smaller mutation sigma (less aggressive jitter)
- Lower iteration cap
- Prefer simpler surfaces

See ``policy/proposer_policy.yaml`` for the full set of options.


Tools Reference
---------------

``tools/propose_batch.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~

The non-LLM proposer.  Reads recent results, checks guardrails, and writes
up to 8 validated case JSON files into ``cases/pending/``.

.. code-block:: bash

   python tools/propose_batch.py [OPTIONS]

Options:

``--batch-size N``
   Number of cases to propose (default: 8).

``--done-dir PATH``
   Directory containing completed summaries (default: ``cases/done``).

``--pending-dir PATH``
   Directory for new pending cases (default: ``cases/pending``).

``--policy PATH``
   Path to ``proposer_policy.yaml`` (default: ``policy/proposer_policy.yaml``).

``--seed N``
   Random seed for reproducibility.

``--dry-run``
   Print proposed cases to stdout without writing files.


``tools/build_context.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Builds a compact JSON context payload from recent completed results.  Used
internally by the proposer and can be used as input for an LLM proposer.

.. code-block:: bash

   python tools/build_context.py [OPTIONS]

Options:

``--done-dir PATH``
   Directory containing completed summaries (default: ``cases/done``).

``--policy PATH``
   Path to ``proposer_policy.yaml`` (default: ``policy/proposer_policy.yaml``).

``--out PATH``
   Write context JSON to this file (default: stdout).

The output includes:

- ``policy`` — batch size, resource caps
- ``failure_stats`` — fail rate, common reasons/classes
- ``top_parents`` — best feasible results with configs
- ``recent_config_hashes`` — for novelty checking
- ``surface_exploration_counts`` — how many runs per surface


CLI Command: ``run-ci-case``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runs a single CI autopilot case from a JSON file.  This is the command
the CI workflow calls for each pending case.

.. code-block:: bash

   stellcoilbench run-ci-case <case_file> [OPTIONS]

Options:

``--output-dir PATH``
   Root directory for completed case results (default: ``cases/done``).

``--policy PATH``
   Path to ``proposer_policy.yaml`` for validation.

The command:

1. Validates the JSON case against resource caps.
2. Writes a ``case.yaml`` from the embedded ``case_config``.
3. Runs coil optimisation.
4. Writes ``summary.json`` with metrics, timing, and the original config.

A ``summary.json`` is **always** written, even on failure (with
``"success": false`` and the exception details).


Running a Test Scan
-------------------

Before letting the autopilot run unattended, you should run a small scan
to verify the pipeline end-to-end.

**Step 1: Dry-run the proposer**

.. code-block:: bash

   python tools/propose_batch.py --batch-size 3 --dry-run --seed 42

This prints 3 proposed cases to stdout without writing anything.
Inspect the output to verify the configs look reasonable.

**Step 2: Propose a small batch for real**

.. code-block:: bash

   python tools/propose_batch.py --batch-size 3 --seed 42

This writes 3 JSON files into ``cases/pending/``.

**Step 3: Run the pending cases locally**

.. code-block:: bash

   for f in cases/pending/*.json; do
     echo "=== Running $f ==="
     stellcoilbench run-ci-case "$f" --output-dir cases/done \
       --policy policy/proposer_policy.yaml
     rm "$f"   # remove from pending after completion
   done

Or run them in parallel (requires GNU ``parallel``):

.. code-block:: bash

   ls cases/pending/*.json | parallel -j 3 \
     'stellcoilbench run-ci-case {} --output-dir cases/done \
       --policy policy/proposer_policy.yaml && rm {}'

**Step 4: Inspect results**

.. code-block:: bash

   # Check summaries
   for f in cases/done/*/summary.json; do
     python -c "import json, sys; d=json.load(open('$f')); \
       print(d['case_id'], 'OK' if d['success'] else 'FAIL', \
             f'score={d[\"total_score\"]:.4e}', \
             f'iters={d[\"iterations_used\"]}', \
             f'wall={d[\"walltime_sec\"]:.0f}s')"
   done

**Step 5: Build context and verify**

.. code-block:: bash

   python tools/build_context.py | python -m json.tool

Check that the context shows the completed cases, correct failure stats,
and top parents.

**Step 6: Run a second proposal round**

.. code-block:: bash

   python tools/propose_batch.py --batch-size 3 --dry-run

The proposer should now pick up the results from step 3 as potential
parents for mutations.  Verify the ``parent_ids`` field in exploit cases.


Emergency Stop
--------------

Create a ``PAUSE_AUTORUN`` file in the repo root to immediately halt
proposals:

.. code-block:: bash

   echo "Manual pause" > PAUSE_AUTORUN

Remove it to resume:

.. code-block:: bash

   rm PAUSE_AUTORUN


Tuning Recommendations
----------------------

After running 50–200 cases with the non-LLM proposer, review:

- **Failure rate**: If consistently above 30%, widen thresholds and lower
  iteration counts in the exploration ranges.
- **Score distribution**: If scores plateau, increase ``weight_sigma`` and
  ``threshold_sigma`` to explore more aggressively.
- **Surface coverage**: Check ``surface_exploration_counts`` in the context
  and adjust the surface list if some surfaces are underexplored.
- **Safe mode triggers**: If safe mode fires too often, raise
  ``safe_mode.threshold``.

Only after the non-LLM loop is stable (consistent batch completion, informative
failure reasons, guardrails trigger when expected) should you consider adding
an LLM proposer.

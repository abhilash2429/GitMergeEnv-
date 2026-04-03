# AGENTS.md - GitMergeEnv Current Codebase Guide

This document is the current-state guide for AI agents working on this repository.
It is not the old "build this from scratch" spec. Treat the codebase itself as the
source of truth, and use this file as the map that explains how the pieces fit.

If you are an agent editing this repo, read this file before making changes.

## 1. Project Identity

GitMergeEnv is an OpenEnv-style FastAPI environment where an agent resolves git
merge conflicts in Python files.

The environment is:
- backend-only
- stateful per running process, but not persistent across restarts
- deterministic on the environment/grader side
- self-contained, with no database and no external APIs inside the environment

The only external model/API usage is in `inference.py`, which is the baseline
agent runner and not part of the environment's deterministic grader logic.

## 2. Prime Directives

When changing this repo, preserve these invariants unless the user explicitly
asks to break them:

1. Do not change the core task scenarios casually.
   The conflicted files, ground truths, required elements, forbidden elements,
   and task difficulty progression are the benchmark itself.

2. Keep grading deterministic.
   No randomness, no LLM calls, no network calls, no clock-based scoring logic
   in `server/grader.py`.

3. Keep the environment endpoint semantics stable.
   `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/validate`, and `/baseline`
   are part of the public environment surface.

4. Do not remove the judge-facing Hugging Face path.
   The primary production path is still Hugging Face router with:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`

5. NVIDIA NIM is development-only support.
   It exists to avoid HF rate limits during local testing. Do not let NIM-specific
   changes break the HF path.

6. Keep the environment parseable and deployable as a Hugging Face Docker Space.

## 3. Actual Repository Layout

This is the real repo shape today.

```text
GitMergeEnv/
|- AGENTS.md
|- README.md
|- Rules.md
|- openenv.yaml
|- pyproject.toml
|- .env.example
|- .env
|- .gitignore
|- app.py
|- client.py
|- inference.py
|- models.py
|- Dockerfile
|- __init__.py
|- uv.lock
|- outputs/
|  `- .gitkeep
`- server/
   |- __init__.py
   |- app.py
   |- environment.py
   |- grader.py
   |- Dockerfile
   |- requirements.txt
   `- tasks/
      |- __init__.py
      |- task1.py
      |- task2.py
      `- task3.py
```

Notes:
- `app.py` at repo root is a compatibility shim so `uvicorn app:app` works.
- `server/app.py` is the real FastAPI entrypoint.
- `Dockerfile` at repo root is the main deployment Dockerfile for the root app shim.
- `server/Dockerfile` is an alternate server-scoped container entrypoint.
- `inference.py` is the baseline agent runner that calls the environment HTTP API.
- `outputs/` exists but is not a core runtime dependency.

## 4. High-Level Architecture

The system has four important layers:

1. Static task definitions
   Files: `server/tasks/task1.py`, `task2.py`, `task3.py`

2. Environment state machine
   File: `server/environment.py`

3. Deterministic grader
   File: `server/grader.py`

4. Model-driven baseline agent
   File: `inference.py`

The request flow for a normal episode is:

1. Client calls `/reset?task_id=...`
2. `GitMergeEnvironment.reset()` loads a hardcoded task dict
3. The current conflicted file is stored in memory
4. Agent repeatedly calls `/step` with `inspect`, `resolve`, or `submit`
5. The environment updates `self.resolutions`, rebuilds `current_file`, and
   returns shaped rewards
6. Final grading is done by `ConflictGrader.grade(current_file, task)`

## 5. Public Runtime Entry Points

### 5.1 `server/app.py`

This is the real FastAPI app.

Important implementation details:
- Uses a FastAPI lifespan hook
- Stores a single `GitMergeEnvironment` instance on `app.state.env`
- Returns typed response models from `models.py`
- `/step` is intentionally resilient and returns a negative-reward `StepResult`
  on internal exceptions instead of propagating a 500 for action-processing errors

### 5.2 `app.py`

This is only a compatibility shim:

```python
from server.app import app, main
```

It exists so both of these work:
- `uvicorn app:app`
- `python app.py`

### 5.3 `inference.py`

This is the baseline runner and debugging harness. It is not the environment.

It:
- calls the environment HTTP endpoints
- talks to an OpenAI-compatible model API
- runs all three tasks in sequence
- prints task-level scores

## 6. Environment Variables

### 6.1 Judge-facing / production path

The intended judge path is:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

`.env.example` is intentionally minimal and keeps the default model on the
Hugging Face router:

```env
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
HF_TOKEN=your_huggingface_token_here
BASE_URL=http://localhost:7860
PORT=7860
```

### 6.2 Development path

For local testing, `inference.py` and `/baseline` also accept:
- `API_KEY`
- `NVIDIA_API_KEY`

If `API_BASE_URL` points to NVIDIA NIM (`integrate.api.nvidia.com`), the key
priority is:

1. `NVIDIA_API_KEY`
2. `API_KEY`
3. `HF_TOKEN`

Otherwise the key priority is:

1. `HF_TOKEN`
2. `API_KEY`
3. `NVIDIA_API_KEY`

This exists only to make local development easier when HF rate limits are a problem.

## 7. API Surface

### `GET /`
Returns basic service metadata.

### `GET /health`
Simple health check.

### `POST /reset?task_id=task1|task2|task3`
Starts a fresh episode and returns `MergeObservation`.

### `POST /step`
Accepts `MergeAction`, returns `StepResult`.

### `GET /state`
Returns `EpisodeState`.

### `GET /tasks`
Returns task metadata plus action schema.

### `POST /grader`
Grades the current in-memory file without ending the episode.

### `POST /validate`
Runs deterministic self-checks on the grader using:
- perfect input
- empty input
- unresolved conflicted input

This endpoint should return `validation_passed: true`.

### `POST /baseline`
Imports `run_baseline()` from `inference.py` and runs the baseline agent against
all three tasks.

Credential validation in `/baseline` currently accepts:
- `HF_TOKEN`
- `API_KEY`
- `NVIDIA_API_KEY`

## 8. Models and Schema

File: `models.py`

Core models:
- `MergeAction`
- `MergeObservation`
- `MergeReward`
- `StepResult`
- `EpisodeState`
- `TaskInfo`
- `GraderResult`
- `BaselineResult`

Important current details:

1. `StepResult.reward` is a simple `float`, not a MergeReward object.

2. `MergeObservation` includes these fields (per spec):
   - `file_name: str`
   - `total_conflicts: int`
   - `resolved_conflicts: int`
   - `unresolved_conflict_ids: List[int]`
   - `current_file_preview: str`
   - `last_action_feedback: str`
   - `last_reward: float`
   - `steps_remaining: int`
   
   Note: The `hint` field was removed to match the original spec.

3. `BaselineResult` has:
   - `model_config = {"protected_namespaces": ()}`

   This suppresses the Pydantic protected namespace warning for `model_used`.

## 9. Environment State Machine

File: `server/environment.py`

Class: `GitMergeEnvironment`

Important state fields:
- `episode_id`
- `task_id`
- `task`
- `step_count`
- `done`
- `total_reward`
- `current_file`
- `original_file`
- `ground_truth_file`
- `ground_truth_blocks`
- `resolutions`
- `conflict_blocks`

### 9.1 Reset behavior

`reset(task_id)`:
- validates the task id
- clears all state
- loads the task dict from `ALL_TASKS`
- parses conflict blocks from the raw conflicted file
- sets `current_file = original_file`

### 9.2 Conflict parsing

Conflict blocks are extracted with regex and stored as dicts containing:
- `id`
- `head_content`
- `incoming_content`
- `full_marker_text`
- `start`
- `end`

### 9.3 Step behavior

`step(action)` does this in order:

1. rejects episode-already-done
2. rejects no-active-episode
3. increments `step_count`
4. routes to:
   - `_handle_inspect`
   - `_handle_resolve`
   - `_handle_submit`
5. subtracts `STEP_PENALTY`
6. updates `total_reward`
7. terminates if step limit reached
8. returns observation + reward + done + info

`STEP_PENALTY = 0.01`

This means the visible net reward is always action reward minus `0.01`.

Examples:
- inspect action reward `+0.02` becomes visible `+0.01`
- perfect resolve `+0.15` becomes visible `+0.14`

### 9.4 Inspect action

`_handle_inspect()`:
- requires `conflict_id`
- checks range
- returns HEAD and INCOMING block contents
- includes current resolution if the block is already resolved
- reward before step penalty: `+0.02`

### 9.5 Resolve action

`_handle_resolve()`:
- requires `conflict_id`
- requires non-empty resolution
- checks id range
- rejects suspiciously short resolutions
- rejects conflict markers in submitted resolution
- rejects unusually long resolutions
- rejects duplicate identical re-resolution
- grades the single block using `ConflictGrader.grade_block()`
- stores the resolution
- rebuilds the entire file via `_apply_resolutions()`

Immediate block reward bands before step penalty:
- `1.0` block score -> `+0.15`
- `>= 0.7` -> `+0.08`
- `>= 0.4` -> `+0.02`
- `> 0.0` -> `-0.02`
- `0.0` -> `-0.08`

Validation edge cases currently implemented:
- too short -> `resolution_too_short`
- conflict markers -> `resolution_contains_markers`
- too long -> `resolution_too_long`
- duplicate identical resolution -> `duplicate_resolution`

### 9.6 Submit action

`_handle_submit()`:
- runs the full deterministic grader
- computes unresolved penalty
- computes efficiency bonus
- computes consistency bonus
- returns terminal reward and component info

Current terminal logic:
- unresolved penalty = `0.10 * unresolved_count`
- efficiency bonus = `0.05` only if:
  - final score `>= 0.9`
  - steps used `<= 50%` of max steps
- consistency bonus comes from `_check_resolution_consistency()`

### 9.7 Consistency bonus

`_check_resolution_consistency()` is a shaped reward helper for multi-block tasks.

It looks for mixed old/new patterns in all submitted resolutions:
- `Session(engine)` vs `cursor.execute`
- `CustomError` vs `ValueError`
- `import logging` vs `print(`

Current bonus rules:
- no mixed pairs -> `+0.08`
- one mixed pair -> `+0.03`
- multiple mixed pairs -> `+0.00`

### 9.8 Hint system

`_build_observation()` adds a `hint` after the agent has consumed more than half
its step budget and still has unresolved blocks.

This is an observation-only helper and does not directly change scoring.

## 10. Grader Internals

File: `server/grader.py`

Class: `ConflictGrader`

This file must remain deterministic.

### 10.1 Current grading pipeline

`grade(agent_file, task)` currently works like this:

1. Empty file:
   - returns `0.0`
   - sets only `parses_cleanly` and `no_conflict_markers`

2. File still containing conflict markers:
   - returns `0.0`
   - markers are an immediate hard failure path

3. File without markers but syntactically invalid:
   - returns low/capped score based only on parse component

4. Valid resolved Python file:
   - computes weighted components
   - applies forbidden-element penalty multiplicatively

### 10.2 Components used today

Weighted components (defined in task grader_weights):
- `parses_cleanly`
- `no_conflict_markers`
- `block_match`
- `required_elements`
- `architectural_consistency` (task 3 only)

Unweighted but applied multiplicatively:
- `forbidden_penalty`

Note: The `structural_similarity` component was removed to match the original spec.
All task weights now allocate full weight to `block_match` instead of splitting
between block_match and structural_similarity.

### 10.3 Block matching

Block scoring is not AST-based.
It is token-presence scoring over each `ground_truth_block`.

Current behavior:
- collects significant tokens from each ground truth block
- checks how many appear in the final agent file
- averages that score across blocks

### 10.4 Grade-block behavior

`grade_block(agent_block, ground_truth_block)` is used for immediate resolve-step
feedback inside the environment.

Current implementation:
- exact normalized match -> `1.0`
- otherwise Jaccard token overlap scaled by `0.9`

So a "pretty good" block can score around `0.7` to `0.9`.

### 10.5 Forbidden penalty

`_compute_forbidden_penalty()` reduces the score multiplier by `0.15` for each
forbidden element found, with a floor of `0.10`.

This is multiplicative, not additive.

## 11. Task Canon

The tasks are hardcoded dictionaries in `server/tasks/`.
These are benchmark artifacts, not generated data.

### 11.1 Task 1

File: `server/tasks/task1.py`

Summary:
- file: `processor.py`
- max steps: `6`
- conflicts: `1`
- difficulty: easy

Intent:
- Developer A renamed `user_data` -> `user_info`
- Developer B added `timeout=30`
- correct merge keeps both

Common failure modes:
- preserving the old variable name in `transform(...)`
- dropping the timeout parameter

### 11.2 Task 2

File: `server/tasks/task2.py`

Summary:
- file: `data_service.py`
- max steps: `12`
- conflicts: `3`
- difficulty: medium

Current actual block order in the file:
0. imports
1. class docstring
2. method body

This order matters. Do not assume an older prose description that says the method
body is block 1 and the docstring is block 2. The code is the truth.

Intent:
- merge custom exception usage with structured logging
- preserve valid Python class/docstring structure

Critical failure mode:
- agents often merge semantics correctly but break syntax by:
  - turning the docstring into plain prose without triple quotes
  - losing indentation under `class DataService:`
  - losing indentation under `if not record:`

This task is where syntax preservation matters most.

### 11.3 Task 3

File: `server/tasks/task3.py`

Summary:
- file: `db_access.py`
- max steps: `18`
- conflicts: `5`
- difficulty: hard

This task is architecturally dependent across all blocks.

Current actual scenario:
- ORM migration wins
- raw sqlite path must be fully removed
- the file includes:
  - `datetime`
  - `DateTime`
  - `deleted_at`
  - `create_users(users: list[dict])`
  - soft delete using `deleted_at = datetime.utcnow()`

Important note:
- older specs for this repo described a different task 3 with `create_user(...)`
  and hard delete semantics
- that is no longer the current benchmark
- the actual task file in the repo is the source of truth

Common failure modes:
- mixing `Session(engine)` with `cursor.execute`
- carrying over `sqlite3.connect`
- breaking consistency across blocks
- partially resolving blocks independently instead of choosing the ORM path globally

## 12. Current Grader Weights

These live in the task dicts and must sum correctly.

### Task 1
- parses_cleanly: `0.15`
- no_conflict_markers: `0.10`
- block_match: `0.40`
- required_elements: `0.25`
- structural_similarity: `0.10`

### Task 2
- parses_cleanly: `0.10`
- no_conflict_markers: `0.10`
- block_match: `0.40`
- required_elements: `0.30`
- structural_similarity: `0.10`

### Task 3
- parses_cleanly: `0.05`
- no_conflict_markers: `0.10`
- block_match: `0.35`
- required_elements: `0.25`
- architectural_consistency: `0.15`
- structural_similarity: `0.10`

## 13. Baseline Inference Agent

File: `inference.py`

This script is both:
- the baseline runner
- the easiest way to debug model behavior against the environment

### 13.1 Provider behavior

The script uses the OpenAI Python client against an OpenAI-compatible endpoint.

Supported paths:
- Hugging Face router
- NVIDIA NIM

### 13.2 Model name normalization

Current normalization:
- `meta/llama-3_3-70b-instruct` -> `meta/llama-3.3-70b-instruct`

This exists because NIM returns `404 page not found` for the underscore variant.

### 13.3 NIM-specific behavior

If `API_BASE_URL` contains `integrate.api.nvidia.com`, the script:
- treats the provider as NVIDIA NIM
- uses `stream=True`
- uses:
  - `temperature=0.2`
  - `top_p=0.7`
  - `max_tokens=1024`

Otherwise it uses the non-streaming HF-compatible call with:
- `temperature=0.0`
- `max_tokens=500`

### 13.4 Retry and fallback

`_create_completion_text()` retries rate limits up to 3 attempts with short backoff.

If model generation still fails inside `run_task()`:
- the step falls back to:
  - `{"action_type": "inspect", "conflict_id": 0}`

If an entire task run crashes:
- `run_baseline()` catches it
- records score `0.0`
- continues to remaining tasks

### 13.5 Action parsing

The script is defensive about messy model output:
- strips `<think>...</think>`
- strips markdown fences
- attempts JSON parse
- falls back to regex extraction for malformed JSON
- handles multiline `resolution` payloads

### 13.6 Resolution normalization

This is a critical current feature.

Before sending a `resolve` action to the environment, the script normalizes the
resolution using the context captured from an earlier `inspect`.

It currently repairs:
- block indentation
- docstring blocks that lost triple quotes

This was added because models, especially on task 2, frequently:
- output prose instead of a real docstring
- lose indentation inside class and `if` blocks
- produce semantically good but syntactically invalid Python

### 13.7 Pre-submit guard

If all conflicts are resolved, the script does not blindly submit.

It first:
1. parses `current_file_preview` with `ast.parse`
2. calls `/grader`

If parse fails:
- it prints `Parse error: ...`
- forces an inspect on the weakest block instead of submitting

If parse succeeds but the pre-submit grader score is still low:
- it also forces additional inspection instead of submitting

This guard exists to avoid ending runs with an obviously broken file.

### 13.8 Current terminal output policy

The user requested simplified logging.
The script currently prints:
- task headers
- step action summary
- reward / resolved count
- `Block score: ...` when available
- `Parse error: ...` when relevant
- final grader scores

It intentionally does not dump full raw model outputs or giant JSON traces anymore.

## 14. Deployment and Containers

### 14.1 Root `Dockerfile`

This is the main deployable Dockerfile.

It:
- installs Python deps from `server/requirements.txt`
- copies root `app.py`, `models.py`, `inference.py`, and `server/`
- exposes port `7860`
- runs:
  - `uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1`

### 14.2 `server/Dockerfile`

This is a server-scoped variant.

It runs:
- `uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1`

### 14.3 Root app shim

The root Dockerfile relies on the root `app.py` shim.
Do not remove that file unless you also change the container entrypoint.

## 15. Dependencies

The current repo is not using the older pinned versions from the original spec.
The actual current versions are the ones in:
- `pyproject.toml`
- `server/requirements.txt`

Current notable versions:
- `fastapi==0.115.12`
- `uvicorn[standard]==0.30.6`
- `pydantic==2.8.2`
- `openenv-core==0.2.1`
- `python-dotenv==1.0.1`
- `httpx==0.28.1`
- `openai==2.7.2`

If you update these, update both files consistently.

## 16. Known Spec Drift and Current Truth

There has been drift between old instructions, older hackathon specs, README text,
and the current running code.

Important examples:

1. The repo uses `inference.py` for the baseline agent runner.

2. The old task 3 spec described a different scenario.
   Current task 3 uses:
   - `create_users(...)`
   - soft delete via `deleted_at`
   - `datetime.utcnow()`

3. `MergeObservation` now has a `hint` field.
   `openenv.yaml` and some documentation still describe the older observation shape.

4. README reward descriptions are conceptual.
   The actual visible step rewards include the `-0.01` step penalty.

When in doubt:
- runtime code beats stale prose
- task files beat older written summaries
- FastAPI response models beat older spec comments

## 17. Known Failure Modes

These are the mistakes future agents should expect to see.

### 17.1 Task 2 syntax corruption

Most common:
- docstring turned into plain text
- indentation lost under class body
- indentation lost under `if not record:`

The grader is not wrong when these score `0.0`; the file genuinely does not parse.

### 17.2 Task 3 architectural mixing

Common bad resolution:
- preserving some ORM blocks
- keeping some raw SQL blocks

This usually harms:
- `required_elements`
- `architectural_consistency`
- `forbidden_penalty`

### 17.3 NIM rate limits

NVIDIA NIM can return `429 Too Many Requests`.
Current mitigation:
- short retry/backoff in `inference.py`
- fallback inspect if calls still fail

### 17.4 Wrong NIM model identifier

Bad:
- `meta/llama-3_3-70b-instruct`

Good:
- `meta/llama-3.3-70b-instruct`

## 18. Local Development Workflow

Typical workflow:

1. Start server
   - `uvicorn server.app:app --host 0.0.0.0 --port 7860`

2. Validate grader
   - `POST /validate`

3. Run baseline locally
   - `python inference.py`

4. If HF limits are a problem, switch only the inference path to NVIDIA NIM

Useful checks:
- `/health`
- `/tasks`
- `/reset?task_id=task1`
- `/grader`
- `/validate`

There is no formal pytest suite in the repo right now; `/validate` is the main
built-in correctness smoke test for the grader.

## 19. Modification Rules for Future Agents

If you are asked to change the code:

### Safe changes
- documentation updates
- inference prompt/agent behavior
- local developer ergonomics
- endpoint docs
- deployment wiring
- debug output changes
- baseline-runner improvements that do not break the HF judge path

### High-risk changes
- task dict contents
- grader formulas
- reward shaping
- observation schema
- endpoint response models
- Docker entrypoints

If you change any of the high-risk areas, verify all of these:
- `/validate` still passes
- `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/baseline` still work
- root Dockerfile still starts the app
- judge-facing HF environment-variable flow still works

## 20. If You Need to Touch Specific Files

### `server/tasks/*.py`
Only change if the user explicitly wants benchmark/task edits.
Otherwise treat as locked benchmark data.

### `server/grader.py`
Keep deterministic.
If you add a new weighted component:
- update each task's `grader_weights`
- keep weights coherent
- keep `/validate` expectations realistic

### `server/environment.py`
Keep reward shaping understandable.
Do not make invalid actions crash the API.

### `models.py`
If schema changes:
- update FastAPI response models
- update docs
- update `openenv.yaml` if needed

### `inference.py`
Remember there are two audiences:
- judges: Hugging Face router
- developers: local debugging and NIM

Do not reintroduce complicated provider branching that breaks the simple judge path.

### `Dockerfile` and `app.py`
These support root-level deploy/run compatibility.
Do not remove them casually.

## 21. Minimal Mental Model for New Agents

If you only remember a few things, remember these:

1. Tasks are hardcoded benchmark artifacts.
2. Environment and grader must stay deterministic.
3. Task 2 fails mostly because of syntax preservation mistakes.
4. Task 3 fails mostly because of cross-block architectural inconsistency.
5. `inference.py` is the non-deterministic baseline agent and local debug harness.
6. Hugging Face is the production path; NVIDIA NIM is dev-only support.
7. Root `app.py` and root `Dockerfile` are intentional compatibility layers.

## 22. Source of Truth Order

If documents disagree, trust them in this order:

1. current Python runtime files
2. task dicts in `server/tasks/`
3. FastAPI models and endpoint code
4. Dockerfiles
5. README / openenv.yaml / older prose docs

This file exists to reduce that drift, but the code still wins.

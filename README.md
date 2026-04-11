---
title: GitMergeEnv
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# GitMergeEnv

GitMergeEnv is a reinforcement learning environment for resolving git merge conflicts in Python files.

The core idea is simple: instead of treating merge conflict resolution as a one-shot text editing problem, this project treats it as a sequential decision-making problem. An agent inspects conflict blocks, resolves them one at a time, and receives reward based on both local progress and the quality of the final merged file.

This repository contains:

- a FastAPI environment server
- a deterministic grader
- three fixed benchmark tasks
- a baseline inference runner

## At a Glance

| Item | Value |
|---|---|
| Project type | Reinforcement learning environment |
| Domain | Python merge conflict resolution |
| Interaction style | Multi-step `reset -> step -> submit` |
| Grading | Deterministic, programmatic, no LLM in grader |
| Task count | 3 |
| Difficulty structure | Easy, Medium, Hard |
| Main differentiator | Rewarding whole-file architectural consistency |

## What This Project Is

| Question | Answer |
|---|---|
| Is this a model? | No. It is an environment that an agent interacts with. |
| Is this a benchmark only? | No. It includes environment state, reward logic, and episode flow. |
| Is the grader LLM-based? | No. The grader is deterministic and fully programmatic. |
| Why use RL here? | Because the correct resolution for one block can depend on decisions made in other blocks. |

In practical terms, GitMergeEnv is a controlled environment that shows an agent a conflicted Python file, lets it take structured actions, and scores the results in a reproducible way.

## Why This Problem Matters

Merge conflicts are common in software development, but the hardest failures are not always syntax errors. The more dangerous failures are merges that look valid while combining incompatible design choices.

Example:

- one developer migrates a module from raw `sqlite3` to SQLAlchemy ORM
- another developer adds new functionality using the old raw SQL style
- a naive merge can preserve both
- the file still parses
- the final result is logically inconsistent

That class of error is exactly what GitMergeEnv is built to model.

## Why Reinforcement Learning Fits

Supervised learning works well when each answer can be scored independently. This problem does not always work that way.

In merge conflict resolution:

- a good resolution for block 4 may depend on what happened in block 0
- a locally plausible decision can hurt global file consistency
- the full quality of a decision sequence is often visible only at the end of the episode

That makes the task a natural fit for reinforcement learning:

- the agent acts sequentially
- the environment returns intermediate reward
- the final merged file receives a delayed terminal score

## How The Environment Works

### Episode flow

| Step | What happens | Why it matters |
|---|---|---|
| 1 | Client calls `POST /reset?task_id=...` | Starts a fresh episode with one conflicted file |
| 2 | Environment returns an observation | Agent sees file state and conflict metadata |
| 3 | Agent calls `POST /step` with `inspect` or `resolve` | Agent gathers context or proposes a fix |
| 4 | Environment returns new observation and reward | Agent gets immediate learning signal |
| 5 | Agent calls `submit` when ready | Environment runs terminal grading |

### Action space

| Field | Type | Required | Description |
|---|---|---|---|
| `action_type` | string | Yes | One of `inspect`, `resolve`, `submit` |
| `conflict_id` | integer | For `inspect` and `resolve` | Conflict block index |
| `resolution` | string | For `resolve` | Proposed merged content |

### Observation space

| Field | Type | Meaning |
|---|---|---|
| `file_name` | string | Current task file |
| `total_conflicts` | integer | Number of conflict blocks in the task |
| `resolved_conflicts` | integer | Number of blocks already resolved |
| `unresolved_conflict_ids` | list[int] | Remaining unresolved block IDs |
| `current_file_preview` | string | Current file state with applied resolutions |
| `last_action_feedback` | string | Human-readable explanation of the last action result |
| `last_reward` | float | Reward from the last action |
| `steps_remaining` | integer | Remaining step budget |

### What each action does

| Action | Environment behavior | Typical use |
|---|---|---|
| `inspect` | Shows HEAD and INCOMING versions for one block | Gather context before editing |
| `resolve` | Applies a proposed resolution and scores it locally | Make progress on one conflict |
| `submit` | Grades the current file and ends the episode | Finish the merge |

### Example interaction

```text
POST /reset?task_id=task3
-> returns conflicted file and metadata

POST /step
{"action_type":"inspect","conflict_id":0}
-> returns block details and small positive reward

POST /step
{"action_type":"resolve","conflict_id":0,"resolution":"..."}
-> applies resolution and returns immediate reward

POST /step
{"action_type":"submit"}
-> returns terminal score and ends episode
```

## Benchmark Tasks

All tasks are fixed and reproducible. There is no random task generation.

| Task | File | Difficulty | Conflicts | Max Steps | Main reasoning challenge |
|---|---|---:|---:|---:|---|
| `task1` | `processor.py` | Easy | 1 | 6 | Combine two compatible edits in one function |
| `task2` | `data_service.py` | Medium | 3 | 12 | Preserve logging and exception migration across multiple blocks |
| `task3` | `db_access.py` | Hard | 5 | 18 | Maintain one architecture across the entire file |

### Task details

| Task | Scenario | Correct behavior |
|---|---|---|
| `task1` | One branch renames `user_data` to `user_info`; the other adds `timeout=30` | Keep both the rename and the new argument |
| `task2` | One branch migrates to `CustomError`; the other adds structured logging | Keep both logging and exception migration without breaking indentation or docstrings |
| `task3` | One branch migrates raw SQL code to SQLAlchemy ORM; the other adds new raw-SQL features | Resolve all blocks consistently in ORM style |

Task 3 is the most important benchmark in the suite because it tests whether the agent can reason about architectural dependency across multiple blocks.

## Reward Design

The reward function is designed to make the environment learnable.

If reward were only given at the end of the episode, the task would be much harder to train on. GitMergeEnv therefore mixes local reward with terminal reward.

### Per-step reward

| Signal | Value | Purpose |
|---|---:|---|
| Step penalty | `-0.01` | Discourage unnecessary actions |
| Inspect reward | `+0.02` before step penalty | Reward useful information gathering |
| Resolve exact match | `+0.15` before step penalty | Reward perfect block resolution |
| Resolve high partial match | `+0.08` before step penalty | Reward strong partial progress |
| Resolve low partial match | `+0.02` before step penalty | Preserve a small positive learning signal |
| Resolve near-zero match | `-0.02` before step penalty | Penalize weak guesses |
| Resolve no match | `-0.08` before step penalty | Penalize incorrect resolutions |

### Repeated resolve penalty

| Attempt number | Multiplier |
|---|---:|
| 1 | `1.0` |
| 2 | `0.7` |
| 3+ | `0.4` |

This makes repeated low-quality retries less attractive.

### Terminal reward

| Component | Formula |
|---|---|
| Base score | deterministic grader score |
| Unresolved penalty | `-0.10 * unresolved_count` |
| Efficiency bonus | `+0.05` for high score in at most half the step budget |
| Consistency bonus | `+0.08`, `+0.03`, or `+0.00` depending on mixed architectural patterns |

### Reward philosophy

| Design choice | Why it exists |
|---|---|
| Dense local signal | Helps the agent learn before the end of the episode |
| Terminal file-level score | Keeps the real objective tied to whole-file quality |
| Repetition decay | Prevents reward farming |
| Consistency bonus | Encourages coherent full-file design, not only local fixes |

## Deterministic Grader

The grader in [`server/grader.py`](server/grader.py) is deterministic.

That means:

- no randomness
- no judge-model calls
- same input always produces the same score

### Block-level grading

| Rule | Effect |
|---|---|
| Exact normalized match | Returns `1.0` |
| Non-exact match | Uses line-overlap F1-style scoring |
| Non-exact cap | Maximum `0.85` |

### Terminal grading components

| Component | task1 | task2 | task3 |
|---|---:|---:|---:|
| `no_conflict_markers` | 0.15 | 0.10 | 0.05 |
| `block_match` | 0.55 | 0.50 | 0.40 |
| `required_elements` | 0.30 | 0.40 | 0.25 |
| `architectural_consistency` | - | - | 0.25 |
| `indentation_consistency` | - | - | 0.05 |

### Additional grading rules

| Rule | Effect |
|---|---|
| Parse failure | Multiplies score by `0.5` |
| Forbidden elements | Applies multiplicative penalty |
| Empty file | Floors to minimum terminal score |
| Final grader range | Clamped to `(0.01, 0.99)` |

## API Reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/tasks` | Return available tasks and action schema |
| `POST` | `/reset?task_id=task1` | Start a new episode |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Return current episode state |
| `POST` | `/grader` | Score current file without ending episode |
| `POST` | `/validate` | Run deterministic self-checks |
| `POST` | `/baseline` | Run the baseline agent across all tasks |
| `GET` | `/health` | Health check |
| `GET` | `/docs-home` | Project documentation page |

## Recorded Inference Run

The following results are from a real run of `python inference.py`.

### Score summary

| Task | Score |
|---|---:|
| `task1` | `0.6554` |
| `task2` | `0.9900` |
| `task3` | `0.8500` |
| Average | `0.8318` |

### Full output

```text
[inference] Base URL: https://router.huggingface.co/v1
[inference] Model: Qwen/Qwen2.5-72B-Instruct
[START] task=task1 env=GitMergeEnv model=Qwen/Qwen2.5-72B-Instruct
[inference] Task started. File: processor.py, Conflicts: 1
[STEP] step=1 action=inspect(0) reward=0.01 done=false error=null
[inference]   Resolved: 0/1 | Steps left: 5
[STEP] step=2 action=resolve(0) reward=0.01 done=false error=null
[inference]   Resolved: 1/1 | Steps left: 4
[inference]   Block score: 0.5667
[inference] Step 2: All blocks resolved — forcing submit
[STEP] step=3 action=submit reward=0.65 done=true error=null
[inference]   Resolved: 1/1 | Steps left: 3
[inference] Episode done. Final grader score: 0.6554
[END] success=true steps=3 score=0.655 rewards=0.01,0.01,0.65
[START] task=task2 env=GitMergeEnv model=Qwen/Qwen2.5-72B-Instruct
[inference] Task started. File: data_service.py, Conflicts: 3
[STEP] step=1 action=inspect(0) reward=0.01 done=false error=null
[inference]   Resolved: 0/3 | Steps left: 11
[STEP] step=2 action=resolve(0) reward=0.07 done=false error=null
[inference]   Resolved: 1/3 | Steps left: 10
[inference]   Block score: 0.8500
[STEP] step=3 action=inspect(1) reward=0.01 done=false error=null
[inference]   Resolved: 1/3 | Steps left: 9
[STEP] step=4 action=resolve(1) reward=0.14 done=false error=null
[inference]   Resolved: 2/3 | Steps left: 8
[inference]   Block score: 1.0000
[STEP] step=5 action=inspect(2) reward=0.01 done=false error=null
[inference]   Resolved: 2/3 | Steps left: 7
[STEP] step=6 action=resolve(2) reward=0.14 done=false error=null
[inference]   Resolved: 3/3 | Steps left: 6
[inference]   Block score: 1.0000
[STEP] step=7 action=submit reward=1.06 done=true error=null
[inference]   Resolved: 3/3 | Steps left: 5
[inference] Episode done. Final grader score: 0.9900
[END] success=true steps=7 score=0.990 rewards=0.01,0.07,0.01,0.14,0.01,0.14,1.06
[START] task=task3 env=GitMergeEnv model=Qwen/Qwen2.5-72B-Instruct
[inference] Task started. File: db_access.py, Conflicts: 5
[STEP] step=1 action=inspect(0) reward=0.01 done=false error=null
[inference]   Resolved: 0/5 | Steps left: 17
[STEP] step=2 action=resolve(0) reward=0.07 done=false error=null
[inference]   Resolved: 1/5 | Steps left: 16
[inference]   Block score: 0.7000
[STEP] step=3 action=inspect(1) reward=0.01 done=false error=null
[inference]   Resolved: 1/5 | Steps left: 15
[STEP] step=4 action=resolve(1) reward=0.14 done=false error=null
[inference]   Resolved: 2/5 | Steps left: 14
[inference]   Block score: 1.0000
[STEP] step=5 action=inspect(2) reward=0.01 done=false error=null
[inference]   Resolved: 2/5 | Steps left: 13
[STEP] step=6 action=resolve(2) reward=0.14 done=false error=null
[inference]   Resolved: 3/5 | Steps left: 12
[inference]   Block score: 1.0000
[STEP] step=7 action=inspect(3) reward=0.01 done=false error=null
[inference]   Resolved: 3/5 | Steps left: 11
[STEP] step=8 action=resolve(3) reward=0.14 done=false error=null
[inference]   Resolved: 4/5 | Steps left: 10
[inference]   Block score: 1.0000
[STEP] step=9 action=inspect(4) reward=0.01 done=false error=null
[inference]   Resolved: 4/5 | Steps left: 9
[STEP] step=10 action=resolve(4) reward=0.14 done=false error=null
[inference]   Resolved: 5/5 | Steps left: 8
[inference]   Block score: 1.0000
[STEP] step=11 action=submit reward=0.92 done=true error=null
[inference]   Resolved: 5/5 | Steps left: 7
[inference] Episode done. Final grader score: 0.8500
[END] success=true steps=11 score=0.850 rewards=0.01,0.07,0.01,0.14,0.01,0.14,0.01,0.14,0.01,0.14,0.92

[inference] BASELINE RESULTS
[inference]   task1: 0.6554
[inference]   task2: 0.9900
[inference]   task3: 0.8500
[inference]   Average: 0.8318
```

## Running The Project

### Environment variables

Use `.env.example` as the starting point:

```bash
API_BASE_URL=https://router.huggingface.co/v1
API_KEY=your_huggingface_token_here
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
BASE_URL=http://localhost:7860
PORT=7860
```

### Run locally with Docker

```bash
docker build -t gitmergeenv .
docker run -p 7860:7860 --env-file .env gitmergeenv
```

### Basic API checks

```bash
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl -X POST "http://localhost:7860/reset?task_id=task1"
```

### Run the baseline agent

```bash
python inference.py
```

## Hugging Face Space Usage

If the project is deployed as a Hugging Face Space, use the Space URL as `BASE_URL`.

Example:

```bash
BASE_URL=https://abhilash2429-gitmergeenv.hf.space
```

Then:

```bash
python inference.py
```

API calls work the same way:

```bash
curl https://abhilash2429-gitmergeenv.hf.space/health
curl https://abhilash2429-gitmergeenv.hf.space/tasks
curl -X POST "https://abhilash2429-gitmergeenv.hf.space/reset?task_id=task1"
```

## Repository Structure

| Path | Purpose |
|---|---|
| `inference.py` | Baseline agent runner |
| `models.py` | Request and response models |
| `openenv.yaml` | Environment metadata |
| `server/app.py` | FastAPI app and HTTP endpoints |
| `server/environment.py` | Episode state and reward logic |
| `server/grader.py` | Deterministic grader |
| `server/tasks/task1.py` | Easy task definition |
| `server/tasks/task2.py` | Medium task definition |
| `server/tasks/task3.py` | Hard task definition |

## Submission Strengths

| Strength | Why it matters |
|---|---|
| Real-world problem | Models a real class of merge failures rather than a toy task |
| Learnable reward | Gives agents useful feedback before the episode ends |
| Deterministic scoring | Makes evaluation reproducible and judge-friendly |
| Curriculum structure | Tasks increase difficulty in a meaningful way |
| Whole-file coherence | Rewards architectural consistency, not only local patch quality |


---
title: GitMergeEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# GitMergeEnv — Git Merge Conflict Resolution RL Environment

> An RL environment where agents learn to resolve merge conflicts with semantic correctness —
> rewarded for architectural consistency across the whole file, not just individual blocks.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)]()

## The Problem

Merge conflicts are among the most common sources of subtle bugs introduced during collaborative development. Automated resolution tools fail because they optimize for syntactic merging — they cannot detect when a developer mixes SQLAlchemy ORM patterns with raw sqlite3 calls across resolved blocks, producing code that parses cleanly but is architecturally broken. No diff tool catches this class of error. This environment trains agents to make semantically correct, architecturally consistent resolution decisions by rewarding global file coherence, not just local block correctness.

## Why RL

The correct resolution in a merge conflict often depends on understanding the global architectural intent of the file, not just the two conflicting local versions. A reward signal that reflects full-file consistency — only fully realized at submit time — is uniquely suited to RL. Supervised learning cannot model this because the label for any single block depends on how all other blocks are resolved. The agent must learn to plan across the entire episode.

## Environment Overview

### Action Space

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `action_type` | string | yes | One of: `inspect`, `resolve`, `submit` |
| `conflict_id` | integer | for inspect/resolve | Index of the conflict block (0-based) |
| `resolution` | string | for resolve | Proposed resolution text |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `file_name` | string | Name of the file being resolved |
| `total_conflicts` | integer | Total conflict blocks in this task |
| `resolved_conflicts` | integer | How many blocks resolved so far |
| `unresolved_conflict_ids` | list[int] | IDs of remaining unresolved blocks |
| `current_file_preview` | string | Current state of the file with resolutions applied |
| `last_action_feedback` | string | Human-readable feedback from last action |
| `last_reward` | float | Reward from last action |
| `steps_remaining` | integer | Steps left before forced termination |

### Episode Flow

```
POST /reset {"task_id": "task1"}
  → returns initial observation

while not done:
    POST /step {"action_type": "inspect", "conflict_id": 0}
    POST /step {"action_type": "resolve", "conflict_id": 0, "resolution": "..."}

POST /step {"action_type": "submit"}
  → returns terminal reward + grader score
```

## Tasks

| Task | File | Conflicts | Max Steps | Difficulty | Scenario |
|------|------|-----------|-----------|------------|----------|
| `task1` | `processor.py` | 1 | 6 | 🟢 Easy | Variable rename + new argument. Must preserve both developer A's rename (`user_data` → `user_info`) and developer B's new parameter (`timeout=30`). |
| `task2` | `data_service.py` | 3 | 12 | 🟡 Medium | `CustomError` migration + `logging` addition. Contains docstring syntax traps and indentation-sensitive blocks. Wrong indentation = parse failure. |
| `task3` | `db_access.py` | 5 | 18 | 🔴 Hard | SQLAlchemy ORM vs raw `sqlite3`. ORM must win globally across all 5 blocks. Mixing patterns across resolutions is detected and penalized via consistency scoring. |

> All tasks use fixed, hardcoded scenarios with precomputed ground truth. Nothing is randomly generated. Results are fully reproducible across all runs.

## Reward Design

The reward system is multi-component and non-binary. Partial credit exists at every level.

### Per-Step Signals (every action)

| Signal | Value |
|--------|-------|
| Step penalty | `-0.01` per action (all action types) |
| Inspect reward (before penalty) | `+0.02` |

### Resolve Signals (immediate, before step penalty)

| Block Match Quality | Reward |
|--------------------|--------|
| Exact match (score = 1.0) | `+0.15` |
| High partial (score ≥ 0.7) | `+0.08` |
| Low partial (score ≥ 0.4) | `+0.02` |
| Near-zero match (score > 0.0) | `-0.02` |
| No match (score = 0.0) | `-0.08` |

**Repetition penalty** — re-resolving the same block multiplies the reward:
- 1st attempt: `×1.0`
- 2nd attempt: `×0.7`
- 3rd+ attempt: `×0.4`

**Syntax warning** — if all blocks resolved but file fails to parse, positive rewards are multiplied by `×0.3`.

### Terminal Signals (on submit)

| Component | Formula |
|-----------|---------|
| Base terminal reward | Grader score (0.01–0.99) |
| Unresolved penalty | `-0.10 × unresolved_count` |
| Efficiency bonus | `+0.05` if score ≥ 0.9 AND steps ≤ half max_steps |
| Consistency bonus (0 mixed patterns) | `+0.08` |
| Consistency bonus (1 mixed pattern) | `+0.03` |
| Consistency bonus (2+ mixed patterns) | `+0.00` |

> All final scores are clamped to the strict open interval **(0.01, 0.99)**. The grader never returns exactly 0 or exactly 1.

## Grader Design

The grader (`server/grader.py`) is fully deterministic and programmatic — no LLM calls.

**Per-block grading (`grade_block`):**
- Exact normalized match → `1.0`
- Otherwise → line-level F1 precision/recall score, capped at `0.85`

**Terminal grading (`grade`) — weighted components:**

| Component | task1 | task2 | task3 |
|-----------|-------|-------|-------|
| `no_conflict_markers` | 0.15 | 0.10 | 0.05 |
| `block_match` | 0.55 | 0.50 | 0.40 |
| `required_elements` | 0.30 | 0.40 | 0.25 |
| `architectural_consistency` | — | — | 0.25 |
| `indentation_consistency` | — | — | 0.05 |

**Multiplicative penalties:**
- Parse failure: `×0.5`
- Forbidden elements present: multiplicative `forbidden_penalty`

## Exploit Resistance

| Exploit Attempt | Defense Mechanism |
|----------------|-------------------|
| Submit empty file | `MIN_TERMINAL_SCORE = 0.01` floor + unresolved conflict penalty |
| Spam inspect to avoid committing | `-0.01` step penalty per action + hard step limit |
| Re-resolve same block repeatedly | Reward multiplier decays: `1.0 → 0.7 → 0.4` |
| Brute-force resolution text similarity | Non-exact `grade_block` matches capped at `0.85` |
| Inject conflict markers in resolution | Hard rejected in `step()` before any grading |
| Grader collapsing to 0 or 1 | All scores clamped to strict open interval `(0.01, 0.99)` |
| Mixed architectural patterns | Consistency bonus lost; terminal reward reduced |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/tasks` | List all tasks with metadata, difficulty, and max_steps |
| `POST` | `/reset` | Start new episode. Body: `{"task_id": "task1"}` |
| `POST` | `/step` | Submit action. Body: `MergeAction`. Returns `StepResult` |
| `GET` | `/state` | Get current episode metadata as `EpisodeState` |
| `POST` | `/grader` | Run deterministic grader on an agent-submitted file |
| `POST` | `/validate` | Validate environment schema compliance |
| `POST` | `/baseline` | Run LLM baseline agent across all three tasks |
| `GET` | `/health` | Health check |
| `GET` | `/docs-home` | This documentation page |

## Baseline

A strong baseline agent is included (`inference.py`) using chain-of-thought prompting with conflict-aware repair heuristics. It uses a sliding message window (system + first message + last 4 exchanges), retries on rate limits, and enforces submit when all conflicts are resolved. It runs all three tasks in sequence and emits structured judge-compatible output to stdout in the format:

```
[START] task=<task> env=GitMergeEnv model=<model>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
```

Debug lines go to stderr. Stdout is reserved for judge-compatible output only.

## Quick Start

```bash
# Clone and run locally
git clone <your-repo-url>
cd gitmergeenv
cp .env.example .env

# Edit .env:
# API_BASE_URL=https://router.huggingface.co/v1
# API_KEY=your_huggingface_token
# MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
# BASE_URL=http://localhost:7860
# PORT=7860

docker build -t gitmergeenv .
docker run -p 7860:7860 --env-file .env gitmergeenv

# Test it
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "task1"}'
```

## Run Baseline

```bash
# Inside running container or with env vars set:
python inference.py
```

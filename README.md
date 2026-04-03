---
title: GitMergeEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---


# GitMergeEnv

> An OpenEnv environment for training AI agents to resolve git merge conflicts
> in Python source files.

## Overview

GitMergeEnv presents an agent with a Python file containing git merge conflict
markers. The agent inspects conflict blocks, proposes resolutions one by one,
and is scored deterministically against a ground truth resolution.

Built for the Meta PyTorch x OpenEnv Hackathon.

## Motivation

Git merge conflicts are one of the most universally painful experiences in
software development. Every engineer hits them. They require understanding
the intent behind two diverging changes and synthesizing a correct resolution
that preserves both. This is a genuine multi-step reasoning task with
deterministic ground truth — an ideal domain for RL agent training.

## Action Space

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| action_type | string | Yes | One of: inspect, resolve, submit |
| conflict_id | int | Conditional | Required for inspect and resolve |
| resolution | string | Conditional | Required for resolve |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| file_name | string | Name of the file being merged |
| total_conflicts | int | Total conflict blocks in file |
| resolved_conflicts | int | Blocks the agent has resolved |
| unresolved_conflict_ids | list[int] | Block IDs not yet resolved |
| current_file_preview | string | File with resolutions applied so far |
| last_action_feedback | string | Feedback on last action |
| last_reward | float | Reward from last action |
| steps_remaining | int | Steps before forced termination |

## Tasks

### Task 1 — Easy (max 6 steps)
Single conflict block. Developer A renamed a variable; Developer B added a
new parameter. Agent must synthesize both changes.
Expected baseline score: 0.75–0.95

### Task 2 — Medium (max 12 steps)
Three conflict blocks across imports, method body, and docstring.
Developer A added custom exceptions; Developer B added structured logging.
Agent must merge both sets of changes consistently.
Expected baseline score: 0.45–0.70

### Task 3 — Hard (max 18 steps)
Five architecturally dependent conflict blocks.
Developer A migrated to SQLAlchemy ORM; Developer B added features using
old raw sqlite3. Conflicts cannot be resolved independently — all five must
be resolved consistently with the ORM approach.
Expected baseline score: 0.20–0.45

## Reward Design

Rewards are dense and multi-component:
- inspect: +0.02 (encourage information gathering)
- resolve correct block: +0.15 exact, +0.08 good, +0.02 partial
- resolve wrong block: -0.02 to -0.08
- invalid action: -0.05 to -0.10
- step penalty: -0.01 per step (encourages efficiency)
- submit: full grader score minus unresolved block penalty plus efficiency bonus

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /reset | POST | Start new episode |
| /step | POST | Execute one action |
| /state | GET | Get episode state |
| /tasks | GET | List all tasks |
| /grader | POST | Score current state |
| /validate | POST | Run grader self-checks |
| /baseline | POST | Run baseline agent |
| /health | GET | Health check |

## Setup

Set these three variables:

**Windows PowerShell:**
```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:HF_TOKEN = "your_huggingface_token_here"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

**Linux/Mac:**
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export HF_TOKEN=your_huggingface_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

Get your HF token at: https://huggingface.co/settings/tokens

### Run Locally
```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run Baseline
```bash
# Windows PowerShell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:HF_TOKEN = "your_huggingface_token_here"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
python inference.py

# Linux/Mac
export API_BASE_URL=https://router.huggingface.co/v1
export HF_TOKEN=your_huggingface_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Baseline Scores

Evaluated using `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace router.

| Task | Score | Difficulty | Notes |
|------|-------|------------|-------|
| task1 | ~0.77 | Easy | Single conflict, variable rename |
| task2 | ~0.60 | Medium | Three conflicts, requires merging both changes |
| task3 | ~0.20 | Hard | Five architecturally dependent conflicts |
| **Average** | **~0.52** | | |

Scores reflect genuine difficulty progression. Task3 is designed to challenge
models that resolve conflicts independently without tracking architectural
consistency across blocks.

## OpenEnv Compliance

Passes `openenv validate`. Deployed on Hugging Face Spaces.
Tagged with `openenv`.

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
| /baseline | POST | Run baseline agent |
| /health | GET | Health check |

## Setup

### Environment Variables

Copy .env.example to .env and fill in your credentials:
```bash
copy .env.example .env
```

For local testing with Groq:
```
INFERENCE_PROVIDER=groq
GROQ_API_KEY=your_groq_key
MODEL_NAME=moonshotai/kimi-k2-instruct
```

For production and submission with HuggingFace:
```
INFERENCE_PROVIDER=huggingface
HF_TOKEN=your_hf_token
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
```

### Run Locally
```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run Baseline
```bash
# Groq (local testing)
set INFERENCE_PROVIDER=groq
set GROQ_API_KEY=your_key
python inference.py

# HuggingFace (submission)
set INFERENCE_PROVIDER=huggingface
set HF_TOKEN=your_token
python inference.py
```

### Docker
```bash
# Build
docker build -f server/Dockerfile -t git_merge_env .

# Run with Groq
docker run -p 7860:7860 \
  -e INFERENCE_PROVIDER=groq \
  -e GROQ_API_KEY=your_groq_key \
  -e MODEL_NAME=moonshotai/kimi-k2-instruct \
  git_merge_env

# Run with HuggingFace
docker run -p 7860:7860 \
  -e INFERENCE_PROVIDER=huggingface \
  -e HF_TOKEN=your_hf_token \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  git_merge_env
```

## Baseline Scores

| Task | Score | Model |
|------|-------|-------|
| task1 | ~0.82 | gpt-4o-mini |
| task2 | ~0.54 | gpt-4o-mini |
| task3 | ~0.31 | gpt-4o-mini |

## OpenEnv Compliance

Passes `openenv validate`. Deployed on Hugging Face Spaces.
Tagged with `openenv`.

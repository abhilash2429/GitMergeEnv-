"""
inference.py

Baseline inference script for GitMergeEnv.
Runs a GPT model as an agent against all 3 tasks.
Uses the OpenAI API client with the environment's HTTP API.

Usage:
    export HF_TOKEN=your_huggingface_token
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=nvidia/llama-3.1-nemotron-70b-instruct
    export BASE_URL=http://localhost:7860   # or your HF Space URL
    python inference.py

Environment variables:
    HF_TOKEN         — preferred
    API_KEY          — fallback
    API_BASE_URL     — inference API URL
    MODEL_NAME       — model to use
    BASE_URL         — environment URL (default: http://localhost:7860)
"""

import json
import os
import signal

import httpx
from openai import OpenAI

BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/llama-3.1-nemotron-70b-instruct")
MAX_STEPS_OVERRIDE = 20  # safety cap

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


SYSTEM_PROMPT = """\
You are an expert software engineer resolving git merge conflicts in Python files.

You will be given a Python file with git conflict markers (<<<<<<< HEAD, =======, >>>>>>>).
Your goal is to resolve all conflicts and produce clean, working Python code.

You interact with an environment through structured actions:

1. inspect — examine a specific conflict block in detail
   {"action_type": "inspect", "conflict_id": 0}

2. resolve — submit your resolution for one conflict block
   {"action_type": "resolve", "conflict_id": 0, "resolution": "your resolved code here"}

3. submit — finalize when all conflicts are resolved
   {"action_type": "submit"}

Strategy:
- Always inspect a conflict block before resolving it
- Read both HEAD and INCOMING versions carefully
- Understand the developer intent behind each change
- When both sides add different features, include BOTH
- When changes conflict architecturally, prefer the more complete refactor
- Ensure all resolved blocks are internally consistent
- Submit only when all conflicts are resolved

Respond ONLY with a valid JSON action. No explanation. No markdown. Just JSON.
"""


def call_env(endpoint: str, method: str = "POST", body: dict = None) -> dict:
    """Make HTTP request to the environment."""
    url = f"{BASE_URL}{endpoint}"
    with httpx.Client(timeout=30.0) as client:
        if method == "GET":
            response = client.get(url)
        else:
            response = client.post(url, json=body or {})
    response.raise_for_status()
    return response.json()


def run_task(client: OpenAI, task_id: str) -> float:
    """
    Run one complete episode for a given task.
    Returns the final grader score.
    """
    print(f"\n{'=' * 60}")
    print(f"Running task: {task_id}")
    print(f"{'=' * 60}")

    obs = call_env(f"/reset?task_id={task_id}")
    print(f"Task started. File: {obs['file_name']}, Conflicts: {obs['total_conflicts']}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"File to resolve: {obs['file_name']}\n"
                f"Total conflicts: {obs['total_conflicts']}\n\n"
                f"Current file state:\n\n{obs['current_file_preview']}\n\n"
                f"{obs['last_action_feedback']}\n\n"
                f"Begin resolving. Start by inspecting conflict 0."
            ),
        },
    ]

    final_score = 0.0

    for step_num in range(MAX_STEPS_OVERRIDE):
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True,
        )

        raw_response_parts = []
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                raw_response_parts.append(chunk.choices[0].delta.content)

        raw_response = "".join(raw_response_parts).strip()

        action_str = raw_response
        if "```" in action_str:
            action_str = action_str.split("```")[1]
            if action_str.startswith("json"):
                action_str = action_str[4:]
            action_str = action_str.strip()

        try:
            action = json.loads(action_str)
        except json.JSONDecodeError:
            print(f"Step {step_num}: LLM returned invalid JSON: {raw_response[:100]}")
            action = {"action_type": "inspect", "conflict_id": 0}

        print(
            f"Step {step_num}: {action.get('action_type', 'unknown')} "
            f"(block {action.get('conflict_id', '-')})"
        )

        result = call_env("/step", body=action)
        obs = result["observation"]
        reward = result["reward"]["value"]
        done = result["done"]

        print(
            f"  Reward: {reward:.4f} | Resolved: "
            f"{obs['resolved_conflicts']}/{obs['total_conflicts']}"
        )

        messages.append({"role": "assistant", "content": raw_response})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Result: {obs['last_action_feedback']}\n\n"
                    f"Current file:\n\n{obs['current_file_preview']}\n\n"
                    f"Resolved: {obs['resolved_conflicts']}/{obs['total_conflicts']} conflicts.\n"
                    f"Unresolved IDs: {obs['unresolved_conflict_ids']}\n"
                    f"Steps remaining: {obs['steps_remaining']}\n\n"
                    + (
                        "All conflicts resolved! Call submit now."
                        if obs["resolved_conflicts"] == obs["total_conflicts"] and not done
                        else "Continue resolving remaining conflicts or submit when ready."
                    )
                ),
            }
        )

        if done:
            grader_result = call_env("/grader", method="POST")
            final_score = grader_result["score"]
            print(f"Episode done. Final grader score: {final_score:.4f}")
            break
    else:
        grader_result = call_env("/grader", method="POST")
        final_score = grader_result["score"]
        print(f"Step limit reached. Grader score: {final_score:.4f}")

    return final_score


def run_baseline() -> dict:
    """
    Run baseline agent against all 3 tasks.
    Returns dict mapping task_id to score.
    """
    if not API_KEY:
        raise ValueError("HF_TOKEN or API_KEY environment variable must be set")

    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        score = run_task(client, task_id)
        scores[task_id] = round(score, 4)

    print(f"\n{'=' * 60}")
    print("BASELINE RESULTS")
    print(f"{'=' * 60}")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  Average: {avg:.4f}")

    return scores


def _timeout_handler(signum, frame):
    raise TimeoutError("Inference script exceeded 20 minute runtime limit")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(1140)  # 19 minutes — 60 second buffer before hard 20min limit
    scores = run_baseline()

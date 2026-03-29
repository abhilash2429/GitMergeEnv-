"""
inference.py

Baseline inference script for GitMergeEnv.
Runs a GPT model as an agent against all 3 tasks.
Uses the OpenAI API client with the environment's HTTP API.

Usage:
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
    export HF_TOKEN=your_huggingface_token
    export BASE_URL=http://localhost:7860   # or your HF Space URL
    python inference.py

Environment variables:
    HF_TOKEN         — Hugging Face token
    API_KEY          — Hugging Face fallback
    API_BASE_URL     — OpenAI-compatible inference API URL
    MODEL_NAME       — model to use
    BASE_URL         — environment URL (default: http://localhost:7860)
"""

import _thread
import json
import os
import re
import signal
import threading

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

if not API_KEY:
    raise ValueError(
        "HF_TOKEN environment variable must be set. "
        "Get your token at https://huggingface.co/settings/tokens"
    )

if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable must be set.")

print(f"[inference] Base URL: {API_BASE_URL}")
print(f"[inference] Model: {MODEL_NAME}")

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


def _normalize_action_text(raw_response: str) -> str:
    """Strip reasoning and markdown wrappers from model output."""
    action_str = raw_response or ""

    if "<think>" in action_str:
        action_str = re.sub(r"<think>.*?</think>", "", action_str, flags=re.DOTALL).strip()

    if "```" in action_str:
        parts = action_str.split("```")
        if len(parts) > 1:
            action_str = parts[1]
            if action_str.startswith("json"):
                action_str = action_str[4:]

    return action_str.strip()


def _extract_resolution_value(action_str: str) -> str | None:
    """Best-effort extraction of a multiline resolution payload."""
    match = re.search(r'["\']resolution["\']\s*:\s*', action_str)
    if not match:
        return None

    value = action_str[match.end():].strip()
    if value.endswith("}"):
        value = value[:-1].rstrip()
    if value.endswith(","):
        value = value[:-1].rstrip()

    if value.startswith('"""') or value.startswith("'''"):
        quote = value[:3]
        return value[3:-3] if value.endswith(quote) else value[3:]

    if value.startswith('"') or value.startswith("'"):
        quote = value[0]
        value = value[1:-1] if value.endswith(quote) else value[1:]
        value = value.replace(r"\r\n", "\n")
        value = value.replace(r"\n", "\n")
        value = value.replace(r"\t", "\t")
        value = value.replace(r"\"", '"')
        value = value.replace(r"\'", "'")
        value = value.replace(r"\\", "\\")
        return value

    return value


def _parse_action(raw_response: str) -> dict | None:
    """Parse a model action, including malformed JSON with multiline code."""
    action_str = _normalize_action_text(raw_response)

    try:
        return json.loads(action_str)
    except json.JSONDecodeError:
        pass

    start = action_str.find("{")
    end = action_str.rfind("}")
    if start != -1 and end > start:
        candidate = action_str[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            action_str = candidate

    action: dict = {}

    action_type_match = re.search(r'["\']action_type["\']\s*:\s*["\']([^"\']+)["\']', action_str)
    if action_type_match:
        action["action_type"] = action_type_match.group(1)

    conflict_id_match = re.search(r'["\']conflict_id["\']\s*:\s*(-?\d+)', action_str)
    if conflict_id_match:
        action["conflict_id"] = int(conflict_id_match.group(1))

    resolution = _extract_resolution_value(action_str)
    if resolution is not None:
        action["resolution"] = resolution

    return action or None


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

    if obs["total_conflicts"] <= 0:
        raise RuntimeError(
            f"Environment reset returned no conflicts for {task_id}. "
            f"Feedback: {obs.get('last_action_feedback', '')}"
        )

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
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )
            raw_response = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"Step {step_num}: API call failed ({exc}). Using fallback inspect action.")
            raw_response = '{"action_type": "inspect", "conflict_id": 0}'

        action = _parse_action(raw_response)
        if action is None:
            action_preview = _normalize_action_text(raw_response)
            print(f"Step {step_num}: LLM returned invalid JSON: {action_preview[:100]}")
            action = {"action_type": "inspect", "conflict_id": 0}

        print(
            f"Step {step_num}: {action.get('action_type', 'unknown')} "
            f"(block {action.get('conflict_id', '-')})"
        )

        result = call_env("/step", body=action)
        obs = result["observation"]
        reward = result["reward"]["value"]
        done = result["done"]

        if (
            obs.get("file_name") == "unknown"
            or obs.get("total_conflicts", 0) == 0
            or obs.get("last_action_feedback", "").startswith("Internal error processing action:")
        ):
            raise RuntimeError(
                "Environment /step returned an internal-error safety response: "
                f"{obs.get('last_action_feedback', '')}"
            )

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
    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        try:
            score = run_task(client, task_id)
        except Exception as exc:
            print(f"Task {task_id} failed entirely: {exc}. Recording score 0.0")
            score = 0.0
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


def _timeout_interrupt():
    _thread.interrupt_main()


if __name__ == "__main__":
    timer = None
    try:
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(1140)  # 19 minutes — 60 second buffer before hard 20min limit
        else:
            timer = threading.Timer(1140, _timeout_interrupt)
            timer.daemon = True
            timer.start()

        scores = run_baseline()
    except KeyboardInterrupt as exc:
        if timer is not None:
            raise TimeoutError("Inference script exceeded 20 minute runtime limit") from exc
        raise
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        if timer is not None:
            timer.cancel()

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
    NVIDIA_API_KEY   — NVIDIA NIM key for development testing
    API_BASE_URL     — OpenAI-compatible inference API URL
    MODEL_NAME       — model to use
    BASE_URL         — environment URL (default: http://localhost:7860)
"""

import _thread
import ast
import json
import os
import re
import signal
import threading
import time
import textwrap

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
RAW_MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")


def _normalize_model_name(model_name: str) -> str:
    """Normalize common provider-specific model-name typos."""
    normalized = model_name.strip()
    if normalized.startswith("meta/llama-3_3"):
        normalized = normalized.replace("llama-3_3", "llama-3.3", 1)
    return normalized


MODEL_NAME = _normalize_model_name(RAW_MODEL_NAME)
IS_NVIDIA_NIM = "integrate.api.nvidia.com" in API_BASE_URL

if IS_NVIDIA_NIM:
    API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("API_KEY") or os.getenv("HF_TOKEN")
else:
    API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    raise ValueError(
        "HF_TOKEN, API_KEY, or NVIDIA_API_KEY environment variable must be set. "
        "For Hugging Face tokens, get one at https://huggingface.co/settings/tokens"
    )

if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable must be set.")

print(f"[inference] Base URL: {API_BASE_URL}")
print(f"[inference] Model: {MODEL_NAME}")
if RAW_MODEL_NAME != MODEL_NAME:
    print(f"[inference] Normalized model name from '{RAW_MODEL_NAME}' to '{MODEL_NAME}'")

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


def _create_completion_text(messages: list[dict]) -> str:
    """Create a chat completion and return plain response text."""
    for attempt in range(3):
        try:
            if IS_NVIDIA_NIM:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024,
                    stream=True,
                )

                chunks: list[str] = []
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        chunks.append(chunk.choices[0].delta.content)
                return "".join(chunks)

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            error_text = str(exc)
            is_rate_limited = "429" in error_text or "Too Many Requests" in error_text
            if is_rate_limited and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise


def _parses_cleanly(code: str) -> tuple[bool, str | None]:
    """Check whether the current file preview parses as Python."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno else "unknown line"
        return False, f"{exc.msg} at {location}"


def _pick_focus_block(obs: dict, block_scores: dict[int, float]) -> int:
    """Choose the most likely problematic block for inspection."""
    unresolved = obs.get("unresolved_conflict_ids") or []
    if unresolved:
        return unresolved[0]

    if block_scores:
        return min(block_scores, key=block_scores.get)

    return 0


def _extract_inspection_context(feedback: str) -> dict | None:
    """Extract HEAD and INCOMING block text from inspect feedback."""
    pattern = re.compile(
        r"HEAD version \(current branch\):\n(.*?)\n\nINCOMING version \(feature branch\):\n(.*?)\n\nHint:",
        re.DOTALL,
    )
    match = pattern.search(feedback)
    if not match:
        return None
    return {
        "head": match.group(1),
        "incoming": match.group(2),
    }


def _block_base_indent(context: dict) -> str:
    """Infer the baseline indentation for a conflict block."""
    indent_levels = []
    for text in (context["head"], context["incoming"]):
        for line in text.splitlines():
            if line.strip():
                indent_levels.append(len(line) - len(line.lstrip(" ")))
    return " " * min(indent_levels) if indent_levels else ""


def _normalize_indented_block(text: str, indent: str) -> str:
    """Normalize a resolution to the expected block indentation."""
    normalized = textwrap.dedent(text).strip("\n")
    if not normalized:
        return normalized
    if not indent:
        return normalized
    return "\n".join(
        f"{indent}{line}" if line.strip() else ""
        for line in normalized.splitlines()
    )


def _normalize_docstring_block(text: str, indent: str) -> str:
    """Ensure docstring resolutions remain valid Python docstrings."""
    normalized = textwrap.dedent(text).strip()
    if '"""' in normalized or "'''" in normalized:
        return _normalize_indented_block(normalized, indent)

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        lines = [normalized]

    inner = "\n".join(f"{indent}{line}" for line in lines)
    return f'{indent}"""\n{inner}\n{indent}"""'


def _normalize_resolution(conflict_id: int, resolution: str, inspected_blocks: dict[int, dict]) -> str:
    """Apply syntax-preserving normalization based on the inspected block context."""
    context = inspected_blocks.get(conflict_id)
    if not context:
        return resolution

    indent = _block_base_indent(context)
    is_docstring_block = '"""' in context["head"] and '"""' in context["incoming"]

    if is_docstring_block:
        return _normalize_docstring_block(resolution, indent)
    return _normalize_indented_block(resolution, indent)


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
    block_scores: dict[int, float] = {}
    inspected_blocks: dict[int, dict] = {}
    forced_review_active = False
    forced_review_block = 0

    for step_num in range(MAX_STEPS_OVERRIDE):
        action = None
        raw_response = ""

        if obs["resolved_conflicts"] == obs["total_conflicts"] and not forced_review_active:
            parses_cleanly, parse_error = _parses_cleanly(obs["current_file_preview"])
            if not parses_cleanly:
                forced_review_block = _pick_focus_block(obs, block_scores)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Do not submit yet. The current file does not parse as Python "
                            f"({parse_error}). Re-inspect block {forced_review_block} and repair it."
                        ),
                    }
                )
                action = {"action_type": "inspect", "conflict_id": forced_review_block}
                raw_response = json.dumps(action)
                forced_review_active = True
                print(f"  Parse error: {parse_error}")
            else:
                grader_result = call_env("/grader", method="POST")
                if grader_result["score"] < 0.8:
                    forced_review_block = _pick_focus_block(obs, block_scores)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Do not submit yet. Pre-submit grader score is "
                                f"{grader_result['score']:.4f}. Re-inspect block "
                                f"{forced_review_block} and improve the merge before submitting."
                            ),
                        }
                    )
                    action = {"action_type": "inspect", "conflict_id": forced_review_block}
                    raw_response = json.dumps(action)
                    forced_review_active = True

        if action is None:
            try:
                raw_response = _create_completion_text(messages)
            except Exception as exc:
                print(f"Step {step_num}: API call failed ({exc}). Using fallback inspect action.")
                raw_response = '{"action_type": "inspect", "conflict_id": 0}'

            action = _parse_action(raw_response)
            if action is None:
                action_preview = _normalize_action_text(raw_response)
                print(f"Step {step_num}: LLM returned invalid JSON: {action_preview[:100]}")
                action = {"action_type": "inspect", "conflict_id": 0}

            if forced_review_active and action.get("action_type") == "submit":
                action = {"action_type": "inspect", "conflict_id": forced_review_block}
                raw_response = json.dumps(action)

        print(
            f"Step {step_num}: {action.get('action_type', 'unknown')} "
            f"(block {action.get('conflict_id', '-')})"
        )

        if action.get("action_type") == "resolve" and action.get("conflict_id") is not None:
            action["resolution"] = _normalize_resolution(
                action["conflict_id"],
                action["resolution"],
                inspected_blocks,
            )

        result = call_env("/step", body=action)
        obs = result["observation"]
        reward = result["reward"]  # Now a simple float
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

        if "block_score" in result["info"] and action.get("conflict_id") is not None:
            block_scores[action["conflict_id"]] = result["info"]["block_score"]
            print(f"  Block score: {result['info']['block_score']:.4f}")

        if action.get("action_type") == "inspect" and action.get("conflict_id") is not None:
            context = _extract_inspection_context(obs["last_action_feedback"])
            if context is not None:
                inspected_blocks[action["conflict_id"]] = context

        if action.get("action_type") == "resolve":
            forced_review_active = False

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

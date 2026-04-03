import os
import re
import json
import time
import sys
from openai import OpenAI

# -------------------------------------------------------
# Credentials — judges set these three variables
# -------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

# NOTE: Key validation and client creation are deferred to run_baseline()
# so importing this module is 100% side-effect free.
# The /baseline endpoint does `from inference import run_baseline` and must
# not crash or block even when no API key is set.

BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")
MAX_STEPS_OVERRIDE = 20
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5   # task score >= this → success=true in [END]
BENCHMARK = "GitMergeEnv"

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

CRITICAL SUBMISSION RULE:
Once observation shows resolved_conflicts == total_conflicts, you MUST call submit on your very next action.
"""


# -------------------------------------------------------
# Mandatory structured stdout logging — judge format
# -------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Emit the mandatory [START] line to stdout."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    """Emit a mandatory [STEP] line to stdout after every env.step() call."""
    error_val = str(error) if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    """Emit the mandatory [END] line to stdout after the episode ends."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _action_str(action: dict) -> str:
    """Compact single-token action label for the [STEP] line."""
    atype = action.get("action_type", "unknown")
    cid = action.get("conflict_id")
    if atype == "inspect" and cid is not None:
        return f"inspect({cid})"
    if atype == "resolve" and cid is not None:
        return f"resolve({cid})"
    if atype == "submit":
        return "submit"
    return atype


# -------------------------------------------------------
# Environment HTTP helpers
# -------------------------------------------------------

def call_env(endpoint: str, method: str = "POST", body: dict = None) -> dict:
    """Make an HTTP request to the GitMergeEnv server."""
    import httpx

    url = f"{BASE_URL}{endpoint}"
    with httpx.Client(timeout=30.0) as http:
        if method == "GET":
            response = http.get(url)
        else:
            response = http.post(url, json=body or {})
    response.raise_for_status()
    return response.json()


# -------------------------------------------------------
# LLM completion helper
# -------------------------------------------------------

def _create_completion_text(messages: list[dict]) -> str:
    """Call the model and return plain response text, with rate-limit retry."""
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            error_text = str(exc)
            is_rate_limited = "429" in error_text or "Too Many Requests" in error_text
            if is_rate_limited and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise


# -------------------------------------------------------
# Inspection context helpers
# -------------------------------------------------------

def _parses_cleanly(code: str) -> tuple[bool, str | None]:
    """Return (True, None) if the code parses as valid Python."""
    import ast
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno else "unknown line"
        return False, f"{exc.msg} at {location}"


def _pick_focus_block(obs: dict, block_scores: dict[int, float]) -> int:
    """Pick the most likely problematic block for re-inspection."""
    unresolved = obs.get("unresolved_conflict_ids") or []
    if unresolved:
        return unresolved[0]
    if block_scores:
        return min(block_scores, key=block_scores.get)
    return 0


def _extract_inspection_context(feedback: str) -> dict | None:
    """Extract HEAD / INCOMING text from an inspect feedback message."""
    pattern = re.compile(
        r"HEAD version \(current branch\):\n(.*?)\n\nINCOMING version \(feature branch\):\n(.*?)\n\nHint:",
        re.DOTALL,
    )
    match = pattern.search(feedback)
    if not match:
        return None
    return {"head": match.group(1), "incoming": match.group(2)}


# -------------------------------------------------------
# Resolution normalisation helpers
# -------------------------------------------------------

def _block_base_indent(context: dict) -> str:
    indent_levels = []
    for text in (context["head"], context["incoming"]):
        for line in text.splitlines():
            if line.strip():
                indent_levels.append(len(line) - len(line.lstrip(" ")))
    return " " * min(indent_levels) if indent_levels else ""


def _normalize_indented_block(text: str, indent: str) -> str:
    import textwrap
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
    import textwrap
    normalized = textwrap.dedent(text).strip()
    if '"""' in normalized or "'''" in normalized:
        return _normalize_indented_block(normalized, indent)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        lines = [normalized]
    inner = "\n".join(f"{indent}{line}" for line in lines)
    return f'{indent}"""\n{inner}\n{indent}"""'


def _normalize_resolution(conflict_id: int, resolution: str, inspected_blocks: dict[int, dict]) -> str:
    context = inspected_blocks.get(conflict_id)
    if not context:
        return resolution
    indent = _block_base_indent(context)
    is_docstring_block = '"""' in context["head"] and '"""' in context["incoming"]
    if is_docstring_block:
        return _normalize_docstring_block(resolution, indent)
    return _normalize_indented_block(resolution, indent)


# -------------------------------------------------------
# Action parsing helpers
# -------------------------------------------------------

def _normalize_action_text(raw_response: str) -> str:
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
        value = value.replace(r"\r\n", "\n").replace(r"\n", "\n")
        value = value.replace(r"\t", "\t").replace(r"\"", '"')
        value = value.replace(r"\'", "'").replace(r"\\", "\\")
        return value
    return value


def _parse_action(raw_response: str) -> dict | None:
    action_str = _normalize_action_text(raw_response)
    try:
        return json.loads(action_str)
    except json.JSONDecodeError:
        pass
    start = action_str.find("{")
    end = action_str.rfind("}")
    if start != -1 and end > start:
        candidate = action_str[start: end + 1]
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


# -------------------------------------------------------
# Task runner
# -------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> float:
    """
    Run one complete episode for a given task.
    Emits [START], one [STEP] per step, and [END] to stdout.
    Returns the final grader score.
    """
    step_rewards: list[float] = []
    step_num = 0
    final_score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = call_env(f"/reset?task_id={task_id}")
        print(
            f"[inference] Task started. File: {obs['file_name']}, "
            f"Conflicts: {obs['total_conflicts']}",
            file=sys.stderr, flush=True,
        )

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

        block_scores: dict[int, float] = {}
        inspected_blocks: dict[int, dict] = {}
        forced_review_active = False
        forced_review_block = 0

        for step_idx in range(MAX_STEPS_OVERRIDE):
            step_num = step_idx + 1
            action = None
            raw_response = ""

            # ── Pre-submit guard ────────────────────────────────────────────
            if obs["resolved_conflicts"] == obs["total_conflicts"] and not forced_review_active:
                parses_cleanly, parse_error = _parses_cleanly(obs["current_file_preview"])
                if not parses_cleanly:
                    forced_review_block = _pick_focus_block(obs, block_scores)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Do not submit yet. The current file does not parse as Python "
                            f"({parse_error}). Re-inspect block {forced_review_block} and repair it."
                        ),
                    })
                    action = {"action_type": "inspect", "conflict_id": forced_review_block}
                    raw_response = json.dumps(action)
                    forced_review_active = True
                    print(f"[inference]   Parse error: {parse_error}", file=sys.stderr, flush=True)
                else:
                    grader_result = call_env("/grader", method="POST")
                    if grader_result["score"] < 0.8:
                        forced_review_block = _pick_focus_block(obs, block_scores)
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Do not submit yet. Pre-submit grader score is "
                                f"{grader_result['score']:.4f}. Re-inspect block "
                                f"{forced_review_block} and improve the merge before submitting."
                            ),
                        })
                        action = {"action_type": "inspect", "conflict_id": forced_review_block}
                        raw_response = json.dumps(action)
                        forced_review_active = True

            # ── Model call ──────────────────────────────────────────────────
            if action is None:
                try:
                    raw_response = _create_completion_text(messages)
                except Exception as exc:
                    print(
                        f"[inference] Step {step_idx}: API call failed ({exc}). Using fallback.",
                        file=sys.stderr, flush=True,
                    )
                    raw_response = '{"action_type": "inspect", "conflict_id": 0}'

                action = _parse_action(raw_response)
                if action is None:
                    action_preview = _normalize_action_text(raw_response)
                    print(
                        f"[inference] Step {step_idx}: invalid JSON: {action_preview[:100]}",
                        file=sys.stderr, flush=True,
                    )
                    action = {"action_type": "inspect", "conflict_id": 0}

                if forced_review_active and action.get("action_type") == "submit":
                    action = {"action_type": "inspect", "conflict_id": forced_review_block}
                    raw_response = json.dumps(action)

            # ── Force submit when all resolved ──────────────────────────────
            if (
                obs.get("resolved_conflicts") == obs.get("total_conflicts")
                and obs.get("total_conflicts", 0) > 0
                and action.get("action_type") != "submit"
            ):
                print(
                    f"[inference] Step {step_idx}: All blocks resolved — forcing submit",
                    file=sys.stderr, flush=True,
                )
                action = {"action_type": "submit"}
                raw_response = json.dumps(action)

            # ── Normalise resolution ────────────────────────────────────────
            if action.get("action_type") == "resolve" and action.get("conflict_id") is not None:
                action["resolution"] = _normalize_resolution(
                    action["conflict_id"],
                    action.get("resolution", ""),
                    inspected_blocks,
                )

            # ── Call environment ────────────────────────────────────────────
            result = call_env("/step", body=action)
            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]

            # Detect internal environment errors
            error = None
            feedback = obs.get("last_action_feedback", "")
            if feedback.startswith("Internal error processing action:"):
                error = "internal_env_error"

            step_rewards.append(reward)

            # ── Mandatory [STEP] log ────────────────────────────────────────
            log_step(
                step=step_num,
                action=_action_str(action),
                reward=reward,
                done=done,
                error=error,
            )

            # Debug info → stderr only
            print(
                f"[inference]   Resolved: {obs['resolved_conflicts']}/{obs['total_conflicts']} "
                f"| Steps left: {obs['steps_remaining']}",
                file=sys.stderr, flush=True,
            )

            if "block_score" in result["info"] and action.get("conflict_id") is not None:
                bs = result["info"]["block_score"]
                block_scores[action["conflict_id"]] = bs
                print(f"[inference]   Block score: {bs:.4f}", file=sys.stderr, flush=True)

            if action.get("action_type") == "inspect" and action.get("conflict_id") is not None:
                context = _extract_inspection_context(obs["last_action_feedback"])
                if context is not None:
                    inspected_blocks[action["conflict_id"]] = context

            if action.get("action_type") == "resolve":
                forced_review_active = False

            # Update message history
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
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
            })
            if len(messages) > 10:
                messages = [messages[0], messages[1]] + messages[-8:]

            if done:
                grader_result = call_env("/grader", method="POST")
                final_score = grader_result["score"]
                print(
                    f"[inference] Episode done. Final grader score: {final_score:.4f}",
                    file=sys.stderr, flush=True,
                )
                break

        else:
            # Step limit reached without submit
            grader_result = call_env("/grader", method="POST")
            final_score = grader_result["score"]
            print(
                f"[inference] Step limit reached. Grader score: {final_score:.4f}",
                file=sys.stderr, flush=True,
            )

        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[inference] Task {task_id} failed: {exc}", file=sys.stderr, flush=True)
        success = False
        final_score = 0.0

    finally:
        # Always emit [END] — even on exception
        log_end(
            success=success,
            steps=step_num,
            score=final_score,
            rewards=step_rewards,
        )

    return final_score


# -------------------------------------------------------
# Baseline runner
# -------------------------------------------------------

def run_baseline() -> dict:
    """
    Run the baseline agent against all 3 tasks.
    Returns {task_id: score}.
    """
    # Validate credentials here — not at import time
    if not API_KEY:
        raise ValueError(
            "HF_TOKEN environment variable must be set. "
            "Get your token at https://huggingface.co/settings/tokens"
        )

    print(f"[inference] Base URL: {API_BASE_URL}", file=sys.stderr, flush=True)
    print(f"[inference] Model:    {MODEL_NAME}", file=sys.stderr, flush=True)

    # Build the OpenAI client once, pass it to each task run
    _client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=30.0,
        max_retries=2,
    )

    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        try:
            score = run_task(_client, task_id)
        except Exception as exc:
            print(
                f"[inference] Task {task_id} failed entirely: {exc}. Recording 0.0",
                file=sys.stderr, flush=True,
            )
            score = 0.0
        scores[task_id] = round(score, 4)

    avg = sum(scores.values()) / len(scores)
    print(f"\n[inference] BASELINE RESULTS", file=sys.stderr, flush=True)
    for task_id, score in scores.items():
        print(f"[inference]   {task_id}: {score:.4f}", file=sys.stderr, flush=True)
    print(f"[inference]   Average: {avg:.4f}", file=sys.stderr, flush=True)

    return scores


# -------------------------------------------------------
# Entry point
# -------------------------------------------------------

if __name__ == "__main__":
    # 19-minute hard limit (infra requirement: runtime < 20 min)
    import threading

    def _timeout_handler():
        print("\n[inference] 19-minute time limit reached. Stopping.", file=sys.stderr, flush=True)
        sys.exit(1)

    timer = threading.Timer(1140, _timeout_handler)
    timer.daemon = True
    timer.start()

    try:
        run_baseline()
    finally:
        timer.cancel()

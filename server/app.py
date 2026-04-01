import json
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from models import (
    BaselineResult,
    EpisodeState,
    GraderResult,
    MergeAction,
    MergeObservation,
    MergeReward,
    StepResult,
    TaskInfo,
)
from server.environment import GitMergeEnvironment
from server.grader import ConflictGrader
from server.tasks import ALL_TASKS, TASK_LIST


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    app.state.env = GitMergeEnvironment()
    yield


app = FastAPI(
    title="GitMergeEnv",
    description=(
        "OpenEnv environment for git merge conflict resolution. "
        "An AI agent resolves Python file merge conflicts step by step "
        "and is scored deterministically against ground truth."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def get_env() -> GitMergeEnvironment:
    return app.state.env


@app.get("/", tags=["health"])
async def root():
    return {
        "status": "ok",
        "environment": "GitMergeEnv",
        "version": "0.1.0",
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


@app.post("/reset", response_model=MergeObservation, tags=["openenv"])
async def reset(task_id: str = "task1", env: GitMergeEnvironment = Depends(get_env)):
    """
    Reset the environment and start a new episode.

    Args:
        task_id: which task to run ("task1", "task2", "task3")

    Returns:
        Initial MergeObservation
    """
    try:
        return env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(exc)}") from exc


@app.post("/step", response_model=StepResult, tags=["openenv"])
async def step(action: MergeAction, env: GitMergeEnvironment = Depends(get_env)):
    """
    Execute one agent action.

    Returns observation, reward, done flag, and info dict.
    Never returns 500 on invalid actions — returns negative reward instead.
    """
    try:
        obs, reward, done, info = env.step(action)
        return StepResult(
            observation=obs,
            reward=MergeReward(
                value=reward,
                components=info,
                cumulative=round(env.total_reward, 4),
            ),
            done=done,
            info=info,
        )
    except Exception as exc:
        return StepResult(
            observation=MergeObservation(
                file_name="unknown",
                total_conflicts=0,
                resolved_conflicts=0,
                unresolved_conflict_ids=[],
                current_file_preview="",
                last_action_feedback=f"Internal error processing action: {str(exc)}",
                last_reward=-0.10,
                steps_remaining=0,
            ),
            reward=MergeReward(
                value=-0.10,
                components={"error": str(exc)},
                cumulative=0.0,
            ),
            done=False,
            info={"error": str(exc)},
        )


@app.get("/state", response_model=EpisodeState, tags=["openenv"])
async def state(env: GitMergeEnvironment = Depends(get_env)):
    """Return current episode state metadata."""
    try:
        return env.state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/tasks", response_model=list[TaskInfo], tags=["openenv"])
async def tasks():
    """
    Return list of all tasks with their action schema.
    Required endpoint per OpenEnv spec.
    """
    action_schema = {
        "action_type": {
            "type": "string",
            "enum": ["inspect", "resolve", "submit"],
            "required": True,
            "description": "Type of action to take",
        },
        "conflict_id": {
            "type": "integer",
            "required": False,
            "description": "0-indexed conflict block ID. Required for inspect and resolve.",
        },
        "resolution": {
            "type": "string",
            "required": False,
            "description": "Resolved content for the block. Required for resolve.",
        },
    }

    return [
        TaskInfo(
            id=task["id"],
            name=task["name"],
            difficulty=task["difficulty"],
            description=task["description"],
            max_steps=task["max_steps"],
            num_conflicts=task["num_conflicts"],
            action_schema=action_schema,
        )
        for task in TASK_LIST
    ]


@app.post("/grader", response_model=GraderResult, tags=["openenv"])
async def grader(env: GitMergeEnvironment = Depends(get_env)):
    """
    Score the current episode's state against ground truth.
    Can be called at any point during an episode for intermediate feedback.
    This does NOT end the episode.
    """
    if env.task is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    grader_instance = ConflictGrader()
    score, components = grader_instance.grade(env.current_file, env.task)

    unresolved = len(env.conflict_blocks) - len(env.resolutions)
    feedback_parts = [f"Current score: {score:.4f}."]
    if unresolved > 0:
        feedback_parts.append(f"{unresolved} conflict(s) still unresolved.")
    feedback_parts.append(f"Components: {json.dumps(components)}")

    return GraderResult(
        task_id=env.task_id,
        score=score,
        components=components,
        feedback=" ".join(feedback_parts),
    )


@app.post("/validate", tags=["openenv"])
async def validate():
    """
    Self-validation endpoint. Runs the grader against known inputs
    and verifies it produces expected outputs. Judges can use this
    to confirm the grader is deterministic and correctly implemented.
    """
    g = ConflictGrader()
    results = {}

    for task_id, task in ALL_TASKS.items():
        perfect_score, _ = g.grade(task["ground_truth_file"], task)
        empty_score, _ = g.grade("", task)
        marker_score, _ = g.grade(task["conflicted_file"], task)

        results[task_id] = {
            "perfect_input_score": perfect_score,
            "empty_input_score": empty_score,
            "unresolved_input_score": marker_score,
            "grader_behaves_correctly": (
                perfect_score > 0.7 and
                empty_score < 0.2 and
                marker_score < 0.3
            ),
        }

    all_correct = all(result["grader_behaves_correctly"] for result in results.values())

    return {
        "validation_passed": all_correct,
        "task_results": results,
        "message": "All graders behaving correctly" if all_correct else "Some graders need attention",
    }


@app.post("/baseline", response_model=BaselineResult, tags=["openenv"])
async def baseline():
    try:
        if not (os.getenv("HF_TOKEN") or os.getenv("API_KEY")):
            raise HTTPException(
                status_code=400,
                detail="HF_TOKEN environment variable not set."
            )

        from inference import run_baseline
        scores = run_baseline()
        avg = sum(scores.values()) / len(scores)
        return BaselineResult(
            task_scores=scores,
            average_score=round(avg, 4),
            model_used=os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct"),
        )
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=500, detail="inference.py not found in root directory")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline run failed: {str(e)}")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )


if __name__ == "__main__":
    main()

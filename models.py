from typing import List, Optional

from pydantic import BaseModel, Field


class MergeAction(BaseModel):
    """
    Action the agent takes each step.

    action_type must be one of: "inspect", "resolve", "submit"

    inspect:
        - conflict_id: required (int, 0-indexed)
        - resolution: not used
        - Effect: returns detailed context for that conflict block
        - Reward: +0.02 (small positive for information gathering)

    resolve:
        - conflict_id: required (int, 0-indexed)
        - resolution: required (str, the resolved content)
        - Effect: records the resolution for that block, computes immediate reward
        - Reward: +0.15 exact match, +0.05 partial, -0.02 wrong, -0.08 garbage

    submit:
        - conflict_id: not used
        - resolution: not used
        - Effect: finalizes the episode, runs terminal grader, sets done=True
        - Reward: full grader score minus unresolved penalty minus step waste penalty
    """

    action_type: str = Field(..., description="One of: inspect, resolve, submit")
    conflict_id: Optional[int] = Field(None, description="0-indexed conflict block ID")
    resolution: Optional[str] = Field(None, description="Resolved content for the block")


class MergeObservation(BaseModel):
    """
    Observation returned to the agent after every step and reset.
    """

    file_name: str = Field(..., description="Name of the file being merged")
    total_conflicts: int = Field(..., description="Total number of conflict blocks in file")
    resolved_conflicts: int = Field(..., description="How many blocks the agent has resolved")
    unresolved_conflict_ids: List[int] = Field(..., description="Block IDs not yet resolved")
    current_file_preview: str = Field(..., description="File content with resolutions applied so far")
    last_action_feedback: str = Field(..., description="Human-readable feedback on last action")
    last_reward: float = Field(..., description="Reward received for the last action")
    steps_remaining: int = Field(..., description="Steps left before forced episode termination")
    hint: Optional[str] = Field(None, description="Contextual hint when agent is running low on steps")


class MergeReward(BaseModel):
    """
    Reward model returned alongside observation.
    """

    value: float = Field(..., description="Reward value for the last action")
    components: dict = Field(..., description="Breakdown of reward components")
    cumulative: float = Field(..., description="Total reward accumulated this episode")


class StepResult(BaseModel):
    """
    Full result returned by /step endpoint.
    """

    observation: MergeObservation
    reward: MergeReward
    done: bool
    info: dict


class EpisodeState(BaseModel):
    """
    Returned by /state endpoint.
    """

    episode_id: str
    task_id: str
    step_count: int
    max_steps: int
    done: bool
    total_reward: float
    resolved_conflicts: int
    total_conflicts: int


class TaskInfo(BaseModel):
    """
    Returned by /tasks endpoint for each task.
    """

    id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    num_conflicts: int
    action_schema: dict


class GraderResult(BaseModel):
    """
    Returned by /grader endpoint.
    """

    task_id: str
    score: float
    components: dict
    feedback: str


class BaselineResult(BaseModel):
    """
    Returned by /baseline endpoint.
    """

    model_config = {"protected_namespaces": ()}

    task_scores: dict
    average_score: float
    model_used: str

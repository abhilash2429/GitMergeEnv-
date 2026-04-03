"""
GitMergeEnv client for programmatic access.
"""

import httpx

from models import EpisodeState, MergeAction, MergeObservation, StepResult


class GitMergeEnvClient:
    """
    Synchronous client for GitMergeEnv.

    Usage:
        client = GitMergeEnvClient(base_url="https://your-space.hf.space")
        obs = client.reset(task_id="task1")
        result = client.step(MergeAction(action_type="inspect", conflict_id=0))
        state = client.state()
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self, task_id: str = "task1") -> MergeObservation:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/reset?task_id={task_id}")
            response.raise_for_status()
            return MergeObservation(**response.json())

    def step(self, action: MergeAction) -> StepResult:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/step",
                json=action.model_dump(),
            )
            response.raise_for_status()
            return StepResult(**response.json())

    def state(self) -> EpisodeState:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/state")
            response.raise_for_status()
            return EpisodeState(**response.json())

    def tasks(self) -> list:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/tasks")
            response.raise_for_status()
            return response.json()

    def grader(self) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/grader")
            response.raise_for_status()
            return response.json()

    def validate(self) -> dict:
        """Call the /validate endpoint to run self-checks on the grader."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/validate")
            response.raise_for_status()
            return response.json()

    def baseline(self) -> dict:
        """Run the baseline agent against all tasks. Requires API credentials in env."""
        with httpx.Client(timeout=60.0) as client:  # Longer timeout for baseline
            response = client.post(f"{self.base_url}/baseline")
            response.raise_for_status()
            return response.json()

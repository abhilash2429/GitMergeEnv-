import re
import uuid
from typing import Optional

from models import EpisodeState, MergeAction, MergeObservation
from server.grader import ConflictGrader
from server.tasks import ALL_TASKS

CONFLICT_PATTERN = re.compile(
    r"<<<<<<< .+?\n(.*?)=======\n(.*?)>>>>>>> .+?\n",
    re.DOTALL,
)


class GitMergeEnvironment:
    """
    Core environment implementing reset(), step(), state().

    State is held in instance variables. Each call to reset()
    wipes all state and starts a fresh episode.

    Thread safety: not guaranteed. Each request should use its own instance,
    managed by the FastAPI dependency injection in app.py.
    """

    STEP_PENALTY = 0.01

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        """Zero out all episode state."""
        self.episode_id: str = ""
        self.task_id: str = ""
        self.task: Optional[dict] = None
        self.step_count: int = 0
        self.done: bool = False
        self.total_reward: float = 0.0
        self.current_file: str = ""
        self.original_file: str = ""
        self.ground_truth_file: str = ""
        self.ground_truth_blocks: list = []
        self.resolutions: dict = {}
        self.resolve_attempts: dict = {}
        self.conflict_blocks: list = []

    def reset(self, task_id: str = "task1") -> MergeObservation:
        """
        Initialize a fresh episode for the given task.

        Args:
            task_id: one of "task1", "task2", "task3"

        Returns:
            MergeObservation with initial state.

        Raises:
            ValueError if task_id is not recognized.
        """
        if task_id not in ALL_TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Must be one of: {list(ALL_TASKS.keys())}")

        self._reset_state()

        self.episode_id = str(uuid.uuid4())
        self.task_id = task_id
        self.task = ALL_TASKS[task_id]

        self.original_file = self.task["conflicted_file"]
        self.current_file = self.original_file
        self.ground_truth_file = self.task["ground_truth_file"]
        self.ground_truth_blocks = self.task["ground_truth_blocks"]

        self.conflict_blocks = self._parse_conflict_blocks(self.original_file)

        return MergeObservation(
            file_name=self.task["file_name"],
            total_conflicts=len(self.conflict_blocks),
            resolved_conflicts=0,
            unresolved_conflict_ids=list(range(len(self.conflict_blocks))),
            current_file_preview=self.current_file,
            last_action_feedback=(
                f"Episode started. File '{self.task['file_name']}' has "
                f"{len(self.conflict_blocks)} conflict(s) to resolve. "
                f"You have {self.task['max_steps']} steps. "
                f"Use 'inspect' to examine a conflict, 'resolve' to fix it, "
                f"'submit' when done."
            ),
            last_reward=0.0,
            steps_remaining=self.task["max_steps"],
        )

    def step(self, action: MergeAction):
        """
        Process one agent action.

        Returns:
            (MergeObservation, reward, done, info)

        Never raises. Invalid actions return negative reward and continue episode.
        """
        if self.done:
            obs = self._build_observation("Episode already done. Call reset() to start new episode.", 0.0)
            return obs, 0.0, True, {"error": "episode_already_done"}

        if self.task is None:
            obs = self._build_observation("No active episode. Call reset() first.", -0.1)
            return obs, -0.1, False, {"error": "no_active_episode"}

        self.step_count += 1
        steps_remaining = self.task["max_steps"] - self.step_count

        reward = 0.0
        feedback = ""
        info = {}

        if action.action_type == "inspect":
            reward, feedback, info = self._handle_inspect(action)
        elif action.action_type == "resolve":
            reward, feedback, info = self._handle_resolve(action)
        elif action.action_type == "submit":
            reward, feedback, info = self._handle_submit(action)
            self.done = True
        else:
            # Unknown action type — negative reward, episode continues
            reward = -0.10
            feedback = (
                f"Unknown action_type '{action.action_type}'. "
                f"Must be one of: inspect, resolve, submit. "
                f"No state change made."
            )
            info = {"error": "invalid_action_type"}

        reward -= self.STEP_PENALTY
        self.total_reward += reward

        if steps_remaining <= 0 and not self.done:
            self.done = True
            feedback += " [STEP LIMIT REACHED — episode terminated]"
            info["terminated_by_step_limit"] = True

        obs = self._build_observation(feedback, reward, steps_remaining=max(steps_remaining, 0))
        return obs, round(reward, 4), self.done, info

    def state(self) -> EpisodeState:
        """Return current episode metadata."""
        return EpisodeState(
            episode_id=self.episode_id,
            task_id=self.task_id,
            step_count=self.step_count,
            max_steps=self.task["max_steps"] if self.task else 0,
            done=self.done,
            total_reward=round(self.total_reward, 4),
            resolved_conflicts=len(self.resolutions),
            total_conflicts=len(self.conflict_blocks),
        )

    def _handle_inspect(self, action: MergeAction):
        """
        Return detailed context for a specific conflict block.
        Small positive reward for information gathering behavior.
        """
        if action.conflict_id is None:
            return -0.05, "inspect requires conflict_id to be set.", {"error": "missing_conflict_id"}

        if action.conflict_id < 0 or action.conflict_id >= len(self.conflict_blocks):
            return -0.05, (
                f"conflict_id {action.conflict_id} out of range. "
                f"Valid IDs: 0 to {len(self.conflict_blocks) - 1}."
            ), {"error": "conflict_id_out_of_range"}

        block = self.conflict_blocks[action.conflict_id]
        already_resolved = action.conflict_id in self.resolutions

        feedback = (
            f"--- Conflict Block {action.conflict_id} ---\n"
            f"Status: {'RESOLVED' if already_resolved else 'UNRESOLVED'}\n\n"
            f"HEAD version (current branch):\n{block['head_content']}\n\n"
            f"INCOMING version (feature branch):\n{block['incoming_content']}\n\n"
            f"Hint: Consider what each developer was trying to achieve. "
            f"The correct resolution may incorporate changes from both sides."
        )

        if already_resolved:
            feedback += f"\n\nYour current resolution:\n{self.resolutions[action.conflict_id]}"

        return 0.02, feedback, {"inspected_block": action.conflict_id}

    def _handle_resolve(self, action: MergeAction):
        """
        Record the agent's resolution for one conflict block.
        Gives immediate partial reward based on block-level grader.
        """
        if action.conflict_id is None:
            return -0.05, "resolve requires conflict_id to be set.", {"error": "missing_conflict_id"}

        if action.resolution is None or action.resolution.strip() == "":
            return -0.08, "resolve requires a non-empty resolution string.", {"error": "empty_resolution"}

        if action.conflict_id < 0 or action.conflict_id >= len(self.conflict_blocks):
            return -0.05, (
                f"conflict_id {action.conflict_id} out of range. "
                f"Valid range: 0 to {len(self.conflict_blocks) - 1}."
            ), {"error": "conflict_id_out_of_range"}

        # Track repeated resolve attempts per block and reduce reward for
        # repair loops that fail to make real progress.
        attempt_num = self.resolve_attempts.get(action.conflict_id, 0) + 1
        self.resolve_attempts[action.conflict_id] = attempt_num

        if attempt_num == 2:
            repetition_multiplier = 0.7
        elif attempt_num >= 3:
            repetition_multiplier = 0.4
        else:
            repetition_multiplier = 1.0

        # Check for conflict markers in the resolution itself — invalid
        if any(m in action.resolution for m in ["<<<<<<<", "=======", ">>>>>>>"]):
            return -0.10, (
                "Resolution contains git conflict markers. "
                "Your resolution must be clean code, not a conflict block."
            ), {"error": "resolution_contains_markers"}

        grader = ConflictGrader()

        if action.conflict_id < len(self.ground_truth_blocks):
            gt_block = self.ground_truth_blocks[action.conflict_id]
            block_score = grader.grade_block(action.resolution, gt_block)
        else:
            block_score = 0.0

        self.resolutions[action.conflict_id] = action.resolution
        self.current_file = self._apply_resolutions()

        if block_score == 1.0:
            reward = 0.15
            quality = "PERFECT"
        elif block_score >= 0.7:
            reward = 0.08
            quality = "GOOD"
        elif block_score >= 0.4:
            reward = 0.02
            quality = "PARTIAL"
        elif block_score > 0.0:
            reward = -0.02
            quality = "POOR"
        else:
            reward = -0.08
            quality = "INCORRECT"

        syntax_ok = True
        if len(self.resolutions) == len(self.conflict_blocks):
            try:
                import ast

                ast.parse(self.current_file)
            except SyntaxError:
                syntax_ok = False

        if not syntax_ok and reward > 0:
            reward = reward * 0.3

        unresolved = [index for index in range(len(self.conflict_blocks)) if index not in self.resolutions]
        feedback = (
            f"Block {action.conflict_id} resolved. "
            f"Immediate quality: {quality} (block score: {block_score:.2f}). "
            f"Resolved: {len(self.resolutions)}/{len(self.conflict_blocks)}. "
            f"Unresolved blocks: {unresolved}."
        )

        if not syntax_ok:
            feedback += " WARNING: current merged file has a syntax error. Review your indentation."

        reward = reward * repetition_multiplier
        if attempt_num > 1:
            feedback += (
                f" (attempt {attempt_num} on block {action.conflict_id} — "
                "diminishing returns apply)"
            )

        return reward, feedback, {
            "resolved_block": action.conflict_id,
            "block_score": block_score,
            "quality": quality,
        }

    def _handle_submit(self, action: MergeAction):
        """
        Finalize episode. Run full grader. Compute terminal reward.
        """
        grader = ConflictGrader()
        final_score, components = grader.grade(self.current_file, self.task)

        unresolved_count = len(self.conflict_blocks) - len(self.resolutions)
        unresolved_penalty = 0.10 * unresolved_count

        steps_used = self.step_count
        max_steps = self.task["max_steps"]
        efficiency_bonus = 0.0
        if final_score >= 0.9 and steps_used <= (max_steps * 0.5):
            efficiency_bonus = 0.05

        consistency_bonus = self._check_resolution_consistency()

        terminal_reward = final_score - unresolved_penalty + efficiency_bonus + consistency_bonus

        if not grader._parses_cleanly(self.current_file):
            score_explanation = (
                "LOW SCORE REASON: merged file has a syntax error. "
                "Review indentation in your resolutions."
            )
        elif final_score < 0.3 and len(self.resolutions) == len(self.conflict_blocks):
            score_explanation = (
                "LOW SCORE REASON: all blocks resolved but content does not "
                "match ground truth patterns."
            )
        else:
            score_explanation = ""

        feedback = (
            f"Episode complete. Final score: {final_score:.4f}. "
            f"{score_explanation} "
            f"Unresolved penalty: -{unresolved_penalty:.2f}. "
            f"Efficiency bonus: +{efficiency_bonus:.2f}. "
            f"Consistency bonus: +{consistency_bonus:.2f}. "
            f"Terminal reward: {terminal_reward:.4f}. "
            f"Components: {components}."
        )

        return round(terminal_reward, 4), feedback, {
            "final_score": final_score,
            "components": components,
            "unresolved_penalty": unresolved_penalty,
            "efficiency_bonus": efficiency_bonus,
            "consistency_bonus": consistency_bonus,
        }

    def _check_resolution_consistency(self) -> float:
        """
        For multi-conflict tasks, check if all resolutions are internally consistent.
        Detects cases where agent mixed old and new patterns across blocks.
        Returns 0.0 to 0.10 bonus.
        """
        if len(self.resolutions) < 2:
            return 0.0

        all_resolved = "\n".join(self.resolutions.values())

        conflict_pairs = [
            ("Session(engine)", "cursor.execute"),
            ("CustomError", "ValueError"),
            ("import logging", "print("),
        ]

        mixed_count = 0
        for new_pattern, old_pattern in conflict_pairs:
            if new_pattern in all_resolved and old_pattern in all_resolved:
                mixed_count += 1

        if mixed_count == 0:
            return 0.08
        if mixed_count == 1:
            return 0.03
        return 0.0

    def _parse_conflict_blocks(self, file_content: str) -> list:
        """
        Parse all conflict blocks from a conflicted file.
        Returns list of dicts with id, head_content, incoming_content.
        """
        blocks = []
        pattern = re.compile(
            r"<<<<<<< [^\n]+\n(.*?)=======\n(.*?)>>>>>>> [^\n]+\n",
            re.DOTALL,
        )
        for index, match in enumerate(pattern.finditer(file_content)):
            blocks.append(
                {
                    "id": index,
                    "head_content": match.group(1),
                    "incoming_content": match.group(2),
                    "full_marker_text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        return blocks

    def _apply_resolutions(self) -> str:
        """
        Rebuild the file with all current resolutions applied.
        Unresolved blocks remain as conflict markers.
        """
        result = self.original_file
        for block_id in sorted(self.resolutions.keys(), reverse=True):
            if block_id < len(self.conflict_blocks):
                block = self.conflict_blocks[block_id]
                resolution = self.resolutions[block_id]
                result = result.replace(block["full_marker_text"], resolution + "\n", 1)
        return result

    def _build_observation(
        self,
        feedback: str,
        last_reward: float,
        steps_remaining: Optional[int] = None,
    ) -> MergeObservation:
        """Build a MergeObservation from current state."""
        if steps_remaining is None and self.task:
            steps_remaining = max(self.task["max_steps"] - self.step_count, 0)

        unresolved = [
            index for index in range(len(self.conflict_blocks))
            if index not in self.resolutions
        ]

        return MergeObservation(
            file_name=self.task["file_name"] if self.task else "unknown",
            total_conflicts=len(self.conflict_blocks),
            resolved_conflicts=len(self.resolutions),
            unresolved_conflict_ids=unresolved,
            current_file_preview=self.current_file,
            last_action_feedback=feedback,
            last_reward=round(last_reward, 4),
            steps_remaining=steps_remaining or 0,
        )

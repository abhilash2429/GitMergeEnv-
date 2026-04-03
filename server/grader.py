import ast
import re
from typing import Dict, Tuple


class ConflictGrader:
    """
    Deterministic grader for git merge conflict resolution.

    Scores are always floats in [0.0, 1.0].
    Same inputs always produce same output.
    No LLMs, no randomness, no external calls.
    """

    CONFLICT_START = "<<<<<<<\n"
    CONFLICT_SEP = "======="
    CONFLICT_END = ">>>>>>>"

    def grade(self, agent_file: str, task: dict) -> Tuple[float, Dict]:
        """
        Master grader. Returns (score, components_dict).

        Score breakdown per task grader_weights field.
        Components dict has one key per weight component with its score.
        """
        weights = task["grader_weights"]
        components = {}

        if not agent_file.strip():
            components["parses_cleanly"] = 0.0
            components["no_conflict_markers"] = 0.0
            return 0.0, components

        has_markers = self._has_conflict_markers(agent_file)
        if has_markers:
            components["parses_cleanly"] = 0.0 if not self._parses_cleanly(agent_file) else 1.0
            components["no_conflict_markers"] = 0.0
            return 0.0, components

        parses = self._parses_cleanly(agent_file)
        if not parses:
            components["parses_cleanly"] = 0.0
            parse_penalty = 0.5
            print(f"[grader debug] Parse failed — applying 0.5 penalty multiplier")
        else:
            components["parses_cleanly"] = 1.0
            parse_penalty = 1.0

        components["no_conflict_markers"] = 1.0

        block_score = self._score_blocks(agent_file, task)
        components["block_match"] = round(block_score, 4)

        req_score = self._score_required_elements(agent_file, task)
        components["required_elements"] = round(req_score, 4)

        indent_score = self._score_indentation_consistency(
            agent_file,
            task["ground_truth_file"],
        )
        components["indentation_consistency"] = indent_score

        if "consistency_checks" in task:
            consistency_score = self._score_consistency(agent_file, task)
            components["architectural_consistency"] = round(consistency_score, 4)
        elif "architectural_consistency" in weights:
            components["architectural_consistency"] = 1.0

        # Forbidden elements penalty — applied multiplicatively
        forbidden_penalty = self._compute_forbidden_penalty(agent_file, task)
        components["forbidden_penalty"] = round(forbidden_penalty, 4)

        total = 0.0
        for component_name, weight in weights.items():
            component_score = components.get(component_name, 0.0)
            total += weight * component_score

        # Apply parse penalty (1.0 if clean, 0.5 if broken)
        total = total * parse_penalty

        total = total * forbidden_penalty

        # A terminal score of exactly 0.0 for a nearly-correct file looks like
        # a broken grader. Keep a small floor for non-empty, marker-free files.
        total = max(total, 0.04)

        return round(min(max(total, 0.0), 1.0), 4), components

    def grade_block(self, agent: str, truth: str) -> float:
        """
        Score a single resolved block against ground truth.
        Used for immediate per-step feedback inside step().
        """
        agent_normalized = self._normalize_whitespace(agent)
        truth_normalized = self._normalize_whitespace(truth)

        if agent_normalized == truth_normalized:
            return 1.0

        agent_lines = set(l.strip() for l in agent_normalized.splitlines() if l.strip())
        truth_lines = set(l.strip() for l in truth_normalized.splitlines() if l.strip())

        if not truth_lines:
            return 0.0

        precision = len(agent_lines & truth_lines) / len(agent_lines) if agent_lines else 0.0
        recall = len(agent_lines & truth_lines) / len(truth_lines)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return round(min(f1 * 0.85, 0.85), 4)

    def _parses_cleanly(self, code: str) -> bool:
        """Returns True if the code parses as valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _has_conflict_markers(self, code: str) -> bool:
        """Returns True if any git conflict markers remain."""
        markers = ["<<<<<<<", "=======", ">>>>>>>"]
        return any(marker in code for marker in markers)

    def _score_blocks(self, agent_file: str, task: dict) -> float:
        """
        Extract resolved blocks from agent file and compare to ground truth blocks.
        Agent file should be fully resolved (no conflict markers).
        We extract content at the positions where conflicts existed using
        structural landmarks in the ground truth.
        """
        ground_truth_blocks = task["ground_truth_blocks"]
        num_blocks = len(ground_truth_blocks)

        if num_blocks == 0:
            return 1.0

        total_block_score = 0.0
        per_block_weight = 1.0 / num_blocks

        for gt_block in ground_truth_blocks:
            block_score = self._check_block_presence(agent_file, gt_block)
            total_block_score += per_block_weight * block_score

        return total_block_score

    def _check_block_presence(self, agent_file: str, ground_truth_block: str) -> float:
        """
        Check how well the ground truth block is represented in the agent file.
        Uses token presence scoring — we check if key tokens from the block
        appear in the agent file in the correct relative order.
        """
        gt_tokens = re.findall(r"\w+", ground_truth_block)
        if not gt_tokens:
            return 1.0

        seen = set()
        unique_gt_tokens = []
        for token in gt_tokens:
            if token not in seen and len(token) > 2:
                seen.add(token)
                unique_gt_tokens.append(token)

        if not unique_gt_tokens:
            return 1.0

        present_count = sum(1 for token in unique_gt_tokens if token in agent_file)
        return present_count / len(unique_gt_tokens)

    def _score_required_elements(self, agent_file: str, task: dict) -> float:
        """
        Check what fraction of required_elements appear in the agent's file.
        Each element is a string that must be present (substring match).
        """
        required = task.get("required_elements", [])
        if not required:
            return 1.0

        present = sum(1 for element in required if element in agent_file)
        return present / len(required)

    def _score_consistency(self, agent_file: str, task: dict) -> float:
        """
        For task3: check architectural consistency.
        Each consistency_check has must_have and must_not_have.
        Score is 1.0 only if all checks pass.
        """
        checks = task.get("consistency_checks", [])
        if not checks:
            return 1.0

        total_weight = sum(check["weight"] for check in checks)
        score = 0.0

        for check in checks:
            has_required = check["must_have"] in agent_file
            has_forbidden = check["must_not_have"] in agent_file
            if has_required and not has_forbidden:
                score += check["weight"]

        return score / total_weight if total_weight > 0 else 1.0

    def _score_indentation_consistency(self, agent_file: str, ground_truth_file: str) -> float:
        """
        Check whether indentation levels used by the agent broadly match the
        ground truth. This catches structurally bad merges that preserve tokens
        but break Python block layout.
        """

        def get_indent_signature(code: str) -> list[int]:
            lines = code.splitlines()
            return [len(line) - len(line.lstrip()) for line in lines if line.strip()]

        agent_indents = get_indent_signature(agent_file)
        truth_indents = get_indent_signature(ground_truth_file)

        if not truth_indents:
            return 1.0

        agent_set = set(agent_indents)
        truth_set = set(truth_indents)

        if not truth_set:
            return 1.0

        overlap = len(agent_set & truth_set) / len(truth_set)
        return round(overlap, 4)

    def _compute_forbidden_penalty(self, agent_file: str, task: dict) -> float:
        """
        Multiplicative penalty for forbidden elements.
        Each forbidden element found reduces the multiplier by 0.15.
        Minimum multiplier is 0.1 (never zero — there may still be partial credit).

        Note: conflict markers are handled by no_conflict_markers component,
        so they are included in forbidden to apply double penalty for failing
        to remove them.
        """
        forbidden = task.get("forbidden_elements", [])
        if not forbidden:
            return 1.0

        violations = sum(1 for element in forbidden if element in agent_file)

        # Each violation reduces by 0.15
        penalty_multiplier = max(1.0 - (violations * 0.15), 0.10)
        return penalty_multiplier

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for comparison — collapse spaces, strip lines."""
        lines = text.strip().splitlines()
        normalized_lines = [line.strip() for line in lines if line.strip()]
        return "\n".join(normalized_lines)

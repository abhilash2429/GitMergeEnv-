"""
Task 1 — Easy: Single Variable Rename Conflict

Scenario:
  Developer A renamed variable 'user_data' to 'user_info' throughout a function.
  Developer B added a 'timeout' parameter to the same function using the old name.
  Git produced one conflict block. Agent must synthesize both changes.

Correct resolution:
  Use Developer A's new name (user_info) AND include Developer B's new parameter (timeout=30).
  This tests whether the agent understands it must merge both changes, not pick one side.

Ground truth required elements:
  - "user_info" must appear (A's rename)
  - "timeout=30" must appear (B's new param)
  - "transform(user_info)" must appear (consistent naming)
  - No conflict markers must remain
  - File must parse with ast.parse()
"""

TASK1 = {
    "id": "task1",
    "name": "Single Conflict — Variable Rename",
    "difficulty": "easy",
    "description": (
        "Two developers modified the same Python function. Developer A renamed "
        "the parameter 'user_data' to 'user_info' for consistency. Developer B "
        "added a new 'timeout' parameter to the function. Git cannot auto-resolve "
        "this. Your task is to merge both changes correctly into a single coherent "
        "function definition."
    ),
    "file_name": "processor.py",
    "max_steps": 6,
    "num_conflicts": 1,
    "conflicted_file": '''\
import logging

logger = logging.getLogger(__name__)


<<<<<<< HEAD
def process_user(user_info, config):
    """Process a user record with the given config."""
    logger.debug("Processing user")
    result = transform(user_info)
    validated = validate(result, config)
    return validated
=======
def process_user(user_data, config, timeout=30):
    """Process a user record with the given config."""
    logger.debug("Processing user")
    result = transform(user_data)
    validated = validate(result, config)
    return validated
>>>>>>> feature/add-timeout


def validate(data, config):
    if not data:
        raise ValueError("Empty data")
    return data
''',
    "ground_truth_file": '''\
import logging

logger = logging.getLogger(__name__)


def process_user(user_info, config, timeout=30):
    """Process a user record with the given config."""
    logger.debug("Processing user")
    result = transform(user_info)
    validated = validate(result, config)
    return validated


def validate(data, config):
    if not data:
        raise ValueError("Empty data")
    return data
''',
    "ground_truth_blocks": [
        '''\
def process_user(user_info, config, timeout=30):
    """Process a user record with the given config."""
    logger.debug("Processing user")
    result = transform(user_info)
    validated = validate(result, config)
    return validated'''
    ],
    "required_elements": [
        "user_info",
        "timeout=30",
        "transform(user_info)",
    ],
    "forbidden_elements": [
        "<<<<<<<",
        "=======",
        ">>>>>>>",
        "transform(user_data)",
    ],
    "grader_weights": {
        "parses_cleanly": 0.15,
        "no_conflict_markers": 0.10,
        "block_match": 0.40,
        "required_elements": 0.25,
        "structural_similarity": 0.10,
    },
    "expected_baseline_score": (0.75, 0.95),
}

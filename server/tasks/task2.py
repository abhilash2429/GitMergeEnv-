"""
Task 2 — Medium: Three Conflicts in a Class

Scenario:
  Developer A refactored exception handling to use custom exceptions.
  Developer B added structured logging throughout the same class.
  Three conflict blocks result across imports, method body, and docstring.

  Conflict 0 (imports): A added CustomError import, B added logging import.
    Correct: include BOTH imports.

  Conflict 1 (method body): A changed raise ValueError to raise CustomError,
    B added logger.warning call before the raise.
    Correct: keep logger.warning AND use CustomError (not ValueError).

  Conflict 2 (docstring): A documented the new exception, B documented logging.
    Correct: mention BOTH in the docstring.

This tests whether the agent can handle multiple independent conflicts
without letting resolutions of early conflicts pollute later ones.
"""

TASK2 = {
    "id": "task2",
    "name": "Three Conflicts — Class Refactor",
    "difficulty": "medium",
    "description": (
        "Two developers modified the same Python class simultaneously. "
        "Developer A refactored all exception handling to use a custom "
        "exception class. Developer B added structured logging throughout "
        "the class. Three conflict blocks were produced across the imports "
        "section, a method body, and the class docstring. "
        "You must resolve all three conflicts, preserving both developers' "
        "changes where they are compatible."
    ),
    "file_name": "data_service.py",
    "max_steps": 12,
    "num_conflicts": 3,
    "conflicted_file": '''\
<<<<<<< HEAD
from exceptions import CustomError, ValidationError
=======
import logging
from exceptions import ValidationError

logger = logging.getLogger(__name__)
>>>>>>> feature/add-logging


class DataService:
<<<<<<< HEAD
    """
    Service for processing data records.
    Raises CustomError on invalid input.
    Uses ValidationError for schema violations.
    """
=======
    """
    Service for processing data records.
    All operations are logged at WARNING level on failure.
    Uses structured logging with context fields.
    """
>>>>>>> feature/add-logging

    def __init__(self, config):
        self.config = config
        self._cache = {}

    def process(self, record):
        if not record:
<<<<<<< HEAD
            raise CustomError("Record cannot be empty", code=400)
=======
            logger.warning("process() called with empty record", extra={"config": self.config})
            raise ValueError("Record cannot be empty")
>>>>>>> feature/add-logging
        return self._transform(record)

    def _transform(self, record):
        return {k: v for k, v in record.items() if v is not None}
''',
    "ground_truth_file": '''\
from exceptions import CustomError, ValidationError
import logging

logger = logging.getLogger(__name__)


class DataService:
    """
    Service for processing data records.
    Raises CustomError on invalid input.
    Uses ValidationError for schema violations.
    All operations are logged at WARNING level on failure.
    Uses structured logging with context fields.
    """

    def __init__(self, config):
        self.config = config
        self._cache = {}

    def process(self, record):
        if not record:
            logger.warning("process() called with empty record", extra={"config": self.config})
            raise CustomError("Record cannot be empty", code=400)
        return self._transform(record)

    def _transform(self, record):
        return {k: v for k, v in record.items() if v is not None}
''',
    "ground_truth_blocks": [
        '''\
from exceptions import CustomError, ValidationError
import logging

logger = logging.getLogger(__name__)''',
        '''\
    """
    Service for processing data records.
    Raises CustomError on invalid input.
    Uses ValidationError for schema violations.
    All operations are logged at WARNING level on failure.
    Uses structured logging with context fields.
    """''',
        '''\
            logger.warning("process() called with empty record", extra={"config": self.config})
            raise CustomError("Record cannot be empty", code=400)''',
    ],
    "required_elements": [
        "CustomError",
        "import logging",
        "logger = logging.getLogger",
        "logger.warning",
        "raise CustomError",
        "code=400",
    ],
    "forbidden_elements": [
        "<<<<<<<",
        "=======",
        ">>>>>>>",
        "raise ValueError",
    ],

    "grader_weights": {
        "no_conflict_markers": 0.10,
        "block_match": 0.50,
        "required_elements": 0.40,
    },

    "expected_baseline_score": (0.45, 0.70),
}

# AGENTS.md — GitMergeEnv: Complete Build Specification

## Prime Directive

You are building a production-ready OpenEnv environment called **GitMergeEnv**.
This is a real-world RL training environment where an AI agent learns to resolve
git merge conflicts in Python source files.

Read this entire document before writing a single line of code.
Follow every instruction exactly. Do not improvise architecture.
Do not add features not specified here.
Do not simplify anything marked as required.

---

## What You Are Building

A FastAPI web server, containerized via Docker, deployable to Hugging Face Spaces,
that implements the OpenEnv spec. The environment presents an agent with a Python
file containing git merge conflict markers. The agent resolves conflicts step by
step and is scored deterministically against a ground truth resolution.

This is a backend service only. No frontend. No database. No external APIs.
Everything is self-contained and stateless between episodes.

---

## Final Project Structure

Reproduce this structure exactly. Every file listed must exist.

```
git_merge_env/
├── AGENTS.md                          # this file (include in repo)
├── README.md                          # documentation (spec below)
├── openenv.yaml                       # OpenEnv manifest (spec below)
├── pyproject.toml                     # package config (spec below)
├── .gitignore                         # standard Python gitignore
├── .env.example                       # example env vars
├── baseline.py                        # baseline inference script (spec below)
├── models.py                          # Pydantic models (spec below)
├── client.py                          # EnvClient subclass (spec below)
└── server/
    ├── __init__.py                    # empty
    ├── app.py                         # FastAPI application (spec below)
    ├── environment.py                 # core environment logic (spec below)
    ├── grader.py                      # ConflictGrader class (spec below)
    ├── requirements.txt               # server dependencies (spec below)
    ├── Dockerfile                     # container spec (spec below)
    └── tasks/
        ├── __init__.py                # exports all three tasks
        ├── task1.py                   # Easy task scenario (spec below)
        ├── task2.py                   # Medium task scenario (spec below)
        └── task3.py                   # Hard task scenario (spec below)
```

---

## Dependency Versions — Pin All of These

### server/requirements.txt

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.8.2
openenv-core==0.2.1
python-dotenv==1.0.1
httpx==0.27.2
openai==1.40.0
```

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "git_merge_env"
version = "0.1.0"
description = "OpenEnv environment for git merge conflict resolution"
requires-python = ">=3.10"
dependencies = [
    "fastapi==0.115.0",
    "uvicorn[standard]==0.30.6",
    "pydantic==2.8.2",
    "openenv-core==0.2.1",
    "python-dotenv==1.0.1",
    "httpx==0.27.2",
    "openai==1.40.0",
]

[project.optional-dependencies]
dev = ["pytest==8.3.2", "pytest-asyncio==0.23.8", "httpx==0.27.2"]

[tool.setuptools.packages.find]
where = ["."]
```

---

## openenv.yaml

```yaml
name: git_merge_env
version: "0.1.0"
description: >
  A real-world RL environment where an agent resolves git merge conflicts
  in Python source files. The agent inspects conflict blocks, proposes
  resolutions, and is scored deterministically against ground truth.
author: "abhilash2429"
tags:
  - openenv
  - software-engineering
  - git
  - code-review
  - real-world
tasks:
  - id: task1
    name: "Single Conflict — Variable Rename"
    difficulty: easy
  - id: task2
    name: "Three Conflicts — Class Refactor"
    difficulty: medium
  - id: task3
    name: "Five Conflicts — Architectural Migration"
    difficulty: hard
action_schema:
  action_type:
    type: string
    enum: ["inspect", "resolve", "submit"]
    required: true
  conflict_id:
    type: integer
    required: false
    description: "0-indexed conflict block ID. Required for inspect and resolve."
  resolution:
    type: string
    required: false
    description: "The resolved content for the conflict block. Required for resolve."
observation_schema:
  file_name: string
  total_conflicts: integer
  resolved_conflicts: integer
  unresolved_conflict_ids: array
  current_file_preview: string
  last_action_feedback: string
  last_reward: float
  steps_remaining: integer
```

---

## models.py — Complete Implementation

```python
from pydantic import BaseModel, Field
from typing import Optional, List


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
    reward: float
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
    task_scores: dict
    average_score: float
    model_used: str
```

---

## server/tasks/task1.py — Easy Task

This file defines Task 1 as a Python dictionary. Hardcoded. No generation.

```python
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

    # The raw conflicted file content exactly as git would produce it
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

    # The single correct resolution — what the merged file should look like
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

    # Ground truth per conflict block (0-indexed list, one entry per conflict)
    "ground_truth_blocks": [
        '''\
def process_user(user_info, config, timeout=30):
    """Process a user record with the given config."""
    logger.debug("Processing user")
    result = transform(user_info)
    validated = validate(result, config)
    return validated'''
    ],

    # Required string elements that MUST appear in the correct final resolution
    # Used by grader semantic component
    "required_elements": [
        "user_info",
        "timeout=30",
        "transform(user_info)",
    ],

    # Elements that must NOT appear — degenerate/wrong resolutions
    "forbidden_elements": [
        "<<<<<<<",
        "=======",
        ">>>>>>>",
        "transform(user_data)",   # old name, wrong
    ],

    # Grader weight breakdown — must sum to 1.0
    "grader_weights": {
        "parses_cleanly": 0.15,
        "no_conflict_markers": 0.10,
        "block_match": 0.50,
        "required_elements": 0.25,
    },

    # Baseline expected score range for a GPT-4 level agent
    "expected_baseline_score": (0.75, 0.95),
}
```

---

## server/tasks/task2.py — Medium Task

```python
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
        # Block 0: imports
        '''\
from exceptions import CustomError, ValidationError
import logging

logger = logging.getLogger(__name__)''',

        # Block 1: docstring
        '''\
    """
    Service for processing data records.
    Raises CustomError on invalid input.
    Uses ValidationError for schema violations.
    All operations are logged at WARNING level on failure.
    Uses structured logging with context fields.
    """''',

        # Block 2: method body
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
        "raise ValueError",      # wrong — must use CustomError
    ],

    "grader_weights": {
        "parses_cleanly": 0.10,
        "no_conflict_markers": 0.10,
        "block_match": 0.50,     # 3 blocks, ~0.167 each
        "required_elements": 0.30,
    },

    "expected_baseline_score": (0.45, 0.70),
}
```

---

## server/tasks/task3.py — Hard Task

```python
"""
Task 3 — Hard: Five Conflicts — Architectural Migration

Scenario:
  Developer A migrated the data access layer from raw sqlite3 to SQLAlchemy ORM.
  Developer B added three new query features using the old raw sqlite3 pattern.
  Five conflict blocks result.

  CRITICAL DEPENDENCY: The conflicts are NOT independently resolvable.
  Conflict 0 establishes which approach wins (ORM must win — it's a breaking migration).
  Conflicts 1-4 must ALL be resolved consistently with the ORM approach.

  An agent that resolves each block independently without tracking architectural
  consistency will produce a logically broken file — mixing ORM and raw SQL.
  This is what separates agents that reason holistically from those that don't.

  Correct approach: SQLAlchemy ORM wins across all five blocks.
  Developer B's new features must be re-implemented using ORM syntax.

Block summary:
  Conflict 0: imports — raw sqlite3 vs sqlalchemy imports. ORM wins.
  Conflict 1: connection setup — sqlite3.connect() vs Session(). ORM wins.
  Conflict 2: basic query — cursor.execute() vs session.query(). ORM wins.
  Conflict 3: new feature (B added) — cursor-based insert. Must convert to ORM.
  Conflict 4: new feature (B added) — cursor-based delete. Must convert to ORM.
"""

TASK3 = {
    "id": "task3",
    "name": "Five Conflicts — Architectural Migration",
    "difficulty": "hard",
    "description": (
        "Two developers modified the same database access module simultaneously. "
        "Developer A completed a full migration from raw sqlite3 to SQLAlchemy ORM. "
        "Developer B, unaware of the migration, added two new features using the old "
        "raw sqlite3 pattern. Five conflict blocks were produced. "
        "The conflicts are architecturally dependent — you must resolve all five "
        "consistently using the SQLAlchemy ORM approach. Developer B's new features "
        "must be re-implemented using ORM syntax, not carried over as-is. "
        "Mixing ORM and raw SQL in the final file is considered a failed resolution."
    ),
    "file_name": "db_access.py",
    "max_steps": 18,
    "num_conflicts": 5,

    "conflicted_file": '''\
<<<<<<< HEAD
from sqlalchemy import create_engine, Column, Integer, String, select, delete
from sqlalchemy.orm import DeclarativeBase, Session

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    role = Column(String, default="user")

engine = create_engine("sqlite:///app.db")
Base.metadata.create_all(engine)
=======
import sqlite3

DB_PATH = "app.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
>>>>>>> feature/new-queries


def get_user_by_id(user_id: int):
<<<<<<< HEAD
    with Session(engine) as session:
        return session.get(User, user_id)
=======
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()
>>>>>>> feature/new-queries


def get_all_users():
<<<<<<< HEAD
    with Session(engine) as session:
        return session.execute(select(User)).scalars().all()
=======
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    return cursor.fetchall()
>>>>>>> feature/new-queries


def create_user(name: str, email: str, role: str = "user"):
<<<<<<< HEAD
    with Session(engine) as session:
        user = User(name=name, email=email, role=role)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
=======
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, email, role) VALUES (?, ?, ?)",
        (name, email, role)
    )
    conn.commit()
    return cursor.lastrowid
>>>>>>> feature/new-queries


def delete_user(user_id: int):
<<<<<<< HEAD
    with Session(engine) as session:
        user = session.get(User, user_id)
        if user:
            session.delete(user)
            session.commit()
            return True
        return False
=======
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    return cursor.rowcount > 0
>>>>>>> feature/new-queries
''',

    "ground_truth_file": '''\
from sqlalchemy import create_engine, Column, Integer, String, select, delete
from sqlalchemy.orm import DeclarativeBase, Session

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    role = Column(String, default="user")

engine = create_engine("sqlite:///app.db")
Base.metadata.create_all(engine)


def get_user_by_id(user_id: int):
    with Session(engine) as session:
        return session.get(User, user_id)


def get_all_users():
    with Session(engine) as session:
        return session.execute(select(User)).scalars().all()


def create_user(name: str, email: str, role: str = "user"):
    with Session(engine) as session:
        user = User(name=name, email=email, role=role)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def delete_user(user_id: int):
    with Session(engine) as session:
        user = session.get(User, user_id)
        if user:
            session.delete(user)
            session.commit()
            return True
        return False
''',

    "ground_truth_blocks": [
        # Block 0: imports and setup
        '''\
from sqlalchemy import create_engine, Column, Integer, String, select, delete
from sqlalchemy.orm import DeclarativeBase, Session

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    role = Column(String, default="user")

engine = create_engine("sqlite:///app.db")
Base.metadata.create_all(engine)''',

        # Block 1: get_user_by_id
        '''\
    with Session(engine) as session:
        return session.get(User, user_id)''',

        # Block 2: get_all_users
        '''\
    with Session(engine) as session:
        return session.execute(select(User)).scalars().all()''',

        # Block 3: create_user
        '''\
    with Session(engine) as session:
        user = User(name=name, email=email, role=role)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user''',

        # Block 4: delete_user
        '''\
    with Session(engine) as session:
        user = session.get(User, user_id)
        if user:
            session.delete(user)
            session.commit()
            return True
        return False''',
    ],

    "required_elements": [
        "from sqlalchemy",
        "Session(engine)",
        "session.get(User",
        "session.add(",
        "session.commit()",
        "session.delete(",
        "select(User)",
    ],

    "forbidden_elements": [
        "<<<<<<<",
        "=======",
        ">>>>>>>",
        "import sqlite3",           # old approach must not survive
        "sqlite3.connect",          # old approach
        "cursor.execute",           # old approach
        "conn.commit()",            # old approach
        "get_connection()",         # old helper must not survive
    ],

    # Architectural consistency check: these pairs must co-exist in final file
    # If ORM imports exist but cursor.execute also exists, that's a failed merge
    "consistency_checks": [
        {
            "must_have": "Session(engine)",
            "must_not_have": "cursor.execute",
            "label": "orm_consistency",
            "weight": 0.15,
        }
    ],

    "grader_weights": {
        "parses_cleanly": 0.05,
        "no_conflict_markers": 0.05,
        "block_match": 0.50,        # 5 blocks, 0.10 each
        "required_elements": 0.25,
        "architectural_consistency": 0.15,
    },

    "expected_baseline_score": (0.20, 0.45),
}
```

---

## server/tasks/__init__.py

```python
from server.tasks.task1 import TASK1
from server.tasks.task2 import TASK2
from server.tasks.task3 import TASK3

ALL_TASKS = {
    "task1": TASK1,
    "task2": TASK2,
    "task3": TASK3,
}

TASK_LIST = [TASK1, TASK2, TASK3]
```

---

## server/grader.py — Complete Implementation

This is the most critical file. Implement exactly as specified.
Every function must be deterministic — same inputs always return same output.
No randomness. No LLM calls. No external API calls.

```python
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

    CONFLICT_START = "<<<<<<<\n"  # simplified marker pattern
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

        # Component: file parses cleanly
        parses = self._parses_cleanly(agent_file)
        components["parses_cleanly"] = 1.0 if parses else 0.0

        # If file doesn't parse, cap total at 0.15 max
        # A broken file is nearly useless even with good content
        if not parses:
            total = weights.get("parses_cleanly", 0.15) * components["parses_cleanly"]
            return round(min(total, 0.15), 4), components

        # Component: no conflict markers remaining
        has_markers = self._has_conflict_markers(agent_file)
        components["no_conflict_markers"] = 0.0 if has_markers else 1.0

        # Component: block-level match
        block_score = self._score_blocks(agent_file, task)
        components["block_match"] = round(block_score, 4)

        # Component: required elements present
        req_score = self._score_required_elements(agent_file, task)
        components["required_elements"] = round(req_score, 4)

        # Component: architectural consistency (task3 only)
        if "consistency_checks" in task:
            consistency_score = self._score_consistency(agent_file, task)
            components["architectural_consistency"] = round(consistency_score, 4)
        elif "architectural_consistency" in weights:
            components["architectural_consistency"] = 1.0

        # Forbidden elements penalty — applied multiplicatively
        forbidden_penalty = self._compute_forbidden_penalty(agent_file, task)
        components["forbidden_penalty"] = round(forbidden_penalty, 4)

        # Weighted sum
        total = 0.0
        for component_name, weight in weights.items():
            component_score = components.get(component_name, 0.0)
            total += weight * component_score

        # Apply forbidden penalty (multiplicative, not additive)
        total = total * forbidden_penalty

        return round(min(max(total, 0.0), 1.0), 4), components

    def grade_block(self, agent_block: str, ground_truth_block: str) -> float:
        """
        Score a single resolved block against ground truth.
        Used for immediate per-step feedback inside step().

        Returns:
          1.0  — exact match (whitespace normalized)
          0.5-0.9 — high token overlap (good but not perfect)
          0.1-0.49 — partial token overlap
          0.0  — no meaningful overlap
        """
        agent_normalized = self._normalize_whitespace(agent_block)
        truth_normalized = self._normalize_whitespace(ground_truth_block)

        # Exact match after normalization
        if agent_normalized == truth_normalized:
            return 1.0

        # Token overlap (Jaccard-like)
        agent_tokens = set(re.findall(r'\w+|[^\w\s]', agent_normalized))
        truth_tokens = set(re.findall(r'\w+|[^\w\s]', truth_normalized))

        if not truth_tokens:
            return 0.0

        intersection = agent_tokens & truth_tokens
        union = agent_tokens | truth_tokens
        jaccard = len(intersection) / len(union)

        # Scale: jaccard of 1.0 = 0.9 (can't get 1.0 without exact match)
        return round(jaccard * 0.9, 4)

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
            # Check if the key identifying tokens of this block appear in agent file
            block_score = self._check_block_presence(agent_file, gt_block)
            total_block_score += per_block_weight * block_score

        return total_block_score

    def _check_block_presence(self, agent_file: str, ground_truth_block: str) -> float:
        """
        Check how well the ground truth block is represented in the agent file.
        Uses token presence scoring — we check if key tokens from the block
        appear in the agent file in the correct relative order.
        """
        gt_tokens = re.findall(r'\w+', ground_truth_block)
        if not gt_tokens:
            return 1.0

        # Deduplicate while preserving order
        seen = set()
        unique_gt_tokens = []
        for t in gt_tokens:
            if t not in seen and len(t) > 2:  # skip tiny tokens like 'if', 'in'
                seen.add(t)
                unique_gt_tokens.append(t)

        if not unique_gt_tokens:
            return 1.0

        present_count = sum(1 for t in unique_gt_tokens if t in agent_file)
        return present_count / len(unique_gt_tokens)

    def _score_required_elements(self, agent_file: str, task: dict) -> float:
        """
        Check what fraction of required_elements appear in the agent's file.
        Each element is a string that must be present (substring match).
        """
        required = task.get("required_elements", [])
        if not required:
            return 1.0

        present = sum(1 for el in required if el in agent_file)
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

        total_weight = sum(c["weight"] for c in checks)
        score = 0.0

        for check in checks:
            has_required = check["must_have"] in agent_file
            has_forbidden = check["must_not_have"] in agent_file
            if has_required and not has_forbidden:
                score += check["weight"]

        return score / total_weight if total_weight > 0 else 1.0

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

        violations = sum(1 for el in forbidden if el in agent_file)

        # Each violation reduces by 0.15
        penalty_multiplier = max(1.0 - (violations * 0.15), 0.10)
        return penalty_multiplier

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for comparison — collapse spaces, strip lines."""
        lines = text.strip().splitlines()
        normalized_lines = [line.strip() for line in lines if line.strip()]
        return "\n".join(normalized_lines)
```

---

## server/environment.py — Core Environment Logic

```python
import uuid
import copy
import re
from typing import Optional
from server.grader import ConflictGrader
from server.tasks import ALL_TASKS
from models import MergeAction, MergeObservation, EpisodeState


CONFLICT_PATTERN = re.compile(
    r'<<<<<<< .+?\n(.*?)=======\n(.*?)>>>>>>> .+?\n',
    re.DOTALL
)


class GitMergeEnvironment:
    """
    Core environment implementing reset(), step(), state().

    State is held in instance variables. Each call to reset()
    wipes all state and starts a fresh episode.

    Thread safety: not guaranteed. Each request should use its own instance,
    managed by the FastAPI dependency injection in app.py.
    """

    # Per-step time penalty applied on every action regardless of type
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

        # Current working copy of the file content
        # Modified as agent resolves blocks
        self.current_file: str = ""

        # Original conflicted file (never mutated after reset)
        self.original_file: str = ""

        # Ground truth (never exposed to agent)
        self.ground_truth_file: str = ""
        self.ground_truth_blocks: list = []

        # Track which blocks have been resolved
        # Key: block_id (int), Value: agent's resolution string
        self.resolutions: dict = {}

        # All conflict blocks parsed from original file
        # List of dicts: {id, head_content, incoming_content, full_marker_text}
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

        # Route to action handler
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

        # Apply step penalty for every action
        reward -= self.STEP_PENALTY
        self.total_reward += reward

        # Force termination if step limit reached
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

    # -------------------------------------------------------------------------
    # Action handlers
    # -------------------------------------------------------------------------

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

        # Check for conflict markers in the resolution itself — invalid
        if any(m in action.resolution for m in ["<<<<<<<", "=======", ">>>>>>>"]):
            return -0.10, (
                "Resolution contains git conflict markers. "
                "Your resolution must be clean code, not a conflict block."
            ), {"error": "resolution_contains_markers"}

        grader = ConflictGrader()

        # Get ground truth for this block
        if action.conflict_id < len(self.ground_truth_blocks):
            gt_block = self.ground_truth_blocks[action.conflict_id]
            block_score = grader.grade_block(action.resolution, gt_block)
        else:
            block_score = 0.0

        # Record resolution and rebuild current file
        self.resolutions[action.conflict_id] = action.resolution
        self.current_file = self._apply_resolutions()

        # Translate block score to step reward
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

        unresolved = [i for i in range(len(self.conflict_blocks)) if i not in self.resolutions]
        feedback = (
            f"Block {action.conflict_id} resolved. "
            f"Immediate quality: {quality} (block score: {block_score:.2f}). "
            f"Resolved: {len(self.resolutions)}/{len(self.conflict_blocks)}. "
            f"Unresolved blocks: {unresolved}."
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

        # Penalty for unresolved blocks
        unresolved_count = len(self.conflict_blocks) - len(self.resolutions)
        unresolved_penalty = 0.10 * unresolved_count

        # Bonus for submitting early if score is high
        steps_used = self.step_count
        max_steps = self.task["max_steps"]
        efficiency_bonus = 0.0
        if final_score >= 0.9 and steps_used <= (max_steps * 0.5):
            efficiency_bonus = 0.05

        terminal_reward = final_score - unresolved_penalty + efficiency_bonus

        feedback = (
            f"Episode complete. Final score: {final_score:.4f}. "
            f"Unresolved penalty: -{unresolved_penalty:.2f}. "
            f"Efficiency bonus: +{efficiency_bonus:.2f}. "
            f"Terminal reward: {terminal_reward:.4f}. "
            f"Score components: {components}."
        )

        return round(terminal_reward, 4), feedback, {
            "final_score": final_score,
            "components": components,
            "unresolved_penalty": unresolved_penalty,
            "efficiency_bonus": efficiency_bonus,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _parse_conflict_blocks(self, file_content: str) -> list:
        """
        Parse all conflict blocks from a conflicted file.
        Returns list of dicts with id, head_content, incoming_content.
        """
        blocks = []
        pattern = re.compile(
            r'<<<<<<< [^\n]+\n(.*?)=======\n(.*?)>>>>>>> [^\n]+\n',
            re.DOTALL
        )
        for i, match in enumerate(pattern.finditer(file_content)):
            blocks.append({
                "id": i,
                "head_content": match.group(1),
                "incoming_content": match.group(2),
                "full_marker_text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })
        return blocks

    def _apply_resolutions(self) -> str:
        """
        Rebuild the file with all current resolutions applied.
        Unresolved blocks remain as conflict markers.
        """
        result = self.original_file
        # Apply in reverse order to preserve character positions
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
            i for i in range(len(self.conflict_blocks))
            if i not in self.resolutions
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
```

---

## server/app.py — FastAPI Application

```python
import os
import json
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Optional

from models import (
    MergeAction, MergeObservation, StepResult,
    EpisodeState, TaskInfo, GraderResult, BaselineResult
)
from server.environment import GitMergeEnvironment
from server.grader import ConflictGrader
from server.tasks import ALL_TASKS, TASK_LIST


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

def get_env() -> GitMergeEnvironment:
    return app.state.env


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# OpenEnv required endpoints
# ---------------------------------------------------------------------------

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
        obs = env.reset(task_id=task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


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
            reward=reward,
            done=done,
            info=info,
        )
    except Exception as e:
        # Safety net — environment should handle all invalid inputs internally
        # but we catch here to guarantee HTTP 200 always
        return StepResult(
            observation=MergeObservation(
                file_name="unknown",
                total_conflicts=0,
                resolved_conflicts=0,
                unresolved_conflict_ids=[],
                current_file_preview="",
                last_action_feedback=f"Internal error processing action: {str(e)}",
                last_reward=-0.10,
                steps_remaining=0,
            ),
            reward=-0.10,
            done=False,
            info={"error": str(e)},
        )


@app.get("/state", response_model=EpisodeState, tags=["openenv"])
async def state(env: GitMergeEnvironment = Depends(get_env)):
    """Return current episode state metadata."""
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Required additional endpoints
# ---------------------------------------------------------------------------

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

    g = ConflictGrader()
    score, components = g.grade(env.current_file, env.task)

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


@app.post("/baseline", response_model=BaselineResult, tags=["openenv"])
async def baseline():
    """
    Run the baseline inference script against all 3 tasks and return scores.
    Uses OPENAI_API_KEY from environment variables.
    Produces reproducible scores.
    """
    try:
        from baseline import run_baseline
        scores = run_baseline()
        avg = sum(scores.values()) / len(scores)
        return BaselineResult(
            task_scores=scores,
            average_score=round(avg, 4),
            model_used=os.getenv("BASELINE_MODEL", "gpt-4o-mini"),
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="baseline.py not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline run failed: {str(e)}")


# ---------------------------------------------------------------------------
# Entry point for local development
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )
```

---

## baseline.py — Inference Script

This runs at the repository root, not inside server/.
It uses the OpenAI API client to run an LLM agent through all 3 tasks.
Reads OPENAI_API_KEY from environment. Produces reproducible scores.

```python
"""
baseline.py

Baseline inference script for GitMergeEnv.
Runs a GPT model as an agent against all 3 tasks.
Uses the OpenAI API client with the environment's HTTP API.

Usage:
    export OPENAI_API_KEY=your_key
    export BASE_URL=http://localhost:7860   # or your HF Space URL
    python baseline.py

Environment variables:
    OPENAI_API_KEY   — required
    BASE_URL         — environment URL (default: http://localhost:7860)
    BASELINE_MODEL   — model to use (default: gpt-4o-mini)
"""

import os
import json
import httpx
from openai import OpenAI

BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")
MODEL = os.getenv("BASELINE_MODEL", "gpt-4o-mini")
MAX_STEPS_OVERRIDE = 20  # safety cap


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


def run_task(client: OpenAI, task_id: str) -> float:
    """
    Run one complete episode for a given task.
    Returns the final grader score.
    """
    print(f"\n{'='*60}")
    print(f"Running task: {task_id}")
    print(f"{'='*60}")

    # Reset environment
    obs = call_env(f"/reset?task_id={task_id}")
    print(f"Task started. File: {obs['file_name']}, Conflicts: {obs['total_conflicts']}")

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
            )
        }
    ]

    final_score = 0.0
    steps_taken = 0

    for step_num in range(MAX_STEPS_OVERRIDE):
        # Get action from LLM
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,   # deterministic
            max_tokens=500,
        )

        raw_response = completion.choices[0].message.content.strip()

        # Parse action — handle markdown fences if present
        action_str = raw_response
        if "```" in action_str:
            action_str = action_str.split("```")[1]
            if action_str.startswith("json"):
                action_str = action_str[4:]
            action_str = action_str.strip()

        try:
            action = json.loads(action_str)
        except json.JSONDecodeError:
            print(f"Step {step_num}: LLM returned invalid JSON: {raw_response[:100]}")
            # Force inspect as recovery
            action = {"action_type": "inspect", "conflict_id": 0}

        print(f"Step {step_num}: {action.get('action_type', 'unknown')} "
              f"(block {action.get('conflict_id', '-')})")

        # Execute action
        result = call_env("/step", body=action)
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        steps_taken += 1

        print(f"  Reward: {reward:.4f} | Resolved: {obs['resolved_conflicts']}/{obs['total_conflicts']}")

        # Add assistant response and environment feedback to conversation
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
                    if obs['resolved_conflicts'] == obs['total_conflicts'] and not done
                    else "Continue resolving remaining conflicts or submit when ready."
                )
            )
        })

        if done:
            # Extract final score from info if available
            grader_result = call_env("/grader", method="POST")
            final_score = grader_result["score"]
            print(f"Episode done. Final grader score: {final_score:.4f}")
            break

    else:
        # Hit step override limit — get grader score anyway
        grader_result = call_env("/grader", method="POST")
        final_score = grader_result["score"]
        print(f"Step limit reached. Grader score: {final_score:.4f}")

    return final_score


def run_baseline() -> dict:
    """
    Run baseline agent against all 3 tasks.
    Returns dict mapping task_id to score.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        score = run_task(client, task_id)
        scores[task_id] = round(score, 4)

    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  Average: {avg:.4f}")

    return scores


if __name__ == "__main__":
    scores = run_baseline()
```

---

## server/Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker cache efficiency
COPY server/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py /app/models.py
COPY baseline.py /app/baseline.py
COPY server/ /app/server/

# Hugging Face Spaces runs as non-root user
RUN useradd -m -u 1000 user
USER user

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

---

## .env.example

```
OPENAI_API_KEY=your_openai_api_key_here
BASE_URL=http://localhost:7860
BASELINE_MODEL=gpt-4o-mini
PORT=7860
```

---

## README.md

Write a README with exactly these sections in order:

```markdown
# GitMergeEnv

> An OpenEnv environment for training AI agents to resolve git merge conflicts
> in Python source files.

## Overview

GitMergeEnv presents an agent with a Python file containing git merge conflict
markers. The agent inspects conflict blocks, proposes resolutions one by one,
and is scored deterministically against a ground truth resolution.

Built for the Meta PyTorch x OpenEnv Hackathon.

## Motivation

Git merge conflicts are one of the most universally painful experiences in
software development. Every engineer hits them. They require understanding
the intent behind two diverging changes and synthesizing a correct resolution
that preserves both. This is a genuine multi-step reasoning task with
deterministic ground truth — an ideal domain for RL agent training.

## Action Space

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| action_type | string | Yes | One of: inspect, resolve, submit |
| conflict_id | int | Conditional | Required for inspect and resolve |
| resolution | string | Conditional | Required for resolve |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| file_name | string | Name of the file being merged |
| total_conflicts | int | Total conflict blocks in file |
| resolved_conflicts | int | Blocks the agent has resolved |
| unresolved_conflict_ids | list[int] | Block IDs not yet resolved |
| current_file_preview | string | File with resolutions applied so far |
| last_action_feedback | string | Feedback on last action |
| last_reward | float | Reward from last action |
| steps_remaining | int | Steps before forced termination |

## Tasks

### Task 1 — Easy (max 6 steps)
Single conflict block. Developer A renamed a variable; Developer B added a
new parameter. Agent must synthesize both changes.
Expected baseline score: 0.75–0.95

### Task 2 — Medium (max 12 steps)
Three conflict blocks across imports, method body, and docstring.
Developer A added custom exceptions; Developer B added structured logging.
Agent must merge both sets of changes consistently.
Expected baseline score: 0.45–0.70

### Task 3 — Hard (max 18 steps)
Five architecturally dependent conflict blocks.
Developer A migrated to SQLAlchemy ORM; Developer B added features using
old raw sqlite3. Conflicts cannot be resolved independently — all five must
be resolved consistently with the ORM approach.
Expected baseline score: 0.20–0.45

## Reward Design

Rewards are dense and multi-component:
- inspect: +0.02 (encourage information gathering)
- resolve correct block: +0.15 exact, +0.08 good, +0.02 partial
- resolve wrong block: -0.02 to -0.08
- invalid action: -0.05 to -0.10
- step penalty: -0.01 per step (encourages efficiency)
- submit: full grader score minus unresolved block penalty plus efficiency bonus

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /reset | POST | Start new episode |
| /step | POST | Execute one action |
| /state | GET | Get episode state |
| /tasks | GET | List all tasks |
| /grader | POST | Score current state |
| /baseline | POST | Run baseline agent |
| /health | GET | Health check |

## Setup

### Local Development

```bash
git clone <repo-url>
cd git_merge_env
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -f server/Dockerfile -t git_merge_env .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key git_merge_env
```

### Baseline

```bash
export OPENAI_API_KEY=your_key
export BASE_URL=http://localhost:7860
python baseline.py
```

## Baseline Scores

| Task | Score | Model |
|------|-------|-------|
| task1 | ~0.82 | gpt-4o-mini |
| task2 | ~0.54 | gpt-4o-mini |
| task3 | ~0.31 | gpt-4o-mini |

## OpenEnv Compliance

Passes `openenv validate`. Deployed on Hugging Face Spaces.
Tagged with `openenv`.
```

---

## client.py

```python
"""
GitMergeEnv client for programmatic access.
"""
import httpx
from models import MergeAction, MergeObservation, StepResult, EpisodeState


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
```

---

## Critical Implementation Rules

The agent writing this code must follow these rules without exception:

**1. No LLM calls inside the environment.**
The grader, reward function, and all environment logic must be pure Python.
No calls to OpenAI, Anthropic, HuggingFace inference, or any external AI service
inside `environment.py`, `grader.py`, or any task file.
LLM calls exist only in `baseline.py`.

**2. Determinism is absolute.**
Same action on same state must always return same reward.
No `random`, no `time.time()`, no `uuid4()` inside step() or grader().
Episode IDs use uuid4 only in reset() for tracking, not for scoring.

**3. The environment never crashes on agent input.**
Every possible action — malformed, empty, wrong type, out of range —
must be handled gracefully inside step() and return a valid StepResult
with HTTP 200. The safety net in app.py is the last resort, not the primary handler.

**4. All scores are floats in [0.0, 1.0].**
Clamp with `min(max(score, 0.0), 1.0)` at every grader return point.
Never return negative scores from the grader.

**5. Ground truth is never exposed.**
The ground_truth_file and ground_truth_blocks fields from task dicts
must never appear in any API response. The observation shows current_file_preview
which is the agent's working version, never the ground truth.

**6. The Dockerfile must build with no network access after pip install.**
All dependencies are in requirements.txt. No runtime pip installs.
No wget or curl during container startup.

**7. The /baseline endpoint must not time out.**
If OPENAI_API_KEY is not set, return a 400 error immediately.
Do not attempt to call the OpenAI API without a key.

**8. Port 7860 is mandatory.**
Hugging Face Spaces runs on port 7860. The server must bind to 0.0.0.0:7860.
Do not hardcode any other port as the default.

---

## Validation Checklist

Before declaring the build complete, verify every item:

- [ ] `docker build -f server/Dockerfile -t git_merge_env .` exits with code 0
- [ ] `docker run -p 7860:7860 git_merge_env` starts without error
- [ ] `GET /health` returns HTTP 200
- [ ] `POST /reset?task_id=task1` returns valid MergeObservation
- [ ] `POST /reset?task_id=task2` returns valid MergeObservation
- [ ] `POST /reset?task_id=task3` returns valid MergeObservation
- [ ] `POST /step` with valid inspect action returns StepResult
- [ ] `POST /step` with valid resolve action returns StepResult
- [ ] `POST /step` with valid submit action returns StepResult with done=True
- [ ] `POST /step` with completely invalid JSON body returns HTTP 422 (FastAPI validation)
- [ ] `POST /step` with unknown action_type returns reward -0.10 and HTTP 200
- [ ] `GET /state` returns EpisodeState
- [ ] `GET /tasks` returns list of 3 TaskInfo objects
- [ ] `POST /grader` returns GraderResult with score in [0.0, 1.0]
- [ ] Grader returns 1.0 for perfect ground truth resolution of task1
- [ ] Grader returns 0.0 for empty string input
- [ ] Grader returns 0.0 for input that still contains conflict markers
- [ ] `openenv validate` passes (run after installing openenv-core)
- [ ] `python baseline.py` runs without error when OPENAI_API_KEY is set
- [ ] `POST /baseline` returns BaselineResult with scores for all 3 tasks

---

## What Success Looks Like

A judge runs this sequence and every step works:

```bash
docker build -f server/Dockerfile -t git_merge_env .
docker run -d -p 7860:7860 -e OPENAI_API_KEY=$KEY git_merge_env

curl http://localhost:7860/health
# {"status":"ok"}

curl -X POST "http://localhost:7860/reset?task_id=task1"
# {...valid MergeObservation...}

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"inspect","conflict_id":0}'
# {...valid StepResult with reward 0.01 (0.02 - 0.01 step penalty)...}

curl -X GET http://localhost:7860/tasks
# [...list of 3 tasks with action schema...]

curl -X POST http://localhost:7860/baseline
# {task_scores: {task1: 0.82, task2: 0.54, task3: 0.31}, average_score: 0.557}
```

That's the bar. Build to it.

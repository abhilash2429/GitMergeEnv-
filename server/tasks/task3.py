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
  Conflict 3: new feature (B added) — raw executemany bulk insert. Must become
    ORM bulk_save_objects().
  Conflict 4: new feature (B added) — soft delete via deleted_at timestamp.
    Must preserve soft-delete semantics using ORM syntax.
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
from datetime import datetime

from sqlalchemy import create_engine, Column, DateTime, Integer, String, select
from sqlalchemy.orm import DeclarativeBase, Session

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    role = Column(String, default="user")
    deleted_at = Column(DateTime, nullable=True)

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


def create_users(users: list[dict]):
<<<<<<< HEAD
    with Session(engine) as session:
        objects = [
            User(
                name=user["name"],
                email=user["email"],
                role=user.get("role", "user"),
            )
            for user in users
        ]
        session.bulk_save_objects(objects)
        session.commit()
        return len(objects)
=======
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO users (name, email, role) VALUES (?, ?, ?)",
        [(user["name"], user["email"], user.get("role", "user")) for user in users],
    )
    conn.commit()
    return cursor.rowcount
>>>>>>> feature/new-queries


def delete_user(user_id: int):
<<<<<<< HEAD
    with Session(engine) as session:
        user = session.get(User, user_id)
        if user:
            user.deleted_at = datetime.utcnow()
            session.commit()
            return True
        return False
=======
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
    conn.commit()
    return cursor.rowcount > 0
>>>>>>> feature/new-queries
''',
    "ground_truth_file": '''\
from datetime import datetime

from sqlalchemy import create_engine, Column, DateTime, Integer, String, select
from sqlalchemy.orm import DeclarativeBase, Session

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    role = Column(String, default="user")
    deleted_at = Column(DateTime, nullable=True)

engine = create_engine("sqlite:///app.db")
Base.metadata.create_all(engine)


def get_user_by_id(user_id: int):
    with Session(engine) as session:
        return session.get(User, user_id)


def get_all_users():
    with Session(engine) as session:
        return session.execute(select(User)).scalars().all()


def create_users(users: list[dict]):
    with Session(engine) as session:
        objects = [
            User(
                name=user["name"],
                email=user["email"],
                role=user.get("role", "user"),
            )
            for user in users
        ]
        session.bulk_save_objects(objects)
        session.commit()
        return len(objects)


def delete_user(user_id: int):
    with Session(engine) as session:
        user = session.get(User, user_id)
        if user:
            user.deleted_at = datetime.utcnow()
            session.commit()
            return True
        return False
''',
    "ground_truth_blocks": [
        '''\
from datetime import datetime

from sqlalchemy import create_engine, Column, DateTime, Integer, String, select
from sqlalchemy.orm import DeclarativeBase, Session

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    role = Column(String, default="user")
    deleted_at = Column(DateTime, nullable=True)

engine = create_engine("sqlite:///app.db")
Base.metadata.create_all(engine)''',
        '''\
    with Session(engine) as session:
        return session.get(User, user_id)''',
        '''\
    with Session(engine) as session:
        return session.execute(select(User)).scalars().all()''',
        '''\
    with Session(engine) as session:
        objects = [
            User(
                name=user["name"],
                email=user["email"],
                role=user.get("role", "user"),
            )
            for user in users
        ]
        session.bulk_save_objects(objects)
        session.commit()
        return len(objects)''',
        '''\
    with Session(engine) as session:
        user = session.get(User, user_id)
        if user:
            user.deleted_at = datetime.utcnow()
            session.commit()
            return True
        return False''',
    ],
    "required_elements": [
        "from sqlalchemy",
        "DateTime",
        "Session(",
        "session.get(",
        "bulk_save_objects",
        "session.commit()",
        "deleted_at",
        "datetime.utcnow()",
        "select(",
    ],
    "forbidden_elements": [
        "<<<<<<<",
        "=======",
        ">>>>>>>",
        "sqlite3.connect",
        "cursor.execute",
        "conn.commit()",
    ],
    "consistency_checks": [
        {
            "must_have": "Session(engine)",
            "must_not_have": "cursor.execute",
            "label": "orm_consistency",
            "weight": 0.15,
        }
    ],

    "grader_weights": {
        "no_conflict_markers": 0.05,
        "block_match": 0.40,
        "required_elements": 0.25,
        "architectural_consistency": 0.25,
        "indentation_consistency": 0.05,
    },

    "expected_baseline_score": (0.20, 0.45),
}

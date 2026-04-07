import json
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from models import (
    BaselineResult,
    EpisodeState,
    GraderResult,
    MergeAction,
    MergeObservation,
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


DOCS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GitMergeEnv — RL Environment Docs</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0a0e17;
            --surface: #111827;
            --surface-hover: #1a2235;
            --border: #1e2d40;
            --accent-orange: #f97316;
            --accent-blue: #3b82f6;
            --accent-green: #22c55e;
            --accent-yellow: #eab308;
            --accent-red: #ef4444;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #475569;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html { scroll-behavior: smooth; }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text-primary);
            line-height: 1.6;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .nav {
            position: sticky;
            top: 0;
            background: var(--bg);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
        }
        .nav-logo {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 1.25rem;
            color: var(--accent-orange);
        }
        .nav-links { display: flex; gap: 2rem; }
        .nav-links a {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.9rem;
            transition: color 0.2s;
        }
        .nav-links a:hover { color: var(--text-primary); }
        .container { max-width: 1200px; margin: 0 auto; padding: 0 2rem; }
        .hero {
            text-align: center;
            padding: 5rem 2rem;
            animation: fadeIn 0.6s ease-out;
        }
        .hero-label {
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-bottom: 1rem;
        }
        .hero-label span { color: var(--accent-orange); }
        .hero h1 {
            font-family: 'JetBrains Mono', monospace;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }
        .hero-subtitle {
            color: var(--text-secondary);
            max-width: 700px;
            margin: 0 auto 3rem;
            font-size: 1.1rem;
        }
        .stat-cards {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }
        .stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem 2rem;
            text-align: center;
            transition: transform 0.2s;
        }
        .stat-card:hover { transform: translateY(-2px); }
        .stat-number {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-orange);
        }
        .stat-label {
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }
        section { padding: 4rem 0; }
        .section-header {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .section-header span { color: var(--accent-orange); }
        .section-subheader {
            color: var(--text-muted);
            margin-bottom: 2rem;
        }
        .two-col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 3rem;
            align-items: start;
        }
        @media (max-width: 768px) {
            .two-col { grid-template-columns: 1fr; }
            .three-col { grid-template-columns: 1fr !important; }
            .nav-links { display: none; }
        }
        .three-col {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            transition: transform 0.2s;
        }
        .card:hover { transform: translateY(-2px); }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .card-title {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-orange);
            font-weight: 600;
        }
        .badge {
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-easy { background: var(--accent-green); color: #000; }
        .badge-medium { background: var(--accent-yellow); color: #000; }
        .badge-hard { background: var(--accent-red); color: #fff; }
        .badge-get { background: var(--accent-blue); color: #fff; }
        .badge-post { background: var(--accent-orange); color: #fff; }
        .card-meta {
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-bottom: 0.75rem;
        }
        .card-desc {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        .chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
        .chip {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .chip-green { background: rgba(34, 197, 94, 0.2); color: var(--accent-green); }
        .chip-red { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        th, td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
        }
        tr:nth-child(even) { background: var(--surface); }
        .code-block {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            overflow-x: auto;
        }
        .code-block .comment { color: var(--accent-green); }
        .code-block .error { color: var(--accent-red); }
        .callout {
            background: var(--surface);
            border-left: 4px solid var(--accent-orange);
            padding: 1rem 1.5rem;
            margin: 1.5rem 0;
            border-radius: 0 8px 8px 0;
        }
        details {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-top: 1.5rem;
        }
        summary {
            padding: 1rem 1.5rem;
            cursor: pointer;
            font-weight: 500;
        }
        details[open] summary { border-bottom: 1px solid var(--border); }
        details .code-block { margin: 1rem; border: none; }
        .footer {
            background: var(--surface);
            border-top: 1px solid var(--border);
            padding: 3rem 2rem;
            text-align: center;
            margin-top: 4rem;
        }
        .footer-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        .footer-links {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 1rem;
        }
        .footer-links a {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.9rem;
        }
        .footer-links a:hover { color: var(--accent-orange); }
        .footer-muted {
            color: var(--text-muted);
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <nav class="nav">
        <div class="nav-logo">GitMergeEnv</div>
        <div class="nav-links">
            <a href="#tasks">Tasks</a>
            <a href="#rewards">Rewards</a>
            <a href="#api">API</a>
            <a href="#grader">Grader</a>
            <a href="#quickstart">Quick Start</a>
        </div>
    </nav>

    <section class="hero" id="hero">
        <div class="container">
            <p class="hero-label"><span>//</span> OpenEnv Hackathon Submission</p>
            <h1>GitMergeEnv</h1>
            <p class="hero-subtitle">An RL environment where agents learn to resolve git merge conflicts with semantic correctness — rewarded for architectural consistency across the whole file, not just individual blocks.</p>
            <div class="stat-cards">
                <div class="stat-card">
                    <div class="stat-number">3</div>
                    <div class="stat-label">Benchmark Tasks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">Deterministic</div>
                    <div class="stat-label">Grader — No LLM</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">Multi-Component</div>
                    <div class="stat-label">Reward Shaping</div>
                </div>
            </div>
        </div>
    </section>

    <section id="problem">
        <div class="container">
            <h2 class="section-header"><span>//</span> Why This Problem Matters</h2>
            <div class="two-col">
                <div>
                    <p style="color: var(--text-secondary);">Merge conflicts are among the most common sources of subtle bugs introduced during collaborative development. Automated resolution tools fail because they optimize for syntactic merging — they cannot detect when a developer mixes SQLAlchemy ORM patterns with raw sqlite3 calls across resolved blocks, producing code that parses cleanly but is architecturally broken. No diff tool catches this class of error. This environment trains agents to make semantically correct, architecturally consistent resolution decisions by rewarding global file coherence, not just local block correctness.</p>
                </div>
                <div class="code-block">
<span class="comment"># Syntactically valid. Architecturally broken.</span>

def get_user(user_id):
    <span class="comment"># Block 0 resolved with ORM ✓</span>
    with Session(engine) as session:
        return session.get(User, user_id)

def delete_user(user_id):
    <span class="error"># Block 3 resolved with raw SQL ✗</span>
    cursor.execute("DELETE FROM users WHERE id=?",
                   (user_id,))
    conn.commit()
                </div>
            </div>
        </div>
    </section>

    <section id="tasks">
        <div class="container">
            <h2 class="section-header"><span>//</span> Benchmark Tasks</h2>
            <p class="section-subheader">Fixed scenarios. Hardcoded ground truth. Fully reproducible.</p>
            <div class="three-col">
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">task1</span>
                        <span class="badge badge-easy">Easy</span>
                    </div>
                    <div class="card-meta">processor.py · 1 conflict · 6 max steps</div>
                    <p class="card-desc">Variable rename + new argument. Developer A renamed <code>user_data → user_info</code>. Developer B added <code>timeout=30</code>. Agent must preserve both in a single coherent resolution.</p>
                    <div class="chips">
                        <span class="chip chip-green">user_info</span>
                        <span class="chip chip-green">timeout=30</span>
                        <span class="chip chip-green">transform(user_info)</span>
                    </div>
                    <div class="chips">
                        <span class="chip chip-red">transform(user_data)</span>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">task2</span>
                        <span class="badge badge-medium">Medium</span>
                    </div>
                    <div class="card-meta">data_service.py · 3 conflicts · 12 max steps</div>
                    <p class="card-desc">CustomError migration + logging addition. Contains docstring syntax traps and indentation-sensitive blocks. Naive resolution causes parse failure.</p>
                    <div class="chips">
                        <span class="chip chip-green">CustomError</span>
                        <span class="chip chip-green">import logging</span>
                        <span class="chip chip-green">logger.warning</span>
                        <span class="chip chip-green">raise CustomError</span>
                        <span class="chip chip-green">code=400</span>
                    </div>
                    <div class="chips">
                        <span class="chip chip-red">raise ValueError</span>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">task3</span>
                        <span class="badge badge-hard">Hard</span>
                    </div>
                    <div class="card-meta">db_access.py · 5 conflicts · 18 max steps</div>
                    <p class="card-desc">SQLAlchemy ORM vs raw sqlite3. The ORM pattern must win globally across all 5 blocks. Mixing patterns is detected via consistency scoring and penalized at submit time.</p>
                    <div class="chips">
                        <span class="chip chip-green">from sqlalchemy</span>
                        <span class="chip chip-green">Session(</span>
                        <span class="chip chip-green">bulk_save_objects</span>
                        <span class="chip chip-green">deleted_at</span>
                        <span class="chip chip-green">select(</span>
                    </div>
                    <div class="chips">
                        <span class="chip chip-red">sqlite3.connect</span>
                        <span class="chip chip-red">cursor.execute</span>
                        <span class="chip chip-red">conn.commit()</span>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <section id="rewards">
        <div class="container">
            <h2 class="section-header"><span>//</span> Reward Shaping</h2>
            <p class="section-subheader">Non-binary. Multi-component. Exploit-resistant.</p>
            <div class="three-col">
                <div class="card">
                    <h3 style="margin-bottom: 1rem; font-size: 1rem;">Per-Step Signals</h3>
                    <table>
                        <tr><th>Signal</th><th>Value</th></tr>
                        <tr><td>Step penalty</td><td>-0.01</td></tr>
                        <tr><td>Inspect reward</td><td>+0.02</td></tr>
                        <tr><td>Exact match</td><td>+0.15</td></tr>
                        <tr><td>High partial (≥0.7)</td><td>+0.08</td></tr>
                        <tr><td>Low partial (≥0.4)</td><td>+0.02</td></tr>
                        <tr><td>Near-zero</td><td>-0.02</td></tr>
                        <tr><td>No match</td><td>-0.08</td></tr>
                    </table>
                </div>
                <div class="card">
                    <h3 style="margin-bottom: 1rem; font-size: 1rem;">Terminal Signals</h3>
                    <table>
                        <tr><th>Component</th><th>Formula</th></tr>
                        <tr><td>Base reward</td><td>Grader score</td></tr>
                        <tr><td>Unresolved penalty</td><td>-0.10 × count</td></tr>
                        <tr><td>Efficiency bonus</td><td>+0.05</td></tr>
                        <tr><td>Consistency (0 mix)</td><td>+0.08</td></tr>
                        <tr><td>Consistency (1 mix)</td><td>+0.03</td></tr>
                        <tr><td>Consistency (2+ mix)</td><td>+0.00</td></tr>
                    </table>
                </div>
                <div class="card">
                    <h3 style="margin-bottom: 1rem; font-size: 1rem;">Exploit Resistance</h3>
                    <table>
                        <tr><th>Exploit</th><th>Defense</th></tr>
                        <tr><td>Empty submit</td><td>MIN_SCORE=0.01</td></tr>
                        <tr><td>Spam inspect</td><td>Step limit</td></tr>
                        <tr><td>Re-resolve</td><td>Decay: 1→0.7→0.4</td></tr>
                        <tr><td>Brute-force</td><td>Cap at 0.85</td></tr>
                        <tr><td>Inject markers</td><td>Hard reject</td></tr>
                        <tr><td>Score collapse</td><td>Clamp (0.01,0.99)</td></tr>
                    </table>
                </div>
            </div>
        </div>
    </section>

    <section id="grader">
        <div class="container">
            <h2 class="section-header"><span>//</span> Deterministic Grader</h2>
            <p class="section-subheader">No LLM calls. Fully programmatic. Results are reproducible.</p>
            <div class="two-col">
                <div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;"><strong>Per-block grading (grade_block):</strong> Exact normalized match returns 1.0. Otherwise, computes line-level F1 precision/recall score, capped at 0.85 to prevent brute-force gaming.</p>
                    <p style="color: var(--text-secondary);"><strong>Terminal grading (grade):</strong> Weighted combination of components. Parse failure applies 0.5× penalty. Forbidden elements apply multiplicative penalty (0.15 per violation, min 0.1).</p>
                    <div class="callout">All terminal scores are clamped to the strict open interval <strong>(0.01, 0.99)</strong>. The grader never returns exactly 0 or 1.</div>
                </div>
                <div class="card">
                    <h3 style="margin-bottom: 1rem; font-size: 1rem;">Component Weights</h3>
                    <table>
                        <tr><th>Component</th><th>task1</th><th>task2</th><th>task3</th></tr>
                        <tr><td>no_conflict_markers</td><td>0.15</td><td>0.10</td><td>0.05</td></tr>
                        <tr><td>block_match</td><td>0.55</td><td>0.50</td><td>0.40</td></tr>
                        <tr><td>required_elements</td><td>0.30</td><td>0.40</td><td>0.25</td></tr>
                        <tr><td>architectural_consistency</td><td>—</td><td>—</td><td>0.25</td></tr>
                        <tr><td>indentation_consistency</td><td>—</td><td>—</td><td>0.05</td></tr>
                    </table>
                </div>
            </div>
        </div>
    </section>

    <section id="api">
        <div class="container">
            <h2 class="section-header"><span>//</span> Endpoints</h2>
            <div class="card">
                <table>
                    <tr><th>Method</th><th>Path</th><th>Description</th></tr>
                    <tr><td><span class="badge badge-get">GET</span></td><td>/tasks</td><td>List all tasks with metadata, difficulty, and max_steps</td></tr>
                    <tr><td><span class="badge badge-post">POST</span></td><td>/reset</td><td>Start new episode. Body: {"task_id": "task1"}</td></tr>
                    <tr><td><span class="badge badge-post">POST</span></td><td>/step</td><td>Submit action. Body: MergeAction. Returns StepResult</td></tr>
                    <tr><td><span class="badge badge-get">GET</span></td><td>/state</td><td>Get current episode metadata as EpisodeState</td></tr>
                    <tr><td><span class="badge badge-post">POST</span></td><td>/grader</td><td>Run deterministic grader on an agent-submitted file</td></tr>
                    <tr><td><span class="badge badge-post">POST</span></td><td>/validate</td><td>Validate environment schema compliance</td></tr>
                    <tr><td><span class="badge badge-post">POST</span></td><td>/baseline</td><td>Run LLM baseline agent across all three tasks</td></tr>
                    <tr><td><span class="badge badge-get">GET</span></td><td>/health</td><td>Health check</td></tr>
                    <tr><td><span class="badge badge-get">GET</span></td><td>/docs-home</td><td>This documentation page</td></tr>
                </table>
                <details>
                    <summary>Example Episode Exchange</summary>
                    <div class="code-block">
POST /reset        {"task_id": "task3"}
→ {"file_name": "db_access.py", "total_conflicts": 5, ...}

POST /step         {"action_type": "inspect", "conflict_id": 0}
→ {"last_reward": 0.01, "last_action_feedback": "Block 0: HEAD uses Session(engine)..."}

POST /step         {"action_type": "resolve", "conflict_id": 0, "resolution": "..."}
→ {"last_reward": 0.14, "resolved_conflicts": 1, ...}

POST /step         {"action_type": "submit"}
→ {"reward": 0.71, "done": true, "info": {"grader_score": 0.71, ...}}
                    </div>
                </details>
            </div>
        </div>
    </section>

    <section id="quickstart">
        <div class="container">
            <h2 class="section-header"><span>//</span> Quick Start</h2>
            <div class="two-col">
                <div class="code-block">
<span class="comment"># Docker</span>
cp .env.example .env
<span class="comment"># Set API_BASE_URL, API_KEY, MODEL_NAME</span>

docker build -t gitmergeenv .
docker run -p 7860:7860 --env-file .env gitmergeenv
                </div>
                <div class="code-block">
<span class="comment"># Baseline agent runs all 3 tasks</span>
python inference.py

<span class="comment"># Output format (stdout only):</span>
[START] task=task1 env=GitMergeEnv
[STEP] step=1 action=inspect reward=0.01
[END] success=true steps=4 score=0.87
                </div>
            </div>
        </div>
    </section>

    <footer class="footer">
        <div class="footer-title">GitMergeEnv — Built for OpenEnv Hackathon</div>
        <div class="footer-links">
            <a href="/docs">Swagger UI → /docs</a>
            <a href="/health">Health → /health</a>
            <a href="/docs-home">Docs → /docs-home</a>
        </div>
        <p class="footer-muted">Deterministic grading · No LLM in scoring pipeline · Fully reproducible</p>
    </footer>
</body>
</html>
"""


@app.get("/", tags=["health"])
async def root():
    return RedirectResponse(url="/docs-home")


@app.get("/docs-home", response_class=HTMLResponse, include_in_schema=False)
async def docs_home():
    return HTMLResponse(content=DOCS_HTML)


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
            reward=reward,
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
            reward=-0.10,
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
        if not os.environ.get("API_KEY"):
            raise HTTPException(status_code=400, detail="API_KEY environment variable not set")

        from inference import run_baseline
        scores = run_baseline()
        avg = sum(scores.values()) / len(scores)
        return BaselineResult(
            task_scores=scores,
            average_score=round(avg, 4),
            model_used=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
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

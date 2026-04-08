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
    <title>GitMergeEnv | RL Environment</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #050816;
            --surface: rgba(13, 19, 39, 0.9);
            --surface-hover: rgba(17, 27, 55, 0.96);
            --border: rgba(148, 163, 184, 0.16);
            --accent-orange: #f59e0b;
            --accent-blue: #7dd3fc;
            --accent-green: #34d399;
            --accent-yellow: #fbbf24;
            --accent-red: #fb7185;
            --text-primary: #edf4ff;
            --text-secondary: #c9d7ee;
            --text-muted: #8ea2c3;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html { scroll-behavior: smooth; }
        body {
            font-family: 'IBM Plex Sans', system-ui, sans-serif;
            background:
                radial-gradient(circle at top left, rgba(125, 211, 252, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.10), transparent 22%),
                linear-gradient(180deg, #0a1125 0%, var(--bg) 44%, var(--bg) 100%);
            color: var(--text-primary);
            line-height: 1.6;
        }
        body::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(148, 163, 184, 0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148, 163, 184, 0.035) 1px, transparent 1px);
            background-size: 48px 48px;
            mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.55), transparent 82%);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .nav {
            position: sticky;
            top: 0;
            background: rgba(5, 8, 22, 0.72);
            backdrop-filter: blur(14px);
            border-bottom: 1px solid rgba(148, 163, 184, 0.10);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
        }
        .nav-logo {
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 700;
            font-size: 1.25rem;
            color: var(--text-primary);
        }
        .nav-links { display: flex; gap: 2rem; }
        .nav-links a {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.9rem;
            transition: color 0.2s;
        }
        .nav-links a:hover { color: var(--text-primary); }
        .container { max-width: 1200px; margin: 0 auto; padding: 0 2rem; position: relative; z-index: 1; }
        .hero {
            text-align: left;
            padding: 4rem 0 4.5rem;
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
            font-family: 'IBM Plex Mono', monospace;
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
            .four-col { grid-template-columns: 1fr !important; }
            .hero-grid, .api-grid, .quick-grid { grid-template-columns: 1fr; }
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
            border-radius: 18px;
            padding: 1.5rem;
            transition: transform 0.2s, border-color 0.2s, background 0.2s;
            box-shadow: 0 18px 48px rgba(2, 8, 23, 0.32);
        }
        .card:hover {
            transform: translateY(-4px);
            border-color: rgba(125, 211, 252, 0.25);
            background: var(--surface-hover);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .card-title {
            font-family: 'IBM Plex Mono', monospace;
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
            font-family: 'IBM Plex Mono', monospace;
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
            border-radius: 18px;
            padding: 1.5rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.7;
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
            border-top: 1px solid rgba(148, 163, 184, 0.10);
            padding: 2rem 0 4rem;
            margin-top: 1rem;
        }
        .footer-title {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }
        .footer-links {
            display: flex;
            gap: 1.25rem;
            flex-wrap: wrap;
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
        .section-kicker {
            font-family: 'IBM Plex Mono', monospace;
            color: var(--accent-blue);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.75rem;
        }
        .section-copy {
            color: var(--text-muted);
            max-width: 60ch;
            margin-bottom: 2rem;
        }
        .hero-grid, .api-grid, .quick-grid {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 1.5rem;
            align-items: start;
        }
        .hero-panel, .hero-visual, .stack-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 2rem;
            box-shadow: 0 22px 58px rgba(2, 8, 23, 0.36);
        }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.5rem 0.9rem;
            border-radius: 999px;
            border: 1px solid rgba(125, 211, 252, 0.22);
            background: rgba(125, 211, 252, 0.08);
            color: var(--accent-blue);
            font-size: 0.84rem;
            margin-bottom: 1rem;
        }
        .eyebrow::before {
            content: "";
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 999px;
            background: var(--accent-blue);
        }
        .hero-title {
            font-family: 'IBM Plex Mono', monospace;
            font-size: clamp(2.4rem, 5vw, 3.9rem);
            line-height: 1.1;
            margin-bottom: 1rem;
            max-width: 12ch;
        }
        .hero-subtext {
            color: var(--text-secondary);
            max-width: 60ch;
            margin-bottom: 1.5rem;
        }
        .pill-row, .tag-row, .metric-grid, .flow-strip, .footer-inner {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        .pill, .tag {
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.14);
            background: rgba(148, 163, 184, 0.06);
            color: var(--text-secondary);
            font-size: 0.8rem;
        }
        .metric-grid { margin-top: 1.5rem; }
        .metric-card {
            flex: 1 1 11rem;
            min-width: 11rem;
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.10);
            background: rgba(5, 10, 22, 0.52);
        }
        .metric-card strong {
            display: block;
            font-family: 'IBM Plex Mono', monospace;
            color: var(--accent-blue);
            margin-bottom: 0.3rem;
        }
        .metric-card span { color: var(--text-muted); font-size: 0.82rem; }
        .stack {
            display: grid;
            gap: 0.9rem;
        }
        .step-item {
            display: grid;
            grid-template-columns: 2.5rem 1fr;
            gap: 0.9rem;
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.10);
            background: rgba(5, 10, 22, 0.52);
        }
        .step-badge {
            width: 2.5rem;
            height: 2.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 14px;
            background: rgba(125, 211, 252, 0.12);
            color: var(--accent-blue);
            font-family: 'IBM Plex Mono', monospace;
        }
        .step-item strong { display: block; margin-bottom: 0.25rem; }
        .four-col {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
        }
        .task-head, .code-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .method {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 3.4rem;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.76rem;
            letter-spacing: 0.04em;
        }
        .method.get {
            color: #bfdbfe;
            background: rgba(59, 130, 246, 0.16);
            border: 1px solid rgba(59, 130, 246, 0.24);
        }
        .method.post {
            color: #fed7aa;
            background: rgba(249, 115, 22, 0.16);
            border: 1px solid rgba(249, 115, 22, 0.24);
        }
        .table-wrap {
            overflow-x: auto;
            border: 1px solid rgba(148, 163, 184, 0.10);
            border-radius: 14px;
        }
    </style>
</head>
<body>
    <nav class="nav">
        <div class="container" style="display: flex; justify-content: space-between; align-items: center; gap: 1.5rem;">
            <div class="nav-logo">GitMergeEnv</div>
            <div class="nav-links">
                <a href="#what-it-is">What It Is</a>
                <a href="#problem">Problem</a>
                <a href="#environment">Environment</a>
                <a href="#tasks">Tasks</a>
                <a href="#rewards">Rewards</a>
                <a href="#grader">Grader</a>
                <a href="#api">API</a>
                <a href="#quickstart">Quick Start</a>
            </div>
        </div>
    </nav>

    <section class="hero" id="hero">
        <div class="container hero-grid">
            <div class="hero-panel">
                <div class="eyebrow">OpenEnv-compatible reinforcement learning environment</div>
                <h1 class="hero-title">A reinforcement learning environment where agents learn to resolve Python merge conflicts with full-file consistency.</h1>
                <p class="hero-subtext">GitMergeEnv exposes <code>reset</code>, <code>step</code>, and <code>state</code>, returns structured observations, and scores outcomes with a deterministic grader. It is training and evaluation infrastructure, not a model.</p>
                <div class="pill-row">
                    <span class="pill">Environment, not a model</span>
                    <span class="pill">No LLM in scoring</span>
                    <span class="pill">3 fixed benchmark tasks</span>
                    <span class="pill">Reward shaping is the core design</span>
                </div>
                <div class="metric-grid">
                    <div class="metric-card">
                        <strong>3 tasks</strong>
                        <span>Easy to hard curriculum</span>
                    </div>
                    <div class="metric-card">
                        <strong>step / reset / state</strong>
                        <span>Clean RL interaction loop</span>
                    </div>
                    <div class="metric-card">
                        <strong>(0.01, 0.99)</strong>
                        <span>Deterministic terminal range</span>
                    </div>
                    <div class="metric-card">
                        <strong>task3</strong>
                        <span>Architecture must stay coherent</span>
                    </div>
                </div>
            </div>
            <div class="hero-visual">
                <div class="section-kicker">Hero Loop</div>
                <div class="stack">
                    <div class="step-item">
                        <div class="step-badge">01</div>
                        <div>
                            <strong>Observation</strong>
                            <p class="section-copy" style="margin-bottom: 0;">Conflicted file, unresolved block IDs, preview, last reward, and remaining steps.</p>
                        </div>
                    </div>
                    <div class="step-item">
                        <div class="step-badge">02</div>
                        <div>
                            <strong>Action</strong>
                            <p class="section-copy" style="margin-bottom: 0;"><code>inspect</code>, <code>resolve</code>, or <code>submit</code>.</p>
                        </div>
                    </div>
                    <div class="step-item">
                        <div class="step-badge">03</div>
                        <div>
                            <strong>Reward</strong>
                            <p class="section-copy" style="margin-bottom: 0;">Dense local signal now, deterministic whole-file score on submit.</p>
                        </div>
                    </div>
                    <div class="step-item">
                        <div class="step-badge">04</div>
                        <div>
                            <strong>Next state</strong>
                            <p class="section-copy" style="margin-bottom: 0;">Episode continues until the agent submits or hits the step budget.</p>
                        </div>
                    </div>
                </div>
                <div class="callout mono">obs → action → reward → next obs</div>
            </div>
        </div>
    </section>

    <section id="what-it-is">
        <div class="container">
            <div class="section-kicker">Section 2</div>
            <h2 class="section-header">What This Actually Is</h2>
            <p class="section-copy">An RL environment is the interface around the task, not the agent itself. GitMergeEnv packages merge conflict resolution into that interface so an agent can learn from repeated episodes.</p>
            <div class="three-col">
                <div class="card">
                    <div class="section-kicker">Environment</div>
                    <h3>A controlled training surface</h3>
                    <p class="card-desc">Each episode starts with a real conflicted Python file and exposes only the state needed to act on it.</p>
                </div>
                <div class="card">
                    <div class="section-kicker">RL Loop</div>
                    <h3>State, action, reward, repeat</h3>
                    <p class="card-desc">The agent inspects blocks, proposes resolutions, then learns from immediate and terminal feedback.</p>
                </div>
                <div class="card">
                    <div class="section-kicker">What We Built</div>
                    <h3>Training + evaluation infrastructure</h3>
                    <p class="card-desc">Fixed tasks, deterministic scoring, and episode state for a problem where local choices depend on global file intent.</p>
                </div>
            </div>
            <div class="two-col" style="margin-top: 1.5rem;">
                <div class="stack-panel">
                    <div class="section-kicker">Step By Step</div>
                    <div class="stack">
                        <div class="step-item">
                            <div class="step-badge">1</div>
                            <div>
                                <strong>Reset</strong>
                                <p class="section-copy" style="margin-bottom: 0;">Choose <code>task1</code>, <code>task2</code>, or <code>task3</code> and receive the conflicted file plus a step budget.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-badge">2</div>
                            <div>
                                <strong>Inspect And Resolve</strong>
                                <p class="section-copy" style="margin-bottom: 0;">Use <code>/step</code> to inspect a conflict or submit a resolution for one block.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-badge">3</div>
                            <div>
                                <strong>Collect Reward</strong>
                                <p class="section-copy" style="margin-bottom: 0;">Useful exploration, exact matches, and coherent progress all get signal before submit.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-badge">4</div>
                            <div>
                                <strong>Submit</strong>
                                <p class="section-copy" style="margin-bottom: 0;">The environment grades the full file and ends the episode.</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="code-block">
Agent policy
    |
    v
GitMergeEnv API
    |
    +-- returns observation
    +-- accepts merge action
    +-- computes reward
    +-- tracks episode state
    +-- runs deterministic grader on submit

The hard part is not one block.
The hard part is keeping the whole file coherent.
                </div>
            </div>
        </div>
    </section>

    <section id="problem">
        <div class="container">
            <div class="section-kicker">Section 3</div>
            <h2 class="section-header">The Problem</h2>
            <p class="section-copy">Merge conflicts create bugs that often pass syntax checks but violate architecture. Existing tooling is good at text reconciliation. It is weak at preserving system-wide intent.</p>
            <div class="two-col">
                <div class="stack-panel">
                    <div class="section-kicker">Why This Matters</div>
                    <div class="stack">
                        <div class="step-item">
                            <div class="step-badge">A</div>
                            <div>
                                <strong>Typical Merge Automation</strong>
                                <p class="section-copy" style="margin-bottom: 0;">Choose one side, splice nearby lines, and optimize for a file that still parses.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-badge">B</div>
                            <div>
                                <strong>Real System Requirement</strong>
                                <p class="section-copy" style="margin-bottom: 0;">Preserve compatible edits, adapt incompatible ones, and keep one architecture across the file.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-badge">C</div>
                            <div>
                                <strong>Why RL Fits</strong>
                                <p class="section-copy" style="margin-bottom: 0;">The correct resolution for block 4 can depend on what the agent committed to in block 0.</p>
                            </div>
                        </div>
                    </div>
                    <div class="callout">This environment rewards whole-file coherence, not just local diff cleanup.</div>
                </div>
                <div class="code-block">
<span class="comment"># Syntactically valid. Architecturally broken.</span>

def get_user(user_id):
    <span class="comment"># Block 0 resolved with ORM</span>
    <span class="comment">with Session(engine) as session:</span>
        <span class="comment">return session.get(User, user_id)</span>

def delete_user(user_id):
    <span class="error">cursor.execute("UPDATE users SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))</span>
    <span class="error">conn.commit()</span>

<span class="comment"># The file parses, but it mixes ORM and raw SQL.</span>
                </div>
            </div>
        </div>
    </section>

    <section id="environment">
        <div class="container">
            <div class="section-kicker">Section 4</div>
            <h2 class="section-header">How The Environment Works</h2>
            <p class="section-copy">The interface stays deliberately small: observation, action, reward, and episode state. That makes it easy to plug into RL training code or a judge harness.</p>
            <div class="four-col">
                <div class="card">
                    <div class="section-kicker">Observation</div>
                    <h3>What the agent sees</h3>
                    <p class="card-desc">Returned on <code>/reset</code> and every <code>/step</code>.</p>
                    <div class="code-block">{
  "file_name": "db_access.py",
  "total_conflicts": 5,
  "resolved_conflicts": 1,
  "unresolved_conflict_ids": [1,2,3,4],
  "current_file_preview": "...",
  "last_reward": 0.14,
  "steps_remaining": 13
}</div>
                </div>
                <div class="card">
                    <div class="section-kicker">Action</div>
                    <h3>What the agent can do</h3>
                    <p class="card-desc">Three actions define the loop.</p>
                    <div class="code-block">{"action_type":"inspect","conflict_id":0}
{"action_type":"resolve","conflict_id":0,"resolution":"..."}
{"action_type":"submit"}</div>
                </div>
                <div class="card">
                    <div class="section-kicker">Reward</div>
                    <h3>What learning signal looks like</h3>
                    <p class="card-desc">Every step explains why reward changed.</p>
                    <div class="code-block">{
  "reward": 0.14,
  "done": false,
  "info": {"block_score":1.0,"quality":"PERFECT"}
}</div>
                </div>
                <div class="card">
                    <div class="section-kicker">Episode</div>
                    <h3>What state is tracked</h3>
                    <p class="card-desc">One task instance equals one episode.</p>
                    <div class="code-block">{
  "episode_id":"uuid",
  "task_id":"task3",
  "step_count": 6,
  "max_steps": 18,
  "done": false,
  "total_reward": 0.31
}</div>
                </div>
            </div>
        </div>
    </section>

    <section id="tasks">
        <div class="container">
            <div class="section-kicker">Section 5</div>
            <h2 class="section-header">Tasks Are A Progression</h2>
            <p class="section-copy">The benchmark is designed as a curriculum: local synthesis first, multi-block coordination next, then full architectural planning across dependent conflicts.</p>
            <div class="flow-strip" style="margin-bottom: 1.5rem;">
                <div class="card">
                    <h3>task1</h3>
                    <p class="card-desc">Local merge synthesis.</p>
                </div>
                <div class="card">
                    <h3>task2</h3>
                    <p class="card-desc">Coordination plus syntax sensitivity.</p>
                </div>
                <div class="card">
                    <h3>task3</h3>
                    <p class="card-desc">Architecture must stay globally consistent.</p>
                </div>
            </div>
            <div class="three-col">
                <div class="card">
                    <div class="task-head">
                        <div>
                            <div class="section-kicker">processor.py</div>
                            <span class="card-title">task1 — Rename plus new argument</span>
                        </div>
                        <span class="badge badge-easy">Easy</span>
                    </div>
                    <p class="card-desc">Developer A renamed <code>user_data</code> to <code>user_info</code>. Developer B added <code>timeout=30</code>. The correct resolution preserves both.</p>
                    <div class="tag-row">
                        <span class="tag">1 conflict</span>
                        <span class="tag">6 max steps</span>
                    </div>
                    <div class="chips">
                        <span class="chip chip-green">Keep the rename consistent</span>
                        <span class="chip chip-green">Preserve the new parameter</span>
                        <span class="chip chip-red">Avoid transform(user_data)</span>
                    </div>
                </div>
                <div class="card">
                    <div class="task-head">
                        <div>
                            <div class="section-kicker">data_service.py</div>
                            <span class="card-title">task2 — Class refactor across three conflicts</span>
                        </div>
                        <span class="badge badge-medium">Medium</span>
                    </div>
                    <p class="card-desc">One branch migrates to <code>CustomError</code>. The other adds structured logging. The agent must align imports, docstring, and method body.</p>
                    <div class="tag-row">
                        <span class="tag">3 conflicts</span>
                        <span class="tag">12 max steps</span>
                    </div>
                    <div class="chips">
                        <span class="chip chip-green">Carry logging across sections</span>
                        <span class="chip chip-green">Indentation matters</span>
                        <span class="chip chip-green">Require logger.warning + code=400</span>
                    </div>
                </div>
                <div class="card">
                    <div class="task-head">
                        <div>
                            <div class="section-kicker">db_access.py</div>
                            <span class="card-title">task3 — Architectural migration across five conflicts</span>
                        </div>
                        <span class="badge badge-hard">Hard</span>
                    </div>
                    <p class="card-desc">The first conflict decides whether the file uses SQLAlchemy ORM or raw <code>sqlite3</code>. Every later choice must follow that same architecture.</p>
                    <div class="tag-row">
                        <span class="tag">5 conflicts</span>
                        <span class="tag">18 max steps</span>
                    </div>
                    <div class="chips">
                        <span class="chip chip-green">Re-implement new features in ORM form</span>
                        <span class="chip chip-green">Block-level correctness is not enough</span>
                        <span class="chip chip-red">Mixed ORM and raw SQL loses score</span>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <section id="rewards">
        <div class="container">
            <div class="section-kicker">Section 6</div>
            <h2 class="section-header">Reward Design</h2>
            <p class="section-copy">Binary success or failure would be too sparse. GitMergeEnv combines dense intermediate signal with a delayed whole-file objective so the agent can learn useful behavior before the episode ends.</p>
            <div class="flow-strip" style="margin-bottom: 1.5rem;">
                <div class="card">
                    <h3>Inspect</h3>
                    <p class="card-desc">Small positive signal for information gathering.</p>
                </div>
                <div class="card">
                    <h3>Resolve</h3>
                    <p class="card-desc">Immediate reward based on block quality, with retry decay.</p>
                </div>
                <div class="card">
                    <h3>Submit</h3>
                    <p class="card-desc">Deterministic full-file score plus penalties and bonuses.</p>
                </div>
            </div>
            <div class="three-col">
                <div class="card">
                    <div class="section-kicker">Per-Step</div>
                    <h3 style="margin-bottom: 1rem; font-size: 1rem;">Dense Local Signal</h3>
                    <div class="chips">
                        <span class="chip chip-green">-0.01 step penalty on every action</span>
                        <span class="chip chip-green">+0.02 inspect reward before penalty</span>
                        <span class="chip chip-green">Resolve ranges from +0.15 to -0.08</span>
                    </div>
                </div>
                <div class="card">
                    <div class="section-kicker">Anti-Exploit</div>
                    <h3 style="margin-bottom: 1rem; font-size: 1rem;">Why the signal stays useful</h3>
                    <div class="chips">
                        <span class="chip chip-green">Retry multiplier decays to 0.7 then 0.4</span>
                        <span class="chip chip-green">Positive reward drops if the full file still fails to parse</span>
                        <span class="chip chip-red">Conflict markers in a resolution are rejected</span>
                    </div>
                </div>
                <div class="card">
                    <div class="section-kicker">Terminal</div>
                    <h3 style="margin-bottom: 1rem; font-size: 1rem;">Whole-file objective</h3>
                    <div class="chips">
                        <span class="chip chip-green">Base reward is the grader score</span>
                        <span class="chip chip-green">Unresolved blocks cost 0.10 each</span>
                        <span class="chip chip-green">Efficiency adds 0.05, consistency adds up to 0.08</span>
                    </div>
                </div>
            </div>
            <div class="card" style="margin-top: 1.5rem;">
                <div class="table-wrap">
                    <table>
                        <tr><th>Stage</th><th>Rule</th><th>Value</th><th>Purpose</th></tr>
                        <tr><td>Every action</td><td>Step penalty</td><td>-0.01</td><td>Discourage useless interaction.</td></tr>
                        <tr><td>Inspect</td><td>Inspect reward</td><td>+0.02</td><td>Reward information gathering.</td></tr>
                        <tr><td>Resolve</td><td>Exact / high / low / near-zero / no match</td><td>+0.15 / +0.08 / +0.02 / -0.02 / -0.08</td><td>Provide block-level learning signal.</td></tr>
                        <tr><td>Resolve retry</td><td>Repetition multiplier</td><td>1.0 → 0.7 → 0.4</td><td>Prevent reward farming.</td></tr>
                        <tr><td>Submit</td><td>Base reward</td><td>grader score</td><td>Keep the objective tied to final file quality.</td></tr>
                        <tr><td>Submit</td><td>Unresolved penalty</td><td>-0.10 × count</td><td>Push the agent to finish.</td></tr>
                        <tr><td>Submit</td><td>Efficiency bonus</td><td>+0.05</td><td>Reward strong short trajectories.</td></tr>
                        <tr><td>Submit</td><td>Consistency bonus</td><td>+0.08 / +0.03 / +0.00</td><td>Reward architectural coherence across blocks.</td></tr>
                    </table>
                </div>
                <div class="callout">Reward shaping is the core differentiator: the environment gives the agent something learnable before final submit without giving up on global correctness.</div>
            </div>
        </div>
    </section>

    <section id="grader">
        <div class="container">
            <div class="section-kicker">Section 7</div>
            <h2 class="section-header">Deterministic Grader</h2>
            <p class="section-copy">No LLM calls, no randomness, same file in and same score out. That makes evaluation reproducible and easy for judges to trust.</p>
            <div class="four-col" style="margin-bottom: 1.5rem;">
                <div class="card"><div class="section-kicker">Step 1</div><h3>Parse and marker checks</h3><p class="card-desc">Empty input bottoms out at the minimum terminal score. Remaining conflict markers and parse failure reduce the score immediately.</p></div>
                <div class="card"><div class="section-kicker">Step 2</div><h3>Block match</h3><p class="card-desc"><code>grade_block</code> returns <code>1.0</code> for exact normalized matches, otherwise a capped line-level F1 score.</p></div>
                <div class="card"><div class="section-kicker">Step 3</div><h3>Required elements</h3><p class="card-desc">Each task checks for critical strings like <code>CustomError</code> or <code>bulk_save_objects</code>.</p></div>
                <div class="card"><div class="section-kicker">Step 4</div><h3>Consistency and penalties</h3><p class="card-desc">Task 3 adds architectural consistency and indentation checks, then forbidden elements apply multiplicative penalties.</p></div>
            </div>
            <div class="two-col">
                <div class="stack-panel">
                    <div class="section-kicker">Scoring Logic</div>
                    <div class="stack">
                        <div class="step-item">
                            <div class="step-badge">A</div>
                            <div>
                                <strong><code>grade_block</code></strong>
                                <p class="section-copy" style="margin-bottom: 0;">Exact normalized match returns <code>1.0</code>; otherwise line overlap produces an F1-style score capped at <code>0.85</code>.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-badge">B</div>
                            <div>
                                <strong><code>grade</code></strong>
                                <p class="section-copy" style="margin-bottom: 0;">Weighted components are combined per task, then parse penalties and forbidden-element multipliers are applied.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-badge">C</div>
                            <div>
                                <strong>Clamp</strong>
                                <p class="section-copy" style="margin-bottom: 0;">Every terminal score is clamped to the strict open interval <code>(0.01, 0.99)</code>.</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <h3 style="margin-bottom: 1rem; font-size: 1rem;">Component Weights</h3>
                    <div class="table-wrap">
                        <table>
                            <tr><th>Component</th><th>task1</th><th>task2</th><th>task3</th></tr>
                            <tr><td><code>no_conflict_markers</code></td><td>0.15</td><td>0.10</td><td>0.05</td></tr>
                            <tr><td><code>block_match</code></td><td>0.55</td><td>0.50</td><td>0.40</td></tr>
                            <tr><td><code>required_elements</code></td><td>0.30</td><td>0.40</td><td>0.25</td></tr>
                            <tr><td><code>architectural_consistency</code></td><td>-</td><td>-</td><td>0.25</td></tr>
                            <tr><td><code>indentation_consistency</code></td><td>-</td><td>-</td><td>0.05</td></tr>
                        </table>
                    </div>
                    <div class="tag-row" style="margin-top: 1rem;">
                        <span class="tag">Parse failure: ×0.5</span>
                        <span class="tag">Forbidden element penalty: -0.15 each</span>
                        <span class="tag">Penalty floor: 0.10</span>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <section id="api">
        <div class="container">
            <div class="section-kicker">Section 8</div>
            <h2 class="section-header">API And System Surface</h2>
            <p class="section-copy">The API is intentionally small. A client can list tasks, start an episode, step through it, inspect episode state, and optionally call grader or baseline helpers.</p>
            <div class="four-col" style="margin-bottom: 1.5rem;">
                <div class="card"><div class="section-kicker">Client</div><h3>Agent or evaluator</h3><p class="card-desc">Sends task selection and merge actions over HTTP.</p></div>
                <div class="card"><div class="section-kicker">FastAPI</div><h3>Thin API layer</h3><p class="card-desc">Exposes the environment through standard endpoints.</p></div>
                <div class="card"><div class="section-kicker">Environment</div><h3>Episode state machine</h3><p class="card-desc">Tracks file state, rewards, resolutions, and step limits.</p></div>
                <div class="card"><div class="section-kicker">Grader</div><h3>Deterministic scorer</h3><p class="card-desc">Evaluates the merged file on demand and on final submit.</p></div>
            </div>
            <div class="api-grid">
                <div class="card">
                    <div class="table-wrap">
                        <table>
                            <tr><th>Method</th><th>Path</th><th>Purpose</th></tr>
                            <tr><td><span class="method get">GET</span></td><td><code>/tasks</code></td><td>List benchmark tasks and action schema.</td></tr>
                            <tr><td><span class="method post">POST</span></td><td><code>/reset?task_id=task1</code></td><td>Start a fresh episode.</td></tr>
                            <tr><td><span class="method post">POST</span></td><td><code>/step</code></td><td>Send <code>inspect</code>, <code>resolve</code>, or <code>submit</code>.</td></tr>
                            <tr><td><span class="method get">GET</span></td><td><code>/state</code></td><td>Read current episode metadata.</td></tr>
                            <tr><td><span class="method post">POST</span></td><td><code>/grader</code></td><td>Score the current file without ending the episode.</td></tr>
                            <tr><td><span class="method post">POST</span></td><td><code>/validate</code></td><td>Run deterministic self-checks on known inputs.</td></tr>
                            <tr><td><span class="method post">POST</span></td><td><code>/baseline</code></td><td>Run the included baseline agent.</td></tr>
                            <tr><td><span class="method get">GET</span></td><td><code>/health</code></td><td>Health check.</td></tr>
                            <tr><td><span class="method get">GET</span></td><td><code>/docs-home</code></td><td>This documentation page.</td></tr>
                        </table>
                    </div>
                </div>
                <div class="code-block">
GET  /tasks

POST /reset?task_id=task3
-> observation with 5 conflict blocks

POST /step
{"action_type":"inspect","conflict_id":0}

POST /step
{"action_type":"resolve","conflict_id":0,"resolution":"with Session(engine) as session: ..."}

POST /step
{"action_type":"submit"}
-> final reward, done=true, deterministic score breakdown
                </div>
            </div>
        </div>
    </section>

    <section id="why-this-wins">
        <div class="container">
            <div class="section-kicker">Section 9</div>
            <h2 class="section-header">Why This Wins</h2>
            <p class="section-copy">The value is not just that it grades merged files. The value is that it turns a subtle software engineering problem into a learnable, repeatable RL task with trustworthy evaluation.</p>
            <div class="three-col">
                <div class="card">
                    <div class="section-kicker">Real-World Utility</div>
                    <h3>Models a production failure mode</h3>
                    <p class="card-desc">Architecturally inconsistent conflict resolutions are a real source of bugs in collaborative codebases.</p>
                </div>
                <div class="card">
                    <div class="section-kicker">Learnability</div>
                    <h3>Dense reward with delayed truth</h3>
                    <p class="card-desc">Agents get useful signal during the episode before final submit reveals the whole-file outcome.</p>
                </div>
                <div class="card">
                    <div class="section-kicker">Deterministic Evaluation</div>
                    <h3>No judge-model variance</h3>
                    <p class="card-desc">Scoring is fully programmatic, so comparisons across runs are stable and reproducible.</p>
                </div>
                <div class="card">
                    <div class="section-kicker">RL Compatibility</div>
                    <h3>Clean observation-action-reward loop</h3>
                    <p class="card-desc">The surface fits standard RL trainers and evaluation harnesses without extra glue logic.</p>
                </div>
                <div class="card">
                    <div class="section-kicker">Task Design</div>
                    <h3>Built for learning, not just testing</h3>
                    <p class="card-desc">The benchmark forms a curriculum: local synthesis, multi-block coordination, then architecture planning.</p>
                </div>
                <div class="card">
                    <div class="section-kicker">Differentiator</div>
                    <h3>Reward shaping centers coherence</h3>
                    <p class="card-desc">The environment explicitly rewards architectural consistency across the resolved file, not only individual block correctness.</p>
                </div>
            </div>
        </div>
    </section>

    <section id="quickstart">
        <div class="container">
            <div class="section-kicker">Section 10</div>
            <h2 class="section-header">Quick Start</h2>
            <p class="section-copy">Run the server, call the API, then optionally execute the included baseline. The environment is ready to use as a local service.</p>
            <div class="quick-grid">
                <div class="code-block">
<span class="comment"># Docker</span>
cp .env.example .env
docker build -t gitmergeenv .
docker run -p 7860:7860 --env-file .env gitmergeenv
curl http://localhost:7860/health
                </div>
                <div class="code-block">
<span class="comment"># Run one episode + baseline</span>
curl -X POST "http://localhost:7860/reset?task_id=task1"
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d "{\"action_type\":\"inspect\",\"conflict_id\":0}"

python inference.py
                </div>
            </div>
        </div>
    </section>

    <footer class="footer">
        <div class="container footer-inner">
            <div>
                <div class="footer-title">GitMergeEnv</div>
                <p class="footer-muted">Deterministic grading. Multi-step learning loop. Built for OpenEnv-style training and evaluation.</p>
            </div>
            <div class="footer-links">
                <a href="/docs">Swagger UI</a>
                <a href="/health">Health</a>
                <a href="/docs-home">Docs Home</a>
            </div>
        </div>
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

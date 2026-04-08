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


DOCS_HTML = """ <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GitMergeEnv — RL Environment for Merge Conflict Resolution</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --cream: #F9F6F0;
            --cream-dark: #F0EBE1;
            --ink: #1A1410;
            --ink-light: #4A4035;
            --ink-muted: #8A7F72;
            --forest: #1B4332;
            --forest-light: #2D6A4F;
            --forest-pale: #D8F3DC;
            --amber: #B45309;
            --amber-pale: #FEF3C7;
            --red-pale: #FEE2E2;
            --red: #991B1B;
            --border: rgba(26,20,16,0.10);
            --border-strong: rgba(26,20,16,0.18);
            --shadow: 0 2px 20px rgba(26,20,16,0.08);
            --shadow-lg: 0 8px 48px rgba(26,20,16,0.12);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        html {
            scroll-behavior: smooth;
        }

        body {
            font-family: 'DM Sans', sans-serif;
            background: var(--cream);
            color: var(--ink);
            line-height: 1.65;
            overflow-x: hidden;
        }

        /* Grain texture overlay */
        body::after {
            content: '';
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.025;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
            background-size: 200px;
            z-index: 9999;
        }

        /* ─── NAV ─── */
        nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 100;
            padding: 1rem 2.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(249,246,240,0.88);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            transition: all 0.3s ease;
        }

        .nav-logo {
            font-family: 'DM Mono', monospace;
            font-size: 0.95rem;
            font-weight: 500;
            color: var(--forest);
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .nav-logo::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--forest);
            animation: pulse 2.4s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.7); }
        }

        .nav-links {
            display: flex;
            align-items: center;
            gap: 2rem;
        }

        .nav-links a {
            color: var(--ink-muted);
            text-decoration: none;
            font-size: 0.875rem;
            font-weight: 500;
            transition: color 0.2s;
        }

        .nav-links a:hover { color: var(--forest); }

        .nav-badge {
            padding: 0.35rem 0.8rem;
            background: var(--forest);
            color: #fff;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        /* ─── CONTAINER ─── */
        .container {
            max-width: 1180px;
            margin: 0 auto;
            padding: 0 2.5rem;
        }

        /* ─── HERO ─── */
        .hero {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 8rem 0 6rem;
            position: relative;
            overflow: hidden;
        }

        .hero-bg {
            position: absolute;
            inset: 0;
            pointer-events: none;
        }

        .hero-bg-circle {
            position: absolute;
            border-radius: 50%;
            opacity: 0.06;
            background: var(--forest);
        }

        .hero-bg-circle:nth-child(1) {
            width: 600px; height: 600px;
            top: -200px; right: -100px;
            animation: float1 18s ease-in-out infinite;
        }

        .hero-bg-circle:nth-child(2) {
            width: 300px; height: 300px;
            bottom: 100px; left: -80px;
            opacity: 0.04;
            animation: float2 22s ease-in-out infinite;
        }

        @keyframes float1 {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(-30px, 20px) scale(1.05); }
            66% { transform: translate(20px, -15px) scale(0.97); }
        }

        @keyframes float2 {
            0%, 100% { transform: translate(0, 0); }
            50% { transform: translate(20px, -30px); }
        }

        .hero-inner {
            position: relative;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4rem;
            align-items: center;
        }

        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.4rem 0.9rem;
            border: 1px solid var(--border-strong);
            border-radius: 999px;
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            color: var(--ink-muted);
            margin-bottom: 1.5rem;
            background: rgba(255,255,255,0.6);
            animation: fadeUp 0.6s ease-out both;
        }

        .hero-eyebrow-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--forest);
        }

        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: clamp(2.8rem, 5vw, 4.2rem);
            font-weight: 900;
            line-height: 1.08;
            letter-spacing: -0.02em;
            margin-bottom: 1.25rem;
            animation: fadeUp 0.6s 0.1s ease-out both;
        }

        .hero-title em {
            font-style: italic;
            color: var(--forest);
        }

        .hero-sub {
            color: var(--ink-light);
            font-size: 1.05rem;
            max-width: 46ch;
            margin-bottom: 2rem;
            animation: fadeUp 0.6s 0.2s ease-out both;
        }

        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 2.5rem;
            animation: fadeUp 0.6s 0.3s ease-out both;
        }

        .pill {
            padding: 0.35rem 0.75rem;
            border: 1px solid var(--border-strong);
            border-radius: 999px;
            font-size: 0.8rem;
            color: var(--ink-light);
            background: rgba(255,255,255,0.5);
        }

        .hero-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            animation: fadeUp 0.6s 0.4s ease-out both;
        }

        .metric {
            padding: 1.1rem;
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 12px;
            box-shadow: var(--shadow);
        }

        .metric-val {
            font-family: 'DM Mono', monospace;
            font-size: 1.1rem;
            font-weight: 500;
            color: var(--forest);
            margin-bottom: 0.2rem;
        }

        .metric-label {
            font-size: 0.78rem;
            color: var(--ink-muted);
        }

        /* Hero right — loop visualization */
        .hero-loop {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: var(--shadow-lg);
            animation: fadeUp 0.7s 0.2s ease-out both;
        }

        .loop-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.72rem;
            color: var(--ink-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 1.5rem;
        }

        .loop-steps {
            display: flex;
            flex-direction: column;
            gap: 0;
        }

        .loop-step {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            padding: 1rem 0;
            border-bottom: 1px solid var(--border);
            opacity: 0;
            transform: translateX(12px);
        }

        .loop-step.visible {
            animation: slideIn 0.4s ease-out forwards;
        }

        .loop-step:last-child { border-bottom: none; }

        @keyframes slideIn {
            to { opacity: 1; transform: translateX(0); }
        }

        .loop-num {
            width: 28px;
            height: 28px;
            border-radius: 8px;
            background: var(--forest-pale);
            color: var(--forest);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            font-weight: 500;
            flex-shrink: 0;
        }

        .loop-step-title {
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.2rem;
        }

        .loop-step-desc {
            font-size: 0.82rem;
            color: var(--ink-muted);
        }

        .loop-arrow {
            text-align: center;
            padding: 0.5rem 0;
            color: var(--ink-muted);
            font-size: 0.75rem;
        }

        /* ─── SECTION BASE ─── */
        section {
            padding: 6rem 0;
            border-top: 1px solid var(--border);
        }

        .section-kicker {
            font-family: 'DM Mono', monospace;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--forest);
            margin-bottom: 0.75rem;
        }

        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: clamp(1.8rem, 3vw, 2.5rem);
            font-weight: 700;
            line-height: 1.15;
            letter-spacing: -0.02em;
            margin-bottom: 0.75rem;
        }

        .section-sub {
            color: var(--ink-light);
            max-width: 55ch;
            font-size: 1rem;
            margin-bottom: 3rem;
        }

        /* ─── CARDS ─── */
        .card {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: var(--shadow);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }

        .card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }

        .card-kicker {
            font-family: 'DM Mono', monospace;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--ink-muted);
            margin-bottom: 0.6rem;
        }

        .card-title {
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }

        .card-body {
            font-size: 0.875rem;
            color: var(--ink-light);
            line-height: 1.6;
        }

        /* ─── GRIDS ─── */
        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.25rem;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            align-items: start;
        }

        .grid-4 {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.25rem;
        }

        /* ─── TASKS ─── */
        .task-card {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: var(--shadow);
            transition: transform 0.25s, box-shadow 0.25s;
            position: relative;
            overflow: hidden;
        }

        .task-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
        }

        .task-card.easy::before { background: #10B981; }
        .task-card.medium::before { background: #F59E0B; }
        .task-card.hard::before { background: #EF4444; }

        .task-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
        }

        .task-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .task-badge.easy { background: #D1FAE5; color: #065F46; }
        .task-badge.medium { background: #FEF3C7; color: #92400E; }
        .task-badge.hard { background: #FEE2E2; color: #991B1B; }

        .task-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        }

        .task-file {
            font-family: 'DM Mono', monospace;
            font-size: 0.75rem;
            color: var(--ink-muted);
            margin-bottom: 0.4rem;
        }

        .task-name {
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 0.6rem;
        }

        .task-desc {
            font-size: 0.875rem;
            color: var(--ink-light);
            margin-bottom: 1.25rem;
        }

        .task-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .tag {
            padding: 0.25rem 0.6rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 0.75rem;
            color: var(--ink-muted);
            font-family: 'DM Mono', monospace;
        }

        .tag.green { background: var(--forest-pale); color: var(--forest); border-color: transparent; }
        .tag.red { background: var(--red-pale); color: var(--red); border-color: transparent; }
        .tag.amber { background: var(--amber-pale); color: var(--amber); border-color: transparent; }

        /* ─── CODE ─── */
        .code-block {
            background: #1A1410;
            color: #E8DCC8;
            border-radius: 12px;
            padding: 1.5rem;
            font-family: 'DM Mono', monospace;
            font-size: 0.82rem;
            line-height: 1.75;
            overflow-x: auto;
        }

        .code-block .comment { color: #6B7860; }
        .code-block .string { color: #A8C5A0; }
        .code-block .key { color: #82AACC; }
        .code-block .val { color: #F0B472; }
        .code-block .err { color: #E06C75; }
        .code-block .ok { color: #98C379; }

        /* ─── REWARD TABLE ─── */
        .reward-table-wrap {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }

        th {
            padding: 0.85rem 1.25rem;
            text-align: left;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--ink-muted);
            font-weight: 600;
            background: var(--cream-dark);
            border-bottom: 1px solid var(--border);
        }

        td {
            padding: 0.85rem 1.25rem;
            border-bottom: 1px solid var(--border);
            color: var(--ink-light);
            vertical-align: top;
        }

        tr:last-child td { border-bottom: none; }

        tr:hover td { background: var(--cream); }

        td code {
            font-family: 'DM Mono', monospace;
            font-size: 0.8rem;
            background: var(--cream-dark);
            padding: 0.1rem 0.35rem;
            border-radius: 4px;
            color: var(--forest);
        }

        /* ─── API TABLE ─── */
        .method-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.55rem;
            border-radius: 6px;
            font-family: 'DM Mono', monospace;
            font-size: 0.72rem;
            font-weight: 500;
        }

        .method-get { background: #DBEAFE; color: #1E40AF; }
        .method-post { background: #FEF3C7; color: #92400E; }

        /* ─── CALLOUT ─── */
        .callout {
            border-left: 3px solid var(--forest);
            background: var(--forest-pale);
            padding: 1rem 1.25rem;
            border-radius: 0 10px 10px 0;
            font-size: 0.9rem;
            color: var(--forest);
            margin: 1.5rem 0;
        }

        /* ─── STEP ITEMS ─── */
        .step-list {
            display: flex;
            flex-direction: column;
            gap: 0;
        }

        .step-item {
            display: flex;
            gap: 1rem;
            padding: 1.25rem 0;
            border-bottom: 1px solid var(--border);
        }

        .step-item:last-child { border-bottom: none; }

        .step-num {
            width: 32px;
            height: 32px;
            border-radius: 10px;
            background: var(--forest-pale);
            color: var(--forest);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'DM Mono', monospace;
            font-size: 0.8rem;
            font-weight: 500;
            flex-shrink: 0;
            margin-top: 0.1rem;
        }

        .step-content strong {
            display: block;
            font-weight: 600;
            margin-bottom: 0.2rem;
            font-size: 0.95rem;
        }

        .step-content p {
            font-size: 0.875rem;
            color: var(--ink-light);
        }

        /* ─── GRADER WEIGHTS ─── */
        .weight-bar-wrap {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .weight-row {
            display: grid;
            grid-template-columns: 14rem 1fr 3rem;
            gap: 0.75rem;
            align-items: center;
        }

        .weight-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            color: var(--ink-light);
        }

        .weight-bar-bg {
            height: 8px;
            background: var(--cream-dark);
            border-radius: 999px;
            overflow: hidden;
        }

        .weight-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: var(--forest);
            transform-origin: left;
            transform: scaleX(0);
            transition: transform 1s cubic-bezier(0.25,0.46,0.45,0.94);
        }

        .weight-bar-fill.animated {
            transform: scaleX(1);
        }

        .weight-pct {
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            color: var(--ink-muted);
            text-align: right;
        }

        /* ─── WHY THIS WINS ─── */
        .wins-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.25rem;
        }

        .win-card {
            padding: 1.75rem;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: #fff;
            box-shadow: var(--shadow);
            transition: transform 0.25s, box-shadow 0.25s;
        }

        .win-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }

        .win-icon {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            background: var(--forest-pale);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }

        .win-title {
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.4rem;
        }

        .win-body {
            font-size: 0.84rem;
            color: var(--ink-light);
            line-height: 1.6;
        }

        /* ─── QUICKSTART ─── */
        .qs-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        .qs-card {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }

        .qs-card-head {
            padding: 0.85rem 1.25rem;
            background: var(--cream-dark);
            border-bottom: 1px solid var(--border);
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            color: var(--ink-muted);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .qs-card-head::before {
            content: '';
            display: flex;
            gap: 4px;
        }

        .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 3px; }
        .dot-red { background: #EF4444; }
        .dot-yellow { background: #F59E0B; }
        .dot-green { background: #10B981; }

        /* ─── FOOTER ─── */
        footer {
            border-top: 1px solid var(--border);
            padding: 3rem 0;
            margin-top: 0;
        }

        .footer-inner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 2rem;
            flex-wrap: wrap;
        }

        .footer-logo {
            font-family: 'DM Mono', monospace;
            font-weight: 500;
            color: var(--forest);
            font-size: 0.95rem;
        }

        .footer-desc {
            font-size: 0.82rem;
            color: var(--ink-muted);
            margin-top: 0.25rem;
        }

        .footer-links {
            display: flex;
            gap: 1.5rem;
        }

        .footer-links a {
            color: var(--ink-muted);
            text-decoration: none;
            font-size: 0.875rem;
            transition: color 0.2s;
        }

        .footer-links a:hover { color: var(--forest); }

        /* ─── ANIMATIONS ─── */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(18px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .reveal {
            opacity: 0;
            transform: translateY(24px);
            transition: opacity 0.55s ease, transform 0.55s ease;
        }

        .reveal.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .reveal-delay-1 { transition-delay: 0.1s; }
        .reveal-delay-2 { transition-delay: 0.2s; }
        .reveal-delay-3 { transition-delay: 0.3s; }
        .reveal-delay-4 { transition-delay: 0.4s; }

        /* ─── RESPONSIVE ─── */
        @media (max-width: 900px) {
            .hero-inner, .grid-2, .grid-3, .grid-4, .wins-grid, .qs-grid {
                grid-template-columns: 1fr;
            }
            .hero-metrics { grid-template-columns: repeat(3, 1fr); }
            .nav-links { display: none; }
            .weight-row { grid-template-columns: 10rem 1fr 3rem; }
        }

        /* ─── CODE HEADER ROW ─── */
        .code-wrap {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .code-header {
            background: var(--cream-dark);
            border-bottom: 1px solid var(--border);
            padding: 0.6rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'DM Mono', monospace;
            font-size: 0.75rem;
            color: var(--ink-muted);
        }

        .code-wrap .code-block {
            border-radius: 0;
            border: none;
        }

        /* ─── COMPONENT WEIGHTS TABLE ─── */
        .weight-grid {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: var(--shadow);
        }

        .weight-grid-title {
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }

        /* horizontal rule */
        hr {
            border: none;
            border-top: 1px solid var(--border);
            margin: 0;
        }
    </style>
</head>
<body>

<!-- NAV -->
<nav>
    <div class="nav-logo">GitMergeEnv</div>
    <div class="nav-links">
        <a href="#tasks">Tasks</a>
        <a href="#rewards">Rewards</a>
        <a href="#grader">Grader</a>
        <a href="#api">API</a>
        <a href="#quickstart">Quick Start</a>
        <a href="/docs" class="nav-badge">Swagger UI</a>
    </div>
</nav>

<!-- HERO -->
<section class="hero" id="hero">
    <div class="hero-bg">
        <div class="hero-bg-circle"></div>
        <div class="hero-bg-circle"></div>
    </div>
    <div class="container">
        <div class="hero-inner">
            <div>
                <div class="hero-eyebrow">
                    <div class="hero-eyebrow-dot"></div>
                    OpenEnv-compatible RL Environment
                </div>
                <h1 class="hero-title">
                    Agents learn to resolve <em>merge conflicts</em> with full-file coherence.
                </h1>
                <p class="hero-sub">
                    GitMergeEnv is training and evaluation infrastructure — not a model.
                    It exposes <code>reset</code>, <code>step</code>, and <code>state</code>,
                    scores outcomes with a deterministic grader, and rewards architectural
                    consistency, not just local diff cleanup.
                </p>
                <div class="hero-pills">
                    <span class="pill">No LLM in grading</span>
                    <span class="pill">3 benchmark tasks</span>
                    <span class="pill">Dense reward shaping</span>
                    <span class="pill">Deterministic scoring</span>
                </div>
                <div class="hero-metrics">
                    <div class="metric">
                        <div class="metric-val">3 tasks</div>
                        <div class="metric-label">Easy → Hard curriculum</div>
                    </div>
                    <div class="metric">
                        <div class="metric-val">step / reset</div>
                        <div class="metric-label">Clean RL API loop</div>
                    </div>
                    <div class="metric">
                        <div class="metric-val">(0.01, 0.99)</div>
                        <div class="metric-label">Terminal score range</div>
                    </div>
                </div>
            </div>

            <div class="hero-loop">
                <div class="loop-label">RL Interaction Loop</div>
                <div class="loop-steps" id="loopSteps">
                    <div class="loop-step">
                        <div class="loop-num">01</div>
                        <div>
                            <div class="loop-step-title">Observation</div>
                            <div class="loop-step-desc">Conflicted file, unresolved block IDs, current preview, last reward, steps remaining.</div>
                        </div>
                    </div>
                    <div class="loop-step">
                        <div class="loop-num">02</div>
                        <div>
                            <div class="loop-step-title">Action</div>
                            <div class="loop-step-desc"><code>inspect</code>, <code>resolve</code>, or <code>submit</code> — three actions define the loop.</div>
                        </div>
                    </div>
                    <div class="loop-step">
                        <div class="loop-num">03</div>
                        <div>
                            <div class="loop-step-title">Dense Reward</div>
                            <div class="loop-step-desc">Block-level feedback now. Whole-file deterministic score on submit.</div>
                        </div>
                    </div>
                    <div class="loop-step">
                        <div class="loop-num">04</div>
                        <div>
                            <div class="loop-step-title">Next State</div>
                            <div class="loop-step-desc">Episode continues until submit or step budget is exhausted.</div>
                        </div>
                    </div>
                </div>
                <div class="callout" style="margin-top: 1.25rem; margin-bottom: 0; font-family: 'DM Mono', monospace; font-size: 0.82rem;">
                    obs → action → reward → next obs
                </div>
            </div>
        </div>
    </div>
</section>

<!-- PROBLEM -->
<section id="problem">
    <div class="container">
        <div class="section-kicker">The Problem</div>
        <h2 class="section-title">Why this task is genuinely hard.</h2>
        <p class="section-sub">
            Merge conflicts create bugs that pass syntax checks but violate architecture.
            Existing tooling optimizes for text reconciliation. It's weak at preserving system-wide intent.
        </p>
        <div class="grid-2 reveal">
            <div>
                <div class="step-list">
                    <div class="step-item">
                        <div class="step-num">A</div>
                        <div class="step-content">
                            <strong>Typical merge automation</strong>
                            <p>Choose one side, splice nearby lines, and optimize for a file that still parses.</p>
                        </div>
                    </div>
                    <div class="step-item">
                        <div class="step-num">B</div>
                        <div class="step-content">
                            <strong>Real system requirement</strong>
                            <p>Preserve compatible edits, adapt incompatible ones, and keep one architecture across the whole file.</p>
                        </div>
                    </div>
                    <div class="step-item">
                        <div class="step-num">C</div>
                        <div class="step-content">
                            <strong>Why RL fits</strong>
                            <p>The correct resolution for block 4 can depend on the architectural choice committed in block 0. That's a multi-step dependency — exactly what RL trains for.</p>
                        </div>
                    </div>
                </div>
                <div class="callout">
                    This environment rewards whole-file coherence, not just local diff cleanup.
                </div>
            </div>
            <div class="code-wrap reveal reveal-delay-2">
                <div class="code-header">
                    <span class="dot dot-red"></span>
                    <span class="dot dot-yellow"></span>
                    <span class="dot dot-green"></span>
                    db_access.py — syntactically valid, architecturally broken
                </div>
                <div class="code-block">
<span class="comment"># Block 0 resolved with ORM ✓</span>
<span class="ok">def get_user(user_id):</span>
<span class="ok">    with Session(engine) as session:</span>
<span class="ok">        return session.get(User, user_id)</span>

<span class="comment"># Block 3 mixed with raw SQL ✗</span>
<span class="err">def delete_user(user_id):</span>
<span class="err">    cursor.execute(</span>
<span class="err">        "DELETE FROM users WHERE id = ?",</span>
<span class="err">        (user_id,)</span>
<span class="err">    )</span>
<span class="err">    conn.commit()</span>

<span class="comment"># File parses. Logic is broken.</span>
<span class="comment"># ORM and raw SQL can't coexist here.</span></div>
            </div>
        </div>
    </div>
</section>

<!-- TASKS -->
<section id="tasks">
    <div class="container">
        <div class="section-kicker">Benchmark Tasks</div>
        <h2 class="section-title reveal">A curriculum from local to architectural.</h2>
        <p class="section-sub reveal">
            Three fixed scenarios with increasing complexity.
            Local synthesis first, multi-block coordination next, then full architectural planning across dependent conflicts.
        </p>
        <div class="grid-3">
            <div class="task-card easy reveal">
                <div class="task-head">
                    <div>
                        <div class="task-file">processor.py</div>
                        <div class="task-name">Rename + new argument</div>
                    </div>
                    <span class="task-badge easy">Easy</span>
                </div>
                <p class="task-desc">
                    Developer A renamed <code>user_data</code> to <code>user_info</code> throughout.
                    Developer B added <code>timeout=30</code>. The correct resolution preserves both changes.
                </p>
                <div class="task-tags">
                    <span class="tag">1 conflict</span>
                    <span class="tag">6 max steps</span>
                    <span class="tag green">Rename consistent</span>
                    <span class="tag green">New param preserved</span>
                    <span class="tag red">Avoid transform(user_data)</span>
                </div>
            </div>
            <div class="task-card medium reveal reveal-delay-1">
                <div class="task-head">
                    <div>
                        <div class="task-file">data_service.py</div>
                        <div class="task-name">Class refactor — 3 conflicts</div>
                    </div>
                    <span class="task-badge medium">Medium</span>
                </div>
                <p class="task-desc">
                    One branch migrates to <code>CustomError</code>. The other adds structured logging.
                    All three conflict blocks — imports, docstring, method body — must align.
                </p>
                <div class="task-tags">
                    <span class="tag">3 conflicts</span>
                    <span class="tag">12 max steps</span>
                    <span class="tag green">Carry logging</span>
                    <span class="tag green">CustomError + code=400</span>
                    <span class="tag red">Avoid ValueError</span>
                </div>
            </div>
            <div class="task-card hard reveal reveal-delay-2">
                <div class="task-head">
                    <div>
                        <div class="task-file">db_access.py</div>
                        <div class="task-name">Architectural migration — 5 conflicts</div>
                    </div>
                    <span class="task-badge hard">Hard</span>
                </div>
                <p class="task-desc">
                    Conflict 0 decides whether the file uses SQLAlchemy ORM or raw sqlite3.
                    Every subsequent conflict must follow that architectural choice — they're not independent.
                </p>
                <div class="task-tags">
                    <span class="tag">5 conflicts</span>
                    <span class="tag">18 max steps</span>
                    <span class="tag green">ORM must win</span>
                    <span class="tag red">Mixed approach penalized</span>
                    <span class="tag red">Avoid cursor.execute</span>
                </div>
            </div>
        </div>
    </div>
</section>

<!-- REWARDS -->
<section id="rewards">
    <div class="container">
        <div class="section-kicker">Reward Design</div>
        <h2 class="section-title reveal">Dense signal. Delayed truth.</h2>
        <p class="section-sub reveal">
            Binary success/failure would be too sparse. The environment combines immediate block-level feedback with a
            whole-file objective so the agent learns useful behavior before the episode ends.
        </p>
        <div class="grid-3 reveal" style="margin-bottom: 2rem;">
            <div class="card">
                <div class="card-kicker">inspect</div>
                <div class="card-title">+0.02 before penalty</div>
                <p class="card-body">Small positive reward for information gathering. Encourages agents to look before they resolve.</p>
            </div>
            <div class="card">
                <div class="card-kicker">resolve</div>
                <div class="card-title">+0.15 to −0.08</div>
                <p class="card-body">Immediate block quality signal. Retries decay: 1st attempt full, 2nd 70%, 3rd+ 40%. Prevents reward farming.</p>
            </div>
            <div class="card">
                <div class="card-kicker">submit</div>
                <div class="card-title">grader score + bonuses</div>
                <p class="card-body">Terminal reward is the full grader score minus unresolved penalties, plus efficiency and consistency bonuses.</p>
            </div>
        </div>
        <div class="reward-table-wrap reveal">
            <table>
                <thead>
                    <tr>
                        <th>Stage</th>
                        <th>Rule</th>
                        <th>Value</th>
                        <th>Purpose</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Every action</td><td>Step penalty</td><td><code>−0.01</code></td><td>Discourage useless interaction</td></tr>
                    <tr><td>Inspect</td><td>Inspect reward</td><td><code>+0.02</code></td><td>Reward information gathering</td></tr>
                    <tr><td>Resolve (exact)</td><td>Perfect block match</td><td><code>+0.15</code></td><td>Strong signal for correct resolutions</td></tr>
                    <tr><td>Resolve (high)</td><td>Near-correct block</td><td><code>+0.08</code></td><td>Partial credit for close attempts</td></tr>
                    <tr><td>Resolve retry</td><td>Repetition multiplier</td><td><code>0.7 → 0.4</code></td><td>Prevent reward farming</td></tr>
                    <tr><td>Resolve (bad)</td><td>Wrong resolution</td><td><code>−0.08</code></td><td>Penalize incorrect resolutions</td></tr>
                    <tr><td>Submit</td><td>Base reward</td><td>grader score</td><td>Tie objective to final file quality</td></tr>
                    <tr><td>Submit</td><td>Unresolved penalty</td><td><code>−0.10 × count</code></td><td>Push the agent to finish</td></tr>
                    <tr><td>Submit</td><td>Efficiency bonus</td><td><code>+0.05</code></td><td>Reward strong short trajectories</td></tr>
                    <tr><td>Submit</td><td>Consistency bonus</td><td><code>+0.08 / +0.03 / 0</code></td><td>Reward architectural coherence</td></tr>
                </tbody>
            </table>
        </div>
    </div>
</section>

<!-- GRADER -->
<section id="grader">
    <div class="container">
        <div class="section-kicker">Deterministic Grader</div>
        <h2 class="section-title reveal">No LLM calls. No randomness. Same input, same score.</h2>
        <p class="section-sub reveal">
            Scoring is fully programmatic, making comparisons across runs stable and reproducible.
            A judge can trust that a 0.72 means the same thing every time.
        </p>
        <div class="grid-2 reveal">
            <div>
                <div class="step-list" style="margin-bottom: 2rem;">
                    <div class="step-item">
                        <div class="step-num">1</div>
                        <div class="step-content">
                            <strong>Parse and marker checks</strong>
                            <p>Empty input floors at minimum terminal score. Remaining conflict markers and parse failure reduce the score via a 0.5× multiplier — not a hard zero.</p>
                        </div>
                    </div>
                    <div class="step-item">
                        <div class="step-num">2</div>
                        <div class="step-content">
                            <strong>Block match via line-level F1</strong>
                            <p><code>grade_block</code> returns 1.0 for exact normalized matches. Otherwise line-overlap F1 capped at 0.85 — partial credit without gaming.</p>
                        </div>
                    </div>
                    <div class="step-item">
                        <div class="step-num">3</div>
                        <div class="step-content">
                            <strong>Required element checks</strong>
                            <p>Each task checks for critical strings: <code>CustomError</code>, <code>bulk_save_objects</code>, <code>Session(engine)</code>, and others.</p>
                        </div>
                    </div>
                    <div class="step-item">
                        <div class="step-num">4</div>
                        <div class="step-content">
                            <strong>Architectural consistency (task3)</strong>
                            <p>Checks that ORM and raw SQL patterns don't coexist. Each violation is a multiplicative penalty (×0.85 per violation, floor 0.25).</p>
                        </div>
                    </div>
                </div>
                <div class="callout">
                    Grader output is clamped to (0.01, 0.99) — a 0.0 from the grader means it's broken, not hard.
                </div>
            </div>
            <div class="weight-grid reveal reveal-delay-2">
                <div class="weight-grid-title">Component Weights by Task</div>
                <div style="margin-bottom: 1.5rem;">
                    <div style="font-size: 0.75rem; color: var(--ink-muted); font-family: 'DM Mono', monospace; margin-bottom: 0.75rem;">task1 — Easy</div>
                    <div class="weight-bar-wrap" id="weights1">
                        <div class="weight-row">
                            <div class="weight-label">no_conflict_markers</div>
                            <div class="weight-bar-bg"><div class="weight-bar-fill" style="--w: 0.15"></div></div>
                            <div class="weight-pct">15%</div>
                        </div>
                        <div class="weight-row">
                            <div class="weight-label">block_match</div>
                            <div class="weight-bar-bg"><div class="weight-bar-fill" style="--w: 0.55"></div></div>
                            <div class="weight-pct">55%</div>
                        </div>
                        <div class="weight-row">
                            <div class="weight-label">required_elements</div>
                            <div class="weight-bar-bg"><div class="weight-bar-fill" style="--w: 0.30"></div></div>
                            <div class="weight-pct">30%</div>
                        </div>
                    </div>
                </div>
                <hr style="margin: 1rem 0;">
                <div style="margin-bottom: 1.5rem;">
                    <div style="font-size: 0.75rem; color: var(--ink-muted); font-family: 'DM Mono', monospace; margin-bottom: 0.75rem;">task3 — Hard</div>
                    <div class="weight-bar-wrap" id="weights3">
                        <div class="weight-row">
                            <div class="weight-label">block_match</div>
                            <div class="weight-bar-bg"><div class="weight-bar-fill" style="--w: 0.40"></div></div>
                            <div class="weight-pct">40%</div>
                        </div>
                        <div class="weight-row">
                            <div class="weight-label">required_elements</div>
                            <div class="weight-bar-bg"><div class="weight-bar-fill" style="--w: 0.25"></div></div>
                            <div class="weight-pct">25%</div>
                        </div>
                        <div class="weight-row">
                            <div class="weight-label">architectural_consistency</div>
                            <div class="weight-bar-bg"><div class="weight-bar-fill" style="--w: 0.25"></div></div>
                            <div class="weight-pct">25%</div>
                        </div>
                        <div class="weight-row">
                            <div class="weight-label">indentation_consistency</div>
                            <div class="weight-bar-bg"><div class="weight-bar-fill" style="--w: 0.05"></div></div>
                            <div class="weight-pct">5%</div>
                        </div>
                    </div>
                </div>
                <hr style="margin: 1rem 0;">
                <div class="task-tags" style="margin-top: 1rem;">
                    <span class="tag">Parse failure: ×0.5</span>
                    <span class="tag">Forbidden element: −0.15</span>
                    <span class="tag">Penalty floor: 0.25</span>
                </div>
            </div>
        </div>
    </div>
</section>

<!-- API -->
<section id="api">
    <div class="container">
        <div class="section-kicker">API Surface</div>
        <h2 class="section-title reveal">Small, intentional, complete.</h2>
        <p class="section-sub reveal">
            A client can list tasks, start an episode, step through it, inspect state,
            and call grader or baseline helpers. Nothing more.
        </p>
        <div class="grid-2 reveal">
            <div class="reward-table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Method</th>
                            <th>Endpoint</th>
                            <th>Purpose</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td><span class="method-badge method-get">GET</span></td><td><code>/tasks</code></td><td>List tasks and action schema</td></tr>
                        <tr><td><span class="method-badge method-post">POST</span></td><td><code>/reset?task_id=</code></td><td>Start a fresh episode</td></tr>
                        <tr><td><span class="method-badge method-post">POST</span></td><td><code>/step</code></td><td>inspect / resolve / submit</td></tr>
                        <tr><td><span class="method-badge method-get">GET</span></td><td><code>/state</code></td><td>Read episode metadata</td></tr>
                        <tr><td><span class="method-badge method-post">POST</span></td><td><code>/grader</code></td><td>Score file without ending episode</td></tr>
                        <tr><td><span class="method-badge method-post">POST</span></td><td><code>/validate</code></td><td>Run deterministic self-checks</td></tr>
                        <tr><td><span class="method-badge method-post">POST</span></td><td><code>/baseline</code></td><td>Run included baseline agent</td></tr>
                        <tr><td><span class="method-badge method-get">GET</span></td><td><code>/health</code></td><td>Health check</td></tr>
                    </tbody>
                </table>
            </div>
            <div class="code-wrap reveal reveal-delay-2">
                <div class="code-header">
                    <span class="dot dot-red"></span>
                    <span class="dot dot-yellow"></span>
                    <span class="dot dot-green"></span>
                    Example episode — task3
                </div>
                <div class="code-block">
<span class="comment"># 1. Start episode</span>
<span class="key">POST</span> /reset?task_id=task3

<span class="comment"># 2. Inspect conflict 0 (establishes architecture)</span>
<span class="key">POST</span> /step
<span class="val">{"action_type":"inspect","conflict_id":0}</span>

<span class="comment"># 3. Resolve with ORM — commits the architecture</span>
<span class="key">POST</span> /step
<span class="val">{"action_type":"resolve","conflict_id":0,</span>
<span class="val"> "resolution":"from sqlalchemy ..."}</span>

<span class="comment"># 4. Resolve remaining 4 blocks consistently</span>
<span class="comment"># ...</span>

<span class="comment"># 5. Submit — grader scores full file</span>
<span class="key">POST</span> /step
<span class="val">{"action_type":"submit"}</span>
<span class="comment"># → final_score, done=true, components breakdown</span></div>
            </div>
        </div>
    </div>
</section>

<!-- WHY THIS WINS -->
<section id="why">
    <div class="container">
        <div class="section-kicker">Why This Wins</div>
        <h2 class="section-title reveal">The value is in what can actually be learned.</h2>
        <p class="section-sub reveal">
            Not just that it grades merged files — but that it turns a subtle software engineering
            failure mode into a repeatable RL task with trustworthy evaluation.
        </p>
        <div class="wins-grid">
            <div class="win-card reveal">
                <div class="win-icon">🎯</div>
                <div class="win-title">Models a real production failure mode</div>
                <div class="win-body">Architecturally inconsistent conflict resolutions are a real source of bugs in collaborative codebases. This environment targets that exact failure.</div>
            </div>
            <div class="win-card reveal reveal-delay-1">
                <div class="win-icon">📈</div>
                <div class="win-title">Dense reward before delayed truth</div>
                <div class="win-body">Agents receive useful signal during the episode before final submit reveals the whole-file outcome. That's what makes it trainable.</div>
            </div>
            <div class="win-card reveal reveal-delay-2">
                <div class="win-icon">🔒</div>
                <div class="win-title">No judge-model variance</div>
                <div class="win-body">Scoring is fully programmatic. Comparisons across runs are stable and reproducible — no LLM-as-judge instability.</div>
            </div>
            <div class="win-card reveal reveal-delay-1">
                <div class="win-icon">🔄</div>
                <div class="win-title">Clean observation-action-reward loop</div>
                <div class="win-body">The surface fits standard RL trainers and evaluation harnesses without extra glue. Plug in any agent and go.</div>
            </div>
            <div class="win-card reveal reveal-delay-2">
                <div class="win-icon">📚</div>
                <div class="win-title">Built as a curriculum</div>
                <div class="win-body">Three tasks: local synthesis, multi-block coordination, then architecture planning. Easy, medium, hard — not arbitrary difficulty labels.</div>
            </div>
            <div class="win-card reveal reveal-delay-3">
                <div class="win-icon">🏗️</div>
                <div class="win-title">Coherence as a first-class reward</div>
                <div class="win-body">Architectural consistency is explicitly rewarded at submit time. Block-level correctness alone isn't enough — the whole file must make sense.</div>
            </div>
        </div>
    </div>
</section>

<!-- QUICKSTART -->
<section id="quickstart">
    <div class="container">
        <div class="section-kicker">Quick Start</div>
        <h2 class="section-title reveal">Up and running in three commands.</h2>
        <p class="section-sub reveal">Run the server, call the API, execute the baseline. Set three environment variables and you're evaluating.</p>
        <div class="qs-grid reveal">
            <div class="qs-card">
                <div class="qs-card-head">
                    <span class="dot dot-red"></span>
                    <span class="dot dot-yellow"></span>
                    <span class="dot dot-green"></span>
                    Docker
                </div>
                <div class="code-block" style="border-radius: 0;">
<span class="comment"># Build and run</span>
cp .env.example .env
docker build -t gitmergeenv .
docker run -p 7860:7860 \
  -e API_BASE_URL=... \
  -e API_KEY=... \
  -e MODEL_NAME=... \
  gitmergeenv

<span class="comment"># Health check</span>
curl http://localhost:7860/health</div>
            </div>
            <div class="qs-card">
                <div class="qs-card-head">
                    <span class="dot dot-red"></span>
                    <span class="dot dot-yellow"></span>
                    <span class="dot dot-green"></span>
                    Baseline inference
                </div>
                <div class="code-block" style="border-radius: 0;">
<span class="comment"># Required environment variables</span>
<span class="key">API_BASE_URL</span>=https://router.huggingface.co/v1
<span class="key">API_KEY</span>=your_api_key_here
<span class="key">MODEL_NAME</span>=Qwen/Qwen2.5-72B-Instruct

<span class="comment"># Run baseline against all 3 tasks</span>
python inference.py

<span class="comment"># Or trigger via API</span>
curl -X POST localhost:7860/baseline</div>
            </div>
        </div>
        <div class="callout reveal" style="margin-top: 1.5rem;">
            Judges inject <code>API_BASE_URL</code>, <code>API_KEY</code>, and <code>MODEL_NAME</code> directly. The inference script uses these variables with no hardcoded fallbacks.
        </div>
    </div>
</section>

<!-- FOOTER -->
<footer>
    <div class="container">
        <div class="footer-inner">
            <div>
                <div class="footer-logo">GitMergeEnv</div>
                <div class="footer-desc">Deterministic grading. Multi-step learning loop. Built for OpenEnv-compatible RL training and evaluation.</div>
            </div>
            <div class="footer-links">
                <a href="/docs">Swagger UI</a>
                <a href="/health">Health</a>
                <a href="/validate" onclick="event.preventDefault(); fetch('/validate',{method:'POST'}).then(r=>r.json()).then(d=>alert(JSON.stringify(d,null,2)))">Run Validate</a>
            </div>
        </div>
    </div>
</footer>

<script>
    // Scroll reveal
    const reveals = document.querySelectorAll('.reveal');
    const io = new IntersectionObserver((entries) => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                e.target.classList.add('visible');
                io.unobserve(e.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
    reveals.forEach(el => io.observe(el));

    // Loop steps staggered animation
    const loopSteps = document.querySelectorAll('.loop-step');
    loopSteps.forEach((step, i) => {
        setTimeout(() => {
            step.style.animationDelay = `${i * 0.15}s`;
            step.classList.add('visible');
        }, 400 + i * 150);
    });

    // Weight bars
    const weightBars = document.querySelectorAll('.weight-bar-fill');
    const wio = new IntersectionObserver((entries) => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                e.target.style.width = (parseFloat(e.target.style.getPropertyValue('--w')) * 100) + '%';
                e.target.classList.add('animated');
                wio.unobserve(e.target);
            }
        });
    }, { threshold: 0.3 });
    weightBars.forEach(bar => {
        bar.style.width = '0%';
        wio.observe(bar);
    });

    // Nav scroll effect
    window.addEventListener('scroll', () => {
        const nav = document.querySelector('nav');
        if (window.scrollY > 40) {
            nav.style.borderBottomColor = 'rgba(26,20,16,0.14)';
        } else {
            nav.style.borderBottomColor = 'rgba(26,20,16,0.10)';
        }
    });
</script>

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

# ⚽ Gegenbot

An autonomous, AI-augmented Fantasy Premier League (FPL) management daemon, solver, and web dashboard built to run 24/7 on lightweight hardware (such as a Raspberry Pi) via Docker.

Gegenbot pairs mathematical linear programming (MILP) with Google Gemini to optimize lineups, automate transfers, handle injury rotation, manage chip deployment, and dispatch natural-language briefings to Telegram before every gameweek deadline.

---

## ⚡ Architecture Overview

```text
┌──────────────────────────────────────────────────────────┐
│                      Official FPL API                    │
└──────────────▲─────────────────────────────┬─────────────┘
               │ (Transfers & Lineups)       │ (Bootstrap, Live, Fixtures)
┌──────────────┴─────────────────────────────▼─────────────┐
│                       fpl-worker                         │
│  ┌────────────────────┐          ┌────────────────────┐  │
│  │    MILP Solver     │ ◄──────► │     Gemini LLM     │  │
│  │ (xP Optimization)  │          │(Narrative & Context│  │
│  └────────────────────┘          └────────────────────┘  │
│             │                               │            │
│             ▼                               ▼            │
│    Automated Execution             Telegram Match Digest │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                      fpl-dashboard                       │
│        (Flask Web UI • Live Pitch • Mini-Leagues)        │
└──────────────────────────────────────────────────────────┘

# ⚽ Gegenbot

An autonomous, AI-augmented Fantasy Premier League (FPL) tactical decision daemon, mathematical solver, and live web dashboard built to run 24/7 on lightweight hardware (such as a Raspberry Pi) via Docker or directly with Python.

Gegenbot pairs mathematical mixed-integer linear programming (**HiGHS MILP**) with an **AI Decision Director** (powered by LLMs like Google Gemini / OpenAI) to optimize lineups, automate transfers, evaluate game-theoretic Effective Ownership (EO%), decode live press conference quotes, manage chip deployment, and dispatch natural-language briefings to Telegram before every gameweek deadline.

---

## ⚡ Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Live Data Feeds & Real-Time Intelligence                        │
│  ┌───────────────────────┐  ┌─────────────────────────┐  ┌──────────────────────────┐  │
│  │   Official FPL API    │  │    FPL Core Insights    │  │   Vaastav FPL Dataset    │  │
│  │ • Bootstrap & Prices  │  │ • Live xP & xGI/90      │  │ • 2026-27 Match Logs     │  │
│  │ • Mini-League Rivals  │  │ • Defensive Contrib.    │  │ • Rolling xGI/xGC/90     │  │
│  └───────────┬───────────┘  └────────────┬────────────┘  └────────────┬─────────────┘  │
└──────────────┼───────────────────────────┼────────────────────────────┼────────────────┘
               ▼                           ▼                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   fpl-worker                                           │
│  ┌─────────────────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │        HiGHS MILP Solver Engine         │  │        AI Decision Director         │  │
│  │ • Multi-Period Lookahead Decay (γᵗ)     │  │ • Press Conference Quote Decryption │  │
│  │ • Bench Sub-Factor Utility Weighting    │  │ • Dynamic Armband & Shielding       │  │
│  │ • Exact Half-Profit Selling Prices      │  │ • Late Injury Veto & Re-Solve Loop  │  │
│  │ • EO% Mini-League Shielding & Daggers   │  │ • Contextual Tactical Rationale     │  │
│  │ • Banked Free Transfer Valuation (+1.5) │  └──────────────────┬──────────────────┘  │
│  └────────────────────┬────────────────────┘                     │                     │
│                       │                                          │                     │
│                       ▼                                          ▼                     │
│              Automated Execution                       Telegram Match Digest           │
└───────────────────────┬────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 fpl-dashboard                                          │
│                   (Flask Web UI • Live Pitch • Mini-Leagues)                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Engine Capabilities

### 1. Mathematical Solver (HiGHS MILP)
* **Lightning-Fast Open-Source Solving:** Solves the entire 15-man squad integer program with positional and budget constraints in **$\approx 15\text{ms}$** using `highspy`.
* **Multi-Gameweek Horizon Decay ($\gamma^t$):** Optimizes across a configurable lookahead horizon ($N=1\dots 5$, default $N=3$ with $\gamma=0.85$), preventing short-sighted 1-week punts.
* **Bench Strength / Sub-Factor Optimization:** Weighs substitute players at $10\%$ expected auto-sub utility, ensuring reliable £4.0m/£4.5m playing bench depth while avoiding non-playing deadwood.
* **Exact Selling Price Accounting:** Ingests live `selling_price` from the FPL API to calculate precise squad liquidation values (half-profit rule), preventing HTTP 400 transfer rejections.
* **Game-Theoretic Effective Ownership (EO%) Shielding:** Scans rival squads in configured mini-leagues to identify **Shields** (high-ownership rivals), **Vulnerabilities**, and **Daggers** (differentials) to protect or gain rank.
* **Banked Free Transfer Valuation:** Awards $+1.5\text{ xP}$ strategic value to banking a free transfer, preventing unnecessary lateral churn.

### 2. Multi-Source Ingestion & Underlying Statistics
* **`olbauday/FPL-Core-Insights`:** Automatically ingests live expected points ($xP$), expected goal involvement ($xGI/90$), and defensive contribution metrics.
* **`vaastav/Fantasy-Premier-League`:** Ingests match logs and computes rolling $xGI/90$, $xGC/90$, average minutes in recent appearances, and starts reliability.
* **Proactive Injury & Minutes Reliability Multipliers:** Dynamically scales availability ($75\% \rightarrow 0.8\times$, $50\% \rightarrow 0.4\times$, $25\% \rightarrow 0.1\times$, $0\% \rightarrow 0.0\times$) and flags minutes fragility.

### 3. Expanded AI Decision Director
* **Manager Press Conference Decryption:** Decodes nuanced managerial quotes (e.g. Pep, Arteta, Emery) for key targets and captains.
* **Emergency Veto & Instant Re-Solve Loop:** If late-breaking news reveals an unexpected injury or benching, the Director issues an emergency veto and the HiGHS solver instantly re-solves in 15ms for a clean alternative.
* **Armband Authority:** Intelligently validates or overrides Captain and Vice-Captain picks based on late weather, press quotes, and mini-league risk mode (`DEFEND`, `CHASE`, `NEUTRAL`).

---

## 🚀 Quick Start

### Prerequisites
* Python 3.10+ or Docker & Docker Compose
* FPL account credentials or authentication cookie
* Telegram Bot Token & Chat ID (for deadline alerts)
* LLM API Key (Google Gemini, OpenAI, Groq, DeepSeek, or local Ollama)

### 1. Clone the Repository
```bash
git clone git@github.com:<YOUR_GITHUB_USERNAME>/gegenbot.git
cd gegenbot
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
```

Populate your `.env` configuration:

```env
# --- FPL Credentials & IDs ---
FPL_TEAM_ID=your_team_id
FPL_EMAIL=your_fpl_email
FPL_PASSWORD=your_fpl_password
FPL_LEAGUE_ID=your_primary_mini_league_id

# --- Data Feeds & Underlying Stats ---
VAASTAV_DATA_ENABLED=true
VAASTAV_SEASON=2026-27

# --- Mathematical Solver Configuration ---
DECAY_FACTOR=0.85                # Horizon Decay Factor (gamma)
HORIZON_LENGTH=3                 # Lookahead horizon in Gameweeks (1 to 5)
BENCH_WEIGHT_FACTOR=0.10         # Auto-sub utility weight for bench players
HIT_MIN_NET_XP_GAIN=5.0          # Minimum net xP gain required to justify taking a -4 hit
ROLLING_BONUS_XP=1.5             # Strategic valuation for rolling/banking a Free Transfer

# --- AI Decision Director (LLM) ---
LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
ENABLE_LIVE_NEWS=true

# --- Telegram Alert Dispatcher ---
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Run with Python or Docker Compose

#### Option A: Run Directly with Python
```bash
# Install dependencies
pip install -r requirements.txt

# Run a dry-run preview (generates candidates, consults AI Director, dispatches Telegram alert)
python3 src/main.py --dry-run

# Run live transfer execution
python3 src/main.py --execute
```

#### Option B: Launch via Docker Compose
```bash
docker compose up -d --build
```
Access the web dashboard at `http://localhost:5000`.

---

## 🛠️ Testing & Verification

Gegenbot includes a comprehensive unit test suite covering auth, data fetchers, news tracking, HiGHS solver formulations, chip evaluations, and AI Director decision logic.

```bash
pytest
```
*All 61 tests passing.*

---

## 📁 Repository Structure

```text
├── docker-compose.yml       # Multi-container orchestration (worker, dashboard)
├── Dockerfile               # Container build definition
├── requirements.txt         # Python dependencies (PuLP, highspy, requests, pandas, Flask, etc.)
├── .env.example             # Template configuration file
├── data/                    # Local cache (projections, decisions, auth tokens)
├── tests/                   # Full unit and integration test suite
└── src/
    ├── agent/               # AI Decision Director (LLM reasoning, quote decryption, veto logic)
    ├── api/                 # FPL API clients, auth token manager, and endpoint handlers
    ├── dashboard/           # Flask web UI, live pitch visualization, and mini-league tables
    ├── engine/              # HiGHS MILP optimizer, horizon scoring, and metrics calculation
    ├── notifier/            # Telegram alerting module with press conference alerts
    ├── tracker/             # Mini-league scanner, EO% analyzer, and live news tracker
    ├── data_fetcher.py      # Multi-source data pipeline (FPL Core Insights & Vaastav)
    └── main.py              # CLI entry point, scheduler loop, and pipeline orchestrator
```

---

## 🛡️ License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
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
```

* **FPL API Client & Auth:** Authenticated session handling for fetching live squad data, gameweek fixtures, and executing roster updates.
* **Optimization Engine:** Mixed-Integer Linear Programming (MILP) model that maximizes multi-week expected points ($xP$) subject to budget constraints, player club limits, and valid formations.
* **Injury & Bench Hierarchy:** Dynamic status discounting (75%, 50%, 25%, 0%) with automatic bench order sorting (positions 12–15) and safe vice-captain assignment.
* **Chip Strategy & DGW Planner:** Automated threshold triggers for Wildcard, Free Hit, Bench Boost, and Triple Captain with minutes scaling for Double Gameweeks.
* **Gemini LLM Layer:** Translates raw optimization output into human-readable managerial briefings and press-conference injury context.
* **Flask Web Dashboard:** Visual pitch display of your active squad, bench order, and live rank tracking across both Global and Private mini-leagues.
* **Telegram Bot:** Real-time notifications for automated transfer executions, deadline alerts, and gameweek reviews.

---

## 🚀 Quick Start

### Prerequisites
* Docker & Docker Compose
* Raspberry Pi / Linux Server / macOS / Windows
* FPL Account credentials or authentication token
* Telegram Bot Token & Chat ID (optional, for alerts)
* Google Gemini API Key

### 1. Clone the Repository
```bash
git clone git@github.com:<YOUR_GITHUB_USERNAME>/gegenbot.git
cd gegenbot
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Populate the required configuration:

```env
# FPL Credentials & Team ID
FPL_TEAM_ID=your_team_id
FPL_EMAIL=your_fpl_email
FPL_PASSWORD=your_fpl_password

# Execution Controls
ENABLE_AUTO_TRANSFERS=true
AUTO_EXECUTE=true
ENABLE_AUTO_CHIPS=true

# AI & Notifications
GEMINI_API_KEY=your_gemini_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
TELEGRAM_NOTIFICATIONS_ENABLED=true

# Solver & Chip Strategy Thresholds
WILDCARD_MIN_XP_GAIN=18.0
FREE_HIT_MIN_XP_GAIN=14.0
BENCH_BOOST_MIN_BENCH_XP=16.0
TRIPLE_CAPTAIN_MIN_XP=11.5
DGW_MINUTES_SCALING=1.75
MIN_CHANCE_START_THRESHOLD=75
```

### 3. Launch via Docker Compose
Build and run the background worker daemon and web dashboard:

```bash
docker compose up -d --build
```

Access the web interface at `http://localhost:5000` (or `http://<PI_IP>:5000`).

---

## 🛠️ CLI & Manual Operations

### Run a Dry Run
Simulate a full solve, injury evaluation, and lineup selection without executing live transfers:

```bash
docker compose exec fpl-worker python src/main.py --dry-run
```

### Check Container Logs
```bash
# Monitor the background scheduler daemon
docker compose logs -f --tail=50 fpl-worker

# Check dashboard access logs
docker compose logs -f --tail=50 fpl-dashboard
```

---

## 📁 Repository Structure

```text
├── docker-compose.yml       # Multi-container service definitions (worker, dashboard)
├── Dockerfile               # Container build configuration
├── requirements.txt         # Python dependencies
├── .env.example             # Template configuration file
├── data/                    # Persistent storage (cached stats, projections)
└── src/
    ├── api/                 # FPL API clients and authentication handlers
    ├── dashboard/           # Flask app, HTML pitch templates, static assets
    ├── llm/                 # Gemini integration for narrative generation
    ├── notifications/       # Telegram bot alerting module
    ├── solver/              # MILP mathematical optimization models & heuristics
    ├── data_fetcher.py      # Fixture, player, and injury ingestion pipeline
    └── main.py              # CLI entry point and scheduler loop
```

---

## 🛡️ License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

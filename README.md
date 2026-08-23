# ⚽ Gegenbot

An autonomous, AI-augmented Fantasy Premier League (FPL) management daemon, solver, and web dashboard built to run 24/7 on lightweight hardware (such as a Raspberry Pi) via Docker.

Gegenbot pairs mathematical mixed-integer linear programming (MILP) with an LLM Decision Director (powered by Google Gemini) to optimize lineups, automate transfers, calculate real squad liquidation values, evaluate game-theoretic Effective Ownership (EO%), manage chip deployment, and dispatch natural-language briefings to Telegram before every gameweek deadline.

---

## ⚡ Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          Official FPL API                              │
└──────────────▲───────────────────────────────────────────┬─────────────┘
               │ (Transfers, Lineups, Auth Validation)     │ (Bootstrap, Live, Fixtures)
┌──────────────┴───────────────────────────────────────────▼─────────────┐
│                           fpl-worker                                   │
│  ┌─────────────────────────┐          ┌─────────────────────────────┐  │
│  │   MILP Engine (PuLP)    │ ◄──────► │     AI Decision Director    │  │
│  │ • Multi-Period Lookahead│          │      (Google Gemini)        │  │
│  │ • Exact Selling Price   │          │ • Managerial Tactical Rationale│
│  │ • EO% Shield/Dagger Scan│          │ • Context & Injury Synthesis│  │
│  │ • Rolling FT Valuation  │          └──────────────┬──────────────┘  │
│  └───────────┬─────────────┘                         │                 │
│              │                                       │                 │
│              ▼                                       ▼                 │
│     Automated Execution                    Telegram Match Digest       │
└────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         fpl-dashboard                                  │
│         (Flask Web UI • Live Pitch • Global & Mini-Leagues)            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Engine Capabilities

* **Exact Selling Price Accounting (Half-Profit Rule):** Ingests live `selling_price` from the FPL API to calculate precise squad liquidation values, preventing HTTP 400 transfer rejections caused by price-rise inflation.
* **Multi-Period Horizon Lookahead & Rolling FT Valuation:** Optimizes across multi-gameweek horizons ($GW_t \dots GW_{t+2}$) and assigns a $+1.5\text{ xP}$ strategic value bonus to banking a free transfer, curbing unnecessary lateral churn.
* **Game-Theoretic Effective Ownership (EO%) Shielding:** Scans rival squads in configured mini-leagues prior to optimization, identifying **Shields** (protection), **Vulnerabilities**, and **Daggers** (differentials) to guard rank when player $xP$ margins are tight.
* **Concurrent Mini-League Scanner:** Uses multithreaded worker pools with TTL caching to complete mini-league ownership scanning in $<1\text{s}$.
* **Proactive Authentication Health Checks:** Validates tokens and cookies prior to pipeline runs to ensure execution safety.
* **Automated Chip Evaluation & DGW Guards:** Checks expected point gains against configurable thresholds for Wildcard, Free Hit, Bench Boost, and Triple Captain while guarding against low-upside single-gameweek activations.
* **Dynamic Bench & Vice-Captain Rules:** Automatically applies injury multipliers ($75\% \rightarrow 0.8\times$, $50\% \rightarrow 0.4\times$, $25\% \rightarrow 0.1\times$, $0\% \rightarrow 0.0\times$), sorts substitutes dynamically by discounted $xP$, and locks vice-captaincy to a $100\%$ fit starter.

---

## 🚀 Quick Start

### Prerequisites
* Docker & Docker Compose
* Raspberry Pi / Linux Server / macOS / Windows
* FPL account credentials or authentication cookie
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

Populate your configuration:

```env
# FPL Credentials & IDs
FPL_TEAM_ID=your_team_id
FPL_EMAIL=your_fpl_email
FPL_PASSWORD=your_fpl_password
FPL_LEAGUE_ID=your_primary_mini_league_id

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

Access the dashboard at `http://localhost:5000` (or `http://<PI_IP>:5000`).

---

## 🛠️ CLI & Manual Operations

### Run a Dry Run
Simulate a full solve, mini-league scan, chip evaluation, and lineup selection without executing live transfers:

```bash
docker compose exec fpl-worker python src/main.py --dry-run
```

### Check Container Logs
```bash
# Monitor the background worker & scheduler daemon
docker compose logs -f --tail=50 fpl-worker

# Check dashboard access logs
docker compose logs -f --tail=50 fpl-dashboard
```

---

## 📁 Repository Structure

```text
├── docker-compose.yml       # Multi-container orchestration (worker, dashboard)
├── Dockerfile               # Container build definition
├── requirements.txt         # Python dependencies (PuLP, requests, google-genai, Flask, etc.)
├── .env.example             # Template configuration file
├── data/                    # Persistent storage (cached stats, decisions, state)
└── src/
    ├── agent/               # AI Decision Director (Gemini LLM reasoning & rationale)
    ├── api/                 # FPL API clients, auth validation, and endpoint handlers
    ├── dashboard/           # Flask web UI, pitch templates, and mini-league tables
    ├── engine/              # MILP mathematical optimizer, horizon scoring & selling price models
    ├── notifier/            # Telegram alerting module
    ├── tracker/             # Multithreaded mini-league scanner & EO% analyzer
    ├── data_fetcher.py      # Fixture, player, and injury ingestion pipeline
    └── main.py              # CLI entry point, dry-run runner, and scheduler loop
```

---

## 🛡️ License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
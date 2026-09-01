# ⚽ Gegenbot

An autonomous, AI-augmented Fantasy Premier League (FPL) tactical decision daemon, mathematical solver, and live web dashboard built to run 24/7 on lightweight hardware (such as a Raspberry Pi) via Docker or directly with Python.

Gegenbot pairs mathematical mixed-integer linear programming (**HiGHS MILP**) with an **AI Decision Director** (powered by LLMs like Google Gemini / OpenAI) to optimize lineups, automate transfers, evaluate game-theoretic Effective Ownership (EO%), decode live press conference quotes, manage chip deployment, and dispatch natural-language briefings to Telegram before every gameweek deadline.

---
## ⚡ Architecture Overview

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             Live Data Feeds & Real-Time Intelligence                             │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐ ┌─────────────┐ │
│  │ Official FPL API │ │FPL Core Insights │ │Vaastav Match Logs│ │ The-Odds-API │ │Price Tracker│ │
│  │•Bootstrap & Sells│ │•Multi-GW xP Proj.│ │•xGI/90 & xGC/90  │ │•Vig Removal  │ │•Net Velocity│ │
│  │•Set-Piece Roles  │ │•Def. Contribution│ │•xGI Mean Revers. │ │•Implied CS % │ │•Rise/Fall Δ │ │
│  │•Mini-League Ranks│ │•Underlying Threat│ │•Rolling Mins/BPS │ │•Implied Goal%│ │•±95% Targets│ │
│  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘ └──────┬───────┘ └──────┬──────┘ │
└───────────┼────────────────────┼────────────────────┼──────────────────┼────────────────┼────────┘
            ▼                    ▼                    ▼                  ▼                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           fpl-worker                                             │
│  ┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────────┐  │
│  │           HiGHS MILP Solver Engine           │  │     Moneyball & Sabermetrics Engine      │  │
│  │ • Multi-Period Lookahead Decay (γᵗ)          │  │ • Mean Reversion (xGI - (G+A) Δ)         │  │
│  │ • Exact Half-Profit Selling Prices           │  │ • VORP per Marginal Million (£m Yield)  │  │
│  │ • Bench Sub-Factor Utility Weighting (10%)   │  │ • "Barbell" Portfolio (Anchor + Enabler) │  │
│  │ • Banked Free Transfer Valuation (+1.5 xP)   │  │ • Set-Piece Baseline Bonus (Pen/FK/Cor)  │  │
│  └──────────────────────┬───────────────────────┘  └────────────────────┬─────────────────────┘  │
│                         │                                               │                        │
│                         ▼                                               ▼                        │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                  AI Decision Director                                      │  │
│  │ • Press Conference Decryption     • Emergency Veto Loop      • Dynamic Captaincy Armband   │  │
│  │ • Mini-League Shielding & Daggers • Risk Mode (DEFEND/CHASE) • Contextual Tactical Rationale│  │
│  └──────────────────────────────────────────────┬─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┼────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │              Automated Dispatch                 │
                         │  • Live Squad Execution & FPL API Lock-in       │
                         │  • Natural-Language Telegram Match Briefing     │
                         │  • Flask Live Dashboard Pitch Visualization     │
                         └─────────────────────────────────────────────────┘
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

### 2. Moneyball Analytics & Sabermetrics Engine
* **Luck vs. Skill Mean Reversion ($xGI\text{ Delta}$):** Quantifies variance between underlying threat and actual returns ($xGI\text{ Delta} = xGI - (G+A)$) to spot high-threat assets *before* they haul while flagging overvalued, lucky haulers.
* **VORP per Marginal Million (£m Efficiency):** Calculates Value Over Replacement Player relative to positional floors (GKP: £4.0m, DEF: £4.0m, MID: £4.5m, FWD: £4.5m) to find hyper-efficient budget enablers.
* **"Barbell" Portfolio Architecture:** Maximizes marginal points per pound across the £4.5m–£6.5m supporting cast, seamlessly funding elite £15.0m captaincy engines (Haaland / Salah).
* **Set-Piece & Penalty Hierarchy Registry:** Ingests official penalty orders, direct free-kick orders, and corner duties to award baseline threat boosts and spot cheap set-piece specialists.
* **Price Change Target Predictor:** Computes net transfer velocity targets to identify imminent $+£0.1\text{m}$ price rises and $-£0.1\text{m}$ price falls, locking in squad value.
* **Mini-League `CHASE` Mode Daggers:** Injects high-EV, low-EO% Moneyball differentials into tactical candidate screening to overcome mini-league point deficits.

| Tag Archetype | Criteria | Strategic Action |
| :--- | :--- | :--- |
| `ELITE_ANCHOR` | Cost $\ge £9.5\text{m}$, $\text{xP} \ge 5.5$ | Permanent captaincy engine; protected from budget stripping |
| `UNDERVALUED_REGRESSION` | $xGI\text{ Delta} \ge +0.60$, Cost $\le £8.5\text{m}$ | Prime buy target; expected positive mean reversion |
| `HIGH_EFFICIENCY_ENABLER` | $\text{VORP/£m} \ge 1.50$, Cost $\le £6.5\text{m}$ | Starting XI budget enabler to fund elite premiums |
| `OVERVALUED_HAULER` | $xGI\text{ Delta} \le -1.20$, $\text{Form} \ge 5.0$ | Sell candidate; lucky conversion rate overdue negative regression |

### 3. Multi-Source Ingestion & Underlying Statistics
* **`olbauday/FPL-Core-Insights`:** Automatically ingests live expected points ($xP$), expected goal involvement ($xGI/90$), and defensive contribution metrics.
* **`vaastav/Fantasy-Premier-League`:** Ingests match logs and computes rolling $xGI/90$, $xGC/90$, average minutes in recent appearances, and starts reliability.
* **Proactive Injury & Minutes Reliability Multipliers:** Dynamically scales availability ($75\% \rightarrow 0.8\times$, $50\% \rightarrow 0.4\times$, $25\% \rightarrow 0.1\times$, $0\% \rightarrow 0.0\times$) and flags minutes fragility.

### 4. Bookmaker Odds & Market Implied Probabilities
* **Vig-Free Probability Modeling:** Ingests live betting market odds (via `The-Odds-API` or calibrated team strength models) and removes bookmaker margin (overround) to derive pure statistical likelihoods.
* **Implied Clean Sheet ($P(\text{CS})$) & Goalscorer ($P(\text{Goal})$):** Translates liquid multi-million pound betting markets into fair percentages per player for tactical decision-making.
* **Market-Priced Expected Points ($xP_{\text{odds}}$):** Calculates independent mathematical valuations according to positional scoring rules, directly boosting high-ceiling captaincy locks and clean sheet anchors.

### 5. Expanded AI Decision Director
* **Manager Press Conference Decryption:** Decodes nuanced managerial quotes (e.g. Arteta, Slot, Emery, Maresca) for key targets and captains.
* **Emergency Veto & Instant Re-Solve Loop:** If late-breaking news reveals an unexpected injury or benching, the Director issues an emergency veto and the HiGHS solver instantly re-solves in 15ms for a clean alternative.
* **Armband Authority & Moneyball Intelligence:** Intelligently validates or overrides Captain and Vice-Captain picks based on market odds, press quotes, Moneyball regression tags, and mini-league risk mode (`DEFEND`, `CHASE`, `NEUTRAL`).

---

## 🚀 Quick Start

### Prerequisites
* Python 3.10+ or Docker & Docker Compose
* FPL account credentials or authentication cookie
* Telegram Bot Token & Chat ID (for deadline alerts)
* LLM API Key (Google Gemini, OpenAI, Groq, DeepSeek, or local Ollama)
* *(Optional)* The-Odds-API Key for live bookmaker lines

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
THE_ODDS_API_KEY=your_optional_odds_api_key

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

Gegenbot includes a comprehensive unit test suite covering auth, data fetchers, news tracking, Moneyball metrics, HiGHS solver formulations, chip evaluations, and AI Director decision logic.

```bash
pytest
```
*All 73 tests passing (100%).*

---

## 📁 Repository Structure

```text
├── docker-compose.yml       # Multi-container orchestration (worker, dashboard)
├── Dockerfile               # Container build definition
├── requirements.txt         # Python dependencies (PuLP, highspy, requests, pandas, Flask, etc.)
├── .env.example             # Template configuration file
├── data/                    # Local cache (projections, decisions, auth tokens)
├── tests/                   # Full unit and integration test suite (73 tests)
└── src/
    ├── agent/               # AI Decision Director (LLM reasoning, quote decryption, veto logic)
    ├── api/                 # FPL API clients, auth token manager, and endpoint handlers
    ├── dashboard/           # Flask web UI, live pitch visualization, and mini-league tables
    ├── engine/              # HiGHS MILP optimizer, horizon scoring, and metrics calculation
    ├── notifier/            # Telegram alerting module with press conference alerts
    ├── tracker/             # Mini-league scanner, EO% analyzer, and live news tracker
    ├── data_fetcher.py      # Multi-source data pipeline (FPL Core Insights & Vaastav)
    ├── odds_tracker.py      # Bookmaker odds ingestion, vig removal, and implied probabilities
    ├── price_tracker.py     # Price change target predictor and transfer velocity tracker
    └── main.py              # CLI entry point, scheduler loop, and pipeline orchestrator
```

---

## 🛡️ License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
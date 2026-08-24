"""Lightweight Flask web dashboard for autonomous FPL visual analysis."""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, render_template_string

logger = logging.getLogger(__name__)

# In-memory entry cache with 5-minute TTL to prevent API spam
_ENTRY_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_ENTRY_CACHE_TTL = 300  # seconds

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous FPL Engine | Tactical Command Center</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: {
                            50: '#ecfdf5',
                            500: '#10b981',
                            600: '#059669',
                            700: '#047857',
                            900: '#064e3b',
                        },
                        pitch: {
                            dark: '#0f3822',
                            light: '#14462b',
                            line: '#2d6a4f',
                        }
                    }
                }
            }
        }
    </script>
    <style>
        .pitch-pattern {
            background: repeating-linear-gradient(
                0deg,
                #134e2a,
                #134e2a 35px,
                #104424 35px,
                #104424 70px
            );
            border: 2px solid rgba(255, 255, 255, 0.2);
            position: relative;
        }
        .pitch-center-circle {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 100px;
            height: 100px;
            border: 2px solid rgba(255, 255, 255, 0.15);
            border-radius: 50%;
            pointer-events: none;
        }
        .pitch-center-line {
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 2px;
            background: rgba(255, 255, 255, 0.15);
            pointer-events: none;
        }
        .pitch-penalty-top {
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 180px;
            height: 70px;
            border: 2px solid rgba(255, 255, 255, 0.15);
            border-top: none;
            pointer-events: none;
        }
        .pitch-penalty-bottom {
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 180px;
            height: 70px;
            border: 2px solid rgba(255, 255, 255, 0.15);
            border-bottom: none;
            pointer-events: none;
        }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased selection:bg-emerald-500 selection:text-white">

    <!-- Navbar -->
    <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-9 h-9 rounded-lg bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
                    <span class="text-xl">⚽</span>
                </div>
                <div>
                    <div class="flex items-center space-x-2">
                        <h1 class="font-bold text-lg leading-tight tracking-tight text-white">FPL AI Engine</h1>
                        {% if entry_data and entry_data.name %}
                        <span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-emerald-400 font-medium border border-slate-700">
                            {{ entry_data.name }}
                        </span>
                        {% endif %}
                    </div>
                    <p class="text-xs text-slate-400">
                        {% if entry_data and (entry_data.player_first_name or entry_data.player_last_name) %}
                        Manager: {{ entry_data.player_first_name }} {{ entry_data.player_last_name }} ·
                        {% endif %}
                        Autonomous MILP Optimization & Decision Director
                    </p>
                </div>
            </div>

            <div class="flex items-center space-x-4">
                {% if data and data.status %}
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-950 text-emerald-300 border border-emerald-800">
                    <span class="w-1.5 h-1.5 mr-1.5 bg-emerald-400 rounded-full animate-pulse"></span>
                    GW {{ data.gameweek }} Active
                </span>
                <span class="text-xs text-slate-400 hidden sm:inline">Updated: {{ data.timestamp[:19].replace('T', ' ') if data.timestamp else 'Recent' }}</span>
                {% else %}
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-950 text-amber-300 border border-amber-800">
                    <span class="w-1.5 h-1.5 mr-1.5 bg-amber-400 rounded-full"></span>
                    Standby / No Data
                </span>
                {% endif %}
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        <!-- Global Rankings Summary Section -->
        {% if global_leagues or global_summary %}
        <section class="space-y-3">
            <div class="flex items-center justify-between">
                <h2 class="text-sm font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-2">
                    <span>🌍 Global Rankings & Benchmarks</span>
                </h2>
                {% if entry_data and entry_data.summary_overall_points %}
                <span class="text-xs font-semibold text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-2.5 py-0.5 rounded-full">
                    Total Points: {{ entry_data.summary_overall_points | comma }} pts
                </span>
                {% endif %}
            </div>

            <!-- Primary Global Ranking Cards (Overall, Country, Favorite Club) -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                
                <!-- 1. Overall Rank Card -->
                <div class="bg-gradient-to-br from-slate-900 via-slate-900 to-indigo-950/30 border border-indigo-900/40 rounded-xl p-4 shadow-sm relative overflow-hidden">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-xs font-semibold text-indigo-400 uppercase tracking-wider flex items-center space-x-1.5">
                            <span>🌐</span>
                            <span>Overall Rank</span>
                        </span>
                        {% if global_summary.overall and global_summary.overall.direction == 'up' %}
                        <span class="inline-flex items-center text-xs font-bold text-emerald-400 bg-emerald-950/80 border border-emerald-800 px-2 py-0.5 rounded-full">
                            ▲ +{{ global_summary.overall.movement | comma }}
                        </span>
                        {% elif global_summary.overall and global_summary.overall.direction == 'down' %}
                        <span class="inline-flex items-center text-xs font-bold text-rose-400 bg-rose-950/80 border border-rose-800 px-2 py-0.5 rounded-full">
                            ▼ {{ (global_summary.overall.movement | abs) | comma }}
                        </span>
                        {% elif global_summary.overall and global_summary.overall.direction == 'same' %}
                        <span class="inline-flex items-center text-xs font-medium text-slate-400 bg-slate-800 px-2 py-0.5 rounded-full">
                            ▬ Unchanged
                        </span>
                        {% endif %}
                    </div>
                    
                    <div class="flex items-baseline space-x-2">
                        <span class="text-2xl font-black text-white font-mono">
                            #{{ global_summary.overall.rank | comma if global_summary.overall and global_summary.overall.rank else (entry_data.summary_overall_rank | comma if entry_data and entry_data.summary_overall_rank else 'N/A') }}
                        </span>
                    </div>

                    <div class="mt-2 flex items-center justify-between text-xs text-slate-400 border-t border-slate-800/80 pt-2">
                        <span>
                            {% if global_summary.overall and global_summary.overall.last_rank %}
                            Prev: #{{ global_summary.overall.last_rank | comma }}
                            {% else %}
                            Global Benchmark
                            {% endif %}
                        </span>
                        <span>
                            {% if global_summary.overall and global_summary.overall.total_players %}
                            of {{ global_summary.overall.total_players | comma }}
                            {% else %}
                            Global League
                            {% endif %}
                        </span>
                    </div>
                </div>

                <!-- 2. Country / Regional Rank Card -->
                <div class="bg-gradient-to-br from-slate-900 via-slate-900 to-emerald-950/30 border border-emerald-900/40 rounded-xl p-4 shadow-sm relative overflow-hidden">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-xs font-semibold text-emerald-400 uppercase tracking-wider flex items-center space-x-1.5">
                            <span>📍</span>
                            <span class="truncate max-w-[140px]">{{ global_summary.country.name if global_summary.country else (entry_data.player_region_name if entry_data and entry_data.player_region_name else 'Regional') }}</span>
                        </span>
                        {% if global_summary.country and global_summary.country.direction == 'up' %}
                        <span class="inline-flex items-center text-xs font-bold text-emerald-400 bg-emerald-950/80 border border-emerald-800 px-2 py-0.5 rounded-full">
                            ▲ +{{ global_summary.country.movement | comma }}
                        </span>
                        {% elif global_summary.country and global_summary.country.direction == 'down' %}
                        <span class="inline-flex items-center text-xs font-bold text-rose-400 bg-rose-950/80 border border-rose-800 px-2 py-0.5 rounded-full">
                            ▼ {{ (global_summary.country.movement | abs) | comma }}
                        </span>
                        {% elif global_summary.country and global_summary.country.direction == 'same' %}
                        <span class="inline-flex items-center text-xs font-medium text-slate-400 bg-slate-800 px-2 py-0.5 rounded-full">
                            ▬ Unchanged
                        </span>
                        {% endif %}
                    </div>
                    
                    <div class="flex items-baseline space-x-2">
                        <span class="text-2xl font-black text-white font-mono">
                            #{{ global_summary.country.rank | comma if global_summary.country and global_summary.country.rank else 'N/A' }}
                        </span>
                    </div>

                    <div class="mt-2 flex items-center justify-between text-xs text-slate-400 border-t border-slate-800/80 pt-2">
                        <span>
                            {% if global_summary.country and global_summary.country.last_rank %}
                            Prev: #{{ global_summary.country.last_rank | comma }}
                            {% else %}
                            National Rank
                            {% endif %}
                        </span>
                        <span>
                            {% if global_summary.country and global_summary.country.total_players %}
                            of {{ global_summary.country.total_players | comma }}
                            {% else %}
                            Country League
                            {% endif %}
                        </span>
                    </div>
                </div>

                <!-- 3. Favorite Club / Supporters Rank Card -->
                <div class="bg-gradient-to-br from-slate-900 via-slate-900 to-amber-950/30 border border-amber-900/40 rounded-xl p-4 shadow-sm relative overflow-hidden">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-xs font-semibold text-amber-400 uppercase tracking-wider flex items-center space-x-1.5">
                            <span>⚽</span>
                            <span class="truncate max-w-[140px]">{{ global_summary.club.name if global_summary.club else 'Favorite Club' }}</span>
                        </span>
                        {% if global_summary.club and global_summary.club.direction == 'up' %}
                        <span class="inline-flex items-center text-xs font-bold text-emerald-400 bg-emerald-950/80 border border-emerald-800 px-2 py-0.5 rounded-full">
                            ▲ +{{ global_summary.club.movement | comma }}
                        </span>
                        {% elif global_summary.club and global_summary.club.direction == 'down' %}
                        <span class="inline-flex items-center text-xs font-bold text-rose-400 bg-rose-950/80 border border-rose-800 px-2 py-0.5 rounded-full">
                            ▼ {{ (global_summary.club.movement | abs) | comma }}
                        </span>
                        {% elif global_summary.club and global_summary.club.direction == 'same' %}
                        <span class="inline-flex items-center text-xs font-medium text-slate-400 bg-slate-800 px-2 py-0.5 rounded-full">
                            ▬ Unchanged
                        </span>
                        {% endif %}
                    </div>
                    
                    <div class="flex items-baseline space-x-2">
                        <span class="text-2xl font-black text-white font-mono">
                            #{{ global_summary.club.rank | comma if global_summary.club and global_summary.club.rank else 'N/A' }}
                        </span>
                    </div>

                    <div class="mt-2 flex items-center justify-between text-xs text-slate-400 border-t border-slate-800/80 pt-2">
                        <span>
                            {% if global_summary.club and global_summary.club.last_rank %}
                            Prev: #{{ global_summary.club.last_rank | comma }}
                            {% else %}
                            Fan Rank
                            {% endif %}
                        </span>
                        <span>
                            {% if global_summary.club and global_summary.club.total_players %}
                            of {{ global_summary.club.total_players | comma }}
                            {% else %}
                            Supporters League
                            {% endif %}
                        </span>
                    </div>
                </div>

            </div>

            <!-- Additional Global Badges (e.g. Gameweek 1, Second Chance, etc.) -->
            {% if global_leagues and global_leagues|length > 3 %}
            <div class="flex flex-wrap gap-2 pt-1">
                {% for gl in global_leagues %}
                {% if gl.id != global_summary.overall.id and gl.id != global_summary.country.id and gl.id != global_summary.club.id %}
                <div class="inline-flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300">
                    <span class="text-slate-400 font-medium">{{ gl.name }}:</span>
                    <span class="font-mono font-bold text-white">#{{ gl.rank | comma }}</span>
                    {% if gl.direction == 'up' %}
                    <span class="text-[11px] text-emerald-400 font-semibold">▲+{{ gl.movement | comma }}</span>
                    {% elif gl.direction == 'down' %}
                    <span class="text-[11px] text-rose-400 font-semibold">▼{{ (gl.movement | abs) | comma }}</span>
                    {% endif %}
                    {% if gl.total_players %}
                    <span class="text-[10px] text-slate-500 font-mono">/ {{ gl.total_players | comma }}</span>
                    {% endif %}
                </div>
                {% endif %}
                {% endfor %}
            </div>
            {% endif %}

        </section>
        {% endif %}

        {% if not data or not data.decision %}
        <!-- Empty State -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center max-w-2xl mx-auto my-12 shadow-xl">
            <div class="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl text-slate-400">
                ⏳
            </div>
            <h2 class="text-xl font-bold text-white mb-2">No Optimization Run Recorded Yet</h2>
            <p class="text-slate-400 text-sm mb-6 leading-relaxed">
                The engine has not yet produced a <code class="text-emerald-400 bg-slate-800 px-1.5 py-0.5 rounded">data/latest_decision.json</code> state file.
                Run the orchestrator via terminal or daemon to generate the tactical lineup.
            </p>
            <div class="bg-slate-950 p-4 rounded-xl text-left border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto">
                <span class="text-slate-500"># Run dry-run simulation</span><br>
                python src/main.py --dry-run<br><br>
                <span class="text-slate-500"># Or execute live transfer & lineup</span><br>
                python src/main.py --execute
            </div>
        </div>
        {% else %}

        <!-- Top Metrics Cards -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
                <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Projected Net xP</div>
                <div class="mt-1 text-2xl font-black text-emerald-400">
                    {{ data.decision.projected_net_xp }}
                    <span class="text-xs font-normal text-slate-400">pts</span>
                </div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
                <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Tactical Move</div>
                <div class="mt-1 text-sm font-bold text-white truncate" title="{{ data.decision.chosen_move_name }}">
                    {{ data.decision.chosen_move_name }}
                </div>
                <div class="text-xs text-slate-400 truncate mt-0.5">{{ data.decision.transfers_description }}</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
                <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Bank Balance</div>
                <div class="mt-1 text-2xl font-bold text-white">
                    £{{ data.decision.selected_candidate.bank_remaining_m }}m
                </div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
                <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Decision Source</div>
                <div class="mt-1 flex items-center space-x-1.5">
                    {% if data.decision.source == 'LLM_DIRECTOR' %}
                    <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-950 text-purple-300 border border-purple-800">
                        🤖 AI Director
                    </span>
                    {% else %}
                    <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-950 text-blue-300 border border-blue-800">
                        📐 PuLP MILP
                    </span>
                    {% endif %}
                </div>
            </div>
        </div>

        <!-- AI Director Tactical Rationale Banner -->
        <div class="bg-gradient-to-r from-slate-900 via-slate-900 to-emerald-950/40 border border-emerald-900/50 rounded-2xl p-6 shadow-xl relative overflow-hidden">
            <div class="flex items-start space-x-4">
                <div class="p-3 bg-emerald-500/10 rounded-xl border border-emerald-500/20 text-2xl shrink-0">
                    🧠
                </div>
                <div class="space-y-1">
                    <div class="flex items-center space-x-2">
                        <h3 class="text-sm font-semibold text-emerald-400 tracking-wide uppercase">AI Director Tactical Directive</h3>
                        <span class="text-xs text-slate-500">Formation: {{ data.decision.selected_candidate.formation }}</span>
                    </div>
                    <p class="text-base text-slate-200 leading-relaxed italic">
                        "{{ data.decision.rationale }}"
                    </p>
                </div>
            </div>
        </div>

        {% if data.decision.news_alerts and data.decision.news_alerts|length > 0 %}
        <!-- Breaking Press Conference & News Alerts Card -->
        <div class="bg-slate-900/90 border border-amber-500/30 rounded-2xl p-5 shadow-lg relative overflow-hidden">
            <div class="flex items-start space-x-3">
                <div class="p-2.5 bg-amber-500/10 rounded-xl border border-amber-500/20 text-xl shrink-0">
                    📰
                </div>
                <div class="space-y-2 flex-1">
                    <div class="flex items-center justify-between">
                        <h3 class="text-xs font-bold text-amber-400 tracking-wider uppercase flex items-center gap-1.5">
                            <span>Press Conference & Injury Intelligence</span>
                            <span class="bg-amber-950 text-amber-300 text-[10px] px-2 py-0.5 rounded border border-amber-800/60 font-mono">Live Grounding</span>
                        </h3>
                        <span class="text-[11px] text-slate-400 font-mono">{{ data.decision.news_alerts|length }} active alert(s)</span>
                    </div>
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2">
                        {% for alert in data.decision.news_alerts %}
                        <div class="bg-slate-950/60 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-300 flex items-start space-x-2">
                            <span class="text-amber-400 mt-0.5">⚠️</span>
                            <span class="leading-relaxed">{{ alert }}</span>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Main Layout: Pitch on Left, Threat Matrix / Leagues on Right -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            
            <!-- Pitch View Column (7 cols) -->
            <div class="lg:col-span-7 space-y-4">
                <div class="flex items-center justify-between">
                    <h2 class="text-lg font-bold text-white flex items-center space-x-2">
                        <span>Starting XI Lineup</span>
                        <span class="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded font-normal">Formation: {{ data.decision.selected_candidate.formation }}</span>
                    </h2>
                    <span class="text-xs text-slate-400">Armband: <strong class="text-amber-400">{{ data.decision.captain_name }} (C)</strong></span>
                </div>

                <!-- Football Pitch -->
                <div class="pitch-pattern rounded-2xl p-6 min-h-[540px] flex flex-col justify-between shadow-2xl overflow-hidden relative">
                    <div class="pitch-center-line"></div>
                    <div class="pitch-center-circle"></div>
                    <div class="pitch-penalty-top"></div>
                    <div class="pitch-penalty-bottom"></div>

                    <!-- GKP Row -->
                    <div class="flex justify-center items-center gap-4 relative z-10">
                        {% for p in starters_by_pos.GKP %}
                        <div class="flex flex-col items-center group transition transform hover:-translate-y-1">
                            <div class="relative">
                                <div class="w-11 h-11 rounded-full bg-amber-500/90 border-2 border-white/80 flex items-center justify-center text-sm font-bold text-slate-950 shadow-md">
                                    🧤
                                </div>
                                {% if p.is_captain %}
                                <span class="absolute -top-1.5 -right-2 bg-amber-400 text-slate-950 text-[10px] font-black px-1 rounded shadow">C</span>
                                {% elif p.is_vice_captain %}
                                <span class="absolute -top-1.5 -right-2 bg-slate-300 text-slate-950 text-[10px] font-black px-1 rounded shadow">V</span>
                                {% endif %}
                            </div>
                            <div class="mt-1.5 bg-slate-950/85 backdrop-blur px-2 py-0.5 rounded border border-white/10 text-center shadow">
                                <div class="text-xs font-semibold text-white leading-tight truncate max-w-[85px]">{{ p.web_name }}</div>
                                <div class="text-[10px] text-emerald-400 font-mono">{{ p.xp }} xP</div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>

                    <!-- DEF Row -->
                    <div class="flex justify-around items-center gap-2 relative z-10">
                        {% for p in starters_by_pos.DEF %}
                        <div class="flex flex-col items-center group transition transform hover:-translate-y-1">
                            <div class="relative">
                                <div class="w-10 h-10 rounded-full bg-blue-600/90 border-2 border-white/80 flex items-center justify-center text-xs font-bold text-white shadow-md">
                                    {{ p.team_code }}
                                </div>
                                {% if p.is_captain %}
                                <span class="absolute -top-1.5 -right-2 bg-amber-400 text-slate-950 text-[10px] font-black px-1 rounded shadow">C</span>
                                {% elif p.is_vice_captain %}
                                <span class="absolute -top-1.5 -right-2 bg-slate-300 text-slate-950 text-[10px] font-black px-1 rounded shadow">V</span>
                                {% endif %}
                            </div>
                            <div class="mt-1.5 bg-slate-950/85 backdrop-blur px-2 py-0.5 rounded border border-white/10 text-center shadow">
                                <div class="text-xs font-semibold text-white leading-tight truncate max-w-[80px]">{{ p.web_name }}</div>
                                <div class="text-[10px] text-emerald-400 font-mono">{{ p.xp }} xP</div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>

                    <!-- MID Row -->
                    <div class="flex justify-around items-center gap-2 relative z-10">
                        {% for p in starters_by_pos.MID %}
                        <div class="flex flex-col items-center group transition transform hover:-translate-y-1">
                            <div class="relative">
                                <div class="w-10 h-10 rounded-full bg-emerald-600/90 border-2 border-white/80 flex items-center justify-center text-xs font-bold text-white shadow-md">
                                    {{ p.team_code }}
                                </div>
                                {% if p.is_captain %}
                                <span class="absolute -top-1.5 -right-2 bg-amber-400 text-slate-950 text-[10px] font-black px-1 rounded shadow">C</span>
                                {% elif p.is_vice_captain %}
                                <span class="absolute -top-1.5 -right-2 bg-slate-300 text-slate-950 text-[10px] font-black px-1 rounded shadow">V</span>
                                {% endif %}
                            </div>
                            <div class="mt-1.5 bg-slate-950/85 backdrop-blur px-2 py-0.5 rounded border border-white/10 text-center shadow">
                                <div class="text-xs font-semibold text-white leading-tight truncate max-w-[80px]">{{ p.web_name }}</div>
                                <div class="text-[10px] text-emerald-400 font-mono">{{ p.xp }} xP</div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>

                    <!-- FWD Row -->
                    <div class="flex justify-around items-center gap-2 relative z-10">
                        {% for p in starters_by_pos.FWD %}
                        <div class="flex flex-col items-center group transition transform hover:-translate-y-1">
                            <div class="relative">
                                <div class="w-10 h-10 rounded-full bg-rose-600/90 border-2 border-white/80 flex items-center justify-center text-xs font-bold text-white shadow-md">
                                    {{ p.team_code }}
                                </div>
                                {% if p.is_captain %}
                                <span class="absolute -top-1.5 -right-2 bg-amber-400 text-slate-950 text-[10px] font-black px-1 rounded shadow">C</span>
                                {% elif p.is_vice_captain %}
                                <span class="absolute -top-1.5 -right-2 bg-slate-300 text-slate-950 text-[10px] font-black px-1 rounded shadow">V</span>
                                {% endif %}
                            </div>
                            <div class="mt-1.5 bg-slate-950/85 backdrop-blur px-2 py-0.5 rounded border border-white/10 text-center shadow">
                                <div class="text-xs font-semibold text-white leading-tight truncate max-w-[80px]">{{ p.web_name }}</div>
                                <div class="text-[10px] text-emerald-400 font-mono">{{ p.xp }} xP</div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>

                </div>

                <!-- Bench Card -->
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                    <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Bench Priority Hierarchy</h3>
                    <div class="grid grid-cols-4 gap-2">
                        {% for p in data.decision.selected_candidate.bench %}
                        <div class="bg-slate-950/60 border border-slate-800 rounded-lg p-2.5 text-center">
                            <div class="text-[10px] font-bold text-slate-500 uppercase">
                                {% if p.bench_order == 0 %}GK Sub{% else %}Sub {{ p.bench_order }}{% endif %}
                            </div>
                            <div class="text-xs font-semibold text-white truncate mt-1">{{ p.web_name }}</div>
                            <div class="text-[11px] text-slate-400">{{ p.position }} · {{ p.team_code }}</div>
                            <div class="text-xs font-mono text-emerald-400 mt-1">{{ p.xp }} xP</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>

            </div>

            <!-- Threat Matrix & Mini-Leagues Column (5 cols) -->
            <div class="lg:col-span-5 space-y-6">
                
                <!-- Dedicated Mini-Leagues Table -->
                {% if mini_leagues %}
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-3">
                    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
                        <h2 class="font-bold text-white flex items-center space-x-2 text-sm">
                            <span>🏆 Private Mini-Leagues</span>
                        </h2>
                        <span class="text-xs text-slate-400 font-mono">{{ mini_leagues|length }} leagues active</span>
                    </div>

                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs text-slate-300">
                            <thead class="text-[10px] uppercase text-slate-400 border-b border-slate-800 bg-slate-950/50">
                                <tr>
                                    <th class="py-2.5 px-2">League</th>
                                    <th class="py-2.5 px-2 text-center">Rank</th>
                                    <th class="py-2.5 px-2 text-center">Trajectory</th>
                                    <th class="py-2.5 px-2 text-right">Size</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-800/60">
                                {% for ml in mini_leagues %}
                                <tr class="hover:bg-slate-800/40 transition">
                                    <td class="py-2.5 px-2">
                                        <div class="font-semibold text-white truncate max-w-[130px]" title="{{ ml.name }}">
                                            {{ ml.name }}
                                        </div>
                                        <div class="text-[10px] text-slate-500">
                                            {% if ml.last_rank %}Prev: #{{ ml.last_rank | comma }}{% else %}Private Classic{% endif %}
                                        </div>
                                    </td>
                                    <td class="py-2.5 px-2 text-center font-mono font-bold text-white">
                                        #{{ ml.rank | comma if ml.rank else '-' }}
                                    </td>
                                    <td class="py-2.5 px-2 text-center">
                                        {% if ml.direction == 'up' %}
                                        <span class="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-950/80 text-emerald-400 border border-emerald-800 shadow-sm" title="Climbed {{ ml.movement }} places">
                                            ▲ +{{ ml.movement | comma }}
                                        </span>
                                        {% elif ml.direction == 'down' %}
                                        <span class="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-bold bg-rose-950/80 text-rose-400 border border-rose-800 shadow-sm" title="Dropped {{ ml.movement | abs }} places">
                                            ▼ {{ (ml.movement | abs) | comma }}
                                        </span>
                                        {% elif ml.direction == 'same' %}
                                        <span class="inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-medium text-slate-400 bg-slate-800" title="Rank unchanged">
                                            ▬ 0
                                        </span>
                                        {% else %}
                                        <span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium text-slate-400 bg-slate-800/60">
                                            NEW
                                        </span>
                                        {% endif %}
                                    </td>
                                    <td class="py-2.5 px-2 text-right font-mono text-slate-400">
                                        {{ ml.total_players | comma if ml.total_players else '-' }}
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                {% endif %}

                <!-- Threat Matrix Tabs -->
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
                    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
                        <h2 class="font-bold text-white flex items-center space-x-2 text-sm">
                            <span>🎯 Tactical Threat Matrix</span>
                        </h2>
                        <span class="text-xs text-slate-400">Mini-League EO%</span>
                    </div>

                    <!-- Shields -->
                    <div class="space-y-2">
                        <div class="flex items-center justify-between text-xs font-semibold text-emerald-400 uppercase tracking-wider">
                            <span>🛡️ Active Shields (Rank Protection)</span>
                            <span class="text-slate-500">Owned High EO</span>
                        </div>
                        {% if data.league_analysis and data.league_analysis.threat_matrix.shields %}
                        <div class="space-y-1.5">
                            {% for p in data.league_analysis.threat_matrix.shields %}
                            <div class="flex items-center justify-between p-2 rounded-lg bg-emerald-950/30 border border-emerald-900/40 text-xs">
                                <div>
                                    <span class="font-bold text-white">{{ p.web_name }}</span>
                                    <span class="text-slate-400">({{ p.team_code }})</span>
                                </div>
                                <span class="font-mono font-bold text-emerald-300 bg-emerald-900/60 px-2 py-0.5 rounded">{{ p.eo_percent }}% EO</span>
                            </div>
                            {% endfor %}
                        </div>
                        {% else %}
                        <div class="text-xs text-slate-500 italic p-2 bg-slate-950/40 rounded border border-slate-800">No active shield assets flagged.</div>
                        {% endif %}
                    </div>

                    <!-- Vulnerabilities -->
                    <div class="space-y-2 pt-2 border-t border-slate-800">
                        <div class="flex items-center justify-between text-xs font-semibold text-rose-400 uppercase tracking-wider">
                            <span>⚠️ Vulnerabilities (Rank Hazards)</span>
                            <span class="text-slate-500">Unowned High EO</span>
                        </div>
                        {% if data.league_analysis and data.league_analysis.threat_matrix.vulnerabilities %}
                        <div class="space-y-1.5">
                            {% for p in data.league_analysis.threat_matrix.vulnerabilities %}
                            <div class="flex items-center justify-between p-2 rounded-lg bg-rose-950/30 border border-rose-900/40 text-xs">
                                <div>
                                    <span class="font-bold text-white">{{ p.web_name }}</span>
                                    <span class="text-slate-400">({{ p.team_code }})</span>
                                </div>
                                <span class="font-mono font-bold text-rose-300 bg-rose-900/60 px-2 py-0.5 rounded">{{ p.eo_percent }}% EO</span>
                            </div>
                            {% endfor %}
                        </div>
                        {% else %}
                        <div class="text-xs text-slate-500 italic p-2 bg-slate-950/40 rounded border border-slate-800">No major unowned hazards detected.</div>
                        {% endif %}
                    </div>

                    <!-- Daggers -->
                    <div class="space-y-2 pt-2 border-t border-slate-800">
                        <div class="flex items-center justify-between text-xs font-semibold text-amber-400 uppercase tracking-wider">
                            <span>🗡️ Differential Daggers (Rank Upside)</span>
                            <span class="text-slate-500">Owned &lt; 25% EO</span>
                        </div>
                        {% if data.league_analysis and data.league_analysis.threat_matrix.daggers %}
                        <div class="space-y-1.5">
                            {% for p in data.league_analysis.threat_matrix.daggers %}
                            <div class="flex items-center justify-between p-2 rounded-lg bg-amber-950/30 border border-amber-900/40 text-xs">
                                <div>
                                    <span class="font-bold text-white">{{ p.web_name }}</span>
                                    <span class="text-slate-400">({{ p.team_code }})</span>
                                </div>
                                <span class="font-mono font-bold text-amber-300 bg-amber-900/60 px-2 py-0.5 rounded">{{ p.eo_percent }}% EO</span>
                            </div>
                            {% endfor %}
                        </div>
                        {% else %}
                        <div class="text-xs text-slate-500 italic p-2 bg-slate-950/40 rounded border border-slate-800">No low-EO differentials currently active.</div>
                        {% endif %}
                    </div>

                </div>

                <!-- Mini-League Rivals Table -->
                {% if data.league_analysis and data.league_analysis.rivals %}
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-3">
                    <div class="flex items-center justify-between">
                        <h2 class="font-bold text-white text-sm">🏆 Mini-League Intel ({{ data.league_analysis.league_name }})</h2>
                        <span class="text-xs text-slate-400">{{ data.league_analysis.total_managers }} rivals</span>
                    </div>

                    <div class="overflow-x-auto max-h-60 overflow-y-auto">
                        <table class="w-full text-left text-xs text-slate-300">
                            <thead class="text-[10px] uppercase text-slate-400 border-b border-slate-800 bg-slate-950/50 sticky top-0">
                                <tr>
                                    <th class="py-2 px-2">Rank</th>
                                    <th class="py-2 px-2">Manager</th>
                                    <th class="py-2 px-2">Captain</th>
                                    <th class="py-2 px-2 text-right">Pts</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-800/60">
                                {% for r in data.league_analysis.rivals %}
                                <tr class="hover:bg-slate-800/40">
                                    <td class="py-2 px-2 font-mono text-slate-400">#{{ r.rank }}</td>
                                    <td class="py-2 px-2">
                                        <div class="font-semibold text-white truncate max-w-[110px]">{{ r.player_name }}</div>
                                        <div class="text-[10px] text-slate-400 truncate max-w-[110px]">{{ r.entry_name }}</div>
                                    </td>
                                    <td class="py-2 px-2 text-amber-300 truncate max-w-[90px]">{{ r.captain_name or '-' }}</td>
                                    <td class="py-2 px-2 text-right font-bold text-white">{{ r.total_points }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                {% endif %}

            </div>

        </div>

        {% endif %}
    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-800/80 bg-slate-900/40 py-6 mt-12 text-center text-xs text-slate-500">
        Autonomous FPL Engine · MILP Solver + AI Director · Raspberry Pi & Docker Headless Edition
    </footer>

</body>
</html>
"""


def fetch_entry_details(team_id: int, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch entry details from the official FPL API: https://fantasy.premierleague.com/api/entry/{team_id}/
    Caches successful responses in memory for 5 minutes.
    """
    now = time.time()
    if team_id in _ENTRY_CACHE:
        cached_time, cached_data = _ENTRY_CACHE[team_id]
        if (now - cached_time) < _ENTRY_CACHE_TTL:
            return cached_data

    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            _ENTRY_CACHE[team_id] = (now, data)
            return data
        logger.warning(f"FPL API returned HTTP {response.status_code} for entry {team_id}")
    except Exception as e:
        logger.warning(f"Failed to fetch entry details for team_id {team_id}: {e}")
    return None


def extract_leagues(
    entry_data: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separate classic leagues into two lists:
    - global_leagues: leagues where league_type == 's' (Overall, Country/Regional, Fan League, 2nd Chance, etc.)
    - mini_leagues: leagues where league_type == 'x' (Private mini-leagues with friends/workmates)

    Computes rank trajectories (Green up-arrow / Red down-arrow comparing entry_rank vs entry_last_rank).
    """
    if not entry_data or not isinstance(entry_data, dict):
        return [], []

    classic_leagues = entry_data.get("leagues", {}).get("classic", [])
    if not classic_leagues and "classic" in entry_data:
        classic_leagues = entry_data.get("classic", [])

    global_leagues: List[Dict[str, Any]] = []
    mini_leagues: List[Dict[str, Any]] = []

    for item in classic_leagues:
        l_type = item.get("league_type", "")
        name = item.get("name", "Unknown League")
        rank = item.get("entry_rank") if item.get("entry_rank") is not None else item.get("rank")
        last_rank = item.get("entry_last_rank")
        total_players = item.get("total_players")

        # Calculate rank movement and direction
        movement = 0
        direction = "same"
        if rank is not None and last_rank is not None and last_rank > 0 and rank > 0:
            movement = last_rank - rank
            if movement > 0:
                direction = "up"
            elif movement < 0:
                direction = "down"
            else:
                direction = "same"
        elif last_rank is None or last_rank == 0:
            direction = "new"

        league_dict = {
            "id": item.get("id"),
            "name": name,
            "league_type": l_type,
            "entry_rank": rank,
            "entry_last_rank": last_rank,
            "rank": rank,
            "last_rank": last_rank,
            "total_players": total_players,
            "movement": movement,
            "direction": direction,
            "raw": item,
        }

        if l_type == "s":
            global_leagues.append(league_dict)
        elif l_type == "x":
            mini_leagues.append(league_dict)

    return global_leagues, mini_leagues


def build_global_summary(
    global_leagues: List[Dict[str, Any]],
    entry_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build quick summary cards for Overall Rank, Country, and Favorite Club.
    """
    overall_league = None
    country_league = None
    club_league = None

    region_name = (
        entry_data.get("player_region_name", "").strip().lower()
        if entry_data
        else ""
    )

    for gl in global_leagues:
        gl_name = gl["name"].strip().lower()
        # Check Overall
        if gl_name == "overall" or gl.get("id") == 275:
            overall_league = gl
        # Check Country
        elif region_name and region_name in gl_name:
            if not country_league:
                country_league = gl
        # Check other leagues as candidate country/club
        elif not country_league and gl_name in [
            "england", "scotland", "wales", "northern ireland", "ireland",
            "united states", "australia", "canada", "norway", "sweden", "india", "nigeria"
        ]:
            country_league = gl

    # Second pass for candidate club league
    for gl in global_leagues:
        if gl == overall_league or gl == country_league:
            continue
        gl_name = gl["name"].strip().lower()
        if "gameweek" not in gl_name and "second chance" not in gl_name and "cup" not in gl_name:
            if not club_league:
                club_league = gl
                break

    # Fallback overall if not explicitly named Overall
    if not overall_league and entry_data and entry_data.get("summary_overall_rank"):
        overall_league = {
            "name": "Overall",
            "rank": entry_data.get("summary_overall_rank"),
            "entry_rank": entry_data.get("summary_overall_rank"),
            "last_rank": None,
            "entry_last_rank": None,
            "total_players": None,
            "movement": 0,
            "direction": "same",
        }

    return {
        "overall": overall_league,
        "country": country_league,
        "club": club_league,
    }


def load_decision_data(data_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load latest decision state strictly from local disk."""
    path = data_path or Path("data/latest_decision.json")
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return {}


def create_app(data_file_path: Optional[Path] = None) -> Flask:
    """Flask application factory."""
    template_dir = Path(__file__).resolve().parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    state_file = data_file_path or Path("data/latest_decision.json")

    # Template filters
    @app.template_filter("comma")
    def comma_filter(val: Any) -> str:
        if val is None or val == "":
            return "-"
        try:
            return f"{int(val):,}"
        except (ValueError, TypeError):
            return str(val)

    @app.template_filter("abs")
    def abs_filter(val: Any) -> Any:
        try:
            return abs(val)
        except (ValueError, TypeError):
            return val

    @app.route("/")
    def index():
        data = load_decision_data(state_file)

        # Categorize starters by position for pitch view
        starters_by_pos = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        if data and "decision" in data and "selected_candidate" in data["decision"]:
            starters = data["decision"]["selected_candidate"].get("starters", [])
            for p in starters:
                pos = p.get("position", "MID")
                if pos in starters_by_pos:
                    starters_by_pos[pos].append(p)

        # Retrieve team entry details and parse leagues
        entry_data: Optional[Dict[str, Any]] = None

        # Check if entry_data already attached to decision state
        if data and "entry_data" in data and isinstance(data["entry_data"], dict):
            entry_data = data["entry_data"]
        elif data and "entry" in data and isinstance(data["entry"], dict):
            entry_data = data["entry"]

        # If not present in file, fetch from FPL API using team_id
        if not entry_data:
            team_id_raw = (
                os.getenv("FPL_TEAM_ID")
                or os.getenv("TEAM_ID")
                or (str(data.get("team_id", "")) if data else "")
                or (str(data.get("entry_id", "")) if data else "")
            )
            if team_id_raw:
                try:
                    team_id = int(str(team_id_raw).strip())
                    entry_data = fetch_entry_details(team_id)
                except (ValueError, TypeError) as exc:
                    logger.debug(f"Invalid team_id '{team_id_raw}': {exc}")

        # Separate classic leagues into global and mini leagues
        global_leagues, mini_leagues = extract_leagues(entry_data)
        global_summary = build_global_summary(global_leagues, entry_data)

        # Render template index.html if available, fallback to render_template_string
        try:
            return render_template(
                "index.html",
                data=data,
                starters_by_pos=starters_by_pos,
                entry_data=entry_data or {},
                global_leagues=global_leagues,
                mini_leagues=mini_leagues,
                global_summary=global_summary,
            )
        except Exception:
            return render_template_string(
                DASHBOARD_HTML,
                data=data,
                starters_by_pos=starters_by_pos,
                entry_data=entry_data or {},
                global_leagues=global_leagues,
                mini_leagues=mini_leagues,
                global_summary=global_summary,
            )

    @app.route("/api/decision")
    def api_decision():
        """REST endpoint returning current decision state."""
        data = load_decision_data(state_file)
        return jsonify(data)

    @app.route("/healthz")
    def health():
        return jsonify({"status": "healthy"})

    return app


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    app = create_app()
    app.run(host=host, port=port, debug=False)

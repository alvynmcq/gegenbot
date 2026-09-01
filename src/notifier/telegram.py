"""Telegram alert dispatcher for pre-deadline lock-ins and post-deadline mini-league intel."""

import logging
import os
from typing import Optional
import requests

from src.agent.director import DecisionOutput
from src.tracker.league_scanner import LeagueAnalysis

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Dispatches markdown notifications to Telegram via direct Bot API HTTP requests."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send message to Telegram chat with error handling and automatic plain-text fallback."""
        if not self.is_configured:
            logger.info("Telegram notifier not configured (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing).")
            logger.info(f"Notification Preview:\n{text}")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info("Telegram notification successfully dispatched.")
            return True
        except Exception as e:
            logger.warning(f"Telegram dispatch with parse_mode={parse_mode} failed ({e}). Retrying with plain text fallback...")
            try:
                payload.pop("parse_mode", None)
                resp = requests.post(url, json=payload, timeout=10)
                resp.raise_for_status()
                logger.info("Telegram notification successfully dispatched (plain text fallback).")
                return True
            except Exception as e2:
                logger.error(f"Failed to dispatch Telegram message: {e2}")
                return False

    def build_pre_deadline_alert(
        self,
        decision: DecisionOutput,
        gameweek: int,
        is_live_execution: bool = False,
    ) -> str:
        """Build formatted Markdown alert for pre-deadline squad lock-in."""
        cand = decision.selected_candidate
        status_tag = "🚀 *[LIVE SQUAD LOCKED IN]*" if is_live_execution else "🔬 *[DRY-RUN PROPOSED LINEUP]*"

        # Format Transfers
        if cand.transfers_count == 0:
            tx_block = "• *Move:* Rolled / Banked free transfer (0 moves)"
        else:
            tx_lines = []
            for t in cand.transfers:
                p_in = t.player_in
                odds_tag = ""
                if p_in.implied_goal_pct and p_in.implied_goal_pct >= 35.0:
                    odds_tag = f" _(🎯 {p_in.implied_goal_pct:.0f}% Goal Prob)_"
                elif p_in.implied_cs_pct and p_in.implied_cs_pct >= 35.0 and p_in.position in ["GKP", "DEF"]:
                    odds_tag = f" _(🛡️ {p_in.implied_cs_pct:.0f}% CS Prob)_"
                tx_lines.append(
                    f"• OUT: *{t.player_out.web_name}* (£{t.player_out.cost_m}m) ➔ IN: *{p_in.web_name}* (£{p_in.cost_m}m){odds_tag}"
                )
            tx_block = "\n".join(tx_lines)

        # Starters list with (C) and (VC)
        starters_lines = []
        for p in cand.starters:
            tag = ""
            if p.is_captain:
                odds_detail = f" _(🎯 {p.implied_goal_pct:.0f}% Goal)_" if p.implied_goal_pct else ""
                tag = f" 👑 *(C)*{odds_detail}"
            elif p.is_vice_captain:
                tag = " 🛡️ *(VC)*"
            elif p.implied_goal_pct and p.implied_goal_pct >= 45.0:
                tag = f" _(🎯 {p.implied_goal_pct:.0f}%)_"
            elif p.implied_cs_pct and p.implied_cs_pct >= 40.0 and p.position in ["GKP", "DEF"]:
                tag = f" _(🛡️ {p.implied_cs_pct:.0f}%)_"
            starters_lines.append(f"• `{p.position}` *{p.web_name}* ({p.team_code}) - {p.xp} xP{tag}")

        # Bench list
        bench_lines = []
        for p in cand.bench:
            order_label = "GK Sub" if p.bench_order == 0 else f"Sub {p.bench_order}"
            bench_lines.append(f"• `{order_label}` *{p.web_name}* ({p.team_code}) - {p.xp} xP")

        active_chip_line = f"⚡ *Active Chip:* `{cand.active_chip.upper()}`\n" if cand.active_chip else ""

        news_block = ""
        if decision.news_alerts:
            # Escape Telegram markdown characters without losing content
            cleaned_alerts = [f"• 🚨 {a.replace('*', '').replace('`', '')}" for a in decision.news_alerts[:3]]
            alerts_str = "\n".join(cleaned_alerts)
            news_block = f"📰 *PRESS CONFERENCE & NEWS ALERTS:*\n{alerts_str}\n\n"

        clean_rationale = decision.rationale.replace("*", "").replace("`", "")

        message = (
            f"{status_tag}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚽ *GAMEWEEK {gameweek} SQUAD BRIEFING*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📋 *TACTICAL MOVE:*\n"
            f"{cand.name}\n"
            f"{active_chip_line}"
            f"{tx_block}\n"
            f"💰 *Bank Remaining:* £{cand.bank_remaining_m:.1f}m | *Hit Cost:* -{cand.hit_cost} pts\n\n"
            f"{news_block}"
            f"🧠 *AI DIRECTOR RATIONALE:*\n"
            f"_{clean_rationale}_\n\n"
            f"👥 *STARTING XI ({cand.formation}):*\n"
            + "\n".join(starters_lines)
            + f"\n\n🪑 *BENCH HIERARCHY:*\n"
            + "\n".join(bench_lines)
            + f"\n\n📊 *PROJECTED SQUAD NET xP:* `{cand.net_xp:.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        return message

    def build_chip_alert(
        self,
        chip_name: str,
        display_name: str,
        projected_xp: float,
        xp_gain: float,
        reason: str,
        gameweek: int,
        is_live_execution: bool = False,
    ) -> str:
        """Build formatted Markdown alert when an automated chip is triggered."""
        status_tag = "⚡ *[LIVE CHIP ACTIVATED]*" if is_live_execution else "💡 *[CHIP THRESHOLD TRIGGERED]*"
        chip_emoji = {
            "wildcard": "🃏",
            "freehit": "🎴",
            "bboost": "🚀",
            "3xc": "👑",
        }.get(chip_name, "⭐")

        message = (
            f"{status_tag}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{chip_emoji} *{display_name.upper()} TRIGGERED (GW {gameweek})*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📈 *Projected Gain:* `+{xp_gain:.2f} xP`\n"
            f"📊 *Total Projected Score:* `{projected_xp:.2f} pts`\n\n"
            f"🎯 *Trigger Rationale:*\n"
            f"{reason}\n\n"
            f"⚙️ *Execution Mode:* `{'LIVE SUBMITTED' if is_live_execution else 'SIMULATION / DRY-RUN'}`\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        return message

    def notify_chip_triggered(
        self,
        chip_name: str,
        display_name: str,
        projected_xp: float,
        xp_gain: float,
        reason: str,
        gameweek: int,
        is_live_execution: bool = False,
    ) -> bool:
        """Send explicit Telegram alert when an automated chip is triggered."""
        text = self.build_chip_alert(
            chip_name=chip_name,
            display_name=display_name,
            projected_xp=projected_xp,
            xp_gain=xp_gain,
            reason=reason,
            gameweek=gameweek,
            is_live_execution=is_live_execution,
        )
        return self.send_message(text)

    def build_post_deadline_briefing(
        self,
        analysis: LeagueAnalysis,
    ) -> str:
        """Build formatted Markdown alert for post-deadline mini-league briefing."""
        # Rival Captain distribution
        captains_list = [f"• *{c_name}*: {count} manager(s)" for c_name, count in analysis.captain_distribution.items()]
        captains_block = "\n".join(captains_list) if captains_list else "• No rival captaincy data."

        # Hits taken
        hits_taken = []
        for r in analysis.rivals:
            if r.event_transfers_cost > 0:
                hits_taken.append(f"• *{r.player_name}* ({r.entry_name}): -{r.event_transfers_cost} pts")
        hits_block = "\n".join(hits_taken) if hits_taken else "• Zero transfer hits taken across rivals."

        # Threat Matrix
        tm = analysis.threat_matrix
        shields_block = "\n".join([f"• 🛡️ *{p.web_name}* ({p.team_code}) - `{p.eo_percent}% EO` (Owned)" for p in tm.shields]) or "• None"
        vuln_block = "\n".join([f"• ⚠️ *{p.web_name}* ({p.team_code}) - `{p.eo_percent}% EO` (Unowned)" for p in tm.vulnerabilities]) or "• None"
        daggers_block = "\n".join([f"• 🗡️ *{p.web_name}* ({p.team_code}) - `{p.eo_percent}% EO` (Differential)" for p in tm.daggers]) or "• None"

        message = (
            f"📡 *[MINI-LEAGUE POST-DEADLINE INTEL]*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🏆 *League:* {analysis.league_name} (GW {analysis.gameweek})\n"
            f"👥 *Managers Scanned:* {analysis.total_managers}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"👑 *RIVAL CAPTAINCY DISTRIBUTION:*\n"
            f"{captains_block}\n\n"
            f"📉 *TRANSFER HITS TAKEN:*\n"
            f"{hits_block}\n\n"
            f"🎯 *TACTICAL THREAT MATRIX:*\n"
            f"*Active Shields (Rank Protection):*\n{shields_block}\n\n"
            f"*Vulnerabilities (Rank Hazards):*\n{vuln_block}\n\n"
            f"*Daggers (Exclusive Differentials):*\n{daggers_block}\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        return message

    def notify_pre_deadline(
        self,
        decision: DecisionOutput,
        gameweek: int,
        is_live_execution: bool = False,
    ) -> bool:
        """Send pre-deadline lock-in alert."""
        text = self.build_pre_deadline_alert(decision, gameweek, is_live_execution)
        return self.send_message(text)

    def notify_post_deadline(
        self,
        analysis: LeagueAnalysis,
    ) -> bool:
        """Send post-deadline mini-league briefing."""
        text = self.build_post_deadline_briefing(analysis)
        return self.send_message(text)

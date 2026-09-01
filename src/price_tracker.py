import os
import requests
from typing import Dict, Any, List, Optional

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

def fetch_price_data() -> Dict[str, Any]:
    res = requests.get(FPL_BOOTSTRAP_URL, timeout=10)
    res.raise_for_status()
    return res.json()


def calculate_price_change_targets(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Calculate price change target percentages and imminent rise/fall status for all players.
    Returns: {player_id: {"target_pct": float, "status": str, "net_transfers": int}}
    Status: "RISE_IMMINENT" (target >= 95%), "FALL_IMMINENT" (target <= -95%), or "STABLE"
    """
    elements = data.get("elements", [])
    results: Dict[int, Dict[str, Any]] = {}

    for p in elements:
        p_id = p.get("id")
        if not p_id:
            continue

        try:
            transfers_in = int(p.get("transfers_in_event", 0) or 0)
            transfers_out = int(p.get("transfers_out_event", 0) or 0)
            net_transfers = transfers_in - transfers_out
            selected_by_pct = float(p.get("selected_by_percent", 0.0) or 0.0)

            # Dynamic threshold scaling: players with high ownership need more net transfers to shift price
            # e.g. for 5% ownership -> ~20k transfers; for 35% ownership -> ~87.5k transfers
            base_threshold = max(20000.0, selected_by_pct * 2500.0)
            target_pct = round((net_transfers / base_threshold) * 100.0, 1)

            if target_pct >= 95.0 or (net_transfers >= 50000 and target_pct >= 80.0):
                status = "RISE_IMMINENT"
            elif target_pct <= -95.0 or (net_transfers <= -50000 and target_pct <= -80.0):
                status = "FALL_IMMINENT"
            else:
                status = "STABLE"

            results[p_id] = {
                "target_pct": target_pct,
                "status": status,
                "net_transfers": net_transfers,
            }
        except (ValueError, TypeError):
            results[p_id] = {
                "target_pct": 0.0,
                "status": "STABLE",
                "net_transfers": 0,
            }

    return results


def analyze_market_movements(data: Dict[str, Any], threshold_transfers: int = 40000) -> Dict[str, List[Dict[str, Any]]]:
    players = data.get("elements", [])
    teams = {t["id"]: t["short_name"] for t in data.get("teams", [])}
    
    risers = []
    fallers = []
    
    for p in players:
        net_transfers = p.get("transfers_in_event", 0) - p.get("transfers_out_event", 0)
        cost = p.get("now_cost", 0) / 10.0
        
        player_summary = {
            "id": p["id"],
            "web_name": p["web_name"],
            "team": teams.get(p["team"], "UNK"),
            "cost": cost,
            "net_transfers": net_transfers,
        }
        
        if net_transfers >= threshold_transfers:
            risers.append(player_summary)
        elif net_transfers <= -threshold_transfers:
            fallers.append(player_summary)
            
    risers.sort(key=lambda x: x["net_transfers"], reverse=True)
    fallers.sort(key=lambda x: x["net_transfers"])
    
    return {
        "likely_rises": risers[:5],
        "likely_falls": fallers[:5]
    }

def format_price_alert_message(insights: Dict[str, List[Dict[str, Any]]]) -> str:
    lines = ["📈 *FPL Daily Price & Market Watch*", ""]
    
    lines.append("🟢 *High Transfer Velocity (Likely Rises)*")
    if insights["likely_rises"]:
        for p in insights["likely_rises"]:
            net_k = f"{p['net_transfers'] // 1000}k"
            lines.append(f"• *{p['web_name']}* ({p['team']}) — £{p['cost']:.1f}m (Net: +{net_k})")
    else:
        lines.append("• _No aggressive upward movers today._")
        
    lines.append("")
    lines.append("🔴 *High Sell-Off Velocity (Likely Falls)*")
    if insights["likely_falls"]:
        for p in insights["likely_falls"]:
            net_k = f"{abs(p['net_transfers']) // 1000}k"
            lines.append(f"• *{p['web_name']}* ({p['team']}) — £{p['cost']:.1f}m (Net: -{net_k})")
    else:
        lines.append("• _No aggressive downward movers today._")
        
    lines.append("\n_Note: Official price updates occur nightly ~01:30 UTC._")
    return "\n".join(lines)

def send_price_update(telegram_token: str, chat_id: str):
    data = fetch_price_data()
    insights = analyze_market_movements(data)
    message = format_price_alert_message(insights)
    
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    res = requests.post(url, json=payload, timeout=10)
    res.raise_for_status()

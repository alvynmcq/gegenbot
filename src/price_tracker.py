import os
import requests
from typing import Dict, Any, List

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

def fetch_price_data() -> Dict[str, Any]:
    res = requests.get(FPL_BOOTSTRAP_URL, timeout=10)
    res.raise_for_status()
    return res.json()

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

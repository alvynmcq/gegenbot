"""Chip eligibility from live FPL inventory, with a conservative history fallback."""
CHIP_NAMES = {"wildcard", "freehit", "bboost", "3xc"}


def available_chips(gameweek, live_chips=None, history=None):
    """Return eligible chip names, or None when availability cannot be established.

    Live inventory describes individual chip instances and their gameweek windows,
    so it takes precedence over season history (which only records chip names).
    An already active chip prevents activating a different chip this gameweek.
    """
    if isinstance(live_chips, list):
        def in_window(chip):
            return chip.get("start_event", 1) <= gameweek <= chip.get("stop_event", 38)

        if any(c.get("status_for_entry") == "active" and in_window(c) for c in live_chips):
            return []
        return sorted({c["name"] for c in live_chips
                       if c.get("name") in CHIP_NAMES and in_window(c)
                       and c.get("status_for_entry") == "available"
                       and not c.get("played_by_entry")})
    if isinstance(history, dict) and isinstance(history.get("chips"), list):
        used = {c.get("name") for c in history["chips"]}
        return sorted(CHIP_NAMES - used)
    return None

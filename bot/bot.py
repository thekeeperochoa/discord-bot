"""
Discord AI Bot - Multi-provider fallback (Groq → Cerebras → Gemini)
Features:
  - Full personality customization
  - Persistent per-channel memory
  - Chime-in mode (spontaneous messages)
  - Passive watching
  - Emoji reactions
  - Silence check-ins
  - Rotating status
  - Image/meme vision (Jordan can see attachments)
  - Daily recap (auto-posts a summary of the day's chaos)
  - Keyword triggers
"""

import discord
import json
import os
import aiohttp
import asyncio
import logging
import random
import time
import re
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict, deque

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("discord-ai-bot")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
CONFIG_FILE = ROOT / "config" / "personality.json"
MEMORY_DIR  = ROOT / "memory"
RECAP_DIR   = ROOT / "memory" / "daily_log"
MEMORY_DIR.mkdir(exist_ok=True)
RECAP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PERSONALITY = {
    "name": "Aria",
    "system_prompt": "You are a friendly Discord community assistant.",
    "temperature": 0.9,
    "max_memory_messages": 30,
    "respond_to_bots": False,
    "trigger_mode": "mention_or_reply",
    "trigger_keywords": [],
    "allowed_channels": [],
    "blocked_users": [],
    "respected_users": [],
    "chime_in_enabled": False,
    "chime_in_min": 40,
    "chime_in_max": 80,
    "watch_mode_enabled": True,
    "reaction_enabled": True,
    "reaction_chance": 0.05,
    "silence_checkin_enabled": True,
    "silence_minutes": 45,
    "status_messages": [
        "watching", "judging quietly", "peaked", "counting typos",
        "3am thoughts", "making bad decisions", "up way too late"
    ],
    "status_rotation_minutes": 10,
    "max_tokens": 200,
    "groq_model":     "llama-3.3-70b-versatile",
    "cerebras_model": "llama3.1-8b",
    "gemini_model":   "gemini-2.5-flash-lite",
    "provider_order": ["groq", "cerebras", "gemini"],
    # Image understanding
    "vision_enabled": True,
    "vision_model": "gemini-2.5-flash",  # gemini supports images best
    # Daily recap
    "daily_recap_enabled": False,
    "daily_recap_hour_utc": 4,   # 4am UTC = midnight EST
    "daily_recap_channel": "",   # channel ID to post the recap in
    "daily_recap_max_tokens": 500,
    # Dedicated channel for ALL bot notifications (events, recaps, tournaments, etc.)
    "notifications_channel": "1303514703464497192",
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULT_PERSONALITY, **json.load(f)}
    return DEFAULT_PERSONALITY.copy()


def get_notification_channel_id(cfg: dict = None) -> str:
    """The single channel ID where all bot notifications/events should post.
    Falls back to daily_recap_channel if notifications_channel is unset."""
    if cfg is None:
        cfg = load_config()
    nc = cfg.get("notifications_channel", "").strip()
    if nc:
        return nc
    return cfg.get("daily_recap_channel", "").strip()


# ── Chime-in counters & silence tracking ─────────────────────────────────────
message_counters: dict[str, int] = defaultdict(int)
chime_thresholds: dict[str, int] = {}
last_bot_activity: dict[str, float] = {}
last_channel_activity: dict[str, float] = {}

def get_chime_threshold(cfg: dict) -> int:
    return random.randint(cfg["chime_in_min"], cfg["chime_in_max"])


# ── Memory ────────────────────────────────────────────────────────────────────
class MemoryStore:
    def __init__(self, max_messages: int = 30):
        self.max_messages = max_messages
        self._cache: dict[str, deque] = {}

    def _path(self, channel_id: str) -> Path:
        return MEMORY_DIR / f"{channel_id}.json"

    def load(self, channel_id: str) -> deque:
        if channel_id in self._cache:
            return self._cache[channel_id]
        p = self._path(channel_id)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            dq = deque(data, maxlen=self.max_messages)
        else:
            dq = deque(maxlen=self.max_messages)
        self._cache[channel_id] = dq
        return dq

    def append(self, channel_id: str, role: str, content: str):
        dq = self.load(channel_id)
        dq.append({"role": role, "content": content})
        with open(self._path(channel_id), "w") as f:
            json.dump(list(dq), f, indent=2)

    def clear(self, channel_id: str):
        self._cache.pop(channel_id, None)
        p = self._path(channel_id)
        if p.exists():
            p.unlink()

    def get_messages(self, channel_id: str) -> list:
        return list(self.load(channel_id))

    def update_maxlen(self, new_max: int):
        self.max_messages = new_max
        for key, dq in self._cache.items():
            self._cache[key] = deque(dq, maxlen=new_max)


memory = MemoryStore()


# ── Daily log (for recap feature) ────────────────────────────────────────────
class DailyLog:
    """Stores ALL messages from the day for the recap. Kept separate so it
    doesn't bloat the conversation memory that gets sent to the AI."""

    def _path(self, channel_id: str, date_str: str) -> Path:
        return RECAP_DIR / f"{channel_id}_{date_str}.jsonl"

    def append(self, channel_id: str, author: str, content: str):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p = self._path(channel_id, date_str)
        line = json.dumps({"author": author, "content": content, "ts": time.time()})
        with open(p, "a") as f:
            f.write(line + "\n")

    def read_day(self, channel_id: str, date_str: str) -> list:
        p = self._path(channel_id, date_str)
        if not p.exists():
            return []
        lines = []
        with open(p) as f:
            for line in f:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    continue
        return lines

    def cleanup_old(self, days_to_keep: int = 7):
        """Delete log files older than N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        for f in RECAP_DIR.glob("*.jsonl"):
            try:
                # Parse date from filename
                date_part = f.stem.split("_")[-1]
                file_date = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if file_date < cutoff:
                    f.unlink()
            except Exception:
                pass


daily_log = DailyLog()


# ── 💰 Economy ────────────────────────────────────────────────────────────────
ECONOMY_FILE = MEMORY_DIR / "economy.json"
ECONOMY_LOCK = asyncio.Lock()

# Default starting balance
STARTING_BALANCE = 500
# Daily reward amount range
DAILY_MIN, DAILY_MAX = 200, 500
# Weekly reward
WEEKLY_MIN, WEEKLY_MAX = 1500, 3500
# Work command range
WORK_MIN, WORK_MAX = 50, 250
# Rob command - chance and limits
ROB_SUCCESS_CHANCE = 0.45
ROB_MIN_TARGET_BALANCE = 200
ROB_PERCENT_MIN, ROB_PERCENT_MAX = 0.05, 0.25  # 5-25% of victim's balance
ROB_FINE_MIN, ROB_FINE_MAX = 100, 400
# Cooldowns in seconds
COOLDOWNS = {
    "daily":  24 * 3600,
    "weekly": 7 * 24 * 3600,
    "work":   45 * 60,
    "rob":    2 * 3600,
    "beg":    10 * 60,
    "pay":    60,  # 1 min between payments to prevent rapid alt-account farming
    "crime":  2 * 3600,  # separate from rob for clarity
    "wheel":  45 * 60,   # separate from work
}


class Economy:
    """JSON-backed economy. Tracks balance and per-command cooldowns per user."""
    def __init__(self):
        self._data: dict = {"users": {}}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if ECONOMY_FILE.exists():
            try:
                with open(ECONOMY_FILE) as f:
                    self._data = json.load(f)
                if "users" not in self._data:
                    self._data["users"] = {}
            except Exception as e:
                log.warning("Economy load failed: %s", e)
                self._data = {"users": {}}
        self._loaded = True

    def _save(self):
        try:
            with open(ECONOMY_FILE, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            log.warning("Economy save failed: %s", e)

    def _user(self, user_id: int) -> dict:
        self._load()
        uid = str(user_id)
        if uid not in self._data["users"]:
            self._data["users"][uid] = {
                "balance": STARTING_BALANCE,
                "cooldowns": {},
                "stats": {"games_won": 0, "games_lost": 0, "total_earned": 0, "total_lost": 0},
            }
        return self._data["users"][uid]

    def balance(self, user_id: int) -> int:
        return self._user(user_id)["balance"]

    def add(self, user_id: int, amount: int, reason: str = "") -> int:
        u = self._user(user_id)
        u["balance"] += amount
        if amount > 0:
            u["stats"]["total_earned"] = u["stats"].get("total_earned", 0) + amount
        else:
            u["stats"]["total_lost"] = u["stats"].get("total_lost", 0) + abs(amount)
        if u["balance"] < 0:
            u["balance"] = 0
        self._save()
        return u["balance"]

    def transfer(self, from_id: int, to_id: int, amount: int) -> bool:
        if amount <= 0:
            return False
        sender = self._user(from_id)
        if sender["balance"] < amount:
            return False
        sender["balance"] -= amount
        receiver = self._user(to_id)
        receiver["balance"] += amount
        self._save()
        return True

    def record_win(self, user_id: int, game: str = "generic", channel=None):
        u = self._user(user_id)
        u["stats"]["games_won"] = u["stats"].get("games_won", 0) + 1
        self._save()
        # Schedule achievement checks asynchronously
        try:
            asyncio.create_task(trigger_game_win(user_id, game, channel))
            asyncio.create_task(trigger_balance_check(user_id, channel))
        except RuntimeError:
            pass  # No running loop (e.g. during startup)

    def record_loss(self, user_id: int, game: str = "generic", channel=None):
        u = self._user(user_id)
        u["stats"]["games_lost"] = u["stats"].get("games_lost", 0) + 1
        self._save()
        try:
            asyncio.create_task(trigger_game_loss(user_id, game, channel))
            asyncio.create_task(trigger_balance_check(user_id, channel))
        except RuntimeError:
            pass

    def stats(self, user_id: int) -> dict:
        return self._user(user_id)["stats"].copy()

    def get_cooldown_remaining(self, user_id: int, key: str) -> int:
        """Returns 0 if usable, else seconds remaining."""
        u = self._user(user_id)
        last = u["cooldowns"].get(key, 0)
        cd = COOLDOWNS.get(key, 0)
        remaining = int((last + cd) - time.time())
        return max(remaining, 0)

    def set_cooldown(self, user_id: int, key: str):
        u = self._user(user_id)
        u["cooldowns"][key] = time.time()
        self._save()

    def leaderboard(self, top_n: int = 10) -> list:
        self._load()
        users = self._data["users"]
        ranked = sorted(
            users.items(),
            key=lambda x: x[1].get("balance", 0),
            reverse=True,
        )
        return ranked[:top_n]


economy = Economy()


# ─────────────────────────────────────────────────────────────────────────────
# 📊 STATS TRACKING (for web dashboard)
# Logs command uses, economy events, game outcomes, feature usage, activity feed.
# All persisted to MEMORY_DIR/stats.json.
# ─────────────────────────────────────────────────────────────────────────────
STATS_FILE = MEMORY_DIR / "stats.json"
STATS_MAX_FEED = 200  # keep last N events in activity feed
STATS_KEEP_DAYS = 60  # retention window


def _load_stats() -> dict:
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "command_uses": {},          # cmd -> total count
        "command_uses_today": {},    # cmd -> {date: count}
        "economy_events_today": {},  # date -> {earned, spent, transferred, transactions}
        "game_outcomes": {},         # game -> {wins, losses, total_payout, biggest_payout}
        "active_users_today": {},    # date -> [user_ids]
        "new_users": {},             # date -> [user_ids]
        "feature_usage": {},         # feature -> count
        "activity_feed": [],         # list of dicts (newest first)
    }


def _save_stats(data: dict):
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.warning("save stats failed: %s", e)


def _stats_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _stats_trim_old(stats: dict):
    """Drop data older than STATS_KEEP_DAYS."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=STATS_KEEP_DAYS)).strftime("%Y-%m-%d")
    for cmd in list(stats.get("command_uses_today", {}).keys()):
        stats["command_uses_today"][cmd] = {
            d: v for d, v in stats["command_uses_today"][cmd].items() if d >= cutoff
        }
        if not stats["command_uses_today"][cmd]:
            del stats["command_uses_today"][cmd]
    for key in ("economy_events_today", "active_users_today", "new_users"):
        stats[key] = {d: v for d, v in stats.get(key, {}).items() if d >= cutoff}


def track_command_use(command_name: str, user_id: int):
    """Called automatically on every successful slash command."""
    try:
        stats = _load_stats()
        today = _stats_today()
        stats["command_uses"][command_name] = stats["command_uses"].get(command_name, 0) + 1
        if command_name not in stats["command_uses_today"]:
            stats["command_uses_today"][command_name] = {}
        stats["command_uses_today"][command_name][today] = (
            stats["command_uses_today"][command_name].get(today, 0) + 1
        )
        # Active users tracking
        if today not in stats["active_users_today"]:
            stats["active_users_today"][today] = []
        uid = str(user_id)
        if uid not in stats["active_users_today"][today]:
            stats["active_users_today"][today].append(uid)
            # New user = never seen on any other day
            seen_before = any(uid in users for d, users in stats["active_users_today"].items() if d != today)
            if not seen_before:
                stats["new_users"].setdefault(today, []).append(uid)
        _stats_trim_old(stats)
        _save_stats(stats)
    except Exception as e:
        log.warning("track_command_use: %s", e)


def track_economy_event(kind: str, amount: int):
    """kind in {'earned','spent','transferred'}."""
    try:
        stats = _load_stats()
        today = _stats_today()
        if today not in stats["economy_events_today"]:
            stats["economy_events_today"][today] = {"earned": 0, "spent": 0, "transferred": 0, "transactions": 0}
        if kind in ("earned", "spent", "transferred"):
            stats["economy_events_today"][today][kind] += abs(amount)
        stats["economy_events_today"][today]["transactions"] += 1
        _save_stats(stats)
    except Exception as e:
        log.warning("track_economy_event: %s", e)


def track_game_outcome(game_name: str, won: bool, payout: int = 0):
    try:
        stats = _load_stats()
        if game_name not in stats["game_outcomes"]:
            stats["game_outcomes"][game_name] = {"wins": 0, "losses": 0, "total_payout": 0, "biggest_payout": 0}
        g = stats["game_outcomes"][game_name]
        if won:
            g["wins"] += 1
        else:
            g["losses"] += 1
        g["total_payout"] += payout
        if payout > g.get("biggest_payout", 0):
            g["biggest_payout"] = payout
        _save_stats(stats)
    except Exception as e:
        log.warning("track_game_outcome: %s", e)


def track_feature_use(feature: str):
    """feature in {'pet','business','venue','dealer','heist','nightlife','shop'}"""
    try:
        stats = _load_stats()
        stats["feature_usage"][feature] = stats["feature_usage"].get(feature, 0) + 1
        _save_stats(stats)
    except Exception as e:
        log.warning("track_feature_use: %s", e)


def track_activity(event_type: str, user_id: int, user_name: str, detail: str):
    """Add to live activity feed (last 200)."""
    try:
        stats = _load_stats()
        stats["activity_feed"].insert(0, {
            "ts": int(time.time()),
            "type": event_type,
            "user_id": str(user_id),
            "user_name": user_name,
            "detail": detail,
        })
        stats["activity_feed"] = stats["activity_feed"][:STATS_MAX_FEED]
        _save_stats(stats)
    except Exception as e:
        log.warning("track_activity: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# 📅 PERSISTENT SCHEDULER STATE
# Tracks which date each scheduler last fired, so they don't double-fire
# after bot restarts (Railway redeploys, crashes, etc.)
# ─────────────────────────────────────────────────────────────────────────────
SCHEDULER_STATE_FILE = MEMORY_DIR / "scheduler_state.json"


def _load_scheduler_state() -> dict:
    if SCHEDULER_STATE_FILE.exists():
        try:
            with open(SCHEDULER_STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_scheduler_state(data: dict):
    try:
        with open(SCHEDULER_STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.warning("Failed to save scheduler state: %s", e)


def get_last_fire(scheduler_name: str) -> str:
    """Returns the last YYYY-MM-DD this scheduler fired (or '')."""
    state = _load_scheduler_state()
    return state.get(scheduler_name, "")


def set_last_fire(scheduler_name: str, date_str: str):
    state = _load_scheduler_state()
    state[scheduler_name] = date_str
    _save_scheduler_state(state)


# ─────────────────────────────────────────────────────────────────────────────
# 🏆 ACHIEVEMENTS SYSTEM
# Visible badges, one-time coin rewards, permanent perks.
# Stored alongside economy data (in the user record).
# ─────────────────────────────────────────────────────────────────────────────

# Each achievement: id -> {emoji, name, description, reward, perk?, perk_value?, hidden?}
# Perks (applied automatically by the economy/cmd code):
#   "daily_bonus_pct"   : extra % on /daily payouts
#   "rob_protection_pct": % chance to nullify a rob attempt
#   "slots_luck_pct"    : extra % win chance on slots pairs
#   "blackjack_payout_pct": extra % multiplier on blackjack wins
#   "fight_armor_pct"   : reduce fight loss by this %
#   "lottery_discount_pct": % off lottery tickets
#   "work_bonus_pct"    : extra % on /work payouts
#   "wheel_luck_pct"    : extra weight on positive wheel outcomes
#   "passive_income"    : flat coins added per /daily claim
ACHIEVEMENTS: dict[str, dict] = {
    # ── Economy milestones ──
    "first_grand": {
        "emoji": "💵", "name": "First Grand",
        "description": "Reach 1,000 coins.",
        "reward": 100,
    },
    "ten_grand": {
        "emoji": "💰", "name": "Stacks",
        "description": "Reach 10,000 coins.",
        "reward": 500,
        "perk": "daily_bonus_pct", "perk_value": 5,
    },
    "hundred_grand": {
        "emoji": "💎", "name": "Hundred Stacks",
        "description": "Reach 100,000 coins.",
        "reward": 2_500,
        "perk": "daily_bonus_pct", "perk_value": 10,
    },
    "millionaire": {
        "emoji": "👑", "name": "Millionaire",
        "description": "Reach 1,000,000 coins.",
        "reward": 25_000,
        "perk": "passive_income", "perk_value": 100,
    },

    # ── Daily/work grinder ──
    "first_daily": {
        "emoji": "📦", "name": "Day One",
        "description": "Claim your first /daily.",
        "reward": 50,
    },
    "daily_streak_7": {
        "emoji": "🔥", "name": "Week Streak",
        "description": "Claim /daily 7 days in a row.",
        "reward": 500,
        "perk": "daily_bonus_pct", "perk_value": 5,
    },
    "workhorse": {
        "emoji": "🛠️", "name": "Workhorse",
        "description": "Use /work 25 times.",
        "reward": 300,
        "perk": "work_bonus_pct", "perk_value": 10,
    },

    # ── Gambling ──
    "first_jackpot": {
        "emoji": "🎰", "name": "Lucky",
        "description": "Hit a slots jackpot.",
        "reward": 250,
        "perk": "slots_luck_pct", "perk_value": 5,
    },
    "diamond_hands": {
        "emoji": "💠", "name": "Diamond Hands",
        "description": "Hit triple diamonds on slots.",
        "reward": 1_000,
    },
    "lottery_winner": {
        "emoji": "🎟️", "name": "Jackpot Royalty",
        "description": "Win the lottery.",
        "reward": 500,
        "perk": "lottery_discount_pct", "perk_value": 10,
    },
    "blackjack_natural": {
        "emoji": "🃏", "name": "Natural",
        "description": "Get a natural 21 in blackjack.",
        "reward": 300,
        "perk": "blackjack_payout_pct", "perk_value": 5,
    },

    # ── Combat / PvP ──
    "first_blood": {
        "emoji": "🩸", "name": "First Blood",
        "description": "Win your first /duel.",
        "reward": 100,
    },
    "fight_champ": {
        "emoji": "🏆", "name": "Champion",
        "description": "Win 10 /fight matches.",
        "reward": 1_500,
        "perk": "fight_armor_pct", "perk_value": 15,
    },
    "shootout_winner": {
        "emoji": "🚪", "name": "Last One Standing",
        "description": "Win a /shootout.",
        "reward": 500,
    },
    "russian_winner": {
        "emoji": "🎯", "name": "Iron Nerve",
        "description": "Survive Russian roulette /gun.",
        "reward": 400,
    },

    # ── Crime ──
    "successful_thief": {
        "emoji": "🦝", "name": "Pickpocket Pro",
        "description": "Successfully /rob 10 users.",
        "reward": 750,
        "perk": "rob_protection_pct", "perk_value": 10,
    },
    "victim": {
        "emoji": "💸", "name": "Easy Target",
        "description": "Get robbed 5 times.",
        "reward": 200,
        "perk": "rob_protection_pct", "perk_value": 15,  # consolation perk
    },
    "heist_legend": {
        "emoji": "🚐", "name": "Heist Legend",
        "description": "Pull off 5 successful heists.",
        "reward": 800,
    },
    "criminal_mind": {
        "emoji": "🦹", "name": "Career Criminal",
        "description": "Commit 25 successful crimes.",
        "reward": 600,
    },

    # ── Social / Drama ──
    "married": {
        "emoji": "💍", "name": "Wedded",
        "description": "Get married.",
        "reward": 300,
        "perk": "passive_income", "perk_value": 25,
    },
    "divorced": {
        "emoji": "💔", "name": "Heartbreaker",
        "description": "Get divorced.",
        "reward": 100,
    },
    "court_winner": {
        "emoji": "⚖️", "name": "Litigator",
        "description": "Win 3 lawsuits.",
        "reward": 500,
    },

    # ── Misc / Vanity ──
    "fortune_teller": {
        "emoji": "🔮", "name": "Mystic",
        "description": "Get 10 tarot readings.",
        "reward": 200,
    },
    "bomb_passer": {
        "emoji": "💣", "name": "Hot Hands",
        "description": "Successfully pass a /bomb 5 times before it explodes.",
        "reward": 300,
    },
    "wheel_jackpot": {
        "emoji": "🎡", "name": "Big Wheel",
        "description": "Hit the /wheel jackpot (+5,000).",
        "reward": 500,
        "perk": "wheel_luck_pct", "perk_value": 10,
    },
    "completionist": {
        "emoji": "🌟", "name": "Completionist",
        "description": "Earn 20 other achievements.",
        "reward": 5_000,
        "perk": "passive_income", "perk_value": 50,
    },
}


def _get_achievements(user_id: int) -> list:
    """Return list of earned achievement IDs for a user."""
    u = economy._user(user_id)
    return u.setdefault("achievements", [])


def _get_counters(user_id: int) -> dict:
    """Return per-user counters (used for milestone tracking like 'won 10 fights')."""
    u = economy._user(user_id)
    return u.setdefault("counters", {})


def _bump_counter(user_id: int, key: str, by: int = 1) -> int:
    counters = _get_counters(user_id)
    counters[key] = counters.get(key, 0) + by
    economy._save()
    return counters[key]


def _set_counter(user_id: int, key: str, value):
    counters = _get_counters(user_id)
    counters[key] = value
    economy._save()


def _user_has_achievement(user_id: int, ach_id: str) -> bool:
    return ach_id in _get_achievements(user_id)


async def _grant_achievement(user_id: int, ach_id: str, channel=None):
    """Award an achievement (one-time). Pays reward, applies perk, announces in channel if given."""
    if ach_id not in ACHIEVEMENTS:
        return False
    if _user_has_achievement(user_id, ach_id):
        return False
    ach = ACHIEVEMENTS[ach_id]
    earned = _get_achievements(user_id)
    earned.append(ach_id)

    # Pay reward
    reward = ach.get("reward", 0)
    if reward > 0:
        economy.add(user_id, reward, f"achievement: {ach_id}")

    # Apply perk
    perk_key = ach.get("perk")
    perk_value = ach.get("perk_value", 0)
    if perk_key:
        u = economy._user(user_id)
        perks = u.setdefault("perks", {})
        # Stack additively
        perks[perk_key] = perks.get(perk_key, 0) + perk_value
    economy._save()

    # Announce
    if channel:
        try:
            # Reduce spam: only flashy embed for achievements with rewards >= 500 or perks
            is_major = ach.get("reward", 0) >= 500 or ach.get("perk")
            if is_major:
                embed = discord.Embed(
                    title="🏆 ACHIEVEMENT UNLOCKED!",
                    description=f"{ach['emoji']} **{ach['name']}**\n_{ach['description']}_",
                    color=discord.Color.gold(),
                )
                embed.add_field(name="💰 Reward", value=f"+{reward:,} coins", inline=True)
                if perk_key:
                    embed.add_field(
                        name="✨ Perk Unlocked",
                        value=f"+{perk_value}% {_perk_label(perk_key)}",
                        inline=True,
                    )
                embed.add_field(name="🏆 User", value=f"<@{user_id}>", inline=False)
                await channel.send(
                    embed=embed,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            else:
                # Quick text-only announcement for minor achievements
                await channel.send(
                    f"🏆 <@{user_id}> earned **{ach['emoji']} {ach['name']}** (+{reward:,} coins)",
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        except Exception:
            pass

    # Check completionist trigger
    if ach_id != "completionist":
        if len(earned) >= 20 and not _user_has_achievement(user_id, "completionist"):
            await _grant_achievement(user_id, "completionist", channel=channel)
    return True


def _perk_label(key: str) -> str:
    return {
        "daily_bonus_pct":      "daily bonus",
        "rob_protection_pct":   "rob protection chance",
        "slots_luck_pct":       "slots luck",
        "blackjack_payout_pct": "blackjack payout",
        "fight_armor_pct":      "fight armor (less loss)",
        "lottery_discount_pct": "lottery discount",
        "work_bonus_pct":       "work bonus",
        "wheel_luck_pct":       "wheel luck",
        "passive_income":       "coins per daily",
    }.get(key, key)


def get_perk(user_id: int, key: str) -> int:
    """Return the user's stacked perk value for a given key (or 0)."""
    u = economy._user(user_id)
    return u.get("perks", {}).get(key, 0)


def get_user_badges(user_id: int) -> str:
    """Return a string of emoji badges the user has earned."""
    earned = _get_achievements(user_id)
    badges = []
    # VIP badge first if active
    if is_vip(user_id):
        badges.append("💎")
    for ach_id in earned:
        if ach_id in ACHIEVEMENTS:
            badges.append(ACHIEVEMENTS[ach_id]["emoji"])
    return "".join(badges)


def get_user_name_display(user_id: int, display_name: str) -> str:
    """Format display name with custom title if set."""
    title = get_custom_title(user_id)
    if title:
        return f"{display_name} [{title}]"
    return display_name


# ── Trigger functions — called by other commands ─────────────────────────────
async def trigger_balance_check(user_id: int, channel=None):
    """Run balance-based achievement checks."""
    bal = economy.balance(user_id)
    if bal >= 1_000:
        await _grant_achievement(user_id, "first_grand", channel)
    if bal >= 10_000:
        await _grant_achievement(user_id, "ten_grand", channel)
    if bal >= 100_000:
        await _grant_achievement(user_id, "hundred_grand", channel)
    if bal >= 1_000_000:
        await _grant_achievement(user_id, "millionaire", channel)


async def trigger_daily_claim(user_id: int, channel=None):
    """Track daily streaks. Returns nothing — just side-effects."""
    counters = _get_counters(user_id)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    last_claim = counters.get("last_daily_date", "")
    if last_claim == today:
        return  # already counted today
    if last_claim == yesterday:
        streak = counters.get("daily_streak", 0) + 1
    else:
        streak = 1
    counters["daily_streak"] = streak
    counters["last_daily_date"] = today
    economy._save()

    # First daily
    await _grant_achievement(user_id, "first_daily", channel)
    if streak >= 7:
        await _grant_achievement(user_id, "daily_streak_7", channel)


async def trigger_work_used(user_id: int, channel=None):
    count = _bump_counter(user_id, "work_count")
    if count >= 25:
        await _grant_achievement(user_id, "workhorse", channel)


async def trigger_game_win(user_id: int, game: str, channel=None):
    """Generic per-game win counter."""
    count = _bump_counter(user_id, f"{game}_wins")
    # Track quest progress
    try:
        track_quest_progress(user_id, "games_won")
        track_quest_progress(user_id, "games_played")
        add_tournament_score(user_id, games_won=1)
    except Exception:
        pass
    if game == "duel" and count >= 1:
        await _grant_achievement(user_id, "first_blood", channel)
    elif game == "fight" and count >= 10:
        await _grant_achievement(user_id, "fight_champ", channel)
    elif game == "shootout" and count >= 1:
        await _grant_achievement(user_id, "shootout_winner", channel)
    elif game == "gun" and count >= 1:
        await _grant_achievement(user_id, "russian_winner", channel)
    elif game == "heist" and count >= 5:
        await _grant_achievement(user_id, "heist_legend", channel)
    elif game == "crime" and count >= 25:
        await _grant_achievement(user_id, "criminal_mind", channel)
    elif game == "rob" and count >= 10:
        await _grant_achievement(user_id, "successful_thief", channel)
    elif game == "lawsuit" and count >= 3:
        await _grant_achievement(user_id, "court_winner", channel)


async def trigger_event(user_id: int, event: str, channel=None):
    """Special one-off events."""
    if event == "slots_jackpot":
        await _grant_achievement(user_id, "first_jackpot", channel)
    elif event == "slots_diamonds":
        await _grant_achievement(user_id, "diamond_hands", channel)
    elif event == "blackjack_natural":
        await _grant_achievement(user_id, "blackjack_natural", channel)
    elif event == "lottery_won":
        await _grant_achievement(user_id, "lottery_winner", channel)
    elif event == "got_robbed":
        c = _bump_counter(user_id, "robbed_count")
        if c >= 5:
            await _grant_achievement(user_id, "victim", channel)
    elif event == "married":
        await _grant_achievement(user_id, "married", channel)
    elif event == "divorced":
        await _grant_achievement(user_id, "divorced", channel)
    elif event == "tarot_read":
        c = _bump_counter(user_id, "tarot_count")
        if c >= 10:
            await _grant_achievement(user_id, "fortune_teller", channel)
    elif event == "bomb_passed":
        c = _bump_counter(user_id, "bomb_passes")
        if c >= 5:
            await _grant_achievement(user_id, "bomb_passer", channel)
    elif event == "wheel_jackpot":
        await _grant_achievement(user_id, "wheel_jackpot", channel)


# ─────────────────────────────────────────────────────────────────────────────
# 📊 XP / LEVELING SYSTEM
# Users earn XP for chatting and using commands.
# Levels unlock perks and confer prestige.
# ─────────────────────────────────────────────────────────────────────────────

XP_PER_MESSAGE = 5
XP_PER_COMMAND = 15
XP_MESSAGE_COOLDOWN = 60  # seconds — prevents spam farming

# Track last message XP timestamp per user (in-memory, fine to lose on restart)
_xp_last_message: dict[int, float] = {}


def xp_for_level(level: int) -> int:
    """Total XP needed to reach `level`. Quadratic curve."""
    return 100 * level * level


def level_for_xp(xp: int) -> int:
    """What level a given total XP corresponds to."""
    if xp < 100:
        return 0
    # Solve 100 * L^2 <= xp → L = sqrt(xp / 100)
    import math
    return int(math.sqrt(xp / 100))


def _get_xp(user_id: int) -> int:
    return economy._user(user_id).get("xp", 0)


def add_xp(user_id: int, amount: int) -> tuple[int, int, bool]:
    """Add XP. Returns (new_xp, new_level, leveled_up)."""
    u = economy._user(user_id)
    old_xp = u.get("xp", 0)
    old_level = level_for_xp(old_xp)
    new_xp = old_xp + amount
    u["xp"] = new_xp
    economy._save()
    new_level = level_for_xp(new_xp)
    return new_xp, new_level, new_level > old_level


async def grant_xp(user_id: int, amount: int, channel=None):
    """Grant XP and announce level-up if it happens."""
    # Apply XP boost item
    amount = amount * xp_multiplier(user_id)
    new_xp, new_level, leveled = add_xp(user_id, amount)
    if leveled and channel:
        # Award level-up bonus + apply level-based perks
        bonus = 100 * new_level
        economy.add(user_id, bonus, f"level {new_level}")
        try:
            embed = discord.Embed(
                title="📊 LEVEL UP!",
                description=f"<@{user_id}> reached **Level {new_level}**!",
                color=discord.Color.green(),
            )
            embed.add_field(name="💰 Reward", value=f"+{bonus:,} coins", inline=True)
            # Milestone perks at major levels
            milestone_perks = {
                5:  ("daily_bonus_pct", 5,  "Apprentice"),
                10: ("work_bonus_pct", 10,  "Hustler"),
                15: ("slots_luck_pct", 5,   "Gambler"),
                20: ("rob_protection_pct", 10, "Veteran"),
                25: ("blackjack_payout_pct", 5, "Card Shark"),
                30: ("fight_armor_pct", 10, "Brawler"),
                40: ("daily_bonus_pct", 10, "Elite"),
                50: ("passive_income", 100, "Legend"),
            }
            if new_level in milestone_perks:
                pkey, pval, title_name = milestone_perks[new_level]
                u = economy._user(user_id)
                perks = u.setdefault("perks", {})
                perks[pkey] = perks.get(pkey, 0) + pval
                economy._save()
                embed.add_field(
                    name="🏅 Title Unlocked",
                    value=f"**{title_name}**",
                    inline=True,
                )
                embed.add_field(
                    name="✨ Perk",
                    value=f"+{pval}% {_perk_label(pkey)}",
                    inline=False,
                )
            await channel.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
        except Exception:
            pass


def get_user_title(user_id: int) -> str:
    """Returns the user's level-based title."""
    level = level_for_xp(_get_xp(user_id))
    titles = {
        0: "Newbie", 5: "Apprentice", 10: "Hustler", 15: "Gambler",
        20: "Veteran", 25: "Card Shark", 30: "Brawler", 40: "Elite", 50: "Legend",
    }
    # Find the highest title they qualify for
    title = "Newbie"
    for lvl, name in sorted(titles.items()):
        if level >= lvl:
            title = name
    return title


# ─────────────────────────────────────────────────────────────────────────────
# 🔥 LOGIN STREAKS (extends existing /daily streak tracking)
# 7/14/30/100 day streaks unlock huge bonuses.
# ─────────────────────────────────────────────────────────────────────────────
STREAK_BONUSES = {
    7:   ("Week Warrior",     1_000),
    14:  ("Two-Week Grinder", 2_500),
    30:  ("Monthly Loyalty",  10_000),
    100: ("Centurion",        50_000),
}


async def check_streak_bonus(user_id: int, channel=None):
    """Awards streak bonuses at 7/14/30/100. Called by /daily."""
    counters = _get_counters(user_id)
    streak = counters.get("daily_streak", 0)
    awarded = counters.setdefault("streak_bonuses_awarded", [])
    for milestone, (name, bonus) in STREAK_BONUSES.items():
        if streak >= milestone and milestone not in awarded:
            awarded.append(milestone)
            economy.add(user_id, bonus, f"streak {milestone}")
            economy._save()
            if channel:
                try:
                    embed = discord.Embed(
                        title=f"🔥 {milestone}-DAY STREAK!",
                        description=f"<@{user_id}> hit a **{milestone}-day** streak!",
                        color=discord.Color.orange(),
                    )
                    embed.add_field(name="🏆 Title", value=name, inline=True)
                    embed.add_field(name="💰 Bonus", value=f"+{bonus:,} coins", inline=True)
                    await channel.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# 🏆 WEEKLY TOURNAMENTS
# Separate weekly leaderboard. Resets every Monday at recap hour.
# Top 3 win prize coins.
# ─────────────────────────────────────────────────────────────────────────────
TOURNAMENT_FILE = MEMORY_DIR / "tournament.json"
TOURNAMENT_PRIZES = [10_000, 5_000, 2_000]  # 1st, 2nd, 3rd


def _load_tournament() -> dict:
    if TOURNAMENT_FILE.exists():
        try:
            with open(TOURNAMENT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "season_start": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "scores": {},      # user_id -> {coins_earned, games_won, commands_used}
        "history": [],
    }


def _save_tournament(data: dict):
    with open(TOURNAMENT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_tournament_score(user_id: int, coins_earned: int = 0, games_won: int = 0, commands_used: int = 0):
    data = _load_tournament()
    uid = str(user_id)
    if uid not in data["scores"]:
        data["scores"][uid] = {"coins_earned": 0, "games_won": 0, "commands_used": 0}
    data["scores"][uid]["coins_earned"] += max(coins_earned, 0)
    data["scores"][uid]["games_won"] += games_won
    data["scores"][uid]["commands_used"] += commands_used
    _save_tournament(data)


def tournament_score(stats: dict) -> int:
    """Composite tournament score: coins earned + heavy bonus for game wins + small bonus for commands."""
    return stats["coins_earned"] + stats["games_won"] * 500 + stats["commands_used"] * 10


async def tournament_scheduler():
    """Background task: every Monday at recap hour, distribute prizes and reset."""
    await client.wait_until_ready()
    while not client.is_closed():
        await asyncio.sleep(600)  # check every 10 min
        cfg = load_config()
        target_hour = cfg.get("daily_recap_hour_utc", 4)
        now = datetime.now(timezone.utc)
        # Reset on Monday at the configured hour
        if now.weekday() != 0 or now.hour != target_hour:
            continue
        today_str = now.strftime("%Y-%m-%d")
        if get_last_fire("tournament") == today_str:
            continue
        # Mark fired FIRST to prevent races
        set_last_fire("tournament", today_str)

        try:
            data = _load_tournament()
            if not data["scores"]:
                # Reset season anyway
                data["season_start"] = today_str
                _save_tournament(data)
                continue

            ranked = sorted(
                data["scores"].items(),
                key=lambda x: tournament_score(x[1]),
                reverse=True,
            )

            # Award prizes
            announcements = []
            medals = ["🥇", "🥈", "🥉"]
            for i, (uid, stats) in enumerate(ranked[:3]):
                prize = TOURNAMENT_PRIZES[i]
                economy.add(int(uid), prize, f"tournament rank {i+1}")
                announcements.append(
                    f"{medals[i]} <@{uid}> — **{prize:,}** coins (score: {tournament_score(stats):,})"
                )
                # DM the winner
                try:
                    dm_embed = discord.Embed(
                        title=f"{medals[i]} TOURNAMENT WINNER!",
                        description=(
                            f"You placed **#{i+1}** in this week's tournament!\n"
                            f"💰 Prize: **{prize:,}** coins\n"
                            f"🏆 Score: **{tournament_score(stats):,}**"
                        ),
                        color=discord.Color.gold(),
                    )
                    await send_dm(int(uid), "tournament", embed=dm_embed)
                except Exception:
                    pass

            # DM all other participants with their rank
            for i, (uid, stats) in enumerate(ranked[3:], start=4):
                try:
                    dm_embed = discord.Embed(
                        title="🏆 TOURNAMENT RESULTS",
                        description=(
                            f"You placed **#{i}** in this week's tournament.\n"
                            f"📊 Score: **{tournament_score(stats):,}**\n"
                            f"💰 Coins earned: {stats['coins_earned']:,}\n"
                            f"🎮 Games won: {stats['games_won']}\n"
                            f"⌨️ Commands used: {stats['commands_used']}\n\n"
                            f"_Top 3 win prizes. Grind harder next week!_"
                        ),
                        color=discord.Color.blue(),
                    )
                    await send_dm(int(uid), "tournament", embed=dm_embed)
                except Exception:
                    pass

            # Save to history & reset
            data["history"].append({
                "season_start": data["season_start"],
                "season_end": today_str,
                "podium": [
                    {"user_id": uid, "score": tournament_score(s), "rank": i + 1}
                    for i, (uid, s) in enumerate(ranked[:3])
                ],
            })
            data["history"] = data["history"][-12:]  # keep last 12 weeks
            data["season_start"] = today_str
            data["scores"] = {}
            _save_tournament(data)

            # Announce
            recap_channel_id = get_notification_channel_id(cfg)
            if recap_channel_id and announcements:
                try:
                    channel = client.get_channel(int(recap_channel_id))
                    if channel:
                        await channel.send(
                            f"# 🏆 WEEKLY TOURNAMENT RESULTS 🏆\n\n"
                            + "\n".join(announcements)
                            + "\n\n_New season starts now. Grind hard._",
                            allowed_mentions=discord.AllowedMentions.none(),
                        )
                except Exception:
                    pass
        except Exception as e:
            log.exception("tournament_scheduler error: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# 🎲 RANDOM EVENTS
# Bot periodically drops a minigame (duck, math, scramble) in the recap channel.
# First user to answer wins the prize.
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_RANDOM_EVENT: dict[str, dict] = {}  # channel_id -> event state
EVENT_INTERVAL_MIN = 90 * 60   # min 90 minutes between events
EVENT_INTERVAL_MAX = 240 * 60  # max 4 hours

EVENT_TYPES = ["duck", "math", "scramble", "trivia"]


async def random_event_scheduler():
    """Background task: every 1.5-4 hours, drop a random event in the recap channel."""
    await client.wait_until_ready()
    while not client.is_closed():
        await asyncio.sleep(random.randint(EVENT_INTERVAL_MIN, EVENT_INTERVAL_MAX))
        try:
            cfg = load_config()
            recap_channel_id = get_notification_channel_id(cfg)
            if not recap_channel_id:
                continue
            channel = client.get_channel(int(recap_channel_id))
            if not channel:
                continue
            channel_id = str(channel.id)
            if channel_id in ACTIVE_RANDOM_EVENT:
                continue  # one at a time

            event_type = random.choice(EVENT_TYPES)
            await _spawn_random_event(channel, event_type)
        except Exception as e:
            log.exception("random_event_scheduler error: %s", e)


async def _spawn_random_event(channel, event_type: str):
    """Drop a specific random event in the channel."""
    channel_id = str(channel.id)
    prize = random.randint(300, 1500)

    if event_type == "duck":
        answer = "bang"
        prompt = (
            f"# 🦆 A WILD DUCK APPEARED!\n\n"
            f"🦆 Quick! Type **`bang`** to shoot it!\n"
            f"💰 Prize: **{prize:,}** coins"
        )
    elif event_type == "math":
        a, b = random.randint(11, 99), random.randint(11, 99)
        op = random.choice(["+", "-", "*"])
        answer = str(eval(f"{a}{op}{b}"))
        prompt = (
            f"# 🧮 MATH BLITZ!\n\n"
            f"First to type the answer wins:\n"
            f"## **{a} {op} {b} = ?**\n"
            f"💰 Prize: **{prize:,}** coins"
        )
    elif event_type == "scramble":
        words = [
            "balance", "fortune", "jackpot", "victory", "champion",
            "millionaire", "criminal", "shadow", "dynasty", "legend",
            "phantom", "hunter", "warrior", "midnight", "voltage",
        ]
        word = random.choice(words)
        scrambled = "".join(random.sample(word, len(word)))
        while scrambled == word:
            scrambled = "".join(random.sample(word, len(word)))
        answer = word
        prompt = (
            f"# 🔤 SCRAMBLED!\n\n"
            f"Unscramble this word — first to type it wins:\n"
            f"## **`{scrambled.upper()}`**\n"
            f"💰 Prize: **{prize:,}** coins"
        )
    elif event_type == "trivia":
        trivia_pool = [
            ("What's the capital of France?", "paris"),
            ("How many continents are there?", "7"),
            ("What's 2 to the 10th power?", "1024"),
            ("In what year did WW2 end?", "1945"),
            ("How many sides does a hexagon have?", "6"),
            ("What's the largest planet?", "jupiter"),
            ("Who painted the Mona Lisa?", "leonardo da vinci"),
            ("What's the chemical symbol for gold?", "au"),
            ("How many keys are on a standard piano?", "88"),
            ("What's the tallest mountain in the world?", "everest"),
        ]
        question, answer = random.choice(trivia_pool)
        prompt = (
            f"# 🧠 TRIVIA TIME!\n\n"
            f"First to answer wins:\n"
            f"## **{question}**\n"
            f"💰 Prize: **{prize:,}** coins"
        )
    else:
        return

    msg = await channel.send(prompt)
    ACTIVE_RANDOM_EVENT[channel_id] = {
        "type": event_type,
        "answer": answer.lower().strip(),
        "prize": prize,
        "started_at": time.time(),
        "message_id": msg.id,
        "channel_id": channel.id,
    }

    # Auto-expire after 5 minutes
    await asyncio.sleep(300)
    event = ACTIVE_RANDOM_EVENT.get(channel_id)
    if event and event["message_id"] == msg.id:
        ACTIVE_RANDOM_EVENT.pop(channel_id, None)
        try:
            await channel.send(
                f"⏰ The event expired. The answer was: **{event['answer']}**"
            )
        except Exception:
            pass


async def check_random_event_answer(message: discord.Message) -> bool:
    """Called from on_message. Returns True if the message resolved a random event."""
    channel_id = str(message.channel.id)
    event = ACTIVE_RANDOM_EVENT.get(channel_id)
    if not event:
        return False
    if message.content.strip().lower() == event["answer"]:
        # Winner!
        ACTIVE_RANDOM_EVENT.pop(channel_id, None)
        prize = event["prize"]
        economy.add(message.author.id, prize, "random event")
        elapsed = time.time() - event["started_at"]
        try:
            await message.channel.send(
                f"# 🎉 {message.author.mention} won the event!\n\n"
                f"💰 Prize: **{prize:,}** coins\n"
                f"⚡ Reaction time: **{elapsed:.1f}s**\n\n"
                f"_Next event in 1.5-4 hours._",
                allowed_mentions=discord.AllowedMentions(users=[message.author]),
            )
        except Exception:
            pass
        # Track random event wins for tournament
        add_tournament_score(message.author.id, coins_earned=prize, games_won=1)
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 🏆 ACHIEVEMENT SYSTEM
# Tracks milestones across all commands. Unlocks stored per-user in economy.json.
# Each achievement: id, name, emoji, description, reward (coins).
# ─────────────────────────────────────────────────────────────────────────────

ACHIEVEMENTS = {
    # ── Money milestones ─────────────────────────────────────────────────────
    "broke":         {"emoji":"🥺", "name":"Rock Bottom",        "desc":"Hit 0 coins",                              "reward":50},
    "first_grand":   {"emoji":"💵", "name":"First Grand",        "desc":"Earned 1,000 total coins",                 "reward":100},
    "rich":          {"emoji":"💰", "name":"Rich",                "desc":"Reach a balance of 10,000",                "reward":500},
    "millionaire":   {"emoji":"💎", "name":"Millionaire",         "desc":"Reach a balance of 1,000,000",             "reward":10000},
    "frugal":        {"emoji":"🏦", "name":"Frugal",              "desc":"Earn 50,000 total over your lifetime",     "reward":1000},

    # ── Daily login ──────────────────────────────────────────────────────────
    "first_daily":   {"emoji":"📅", "name":"Showed Up",           "desc":"Claimed your first /daily",                "reward":50},
    "daily_streak_7":{"emoji":"🔥", "name":"On Fire",              "desc":"7-day daily streak",                       "reward":500},
    "daily_streak_30":{"emoji":"🌟","name":"Habitual",            "desc":"30-day daily streak",                      "reward":2500},

    # ── Game wins ────────────────────────────────────────────────────────────
    "first_win":     {"emoji":"🥇", "name":"First Blood",         "desc":"Win your first game",                      "reward":100},
    "ten_wins":      {"emoji":"🏅", "name":"Veteran",             "desc":"Win 10 games",                             "reward":300},
    "fifty_wins":    {"emoji":"🎖️", "name":"Champion",            "desc":"Win 50 games",                             "reward":1000},
    "hundred_wins":  {"emoji":"👑", "name":"Legendary",           "desc":"Win 100 games",                            "reward":5000},

    # ── Game losses (lol) ────────────────────────────────────────────────────
    "first_loss":    {"emoji":"💀", "name":"Welcome to the Club", "desc":"Lose your first game",                     "reward":25},
    "loss_streak":   {"emoji":"😭", "name":"Pure Pain",            "desc":"Lose 10 games in a row",                   "reward":500},

    # ── Specific game achievements ───────────────────────────────────────────
    "blackjack_win": {"emoji":"🃏", "name":"21",                   "desc":"Win a /blackjack hand",                    "reward":100},
    "natural_21":    {"emoji":"♠️", "name":"Natural",              "desc":"Hit a natural blackjack",                  "reward":500},
    "slots_jackpot": {"emoji":"🎰", "name":"Jackpot!",            "desc":"Hit a slots jackpot (3x match)",           "reward":1000},
    "slots_diamond": {"emoji":"💎", "name":"Diamond Hands",       "desc":"Hit 3x💎 on slots",                         "reward":5000},
    "lottery_winner":{"emoji":"🎟️", "name":"Lucky One",           "desc":"Win the daily lottery",                    "reward":2000},
    "fight_won":     {"emoji":"🥊", "name":"Heavyweight",          "desc":"Win a /fight",                             "reward":200},
    "fight_streak_3":{"emoji":"🏆", "name":"Undefeated",           "desc":"Win 3 fights in a row",                    "reward":1500},
    "duel_won":      {"emoji":"🔫", "name":"Quick Draw",           "desc":"Win a /duel",                              "reward":150},
    "shootout_won":  {"emoji":"🚪", "name":"Last Door Standing",  "desc":"Win a /shootout",                          "reward":500},
    "russian_winner":{"emoji":"🎯", "name":"Lucky Survivor",      "desc":"Win at /gun",                              "reward":300},
    "bomb_passer":   {"emoji":"💣", "name":"Hot Hands",            "desc":"Pass a bomb in /bomb",                     "reward":50},
    "bomb_caught":   {"emoji":"💥", "name":"Boom Goes the Dynamite","desc":"Get blown up by /bomb",                  "reward":100},
    "heist_success": {"emoji":"💰", "name":"Bank Job",             "desc":"Pull off a successful /heist",             "reward":300},
    "heist_caught":  {"emoji":"🚔", "name":"Cuffed",               "desc":"Get caught on a /heist",                   "reward":50},
    "connect4_won":  {"emoji":"🟡", "name":"4 in a Row",           "desc":"Win at /connect4",                         "reward":200},
    "quest_winner":  {"emoji":"🗡️", "name":"Adventurer",          "desc":"Win a /quest",                             "reward":300},
    "wheel_jackpot": {"emoji":"🎡", "name":"Spin to Win",          "desc":"Hit the wheel jackpot",                    "reward":1500},

    # ── Crime & rob ──────────────────────────────────────────────────────────
    "first_robbery": {"emoji":"🦝", "name":"Sticky Fingers",       "desc":"Successfully rob someone",                 "reward":150},
    "rob_caught":    {"emoji":"👮", "name":"On Probation",         "desc":"Get caught robbing",                       "reward":50},
    "crime_streak":  {"emoji":"🦹", "name":"Career Criminal",      "desc":"Win 5 /crime in a row",                    "reward":750},

    # ── Social ───────────────────────────────────────────────────────────────
    "married":       {"emoji":"💍", "name":"Locked Down",          "desc":"Got married via /marry",                   "reward":500},
    "divorced":      {"emoji":"💔", "name":"Free Again",           "desc":"Got divorced",                             "reward":200},
    "rich_friends":  {"emoji":"💸", "name":"Generous",             "desc":"Send 5,000+ coins via /pay",               "reward":300},
    "victim":        {"emoji":"😢", "name":"The Mark",             "desc":"Got robbed by another user",               "reward":75},
    "court_winner":  {"emoji":"⚖️", "name":"Order in the Court",  "desc":"Win a /lawsuit",                           "reward":300},

    # ── AI command users ─────────────────────────────────────────────────────
    "first_roast":   {"emoji":"🔥", "name":"Roasted",              "desc":"Got roasted by /roast",                    "reward":50},
    "tarot_reader":  {"emoji":"🔮", "name":"Mystic",               "desc":"Got a /tarot reading",                     "reward":50},
    "analyzed":      {"emoji":"🔍", "name":"Self-Aware",           "desc":"Ran /analyze on yourself",                 "reward":50},

    # ── Shop ─────────────────────────────────────────────────────────────────
    "first_role":    {"emoji":"🎨", "name":"Drip",                 "desc":"Buy a colored role",                       "reward":250},
    "perm_role":     {"emoji":"♾️", "name":"Permanent Drip",       "desc":"Buy a PERMANENT colored role",             "reward":1000},
    "protector":     {"emoji":"🛡️", "name":"Untouchable",         "desc":"Buy /protect for the first time",          "reward":150},
    "megaphone":     {"emoji":"📢", "name":"Loudmouth",            "desc":"Use /megaphone for the first time",        "reward":300},

    # ── Hidden / fun ─────────────────────────────────────────────────────────
    "lucky_7":       {"emoji":"7️⃣", "name":"Triple 7s",            "desc":"Hit 3x7️⃣ on slots",                        "reward":2500},
    "lieordie_pro":  {"emoji":"🤥", "name":"Detector",             "desc":"Win 10 /lieordie rounds",                  "reward":500},
    "completionist": {"emoji":"🏆", "name":"Completionist",        "desc":"Earn 25 achievements",                     "reward":5000},
}


def _get_achievements(user_id: int) -> set:
    """Get the set of achievement IDs the user has unlocked."""
    u = economy._user(user_id)
    return set(u.setdefault("achievements", []))


def _get_counters(user_id: int) -> dict:
    """Get the user's progression counters (used for streaks, totals, etc.)."""
    u = economy._user(user_id)
    return u.setdefault("counters", {})


def _save_user_data():
    economy._save()


# Pending achievement notifications, batched per user so they don't spam channels
PENDING_ACHIEVEMENTS: dict[int, list[str]] = defaultdict(list)


async def _grant_achievement(user_id: int, ach_id: str, channel: discord.abc.Messageable | None = None):
    """Grant an achievement if not already earned. Pays reward + queues notification."""
    if ach_id not in ACHIEVEMENTS:
        return False
    earned = _get_achievements(user_id)
    if ach_id in earned:
        return False

    ach = ACHIEVEMENTS[ach_id]
    u = economy._user(user_id)
    u.setdefault("achievements", []).append(ach_id)
    economy.add(user_id, ach["reward"], f"achievement {ach_id}")
    _save_user_data()
    log.info("Achievement %s granted to %s", ach_id, user_id)

    # Send notification immediately if channel given
    if channel is not None:
        try:
            await channel.send(
                f"🎉 {ach['emoji']} **ACHIEVEMENT UNLOCKED**: {ach['name']}\n"
                f"<@{user_id}> earned **{ach['name']}** — _{ach['desc']}_\n"
                f"💰 +{ach['reward']:,} coins!",
                allowed_mentions=discord.AllowedMentions(users=[discord.Object(id=user_id)]) if False else discord.AllowedMentions.none(),
            )
        except Exception:
            pass

    # Check for completionist meta-achievement
    if ach_id != "completionist" and len(_get_achievements(user_id)) >= 25:
        await _grant_achievement(user_id, "completionist", channel)

    return True


def _bump_counter(user_id: int, key: str, by: int = 1) -> int:
    """Increment a counter and return the new value."""
    counters = _get_counters(user_id)
    counters[key] = counters.get(key, 0) + by
    _save_user_data()
    return counters[key]


def _set_counter(user_id: int, key: str, value):
    counters = _get_counters(user_id)
    counters[key] = value
    _save_user_data()


def _reset_counter(user_id: int, key: str):
    counters = _get_counters(user_id)
    counters[key] = 0
    _save_user_data()


# ── Trigger functions called by game commands ───────────────────────────────

async def trigger_game_win(user_id: int, game: str, channel=None):
    """Call after any game win. Updates counters and grants achievements."""
    wins = _bump_counter(user_id, "total_wins")
    _bump_counter(user_id, f"wins_{game}")
    _reset_counter(user_id, "loss_streak")

    if wins == 1:
        await _grant_achievement(user_id, "first_win", channel)
    if wins >= 10:
        await _grant_achievement(user_id, "ten_wins", channel)
    if wins >= 50:
        await _grant_achievement(user_id, "fifty_wins", channel)
    if wins >= 100:
        await _grant_achievement(user_id, "hundred_wins", channel)

    # Game-specific
    game_map = {
        "blackjack": "blackjack_win",
        "fight": "fight_won",
        "duel": "duel_won",
        "shootout": "shootout_won",
        "gun": "russian_winner",
        "connect4": "connect4_won",
        "quest": "quest_winner",
    }
    if game in game_map:
        await _grant_achievement(user_id, game_map[game], channel)

    # Fight win streak
    if game == "fight":
        streak = _bump_counter(user_id, "fight_win_streak")
        if streak >= 3:
            await _grant_achievement(user_id, "fight_streak_3", channel)
    else:
        # If they win something else, fight streak still counts as fights only
        pass

    # Crime streak
    if game == "crime":
        streak = _bump_counter(user_id, "crime_win_streak")
        if streak >= 5:
            await _grant_achievement(user_id, "crime_streak", channel)

    # Lieordie wins
    if game == "lieordie":
        if wins_lod := _get_counters(user_id).get("wins_lieordie", 0):
            if wins_lod >= 10:
                await _grant_achievement(user_id, "lieordie_pro", channel)


async def trigger_game_loss(user_id: int, game: str, channel=None):
    losses = _bump_counter(user_id, "total_losses")
    if losses == 1:
        await _grant_achievement(user_id, "first_loss", channel)

    streak = _bump_counter(user_id, "loss_streak")
    if streak >= 10:
        await _grant_achievement(user_id, "loss_streak", channel)

    # Reset relevant win streaks
    if game == "fight":
        _set_counter(user_id, "fight_win_streak", 0)
    if game == "crime":
        _set_counter(user_id, "crime_win_streak", 0)


async def trigger_balance_check(user_id: int, channel=None):
    """Call whenever balance changes. Checks money milestones."""
    bal = economy.balance(user_id)
    if bal == 0:
        await _grant_achievement(user_id, "broke", channel)
    if bal >= 10_000:
        await _grant_achievement(user_id, "rich", channel)
    if bal >= 1_000_000:
        await _grant_achievement(user_id, "millionaire", channel)

    # Total earned over lifetime
    earned = economy._user(user_id).get("stats", {}).get("total_earned", 0)
    if earned >= 1000:
        await _grant_achievement(user_id, "first_grand", channel)
    if earned >= 50_000:
        await _grant_achievement(user_id, "frugal", channel)


async def trigger_daily_claim(user_id: int, channel=None):
    """Track daily claim streaks."""
    counters = _get_counters(user_id)
    last_claim = counters.get("last_daily_claim", 0)
    now = time.time()

    # First daily ever
    if last_claim == 0:
        await _grant_achievement(user_id, "first_daily", channel)

    # Streak: if last claim was within 25 hours but more than 23, count as continuing streak
    if last_claim and (23 * 3600) <= (now - last_claim) <= (49 * 3600):
        streak = _bump_counter(user_id, "daily_streak")
    else:
        # Either first claim or broken streak
        _set_counter(user_id, "daily_streak", 1)
        streak = 1

    _set_counter(user_id, "last_daily_claim", now)

    if streak >= 7:
        await _grant_achievement(user_id, "daily_streak_7", channel)
    if streak >= 30:
        await _grant_achievement(user_id, "daily_streak_30", channel)


# ── Convenience: a single hook to call from anywhere ─────────────────────────

async def check_event(user_id: int, event: str, channel=None, **kwargs):
    """Generic event hook for misc one-off achievements."""
    table = {
        "natural_21":         "natural_21",
        "slots_jackpot":      "slots_jackpot",
        "slots_diamond":      "slots_diamond",
        "lucky_7":            "lucky_7",
        "lottery_won":        "lottery_winner",
        "wheel_jackpot":      "wheel_jackpot",
        "bomb_passed":        "bomb_passer",
        "bomb_caught":        "bomb_caught",
        "heist_success":      "heist_success",
        "heist_caught":       "heist_caught",
        "rob_success":        "first_robbery",
        "rob_caught":         "rob_caught",
        "got_robbed":         "victim",
        "married":            "married",
        "divorced":           "divorced",
        "court_won":          "court_winner",
        "got_roasted":        "first_roast",
        "got_tarot":          "tarot_reader",
        "got_analyzed":       "analyzed",
        "bought_role":        "first_role",
        "bought_perm_role":   "perm_role",
        "bought_protect":     "protector",
        "bought_megaphone":   "megaphone",
    }
    if event in table:
        await _grant_achievement(user_id, table[event], channel)

    # Special handling for /pay
    if event == "paid_amount":
        amount = kwargs.get("amount", 0)
        total = _bump_counter(user_id, "total_paid", amount)
        if total >= 5000:
            await _grant_achievement(user_id, "rich_friends", channel)


def fmt_cooldown(seconds: int) -> str:
    """Format seconds as h:m or m:s string."""
    if seconds <= 0:
        return "ready"
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h {m}m"
    if seconds >= 60:
        m = seconds // 60
        s = seconds % 60
        return f"{m}m {s}s"
    return f"{seconds}s"


def fmt_coins(amount: int) -> str:
    return f"💰 **{amount:,}**"


# ── Provider implementations ─────────────────────────────────────────────────
class RateLimitError(Exception):
    pass

class ProviderError(Exception):
    pass


async def call_groq(system_prompt, messages, model, temperature, api_key, max_tokens=200):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 429:
                raise RateLimitError("groq rate limited")
            if resp.status == 401:
                raise ProviderError("groq invalid key")
            if resp.status != 200:
                text = await resp.text()
                raise ProviderError(f"groq {resp.status}: {text[:200]}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()


async def call_cerebras(system_prompt, messages, model, temperature, api_key, max_tokens=200):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.cerebras.ai/v1/chat/completions",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 429:
                raise RateLimitError("cerebras rate limited")
            if resp.status == 401:
                raise ProviderError("cerebras invalid key")
            if resp.status != 200:
                text = await resp.text()
                raise ProviderError(f"cerebras {resp.status}: {text[:200]}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()


async def call_gemini(system_prompt, messages, model, temperature, api_key, max_tokens=200, images=None):
    """Gemini supports text OR text + images. Pass images=[(mime, base64_data), ...]"""
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    if contents and contents[-1]["role"] == "model":
        contents.append({"role": "user", "parts": [{"text": "(continue the conversation)"}]})
    if not contents:
        contents = [{"role": "user", "parts": [{"text": "hello"}]}]

    # If there are images, attach them to the LAST user turn
    if images:
        # Find last user message
        for c in reversed(contents):
            if c["role"] == "user":
                for mime, data in images:
                    c["parts"].append({
                        "inlineData": {"mimeType": mime, "data": data}
                    })
                break

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 429:
                raise RateLimitError("gemini rate limited")
            if resp.status in (401, 403):
                raise ProviderError("gemini invalid key")
            if resp.status != 200:
                text = await resp.text()
                raise ProviderError(f"gemini {resp.status}: {text[:200]}")
            data = await resp.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError):
                raise ProviderError(f"gemini malformed response: {str(data)[:200]}")


PROVIDER_COOLDOWN: dict[str, float] = {}


async def ask_ai(system_prompt, messages, cfg, images=None) -> str:
    """Try each provider in order until one succeeds.
    If images are present, forces Gemini (only vision-capable free provider)."""
    temperature = cfg["temperature"]
    max_tokens = cfg.get("max_tokens", 200)
    errors = []
    now = time.time()

    # Images require Gemini
    if images:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return "⚠️ Can't see images — GEMINI_API_KEY not set."
        try:
            reply = await call_gemini(
                system_prompt, messages,
                cfg.get("vision_model", "gemini-2.5-flash"),
                temperature, api_key, max_tokens, images=images
            )
            log.info("✓ Used provider: gemini (vision)")
            return reply
        except RateLimitError:
            PROVIDER_COOLDOWN["gemini"] = time.time() + 60
            return "⚠️ Gemini vision rate limited. Try again in a moment."
        except Exception as e:
            log.exception("Gemini vision error")
            return f"⚠️ Vision error: {e}"

    # Normal text: iterate providers in preferred order
    providers = cfg.get("provider_order", ["groq", "cerebras", "gemini"])
    for provider in providers:
        if PROVIDER_COOLDOWN.get(provider, 0) > now:
            continue
        key_env = f"{provider.upper()}_API_KEY"
        api_key = os.environ.get(key_env, "")
        if not api_key:
            continue
        try:
            if provider == "groq":
                reply = await call_groq(system_prompt, messages, cfg["groq_model"], temperature, api_key, max_tokens)
            elif provider == "cerebras":
                reply = await call_cerebras(system_prompt, messages, cfg["cerebras_model"], temperature, api_key, max_tokens)
            elif provider == "gemini":
                reply = await call_gemini(system_prompt, messages, cfg["gemini_model"], temperature, api_key, max_tokens)
            else:
                continue
            log.info("✓ Used provider: %s", provider)
            return reply
        except RateLimitError:
            log.warning("%s rate limited, cooling down 60s", provider)
            PROVIDER_COOLDOWN[provider] = time.time() + 60
            errors.append(f"{provider}: rate limit")
            continue
        except ProviderError as e:
            log.warning("%s failed: %s", provider, e)
            errors.append(f"{provider}: {e}")
            continue
        except asyncio.TimeoutError:
            log.warning("%s timed out", provider)
            errors.append(f"{provider}: timeout")
            continue
        except Exception as e:
            log.exception("Unexpected error from %s", provider)
            errors.append(f"{provider}: {e}")
            continue

    log.error("All providers failed: %s", errors)
    return "⚠️ All AI providers are busy right now. Try again in a moment."


# ── Image downloader ──────────────────────────────────────────────────────────
async def _fetch_url_as_image(url: str, session: aiohttp.ClientSession) -> tuple | None:
    """Download a URL and return (mime, base64). Returns None on failure."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            ctype = resp.headers.get("Content-Type", "image/png").split(";")[0].strip().lower()
            # Normalize mime
            if ctype not in {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}:
                # Guess from URL if header is useless
                lower = url.lower()
                if lower.endswith(".png"):    ctype = "image/png"
                elif lower.endswith(".jpg") or lower.endswith(".jpeg"): ctype = "image/jpeg"
                elif lower.endswith(".webp"): ctype = "image/webp"
                elif lower.endswith(".gif"):  ctype = "image/gif"
                else:
                    return None
            data = await resp.read()
            if len(data) > 4 * 1024 * 1024:  # 4MB cap
                return None
            return (ctype, base64.b64encode(data).decode("utf-8"))
    except Exception as e:
        log.warning("Failed to fetch %s: %s", url, e)
        return None


async def download_images(message: discord.Message) -> list:
    """Return (mime, base64) for every visual: attachments, stickers,
    Tenor/Giphy GIFs, and custom emojis."""
    images = []
    allowed_mimes = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}

    # 1) Direct attachments
    for att in message.attachments:
        if att.content_type and att.content_type.lower() in allowed_mimes:
            if att.size > 4 * 1024 * 1024:
                continue
            try:
                data = await att.read()
                images.append((att.content_type, base64.b64encode(data).decode("utf-8")))
            except Exception as e:
                log.warning("Failed to download %s: %s", att.filename, e)

    async with aiohttp.ClientSession() as session:
        # 2) Discord stickers (message.stickers is a list of StickerItem)
        for sticker in getattr(message, "stickers", []):
            try:
                url = sticker.url  # always returns a usable CDN URL
                result = await _fetch_url_as_image(url, session)
                if result:
                    images.append(result)
            except Exception as e:
                log.warning("Sticker fetch failed: %s", e)

        # 3) Embeds — Discord auto-generates embeds for Tenor/Giphy/link previews
        for embed in message.embeds:
            # Prefer embed.image, fall back to embed.thumbnail
            candidate_url = None
            if embed.image and embed.image.url:
                candidate_url = embed.image.url
            elif embed.thumbnail and embed.thumbnail.url:
                candidate_url = embed.thumbnail.url
            if candidate_url:
                # Tenor often serves a .gif via a dynamic URL; try to append .gif if missing
                result = await _fetch_url_as_image(candidate_url, session)
                if result:
                    images.append(result)

        # 4) Custom Discord emojis in message content — format <a?:name:id>
        # Limit to first 3 to avoid spam
        emoji_pattern = re.compile(r"<(a?):(\w+):(\d+)>")
        emoji_matches = emoji_pattern.findall(message.content)[:3]
        for animated, _name, eid in emoji_matches:
            ext = "gif" if animated == "a" else "png"
            url = f"https://cdn.discordapp.com/emojis/{eid}.{ext}"
            result = await _fetch_url_as_image(url, session)
            if result:
                images.append(result)

    # Cap total images sent to Gemini to avoid oversized payloads
    return images[:5]


# ── Reactions ─────────────────────────────────────────────────────────────────
REACTION_POOL = [
    "💀", "👀", "🔥", "😂", "🤡", "😒", "🤨", "😮‍💨",
    "🙄", "💅", "🧠", "⚡", "💯", "🫠", "🫡", "🤌"
]


async def send_reply(message: discord.Message, reply: str, chimed_in: bool = False):
    if len(reply) <= 2000:
        if chimed_in:
            await message.channel.send(reply)
        else:
            await message.reply(reply)
    else:
        chunks = [reply[i:i+1990] for i in range(0, len(reply), 1990)]
        for i, chunk in enumerate(chunks):
            if i == 0 and not chimed_in:
                await message.reply(chunk)
            else:
                await message.channel.send(chunk)


# ── Bot ───────────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")


def channel_allowed(channel_id: str, cfg: dict) -> bool:
    return not cfg["allowed_channels"] or channel_id in cfg["allowed_channels"]


def is_direct_trigger(message: discord.Message, cfg: dict) -> bool:
    if message.author == client.user:
        return False
    if not cfg["respond_to_bots"] and message.author.bot:
        return False
    if str(message.author.id) in cfg["blocked_users"]:
        return False
    if not channel_allowed(str(message.channel.id), cfg):
        return False
    if cfg["trigger_mode"] == "all_messages":
        return True
    mentioned = client.user in message.mentions
    is_reply = (
        message.reference is not None
        and message.reference.resolved is not None
        and isinstance(message.reference.resolved, discord.Message)
        and message.reference.resolved.author == client.user
    )
    keyword_hit = False
    for kw in cfg.get("trigger_keywords", []):
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', message.content.lower()):
            keyword_hit = True
            break
    return mentioned or is_reply or keyword_hit


def get_time_context() -> str:
    now = datetime.now(timezone.utc)
    hour = now.hour
    if 0 <= hour < 5:
        return f"(It is currently {hour}:{now.minute:02d} UTC — late night / early morning)"
    if 5 <= hour < 12:
        return f"(It is currently {hour}:{now.minute:02d} UTC — morning)"
    if 12 <= hour < 18:
        return f"(It is currently {hour}:{now.minute:02d} UTC — afternoon)"
    return f"(It is currently {hour}:{now.minute:02d} UTC — evening/night)"


async def rotate_status():
    await client.wait_until_ready()
    while not client.is_closed():
        cfg = load_config()
        statuses = cfg.get("status_messages") or ["online"]
        status = random.choice(statuses)
        try:
            await client.change_presence(activity=discord.CustomActivity(name=status))
        except Exception as e:
            log.warning("Status change failed: %s", e)
        await asyncio.sleep(max(cfg.get("status_rotation_minutes", 10), 2) * 60)


async def silence_checker():
    await client.wait_until_ready()
    while not client.is_closed():
        await asyncio.sleep(180)
        cfg = load_config()
        if not cfg.get("silence_checkin_enabled"):
            continue
        silence_seconds = cfg["silence_minutes"] * 60
        for channel_id, ch_last in list(last_channel_activity.items()):
            if not channel_allowed(channel_id, cfg):
                continue
            try:
                channel = client.get_channel(int(channel_id))
                if channel and hasattr(channel, "name"):
                    cname = channel.name.lower()
                    if any(x in cname for x in ["log", "admin", "backend", "audit", "welcome", "invite"]):
                        continue
            except Exception:
                pass
            bot_last = last_bot_activity.get(channel_id, 0)
            now = time.time()
            if (now - ch_last) < 7200 and (now - bot_last) > silence_seconds:
                try:
                    channel = client.get_channel(int(channel_id))
                    if channel is None:
                        continue
                    history = memory.get_messages(channel_id)
                    system = cfg["system_prompt"] + (
                        "\n\nThe channel has been quiet for a while but people were active recently. "
                        "Drop a short, spontaneous message — 1 sentence max — like you're lurking and decided to say something. "
                        "Don't greet anyone. Don't ask 'is anyone here'. Just say something in-character. "
                        + get_time_context()
                    )
                    reply = await ask_ai(
                        system, history if history else [{"role":"user","content":"..."}],
                        {**cfg, "temperature": min(cfg["temperature"] + 0.15, 1.0)},
                    )
                    if not reply.startswith("⚠️") and reply.strip():
                        await channel.send(reply)
                        last_bot_activity[channel_id] = time.time()
                        memory.append(channel_id, "assistant", reply)
                        log.info("Silence check-in on #%s", getattr(channel, "name", channel_id))
                except Exception as e:
                    log.warning("Silence check-in failed: %s", e)


async def daily_recap_scheduler():
    """Posts a daily recap at the configured UTC hour."""
    await client.wait_until_ready()
    while not client.is_closed():
        await asyncio.sleep(300)  # check every 5 min
        cfg = load_config()
        if not cfg.get("daily_recap_enabled"):
            continue
        recap_channel_id = cfg.get("daily_recap_channel", "").strip()
        if not recap_channel_id:
            continue
        target_hour = cfg.get("daily_recap_hour_utc", 4)
        now = datetime.now(timezone.utc)

        # Only fire within the configured hour, and only once per day (persistent)
        if now.hour != target_hour:
            continue
        today_str = now.strftime("%Y-%m-%d")
        if get_last_fire("daily_recap") == today_str:
            continue

        # Mark fired FIRST to prevent race conditions
        set_last_fire("daily_recap", today_str)

        # Recap the PREVIOUS day's messages
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        # Gather from all channels that had activity yesterday, but primarily the recap channel
        logs = daily_log.read_day(recap_channel_id, yesterday)
        if len(logs) < 10:
            log.info("Not enough messages for recap (%d), skipping", len(logs))
            continue

        # Format for AI
        transcript = "\n".join(f"{m['author']}: {m['content']}" for m in logs[-300:])
        system = cfg["system_prompt"] + (
            "\n\nYou are writing a daily recap of yesterday's chat. "
            "Pick out the most ridiculous, funniest, or most memorable moments. "
            "Call out specific users and quote the wildest things they said. "
            "Format as a short recap with bullet points — 5 to 8 highlights max. "
            "Stay completely in character. Be ruthless. Don't thank anyone. "
            "Lead with a short intro line in your voice, then list the highlights."
        )
        user_msg = f"Here's yesterday's transcript from the chat. Roast it:\n\n{transcript}"

        try:
            recap = await ask_ai(
                system,
                [{"role": "user", "content": user_msg}],
                {**cfg, "max_tokens": cfg.get("daily_recap_max_tokens", 500)},
            )
            if not recap.startswith("⚠️") and recap.strip():
                post_channel_id = get_notification_channel_id(cfg)
                channel = client.get_channel(int(post_channel_id))
                if channel:
                    header = f"📰 **Daily Recap — {yesterday}**\n\n"
                    full = header + recap
                    if len(full) <= 2000:
                        await channel.send(full)
                    else:
                        await channel.send(header)
                        for i in range(0, len(recap), 1990):
                            await channel.send(recap[i:i+1990])
                    log.info("Posted daily recap for %s", yesterday)
        except Exception as e:
            log.exception("Daily recap failed: %s", e)

        # Clean up old logs
        daily_log.cleanup_old(days_to_keep=7)


# ── Slash commands: entertainment ────────────────────────────────────────────
RACE_RACERS = [
    ("🏎️", "Lambo"),
    ("🚗", "Civic"),
    ("🚙", "Minivan"),
    ("🛺", "Tuk-tuk"),
    ("🚜", "Tractor"),
    ("🏍️", "Bike"),
    ("🛴", "Scooter"),
    ("🚲", "Tricycle"),
]

# Track per-channel race lock so two people don't start one at the same time
active_races: set[str] = set()


def _draw_track(positions: list[tuple], track_len: int = 20, finished: list = None) -> str:
    """Draw the race state. positions: [(emoji, name, pos)]"""
    finished = finished or []
    lines = []
    for emoji, name, pos in positions:
        bar = "─" * track_len
        # Place racer
        p = min(pos, track_len)
        line = bar[:p] + emoji + bar[p:] + "🏁"
        # Add name + position indicator
        suffix = ""
        if name in [f[1] for f in finished]:
            place = [f[1] for f in finished].index(name) + 1
            medal = ["🥇","🥈","🥉"][place-1] if place <= 3 else f"#{place}"
            suffix = f"  {medal}"
        lines.append(f"`{line}` **{name}**{suffix}")
    return "\n".join(lines)


@tree.command(name="rs", description="Start a race! Optionally tag up to 3 other racers.")
@discord.app_commands.describe(
    racer1="Tag a user to race against (optional)",
    racer2="Tag another user (optional)",
    racer3="Tag another user (optional)",
)
async def race_command(
    interaction: discord.Interaction,
    racer1: discord.Member = None,
    racer2: discord.Member = None,
    racer3: discord.Member = None,
):
    channel_id = str(interaction.channel_id)
    if channel_id in active_races:
        await interaction.response.send_message("🏁 A race is already running here!", ephemeral=True)
        return

    # Defer immediately so we don't time out
    await interaction.response.defer()

    active_races.add(channel_id)
    try:
        # Build the racer list — start with the command runner, then add tagged racers
        # Each user gets a random vehicle from the pool
        starter = interaction.user
        tagged_users = [u for u in [racer1, racer2, racer3] if u is not None]
        # Deduplicate (in case someone tags themselves or duplicates)
        seen_ids = {starter.id}
        unique_tagged = []
        for u in tagged_users:
            if u.id not in seen_ids:
                seen_ids.add(u.id)
                unique_tagged.append(u)

        all_users = [starter] + unique_tagged

        # Assign each user a random vehicle (no duplicates)
        vehicles = random.sample(RACE_RACERS, min(len(RACE_RACERS), 8))
        racers = []  # [(emoji, name, mention)]
        name_to_user_id: dict[str, int] = {}  # for economy payout
        for i, u in enumerate(all_users):
            emoji, _vehicle = vehicles[i]
            racers.append((emoji, u.display_name, u.mention))
            name_to_user_id[u.display_name] = u.id

        # If only the starter raced, fill in to 4 racers with NPCs
        if len(racers) == 1:
            npc_pool = [v for v in vehicles[1:] if v not in [(r[0], None) for r in racers]]
            for i in range(3):
                emoji, npc_name = vehicles[i + 1]
                racers.append((emoji, npc_name, None))

        track_len = 20
        # positions: [(emoji, name, mention, pos)]
        positions = [(e, n, m, 0) for (e, n, m) in racers]
        finished: list = []  # [(emoji, name, mention)] in finishing order

        def draw(positions, finished):
            lines = []
            finished_names = [f[1] for f in finished]
            for emoji, name, mention, pos in positions:
                bar = "─" * track_len
                p = min(pos, track_len)
                line = bar[:p] + emoji + bar[p:] + "🏁"
                suffix = ""
                if name in finished_names:
                    place = finished_names.index(name) + 1
                    medal = ["🥇","🥈","🥉"][place-1] if place <= 3 else f"#{place}"
                    suffix = f"  {medal}"
                # Use mention if it's a real user, plain bold name for NPCs
                label = mention if mention else f"**{name}**"
                lines.append(f"`{line}` {label}{suffix}")
            return "\n".join(lines)

        silent = discord.AllowedMentions.none()

        # Initial frame
        await interaction.edit_original_response(
            content=f"🏁 **RACE START!**\n\n{draw(positions, finished)}",
            allowed_mentions=silent,
        )

        # Race animation loop — each tick everyone moves 0-2 spaces
        for _tick in range(40):  # max 40 ticks
            await asyncio.sleep(1.2)
            new_positions = []
            finished_names = [f[1] for f in finished]
            for emoji, name, mention, pos in positions:
                if name in finished_names:
                    new_positions.append((emoji, name, mention, track_len))
                    continue
                step = random.choices([0, 1, 2], weights=[1, 2, 1])[0]
                new_pos = pos + step
                if new_pos >= track_len and name not in finished_names:
                    finished.append((emoji, name, mention))
                    finished_names = [f[1] for f in finished]
                    new_pos = track_len
                new_positions.append((emoji, name, mention, new_pos))
            positions = new_positions

            header = "🏁 **RACE IN PROGRESS!**\n\n"
            if len(finished) == len(racers):
                header = "🏁 **RACE FINISHED!**\n\n"
            content = header + draw(positions, finished)

            try:
                await interaction.edit_original_response(content=content, allowed_mentions=silent)
            except discord.HTTPException:
                pass

            if len(finished) == len(racers):
                break

        # Final result — only ping the winner
        winner_emoji, winner_name, winner_mention = finished[0]
        winner_label = winner_mention if winner_mention else f"**{winner_name}**"
        results = "\n".join(
            f"{['🥇','🥈','🥉','4️⃣'][i] if i < 4 else f'#{i+1}'} {e} {(m if m else f'**{n}**')}"
            for i, (e, n, m) in enumerate(finished)
        )

        # Economy: prize for real-user winner only
        prize_text = ""
        winner_uid = name_to_user_id.get(winner_name)
        if winner_uid:
            # Bigger prize when more real users were in the race
            real_count = len([1 for r in racers if r[2] is not None])
            prize = 100 + (real_count - 1) * 100  # 1=100, 2=200, 3=300, 4=400
            economy.add(winner_uid, prize, "race win")
            economy.record_win(winner_uid)
            prize_text = f"\n💰 Earned **{prize:,}** coins!"
            # Mark the other real-user finishers as losses
            for _e, n, _m in finished[1:]:
                uid = name_to_user_id.get(n)
                if uid:
                    economy.record_loss(uid)

        final = (
            f"🏁 **RACE FINISHED!**\n\n"
            f"{draw(positions, finished)}\n\n"
            f"## 🏆 {winner_emoji} {winner_label} WINS!{prize_text}\n\n"
            f"**Final standings:**\n{results}"
        )
        # Allow only the winner to be pinged on the final message
        if winner_mention:
            try:
                await interaction.edit_original_response(content=final)
            except Exception:
                await interaction.edit_original_response(content=final, allowed_mentions=silent)
        else:
            await interaction.edit_original_response(content=final, allowed_mentions=silent)

    finally:
        active_races.discard(channel_id)


# ── Helper: gather user's recent messages from Discord + daily logs ──────────
async def gather_user_messages(interaction: discord.Interaction, user: discord.Member, max_count: int = 50) -> list[str]:
    """Pull a user's recent messages from current channel history + daily logs."""
    user_messages: list[str] = []
    try:
        async for m in interaction.channel.history(limit=1000):
            if m.author.id == user.id and m.content and m.content.strip():
                user_messages.append(m.content.strip())
                if len(user_messages) >= max_count:
                    break
    except Exception as e:
        log.warning("Channel history fetch failed: %s", e)

    today = datetime.now(timezone.utc)
    target_name = user.display_name
    seen = set(user_messages)
    for days_back in range(7):
        date_str = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        for log_file in RECAP_DIR.glob(f"*_{date_str}.jsonl"):
            try:
                with open(log_file) as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("author") == target_name:
                                content = entry.get("content", "").strip()
                                if content and content not in seen:
                                    seen.add(content)
                                    user_messages.append(content)
                        except Exception:
                            continue
            except Exception:
                continue
    return user_messages[:80]


# ── ⚖️ /court ────────────────────────────────────────────────────────────────
@tree.command(name="court", description="Criminal trial: charge a user with a crime. Pure roleplay verdict (no $).")
@discord.app_commands.describe(defendant="Who's on trial?", charge="What are they charged with?")
async def court_command(interaction: discord.Interaction, defendant: discord.Member, charge: str):
    cfg = load_config()
    silent = discord.AllowedMentions.none()
    prosecutor = interaction.user

    if defendant.id == prosecutor.id:
        await interaction.response.send_message("You can't put yourself on trial.", ephemeral=True)
        return
    if defendant.bot:
        await interaction.response.send_message("Can't put a bot on trial.", ephemeral=True)
        return
    if str(defendant.id) in cfg.get("respected_users", []):
        await interaction.response.send_message(
            f"❌ {defendant.mention} is the boss. They're above the law.",
            allowed_mentions=silent,
        )
        return

    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    # Animated courtroom intro
    await edit("⚖️ **ALL RISE**\n\n*The court is now in session...*")
    await asyncio.sleep(2.0)
    await edit(
        f"⚖️ **CASE FILE OPENED**\n\n"
        f"**Prosecution:** {prosecutor.mention}\n"
        f"**Defendant:** {defendant.mention}\n"
        f"**Charge:** *{charge}*"
    )
    await asyncio.sleep(2.5)

    # Gather defendant's messages as evidence
    evidence = await gather_user_messages(interaction, defendant, max_count=40)
    evidence_text = "\n".join(f"- {m}" for m in evidence) if evidence else "(no message history available)"

    await edit(
        f"⚖️ **CASE: *{charge}***\n\n"
        f"📜 Court reviewing evidence... ({len(evidence)} messages from {defendant.mention})"
    )
    await asyncio.sleep(2.0)

    # Prosecutor opening statement
    await edit(
        f"⚖️ **PROSECUTION OPENING STATEMENT**\n\n"
        f"👨‍⚖️ *{prosecutor.display_name} approaches the bench...*"
    )
    await asyncio.sleep(1.8)

    prosecutor_system = (
        cfg["system_prompt"] +
        "\n\n=== COURTROOM PROSECUTOR MODE ===\n"
        f"You are the PROSECUTOR. {defendant.display_name} is on trial for: '{charge}'. "
        "Deliver a brief, dramatic 2-3 sentence opening statement against the defendant. "
        "Reference SPECIFIC quotes or patterns from their actual messages below as evidence when possible. "
        "Be theatrical, witty, and devastating. Stay in character. No preamble — just the statement."
    )
    prosecution = await ask_ai(
        prosecutor_system,
        [{"role": "user", "content": f"Defendant's recent messages:\n\n{evidence_text}\n\nDeliver your prosecution statement."}],
        {**cfg, "max_tokens": 250},
    )
    if prosecution.startswith("⚠️"):
        prosecution = "Your honor, the evidence speaks for itself."

    await edit(
        f"⚖️ **PROSECUTION** ({prosecutor.mention})\n\n"
        f"> {prosecution}"
    )
    await asyncio.sleep(4.0)

    # Defense statement
    await edit(
        f"⚖️ **DEFENSE OPENING STATEMENT**\n\n"
        f"🛡️ *{defendant.display_name}'s counsel rises...*"
    )
    await asyncio.sleep(1.8)

    defense_system = (
        cfg["system_prompt"] +
        "\n\n=== COURTROOM DEFENSE MODE ===\n"
        f"You are the DEFENSE ATTORNEY for {defendant.display_name} who is charged with: '{charge}'. "
        "Deliver a brief, dramatic 2-3 sentence defense. "
        "Be desperate, dramatic, and reach for any excuse. Use their actual messages as context. "
        "Stay in character. No preamble — just the defense argument."
    )
    defense = await ask_ai(
        defense_system,
        [{"role": "user", "content": f"Defendant's recent messages:\n\n{evidence_text}\n\nDeliver your defense statement."}],
        {**cfg, "max_tokens": 250},
    )
    if defense.startswith("⚠️"):
        defense = "Your honor, my client is innocent. Probably."

    await edit(
        f"⚖️ **DEFENSE** ({defendant.mention})\n\n"
        f"> {defense}"
    )
    await asyncio.sleep(4.0)

    # Judge deliberates
    await edit("⚖️ *The judge deliberates...*")
    await asyncio.sleep(2.5)
    await edit("⚖️ *The judge consults precedent...*")
    await asyncio.sleep(2.0)
    await edit("⚖️ 🔨 *gavel raised...*")
    await asyncio.sleep(1.5)

    # Verdict
    judge_system = (
        cfg["system_prompt"] +
        "\n\n=== JUDGE MODE ===\n"
        f"You are the JUDGE in a Discord courtroom comedy. The defendant {defendant.display_name} is charged with: '{charge}'. "
        "You just heard both prosecution and defense. Deliver a verdict — GUILTY or NOT GUILTY — followed by a "
        "sentence (the punishment) which can be hilarious or serious. Total 2-4 sentences. "
        "Format as:\n"
        "**VERDICT:** GUILTY/NOT GUILTY\n**SENTENCE:** [your punishment]\n\nThen one final sentence of judgement."
    )
    verdict_prompt = (
        f"Charge: {charge}\n\n"
        f"Prosecution said: {prosecution}\n\n"
        f"Defense said: {defense}\n\n"
        f"Defendant's messages for context:\n{evidence_text[:1500]}\n\n"
        "Deliver your verdict."
    )
    verdict = await ask_ai(
        judge_system,
        [{"role": "user", "content": verdict_prompt}],
        {**cfg, "max_tokens": 300},
    )
    if verdict.startswith("⚠️"):
        verdict = "**VERDICT:** GUILTY\n**SENTENCE:** Read the room. Court adjourned."

    final = (
        f"⚖️ **🔨 GAVEL DROPS** 🔨\n\n"
        f"## Case: *{charge}*\n"
        f"**Defendant:** {defendant.mention}\n\n"
        f"{verdict}"
    )
    try:
        await interaction.edit_original_response(content=final)
    except Exception:
        pass


# ── 📱 /bio — AI Tinder bio ─────────────────────────────────────────────────
@tree.command(name="bio", description="Generate a savage Tinder bio for a user based on their messages.")
@discord.app_commands.describe(user="Who's getting a bio?")
async def bio_command(interaction: discord.Interaction, user: discord.Member):
    cfg = load_config()
    silent = discord.AllowedMentions.none()

    if user.bot:
        await interaction.response.send_message("Bots don't date.", ephemeral=True)
        return

    is_boss = str(user.id) in cfg.get("respected_users", [])

    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    # Animated "loading profile"
    await edit(f"📱 *opening Tinder...*")
    await asyncio.sleep(0.9)
    await edit(f"📱 *creating profile for {user.mention}...*")
    await asyncio.sleep(1.0)
    await edit(f"📱 *uploading photos...* 📸")
    await asyncio.sleep(1.0)

    # Gather their messages
    user_messages = await gather_user_messages(interaction, user, max_count=50)

    await edit(f"📱 *analyzing personality...* 🧠 ({len(user_messages)} messages found)")
    await asyncio.sleep(1.2)
    await edit(f"📱 *generating bio...* ✍️")
    await asyncio.sleep(1.0)

    transcript = "\n".join(f"- {m}" for m in user_messages) if user_messages else "(this user is a lurker)"

    if is_boss:
        # Generous bio for boss
        bio_system = (
            cfg["system_prompt"] +
            "\n\n=== TINDER BIO MODE — RESPECT MODE ===\n"
            f"Write a flattering, charismatic Tinder bio for {user.display_name} based on their messages. "
            "Keep it confident, alpha, charming. Format with emojis and bullet points. "
            "Include: a one-line tagline, 3-4 short bullet points about them, and a closing line. "
            "Total 100 words max. Stay in character but be respectful."
        )
    else:
        bio_system = (
            cfg["system_prompt"] +
            "\n\n=== TINDER BIO MODE ===\n"
            f"Write a savage, brutally accurate Tinder bio for {user.display_name} based on their actual messages. "
            "Reference real patterns from their chat history. Be funny and ruthless but not actually mean. "
            "Format like a real bio: tagline, bullet points, closing line. Use emojis. "
            "Include height (something stupid like '5'7\" but lies on apps'), occupation (something absurd based on their vibe), and red flags. "
            "Total 100 words max. Stay in character."
        )

    bio = await ask_ai(
        bio_system,
        [{"role": "user", "content": f"Recent messages from {user.display_name}:\n\n{transcript}\n\nWrite their Tinder bio."}],
        {**cfg, "max_tokens": 350},
    )
    if bio.startswith("⚠️"):
        bio = "Bio could not be loaded. Even the AI gave up."

    age = random.randint(19, 42)
    distance = random.randint(1, 25)

    final = (
        f"📱 **TINDER PROFILE GENERATED** 📱\n\n"
        f"╭─────────────────────╮\n"
        f"┃ **{user.display_name}**, {age}\n"
        f"┃ 📍 {distance} miles away\n"
        f"╰─────────────────────╯\n\n"
        f"{bio}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💚 Like     ✖️ Nope     ⭐ Super Like\n"
        f"━━━━━━━━━━━━━━━━━━━━━"
    )
    try:
        await interaction.edit_original_response(content=final)
    except Exception:
        pass



@tree.command(name="roast", description="Generate a brutal personalized roast based on a user's recent messages.")
@discord.app_commands.describe(user="Who to roast")
async def roast_command(interaction: discord.Interaction, user: discord.Member):
    cfg = load_config()
    target_id = str(user.id)
    target_name = user.display_name

    # Don't roast the boss
    if target_id in cfg.get("respected_users", []):
        await interaction.response.send_message(
            f"❌ {user.mention} is the boss. I don't roast the boss.",
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    # Defer because gathering messages + AI call takes longer than 3 seconds
    await interaction.response.defer()

    async def edit(content: str):
        try:
            await interaction.edit_original_response(
                content=content,
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except Exception as e:
            log.warning("edit failed: %s", e)

    await edit(f"🔥 Loading ammunition on {user.mention}...")

    # Gather user's recent messages — pull DIRECTLY from Discord channel history
    # This is far more reliable than our local logs which can be sparse.
    user_messages: list[str] = []
    try:
        # Scan up to last 1000 messages in this channel for messages by target
        async for m in interaction.channel.history(limit=1000):
            if m.author.id == user.id and m.content and m.content.strip():
                user_messages.append(m.content.strip())
                if len(user_messages) >= 50:
                    break
    except Exception as e:
        log.warning("Channel history fetch failed: %s", e)

    # Also scour daily logs (cross-channel) as backup
    today = datetime.now(timezone.utc)
    target_name = user.display_name
    seen = set(user_messages)
    for days_back in range(7):
        date_str = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        for log_file in RECAP_DIR.glob(f"*_{date_str}.jsonl"):
            try:
                with open(log_file) as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("author") == target_name:
                                content = entry.get("content", "").strip()
                                if content and content not in seen:
                                    seen.add(content)
                                    user_messages.append(content)
                        except Exception:
                            continue
            except Exception:
                continue

    # Cap to most recent 80 messages (avoid huge payloads)
    user_messages = user_messages[:80]

    if len(user_messages) < 1:
        # Still allow a roast based purely on display name + reputation
        user_messages = [f"({target_name} barely talks)"]

    # Animation while AI generates
    await asyncio.sleep(0.6)
    await edit(f"🔍 Found {len(user_messages)} messages from {user.mention}...")
    await asyncio.sleep(0.8)
    await edit(f"⚙️ Analyzing patterns, typos, and bad takes...")
    await asyncio.sleep(0.8)
    await edit(f"🎯 Selecting weakest moments...")

    # Build the prompt
    transcript = "\n".join(f"- {m}" for m in user_messages)
    roast_system = (
        cfg["system_prompt"] +
        "\n\n=== ROAST MODE ===\n"
        f"You are about to roast {target_name}. Below are their actual recent messages from the server. "
        "If there's plenty of material, use SPECIFIC details — quote bad takes, mock typos, point out repeated phrases, "
        "expose contradictions. If they barely talk, roast them for being a lurker / NPC / silent observer instead. "
        "Either way be ruthless, witty, and devastating. 4 to 6 sentences max. "
        "Format as a clean paragraph. No bullet points. No 'here's your roast' preamble — just deliver. "
        "Stay completely in character."
    )
    user_prompt = (
        f"Roast {target_name} based on these recent messages of theirs:\n\n{transcript}"
    )

    try:
        roast = await ask_ai(
            roast_system,
            [{"role": "user", "content": user_prompt}],
            {**cfg, "max_tokens": 350, "temperature": min(cfg["temperature"] + 0.1, 1.0)},
        )
    except Exception as e:
        log.exception("Roast failed")
        await edit(f"⚠️ Couldn't generate roast: {e}")
        return

    if roast.startswith("⚠️"):
        await edit(roast)
        return

    final = (
        f"🔥 **ROAST OF {user.mention}** 🔥\n"
        f"_{len(user_messages)} messages analyzed_\n\n"
        f"{roast}"
    )
    # Final message — needs to ping the target so they see the roast
    if len(final) <= 2000:
        try:
            await interaction.edit_original_response(content=final)
        except Exception as e:
            log.warning("Final edit failed: %s", e)
    else:
        try:
            await interaction.edit_original_response(content=f"🔥 **ROAST OF {user.mention}** 🔥")
        except Exception:
            pass
        for i in range(0, len(roast), 1990):
            await interaction.followup.send(roast[i:i+1990])



@tree.command(name="roll", description="Roll a dice (1-100 by default).")
@discord.app_commands.describe(sides="Number of sides on the dice (default 100)")
async def roll_command(interaction: discord.Interaction, sides: int = 100):
    if sides < 2 or sides > 1_000_000:
        await interaction.response.send_message("Pick a sides count between 2 and 1,000,000.", ephemeral=True)
        return
    result = random.randint(1, sides)
    await interaction.response.send_message(f"🎲 **{result}** _(d{sides})_")


@tree.command(name="flip", description="Flip a coin.")
async def flip_command(interaction: discord.Interaction):
    result = random.choice(["🪙 **Heads**", "🪙 **Tails**"])
    # Animate
    await interaction.response.send_message("🪙 *flipping...*")
    msg = await interaction.original_response()
    await asyncio.sleep(1)
    await msg.edit(content="🪙 *spinning...*")
    await asyncio.sleep(1)
    await msg.edit(content=result)


@tree.command(name="8ball", description="Ask the magic 8-ball a question.")
@discord.app_commands.describe(question="Your yes/no question")
async def eightball_command(interaction: discord.Interaction, question: str):
    answers = [
        "Absolutely.", "No chance.", "Without a doubt.", "Definitely not.",
        "Ask again later.", "My sources say yes.", "Very doubtful.",
        "Outlook good.", "It is certain.", "Don't count on it.",
        "Signs point to yes.", "Reply hazy, try again.", "Cannot predict now.",
        "Better not tell you now.", "Concentrate and ask again.",
    ]
    answer = random.choice(answers)
    await interaction.response.send_message(f"🎱 **Q:** {question}\n**A:** {answer}")


@tree.command(name="rps", description="Play rock paper scissors against the bot.")
@discord.app_commands.describe(choice="rock, paper, or scissors")
@discord.app_commands.choices(choice=[
    discord.app_commands.Choice(name="🪨 Rock", value="rock"),
    discord.app_commands.Choice(name="📄 Paper", value="paper"),
    discord.app_commands.Choice(name="✂️ Scissors", value="scissors"),
])
async def rps_command(interaction: discord.Interaction, choice: discord.app_commands.Choice[str]):
    user = choice.value
    bot = random.choice(["rock", "paper", "scissors"])
    emoji = {"rock":"🪨","paper":"📄","scissors":"✂️"}
    if user == bot:
        result = "It's a tie!"
    elif (user == "rock" and bot == "scissors") or \
         (user == "paper" and bot == "rock") or \
         (user == "scissors" and bot == "paper"):
        result = "**You win!** 🎉"
    else:
        result = "**I win!** 😎"
    await interaction.response.send_message(
        f"{emoji[user]} vs {emoji[bot]}\n{result}"
    )


@tree.command(name="rate", description="Rate something out of 10.")
@discord.app_commands.describe(thing="What should I rate?")
async def rate_command(interaction: discord.Interaction, thing: str):
    score = random.randint(0, 10)
    bar = "🟩" * score + "⬜" * (10 - score)
    comments = {
        0:"absolute trash", 1:"barely registers", 2:"yikes", 3:"questionable",
        4:"meh", 5:"average", 6:"not bad", 7:"solid", 8:"impressive",
        9:"elite", 10:"perfection"
    }
    await interaction.response.send_message(
        f"**{thing}**\n{bar} **{score}/10** — _{comments[score]}_"
    )


@tree.command(name="ship", description="Ship two users together. See compatibility %.")
@discord.app_commands.describe(user1="First person (mention or name)", user2="Second person (mention or name)")
async def ship_command(interaction: discord.Interaction, user1: str, user2: str):
    # Try to resolve as Discord members first, fall back to plain text
    def resolve(s: str):
        s = s.strip()
        # Mention format <@123> or <@!123>
        m = re.match(r"<@!?(\d+)>", s)
        if m and interaction.guild:
            member = interaction.guild.get_member(int(m.group(1)))
            if member:
                return member.display_name, member.id, member.mention
        # Try by name in guild
        if interaction.guild:
            for member in interaction.guild.members:
                if s.lower() in (member.display_name.lower(), member.name.lower()):
                    return member.display_name, member.id, member.mention
        # Fallback: just use the string, no mention
        return s, hash(s.lower()), s

    name1, id1, mention1 = resolve(user1)
    name2, id2, mention2 = resolve(user2)

    seed = int(id1) ^ int(id2)
    rng = random.Random(seed)
    score = rng.randint(0, 100)
    bar_len = 20
    filled = round(score / 100 * bar_len)
    ship_name = name1[:max(1, len(name1)//2)] + name2[len(name2)//2:]
    if score < 20:    verdict = "💀 disaster"
    elif score < 40:  verdict = "😬 awkward"
    elif score < 60:  verdict = "🤔 mid"
    elif score < 80:  verdict = "😍 promising"
    else:             verdict = "💍 soulmates"

    header = f"💞 {mention1} + {mention2}\n*Calculating compatibility...*"
    # Suppress mention pings on the animation frames so we don't ping users repeatedly
    silent = discord.AllowedMentions.none()

    await interaction.response.send_message(
        f"💞 {mention1} + {mention2}\n\n🔍 *analyzing chemistry...*",
        allowed_mentions=silent,
    )
    msg = await interaction.original_response()
    await asyncio.sleep(1.0)

    # Loading bar fills up
    for i in range(1, bar_len + 1):
        bar = "💗" * i + "⬜" * (bar_len - i)
        await msg.edit(
            content=f"💞 {mention1} + {mention2}\n{bar}",
            allowed_mentions=silent,
        )
        await asyncio.sleep(0.12)

    await asyncio.sleep(0.4)
    # Brief suspense
    await msg.edit(
        content=f"💞 {mention1} + {mention2}\n{'💗' * bar_len}\n*results incoming...*",
        allowed_mentions=silent,
    )
    await asyncio.sleep(1.0)

    # Number rolls up to final score (last frame pings nobody to avoid double-ping)
    steps = max(score, 1)
    increments = list(range(0, score + 1, max(1, score // 12))) or [0]
    if increments[-1] != score:
        increments.append(score)
    for s in increments:
        f = round(s / 100 * bar_len)
        bar = "💗" * f + "🖤" * (bar_len - f)
        await msg.edit(
            content=f"💞 {mention1} + {mention2}\n{bar}  **{s}%**",
            allowed_mentions=silent,
        )
        await asyncio.sleep(0.18)

    # Final reveal — this one pings the actual users
    final_bar = "💗" * filled + "🖤" * (bar_len - filled)
    final = (
        f"## 💞 {mention1} + {mention2} = **{ship_name}**\n"
        f"{final_bar}\n"
        f"### **{score}%** — _{verdict}_"
    )
    await msg.edit(content=final)


# ── 🔫 /duel ─────────────────────────────────────────────────────────────────
@tree.command(name="duel", description="Quick 1v1 pistol duel (instant outcome, small wager).")
@discord.app_commands.describe(opponent="Who do you challenge?")
async def duel_command(interaction: discord.Interaction, opponent: discord.Member):
    if opponent.id == interaction.user.id:
        await interaction.response.send_message("🤔 You can't duel yourself.", ephemeral=True)
        return
    if opponent.bot:
        await interaction.response.send_message("🤖 You can't duel a bot.", ephemeral=True)
        return

    silent = discord.AllowedMentions.none()
    challenger = interaction.user

    await interaction.response.defer()
    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    await edit(f"🎩 **DUEL AT DAWN**\n{challenger.mention} has challenged {opponent.mention}!")
    await asyncio.sleep(1.5)
    await edit(f"🚶 {challenger.mention} and {opponent.mention} take 10 paces...")
    await asyncio.sleep(1.5)
    for i in range(10, 0, -2):
        await edit(f"🚶 ...{i} paces...")
        await asyncio.sleep(0.6)
    await edit(f"⏸️ ...they turn...")
    await asyncio.sleep(1.5)
    for n in [3, 2, 1]:
        await edit(f"# **{n}**")
        await asyncio.sleep(0.8)
    await edit("# 🔫 **DRAW!**")
    await asyncio.sleep(1.0)
    await edit("💥 **BANG!**\n💥 **BANG!**")
    await asyncio.sleep(1.5)

    winner, loser = random.sample([challenger, opponent], 2)
    flavors = [
        f"{loser.mention} fired wide. {winner.mention} did not.",
        f"{winner.mention}'s aim was true. {loser.mention} hits the dirt.",
        f"{loser.mention}'s gun jammed. {winner.mention} took the shot.",
        f"{winner.mention} was faster on the draw.",
        f"{loser.mention} blinked. That was all {winner.mention} needed.",
    ]
    # Economy: winner gets a prize
    duel_prize = random.randint(150, 400)
    new_bal = economy.add(winner.id, duel_prize, "duel win")
    economy.record_win(winner.id)
    economy.record_loss(loser.id)
    # Claim any bounty on the loser
    try:
        await claim_bounty(winner.id, loser.id, interaction.channel)
    except Exception:
        pass
    await trigger_game_win(winner.id, "duel", channel=interaction.channel)
    await trigger_balance_check(winner.id, channel=interaction.channel)
    final = (
        f"🎩 **DUEL RESULT** 🎩\n\n"
        f"{random.choice(flavors)}\n\n"
        f"## 🏆 {winner.mention} stands victorious.\n"
        f"💀 {loser.mention} lies in the dust.\n\n"
        f"💰 {winner.display_name} earned **{duel_prize:,}** coins!"
    )
    # Final message can ping both users — they earned it
    try:
        await interaction.edit_original_response(content=final)
    except Exception:
        pass


# ── 🎯 /gun (Russian roulette) ───────────────────────────────────────────────
@tree.command(name="gun", description="Russian roulette: 2-4 players take turns. One dies. Last alive wins the pot.")
@discord.app_commands.describe(player2="Player 2", player3="Player 3 (optional)", player4="Player 4 (optional)")
async def gun_command(
    interaction: discord.Interaction,
    player2: discord.Member,
    player3: discord.Member = None,
    player4: discord.Member = None,
):
    silent = discord.AllowedMentions.none()
    starter = interaction.user
    candidates = [starter, player2, player3, player4]
    seen = set()
    players = []
    for u in candidates:
        if u and u.id not in seen and not u.bot:
            seen.add(u.id)
            players.append(u)
    if len(players) < 2:
        await interaction.response.send_message("Need at least 2 unique humans.", ephemeral=True)
        return

    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    alive = list(players)
    chamber = random.randint(1, 6)  # bullet position
    pulls = 0
    log_lines = []

    await edit(
        "🎯 **RUSSIAN ROULETTE**\n\n"
        "One bullet. Six chambers. Take turns.\n\n"
        f"Players: {', '.join(p.mention for p in alive)}"
    )
    await asyncio.sleep(2.0)

    turn_idx = 0
    while len(alive) > 1:
        player = alive[turn_idx % len(alive)]
        pulls += 1
        # Suspense
        await edit(
            f"🎯 **{player.mention} picks up the gun...**\n\n"
            + "\n".join(log_lines[-5:])
        )
        await asyncio.sleep(1.4)
        await edit(
            f"🎯 {player.mention} spins the cylinder...\n"
            f"*click click click...*\n\n"
            + "\n".join(log_lines[-5:])
        )
        await asyncio.sleep(1.4)
        await edit(
            f"🎯 {player.mention} pulls the trigger...\n\n"
            + "\n".join(log_lines[-5:])
        )
        await asyncio.sleep(1.6)

        if pulls == chamber:
            # BANG
            log_lines.append(f"💥 **BANG!** {player.mention} is **OUT**.")
            alive.remove(player)
            await edit(
                f"# 💥 **BANG!**\n\n{player.mention} is out.\n\n"
                + "\n".join(log_lines[-5:])
            )
            await asyncio.sleep(2.0)
            # Reload — new bullet position, reset pulls
            chamber = random.randint(1, 6)
            pulls = 0
            # Don't increment turn_idx because we removed someone
            if turn_idx >= len(alive):
                turn_idx = 0
        else:
            log_lines.append(f"🔘 *click* — {player.mention} survives.")
            await edit(
                f"🔘 *click*\n\n{player.mention} survives.\n\n"
                + "\n".join(log_lines[-5:])
            )
            await asyncio.sleep(1.5)
            turn_idx += 1

    winner = alive[0]
    # Economy: bigger prize = more players
    gun_prize = 250 + (len(players) - 1) * 200  # 2p=450, 3p=650, 4p=850
    new_bal = economy.add(winner.id, gun_prize, "russian roulette win")
    economy.record_win(winner.id)
    for p in players:
        if p.id != winner.id:
            economy.record_loss(p.id)
    await trigger_game_win(winner.id, "gun", channel=interaction.channel)
    await trigger_balance_check(winner.id, channel=interaction.channel)
    final = (
        f"🎯 **GAME OVER**\n\n"
        + "\n".join(log_lines[-8:])
        + f"\n\n## 🏆 {winner.mention} is the last one standing!"
        + f"\n💰 Earned **{gun_prize:,}** coins!"
    )
    try:
        await interaction.edit_original_response(content=final)
    except Exception:
        pass


# ── 💰 /heist ────────────────────────────────────────────────────────────────
HEIST_LOOT = [
    ("💎", "diamond"), ("💵", "stack of bills"), ("💰", "money bag"),
    ("🏆", "golden trophy"), ("👑", "crown"), ("💍", "ring"),
    ("📿", "pearl necklace"), ("🪙", "gold coin"),
]


@tree.command(name="hack", description="Hack into a user's most embarrassing data.")
@discord.app_commands.describe(target="Who to hack")
async def hack_command(interaction: discord.Interaction, target: discord.Member):
    silent = discord.AllowedMentions.none()
    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    await edit(f"```\n> initiating hack on {target.display_name}...\n```")
    await asyncio.sleep(0.8)

    log_lines = [f"> initiating hack on {target.display_name}..."]
    stages = [
        "> bypassing firewall...",
        "> spoofing IP address...",
        "> connecting to mainframe...",
        "> [████░░░░░░] 40%",
        "> [████████░░] 80%",
        "> [██████████] 100%",
        "> ACCESS GRANTED",
        "> decrypting data...",
    ]
    for s in stages:
        log_lines.append(s)
        terminal = "```\n" + "\n".join(log_lines[-8:]) + "\n```"
        await edit(terminal)
        await asyncio.sleep(0.5)

    findings = random.sample(HACK_FAKE_FINDINGS, 4)
    final_terminal = (
        "```\n"
        + "\n".join(log_lines[-4:])
        + "\n\n"
        + "=== CONFIDENTIAL FILE ===\n"
        + f"target: {target.display_name}\n\n"
        + "\n".join(f"  - {f}" for f in findings)
        + "\n\n=== END OF FILE ===\n"
        + "```"
    )
    final = f"💻 **HACK SUCCESSFUL** on {target.mention}\n\n{final_terminal}"
    try:
        await interaction.edit_original_response(content=final)
    except Exception:
        pass


# ── 🥊 /rps-tournament ───────────────────────────────────────────────────────
@tree.command(name="rps-tournament", description="4-player rock paper scissors bracket.")
@discord.app_commands.describe(p2="Player 2", p3="Player 3", p4="Player 4")
async def rps_tournament_command(
    interaction: discord.Interaction,
    p2: discord.Member,
    p3: discord.Member,
    p4: discord.Member,
):
    silent = discord.AllowedMentions.none()
    starter = interaction.user
    candidates = [starter, p2, p3, p4]
    seen = set()
    players = []
    for u in candidates:
        if u and u.id not in seen and not u.bot:
            seen.add(u.id)
            players.append(u)
    if len(players) != 4:
        await interaction.response.send_message("Need exactly 4 unique players.", ephemeral=True)
        return

    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    moves = ["rock", "paper", "scissors"]
    emoji = {"rock":"🪨","paper":"📄","scissors":"✂️"}

    def battle(a, b):
        # Returns (winner, loser, a_move, b_move)
        while True:
            ma = random.choice(moves)
            mb = random.choice(moves)
            if ma == mb:
                continue  # ties replay
            if (ma == "rock" and mb == "scissors") or \
               (ma == "paper" and mb == "rock") or \
               (ma == "scissors" and mb == "paper"):
                return a, b, ma, mb
            return b, a, ma, mb

    bracket_lines = []
    bracket_lines.append("🥊 **RPS TOURNAMENT** 🥊\n")
    bracket_lines.append(f"Players: {' • '.join(p.mention for p in players)}\n")
    await edit("\n".join(bracket_lines))
    await asyncio.sleep(2.0)

    # Semis
    bracket_lines.append("**━━ SEMIFINALS ━━**")
    await edit("\n".join(bracket_lines))
    await asyncio.sleep(1.0)

    bracket_lines.append(f"⚔️ {players[0].mention} vs {players[1].mention}...")
    await edit("\n".join(bracket_lines))
    await asyncio.sleep(1.5)
    w1, l1, mw1, ml1 = battle(players[0], players[1])
    bracket_lines[-1] = f"⚔️ {players[0].mention} {emoji[mw1 if w1 == players[0] else ml1]} vs {emoji[mw1 if w1 == players[1] else ml1]} {players[1].mention} → **{w1.mention} wins**"
    await edit("\n".join(bracket_lines))
    await asyncio.sleep(1.5)

    bracket_lines.append(f"⚔️ {players[2].mention} vs {players[3].mention}...")
    await edit("\n".join(bracket_lines))
    await asyncio.sleep(1.5)
    w2, l2, mw2, ml2 = battle(players[2], players[3])
    bracket_lines[-1] = f"⚔️ {players[2].mention} {emoji[mw2 if w2 == players[2] else ml2]} vs {emoji[mw2 if w2 == players[3] else ml2]} {players[3].mention} → **{w2.mention} wins**"
    await edit("\n".join(bracket_lines))
    await asyncio.sleep(2.0)

    # Final
    bracket_lines.append("\n**━━ FINAL ━━**")
    bracket_lines.append(f"🏆 {w1.mention} vs {w2.mention}...")
    await edit("\n".join(bracket_lines))
    await asyncio.sleep(2.0)
    champ, runner, mc, mr = battle(w1, w2)
    bracket_lines[-1] = f"🏆 {w1.mention} {emoji[mc if champ == w1 else mr]} vs {emoji[mc if champ == w2 else mr]} {w2.mention}"
    bracket_lines.append(f"\n## 🏆 **{champ.mention} IS THE CHAMPION!**")
    try:
        await interaction.edit_original_response(content="\n".join(bracket_lines))
    except Exception:
        pass


# ── 🎰 /slots ────────────────────────────────────────────────────────────────
SLOT_SYMBOLS = ["🍒", "🍋", "🍇", "🔔", "💎", "7️⃣", "🍀", "⭐"]
SLOT_PAYOUT = {
    "💎": 1000, "7️⃣": 500, "⭐": 250, "🔔": 100,
    "🍀": 75, "🍇": 50, "🍒": 25, "🍋": 10,
}

@tree.command(name="slots", description="Pull the lever! Bet coins to spin the slot machine.")
@discord.app_commands.describe(bet="How many coins to bet (default 100)")
async def slots_command(interaction: discord.Interaction, bet: int = 100):
    silent = discord.AllowedMentions.none()
    user = interaction.user

    if bet <= 0:
        await interaction.response.send_message("Bet must be positive.", ephemeral=True)
        return
    if bet > economy.balance(user.id):
        await interaction.response.send_message(
            f"❌ You only have **{economy.balance(user.id):,}** coins.",
            ephemeral=True,
        )
        return

    # Take the bet upfront
    economy.add(user.id, -bet, "slots bet")
    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    # Determine final result first (~12% jackpot, 25% pair, rest no-match)
    roll = random.random()
    if roll < 0.12:
        # Jackpot — three of a kind
        s = random.choice(SLOT_SYMBOLS)
        final_reels = [s, s, s]
    elif roll < 0.37:
        # Pair somewhere
        s = random.choice(SLOT_SYMBOLS)
        odd = random.choice([x for x in SLOT_SYMBOLS if x != s])
        positions = [0, 1, 2]
        random.shuffle(positions)
        final_reels = ["", "", ""]
        final_reels[positions[0]] = s
        final_reels[positions[1]] = s
        final_reels[positions[2]] = odd
    else:
        # All different
        final_reels = random.sample(SLOT_SYMBOLS, 3)

    def render(reels, header="🎰 SLOT MACHINE 🎰"):
        return (
            f"## {header}\n\n"
            f"┏━━━━━━━━━━━━━━━┓\n"
            f"┃   {reels[0]}   {reels[1]}   {reels[2]}   ┃\n"
            f"┗━━━━━━━━━━━━━━━┛\n"
            f"      ⬆ ⬆ ⬆\n"
            f"   {user.mention}"
        )

    await edit(render(["🎰","🎰","🎰"], "🎰 SLOT MACHINE 🎰"))
    await asyncio.sleep(0.8)

    # Spin all three
    for _ in range(8):
        spinning = [random.choice(SLOT_SYMBOLS) for _ in range(3)]
        await edit(render(spinning, "🎰 SPINNING 🎰"))
        await asyncio.sleep(0.18)

    # Stop reel 1
    for _ in range(5):
        await edit(render([final_reels[0], random.choice(SLOT_SYMBOLS), random.choice(SLOT_SYMBOLS)], "🎰 SPINNING 🎰"))
        await asyncio.sleep(0.18)
    # Stop reel 2
    for _ in range(5):
        await edit(render([final_reels[0], final_reels[1], random.choice(SLOT_SYMBOLS)], "🎰 SPINNING 🎰"))
        await asyncio.sleep(0.18)
    # Stop reel 3
    await edit(render(final_reels, "🎰 SPINNING 🎰"))
    await asyncio.sleep(0.6)

    # Result — payouts are multiplied by the bet
    if final_reels[0] == final_reels[1] == final_reels[2]:
        # Jackpot: bet × 10 (or × 20 for diamond)
        symbol = final_reels[0]
        multiplier = 20 if symbol == "💎" else (10 if symbol == "7️⃣" else 7)
        winnings = bet * multiplier
        new_bal = economy.add(user.id, winnings, "slots jackpot")
        economy.record_win(user.id)
        result = (
            f"## 🎉 JACKPOT! 🎉\n\n"
            f"{user.mention} hit **3x {symbol}** ({multiplier}x payout)\n"
            f"Won {COIN_EMOJI} **{winnings:,}** coins!\n"
            f"Balance: **{new_bal:,}**"
        )
        await trigger_event(user.id, "slots_jackpot", channel=interaction.channel)
        if symbol == "💎":
            await trigger_event(user.id, "slots_diamonds", channel=interaction.channel)
        await trigger_balance_check(user.id, channel=interaction.channel)
    elif final_reels[0] == final_reels[1] or final_reels[1] == final_reels[2] or final_reels[0] == final_reels[2]:
        # Pair: bet × 1.5 (small profit)
        pair_symbol = max(set(final_reels), key=final_reels.count)
        winnings = int(bet * 1.5)
        new_bal = economy.add(user.id, winnings, "slots pair")
        economy.record_win(user.id)
        result = (
            f"## ✨ PAIR! ✨\n\n"
            f"{user.mention} hit **2x {pair_symbol}** (1.5x payout)\n"
            f"Won {COIN_EMOJI} **{winnings:,}** coins!\n"
            f"Balance: **{new_bal:,}**"
        )
    else:
        # Apply slots luck perk for retroactive pair check
        slots_luck_pct = get_perk(user.id, "slots_luck_pct")
        if slots_luck_pct and random.randint(1, 100) <= slots_luck_pct:
            # Pity pair payout
            winnings = int(bet * 1.5)
            new_bal = economy.add(user.id, winnings, "slots luck pity")
            economy.record_win(user.id)
            result = (
                f"## 🍀 LUCKY! 🍀\n\n"
                f"{user.mention}'s luck activated.\n"
                f"Won {COIN_EMOJI} **{winnings:,}** coins!\n"
                f"Balance: **{new_bal:,}**"
            )
        else:
            economy.record_loss(user.id)
            new_bal = economy.balance(user.id)
            result = (
                f"## 💸 NO MATCH 💸\n\n"
                f"{user.mention} lost **{bet:,}** coins.\n"
                f"Balance: **{new_bal:,}**"
            )

    final_display = render(final_reels, "🎰 RESULT 🎰") + "\n\n" + result
    try:
        await interaction.edit_original_response(content=final_display)
    except Exception:
        pass


# ── ⚖️ /court ───────────────────────────────────────────────────────────────
# ── 💰 Economy commands ──────────────────────────────────────────────────────
COIN_EMOJI = "💰"


@tree.command(name="balance", description="Check your or someone else's coin balance.")
@discord.app_commands.describe(user="Whose balance to check (defaults to you)")
async def balance_command(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user
    bal = economy.balance(target.id)
    stats = economy.stats(target.id)
    badges = get_user_badges(target.id)
    earned_count = len(_get_achievements(target.id))

    # Custom title
    custom_title = get_custom_title(target.id)
    title_suffix = f" [{custom_title}]" if custom_title else ""

    title = f"{COIN_EMOJI} {target.display_name}{title_suffix}'s Wallet"
    if badges:
        title = f"{badges} {target.display_name}{title_suffix}"

    embed = discord.Embed(title=title, color=discord.Color.gold())
    embed.add_field(name="💰 Balance", value=f"**{bal:,}** coins", inline=False)
    embed.add_field(name="Wins", value=str(stats.get("games_won", 0)), inline=True)
    embed.add_field(name="Losses", value=str(stats.get("games_lost", 0)), inline=True)
    embed.add_field(name="Total Earned", value=f"{stats.get('total_earned',0):,}", inline=True)

    # Pet earnings pending
    pets = _load_pets()
    pet = pets.get(str(target.id))
    if pet:
        info = PET_TYPES.get(pet["type"], {"emoji":"🐾","name":"?"})
        level = _pet_level(pet["xp"])
        hours_since = (time.time() - pet.get("last_collected", time.time())) / 3600
        pending = int(level * PET_DAILY_INCOME_BASE * (hours_since / 24))
        if _pet_hunger(pet) < 30:
            pending = pending // 2
        embed.add_field(
            name=f"{info['emoji']} Pet",
            value=f"**{pet['name']}** lvl {level} • Pending: {pending:,}",
            inline=True,
        )

    # Business pending
    user_bizs = _user_businesses(target.id)
    if user_bizs:
        total_hourly = sum(_business_income_per_hour(b) for b in user_bizs)
        total_pending = sum(_business_pending_income(b) for b in user_bizs)
        embed.add_field(
            name="🏢 Businesses",
            value=f"{len(user_bizs)} owned • {total_hourly:,}/hr • Pending: **{total_pending:,}**",
            inline=True,
        )

    # XP/Level
    xp = _get_xp(target.id)
    level = level_for_xp(xp)
    embed.add_field(
        name="📊 Level",
        value=f"**Lvl {level}** ({xp:,} XP) — _{get_user_title(target.id)}_",
        inline=True,
    )

    # Active boosts
    active_boosts = []
    if is_vip(target.id):
        active_boosts.append(f"💎 VIP ({fmt_cooldown(_user_active_remaining(target.id, 'vip_until'))})")
    if _user_is_active(target.id, "xp_boost_until"):
        active_boosts.append(f"⚡ 2x XP ({fmt_cooldown(_user_active_remaining(target.id, 'xp_boost_until'))})")
    if has_insurance(target.id):
        active_boosts.append(f"🛡️ Insurance ({fmt_cooldown(_user_active_remaining(target.id, 'insurance_until'))})")
    if has_heist_tools(target.id):
        active_boosts.append(f"🦝 Heist Tools ({fmt_cooldown(_user_active_remaining(target.id, 'heist_tools_until'))})")
    if is_protected(target.id):
        prot_data = _load_protections()
        remaining = max(0, int(prot_data.get(str(target.id), 0) - time.time()))
        active_boosts.append(f"🛡️ Rob Immunity ({fmt_cooldown(remaining)})")
    if active_boosts:
        embed.add_field(
            name="✨ Active Boosts",
            value="\n".join(active_boosts),
            inline=False,
        )

    embed.add_field(
        name=f"🏆 Achievements ({earned_count}/{len(ACHIEVEMENTS)})",
        value=(badges or "_none yet — try /achievements_"),
        inline=False,
    )
    # Show next-step suggestion only if checking your own balance
    if target.id == interaction.user.id:
        suggestion = suggest_next_step(target.id)
        if suggestion:
            embed.add_field(name="💡 Next Step", value=suggestion, inline=False)
    await interaction.response.send_message(embed=embed)


@tree.command(name="daily", description="Claim your daily coin reward.")
async def daily_command(interaction: discord.Interaction):
    user = interaction.user
    remaining = economy.get_cooldown_remaining(user.id, "daily")
    if remaining > 0:
        await interaction.response.send_message(
            f"⏰ Your daily resets in **{fmt_cooldown(remaining)}**.",
            ephemeral=True,
        )
        return
    await interaction.response.defer()
    silent = discord.AllowedMentions.none()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    reward = random.randint(DAILY_MIN, DAILY_MAX)
    economy.set_cooldown(user.id, "daily")

    await edit("📦 *Opening your daily box...*")
    await asyncio.sleep(1.0)
    await edit("📦 *Unwrapping...*")
    await asyncio.sleep(0.9)
    await edit("✨ *...*")
    await asyncio.sleep(0.7)
    # Apply daily_bonus_pct perk + passive_income perk
    bonus_pct = get_perk(user.id, "daily_bonus_pct")
    passive = get_perk(user.id, "passive_income")
    final_reward = int(reward * (1 + bonus_pct / 100)) + passive
    new_bal = economy.add(user.id, final_reward, "daily")
    track_economy_event("earned", final_reward)
    track_activity("daily", user.id, user.display_name, f"claimed daily {final_reward:,}")
    bonus_text = ""
    if bonus_pct or passive:
        bonus_text = f"\n_(+{bonus_pct}% bonus, +{passive} passive)_"
    suggestion = suggest_next_step(user.id)
    tip = f"\n\n{suggestion}" if suggestion else ""
    await edit(
        f"🎁 **DAILY REWARD!**\n\n"
        f"You found {COIN_EMOJI} **{final_reward:,}** coins!{bonus_text}\n"
        f"Balance: **{new_bal:,}**{tip}"
    )
    # Achievements
    await trigger_daily_claim(user.id, channel=interaction.channel)
    await trigger_balance_check(user.id, channel=interaction.channel)
    await check_streak_bonus(user.id, channel=interaction.channel)
    add_tournament_score(user.id, coins_earned=final_reward)
    track_quest_progress(user.id, "coins_earned", final_reward)


@tree.command(name="weekly", description="Claim your weekly coin reward (bigger but rarer).")
async def weekly_command(interaction: discord.Interaction):
    user = interaction.user
    remaining = economy.get_cooldown_remaining(user.id, "weekly")
    if remaining > 0:
        await interaction.response.send_message(
            f"⏰ Your weekly resets in **{fmt_cooldown(remaining)}**.",
            ephemeral=True,
        )
        return
    await interaction.response.defer()
    silent = discord.AllowedMentions.none()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    reward = random.randint(WEEKLY_MIN, WEEKLY_MAX)
    economy.set_cooldown(user.id, "weekly")

    await edit("💼 *Opening the weekly safe...*")
    await asyncio.sleep(1.0)
    await edit("🔓 *...combination accepted...*")
    await asyncio.sleep(1.0)
    await edit("💸 *...counting...*")
    await asyncio.sleep(1.0)
    new_bal = economy.add(user.id, reward, "weekly")
    track_economy_event("earned", reward)
    track_activity("weekly", user.id, user.display_name, f"claimed weekly {reward:,}")
    suggestion = suggest_next_step(user.id)
    tip = f"\n\n{suggestion}" if suggestion else ""
    await edit(
        f"💎 **WEEKLY REWARD!**\n\n"
        f"You scored {COIN_EMOJI} **{reward:,}** coins!\n"
        f"Balance: **{new_bal:,}**{tip}"
    )


WORK_JOBS = [
    ("📦", "Amazon warehouse"),
    ("🍕", "Pizza delivery"),
    ("🚗", "Uber driving"),
    ("☕", "Barista"),
    ("🛒", "DoorDashing"),
    ("📱", "Selling sneakers on StockX"),
    ("💻", "Freelance coding"),
    ("🎰", "Pit boss at the casino"),
    ("🛠️", "Handyman gig"),
    ("📸", "Selling stock photos"),
    ("🎤", "Open mic comedy"),
    ("🎮", "Streaming on Twitch"),
]


@tree.command(name="work", description="Work a random job for some coins. 45 min cooldown.")
async def work_command(interaction: discord.Interaction):
    user = interaction.user
    remaining = economy.get_cooldown_remaining(user.id, "work")
    if remaining > 0:
        await interaction.response.send_message(
            f"😴 You're tired. Rest for **{fmt_cooldown(remaining)}**.",
            ephemeral=True,
        )
        return
    await interaction.response.defer()
    silent = discord.AllowedMentions.none()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    emoji, job = random.choice(WORK_JOBS)
    reward = random.randint(WORK_MIN, WORK_MAX)
    work_bonus_pct = get_perk(user.id, "work_bonus_pct")
    if work_bonus_pct:
        reward = int(reward * (1 + work_bonus_pct / 100))
    economy.set_cooldown(user.id, "work")

    await edit(f"{emoji} *Clocking in at {job}...*")
    await asyncio.sleep(1.0)
    await edit(f"{emoji} *Working hard at {job}...*")
    await asyncio.sleep(1.2)
    await edit(f"{emoji} *Almost done at {job}...*")
    await asyncio.sleep(1.0)
    new_bal = economy.add(user.id, reward, "work")
    track_economy_event("earned", reward)
    track_activity("work", user.id, user.display_name, f"worked, earned {reward:,}")
    suggestion = suggest_next_step(user.id)
    tip = f"\n\n{suggestion}" if suggestion else ""
    await edit(
        f"{emoji} **JOB COMPLETE!**\n\n"
        f"You worked at *{job}* and earned {COIN_EMOJI} **{reward:,}** coins.\n"
        f"Balance: **{new_bal:,}**{tip}"
    )
    await trigger_work_used(user.id, channel=interaction.channel)
    await trigger_balance_check(user.id, channel=interaction.channel)


@tree.command(name="beg", description="Beg for spare change. Sometimes you get nothing.")
async def beg_command(interaction: discord.Interaction):
    user = interaction.user
    remaining = economy.get_cooldown_remaining(user.id, "beg")
    if remaining > 0:
        await interaction.response.send_message(
            f"🥺 People are tired of you. Try again in **{fmt_cooldown(remaining)}**.",
            ephemeral=True,
        )
        return
    await interaction.response.defer()
    silent = discord.AllowedMentions.none()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    economy.set_cooldown(user.id, "beg")
    # 25% nothing, 75% small reward
    if random.random() < 0.25:
        flavors = [
            "A stranger spit on you. Disgusting.",
            "Someone laughed at you and walked away.",
            "Someone gave you a piece of gum. You ate it. No coins.",
            "A pigeon stole your hat.",
            "You got told to get a job. You got nothing.",
        ]
        await edit(f"🥺 *begging on the corner...*")
        await asyncio.sleep(1.2)
        await edit(f"😔 {random.choice(flavors)}\n\nYou got **0** coins.")
        return

    reward = random.randint(50, 250)
    await edit(f"🥺 *begging on the corner...*")
    await asyncio.sleep(1.0)
    await edit(f"🤲 *someone is approaching...*")
    await asyncio.sleep(1.0)
    new_bal = economy.add(user.id, reward, "beg")
    track_economy_event("earned", reward)
    track_activity("beg", user.id, user.display_name, f"begged, got {reward:,}")
    flavors = [
        f"A kind stranger gave you {reward} coins.",
        f"You found {reward} coins on the ground.",
        f"A drunk guy threw {reward} coins at you.",
        f"You did a sad face. Worth {reward} coins.",
    ]
    await edit(
        f"🥺 **BEG SUCCESS**\n\n"
        f"{random.choice(flavors)}\n"
        f"Balance: **{new_bal:,}**"
        + (f"\n\n{suggest_next_step(user.id)}" if suggest_next_step(user.id) else "")
    )


@tree.command(name="rob", description="Try to rob another user. 45% success rate. Risky.")
@discord.app_commands.describe(target="Who to rob")
async def rob_command(interaction: discord.Interaction, target: discord.Member):
    user = interaction.user
    if target.id == user.id:
        await interaction.response.send_message("You can't rob yourself.", ephemeral=True)
        return
    if target.bot:
        await interaction.response.send_message("You can't rob a bot.", ephemeral=True)
        return

    remaining = economy.get_cooldown_remaining(user.id, "rob")
    if remaining > 0:
        await interaction.response.send_message(
            f"🚨 You're laying low. Try again in **{fmt_cooldown(remaining)}**.",
            ephemeral=True,
        )
        return

    target_bal = economy.balance(target.id)
    if target_bal < ROB_MIN_TARGET_BALANCE:
        await interaction.response.send_message(
            f"💸 {target.mention} is too broke to rob (under **{ROB_MIN_TARGET_BALANCE}** coins).",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    # Check if target is protected
    if is_protected(target.id):
        await interaction.response.send_message(
            f"🛡️ {target.mention} has rob immunity. You can't touch them.",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    await interaction.response.defer()
    silent = discord.AllowedMentions.none()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    economy.set_cooldown(user.id, "rob")
    track_quest_progress(user.id, "rob_attempts")

    await edit(f"🦝 {user.mention} is sneaking up on {target.mention}...")
    await asyncio.sleep(1.5)
    await edit(f"🦝 *picking the lock...*")
    await asyncio.sleep(1.5)
    await edit(f"🦝 *reaching into the wallet...*")
    await asyncio.sleep(1.5)

    # Apply rob protection perk on target
    target_protection_pct = get_perk(target.id, "rob_protection_pct")
    if target_protection_pct and random.randint(1, 100) <= target_protection_pct:
        await edit(
            f"🛡️ **{target.mention}'s instincts kicked in.**\n\n"
            f"They sensed the threat. {user.mention} bailed empty-handed."
        )
        return

    if random.random() < ROB_SUCCESS_CHANCE + (HEIST_TOOLS_BOOST if has_heist_tools(user.id) else 0):
        # Success
        pct = random.uniform(ROB_PERCENT_MIN, ROB_PERCENT_MAX)
        amount = max(1, int(target_bal * pct))
        economy.add(target.id, -amount, "robbed")
        new_bal = economy.add(user.id, amount, "rob success")
        await edit(
            f"💰 **ROBBERY SUCCESSFUL!**\n\n"
            f"{user.mention} stole **{amount:,}** coins from {target.mention}!\n"
            f"{user.display_name}'s balance: **{new_bal:,}**"
        )
        await trigger_game_win(user.id, "rob", channel=interaction.channel)
        await trigger_event(target.id, "got_robbed", channel=interaction.channel)
        await trigger_balance_check(user.id, channel=interaction.channel)
        # Claim any bounty on the victim
        try:
            await claim_bounty(user.id, target.id, interaction.channel)
        except Exception:
            pass
        # DM the victim
        try:
            embed = discord.Embed(
                title="💸 YOU GOT ROBBED",
                description=(
                    f"**{user.display_name}** robbed you for **{amount:,}** coins.\n"
                    f"Your new balance: **{economy.balance(target.id):,}**\n\n"
                    f"_Consider buying `/protect` to prevent future robberies._"
                ),
                color=discord.Color.dark_red(),
            )
            await send_dm(target.id, "rob", embed=embed)
        except Exception:
            pass
    else:
        # Caught — pay a fine
        fine = random.randint(ROB_FINE_MIN, ROB_FINE_MAX)
        current = economy.balance(user.id)
        actual_fine = min(fine, current)
        economy.add(user.id, -actual_fine, "rob caught")
        # Half goes to victim as compensation
        comp = actual_fine // 2
        if comp > 0:
            economy.add(target.id, comp, "robbery comp")
        new_bal = economy.balance(user.id)
        await edit(
            f"🚨 **YOU GOT CAUGHT!**\n\n"
            f"{user.mention} got tackled trying to rob {target.mention}.\n"
            f"Paid a fine of **{actual_fine:,}** coins. {target.display_name} got **{comp:,}** in compensation.\n"
            f"{user.display_name}'s balance: **{new_bal:,}**"
        )


@tree.command(name="pay", description="Send coins to another user.")
@discord.app_commands.describe(user="Who to pay", amount="How many coins")
async def pay_command(interaction: discord.Interaction, user: discord.Member, amount: int):
    sender = interaction.user
    if user.id == sender.id:
        await interaction.response.send_message("You can't pay yourself.", ephemeral=True)
        return
    if user.bot:
        await interaction.response.send_message("Bots don't accept payments.", ephemeral=True)
        return
    if amount <= 0:
        await interaction.response.send_message("Amount must be positive.", ephemeral=True)
        return

    # Anti-abuse: max single transfer based on sender balance to prevent silent farming
    MAX_PAY_AMOUNT = 50_000
    if amount > MAX_PAY_AMOUNT:
        await interaction.response.send_message(
            f"❌ Max single transfer is **{MAX_PAY_AMOUNT:,}** coins. Send multiple payments.",
            ephemeral=True,
        )
        return

    # Anti-abuse: cooldown to prevent rapid alt-account farming
    pay_cd = economy.get_cooldown_remaining(sender.id, "pay")
    if pay_cd > 0:
        await interaction.response.send_message(
            f"⏰ You can send another payment in **{fmt_cooldown(pay_cd)}**.",
            ephemeral=True,
        )
        return

    success = economy.transfer(sender.id, user.id, amount)
    if not success:
        bal = economy.balance(sender.id)
        await interaction.response.send_message(
            f"❌ Insufficient funds. You have **{bal:,}** coins.",
            ephemeral=True,
        )
        return

    economy.set_cooldown(sender.id, "pay")

    # Quest tracking + balance check for both parties
    track_quest_progress(sender.id, "coins_given", amount)
    await trigger_balance_check(user.id, channel=interaction.channel)

    sender_bal = economy.balance(sender.id)
    await interaction.response.send_message(
        f"💸 {sender.mention} sent {COIN_EMOJI} **{amount:,}** to {user.mention}.\n"
        f"{sender.display_name}'s balance: **{sender_bal:,}**",
        allowed_mentions=discord.AllowedMentions(users=[user]),
    )


@tree.command(name="leaderboard", description="See the richest users in the server.")
async def leaderboard_command(interaction: discord.Interaction):
    await interaction.response.defer()
    top = economy.leaderboard(10)
    if not top:
        await interaction.edit_original_response(content="No one has any coins yet.")
        return

    lines = []
    medals = ["🥇","🥈","🥉"]
    for i, (uid, data) in enumerate(top):
        try:
            member = interaction.guild.get_member(int(uid)) if interaction.guild else None
            name = member.display_name if member else f"User {uid}"
        except Exception:
            name = f"User {uid}"
        prefix = medals[i] if i < 3 else f"`#{i+1}`"
        badges = get_user_badges(int(uid))
        badge_str = f" {badges}" if badges else ""
        lines.append(f"{prefix} **{name}**{badge_str} — {COIN_EMOJI} {data.get('balance', 0):,}")

    embed = discord.Embed(
        title="🏆 Richest Users",
        description="\n".join(lines),
        color=discord.Color.gold(),
    )
    await interaction.edit_original_response(embed=embed)


@tree.command(name="bet", description="Bet coins on a coinflip. Double or nothing.")
@discord.app_commands.describe(amount="How many coins to bet")
async def bet_command(interaction: discord.Interaction, amount: int):
    user = interaction.user
    if amount <= 0:
        await interaction.response.send_message("Amount must be positive.", ephemeral=True)
        return
    if amount > economy.balance(user.id):
        await interaction.response.send_message(
            f"❌ You only have **{economy.balance(user.id):,}** coins.",
            ephemeral=True,
        )
        return

    await interaction.response.defer()
    silent = discord.AllowedMentions.none()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    await edit(f"🪙 {user.mention} bets **{amount:,}** coins. Flipping...")
    await asyncio.sleep(0.8)
    await edit("🪙 *spinning...*")
    await asyncio.sleep(1.0)
    await edit("🪙 *spinning...*")
    await asyncio.sleep(1.0)

    if random.random() < 0.5:
        new_bal = economy.add(user.id, amount, "bet win")
        economy.record_win(user.id)
        # Track quests/tournament
        track_quest_progress(user.id, "games_won")
        track_quest_progress(user.id, "games_played")
        track_quest_progress(user.id, "coins_earned", amount)
        add_tournament_score(user.id, coins_earned=amount, games_won=1)
        await trigger_balance_check(user.id, channel=interaction.channel)
        await edit(
            f"## 🎉 YOU WON!\n"
            f"{user.mention} doubled up — gained **{amount:,}** coins!\n"
            f"Balance: **{new_bal:,}**"
        )
    else:
        new_bal = economy.add(user.id, -amount, "bet loss")
        economy.record_loss(user.id)
        track_quest_progress(user.id, "games_played")
        await edit(
            f"## 💀 YOU LOST!\n"
            f"{user.mention} lost **{amount:,}** coins.\n"
            f"Balance: **{new_bal:,}**"
        )


# ── 🃏 /blackjack ────────────────────────────────────────────────────────────
CARD_SUITS = ["♠", "♥", "♦", "♣"]
CARD_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


def _new_deck() -> list:
    return [(r, s) for r in CARD_RANKS for s in CARD_SUITS]


def _hand_value(hand: list) -> int:
    total = 0
    aces = 0
    for rank, _ in hand:
        if rank in ("J", "Q", "K"):
            total += 10
        elif rank == "A":
            total += 11
            aces += 1
        else:
            total += int(rank)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def _render_hand(hand: list, hide_first: bool = False) -> str:
    cards = []
    for i, (r, s) in enumerate(hand):
        if hide_first and i == 0:
            cards.append("`[??]`")
        else:
            cards.append(f"`[{r}{s}]`")
    return " ".join(cards)


# Track active blackjack games per user
active_blackjack: dict[int, dict] = {}


class BlackjackView(discord.ui.View):
    def __init__(self, user_id: int, bet: int):
        super().__init__(timeout=120)
        self.user_id = user_id
        self.bet = bet

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your game.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Hit", style=discord.ButtonStyle.primary, emoji="🎯")
    async def hit_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._hit(interaction)

    @discord.ui.button(label="Stand", style=discord.ButtonStyle.success, emoji="✋")
    async def stand_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._stand(interaction)

    async def _hit(self, interaction: discord.Interaction):
        game = active_blackjack.get(self.user_id)
        if not game:
            await interaction.response.send_message("Game not found.", ephemeral=True)
            return
        # Draw a card
        card = game["deck"].pop()
        game["player"].append(card)
        player_total = _hand_value(game["player"])

        if player_total > 21:
            # Bust
            del active_blackjack[self.user_id]
            economy.record_loss(self.user_id)
            new_bal = economy.balance(self.user_id)
            self.disable_all()
            await interaction.response.edit_message(
                content=(
                    f"🃏 **BLACKJACK** — Bet: {self.bet:,}\n\n"
                    f"**Dealer:** {_render_hand(game['dealer'])}\n"
                    f"**You:** {_render_hand(game['player'])} = **{player_total}**\n\n"
                    f"## 💥 BUST!\n"
                    f"Lost **{self.bet:,}** coins.\n"
                    f"Balance: **{new_bal:,}**"
                ),
                view=self,
            )
            return

        # Continue
        await interaction.response.edit_message(
            content=(
                f"🃏 **BLACKJACK** — Bet: {self.bet:,}\n\n"
                f"**Dealer:** {_render_hand(game['dealer'], hide_first=True)} = ?\n"
                f"**You:** {_render_hand(game['player'])} = **{player_total}**"
            ),
            view=self,
        )

    async def _stand(self, interaction: discord.Interaction):
        game = active_blackjack.get(self.user_id)
        if not game:
            await interaction.response.send_message("Game not found.", ephemeral=True)
            return

        # Dealer plays
        while _hand_value(game["dealer"]) < 17:
            game["dealer"].append(game["deck"].pop())

        player_total = _hand_value(game["player"])
        dealer_total = _hand_value(game["dealer"])
        del active_blackjack[self.user_id]

        # Resolve
        outcome = ""
        winnings = 0
        if dealer_total > 21:
            winnings = self.bet * 2
            outcome = "## 🎉 DEALER BUSTS! YOU WIN!"
            economy.record_win(self.user_id)
        elif player_total > dealer_total:
            winnings = self.bet * 2
            outcome = "## 🎉 YOU WIN!"
            economy.record_win(self.user_id)
        elif player_total == dealer_total:
            winnings = self.bet  # push, get bet back
            outcome = "## 🤝 PUSH (tie). Bet refunded."
        else:
            winnings = 0
            outcome = "## 💀 DEALER WINS"
            economy.record_loss(self.user_id)

        if winnings > 0:
            economy.add(self.user_id, winnings, "blackjack")
        new_bal = economy.balance(self.user_id)

        self.disable_all()
        await interaction.response.edit_message(
            content=(
                f"🃏 **BLACKJACK** — Bet: {self.bet:,}\n\n"
                f"**Dealer:** {_render_hand(game['dealer'])} = **{dealer_total}**\n"
                f"**You:** {_render_hand(game['player'])} = **{player_total}**\n\n"
                f"{outcome}\n"
                f"{('Won **' + format(winnings, ',') + '** coins.') if winnings > self.bet else ('Lost **' + format(self.bet, ',') + '** coins.') if winnings == 0 else 'Bet refunded.'}\n"
                f"Balance: **{new_bal:,}**"
            ),
            view=self,
        )

    def disable_all(self):
        for child in self.children:
            child.disabled = True

    async def on_timeout(self):
        # If a game is still active, refund bet
        if self.user_id in active_blackjack:
            economy.add(self.user_id, self.bet, "blackjack timeout refund")
            del active_blackjack[self.user_id]


@tree.command(name="blackjack", description="Play blackjack against the dealer.")
@discord.app_commands.describe(bet="How many coins to bet")
async def blackjack_command(interaction: discord.Interaction, bet: int):
    user = interaction.user
    if bet <= 0:
        await interaction.response.send_message("Bet must be positive.", ephemeral=True)
        return
    if user.id in active_blackjack:
        await interaction.response.send_message("You already have a hand in play.", ephemeral=True)
        return
    if bet > economy.balance(user.id):
        await interaction.response.send_message(
            f"❌ You only have **{economy.balance(user.id):,}** coins.",
            ephemeral=True,
        )
        return

    # Take bet upfront
    economy.add(user.id, -bet, "blackjack bet")

    deck = _new_deck()
    random.shuffle(deck)
    player = [deck.pop(), deck.pop()]
    dealer = [deck.pop(), deck.pop()]
    active_blackjack[user.id] = {"deck": deck, "player": player, "dealer": dealer}

    player_total = _hand_value(player)

    # Natural blackjack on the deal?
    if player_total == 21:
        # Pay 2.5x for natural (plus blackjack_payout_pct perk)
        bj_perk = get_perk(user.id, "blackjack_payout_pct")
        winnings = int(bet * (2.5 + bj_perk / 100))
        economy.add(user.id, winnings, "blackjack natural")
        economy.record_win(user.id)
        del active_blackjack[user.id]
        new_bal = economy.balance(user.id)
        await interaction.response.send_message(
            f"🃏 **BLACKJACK!**\n\n"
            f"**Dealer:** {_render_hand(dealer, hide_first=True)}\n"
            f"**You:** {_render_hand(player)} = **21**\n\n"
            f"## 🎉 NATURAL BLACKJACK! 2.5x payout!\n"
            f"Won **{winnings:,}** coins!\n"
            f"Balance: **{new_bal:,}**"
        )
        await trigger_event(user.id, "blackjack_natural", channel=interaction.channel)
        await trigger_balance_check(user.id, channel=interaction.channel)
        track_quest_progress(user.id, "games_won")
        track_quest_progress(user.id, "games_played")
        track_quest_progress(user.id, "coins_earned", winnings)
        add_tournament_score(user.id, coins_earned=winnings, games_won=1)
        return

    view = BlackjackView(user.id, bet)
    await interaction.response.send_message(
        f"🃏 **BLACKJACK** — Bet: {bet:,}\n\n"
        f"**Dealer:** {_render_hand(dealer, hide_first=True)} = ?\n"
        f"**You:** {_render_hand(player)} = **{player_total}**\n\n"
        f"Hit or Stand?",
        view=view,
    )


# ── 🎭 /crime ────────────────────────────────────────────────────────────────
@tree.command(name="crime", description="Commit a random crime. Win or lose coins.")
async def crime_command(interaction: discord.Interaction):
    user = interaction.user
    cfg = load_config()
    silent = discord.AllowedMentions.none()

    remaining = economy.get_cooldown_remaining(user.id, "crime")
    if remaining > 0:
        await interaction.response.send_message(
            f"🚨 You're laying low. Try again in **{fmt_cooldown(remaining)}**.",
            ephemeral=True,
        )
        return

    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    economy.set_cooldown(user.id, "crime")

    # AI generates the crime story
    success = random.random() < 0.55
    crime_system = (
        cfg["system_prompt"] +
        "\n\n=== CRIME GENERATOR ===\n"
        f"You are narrating {user.display_name}'s petty crime. Write a SHORT 2-sentence story "
        f"about them committing a random crime (mugging, scamming, smuggling, picking pockets, "
        f"running a Ponzi scheme, fake check, NFT rug pull, etc.). "
        f"The crime ended in {'SUCCESS' if success else 'FAILURE'}. "
        "Make it funny, specific, and creative. Stay in character. No preamble — just the story."
    )

    await edit(f"🦹 *{user.display_name} is up to no good...*")
    await asyncio.sleep(1.0)
    await edit(f"🦹 *something is happening...*")
    await asyncio.sleep(1.0)

    try:
        story = await ask_ai(
            crime_system,
            [{"role": "user", "content": "Generate the crime story."}],
            {**cfg, "max_tokens": 200, "temperature": min(cfg["temperature"] + 0.1, 1.0)},
        )
    except Exception:
        story = ""
    if story.startswith("⚠️") or not story.strip():
        # Fallback flavor
        if success:
            story = f"{user.display_name} pulled off a quick scam and slipped into the night."
        else:
            story = f"{user.display_name} got caught on camera. Tragic."

    if success:
        amount = random.randint(150, 800)
        new_bal = economy.add(user.id, amount, "crime success")
        economy.record_win(user.id)
        result = f"💰 **+{amount:,} coins**\nBalance: **{new_bal:,}**"
        header = "## 🦹 CRIME SUCCESSFUL"
        await trigger_game_win(user.id, "crime", channel=interaction.channel)
        await trigger_balance_check(user.id, channel=interaction.channel)
    else:
        fine = min(random.randint(100, 400), economy.balance(user.id))
        new_bal = economy.add(user.id, -fine, "crime failed")
        economy.record_loss(user.id)
        result = f"💸 **-{fine:,} coins** (fined)\nBalance: **{new_bal:,}**"
        header = "## 🚔 CAUGHT!"

    await edit(f"{header}\n\n{story}\n\n{result}")


# ── 💍 /marry ────────────────────────────────────────────────────────────────
MARRIAGES_FILE = MEMORY_DIR / "marriages.json"

def _load_marriages() -> dict:
    if MARRIAGES_FILE.exists():
        try:
            with open(MARRIAGES_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_marriages(data: dict):
    with open(MARRIAGES_FILE, "w") as f:
        json.dump(data, f, indent=2)


class MarryView(discord.ui.View):
    def __init__(self, proposer_id: int, target_id: int):
        super().__init__(timeout=60)
        self.proposer_id = proposer_id
        self.target_id = target_id
        self.responded = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.target_id:
            await interaction.response.send_message("This proposal isn't for you.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Accept 💍", style=discord.ButtonStyle.success)
    async def accept(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.responded = True
        marriages = _load_marriages()
        # Check both are single
        if str(self.proposer_id) in marriages or str(self.target_id) in marriages:
            await interaction.response.edit_message(
                content="❌ Someone is already married. This proposal is void.",
                view=None,
            )
            return
        # Marry them
        marriages[str(self.proposer_id)] = {"spouse": str(self.target_id), "since": time.time()}
        marriages[str(self.target_id)] = {"spouse": str(self.proposer_id), "since": time.time()}
        _save_marriages(marriages)
        # Wedding bonus to both
        bonus = 1000
        economy.add(self.proposer_id, bonus, "wedding gift")
        economy.add(self.target_id, bonus, "wedding gift")
        # Achievements
        chan = interaction.channel
        await trigger_event(self.proposer_id, "married", channel=chan)
        await trigger_event(self.target_id, "married", channel=chan)
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(
            content=(
                f"💒 **WEDDING BELLS** 💒\n\n"
                f"<@{self.proposer_id}> and <@{self.target_id}> are now married!\n\n"
                f"💰 Both received **{bonus:,}** coin wedding gifts!\n"
                f"💑 Use `/divorce` to end it (with consequences)."
            ),
            view=self,
        )

    @discord.ui.button(label="Reject 💔", style=discord.ButtonStyle.danger)
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.responded = True
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(
            content=f"💔 <@{self.target_id}> rejected the proposal. Brutal.",
            view=self,
        )

    async def on_timeout(self):
        if not self.responded:
            for child in self.children:
                child.disabled = True


@tree.command(name="marry", description="Propose marriage to another user.")
@discord.app_commands.describe(user="Who you're proposing to")
async def marry_command(interaction: discord.Interaction, user: discord.Member):
    proposer = interaction.user
    if user.id == proposer.id:
        await interaction.response.send_message("You can't marry yourself.", ephemeral=True)
        return
    if user.bot:
        await interaction.response.send_message("Bots can't marry.", ephemeral=True)
        return

    marriages = _load_marriages()
    if str(proposer.id) in marriages:
        spouse_id = marriages[str(proposer.id)]["spouse"]
        await interaction.response.send_message(
            f"❌ You're already married to <@{spouse_id}>. Use `/divorce` first.",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return
    if str(user.id) in marriages:
        await interaction.response.send_message(
            f"❌ {user.mention} is already married.",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    silent = discord.AllowedMentions.none()
    await interaction.response.defer()

    async def edit(content, view=None):
        kwargs = {"content": content, "allowed_mentions": silent}
        if view is not None:
            kwargs["view"] = view
        try:
            await interaction.edit_original_response(**kwargs)
        except Exception:
            pass

    await edit("💍 *getting down on one knee...*")
    await asyncio.sleep(1.5)
    await edit("💍 💍 *opening the box...*")
    await asyncio.sleep(1.5)
    await edit(
        f"# 💍 PROPOSAL\n\n"
        f"{proposer.mention} drops to one knee in front of {user.mention}...\n\n"
        f"*Will you accept?*"
    )
    await asyncio.sleep(1.5)

    view = MarryView(proposer.id, user.id)
    # The proposal needs to ping the target user
    await interaction.edit_original_response(
        content=(
            f"# 💍 PROPOSAL 💍\n\n"
            f"{proposer.mention} has proposed to {user.mention}!\n\n"
            f"{user.mention}, do you accept?"
        ),
        view=view,
        allowed_mentions=discord.AllowedMentions(users=[user]),
    )


@tree.command(name="divorce", description="End your marriage. Costs coins.")
async def divorce_command(interaction: discord.Interaction):
    user = interaction.user
    marriages = _load_marriages()
    if str(user.id) not in marriages:
        await interaction.response.send_message("You're not married.", ephemeral=True)
        return

    spouse_id = int(marriages[str(user.id)]["spouse"])
    cost = 500
    bal = economy.balance(user.id)
    actual = min(cost, bal)
    economy.add(user.id, -actual, "divorce")
    await trigger_event(user.id, "divorced", channel=interaction.channel)

    del marriages[str(user.id)]
    if str(spouse_id) in marriages:
        del marriages[str(spouse_id)]
    _save_marriages(marriages)

    await interaction.response.send_message(
        f"💔 **DIVORCED**\n\n"
        f"{user.mention} divorced <@{spouse_id}>.\n"
        f"Lawyer fees: **{actual:,}** coins.",
        allowed_mentions=discord.AllowedMentions.none(),
    )


# ── 🎡 /wheel ────────────────────────────────────────────────────────────────
WHEEL_OUTCOMES = [
    {"label": "💰 +1000 coins", "type": "gain", "amount": 1000, "weight": 8},
    {"label": "💰 +500 coins",  "type": "gain", "amount": 500,  "weight": 12},
    {"label": "💰 +250 coins",  "type": "gain", "amount": 250,  "weight": 16},
    {"label": "💀 -500 coins",  "type": "loss", "amount": 500,  "weight": 12},
    {"label": "💀 -250 coins",  "type": "loss", "amount": 250,  "weight": 14},
    {"label": "🎰 Double balance", "type": "double", "weight": 3},
    {"label": "💸 Lose half balance", "type": "halve", "weight": 4},
    {"label": "🔄 Swap balance with random user", "type": "swap", "weight": 3},
    {"label": "🍀 Nothing happens", "type": "nothing", "weight": 10},
    {"label": "💎 JACKPOT +5000", "type": "gain", "amount": 5000, "weight": 1},
]


@tree.command(name="wheel", description="Spin the wheel of fortune for a random outcome.")
async def wheel_command(interaction: discord.Interaction):
    user = interaction.user
    silent = discord.AllowedMentions.none()
    remaining = economy.get_cooldown_remaining(user.id, "wheel")
    if remaining > 0:
        await interaction.response.send_message(
            f"⏰ The wheel needs to cool down. Try again in **{fmt_cooldown(remaining)}**.",
            ephemeral=True,
        )
        return

    economy.set_cooldown(user.id, "wheel")
    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    # Animation: cycle through outcomes
    await edit(f"🎡 {user.mention} spins the wheel...")
    await asyncio.sleep(0.8)
    for _ in range(8):
        sample = random.choice(WHEEL_OUTCOMES)
        await edit(f"🎡 *spinning...*\n\n> {sample['label']}")
        await asyncio.sleep(0.25)
    for _ in range(4):
        sample = random.choice(WHEEL_OUTCOMES)
        await edit(f"🎡 *slowing...*\n\n> {sample['label']}")
        await asyncio.sleep(0.5)

    # Pick weighted outcome with wheel_luck_pct perk
    weights = [o["weight"] for o in WHEEL_OUTCOMES]
    luck_pct = get_perk(user.id, "wheel_luck_pct")
    if luck_pct:
        # Boost positive outcomes by perk percentage
        weights = [
            int(w * (1 + luck_pct / 100)) if WHEEL_OUTCOMES[i]["type"] in ("gain", "double") else w
            for i, w in enumerate(weights)
        ]
    outcome = random.choices(WHEEL_OUTCOMES, weights=weights, k=1)[0]

    label = outcome["label"]
    result_text = ""
    if outcome["type"] == "gain":
        new_bal = economy.add(user.id, outcome["amount"], "wheel gain")
        result_text = f"You gained **{outcome['amount']:,}** coins!\nBalance: **{new_bal:,}**"
        if outcome["amount"] >= 5000:
            await trigger_event(user.id, "wheel_jackpot", channel=interaction.channel)
        await trigger_balance_check(user.id, channel=interaction.channel)
    elif outcome["type"] == "loss":
        actual = min(outcome["amount"], economy.balance(user.id))
        new_bal = economy.add(user.id, -actual, "wheel loss")
        result_text = f"You lost **{actual:,}** coins.\nBalance: **{new_bal:,}**"
    elif outcome["type"] == "double":
        bal = economy.balance(user.id)
        new_bal = economy.add(user.id, bal, "wheel double")
        result_text = f"Your balance was **doubled**!\nBalance: **{new_bal:,}**"
    elif outcome["type"] == "halve":
        bal = economy.balance(user.id)
        loss = bal // 2
        new_bal = economy.add(user.id, -loss, "wheel halve")
        result_text = f"You lost **{loss:,}** coins (half).\nBalance: **{new_bal:,}**"
    elif outcome["type"] == "swap":
        # Pick a random other user with a balance
        others = []
        if interaction.guild:
            for m in interaction.guild.members:
                if m.id != user.id and not m.bot and economy.balance(m.id) > 0:
                    others.append(m)
        if others:
            target = random.choice(others)
            user_bal = economy.balance(user.id)
            target_bal = economy.balance(target.id)
            # Set both balances
            u = economy._user(user.id)
            t = economy._user(target.id)
            u["balance"] = target_bal
            t["balance"] = user_bal
            economy._save()
            result_text = (
                f"🔄 You swapped balances with {target.mention}!\n"
                f"You now have **{target_bal:,}** coins, they have **{user_bal:,}**."
            )
        else:
            result_text = "No one to swap with. Spared by chance."
    else:  # nothing
        result_text = "Nothing happens. The universe ignored you."

    await edit(
        f"# 🎡 THE WHEEL HAS SPOKEN\n\n"
        f"## > {label}\n\n"
        f"{result_text}"
    )


# ── 🎰 /casino — hub menu ─────────────────────────────────────────────────────
@tree.command(name="casino", description="Welcome to the casino. See all gambling games.")
async def casino_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🎰 WELCOME TO THE CASINO 🎰",
        description=f"_{interaction.user.display_name}, what's your poison?_",
        color=discord.Color.gold(),
    )
    embed.add_field(name="🃏 /blackjack `<bet>`", value="Beat the dealer to 21", inline=False)
    embed.add_field(name="🎰 /slots `<bet>`",     value="Spin the reels • up to 20x payout", inline=False)
    embed.add_field(name="🪙 /bet `<amount>`",    value="50/50 coinflip • double or nothing", inline=False)
    embed.add_field(name="🎡 /wheel",             value="Spin for a random outcome", inline=False)
    embed.add_field(name="🦹 /crime",             value="Risk it for a profit", inline=False)
    embed.add_field(name="🎯 /gun",               value="Russian roulette w/ friends", inline=False)
    embed.add_field(name="💰 /heist",             value="Rob a bank with your crew", inline=False)
    embed.add_field(name="💸 /rob `<user>`",      value="Pickpocket someone", inline=False)
    bal = economy.balance(interaction.user.id)
    embed.set_footer(text=f"💰 Your balance: {bal:,} coins")
    await interaction.response.send_message(embed=embed)


# ── 🔍 /analyze ──────────────────────────────────────────────────────────────
@tree.command(name="analyze", description="Scan and analyze your last 50 messages.")
async def analyze_command(interaction: discord.Interaction):
    cfg = load_config()
    user = interaction.user
    silent = discord.AllowedMentions.none()

    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    # Animated scanning sequence
    await edit(f"🔍 *scanning {user.mention}'s message history...*")
    await asyncio.sleep(0.9)

    # Pull their last 50 messages from the current channel
    user_messages: list[str] = []
    try:
        async for m in interaction.channel.history(limit=2000):
            if m.author.id == user.id and m.content and m.content.strip():
                user_messages.append(m.content.strip())
                if len(user_messages) >= 50:
                    break
    except Exception as e:
        log.warning("Channel history fetch failed: %s", e)

    if not user_messages:
        await edit(f"🔍 No messages found for {user.mention} in this channel.")
        return

    await edit(f"🔍 *found {len(user_messages)} messages...*")
    await asyncio.sleep(0.9)
    await edit("🧠 *analyzing tone, patterns, vocabulary...*")
    await asyncio.sleep(0.9)
    await edit("📊 *cross-referencing personality markers...*")
    await asyncio.sleep(0.9)
    await edit("✍️ *writing report...*")
    await asyncio.sleep(0.6)

    # Quick stats we can compute locally
    total_chars = sum(len(m) for m in user_messages)
    avg_len = total_chars // len(user_messages)
    word_count = sum(len(m.split()) for m in user_messages)
    questions = sum(1 for m in user_messages if "?" in m)
    exclamations = sum(1 for m in user_messages if "!" in m)
    laughs = sum(1 for m in user_messages if any(l in m.lower() for l in ["lol", "lmao", "lmfao", "😂", "💀", "haha"]))
    caps_msgs = sum(1 for m in user_messages if len(m) > 3 and m.isupper())

    transcript = "\n".join(f"- {m}" for m in user_messages)

    analysis_system = (
        cfg["system_prompt"] +
        "\n\n=== MESSAGE ANALYZER MODE ===\n"
        f"You are analyzing {user.display_name}'s last {len(user_messages)} messages from this channel. "
        "Provide a brutally honest but funny analysis. Format as 4-5 short bullet points covering: "
        "their vibe / personality, common topics they bring up, recurring phrases or quirks, "
        "their texting style, and one final 'verdict' line. "
        "Reference specific things they actually said when relevant. Stay completely in character. "
        "Each bullet is 1 sentence max. No preamble. Just the bullets."
    )

    try:
        analysis = await ask_ai(
            analysis_system,
            [{"role": "user", "content": f"Messages from {user.display_name}:\n\n{transcript}"}],
            {**cfg, "max_tokens": 400},
        )
    except Exception:
        analysis = ""
    if analysis.startswith("⚠️") or not analysis.strip():
        analysis = "Analysis failed. The AI looked at your messages and just left."

    # Build the final report card
    report = (
        f"# 🔍 Analysis: {user.mention}\n"
        f"_Based on last **{len(user_messages)}** messages_\n\n"
        f"{analysis}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"**📊 Stats**\n"
        f"• Total words: **{word_count:,}**\n"
        f"• Avg message length: **{avg_len}** chars\n"
        f"• Questions asked: **{questions}**\n"
        f"• Exclamations: **{exclamations}**\n"
        f"• Laughing reactions: **{laughs}**\n"
        f"• ALL CAPS rants: **{caps_msgs}**"
    )

    if len(report) <= 2000:
        await edit(report)
    else:
        await edit(report[:1990])
        for i in range(1990, len(report), 1990):
            await interaction.followup.send(report[i:i+1990])


# ─────────────────────────────────────────────────────────────────────────────
# 🛒 REDEMPTION COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

# ── 🎨 /buyrole ──────────────────────────────────────────────────────────────
ROLE_PRICES = {
    "24h":       2000,
    "perm":     15000,
}
ROLE_COLORS = {
    "red":     (0xED4245, "🟥 Red"),
    "orange":  (0xE67E22, "🟧 Orange"),
    "yellow":  (0xF1C40F, "🟨 Yellow"),
    "green":   (0x2ECC71, "🟩 Green"),
    "blue":    (0x3498DB, "🟦 Blue"),
    "purple":  (0x9B59B6, "🟪 Purple"),
    "pink":    (0xEB459E, "💗 Pink"),
    "cyan":    (0x1ABC9C, "🩵 Cyan"),
    "gold":    (0xF1C40F, "🟨 Gold"),
    "white":   (0xFFFFFF, "⬜ White"),
}

# Track temporary roles for cleanup
TEMP_ROLES_FILE = MEMORY_DIR / "temp_roles.json"

def _load_temp_roles() -> list:
    if TEMP_ROLES_FILE.exists():
        try:
            with open(TEMP_ROLES_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_temp_roles(data: list):
    with open(TEMP_ROLES_FILE, "w") as f:
        json.dump(data, f, indent=2)


@tree.command(name="buyrole", description="Buy a colored Discord role.")
@discord.app_commands.describe(
    color="Pick a color",
    duration="How long? 24h or permanent",
    custom_name="Optional custom role name (max 30 chars)",
)
@discord.app_commands.choices(
    color=[discord.app_commands.Choice(name=label, value=key) for key, (_, label) in ROLE_COLORS.items()],
    duration=[
        discord.app_commands.Choice(name=f"24 hours ({ROLE_PRICES['24h']:,} coins)", value="24h"),
        discord.app_commands.Choice(name=f"Permanent ({ROLE_PRICES['perm']:,} coins)", value="perm"),
    ],
)
async def buyrole_command(
    interaction: discord.Interaction,
    color: discord.app_commands.Choice[str],
    duration: discord.app_commands.Choice[str],
    custom_name: str = None,
):
    user = interaction.user
    if not interaction.guild:
        await interaction.response.send_message("Server only.", ephemeral=True)
        return

    price = ROLE_PRICES[duration.value]
    bal = economy.balance(user.id)
    if bal < price:
        await interaction.response.send_message(
            f"❌ You need **{price:,}** coins. You have **{bal:,}**.",
            ephemeral=True,
        )
        return

    # Check bot permissions
    bot_member = interaction.guild.me
    if not bot_member.guild_permissions.manage_roles:
        await interaction.response.send_message(
            "❌ I need the **Manage Roles** permission. Ask an admin.",
            ephemeral=True,
        )
        return

    color_int, color_label = ROLE_COLORS[color.value]
    role_name = (custom_name[:30] if custom_name else f"💎 {user.display_name}").strip()
    if not role_name:
        role_name = f"💎 {user.display_name}"

    await interaction.response.defer(ephemeral=False)

    silent = discord.AllowedMentions.none()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    await edit(f"🎨 *Crafting your role...*")
    await asyncio.sleep(1.0)

    try:
        role = await interaction.guild.create_role(
            name=role_name,
            colour=discord.Colour(color_int),
            reason=f"Purchased by {user.display_name}",
            mentionable=False,
        )
        # Move it just below the bot's top role so it's visible
        try:
            top_bot_role = bot_member.top_role
            await role.edit(position=max(top_bot_role.position - 1, 1))
        except Exception:
            pass
        await user.add_roles(role, reason="buyrole purchase")
    except discord.Forbidden:
        await edit("❌ I can't manage roles. Move my role higher than the roles I should manage.")
        return
    except Exception as e:
        log.exception("buyrole failed")
        await edit(f"❌ Failed to create role: {e}")
        return

    # Charge the user
    economy.add(user.id, -price, "buyrole")
    new_bal = economy.balance(user.id)

    # If 24h, schedule cleanup
    if duration.value == "24h":
        temp = _load_temp_roles()
        temp.append({
            "role_id": role.id,
            "guild_id": interaction.guild.id,
            "user_id": user.id,
            "expires_at": time.time() + 24 * 3600,
        })
        _save_temp_roles(temp)
        duration_label = "24 hours"
    else:
        duration_label = "permanently"

    final = (
        f"🎨 **ROLE PURCHASED!**\n\n"
        f"{user.mention} bought {color_label} role **{role_name}** for **{price:,}** coins!\n"
        f"Lasts **{duration_label}**.\n\n"
        f"Balance: **{new_bal:,}**"
    )
    await edit(final)


# Background task to expire 24h roles
async def expire_temp_roles():
    await client.wait_until_ready()
    while not client.is_closed():
        await asyncio.sleep(600)  # check every 10 min
        try:
            temp = _load_temp_roles()
            now = time.time()
            still_active = []
            for entry in temp:
                if entry["expires_at"] <= now:
                    # Remove the role
                    try:
                        guild = client.get_guild(entry["guild_id"])
                        if guild:
                            role = guild.get_role(entry["role_id"])
                            if role:
                                await role.delete(reason="24h temp role expired")
                    except Exception as e:
                        log.warning("Temp role cleanup failed: %s", e)
                else:
                    still_active.append(entry)
            if len(still_active) != len(temp):
                _save_temp_roles(still_active)
        except Exception as e:
            log.warning("expire_temp_roles loop error: %s", e)


# ── 🛡️ /protect ──────────────────────────────────────────────────────────────
PROTECT_FILE = MEMORY_DIR / "protect.json"
PROTECT_PRICE = 1500
PROTECT_DURATION = 12 * 3600  # 12 hours


def _load_protections() -> dict:
    if PROTECT_FILE.exists():
        try:
            with open(PROTECT_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_protections(data: dict):
    with open(PROTECT_FILE, "w") as f:
        json.dump(data, f, indent=2)

def is_protected(user_id: int) -> bool:
    data = _load_protections()
    expiry = data.get(str(user_id), 0)
    return time.time() < expiry


@tree.command(name="protect", description="Buy 12-hour rob immunity.")
async def protect_command(interaction: discord.Interaction):
    user = interaction.user
    bal = economy.balance(user.id)

    # Check existing protection
    data = _load_protections()
    existing_expiry = data.get(str(user.id), 0)
    if time.time() < existing_expiry:
        remaining = int(existing_expiry - time.time())
        await interaction.response.send_message(
            f"🛡️ You're already protected for another **{fmt_cooldown(remaining)}**.",
            ephemeral=True,
        )
        return

    if bal < PROTECT_PRICE:
        await interaction.response.send_message(
            f"❌ Costs **{PROTECT_PRICE:,}** coins. You have **{bal:,}**.",
            ephemeral=True,
        )
        return

    economy.add(user.id, -PROTECT_PRICE, "protect")
    data[str(user.id)] = time.time() + PROTECT_DURATION
    _save_protections(data)
    new_bal = economy.balance(user.id)

    silent = discord.AllowedMentions.none()
    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    await edit("🛡️ *hiring private security...*")
    await asyncio.sleep(1.0)
    await edit("🛡️ *installing biometric locks...*")
    await asyncio.sleep(1.0)
    await edit(
        f"## 🛡️ PROTECTION ACTIVATED\n\n"
        f"{user.mention} is **immune to /rob** for the next **12 hours**.\n"
        f"Cost: **{PROTECT_PRICE:,}** coins.\n"
        f"Balance: **{new_bal:,}**"
    )


# ── 📢 /megaphone ────────────────────────────────────────────────────────────
MEGAPHONE_PRICE = 5000

@tree.command(name="megaphone", description="Have Jordan announce your message to the channel (with @here ping).")
@discord.app_commands.describe(message="What you want announced (max 200 chars)")
async def megaphone_command(interaction: discord.Interaction, message: str):
    user = interaction.user
    if len(message) > 200:
        await interaction.response.send_message("Message too long (max 200 chars).", ephemeral=True)
        return
    bal = economy.balance(user.id)
    if bal < MEGAPHONE_PRICE:
        await interaction.response.send_message(
            f"❌ Costs **{MEGAPHONE_PRICE:,}** coins. You have **{bal:,}**.",
            ephemeral=True,
        )
        return

    # Sanitize — no @everyone or role mentions
    safe_message = message.replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")

    economy.add(user.id, -MEGAPHONE_PRICE, "megaphone")
    new_bal = economy.balance(user.id)

    await interaction.response.defer()

    silent = discord.AllowedMentions.none()

    async def edit(content, allow_here=False):
        kw = {"content": content}
        if allow_here:
            kw["allowed_mentions"] = discord.AllowedMentions(everyone=True, users=False, roles=False)
        else:
            kw["allowed_mentions"] = silent
        try:
            await interaction.edit_original_response(**kw)
        except Exception:
            pass

    await edit("📢 *Jordan grabs the megaphone...*")
    await asyncio.sleep(1.0)
    await edit("📢 *clears throat...*")
    await asyncio.sleep(1.0)

    # Final megaphone message — pings @here
    final = (
        f"📢 **MEGAPHONE** 📢\n\n"
        f"@here — {user.mention} paid {MEGAPHONE_PRICE:,} coins to say:\n\n"
        f"# > {safe_message}"
    )
    try:
        await interaction.edit_original_response(
            content=final,
            allowed_mentions=discord.AllowedMentions(everyone=True, users=[user], roles=False),
        )
    except Exception:
        pass


# ── 🎰 /lottery ──────────────────────────────────────────────────────────────
LOTTERY_FILE = MEMORY_DIR / "lottery.json"
LOTTERY_TICKET_PRICE = 100
LOTTERY_MAX_TICKETS_PER_USER = 50


def _load_lottery() -> dict:
    if LOTTERY_FILE.exists():
        try:
            with open(LOTTERY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"jackpot": 1000, "tickets": {}, "history": []}


def _save_lottery(data: dict):
    with open(LOTTERY_FILE, "w") as f:
        json.dump(data, f, indent=2)


@tree.command(name="lottery", description="Buy lottery tickets or check the jackpot.")
@discord.app_commands.describe(tickets="How many tickets to buy (leave empty just to view)")
async def lottery_command(interaction: discord.Interaction, tickets: int = 0):
    user = interaction.user
    data = _load_lottery()

    # Just viewing
    if tickets == 0:
        my_tickets = data["tickets"].get(str(user.id), 0)
        total_tickets = sum(data["tickets"].values())
        chance = (my_tickets / total_tickets * 100) if total_tickets > 0 else 0
        embed = discord.Embed(
            title="🎰 LOTTERY",
            description=f"Tickets: **{LOTTERY_TICKET_PRICE:,}** coins each. Drawing happens daily at the recap hour.",
            color=discord.Color.gold(),
        )
        embed.add_field(name="💰 Current Jackpot", value=f"**{data['jackpot']:,}** coins", inline=False)
        embed.add_field(name="🎟️ Your Tickets", value=str(my_tickets), inline=True)
        embed.add_field(name="🎟️ Total Tickets", value=str(total_tickets), inline=True)
        embed.add_field(name="📊 Your Win Chance", value=f"{chance:.1f}%", inline=True)
        if data.get("history"):
            last = data["history"][-1]
            embed.add_field(
                name="📜 Last Winner",
                value=f"<@{last['winner_id']}> won **{last['amount']:,}** on {last['date']}",
                inline=False
            )
        embed.set_footer(text="Use /lottery tickets:N to buy. Max 50 per drawing.")
        await interaction.response.send_message(embed=embed)
        return

    if tickets <= 0:
        await interaction.response.send_message("Tickets must be positive.", ephemeral=True)
        return

    current_tickets = data["tickets"].get(str(user.id), 0)
    if current_tickets + tickets > LOTTERY_MAX_TICKETS_PER_USER:
        await interaction.response.send_message(
            f"❌ Max {LOTTERY_MAX_TICKETS_PER_USER} tickets per user per drawing. You already have {current_tickets}.",
            ephemeral=True,
        )
        return

    cost = tickets * LOTTERY_TICKET_PRICE
    discount_pct = get_perk(user.id, "lottery_discount_pct")
    if discount_pct:
        cost = int(cost * (1 - discount_pct / 100))
    bal = economy.balance(user.id)
    if bal < cost:
        await interaction.response.send_message(
            f"❌ {tickets} tickets cost **{cost:,}** coins. You have **{bal:,}**.",
            ephemeral=True,
        )
        return

    economy.add(user.id, -cost, "lottery tickets")
    data["tickets"][str(user.id)] = current_tickets + tickets
    # Half of ticket cost goes to jackpot
    data["jackpot"] += cost // 2
    _save_lottery(data)

    new_bal = economy.balance(user.id)
    total_tickets = sum(data["tickets"].values())
    chance = ((current_tickets + tickets) / total_tickets * 100) if total_tickets > 0 else 0

    await interaction.response.send_message(
        f"🎟️ **{tickets:,}** tickets purchased for {cost:,} coins!\n"
        f"💰 Jackpot: **{data['jackpot']:,}**\n"
        f"📊 Your win chance: **{chance:.1f}%**\n"
        f"Balance: **{new_bal:,}**"
    )


# Background task: daily lottery drawing
async def lottery_drawing_scheduler():
    """Runs lottery drawing once per day at the configured recap hour."""
    await client.wait_until_ready()
    while not client.is_closed():
        await asyncio.sleep(300)  # check every 5 min
        cfg = load_config()
        target_hour = cfg.get("daily_recap_hour_utc", 4)
        now = datetime.now(timezone.utc)
        if now.hour != target_hour:
            continue
        today_str = now.strftime("%Y-%m-%d")
        if get_last_fire("lottery") == today_str:
            continue
        # Mark fired FIRST to prevent races on redeploy
        set_last_fire("lottery", today_str)

        try:
            data = _load_lottery()
            if not data["tickets"]:
                continue

            # Build weighted draw — each user weighted by their ticket count
            entries = []
            for uid, count in data["tickets"].items():
                entries.extend([uid] * count)
            if not entries:
                continue
            winner_uid = random.choice(entries)
            jackpot = data["jackpot"]

            # Pay out
            economy.add(int(winner_uid), jackpot, "lottery jackpot")
            economy.record_win(int(winner_uid))
            # Apply lottery multiplier item if present
            try:
                winner_u = economy._user(int(winner_uid))
                mult = winner_u.get("lottery_mult", 1)
                if mult > 1:
                    bonus = jackpot * (mult - 1)
                    economy.add(int(winner_uid), bonus, "lottery multiplier")
                    winner_u["lottery_mult"] = 1  # consumed
                    economy._save()
                    if channel:
                        await channel.send(
                            f"🎰 **<@{winner_uid}>'s lottery multiplier activated! +{bonus:,} bonus coins!**",
                            allowed_mentions=discord.AllowedMentions.none(),
                        )
            except Exception:
                pass
            # DM the winner
            try:
                embed = discord.Embed(
                    title="🎰 YOU WON THE LOTTERY!",
                    description=(
                        f"You just won **{jackpot:,}** coins in the daily lottery!\n"
                        f"💰 New balance: **{economy.balance(int(winner_uid)):,}**\n\n"
                        f"_Buy more tickets with `/lottery` for tomorrow's drawing._"
                    ),
                    color=discord.Color.gold(),
                )
                await send_dm(int(winner_uid), "lottery", embed=embed)
            except Exception:
                pass
            # Try to grant achievement (no channel ref handy here but channel below works)
            try:
                if channel:
                    await trigger_event(int(winner_uid), "lottery_won", channel=channel)
                    await trigger_balance_check(int(winner_uid), channel=channel)
            except Exception:
                pass
            data["history"] = data.get("history", [])
            data["history"].append({
                "winner_id": winner_uid,
                "amount": jackpot,
                "date": today_str,
                "tickets": data["tickets"][winner_uid],
                "total_tickets": len(entries),
            })
            data["history"] = data["history"][-30:]  # keep last 30 drawings
            # Reset for next round
            data["tickets"] = {}
            data["jackpot"] = 1000  # seed for next round
            _save_lottery(data)

            # Announce in the recap channel if configured
            recap_channel_id = get_notification_channel_id(cfg)
            if recap_channel_id:
                try:
                    channel = client.get_channel(int(recap_channel_id))
                    if channel:
                        await channel.send(
                            f"🎰 **DAILY LOTTERY DRAWING** 🎰\n\n"
                            f"## 🏆 Winner: <@{winner_uid}>\n"
                            f"💰 Won **{jackpot:,}** coins!\n\n"
                            f"_New round starts now. Buy tickets with `/lottery tickets:N`_",
                        )
                except Exception as e:
                    log.warning("Lottery announce failed: %s", e)

            log.info("Lottery drawn: %s won %s", winner_uid, jackpot)
        except Exception as e:
            log.exception("Lottery drawing error: %s", e)


# ── 🛒 /shop — interactive shop hub ──────────────────────────────────────────

# ── Shop item prices/timings (must be defined before view classes use them) ─
INSURANCE_PRICE = 3_000
INSURANCE_DURATION_HOURS = 48
VIP_PRICE = 5_000
VIP_DURATION_DAYS = 7
XP_BOOST_PRICE = 1_500
XP_BOOST_DURATION_HOURS = 24
CUSTOM_TITLE_PRICE = 2_500
LOTTERY_MULT_PRICE = 1_000
LOAN_AMOUNTS = {
    1000:  (1000, 1200,  24),
    5000:  (5000, 6500,  48),
    25000: (25000, 35000, 72),
}
BOUNTY_MIN = 500
PET_FOOD_BUNDLE_PRICE = 250
PET_FOOD_BUNDLE_DAYS = 7
BUSINESS_UPGRADE_PRICE_PER_LEVEL = 5_000
BUSINESS_UPGRADE_BOOST = 0.10
BUSINESS_UPGRADE_MAX_LEVEL = 5
HEIST_TOOLS_PRICE = 2_000
HEIST_TOOLS_DURATION_HOURS = 24
HEIST_TOOLS_BOOST = 0.20
BOUNTIES_FILE = MEMORY_DIR / "bounties.json"
LOANS_FILE = MEMORY_DIR / "loans.json"


class ShopMainView(discord.ui.View):
    """Top-level shop menu — buttons for each category. Page 1 of 3."""
    def __init__(self, user_id: int):
        super().__init__(timeout=180)
        self.user_id = user_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("This isn't your shop session. Open your own with `/shop`.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Colored Role", style=discord.ButtonStyle.primary, emoji="🎨", row=0)
    async def role_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        embed = discord.Embed(
            title="🎨 Buy a Colored Role",
            description=(
                "Get a custom-colored Discord role next to your name.\n\n"
                f"**24 hours** — {ROLE_PRICES['24h']:,} coins\n"
                f"**Permanent** — {ROLE_PRICES['perm']:,} coins\n\n"
                "Pick a color below."
            ),
            color=discord.Color.purple(),
        )
        bal = economy.balance(self.user_id)
        embed.set_footer(text=f"💰 Your balance: {bal:,}")
        await interaction.response.edit_message(embed=embed, view=ShopColorView(self.user_id))

    @discord.ui.button(label="Rob Immunity", style=discord.ButtonStyle.success, emoji="🛡️", row=0)
    async def protect_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        bal = economy.balance(self.user_id)
        already = is_protected(self.user_id)
        if already:
            data = _load_protections()
            remaining = int(data.get(str(self.user_id), 0) - time.time())
            await interaction.response.send_message(
                f"🛡️ You're already protected for another **{fmt_cooldown(remaining)}**.",
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title="🛡️ Rob Immunity",
            description=(
                f"For **{PROTECT_PRICE:,} coins**, no one can `/rob` you for **12 hours**.\n\n"
                "Click confirm to purchase."
            ),
            color=discord.Color.green(),
        )
        embed.set_footer(text=f"💰 Your balance: {bal:,}")
        await interaction.response.edit_message(
            embed=embed, view=ShopConfirmProtectView(self.user_id)
        )

    @discord.ui.button(label="Megaphone", style=discord.ButtonStyle.danger, emoji="📢", row=0)
    async def megaphone_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        bal = economy.balance(self.user_id)
        if bal < MEGAPHONE_PRICE:
            await interaction.response.send_message(
                f"❌ Costs **{MEGAPHONE_PRICE:,}** coins. You have **{bal:,}**.",
                ephemeral=True,
            )
            return
        await interaction.response.send_modal(MegaphoneModal(self.user_id))

    @discord.ui.button(label="Lottery", style=discord.ButtonStyle.secondary, emoji="🎰", row=1)
    async def lottery_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        data = _load_lottery()
        my_tickets = data["tickets"].get(str(self.user_id), 0)
        total_tickets = sum(data["tickets"].values())
        chance = (my_tickets / total_tickets * 100) if total_tickets > 0 else 0

        embed = discord.Embed(
            title="🎰 Lottery",
            description=(
                f"Buy tickets for **{LOTTERY_TICKET_PRICE:,}** coins each.\n"
                f"Drawing happens daily. Max 50 tickets per user per round.\n\n"
                f"💰 **Jackpot:** {data['jackpot']:,} coins\n"
                f"🎟️ **Your tickets:** {my_tickets}\n"
                f"📊 **Win chance:** {chance:.1f}%\n"
                f"🎟️ **Total tickets sold:** {total_tickets}"
            ),
            color=discord.Color.gold(),
        )
        bal = economy.balance(self.user_id)
        embed.set_footer(text=f"💰 Your balance: {bal:,}")
        await interaction.response.edit_message(
            embed=embed, view=ShopLotteryView(self.user_id)
        )

    @discord.ui.button(label="Pets", style=discord.ButtonStyle.primary, emoji="🐶", row=1)
    async def pets_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        bal = economy.balance(self.user_id)
        pets = _load_pets()
        has_pet = str(self.user_id) in pets

        desc_lines = [
            f"Adopt a pet for **{PET_ADOPT_COST:,}** coins.",
            "Pets earn passive coins, level up, need feeding.",
            "",
        ]
        if has_pet:
            p = pets[str(self.user_id)]
            info = PET_TYPES.get(p["type"], {"emoji":"🐾","name":"?"})
            desc_lines.append(f"❌ You already have **{info['emoji']} {p['name']}**. Use `/abandon` first to adopt a new one.")
        else:
            desc_lines.append("Pick a pet type below:")

        embed = discord.Embed(
            title="🐶 Adopt a Pet",
            description="\n".join(desc_lines),
            color=discord.Color.teal(),
        )
        embed.set_footer(text=f"💰 Your balance: {bal:,}")
        await interaction.response.edit_message(
            embed=embed, view=ShopPetView(self.user_id, has_pet=has_pet)
        )

    @discord.ui.button(label="Businesses", style=discord.ButtonStyle.success, emoji="🏢", row=1)
    async def businesses_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        bal = economy.balance(self.user_id)
        user_bizs = _user_businesses(self.user_id)
        total_hourly = sum(_business_income_per_hour(b) for b in user_bizs)

        embed = discord.Embed(
            title="🏢 Buy a Business",
            description=(
                "Businesses earn passive income 24/7.\n"
                "Hire employees with `/hire` to boost income.\n"
                f"_Same-tier purchases cost {int((BUSINESS_QUANTITY_MULTIPLIER-1)*100)}% more each._\n\n"
                f"📊 You own **{len(user_bizs)}** businesses earning **{total_hourly:,}/hr**"
            ),
            color=discord.Color.dark_green(),
        )
        embed.set_footer(text=f"💰 Your balance: {bal:,}")
        await interaction.response.edit_message(
            embed=embed, view=ShopBusinessView(self.user_id, page=0)
        )

    @discord.ui.button(label="Page 2 ▶", style=discord.ButtonStyle.secondary, row=2)
    async def page2_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _show_shop_page2(interaction, self.user_id, edit=True)

    @discord.ui.button(label="Close", style=discord.ButtonStyle.secondary, emoji="✖️", row=2)
    async def close_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        embed = discord.Embed(
            title="🛒 Shop closed",
            description="Come back anytime with `/shop`.",
            color=discord.Color.greyple(),
        )
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(embed=embed, view=None)


class ShopPage2View(discord.ui.View):
    """Page 2 — Boosts & Cosmetics."""
    def __init__(self, user_id: int):
        super().__init__(timeout=180)
        self.user_id = user_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your shop.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="VIP Badge", style=discord.ButtonStyle.primary, emoji="💎", row=0)
    async def vip_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if _user_is_active(self.user_id, "vip_until"):
            remaining = _user_active_remaining(self.user_id, "vip_until")
            await interaction.response.send_message(
                f"💎 Already VIP for **{fmt_cooldown(remaining)}**.", ephemeral=True
            )
            return
        if economy.balance(self.user_id) < VIP_PRICE:
            await interaction.response.send_message(
                f"❌ Need **{VIP_PRICE:,}** coins.", ephemeral=True
            )
            return
        economy.add(self.user_id, -VIP_PRICE, "vip (shop)")
        _user_set_until(self.user_id, "vip_until", VIP_DURATION_DAYS * 24)
        await interaction.response.send_message(
            f"💎 **VIP activated** for {VIP_DURATION_DAYS} days! Badge shows on `/balance` and `/leaderboard`."
        )

    @discord.ui.button(label="Custom Title", style=discord.ButtonStyle.primary, emoji="🏷️", row=0)
    async def title_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if economy.balance(self.user_id) < CUSTOM_TITLE_PRICE:
            await interaction.response.send_message(
                f"❌ Need **{CUSTOM_TITLE_PRICE:,}** coins.", ephemeral=True
            )
            return
        await interaction.response.send_modal(ShopTitleModal(self.user_id))

    @discord.ui.button(label="XP Boost", style=discord.ButtonStyle.primary, emoji="⚡", row=0)
    async def xpboost_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if _user_is_active(self.user_id, "xp_boost_until"):
            remaining = _user_active_remaining(self.user_id, "xp_boost_until")
            await interaction.response.send_message(
                f"⚡ XP boost active for **{fmt_cooldown(remaining)}** more.", ephemeral=True
            )
            return
        if economy.balance(self.user_id) < XP_BOOST_PRICE:
            await interaction.response.send_message(
                f"❌ Need **{XP_BOOST_PRICE:,}** coins.", ephemeral=True
            )
            return
        economy.add(self.user_id, -XP_BOOST_PRICE, "xp boost (shop)")
        _user_set_until(self.user_id, "xp_boost_until", XP_BOOST_DURATION_HOURS)
        await interaction.response.send_message(
            f"⚡ **2x XP boost activated** for {XP_BOOST_DURATION_HOURS} hours!"
        )

    @discord.ui.button(label="Insurance", style=discord.ButtonStyle.success, emoji="🛡️", row=1)
    async def insurance_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if _user_is_active(self.user_id, "insurance_until"):
            remaining = _user_active_remaining(self.user_id, "insurance_until")
            await interaction.response.send_message(
                f"🛡️ Already insured for **{fmt_cooldown(remaining)}**.", ephemeral=True
            )
            return
        if economy.balance(self.user_id) < INSURANCE_PRICE:
            await interaction.response.send_message(
                f"❌ Need **{INSURANCE_PRICE:,}** coins.", ephemeral=True
            )
            return
        economy.add(self.user_id, -INSURANCE_PRICE, "insurance (shop)")
        _user_set_until(self.user_id, "insurance_until", INSURANCE_DURATION_HOURS)
        await interaction.response.send_message(
            f"🛡️ **Business insurance activated** for {INSURANCE_DURATION_HOURS} hours. "
            f"Blocks sabotage and random events."
        )

    @discord.ui.button(label="Heist Tools", style=discord.ButtonStyle.danger, emoji="🦝", row=1)
    async def heist_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if _user_is_active(self.user_id, "heist_tools_until"):
            remaining = _user_active_remaining(self.user_id, "heist_tools_until")
            await interaction.response.send_message(
                f"🦝 Heist tools active for **{fmt_cooldown(remaining)}** more.", ephemeral=True
            )
            return
        if economy.balance(self.user_id) < HEIST_TOOLS_PRICE:
            await interaction.response.send_message(
                f"❌ Need **{HEIST_TOOLS_PRICE:,}** coins.", ephemeral=True
            )
            return
        economy.add(self.user_id, -HEIST_TOOLS_PRICE, "heist tools (shop)")
        _user_set_until(self.user_id, "heist_tools_until", HEIST_TOOLS_DURATION_HOURS)
        await interaction.response.send_message(
            f"🦝 **Heist tools equipped** — +{int(HEIST_TOOLS_BOOST*100)}% rob success for {HEIST_TOOLS_DURATION_HOURS}h!"
        )

    @discord.ui.button(label="Lottery 2x", style=discord.ButtonStyle.secondary, emoji="🎰", row=1)
    async def lotmult_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        u = economy._user(self.user_id)
        if u.get("lottery_mult", 1) > 1:
            await interaction.response.send_message(
                "🎰 You already have a lottery multiplier ready.", ephemeral=True
            )
            return
        if economy.balance(self.user_id) < LOTTERY_MULT_PRICE:
            await interaction.response.send_message(
                f"❌ Need **{LOTTERY_MULT_PRICE:,}** coins.", ephemeral=True
            )
            return
        economy.add(self.user_id, -LOTTERY_MULT_PRICE, "lottery mult (shop)")
        u["lottery_mult"] = 2
        economy._save()
        await interaction.response.send_message(
            f"🎰 **2x multiplier active** on your next lottery win!"
        )

    @discord.ui.button(label="◀ Page 1", style=discord.ButtonStyle.secondary, row=2)
    async def page1_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _show_shop_main(interaction, self.user_id, edit=True)

    @discord.ui.button(label="Page 3 ▶", style=discord.ButtonStyle.secondary, row=2)
    async def page3_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _show_shop_page3(interaction, self.user_id, edit=True)

    @discord.ui.button(label="Close", style=discord.ButtonStyle.secondary, emoji="✖️", row=2)
    async def close_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        for child in self.children:
            child.disabled = True
        embed = discord.Embed(
            title="🛒 Shop closed",
            description="Come back anytime with `/shop`.",
            color=discord.Color.greyple(),
        )
        await interaction.response.edit_message(embed=embed, view=None)


class ShopPage3View(discord.ui.View):
    """Page 3 — Risk & Utility."""
    def __init__(self, user_id: int):
        super().__init__(timeout=180)
        self.user_id = user_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your shop.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Take Out Loan", style=discord.ButtonStyle.danger, emoji="💸", row=0)
    async def loan_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        loans = _load_json_file(LOANS_FILE, {"users": {}})
        if str(self.user_id) in loans["users"]:
            existing = loans["users"][str(self.user_id)]
            remaining = max(0, existing["due_at"] - time.time())
            await interaction.response.send_message(
                f"❌ You owe **{existing['owe']:,}** coins. Due in **{fmt_cooldown(int(remaining))}**.\n"
                f"Repay with `/repay`.",
                ephemeral=True,
            )
            return
        embed = discord.Embed(
            title="💸 Loan Shark",
            description=(
                "Borrow coins now, pay more later. **Miss the deadline and the shark drains 10% per hour.**\n\n"
                + "\n".join(
                    f"💰 **{amt:,}** → owe **{owe:,}** in {hrs}h"
                    for amt, (_, owe, hrs) in LOAN_AMOUNTS.items()
                )
            ),
            color=discord.Color.dark_red(),
        )
        embed.set_footer(text=f"💰 Your balance: {economy.balance(self.user_id):,}")
        await interaction.response.edit_message(embed=embed, view=ShopLoanView(self.user_id))

    @discord.ui.button(label="Place Bounty", style=discord.ButtonStyle.danger, emoji="💰", row=0)
    async def bounty_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if economy.balance(self.user_id) < BOUNTY_MIN:
            await interaction.response.send_message(
                f"❌ Min bounty is **{BOUNTY_MIN}** coins.", ephemeral=True
            )
            return
        await interaction.response.send_modal(ShopBountyModal(self.user_id))

    @discord.ui.button(label="Pet Food (7d)", style=discord.ButtonStyle.success, emoji="🍖", row=0)
    async def petfood_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        pets = _load_pets()
        if str(self.user_id) not in pets:
            await interaction.response.send_message(
                "❌ You don't have a pet. Adopt one first.", ephemeral=True
            )
            return
        if economy.balance(self.user_id) < PET_FOOD_BUNDLE_PRICE:
            await interaction.response.send_message(
                f"❌ Need **{PET_FOOD_BUNDLE_PRICE:,}** coins.", ephemeral=True
            )
            return
        economy.add(self.user_id, -PET_FOOD_BUNDLE_PRICE, "pet food bundle (shop)")
        pet = pets[str(self.user_id)]
        pet["last_fed"] = time.time() + PET_FOOD_BUNDLE_DAYS * 24 * 3600
        pet["xp"] = pet.get("xp", 0) + 50
        _save_pets(pets)
        info = PET_TYPES.get(pet["type"], {"emoji": "🐾"})
        await interaction.response.send_message(
            f"🍖 **{info['emoji']} {pet['name']}** is fed for **{PET_FOOD_BUNDLE_DAYS} days**! (+50 XP bonus)"
        )

    @discord.ui.button(label="Upgrade Business", style=discord.ButtonStyle.success, emoji="📈", row=1)
    async def upgrade_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_bizs = _user_businesses(self.user_id)
        if not user_bizs:
            await interaction.response.send_message(
                "❌ You don't own any businesses to upgrade.", ephemeral=True
            )
            return
        embed = discord.Embed(
            title="📈 Upgrade a Business",
            description=(
                f"Each upgrade adds **+{int(BUSINESS_UPGRADE_BOOST*100)}%** permanent income.\n"
                f"Max **{BUSINESS_UPGRADE_MAX_LEVEL}** levels per business.\n"
                f"Cost: **{BUSINESS_UPGRADE_PRICE_PER_LEVEL:,}** × next level.\n\n"
                "Pick which business type:"
            ),
            color=discord.Color.dark_green(),
        )
        embed.set_footer(text=f"💰 Your balance: {economy.balance(self.user_id):,}")
        await interaction.response.edit_message(
            embed=embed, view=ShopUpgradeBusinessView(self.user_id)
        )

    @discord.ui.button(label="◀ Page 2", style=discord.ButtonStyle.secondary, row=2)
    async def page2_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _show_shop_page2(interaction, self.user_id, edit=True)

    @discord.ui.button(label="Close", style=discord.ButtonStyle.secondary, emoji="✖️", row=2)
    async def close_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        for child in self.children:
            child.disabled = True
        embed = discord.Embed(
            title="🛒 Shop closed",
            description="Come back anytime with `/shop`.",
            color=discord.Color.greyple(),
        )
        await interaction.response.edit_message(embed=embed, view=None)


# ── Modals & sub-views for Page 2/3 ─────────────────────────────────────────
class ShopTitleModal(discord.ui.Modal, title=f"Buy Custom Title"):
    new_title = discord.ui.TextInput(
        label=f"Your title (max 24 chars)",
        placeholder="e.g. Degenerate King",
        required=True,
        max_length=24,
    )

    def __init__(self, user_id: int):
        super().__init__()
        self.user_id = user_id

    async def on_submit(self, interaction: discord.Interaction):
        if economy.balance(self.user_id) < CUSTOM_TITLE_PRICE:
            await interaction.response.send_message(
                f"❌ Need **{CUSTOM_TITLE_PRICE:,}** coins.", ephemeral=True
            )
            return
        text = str(self.new_title).strip().replace("@everyone", "").replace("@here", "")[:24]
        if not text:
            await interaction.response.send_message("❌ Title can't be empty.", ephemeral=True)
            return
        economy.add(self.user_id, -CUSTOM_TITLE_PRICE, "custom title (shop)")
        u = economy._user(self.user_id)
        u["custom_title"] = text
        economy._save()
        await interaction.response.send_message(
            f"🏷️ Your title is now: **{text}**"
        )


class ShopBountyModal(discord.ui.Modal, title="Place a Bounty"):
    target_input = discord.ui.TextInput(
        label="Target (@mention or username)",
        placeholder="e.g. @someuser",
        required=True,
        max_length=100,
    )
    amount_input = discord.ui.TextInput(
        label=f"Amount (min {BOUNTY_MIN} coins)",
        placeholder="e.g. 1000",
        required=True,
        max_length=10,
    )

    def __init__(self, user_id: int):
        super().__init__()
        self.user_id = user_id

    async def on_submit(self, interaction: discord.Interaction):
        # Parse target
        target_str = str(self.target_input).strip()
        target_id = None
        m = re.match(r"<@!?(\d+)>", target_str)
        if m:
            target_id = int(m.group(1))
        else:
            name = target_str.lstrip("@").lower()
            if interaction.guild:
                for member in interaction.guild.members:
                    if name in (member.display_name.lower(), member.name.lower()):
                        target_id = member.id
                        break
        if not target_id:
            await interaction.response.send_message("❌ Couldn't find that user.", ephemeral=True)
            return
        if target_id == self.user_id:
            await interaction.response.send_message("Can't bounty yourself.", ephemeral=True)
            return
        target_member = interaction.guild.get_member(target_id) if interaction.guild else None
        if not target_member or target_member.bot:
            await interaction.response.send_message("❌ Invalid target.", ephemeral=True)
            return
        # Parse amount
        try:
            amount = int(str(self.amount_input).strip())
        except ValueError:
            await interaction.response.send_message("❌ Amount must be a number.", ephemeral=True)
            return
        if amount < BOUNTY_MIN:
            await interaction.response.send_message(f"❌ Min bounty is **{BOUNTY_MIN}** coins.", ephemeral=True)
            return
        if economy.balance(self.user_id) < amount:
            await interaction.response.send_message(
                f"❌ You only have **{economy.balance(self.user_id):,}** coins.", ephemeral=True
            )
            return

        bounties = _load_json_file(BOUNTIES_FILE, {"targets": {}})
        economy.add(self.user_id, -amount, "bounty placed (shop)")
        targets = bounties["targets"].setdefault(str(target_id), [])
        targets.append({
            "placer_id": self.user_id,
            "amount": amount,
            "placed_at": time.time(),
        })
        _save_json_file(BOUNTIES_FILE, bounties)
        total_on_target = sum(b["amount"] for b in targets)
        await interaction.response.send_message(
            f"# 💰 BOUNTY PLACED\n\n"
            f"<@{self.user_id}> put a **{amount:,}** coin bounty on {target_member.mention}!\n"
            f"💀 Total on {target_member.display_name}: **{total_on_target:,}**",
            allowed_mentions=discord.AllowedMentions(users=[target_member]),
        )


class ShopLoanView(discord.ui.View):
    def __init__(self, user_id: int):
        super().__init__(timeout=180)
        self.user_id = user_id
        for amt in LOAN_AMOUNTS:
            cost, owe, hrs = LOAN_AMOUNTS[amt]
            btn = discord.ui.Button(
                label=f"{amt:,} → owe {owe:,}",
                style=discord.ButtonStyle.danger,
                emoji="💸",
            )
            btn.callback = self._make_callback(amt)
            self.add_item(btn)
        back = discord.ui.Button(label="◀ Back", style=discord.ButtonStyle.secondary, row=2)
        back.callback = self._back_callback
        self.add_item(back)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your shop.", ephemeral=True)
            return False
        return True

    def _make_callback(self, amount):
        async def cb(interaction: discord.Interaction):
            loans = _load_json_file(LOANS_FILE, {"users": {}})
            if str(self.user_id) in loans["users"]:
                await interaction.response.send_message(
                    "❌ You already have a loan. Repay it first.", ephemeral=True
                )
                return
            cost, owe, hours = LOAN_AMOUNTS[amount]
            economy.add(self.user_id, amount, "loan borrow (shop)")
            loans["users"][str(self.user_id)] = {
                "borrowed": amount,
                "owe": owe,
                "due_at": time.time() + hours * 3600,
                "borrowed_at": time.time(),
            }
            _save_json_file(LOANS_FILE, loans)
            await interaction.response.send_message(
                f"# 💸 LOAN APPROVED\n\n"
                f"Borrowed **{amount:,}** coins. Owe **{owe:,}** in **{hours}h**.\n"
                f"Use `/repay` to pay back."
            )
        return cb

    async def _back_callback(self, interaction: discord.Interaction):
        await _show_shop_page3(interaction, self.user_id, edit=True)


class ShopUpgradeBusinessView(discord.ui.View):
    def __init__(self, user_id: int):
        super().__init__(timeout=180)
        self.user_id = user_id
        # Only show buttons for businesses the user actually owns
        owned_types = set(b["type"] for b in _user_businesses(user_id))
        for biz_type in BUSINESS_TYPES:
            if biz_type not in owned_types:
                continue
            info = BUSINESS_TYPES[biz_type]
            # Find lowest-level instance to upgrade
            matching = [b for b in _user_businesses(user_id) if b["type"] == biz_type]
            matching.sort(key=lambda b: b.get("upgrade_level", 0))
            biz = matching[0]
            current_level = biz.get("upgrade_level", 0)
            if current_level >= BUSINESS_UPGRADE_MAX_LEVEL:
                continue
            cost = BUSINESS_UPGRADE_PRICE_PER_LEVEL * (current_level + 1)
            btn = discord.ui.Button(
                label=f"{info['name']} L{current_level+1} ({cost:,})",
                emoji=info["emoji"],
                style=discord.ButtonStyle.success,
            )
            btn.callback = self._make_callback(biz_type)
            self.add_item(btn)
        back = discord.ui.Button(label="◀ Back", style=discord.ButtonStyle.secondary, row=4)
        back.callback = self._back_callback
        self.add_item(back)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your shop.", ephemeral=True)
            return False
        return True

    def _make_callback(self, biz_type: str):
        async def cb(interaction: discord.Interaction):
            data = _load_businesses()
            user_bizs = data["users"].get(str(self.user_id), [])
            matching = [b for b in user_bizs if b["type"] == biz_type]
            matching.sort(key=lambda b: b.get("upgrade_level", 0))
            if not matching:
                await interaction.response.send_message("❌ No business to upgrade.", ephemeral=True)
                return
            biz = matching[0]
            current_level = biz.get("upgrade_level", 0)
            if current_level >= BUSINESS_UPGRADE_MAX_LEVEL:
                await interaction.response.send_message("❌ Already maxed.", ephemeral=True)
                return
            cost = BUSINESS_UPGRADE_PRICE_PER_LEVEL * (current_level + 1)
            if economy.balance(self.user_id) < cost:
                await interaction.response.send_message(
                    f"❌ Need **{cost:,}** coins.", ephemeral=True
                )
                return
            economy.add(self.user_id, -cost, "business upgrade (shop)")
            biz["upgrade_level"] = current_level + 1
            _save_businesses(data)
            info = BUSINESS_TYPES[biz_type]
            boost = (current_level + 1) * BUSINESS_UPGRADE_BOOST * 100
            await interaction.response.send_message(
                f"📈 **{info['emoji']} {info['name']}** upgraded to **Level {current_level+1}**!\n"
                f"💰 Total boost: **+{int(boost)}%** income."
            )
        return cb

    async def _back_callback(self, interaction: discord.Interaction):
        await _show_shop_page3(interaction, self.user_id, edit=True)


def shop_back_button():
    """Reusable Back button for sub-views."""
    btn = discord.ui.Button(label="Back to Shop", style=discord.ButtonStyle.secondary, emoji="◀️", row=4)
    return btn


class ShopColorView(discord.ui.View):
    """Pick a color for buyrole."""
    def __init__(self, user_id: int):
        super().__init__(timeout=180)
        self.user_id = user_id

        # Add color buttons dynamically
        colors_list = list(ROLE_COLORS.items())
        for i, (key, (color_int, label)) in enumerate(colors_list):
            row = i // 5  # 5 per row
            self.add_item(self._make_color_button(key, label, row))

        # Back button
        back = discord.ui.Button(label="Back", style=discord.ButtonStyle.secondary, emoji="◀️", row=3)
        async def go_back(interaction: discord.Interaction):
            if interaction.user.id != self.user_id:
                await interaction.response.send_message("Not your session.", ephemeral=True)
                return
            await _show_shop_main(interaction, self.user_id, edit=True)
        back.callback = go_back
        self.add_item(back)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your session.", ephemeral=True)
            return False
        return True

    def _make_color_button(self, key: str, label: str, row: int) -> discord.ui.Button:
        btn = discord.ui.Button(label=label, style=discord.ButtonStyle.primary, row=row)
        async def cb(interaction: discord.Interaction):
            if interaction.user.id != self.user_id:
                await interaction.response.send_message("Not your session.", ephemeral=True)
                return
            # Show duration choice for this color
            color_int, color_label = ROLE_COLORS[key]
            embed = discord.Embed(
                title=f"{color_label} Role",
                description=(
                    "How long do you want this role?\n\n"
                    f"**24 hours** — {ROLE_PRICES['24h']:,} coins\n"
                    f"**Permanent** — {ROLE_PRICES['perm']:,} coins"
                ),
                color=discord.Color(color_int) if color_int != 0xFFFFFF else discord.Color.light_grey(),
            )
            bal = economy.balance(self.user_id)
            embed.set_footer(text=f"💰 Your balance: {bal:,}")
            await interaction.response.edit_message(embed=embed, view=ShopRoleDurationView(self.user_id, key))
        btn.callback = cb
        return btn


class ShopRoleDurationView(discord.ui.View):
    def __init__(self, user_id: int, color_key: str):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.color_key = color_key

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your session.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="24 Hours", style=discord.ButtonStyle.primary, emoji="⏱️")
    async def buy_24h(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._purchase(interaction, "24h")

    @discord.ui.button(label="Permanent", style=discord.ButtonStyle.success, emoji="💎")
    async def buy_perm(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._purchase(interaction, "perm")

    @discord.ui.button(label="Custom Name", style=discord.ButtonStyle.secondary, emoji="✏️", row=1)
    async def custom_name(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(BuyRoleModal(self.user_id, self.color_key))

    @discord.ui.button(label="Back", style=discord.ButtonStyle.secondary, emoji="◀️", row=1)
    async def back(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _show_shop_main(interaction, self.user_id, edit=True)

    async def _purchase(self, interaction: discord.Interaction, duration: str, custom_name: str = None):
        # This re-implements the buyrole logic but inside an interaction
        user = interaction.user
        guild = interaction.guild
        if not guild:
            await interaction.response.send_message("Server only.", ephemeral=True)
            return
        price = ROLE_PRICES[duration]
        bal = economy.balance(user.id)
        if bal < price:
            await interaction.response.send_message(
                f"❌ Need **{price:,}**. You have **{bal:,}**.", ephemeral=True
            )
            return

        bot_member = guild.me
        if not bot_member.guild_permissions.manage_roles:
            await interaction.response.send_message(
                "❌ I need **Manage Roles** permission. Ask an admin.", ephemeral=True
            )
            return

        color_int, color_label = ROLE_COLORS[self.color_key]
        role_name = (custom_name[:30] if custom_name else f"💎 {user.display_name}").strip() or f"💎 {user.display_name}"

        await interaction.response.defer()

        try:
            role = await guild.create_role(
                name=role_name,
                colour=discord.Colour(color_int),
                reason=f"Shop purchase by {user.display_name}",
                mentionable=False,
            )
            try:
                top_bot_role = bot_member.top_role
                await role.edit(position=max(top_bot_role.position - 1, 1))
            except Exception:
                pass
            await user.add_roles(role, reason="shop buyrole")
        except discord.Forbidden:
            await interaction.followup.send(
                "❌ I can't manage roles. Move my role higher in Server Settings.", ephemeral=True
            )
            return
        except Exception as e:
            log.exception("Shop buyrole failed")
            await interaction.followup.send(f"❌ Failed: {e}", ephemeral=True)
            return

        economy.add(user.id, -price, "shop buyrole")
        new_bal = economy.balance(user.id)

        if duration == "24h":
            temp = _load_temp_roles()
            temp.append({
                "role_id": role.id,
                "guild_id": guild.id,
                "user_id": user.id,
                "expires_at": time.time() + 24 * 3600,
            })
            _save_temp_roles(temp)
            duration_label = "24 hours"
        else:
            duration_label = "permanently"

        embed = discord.Embed(
            title="✅ Role Purchased",
            description=(
                f"You bought {color_label} role **{role_name}** for **{price:,}** coins.\n"
                f"Lasts **{duration_label}**."
            ),
            color=discord.Color(color_int) if color_int != 0xFFFFFF else discord.Color.light_grey(),
        )
        embed.set_footer(text=f"💰 New balance: {new_bal:,}")
        for child in self.children:
            child.disabled = True
        await interaction.edit_original_response(embed=embed, view=self)


class BuyRoleModal(discord.ui.Modal, title="Custom Role Name"):
    role_name = discord.ui.TextInput(
        label="Role name",
        placeholder="E.g. 💎 Boss",
        max_length=30,
        required=True,
    )

    def __init__(self, user_id: int, color_key: str):
        super().__init__()
        self.user_id = user_id
        self.color_key = color_key

    async def on_submit(self, interaction: discord.Interaction):
        # Show duration choice with the custom name now stored
        view = ShopRoleDurationViewCustom(self.user_id, self.color_key, self.role_name.value)
        color_int, color_label = ROLE_COLORS[self.color_key]
        embed = discord.Embed(
            title=f"{color_label} — Custom: {self.role_name.value}",
            description=(
                "How long?\n\n"
                f"**24 hours** — {ROLE_PRICES['24h']:,} coins\n"
                f"**Permanent** — {ROLE_PRICES['perm']:,} coins"
            ),
            color=discord.Color(color_int) if color_int != 0xFFFFFF else discord.Color.light_grey(),
        )
        bal = economy.balance(self.user_id)
        embed.set_footer(text=f"💰 Your balance: {bal:,}")
        await interaction.response.edit_message(embed=embed, view=view)


class ShopRoleDurationViewCustom(discord.ui.View):
    """Same as ShopRoleDurationView but with a stored custom name."""
    def __init__(self, user_id: int, color_key: str, custom_name: str):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.color_key = color_key
        self.custom_name = custom_name

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your session.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="24 Hours", style=discord.ButtonStyle.primary, emoji="⏱️")
    async def buy_24h(self, interaction: discord.Interaction, button: discord.ui.Button):
        await ShopRoleDurationView._purchase(self, interaction, "24h", self.custom_name)

    @discord.ui.button(label="Permanent", style=discord.ButtonStyle.success, emoji="💎")
    async def buy_perm(self, interaction: discord.Interaction, button: discord.ui.Button):
        await ShopRoleDurationView._purchase(self, interaction, "perm", self.custom_name)

    @discord.ui.button(label="Back", style=discord.ButtonStyle.secondary, emoji="◀️", row=1)
    async def back(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _show_shop_main(interaction, self.user_id, edit=True)


class ShopConfirmProtectView(discord.ui.View):
    def __init__(self, user_id: int):
        super().__init__(timeout=180)
        self.user_id = user_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your session.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label=f"Buy Protection", style=discord.ButtonStyle.success, emoji="🛡️")
    async def buy(self, interaction: discord.Interaction, button: discord.ui.Button):
        bal = economy.balance(self.user_id)
        if bal < PROTECT_PRICE:
            await interaction.response.send_message(f"❌ Not enough coins.", ephemeral=True)
            return
        if is_protected(self.user_id):
            await interaction.response.send_message("🛡️ Already protected.", ephemeral=True)
            return
        economy.add(self.user_id, -PROTECT_PRICE, "shop protect")
        data = _load_protections()
        data[str(self.user_id)] = time.time() + PROTECT_DURATION
        _save_protections(data)
        new_bal = economy.balance(self.user_id)
        embed = discord.Embed(
            title="🛡️ Protection Activated",
            description=f"You're immune to `/rob` for the next **12 hours**.",
            color=discord.Color.green(),
        )
        embed.set_footer(text=f"💰 New balance: {new_bal:,}")
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(label="Back", style=discord.ButtonStyle.secondary, emoji="◀️")
    async def back(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _show_shop_main(interaction, self.user_id, edit=True)


class MegaphoneModal(discord.ui.Modal, title="Megaphone Message"):
    message = discord.ui.TextInput(
        label=f"Your message (costs {MEGAPHONE_PRICE} coins)",
        placeholder="Pings @here when posted",
        style=discord.TextStyle.paragraph,
        max_length=200,
        required=True,
    )

    def __init__(self, user_id: int):
        super().__init__()
        self.user_id = user_id

    async def on_submit(self, interaction: discord.Interaction):
        user = interaction.user
        bal = economy.balance(user.id)
        if bal < MEGAPHONE_PRICE:
            await interaction.response.send_message(
                f"❌ Costs **{MEGAPHONE_PRICE:,}**. You have **{bal:,}**.", ephemeral=True
            )
            return
        safe_msg = self.message.value.replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
        economy.add(user.id, -MEGAPHONE_PRICE, "shop megaphone")
        new_bal = economy.balance(user.id)
        # Send the megaphone message
        await interaction.response.send_message(
            content=(
                f"📢 **MEGAPHONE** 📢\n\n"
                f"@here — {user.mention} paid {MEGAPHONE_PRICE:,} coins to say:\n\n"
                f"# > {safe_msg}\n\n"
                f"-# Balance: {new_bal:,}"
            ),
            allowed_mentions=discord.AllowedMentions(everyone=True, users=[user], roles=False),
        )


class ShopLotteryView(discord.ui.View):
    def __init__(self, user_id: int):
        super().__init__(timeout=180)
        self.user_id = user_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your session.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="1 ticket", style=discord.ButtonStyle.primary)
    async def buy1(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._buy(interaction, 1)

    @discord.ui.button(label="5 tickets", style=discord.ButtonStyle.primary)
    async def buy5(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._buy(interaction, 5)

    @discord.ui.button(label="10 tickets", style=discord.ButtonStyle.primary)
    async def buy10(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._buy(interaction, 10)

    @discord.ui.button(label="Custom amount", style=discord.ButtonStyle.secondary, emoji="✏️", row=1)
    async def custom(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(LotteryModal(self.user_id))

    @discord.ui.button(label="Back", style=discord.ButtonStyle.secondary, emoji="◀️", row=1)
    async def back(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _show_shop_main(interaction, self.user_id, edit=True)

    async def _buy(self, interaction: discord.Interaction, count: int):
        await _purchase_lottery_tickets(interaction, self.user_id, count)


class LotteryModal(discord.ui.Modal, title="Buy Lottery Tickets"):
    amount = discord.ui.TextInput(
        label=f"How many tickets? ({LOTTERY_TICKET_PRICE} coins each, max 50)",
        placeholder="e.g. 25",
        required=True,
    )

    def __init__(self, user_id: int):
        super().__init__()
        self.user_id = user_id

    async def on_submit(self, interaction: discord.Interaction):
        try:
            n = int(self.amount.value.strip())
        except ValueError:
            await interaction.response.send_message("That's not a number.", ephemeral=True)
            return
        if n <= 0:
            await interaction.response.send_message("Pick a positive number.", ephemeral=True)
            return
        await _purchase_lottery_tickets(interaction, self.user_id, n)


async def _purchase_lottery_tickets(interaction: discord.Interaction, user_id: int, count: int):
    data = _load_lottery()
    current = data["tickets"].get(str(user_id), 0)
    if current + count > LOTTERY_MAX_TICKETS_PER_USER:
        await interaction.response.send_message(
            f"❌ Max {LOTTERY_MAX_TICKETS_PER_USER} per user per drawing. You already have {current}.",
            ephemeral=True,
        )
        return
    cost = count * LOTTERY_TICKET_PRICE
    bal = economy.balance(user_id)
    if bal < cost:
        await interaction.response.send_message(
            f"❌ Costs **{cost:,}**. You have **{bal:,}**.", ephemeral=True
        )
        return
    economy.add(user_id, -cost, "shop lottery")
    data["tickets"][str(user_id)] = current + count
    data["jackpot"] += cost // 2
    _save_lottery(data)
    new_bal = economy.balance(user_id)
    total_tickets = sum(data["tickets"].values())
    chance = ((current + count) / total_tickets * 100) if total_tickets > 0 else 0

    embed = discord.Embed(
        title="🎟️ Tickets Purchased",
        description=(
            f"You bought **{count}** tickets for **{cost:,}** coins.\n\n"
            f"💰 Jackpot: **{data['jackpot']:,}**\n"
            f"🎟️ Your tickets: **{current + count}**\n"
            f"📊 Win chance: **{chance:.1f}%**"
        ),
        color=discord.Color.gold(),
    )
    embed.set_footer(text=f"💰 New balance: {new_bal:,}")
    # Use response if not yet responded, otherwise followup
    if not interaction.response.is_done():
        await interaction.response.send_message(embed=embed, ephemeral=True)
    else:
        await interaction.followup.send(embed=embed, ephemeral=True)


async def _show_shop_main(interaction: discord.Interaction, user_id: int, edit: bool = False):
    bal = economy.balance(user_id)
    user = interaction.guild.get_member(user_id) if interaction.guild else None
    name = user.display_name if user else "You"
    embed = discord.Embed(
        title="🛒 SHOP — Page 1/3 (Essentials)",
        description=(
            f"Welcome, **{name}**. Pick a category below.\n\n"
            f"🎨 **Colored Role** — {ROLE_PRICES['24h']:,}–{ROLE_PRICES['perm']:,} coins\n"
            f"🛡️ **Rob Immunity (12h)** — {PROTECT_PRICE:,} coins\n"
            f"📢 **Megaphone (@here)** — {MEGAPHONE_PRICE:,} coins\n"
            f"🎰 **Lottery Tickets** — {LOTTERY_TICKET_PRICE:,} coins each\n"
            f"🐶 **Adopt a Pet** — {PET_ADOPT_COST:,} coins\n"
            f"🏢 **Buy a Business** — 2,000+ coins (passive income!)\n\n"
            f"_Click ▶ for more boosts, cosmetics, loans, bounties..._"
        ),
        color=discord.Color.gold(),
    )
    embed.set_footer(text=f"💰 Your balance: {bal:,}")
    view = ShopMainView(user_id)
    if edit:
        await interaction.response.edit_message(embed=embed, view=view)
    else:
        await interaction.response.send_message(embed=embed, view=view)


async def _show_shop_page2(interaction: discord.Interaction, user_id: int, edit: bool = False):
    bal = economy.balance(user_id)
    user = interaction.guild.get_member(user_id) if interaction.guild else None
    name = user.display_name if user else "You"
    embed = discord.Embed(
        title="🛒 SHOP — Page 2/3 (Boosts & Cosmetics)",
        description=(
            f"💎 **VIP Badge** — {VIP_PRICE:,} coins ({VIP_DURATION_DAYS}d badge next to your name)\n"
            f"🏷️ **Custom Title** — {CUSTOM_TITLE_PRICE:,} coins (shown on /balance & /leaderboard)\n"
            f"⚡ **XP Boost** — {XP_BOOST_PRICE:,} coins (2x XP for {XP_BOOST_DURATION_HOURS}h)\n"
            f"🛡️ **Business Insurance** — {INSURANCE_PRICE:,} coins ({INSURANCE_DURATION_HOURS}h sabotage/event immunity)\n"
            f"🦝 **Heist Tools** — {HEIST_TOOLS_PRICE:,} coins (+{int(HEIST_TOOLS_BOOST*100)}% rob success for {HEIST_TOOLS_DURATION_HOURS}h)\n"
            f"🎰 **Lottery 2x** — {LOTTERY_MULT_PRICE:,} coins (next lottery win pays double)"
        ),
        color=discord.Color.purple(),
    )
    embed.set_footer(text=f"💰 Your balance: {bal:,}")
    view = ShopPage2View(user_id)
    if edit:
        await interaction.response.edit_message(embed=embed, view=view)
    else:
        await interaction.response.send_message(embed=embed, view=view)


async def _show_shop_page3(interaction: discord.Interaction, user_id: int, edit: bool = False):
    bal = economy.balance(user_id)
    user = interaction.guild.get_member(user_id) if interaction.guild else None
    name = user.display_name if user else "You"
    embed = discord.Embed(
        title="🛒 SHOP — Page 3/3 (Risk & Utility)",
        description=(
            f"💸 **Loan Shark** — borrow 1k/5k/25k now, owe more later\n"
            f"💰 **Place Bounty** — put a hit on a user (min {BOUNTY_MIN}). Anyone who beats them in PvP claims it.\n"
            f"🍖 **Pet Food Bundle** — {PET_FOOD_BUNDLE_PRICE:,} coins ({PET_FOOD_BUNDLE_DAYS}d auto-feed, discounted)\n"
            f"📈 **Upgrade Business** — {BUSINESS_UPGRADE_PRICE_PER_LEVEL:,}+ coins (+{int(BUSINESS_UPGRADE_BOOST*100)}% permanent income per level, max {BUSINESS_UPGRADE_MAX_LEVEL})"
        ),
        color=discord.Color.dark_red(),
    )
    embed.set_footer(text=f"💰 Your balance: {bal:,}")
    view = ShopPage3View(user_id)
    if edit:
        await interaction.response.edit_message(embed=embed, view=view)
    else:
        await interaction.response.send_message(embed=embed, view=view)


# ── 🐶 Shop: Pet Adoption ────────────────────────────────────────────────────
class ShopPetView(discord.ui.View):
    """Pick a pet type. Each button opens a name modal."""
    def __init__(self, user_id: int, has_pet: bool = False):
        super().__init__(timeout=180)
        self.user_id = user_id
        # Build buttons for each pet type
        for i, (key, info) in enumerate(PET_TYPES.items()):
            btn = discord.ui.Button(
                label=info["name"],
                emoji=info["emoji"],
                style=discord.ButtonStyle.primary,
                row=i // 4,
                disabled=has_pet,
            )
            btn.callback = self._make_callback(key)
            self.add_item(btn)
        # Back button
        back = discord.ui.Button(label="Back to Shop", style=discord.ButtonStyle.secondary, emoji="◀️", row=2)
        back.callback = self._back_callback
        self.add_item(back)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your shop.", ephemeral=True)
            return False
        return True

    def _make_callback(self, pet_type: str):
        async def cb(interaction: discord.Interaction):
            # Check balance & pet ownership first
            if economy.balance(self.user_id) < PET_ADOPT_COST:
                await interaction.response.send_message(
                    f"❌ Need **{PET_ADOPT_COST:,}** coins.", ephemeral=True
                )
                return
            pets = _load_pets()
            if str(self.user_id) in pets:
                await interaction.response.send_message(
                    "❌ You already have a pet. Abandon it first.", ephemeral=True
                )
                return
            await interaction.response.send_modal(PetNameModal(self.user_id, pet_type))
        return cb

    async def _back_callback(self, interaction: discord.Interaction):
        await _show_shop_main(interaction, self.user_id, edit=True)


class PetNameModal(discord.ui.Modal, title="Name your pet"):
    pet_name = discord.ui.TextInput(
        label="Pet name",
        placeholder="e.g. Buddy",
        required=True,
        max_length=30,
    )

    def __init__(self, user_id: int, pet_type: str):
        super().__init__()
        self.user_id = user_id
        self.pet_type = pet_type

    async def on_submit(self, interaction: discord.Interaction):
        # Re-check (race protection)
        if economy.balance(self.user_id) < PET_ADOPT_COST:
            await interaction.response.send_message(
                f"❌ Need **{PET_ADOPT_COST:,}** coins.", ephemeral=True
            )
            return
        pets = _load_pets()
        if str(self.user_id) in pets:
            await interaction.response.send_message("❌ You already have a pet.", ephemeral=True)
            return

        name = str(self.pet_name).strip()[:30]
        if not name:
            await interaction.response.send_message("❌ Give your pet a name.", ephemeral=True)
            return

        economy.add(self.user_id, -PET_ADOPT_COST, "pet adoption (shop)")
        info = PET_TYPES[self.pet_type]
        pets[str(self.user_id)] = {
            "type": self.pet_type,
            "name": name,
            "xp": 0,
            "last_fed": time.time(),
            "last_collected": time.time(),
            "adopted_at": time.time(),
        }
        _save_pets(pets)
        await interaction.response.send_message(
            f"# {info['emoji']} You adopted **{name}** the {info['name']}!\n\n"
            f"💰 Earns **{PET_DAILY_INCOME_BASE}** coins per level per day\n"
            f"🍖 Feed daily with `/feed` so they don't starve\n"
            f"📊 Check stats with `/pet`\n"
            f"💎 Collect earnings with `/collect`"
        )


# ── 🏢 Shop: Business Purchase ───────────────────────────────────────────────
class ShopBusinessView(discord.ui.View):
    """Paginated list of business types with buy buttons."""
    PER_PAGE = 4  # 4 buttons + back/page nav

    def __init__(self, user_id: int, page: int = 0):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.page = page
        biz_keys = list(BUSINESS_TYPES.keys())
        start = page * self.PER_PAGE
        page_keys = biz_keys[start:start + self.PER_PAGE]

        for i, key in enumerate(page_keys):
            info = BUSINESS_TYPES[key]
            cost = _business_cost(user_id, key)
            btn = discord.ui.Button(
                label=f"{info['name']} ({cost:,})",
                emoji=info["emoji"],
                style=discord.ButtonStyle.success,
                row=i,
            )
            btn.callback = self._make_callback(key)
            self.add_item(btn)

        # Nav row (row 4 is buttons row max so use proper rows)
        nav_row = (self.PER_PAGE - 1) // 5 + 1
        if page > 0:
            prev_btn = discord.ui.Button(label="◀ Prev", style=discord.ButtonStyle.secondary, row=4)
            prev_btn.callback = self._prev_callback
            self.add_item(prev_btn)
        total_pages = (len(biz_keys) + self.PER_PAGE - 1) // self.PER_PAGE
        if page + 1 < total_pages:
            next_btn = discord.ui.Button(label="Next ▶", style=discord.ButtonStyle.secondary, row=4)
            next_btn.callback = self._next_callback
            self.add_item(next_btn)
        back = discord.ui.Button(label="Back to Shop", style=discord.ButtonStyle.secondary, emoji="◀️", row=4)
        back.callback = self._back_callback
        self.add_item(back)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your shop.", ephemeral=True)
            return False
        return True

    def _make_callback(self, biz_type: str):
        async def cb(interaction: discord.Interaction):
            cost = _business_cost(self.user_id, biz_type)
            bal = economy.balance(self.user_id)
            if bal < cost:
                await interaction.response.send_message(
                    f"❌ Need **{cost:,}** coins. You have **{bal:,}**.", ephemeral=True
                )
                return

            info = BUSINESS_TYPES[biz_type]
            economy.add(self.user_id, -cost, f"bought {biz_type} (shop)")
            data = _load_businesses()
            user_bizs = data["users"].setdefault(str(self.user_id), [])
            user_bizs.append({
                "id": f"{biz_type}_{int(time.time())}_{random.randint(1000,9999)}",
                "type": biz_type,
                "purchased_at": time.time(),
                "last_collected": time.time(),
                "employees": [],
                "damaged_until": 0,
                "lifetime_earned": 0,
            })
            _save_businesses(data)
            new_bal = economy.balance(self.user_id)
            await interaction.response.send_message(
                f"# {info['emoji']} BUSINESS ACQUIRED\n\n"
                f"Bought **{info['name']}** for **{cost:,}** coins.\n"
                f"💰 Income: **{info['income_per_hour']:,}**/hr\n"
                f"👥 Max employees: **{info['max_employees']}**\n\n"
                f"Use `/businesses` to manage, `/collectbusiness` to claim earnings.\n"
                f"Balance: **{new_bal:,}**"
            )
        return cb

    async def _prev_callback(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="🏢 Buy a Business",
            description=(
                "Businesses earn passive income 24/7.\n"
                "Hire employees with `/hire` to boost income.\n"
                f"_Same-tier purchases cost {int((BUSINESS_QUANTITY_MULTIPLIER-1)*100)}% more each._"
            ),
            color=discord.Color.dark_green(),
        )
        bal = economy.balance(self.user_id)
        embed.set_footer(text=f"💰 Your balance: {bal:,}")
        await interaction.response.edit_message(
            embed=embed, view=ShopBusinessView(self.user_id, page=self.page - 1)
        )

    async def _next_callback(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="🏢 Buy a Business",
            description=(
                "Businesses earn passive income 24/7.\n"
                "Hire employees with `/hire` to boost income.\n"
                f"_Same-tier purchases cost {int((BUSINESS_QUANTITY_MULTIPLIER-1)*100)}% more each._"
            ),
            color=discord.Color.dark_green(),
        )
        bal = economy.balance(self.user_id)
        embed.set_footer(text=f"💰 Your balance: {bal:,}")
        await interaction.response.edit_message(
            embed=embed, view=ShopBusinessView(self.user_id, page=self.page + 1)
        )

    async def _back_callback(self, interaction: discord.Interaction):
        await _show_shop_main(interaction, self.user_id, edit=True)


@tree.command(name="shop", description="Browse the shop with interactive buttons.")
async def shop_command(interaction: discord.Interaction):
    await _show_shop_main(interaction, interaction.user.id, edit=False)


# ─────────────────────────────────────────────────────────────────────────────
# 🎮 ADVANCED GAMES
# ─────────────────────────────────────────────────────────────────────────────

# ── 🎲 /quest — AI choose-your-own-adventure ─────────────────────────────────
QUEST_MAX_TURNS = 4  # number of choices before final outcome


class QuestView(discord.ui.View):
    def __init__(self, user_id: int, history: list, turn: int):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.history = history  # list of {"text": ..., "choice": ...}
        self.turn = turn

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your quest.", ephemeral=True)
            return False
        return True

    def add_choice_buttons(self, choices: list):
        # Add up to 3 choice buttons
        for i, choice_text in enumerate(choices[:3]):
            label = choice_text[:75]
            btn = discord.ui.Button(
                label=label,
                style=[discord.ButtonStyle.primary, discord.ButtonStyle.success, discord.ButtonStyle.secondary][i % 3],
                custom_id=f"quest_choice_{i}",
                row=i,
            )
            btn.callback = self._make_callback(choice_text)
            self.add_item(btn)

    def _make_callback(self, choice_text: str):
        async def cb(interaction: discord.Interaction):
            await self._handle_choice(interaction, choice_text)
        return cb

    async def _handle_choice(self, interaction: discord.Interaction, choice: str):
        # Disable buttons and continue the quest
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(view=self)

        # Build context for AI
        cfg = load_config()
        self.history.append({"choice": choice})
        self.turn += 1

        is_final = self.turn >= QUEST_MAX_TURNS

        history_text = "\n\n".join(
            f"Scene {i+1}: {h.get('text','')}\nPlayer chose: {h.get('choice','')}"
            for i, h in enumerate(self.history) if h.get('text')
        )
        # Latest choice is the most recent
        last_choice = self.history[-1]["choice"]

        if is_final:
            sys = (
                cfg["system_prompt"] +
                "\n\n=== QUEST GENERATOR — FINAL SCENE ===\n"
                "You are running a choose-your-own-adventure for a Discord user. "
                "This is the FINAL scene. Based on their journey, write a 2-3 sentence ending. "
                "End with EXACTLY one line in this format: VERDICT: WIN or VERDICT: LOSS. "
                "Win = they survived/triumphed/got the prize. Loss = they died/failed/got robbed. "
                "Stay in character. Be vivid and specific."
            )
            prompt = f"Quest so far:\n\n{history_text}\n\nWrite the final scene and verdict."
        else:
            sys = (
                cfg["system_prompt"] +
                "\n\n=== QUEST GENERATOR ===\n"
                f"You are running a choose-your-own-adventure. This is scene {self.turn + 1} of {QUEST_MAX_TURNS + 1}. "
                "Write 2 short sentences describing what happens after the player's last choice, "
                "then exactly 3 numbered options for what they do next. "
                "Format strictly as:\n"
                "[scene description]\n\n"
                "1) [option 1]\n"
                "2) [option 2]\n"
                "3) [option 3]\n\n"
                "Stay in character. Make options creative and varied."
            )
            prompt = f"Quest so far:\n\n{history_text}\n\nLatest choice: {last_choice}\n\nContinue the story."

        try:
            response = await ask_ai(
                sys, [{"role": "user", "content": prompt}],
                {**cfg, "max_tokens": 400, "temperature": 0.95},
            )
        except Exception as e:
            log.exception("quest AI failed")
            response = ""

        if not response or response.startswith("⚠️"):
            await interaction.followup.send("⚠️ The quest got lost in the void. Try again later.")
            return

        if is_final:
            # Parse verdict
            verdict_match = re.search(r"VERDICT:\s*(WIN|LOSS)", response, re.IGNORECASE)
            verdict = verdict_match.group(1).upper() if verdict_match else random.choice(["WIN", "LOSS"])
            # Remove the VERDICT line from displayed text
            ending_text = re.sub(r"VERDICT:\s*(WIN|LOSS)", "", response, flags=re.IGNORECASE).strip()

            user = interaction.user
            if verdict == "WIN":
                reward = random.randint(500, 2000)
                economy.add(user.id, reward, "quest win")
                economy.record_win(user.id)
                outcome_line = f"## 🏆 VICTORY!\n💰 You earned **{reward:,}** coins.\nBalance: **{economy.balance(user.id):,}**"
            else:
                fine = min(random.randint(200, 600), economy.balance(user.id))
                economy.add(user.id, -fine, "quest loss")
                economy.record_loss(user.id)
                outcome_line = f"## 💀 DEFEAT!\n💸 You lost **{fine:,}** coins.\nBalance: **{economy.balance(user.id):,}**"

            final = f"# 🎲 QUEST COMPLETE\n\n{ending_text}\n\n{outcome_line}"
            await interaction.followup.send(final)
            return

        # Parse the 3 options
        options_match = re.findall(r"^\s*\d+[\)\.]\s*(.+?)\s*$", response, re.MULTILINE)
        if len(options_match) < 2:
            # Couldn't parse; force 3 generic options
            options_match = ["Continue carefully", "Take a risk", "Run away"]
        options = options_match[:3]

        # Strip options from displayed scene
        scene_text = re.sub(r"^\s*\d+[\)\.].*$", "", response, flags=re.MULTILINE).strip()

        self.history[-1]["text"] = scene_text  # store this scene
        new_view = QuestView(self.user_id, self.history, self.turn)
        new_view.add_choice_buttons(options)

        embed = discord.Embed(
            title=f"🎲 Quest — Scene {self.turn + 1}/{QUEST_MAX_TURNS + 1}",
            description=scene_text,
            color=discord.Color.purple(),
        )
        embed.set_footer(text=f"{interaction.user.display_name}'s adventure")
        await interaction.followup.send(embed=embed, view=new_view)


@tree.command(name="quest", description="Start an AI-generated choose-your-own-adventure!")
async def quest_command(interaction: discord.Interaction):
    cfg = load_config()
    user = interaction.user

    await interaction.response.defer()

    # Generate the opening scene
    sys = (
        cfg["system_prompt"] +
        "\n\n=== QUEST GENERATOR — OPENING ===\n"
        f"You are starting a choose-your-own-adventure for {user.display_name}. "
        "Write the opening scene in 2 short sentences (set the stakes!), "
        "then exactly 3 numbered options for what they do next. "
        "Pick a wild scenario: heist, dungeon crawl, gang turf war, ancient ruins, alien abduction, etc. "
        "Format strictly as:\n"
        "[scene description]\n\n"
        "1) [option 1]\n"
        "2) [option 2]\n"
        "3) [option 3]\n\n"
        "Stay in character."
    )
    try:
        response = await ask_ai(
            sys, [{"role": "user", "content": f"Begin the quest for {user.display_name}."}],
            {**cfg, "max_tokens": 400, "temperature": 0.95},
        )
    except Exception:
        response = ""

    if not response or response.startswith("⚠️"):
        await interaction.followup.send("⚠️ Couldn't start the quest. Try again later.")
        return

    options_match = re.findall(r"^\s*\d+[\)\.]\s*(.+?)\s*$", response, re.MULTILINE)
    if len(options_match) < 2:
        options_match = ["Press forward", "Hide and wait", "Run for it"]
    options = options_match[:3]
    scene_text = re.sub(r"^\s*\d+[\)\.].*$", "", response, flags=re.MULTILINE).strip()

    history = [{"text": scene_text, "choice": None}]
    view = QuestView(user.id, history, turn=0)
    view.add_choice_buttons(options)

    embed = discord.Embed(
        title=f"🎲 Quest — Scene 1/{QUEST_MAX_TURNS + 1}",
        description=scene_text,
        color=discord.Color.purple(),
    )
    embed.set_footer(text=f"{user.display_name}'s adventure")
    await interaction.followup.send(embed=embed, view=view)


# ── 🚪 /shootout — door elimination ──────────────────────────────────────────
SHOOTOUT_LOBBY: dict[str, dict] = {}  # channel_id -> {host_id, players, message_id, started, buy_in}


class ShootoutLobbyView(discord.ui.View):
    def __init__(self, channel_id: str, buy_in: int):
        super().__init__(timeout=60)
        self.channel_id = channel_id
        self.buy_in = buy_in

    @discord.ui.button(label="Join Shootout", style=discord.ButtonStyle.success, emoji="🚪")
    async def join(self, interaction: discord.Interaction, _b: discord.ui.Button):
        lobby = SHOOTOUT_LOBBY.get(self.channel_id)
        if not lobby or lobby.get("started"):
            await interaction.response.send_message("Lobby closed.", ephemeral=True)
            return
        if interaction.user.id in lobby["players"]:
            await interaction.response.send_message("You're already in.", ephemeral=True)
            return
        if economy.balance(interaction.user.id) < self.buy_in:
            await interaction.response.send_message(
                f"❌ Need **{self.buy_in:,}** coins to join.", ephemeral=True
            )
            return
        economy.add(interaction.user.id, -self.buy_in, "shootout buy-in")
        lobby["players"].append(interaction.user.id)
        await interaction.response.edit_message(content=_lobby_text(lobby, self.buy_in), view=self)

    @discord.ui.button(label="Start Game", style=discord.ButtonStyle.primary, emoji="▶️")
    async def start(self, interaction: discord.Interaction, _b: discord.ui.Button):
        lobby = SHOOTOUT_LOBBY.get(self.channel_id)
        if not lobby or lobby.get("started"):
            await interaction.response.send_message("Already started.", ephemeral=True)
            return
        if interaction.user.id != lobby["host_id"]:
            await interaction.response.send_message("Only the host can start.", ephemeral=True)
            return
        if len(lobby["players"]) < 2:
            await interaction.response.send_message("Need at least 2 players.", ephemeral=True)
            return
        lobby["started"] = True
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(content=_lobby_text(lobby, self.buy_in), view=self)
        # Start the actual game
        await _run_shootout(interaction, lobby, self.buy_in)


def _lobby_text(lobby: dict, buy_in: int) -> str:
    players = lobby["players"]
    pot = len(players) * buy_in
    player_list = "\n".join(f"• <@{pid}>" for pid in players) if players else "_no one yet_"
    status = "**STARTING...**" if lobby.get("started") else "_waiting for players_"
    return (
        f"# 🚪 SHOOTOUT LOBBY\n\n"
        f"Buy-in: **{buy_in:,}** coins\n"
        f"Pot: **{pot:,}** coins\n"
        f"Status: {status}\n\n"
        f"**Players ({len(players)}):**\n{player_list}\n\n"
        f"_Click Join to enter. Host clicks Start when ready._"
    )


async def _run_shootout(interaction: discord.Interaction, lobby: dict, buy_in: int):
    channel = interaction.channel
    players = list(lobby["players"])
    pot = len(players) * buy_in

    # Wait briefly so people can read the lobby
    await asyncio.sleep(2.0)

    round_num = 1
    while len(players) > 1:
        # Set up round: 4 doors, only N-1 are safe (to guarantee elimination)
        safe_doors = list(range(1, 5))
        random.shuffle(safe_doors)
        # One specific door is the death door
        death_door = random.randint(1, 4)
        safe_doors = [d for d in [1, 2, 3, 4] if d != death_door]

        # Show prompt
        round_msg = await channel.send(
            f"# 🚪 ROUND {round_num} 🚪\n\n"
            f"**Players alive:** {', '.join(f'<@{p}>' for p in players)}\n\n"
            f"Pick a door using the buttons below. **One door is rigged.**\n"
            f"You have **15 seconds** to pick.",
            allowed_mentions=discord.AllowedMentions.none(),
        )

        # Create button view with 4 doors
        view = ShootoutDoorView(players)
        await round_msg.edit(view=view)

        # Wait for picks
        await asyncio.sleep(15)
        view.disable_all()
        try:
            await round_msg.edit(view=view)
        except Exception:
            pass

        # Resolve
        picks = view.picks  # {user_id: door}
        # Auto-pick random door for anyone who didn't choose
        for p in players:
            if p not in picks:
                picks[p] = random.choice([1, 2, 3, 4])

        # Eliminate everyone who picked the death door
        eliminated = [p for p in players if picks[p] == death_door]
        if not eliminated:
            # No one died — pick one random unlucky soul
            unlucky = random.choice(players)
            eliminated = [unlucky]

        survivors = [p for p in players if p not in eliminated]

        # Build round result
        pick_lines = "\n".join(
            f"<@{p}> → Door {picks[p]} {'💀' if p in eliminated else '✅'}"
            for p in players
        )
        await channel.send(
            f"# 💥 ROUND {round_num} RESULT\n\n"
            f"The death door was **Door {death_door}**.\n\n"
            f"{pick_lines}\n\n"
            f"**Eliminated:** {', '.join(f'<@{e}>' for e in eliminated)}",
            allowed_mentions=discord.AllowedMentions.none(),
        )
        await asyncio.sleep(3)

        for p in eliminated:
            economy.record_loss(p)

        players = survivors
        round_num += 1

        if len(players) == 1:
            break

    # Winner
    winner = players[0]
    economy.add(winner, pot, "shootout win")
    economy.record_win(winner)
    await trigger_game_win(winner, "shootout", channel=channel)
    await trigger_balance_check(winner, channel=channel)
    new_bal = economy.balance(winner)
    await channel.send(
        f"# 🏆 SHOOTOUT CHAMPION 🏆\n\n"
        f"<@{winner}> survived and takes the **{pot:,}** coin pot!\n"
        f"Balance: **{new_bal:,}**",
        allowed_mentions=discord.AllowedMentions(users=[interaction.guild.get_member(winner)] if interaction.guild else False),
    )
    SHOOTOUT_LOBBY.pop(lobby.get("channel_id", ""), None)


class ShootoutDoorView(discord.ui.View):
    def __init__(self, players: list):
        super().__init__(timeout=15)
        self.players = players
        self.picks: dict[int, int] = {}

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id not in self.players:
            await interaction.response.send_message("You're not in this game.", ephemeral=True)
            return False
        if interaction.user.id in self.picks:
            await interaction.response.send_message("You already picked.", ephemeral=True)
            return False
        return True

    async def _pick(self, interaction: discord.Interaction, door: int):
        self.picks[interaction.user.id] = door
        await interaction.response.send_message(
            f"🚪 You picked Door **{door}**. Locked in.", ephemeral=True
        )

    @discord.ui.button(label="Door 1", style=discord.ButtonStyle.primary, emoji="🚪")
    async def d1(self, interaction, _b): await self._pick(interaction, 1)

    @discord.ui.button(label="Door 2", style=discord.ButtonStyle.primary, emoji="🚪")
    async def d2(self, interaction, _b): await self._pick(interaction, 2)

    @discord.ui.button(label="Door 3", style=discord.ButtonStyle.primary, emoji="🚪")
    async def d3(self, interaction, _b): await self._pick(interaction, 3)

    @discord.ui.button(label="Door 4", style=discord.ButtonStyle.primary, emoji="🚪")
    async def d4(self, interaction, _b): await self._pick(interaction, 4)

    def disable_all(self):
        for child in self.children:
            child.disabled = True


@tree.command(name="shootout", description="Host a hostage shootout. Players pick doors. Last alive wins the pot.")
@discord.app_commands.describe(buy_in="How many coins each player puts into the pot")
async def shootout_command(interaction: discord.Interaction, buy_in: int = 200):
    if buy_in <= 0:
        await interaction.response.send_message("Buy-in must be positive.", ephemeral=True)
        return
    channel_id = str(interaction.channel_id)
    if channel_id in SHOOTOUT_LOBBY:
        await interaction.response.send_message("A shootout is already running here.", ephemeral=True)
        return
    if economy.balance(interaction.user.id) < buy_in:
        await interaction.response.send_message(
            f"❌ Need **{buy_in:,}** coins to host.", ephemeral=True
        )
        return

    economy.add(interaction.user.id, -buy_in, "shootout host buy-in")
    lobby = {
        "host_id": interaction.user.id,
        "channel_id": channel_id,
        "players": [interaction.user.id],
        "started": False,
    }
    SHOOTOUT_LOBBY[channel_id] = lobby

    view = ShootoutLobbyView(channel_id, buy_in)
    await interaction.response.send_message(content=_lobby_text(lobby, buy_in), view=view)


# ── 💣 /bomb — hot potato ────────────────────────────────────────────────────
ACTIVE_BOMBS: dict[str, dict] = {}  # channel_id -> bomb state


class BombPassModal(discord.ui.Modal, title="💣 Pass the Bomb"):
    target = discord.ui.TextInput(
        label="Tag user to pass to (paste their @mention)",
        placeholder="@username",
        required=True,
        max_length=100,
    )

    def __init__(self, channel_id: str, holder_id: int):
        super().__init__()
        self.channel_id = channel_id
        self.holder_id = holder_id

    async def on_submit(self, interaction: discord.Interaction):
        bomb = ACTIVE_BOMBS.get(self.channel_id)
        if not bomb or bomb.get("exploded"):
            await interaction.response.send_message("Bomb is gone.", ephemeral=True)
            return
        if bomb["holder_id"] != interaction.user.id:
            await interaction.response.send_message("You don't have the bomb!", ephemeral=True)
            return

        # Parse target mention
        m = re.match(r"<@!?(\d+)>", str(self.target).strip())
        target_id = None
        if m:
            target_id = int(m.group(1))
        else:
            # Try by name
            name = str(self.target).strip().lstrip("@").lower()
            if interaction.guild:
                for member in interaction.guild.members:
                    if name in (member.display_name.lower(), member.name.lower()):
                        target_id = member.id
                        break
        if not target_id:
            await interaction.response.send_message("❌ Couldn't find that user.", ephemeral=True)
            return
        if target_id == interaction.user.id:
            await interaction.response.send_message("Can't pass to yourself.", ephemeral=True)
            return
        target = interaction.guild.get_member(target_id) if interaction.guild else None
        if not target or target.bot:
            await interaction.response.send_message("❌ Invalid target.", ephemeral=True)
            return

        bomb["holder_id"] = target_id
        bomb["passes"] = bomb.get("passes", 0) + 1
        # Track who passed the bomb (for achievement)
        await trigger_event(interaction.user.id, "bomb_passed", channel=interaction.channel)
        await interaction.response.send_message(
            f"💣 {interaction.user.mention} passed the bomb to {target.mention}!\n"
            f"⏰ Time remaining: ~**{int(bomb['expires_at'] - time.time())}s**\n"
            f"_Pass it before it explodes!_",
            allowed_mentions=discord.AllowedMentions(users=[target]),
        )


class BombHoldView(discord.ui.View):
    def __init__(self, channel_id: str):
        super().__init__(timeout=None)
        self.channel_id = channel_id

    @discord.ui.button(label="Pass the Bomb 💣", style=discord.ButtonStyle.danger, emoji="🔥")
    async def pass_btn(self, interaction: discord.Interaction, _b: discord.ui.Button):
        bomb = ACTIVE_BOMBS.get(self.channel_id)
        if not bomb or bomb.get("exploded"):
            await interaction.response.send_message("Bomb is gone.", ephemeral=True)
            return
        if interaction.user.id != bomb["holder_id"]:
            await interaction.response.send_message(
                f"You don't have the bomb! <@{bomb['holder_id']}> does.",
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            return
        await interaction.response.send_modal(BombPassModal(self.channel_id, interaction.user.id))


@tree.command(name="bomb", description="Hot potato. Pass the bomb to a user. Whoever holds it when it explodes loses coins.")
@discord.app_commands.describe(target="Who you're throwing the bomb to first", stakes="Coins the loser pays (default 500)")
async def bomb_command(interaction: discord.Interaction, target: discord.Member, stakes: int = 500):
    channel_id = str(interaction.channel_id)
    if channel_id in ACTIVE_BOMBS and not ACTIVE_BOMBS[channel_id].get("exploded"):
        await interaction.response.send_message("A bomb is already in play here.", ephemeral=True)
        return
    if target.id == interaction.user.id:
        await interaction.response.send_message("Can't bomb yourself.", ephemeral=True)
        return
    if target.bot:
        await interaction.response.send_message("Bots don't play.", ephemeral=True)
        return
    if stakes < 50:
        await interaction.response.send_message("Stakes too low (min 50).", ephemeral=True)
        return

    # Random fuse 30-90 seconds
    fuse = random.randint(30, 90)
    expires_at = time.time() + fuse

    ACTIVE_BOMBS[channel_id] = {
        "holder_id": target.id,
        "stakes": stakes,
        "expires_at": expires_at,
        "exploded": False,
        "passes": 0,
    }

    view = BombHoldView(channel_id)
    await interaction.response.send_message(
        f"# 💣 LIVE BOMB! 💣\n\n"
        f"{interaction.user.mention} threw a bomb at {target.mention}!\n"
        f"⏰ Fuse: **{fuse} seconds**\n"
        f"💸 Stakes: **{stakes:,}** coins\n\n"
        f"_{target.mention}, click the button to pass it before it blows!_",
        view=view,
        allowed_mentions=discord.AllowedMentions(users=[target]),
    )

    # Schedule explosion
    asyncio.create_task(_bomb_explode(interaction, channel_id, fuse))


async def _bomb_explode(interaction: discord.Interaction, channel_id: str, fuse: int):
    await asyncio.sleep(fuse)
    bomb = ACTIVE_BOMBS.get(channel_id)
    if not bomb or bomb.get("exploded"):
        return
    bomb["exploded"] = True

    holder_id = bomb["holder_id"]
    stakes = bomb["stakes"]
    actual_loss = min(stakes, economy.balance(holder_id))
    economy.add(holder_id, -actual_loss, "bomb explosion")
    economy.record_loss(holder_id)
    new_bal = economy.balance(holder_id)

    channel = interaction.channel
    await channel.send(
        f"# 💥💥💥 **BOOM** 💥💥💥\n\n"
        f"<@{holder_id}> was holding the bomb when it exploded!\n"
        f"💸 Lost **{actual_loss:,}** coins.\n"
        f"📊 Bomb was passed **{bomb['passes']}** times.\n"
        f"Balance: **{new_bal:,}**",
        allowed_mentions=discord.AllowedMentions.none(),
    )
    ACTIVE_BOMBS.pop(channel_id, None)


# ── 🟡 /connect4 ─────────────────────────────────────────────────────────────
ACTIVE_C4: dict[str, dict] = {}  # channel_id -> game state

C4_ROWS, C4_COLS = 6, 7
EMPTY = "⚪"
P1_PIECE = "🔴"
P2_PIECE = "🟡"


def _new_c4_board():
    return [[EMPTY] * C4_COLS for _ in range(C4_ROWS)]


def _render_c4(board, p1_id, p2_id, current_id, status="", last_col=None):
    cols = " ".join(["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣"])
    rows_str = "\n".join(" ".join(row) for row in board)
    turn_indicator = f"<@{current_id}>'s turn" if current_id else ""
    header = f"## 🟡 CONNECT 4\n<@{p1_id}> {P1_PIECE} vs {P2_PIECE} <@{p2_id}>\n\n"
    if status:
        return header + status + "\n\n" + rows_str + "\n" + cols
    return header + turn_indicator + "\n\n" + rows_str + "\n" + cols


def _drop_piece(board, col, piece) -> int:
    """Drop piece in column. Returns row index or -1 if column full."""
    for r in range(C4_ROWS - 1, -1, -1):
        if board[r][col] == EMPTY:
            board[r][col] = piece
            return r
    return -1


def _check_win(board, piece) -> bool:
    # Horizontal
    for r in range(C4_ROWS):
        for c in range(C4_COLS - 3):
            if all(board[r][c+i] == piece for i in range(4)):
                return True
    # Vertical
    for r in range(C4_ROWS - 3):
        for c in range(C4_COLS):
            if all(board[r+i][c] == piece for i in range(4)):
                return True
    # Diagonal /
    for r in range(3, C4_ROWS):
        for c in range(C4_COLS - 3):
            if all(board[r-i][c+i] == piece for i in range(4)):
                return True
    # Diagonal \
    for r in range(C4_ROWS - 3):
        for c in range(C4_COLS - 3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True
    return False


def _board_full(board) -> bool:
    return all(board[0][c] != EMPTY for c in range(C4_COLS))


class C4View(discord.ui.View):
    def __init__(self, channel_id: str):
        super().__init__(timeout=300)
        self.channel_id = channel_id
        # Add 7 column buttons
        for col in range(C4_COLS):
            label = str(col + 1)
            btn = discord.ui.Button(label=label, style=discord.ButtonStyle.primary, custom_id=f"c4_col_{col}")
            btn.callback = self._make_callback(col)
            self.add_item(btn)

    def _make_callback(self, col: int):
        async def cb(interaction: discord.Interaction):
            await self._handle_drop(interaction, col)
        return cb

    async def _handle_drop(self, interaction: discord.Interaction, col: int):
        game = ACTIVE_C4.get(self.channel_id)
        if not game or game.get("ended"):
            await interaction.response.send_message("Game over.", ephemeral=True)
            return
        if interaction.user.id != game["current_id"]:
            await interaction.response.send_message("Not your turn.", ephemeral=True)
            return

        row = _drop_piece(game["board"], col, game["current_piece"])
        if row == -1:
            await interaction.response.send_message("Column is full.", ephemeral=True)
            return

        # Check win
        if _check_win(game["board"], game["current_piece"]):
            game["ended"] = True
            winner_id = game["current_id"]
            loser_id = game["p2_id"] if winner_id == game["p1_id"] else game["p1_id"]
            economy.add(winner_id, game["pot"], "connect4 win")
            economy.record_win(winner_id)
            economy.record_loss(loser_id)
            new_bal = economy.balance(winner_id)
            for child in self.children:
                child.disabled = True
            status = f"## 🏆 <@{winner_id}> WINS! Took **{game['pot']:,}** coins.\nBalance: **{new_bal:,}**"
            await interaction.response.edit_message(
                content=_render_c4(game["board"], game["p1_id"], game["p2_id"], None, status=status),
                view=self,
                allowed_mentions=discord.AllowedMentions(users=[interaction.guild.get_member(winner_id)] if interaction.guild else False),
            )
            ACTIVE_C4.pop(self.channel_id, None)
            return

        # Check tie
        if _board_full(game["board"]):
            game["ended"] = True
            # Refund both players half the pot
            refund = game["pot"] // 2
            economy.add(game["p1_id"], refund, "connect4 tie")
            economy.add(game["p2_id"], refund, "connect4 tie")
            for child in self.children:
                child.disabled = True
            status = f"## 🤝 TIE! Pot refunded ({refund:,} each)."
            await interaction.response.edit_message(
                content=_render_c4(game["board"], game["p1_id"], game["p2_id"], None, status=status),
                view=self,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            ACTIVE_C4.pop(self.channel_id, None)
            return

        # Switch turns
        if game["current_id"] == game["p1_id"]:
            game["current_id"] = game["p2_id"]
            game["current_piece"] = P2_PIECE
        else:
            game["current_id"] = game["p1_id"]
            game["current_piece"] = P1_PIECE

        await interaction.response.edit_message(
            content=_render_c4(game["board"], game["p1_id"], game["p2_id"], game["current_id"]),
            view=self,
            allowed_mentions=discord.AllowedMentions.none(),
        )


@tree.command(name="connect4", description="Challenge a user to Connect 4. Winner takes the pot.")
@discord.app_commands.describe(opponent="Who to challenge", wager="Coins each player wagers (default 200)")
async def connect4_command(interaction: discord.Interaction, opponent: discord.Member, wager: int = 200):
    challenger = interaction.user
    if opponent.id == challenger.id:
        await interaction.response.send_message("Can't challenge yourself.", ephemeral=True)
        return
    if opponent.bot:
        await interaction.response.send_message("Bots don't play.", ephemeral=True)
        return
    if wager <= 0:
        await interaction.response.send_message("Wager must be positive.", ephemeral=True)
        return
    channel_id = str(interaction.channel_id)
    if channel_id in ACTIVE_C4:
        await interaction.response.send_message("A Connect 4 game is already in this channel.", ephemeral=True)
        return
    if economy.balance(challenger.id) < wager or economy.balance(opponent.id) < wager:
        await interaction.response.send_message(
            f"❌ Both players need **{wager:,}** coins.", ephemeral=True
        )
        return

    # Take wagers
    economy.add(challenger.id, -wager, "connect4 wager")
    economy.add(opponent.id, -wager, "connect4 wager")
    pot = wager * 2

    board = _new_c4_board()
    ACTIVE_C4[channel_id] = {
        "p1_id": challenger.id,
        "p2_id": opponent.id,
        "board": board,
        "current_id": challenger.id,
        "current_piece": P1_PIECE,
        "pot": pot,
        "ended": False,
    }

    view = C4View(channel_id)
    await interaction.response.send_message(
        content=_render_c4(board, challenger.id, opponent.id, challenger.id),
        view=view,
        allowed_mentions=discord.AllowedMentions(users=[opponent]),
    )


# ── 🥊 /fight — animated fight w/ spectator betting ─────────────────────────
ACTIVE_FIGHTS: dict[str, dict] = {}  # channel_id -> fight state

FIGHT_BETTING_WINDOW = 30   # seconds for spectators to place bets
FIGHT_DURATION       = 60   # seconds of actual fighting
FIGHT_MIN_WAGER      = 100
FIGHT_MIN_BET        = 50

# Fight move flavor pool — random combat actions for the animation
FIGHT_MOVES = [
    ("🥊", "lands a clean jab"),
    ("👊", "throws a hook"),
    ("🦵", "drops a vicious low kick"),
    ("💢", "headbutts"),
    ("🤜", "delivers an uppercut"),
    ("🧠", "psychs them out with mind games"),
    ("🌀", "ducks and counters"),
    ("⚡", "lands a flash combo"),
    ("🩸", "draws blood with a sharp elbow"),
    ("🔥", "is on fire — landing everything"),
    ("💀", "lands a knockdown blow"),
    ("🛡️", "blocks and pivots"),
    ("🎯", "lands a precision strike"),
    ("💥", "explodes with a flurry"),
    ("🪃", "fakes left, hits right"),
    ("👀", "spots an opening"),
    ("🥷", "moves like a ghost"),
    ("🐍", "sneaks in a body shot"),
]

FIGHT_TRASH_TALK = [
    "calls them weak",
    "yells 'is that all?!'",
    "starts taunting",
    "whispers something cruel",
    "laughs in their face",
    "blows a kiss",
    "points at the crowd",
]


class FightBetView(discord.ui.View):
    """View shown during the betting window. Spectators click to bet on a fighter."""
    def __init__(self, channel_id: str):
        super().__init__(timeout=FIGHT_BETTING_WINDOW)
        self.channel_id = channel_id

    @discord.ui.button(label="Bet on Fighter A", style=discord.ButtonStyle.danger, emoji="🔴")
    async def bet_a(self, interaction: discord.Interaction, _b: discord.ui.Button):
        await interaction.response.send_modal(FightBetModal(self.channel_id, "a"))

    @discord.ui.button(label="Bet on Fighter B", style=discord.ButtonStyle.primary, emoji="🔵")
    async def bet_b(self, interaction: discord.Interaction, _b: discord.ui.Button):
        await interaction.response.send_modal(FightBetModal(self.channel_id, "b"))


class FightBetModal(discord.ui.Modal, title="Place your bet"):
    amount = discord.ui.TextInput(
        label="Coins to bet",
        placeholder="e.g. 250",
        required=True,
        max_length=10,
    )

    def __init__(self, channel_id: str, side: str):
        super().__init__()
        self.channel_id = channel_id
        self.side = side  # "a" or "b"

    async def on_submit(self, interaction: discord.Interaction):
        fight = ACTIVE_FIGHTS.get(self.channel_id)
        if not fight or fight.get("phase") != "betting":
            await interaction.response.send_message("Betting is closed.", ephemeral=True)
            return

        # Don't let fighters bet on their own match
        if interaction.user.id in (fight["a_id"], fight["b_id"]):
            await interaction.response.send_message(
                "You can't bet on your own fight, slugger.", ephemeral=True
            )
            return

        # Parse amount
        try:
            bet_amount = int(str(self.amount).strip())
        except ValueError:
            await interaction.response.send_message("Bet must be a number.", ephemeral=True)
            return
        if bet_amount < FIGHT_MIN_BET:
            await interaction.response.send_message(
                f"Min bet is **{FIGHT_MIN_BET}** coins.", ephemeral=True
            )
            return
        if bet_amount > economy.balance(interaction.user.id):
            await interaction.response.send_message(
                f"❌ You only have **{economy.balance(interaction.user.id):,}** coins.",
                ephemeral=True,
            )
            return

        # Don't allow switching sides — use existing bet's side
        existing = fight["bets"].get(interaction.user.id)
        if existing and existing["side"] != self.side:
            await interaction.response.send_message(
                f"You already bet on the other fighter. Stick with your pick.", ephemeral=True
            )
            return

        # Take the coins
        economy.add(interaction.user.id, -bet_amount, "fight bet")
        if existing:
            existing["amount"] += bet_amount
        else:
            fight["bets"][interaction.user.id] = {"side": self.side, "amount": bet_amount}

        side_label = f"🔴 {fight['a_name']}" if self.side == "a" else f"🔵 {fight['b_name']}"
        total_bet = fight["bets"][interaction.user.id]["amount"]
        await interaction.response.send_message(
            f"💰 You bet **{bet_amount:,}** more on {side_label}.\n"
            f"Total stake on this fight: **{total_bet:,}** coins.",
            ephemeral=True,
        )


def _fight_status_text(fight: dict, header: str = "") -> str:
    a_pool = sum(b["amount"] for b in fight["bets"].values() if b["side"] == "a")
    b_pool = sum(b["amount"] for b in fight["bets"].values() if b["side"] == "b")
    a_bettors = sum(1 for b in fight["bets"].values() if b["side"] == "a")
    b_bettors = sum(1 for b in fight["bets"].values() if b["side"] == "b")
    wager = fight["wager"]
    body = (
        f"**🔴 {fight['a_mention']}** vs **🔵 {fight['b_mention']}**\n"
        f"Each fighter staked **{wager:,}** coins.\n\n"
        f"**Spectator pool:**\n"
        f"🔴 Backing {fight['a_name']}: **{a_pool:,}** coins ({a_bettors} bettor{'s' if a_bettors != 1 else ''})\n"
        f"🔵 Backing {fight['b_name']}: **{b_pool:,}** coins ({b_bettors} bettor{'s' if b_bettors != 1 else ''})\n"
    )
    if header:
        return header + "\n\n" + body
    return body


def _build_fight_embed(fight: dict, phase: str = "betting", **kwargs) -> discord.Embed:
    """Build the fight embed for any phase: betting, fighting, ended."""
    a_pool = sum(b["amount"] for b in fight["bets"].values() if b["side"] == "a")
    b_pool = sum(b["amount"] for b in fight["bets"].values() if b["side"] == "b")
    a_bettors = sum(1 for b in fight["bets"].values() if b["side"] == "a")
    b_bettors = sum(1 for b in fight["bets"].values() if b["side"] == "b")
    wager = fight["wager"]

    if phase == "betting":
        remaining = kwargs.get("remaining", FIGHT_BETTING_WINDOW)
        title = f"🥊 FIGHT NIGHT — {remaining}s LEFT TO BET"
        color = discord.Color.gold()
    elif phase == "fighting":
        time_left = kwargs.get("time_left", FIGHT_DURATION)
        title = f"🥊 LIVE FIGHT — {time_left}s LEFT"
        color = discord.Color.red()
    elif phase == "ended":
        title = "🏆 FIGHT OVER"
        color = discord.Color.green()
    else:
        title = "🥊 FIGHT"
        color = discord.Color.blue()

    embed = discord.Embed(title=title, color=color)
    embed.add_field(
        name="🥊 Match",
        value=f"🔴 {fight['a_mention']} vs 🔵 {fight['b_mention']}\nWager: **{wager:,}** coins each",
        inline=False,
    )

    # Score (only during fight + after)
    if phase in ("fighting", "ended"):
        a_score = kwargs.get("a_score", 0)
        b_score = kwargs.get("b_score", 0)
        embed.add_field(
            name="📊 Score",
            value=f"🔴 **{fight['a_name']}**: {a_score}\n🔵 **{fight['b_name']}**: {b_score}",
            inline=False,
        )

    # Betting pool — always visible
    pool_value = (
        f"🔴 {fight['a_name']}: **{a_pool:,}** ({a_bettors} bettor{'s' if a_bettors != 1 else ''})\n"
        f"🔵 {fight['b_name']}: **{b_pool:,}** ({b_bettors} bettor{'s' if b_bettors != 1 else ''})"
    )
    embed.add_field(name="💰 Spectator Pool", value=pool_value, inline=False)

    # Recent action log (during/after fight)
    log_lines = kwargs.get("log_lines")
    if log_lines:
        action = "\n".join(log_lines[-6:])
        if len(action) > 1020:
            action = action[-1020:]
        embed.add_field(name="📝 Action", value=action, inline=False)

    # Final result
    if phase == "ended":
        winner_text = kwargs.get("winner_text", "")
        payout_text = kwargs.get("payout_text", "")
        if winner_text:
            embed.add_field(name="🏆 Winner", value=winner_text, inline=False)
        if payout_text:
            embed.add_field(name="💸 Payouts", value=payout_text[:1020], inline=False)

    return embed


async def _edit_fight_message(fight: dict, **kwargs):
    """Edit the single fight message with a new embed."""
    msg = fight.get("message")
    if not msg:
        return
    try:
        view = fight.get("view") if kwargs.get("phase") == "betting" else None
        kw = {"embed": _build_fight_embed(fight, **kwargs)}
        if view is not None:
            kw["view"] = view
        elif kwargs.get("phase") == "ended":
            # Disable view on end
            v = fight.get("view")
            if v:
                for child in v.children:
                    child.disabled = True
                kw["view"] = v
        await msg.edit(**kw)
    except Exception as e:
        log.warning("fight message edit failed: %s", e)


@tree.command(name="fight", description="60-second 1v1 fight with spectator betting. Big wagers, full embed animation.")
@discord.app_commands.describe(opponent="Who you're fighting", wager="Coins each fighter puts up (min 100)")
async def fight_command(interaction: discord.Interaction, opponent: discord.Member, wager: int = 200):
    challenger = interaction.user
    if opponent.id == challenger.id:
        await interaction.response.send_message("Can't fight yourself.", ephemeral=True)
        return
    if opponent.bot:
        await interaction.response.send_message("Bots don't fight.", ephemeral=True)
        return
    if wager < FIGHT_MIN_WAGER:
        await interaction.response.send_message(
            f"Wager must be at least **{FIGHT_MIN_WAGER}** coins.", ephemeral=True
        )
        return

    channel_id = str(interaction.channel_id)
    if channel_id in ACTIVE_FIGHTS:
        await interaction.response.send_message("A fight is already happening here.", ephemeral=True)
        return

    if economy.balance(challenger.id) < wager:
        await interaction.response.send_message(
            f"❌ You need **{wager:,}** coins.", ephemeral=True
        )
        return
    if economy.balance(opponent.id) < wager:
        await interaction.response.send_message(
            f"❌ {opponent.mention} only has **{economy.balance(opponent.id):,}** coins.",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    # Take wagers from both fighters
    economy.add(challenger.id, -wager, "fight wager")
    economy.add(opponent.id, -wager, "fight wager")

    fight = {
        "a_id": challenger.id,
        "a_name": challenger.display_name,
        "a_mention": challenger.mention,
        "b_id": opponent.id,
        "b_name": opponent.display_name,
        "b_mention": opponent.mention,
        "wager": wager,
        "bets": {},        # user_id -> {side, amount}
        "phase": "betting",
    }
    ACTIVE_FIGHTS[channel_id] = fight

    # Send the initial fight card with betting buttons — ONE embed that updates
    view = FightBetView(channel_id)
    embed = _build_fight_embed(fight, phase="betting")
    await interaction.response.send_message(
        embed=embed,
        view=view,
        allowed_mentions=discord.AllowedMentions(users=[opponent]),
    )
    fight_message = await interaction.original_response()
    fight["message"] = fight_message  # store ref so _run_fight can edit it
    fight["view"] = view

    # Run the full fight asynchronously so we don't hold the interaction
    asyncio.create_task(_run_fight(interaction, channel_id))


async def _run_fight(interaction: discord.Interaction, channel_id: str):
    fight = ACTIVE_FIGHTS.get(channel_id)
    if not fight:
        return
    channel = interaction.channel

    # ── Betting window — ticks down by editing the SAME message ──────────────
    elapsed = 0
    while elapsed < FIGHT_BETTING_WINDOW:
        await asyncio.sleep(2)
        elapsed += 2
        remaining = FIGHT_BETTING_WINDOW - elapsed
        await _edit_fight_message(fight, phase="betting", remaining=max(remaining, 0))

    # ── Close betting & start fight ──────────────────────────────────────────
    fight["phase"] = "fighting"
    a_score = 0
    b_score = 0
    log_lines = [
        f"🔔 **DING DING DING!** {fight['a_name']} and {fight['b_name']} step into the ring."
    ]
    await _edit_fight_message(
        fight, phase="fighting",
        time_left=FIGHT_DURATION, a_score=a_score, b_score=b_score, log_lines=log_lines,
    )
    await asyncio.sleep(2.0)

    # ── Fight animation — keeps editing the SAME embed ───────────────────────
    fight_start = time.time()
    while time.time() - fight_start < FIGHT_DURATION:
        await asyncio.sleep(random.uniform(3.5, 5.5))

        attacker_id = random.choice([fight["a_id"], fight["b_id"]])
        attacker_name = fight["a_name"] if attacker_id == fight["a_id"] else fight["b_name"]
        defender_name = fight["b_name"] if attacker_id == fight["a_id"] else fight["a_name"]

        emoji, action = random.choice(FIGHT_MOVES)

        roll = random.random()
        if roll < 0.65:
            # Hit
            damage = random.randint(1, 4)
            if attacker_id == fight["a_id"]:
                a_score += damage
            else:
                b_score += damage
            log_lines.append(f"{emoji} **{attacker_name}** {action} on {defender_name}! **(+{damage})**")
        elif roll < 0.85:
            log_lines.append(f"💨 **{attacker_name}** swings — and {defender_name} dodges it.")
        else:
            damage = random.randint(2, 5)
            if attacker_id == fight["a_id"]:
                b_score += damage
            else:
                a_score += damage
            log_lines.append(
                f"🌀 **{defender_name}** counters! **{attacker_name}** eats it. **(+{damage} for {defender_name})**"
            )

        if random.random() < 0.18:
            talker = random.choice([fight["a_name"], fight["b_name"]])
            log_lines.append(f"🗯️ {talker} {random.choice(FIGHT_TRASH_TALK)}.")

        time_left = max(int(FIGHT_DURATION - (time.time() - fight_start)), 0)
        await _edit_fight_message(
            fight, phase="fighting",
            time_left=time_left, a_score=a_score, b_score=b_score, log_lines=log_lines,
        )

    # ── Decide winner ────────────────────────────────────────────────────────
    fight["phase"] = "ended"
    if a_score > b_score:
        winner_id, loser_id = fight["a_id"], fight["b_id"]
        winner_name = fight["a_name"]
        winning_side = "a"
    elif b_score > a_score:
        winner_id, loser_id = fight["b_id"], fight["a_id"]
        winner_name = fight["b_name"]
        winning_side = "b"
    else:
        if random.choice([True, False]):
            winner_id, loser_id = fight["a_id"], fight["b_id"]
            winner_name = fight["a_name"]
            winning_side = "a"
        else:
            winner_id, loser_id = fight["b_id"], fight["a_id"]
            winner_name = fight["b_name"]
            winning_side = "b"

    fighter_payout = fight["wager"] * 2
    armor = get_perk(loser_id, "fight_armor_pct")
    if armor:
        refund = int(fight["wager"] * armor / 100)
        if refund > 0:
            economy.add(loser_id, refund, "fight armor refund")
    economy.add(winner_id, fighter_payout, "fight win")
    economy.record_win(winner_id)
    economy.record_loss(loser_id)
    # Claim any bounty on the loser
    try:
        await claim_bounty(winner_id, loser_id, channel)
    except Exception:
        pass
    await trigger_game_win(winner_id, "fight", channel=channel)
    await trigger_balance_check(winner_id, channel=channel)

    # Resolve spectator bets
    a_pool = sum(b["amount"] for b in fight["bets"].values() if b["side"] == "a")
    b_pool = sum(b["amount"] for b in fight["bets"].values() if b["side"] == "b")
    total_pool = a_pool + b_pool
    winning_pool = a_pool if winning_side == "a" else b_pool
    losing_pool = b_pool if winning_side == "a" else a_pool

    payout_lines = []
    if winning_pool > 0:
        for uid, b in fight["bets"].items():
            if b["side"] == winning_side:
                share = (b["amount"] / winning_pool) * losing_pool
                payout = b["amount"] + int(share)
                economy.add(uid, payout, "fight bet win")
                profit = payout - b["amount"]
                payout_lines.append(f"✅ <@{uid}> won **{payout:,}** (+{profit:,})")
            else:
                payout_lines.append(f"❌ <@{uid}> lost **{b['amount']:,}**")
    else:
        for uid, b in fight["bets"].items():
            economy.add(uid, b["amount"], "fight bet refund")
            payout_lines.append(f"↩️ <@{uid}> refunded {b['amount']:,}")

    winner_text = (
        f"## 🏆 {winner_name} WINS!\n"
        f"💰 Earned **{fighter_payout:,}** coins"
    )
    payout_text = "\n".join(payout_lines) if payout_lines else "_No spectators placed bets._"

    await _edit_fight_message(
        fight, phase="ended",
        a_score=a_score, b_score=b_score,
        log_lines=log_lines,
        winner_text=winner_text, payout_text=payout_text,
    )

    # Single ping for the winner
    try:
        winner_member = interaction.guild.get_member(winner_id) if interaction.guild else None
        if winner_member:
            await channel.send(
                f"🏆 {winner_member.mention} took the fight!",
                allowed_mentions=discord.AllowedMentions(users=[winner_member]),
            )
    except Exception:
        pass

    ACTIVE_FIGHTS.pop(channel_id, None)


# ── 🔮 /tarot — animated tarot reading ───────────────────────────────────────
TAROT_DECK = [
    "The Fool", "The Magician", "The High Priestess", "The Empress",
    "The Emperor", "The Hierophant", "The Lovers", "The Chariot",
    "Strength", "The Hermit", "Wheel of Fortune", "Justice",
    "The Hanged Man", "Death", "Temperance", "The Devil",
    "The Tower", "The Star", "The Moon", "The Sun",
    "Judgement", "The World",
]


@tree.command(name="tarot", description="Get an AI-interpreted 3-card tarot reading based on your chat history.")
async def tarot_command(interaction: discord.Interaction):
    cfg = load_config()
    user = interaction.user
    silent = discord.AllowedMentions.none()

    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    # Pick 3 unique cards
    cards = random.sample(TAROT_DECK, 3)
    reversed_flags = [random.choice([True, False]) for _ in range(3)]
    positions = ["PAST", "PRESENT", "FUTURE"]

    # Animation: shuffle deck
    await edit(f"🔮 *{user.mention} sits at the table...*")
    await asyncio.sleep(1.0)
    await edit("🔮 *the cards are shuffled...*\n\n🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠 🂠")
    await asyncio.sleep(1.2)
    await edit("🔮 *three cards are drawn from the deck...*\n\n🂠 🂠 🂠")
    await asyncio.sleep(1.3)

    # Reveal each card slowly
    revealed = ["🂠", "🂠", "🂠"]
    for i, (pos, card, rev) in enumerate(zip(positions, cards, reversed_flags)):
        await edit(
            f"🔮 *flipping the **{pos}** card...*\n\n"
            + " ".join(revealed)
        )
        await asyncio.sleep(1.1)
        # Render the card (rev marker if reversed)
        marker = "🔄" if rev else "🃏"
        revealed[i] = marker
        await edit(
            f"🔮 **{pos}: {card}** {'(reversed)' if rev else ''}\n\n"
            + " ".join(revealed) + "\n"
            + " ".join(["PAST    ", "PRESENT", "FUTURE  "])
        )
        await asyncio.sleep(1.5)

    # Gather context for AI interpretation (from this user's recent messages)
    user_messages: list[str] = []
    try:
        async for m in interaction.channel.history(limit=500):
            if m.author.id == user.id and m.content and m.content.strip():
                user_messages.append(m.content.strip())
                if len(user_messages) >= 30:
                    break
    except Exception:
        pass
    history_blob = "\n".join(f"- {m}" for m in user_messages) if user_messages else "(this person doesn't post much)"

    # AI interpretation
    cards_summary = "\n".join(
        f"{pos}: {card}{' (reversed)' if rev else ''}"
        for pos, card, rev in zip(positions, cards, reversed_flags)
    )

    sys = (
        cfg["system_prompt"] +
        "\n\n=== TAROT READER MODE ===\n"
        f"You are reading tarot for {user.display_name}. The 3 cards drawn:\n"
        f"{cards_summary}\n\n"
        "Write a short interpretation: 1 sentence per card (PAST, PRESENT, FUTURE), "
        "weaving in things from their actual chat history when relevant. "
        "End with 1 sentence of overall guidance. "
        "Stay completely in character. Be mystical but cutting. "
        "Format strictly:\n"
        "**PAST:** [interpretation]\n"
        "**PRESENT:** [interpretation]\n"
        "**FUTURE:** [interpretation]\n\n"
        "**THE READING:** [closing line]"
    )

    await edit(f"🔮 *the reader interprets the cards...*\n\n{cards_summary}")
    await asyncio.sleep(1.0)

    try:
        reading = await ask_ai(
            sys,
            [{"role": "user", "content": f"Recent messages from {user.display_name}:\n\n{history_blob}\n\nRead the cards."}],
            {**cfg, "max_tokens": 400},
        )
    except Exception:
        reading = ""
    if reading.startswith("⚠️") or not reading.strip():
        reading = (
            f"**PAST:** Your past holds shadows you'd rather forget.\n"
            f"**PRESENT:** You stand at a crossroads of bad decisions.\n"
            f"**FUTURE:** The path ahead bends toward chaos.\n\n"
            f"**THE READING:** Trust nothing, especially yourself."
        )

    final = (
        f"# 🔮 TAROT READING — {user.display_name}\n\n"
        f"```\n  PAST       PRESENT     FUTURE\n  {cards[0][:9]:<11}{cards[1][:9]:<12}{cards[2][:9]:<10}\n```\n"
        f"{reading}"
    )
    await edit(final)
    await trigger_event(user.id, "tarot_read", channel=interaction.channel)


# ── ⚖️ /lawsuit — sue a user, AI judges, real damages ───────────────────────
LAWSUIT_FILING_FEE = 200


@tree.command(name="lawsuit", description="Civil trial: sue a user for damages. Real coins change hands.")
@discord.app_commands.describe(
    defendant="Who you're suing",
    claim="What they did to you",
)
async def lawsuit_command(interaction: discord.Interaction, defendant: discord.Member, claim: str):
    cfg = load_config()
    silent = discord.AllowedMentions.none()
    plaintiff = interaction.user

    if defendant.id == plaintiff.id:
        await interaction.response.send_message("You can't sue yourself.", ephemeral=True)
        return
    if defendant.bot:
        await interaction.response.send_message("You can't sue a bot.", ephemeral=True)
        return
    if str(defendant.id) in cfg.get("respected_users", []):
        await interaction.response.send_message(
            f"❌ {defendant.mention} is the boss. They're above the law.",
            ephemeral=True, allowed_mentions=silent,
        )
        return

    bal = economy.balance(plaintiff.id)
    if bal < LAWSUIT_FILING_FEE:
        await interaction.response.send_message(
            f"❌ Filing fee is **{LAWSUIT_FILING_FEE:,}** coins. You have **{bal:,}**.",
            ephemeral=True,
        )
        return

    economy.add(plaintiff.id, -LAWSUIT_FILING_FEE, "lawsuit filing fee")
    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    # Animated court intro
    await edit("⚖️ **CIVIL COURT IS NOW IN SESSION**")
    await asyncio.sleep(1.5)
    await edit(
        f"⚖️ **CASE FILED**\n\n"
        f"**Plaintiff:** {plaintiff.mention}\n"
        f"**Defendant:** {defendant.mention}\n"
        f"**Filing fee paid:** {LAWSUIT_FILING_FEE:,} coins\n\n"
        f"📜 **Claim:** _{claim}_"
    )
    await asyncio.sleep(2.5)
    await edit(f"⚖️ *The judge reviews the claim...*")
    await asyncio.sleep(1.8)
    await edit(f"⚖️ 🔨 *gavel raised...*")
    await asyncio.sleep(1.3)

    # AI judge
    judge_system = (
        cfg["system_prompt"] +
        "\n\n=== CIVIL COURT JUDGE MODE ===\n"
        f"You are a comedy judge ruling on a lawsuit between two users. "
        f"Plaintiff {plaintiff.display_name} sued defendant {defendant.display_name} for: '{claim}'. "
        "Deliver a 2-3 sentence ruling — be theatrical and absurd. "
        "End with one of these EXACT formats on a new line:\n"
        "AWARD: <number> coins\n"
        "or\n"
        "DISMISSED\n"
        "or\n"
        "COUNTERSUIT: <number> coins (this means the plaintiff has to pay the defendant)\n\n"
        "Pick AWARD if the case has merit (use a number between 100 and 2000). "
        "Pick DISMISSED if the lawsuit is frivolous. "
        "Pick COUNTERSUIT (rare) if the plaintiff is clearly the bad guy. "
        "Stay in character."
    )
    user_prompt = (
        f"Plaintiff: {plaintiff.display_name}\n"
        f"Defendant: {defendant.display_name}\n"
        f"Claim: {claim}\n\n"
        f"Deliver your ruling."
    )

    try:
        ruling = await ask_ai(
            judge_system,
            [{"role": "user", "content": user_prompt}],
            {**cfg, "max_tokens": 250},
        )
    except Exception:
        ruling = ""
    if ruling.startswith("⚠️") or not ruling.strip():
        ruling = "Court is overworked. Case dismissed.\nDISMISSED"

    # Parse the verdict
    award_m = re.search(r"AWARD:\s*([\d,]+)", ruling, re.IGNORECASE)
    countersuit_m = re.search(r"COUNTERSUIT:\s*([\d,]+)", ruling, re.IGNORECASE)
    dismissed = bool(re.search(r"\bDISMISSED\b", ruling, re.IGNORECASE))

    # Strip the verdict line(s) from displayed ruling text
    display_ruling = re.sub(r"(AWARD:.*|DISMISSED|COUNTERSUIT:.*)", "", ruling, flags=re.IGNORECASE).strip()

    if countersuit_m:
        amount = int(countersuit_m.group(1).replace(",", ""))
        amount = max(50, min(amount, 5000))  # sanity bounds
        actual = min(amount, economy.balance(plaintiff.id))
        economy.add(plaintiff.id, -actual, "lawsuit countersuit")
        economy.add(defendant.id, actual, "lawsuit countersuit award")
        verdict_line = (
            f"# 💀 COUNTERSUIT GRANTED\n"
            f"{plaintiff.mention} pays {defendant.mention} **{actual:,}** coins for filing a frivolous lawsuit."
        )
    elif award_m:
        amount = int(award_m.group(1).replace(",", ""))
        amount = max(100, min(amount, 5000))  # sanity bounds
        actual = min(amount, economy.balance(defendant.id))
        economy.add(defendant.id, -actual, "lawsuit damages")
        economy.add(plaintiff.id, actual, "lawsuit award")
        await trigger_game_win(plaintiff.id, "lawsuit", channel=interaction.channel)
        await trigger_balance_check(plaintiff.id, channel=interaction.channel)
        verdict_line = (
            f"# 🏆 PLAINTIFF WINS\n"
            f"{defendant.mention} owes {plaintiff.mention} **{actual:,}** coins in damages."
        )
    else:
        # Dismissed (default)
        verdict_line = (
            f"# 🪦 CASE DISMISSED\n"
            f"{plaintiff.mention} loses the **{LAWSUIT_FILING_FEE:,}** coin filing fee. "
            f"{defendant.mention} walks free."
        )

    final = (
        f"⚖️ **🔨 GAVEL DROPS** 🔨\n\n"
        f"**Case:** _{claim}_\n"
        f"**Plaintiff:** {plaintiff.mention} | **Defendant:** {defendant.mention}\n\n"
        f"**JUDGE'S RULING:**\n> {display_ruling}\n\n"
        f"{verdict_line}"
    )
    if len(final) > 2000:
        final = final[:1990] + "..."
    try:
        await interaction.edit_original_response(content=final)
    except Exception:
        pass


# ── 🤥 /lieordie — AI generates fact, users vote ────────────────────────────
ACTIVE_LIEORDIE: dict[str, dict] = {}  # channel_id -> game state
LIEORDIE_VOTING_WINDOW = 30  # seconds
LIEORDIE_BET = 100           # coins per vote


class LieOrDieView(discord.ui.View):
    def __init__(self, channel_id: str, target_id: int):
        super().__init__(timeout=LIEORDIE_VOTING_WINDOW)
        self.channel_id = channel_id
        self.target_id = target_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        game = ACTIVE_LIEORDIE.get(self.channel_id)
        if not game or game.get("ended"):
            await interaction.response.send_message("Voting is closed.", ephemeral=True)
            return False
        if interaction.user.id == self.target_id:
            await interaction.response.send_message("You can't vote on a fact about yourself.", ephemeral=True)
            return False
        if interaction.user.id in game["votes"]:
            await interaction.response.send_message("You already voted.", ephemeral=True)
            return False
        if economy.balance(interaction.user.id) < LIEORDIE_BET:
            await interaction.response.send_message(
                f"❌ You need at least **{LIEORDIE_BET}** coins to vote.", ephemeral=True
            )
            return False
        return True

    async def _vote(self, interaction: discord.Interaction, vote: str):
        game = ACTIVE_LIEORDIE[self.channel_id]
        # Take the bet upfront
        economy.add(interaction.user.id, -LIEORDIE_BET, "lieordie vote")
        game["votes"][interaction.user.id] = vote
        true_count = sum(1 for v in game["votes"].values() if v == "true")
        false_count = sum(1 for v in game["votes"].values() if v == "false")
        await interaction.response.send_message(
            f"🗳️ You voted **{vote.upper()}** ({LIEORDIE_BET} coins staked).\n"
            f"Current count: ✅ TRUE: {true_count} | ❌ FALSE: {false_count}",
            ephemeral=True,
        )

    @discord.ui.button(label="TRUE", style=discord.ButtonStyle.success, emoji="✅")
    async def true_btn(self, interaction: discord.Interaction, _b: discord.ui.Button):
        await self._vote(interaction, "true")

    @discord.ui.button(label="FALSE", style=discord.ButtonStyle.danger, emoji="❌")
    async def false_btn(self, interaction: discord.Interaction, _b: discord.ui.Button):
        await self._vote(interaction, "false")


@tree.command(name="lieordie", description="AI generates a fact about a user. Vote TRUE or FALSE — winners take losers' coins.")
@discord.app_commands.describe(target="Who is the fact about?")
async def lieordie_command(interaction: discord.Interaction, target: discord.Member):
    cfg = load_config()
    if target.bot:
        await interaction.response.send_message("Can't run this on a bot.", ephemeral=True)
        return
    # Anti-abuse: can't run on self (would let users vote on themselves for guaranteed wins via alts)
    if target.id == interaction.user.id:
        await interaction.response.send_message(
            "❌ Can't run /lieordie on yourself.", ephemeral=True
        )
        return
    channel_id = str(interaction.channel_id)
    if channel_id in ACTIVE_LIEORDIE and not ACTIVE_LIEORDIE[channel_id].get("ended"):
        await interaction.response.send_message("A round is already running here.", ephemeral=True)
        return

    await interaction.response.defer()
    silent = discord.AllowedMentions.none()

    async def edit(content, view=None):
        kwargs = {"content": content, "allowed_mentions": silent}
        if view is not None:
            kwargs["view"] = view
        try:
            await interaction.edit_original_response(**kwargs)
        except Exception:
            pass

    # ── Gather the target's messages — pull a LOT to get good signal ─────────
    await edit(f"🤥 *gathering intel on {target.mention}...*")

    user_messages: list[str] = []
    try:
        # Scan up to 5000 messages of channel history to get up to 150 of theirs
        async for m in interaction.channel.history(limit=5000):
            if m.author.id == target.id and m.content and m.content.strip():
                user_messages.append(m.content.strip())
                if len(user_messages) >= 150:
                    break
    except Exception as e:
        log.warning("lieordie history fetch failed: %s", e)

    # Also pull from daily logs (last 7 days, all channels)
    today = datetime.now(timezone.utc)
    target_name = target.display_name
    seen = set(user_messages)
    for days_back in range(7):
        date_str = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        for log_file in RECAP_DIR.glob(f"*_{date_str}.jsonl"):
            try:
                with open(log_file) as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("author") == target_name:
                                content = entry.get("content", "").strip()
                                if content and content not in seen:
                                    seen.add(content)
                                    user_messages.append(content)
                        except Exception:
                            continue
            except Exception:
                continue

    if len(user_messages) < 10:
        await edit(
            f"❌ Not enough message history on {target.mention} to generate a fact. "
            f"They need at least 10 messages."
        )
        return

    # Cap to most recent 200 messages
    user_messages = user_messages[:200]
    transcript = "\n".join(f"- {m}" for m in user_messages)

    await edit(f"🤥 *found {len(user_messages)} messages... fabricating a 'fact'...*")
    await asyncio.sleep(1.0)

    # ── AI generates a fact (truth or lie) about the user ────────────────────
    is_true = random.choice([True, False])

    if is_true:
        sys = (
            cfg["system_prompt"] +
            "\n\n=== TRUTH MODE ===\n"
            f"You are generating a TRUE 'fact' about {target.display_name} based on their actual messages. "
            "Pick something specific they actually said, did, or revealed. Phrase it as a SINGLE statement "
            "starting with their name (NOT 'I' — third person). "
            "Make it sound like it COULD be a lie — interesting, weird, or surprising — but it must be "
            "directly supported by their actual messages. Cite no quotes — just state the fact. "
            "Output ONLY the fact, one sentence, max 25 words. No preamble."
        )
    else:
        sys = (
            cfg["system_prompt"] +
            "\n\n=== LIE MODE ===\n"
            f"You are generating a FAKE 'fact' about {target.display_name}. "
            "Make it sound plausible — like something they MIGHT say or do based on their vibe — "
            "but it must NOT be supported by their messages. Phrase it as a SINGLE statement "
            "starting with their name (third person). Make it interesting, weird, or surprising. "
            "Output ONLY the fact, one sentence, max 25 words. No preamble."
        )

    user_prompt = f"Recent messages from {target.display_name}:\n\n{transcript[:6000]}\n\nGenerate the fact."
    try:
        fact = await ask_ai(sys, [{"role": "user", "content": user_prompt}],
                            {**cfg, "max_tokens": 100, "temperature": 0.95})
    except Exception:
        fact = ""
    if fact.startswith("⚠️") or not fact.strip():
        await edit("⚠️ AI failed to generate a fact. Try again.")
        return
    fact = fact.strip().strip('"').strip("'")

    # ── Set up the game ──────────────────────────────────────────────────────
    game = {
        "channel_id": channel_id,
        "target_id": target.id,
        "fact": fact,
        "is_true": is_true,
        "votes": {},  # user_id -> "true"|"false"
        "ended": False,
        "expires_at": time.time() + LIEORDIE_VOTING_WINDOW,
    }
    ACTIVE_LIEORDIE[channel_id] = game

    view = LieOrDieView(channel_id, target.id)
    await edit(
        (
            f"# 🤥 LIE OR DIE 🤥\n\n"
            f"**About:** {target.mention}\n"
            f"**Buy-in:** {LIEORDIE_BET} coins per vote\n"
            f"**Voting closes in {LIEORDIE_VOTING_WINDOW}s**\n\n"
            f"## > {fact}\n\n"
            f"_Click **TRUE** if you think it's real, **FALSE** if you think it's a lie. Winners split the losers' pot._"
        ),
        view=view,
    )

    # ── Run the round ────────────────────────────────────────────────────────
    asyncio.create_task(_resolve_lieordie(interaction, channel_id))


async def _resolve_lieordie(interaction: discord.Interaction, channel_id: str):
    # Periodic count updates while voting is open
    game = ACTIVE_LIEORDIE.get(channel_id)
    if not game:
        return

    # Wait out the voting window
    await asyncio.sleep(LIEORDIE_VOTING_WINDOW)

    game = ACTIVE_LIEORDIE.get(channel_id)
    if not game or game.get("ended"):
        return
    game["ended"] = True

    votes = game["votes"]
    truth = "true" if game["is_true"] else "false"

    winners = [uid for uid, v in votes.items() if v == truth]
    losers = [uid for uid, v in votes.items() if v != truth]

    pot = len(losers) * LIEORDIE_BET
    payout_lines = []

    if winners and pot > 0:
        share = pot // len(winners)
        # Each winner: get their bet back + their share of the loser pool
        for uid in winners:
            economy.add(uid, LIEORDIE_BET + share, "lieordie win")
            economy.record_win(uid)
            payout_lines.append(f"✅ <@{uid}> won **{share + LIEORDIE_BET:,}** ({share:,} profit)")
        for uid in losers:
            economy.record_loss(uid)
            payout_lines.append(f"❌ <@{uid}> lost **{LIEORDIE_BET}**")
    elif winners and not losers:
        # Everyone right — refund only
        for uid in winners:
            economy.add(uid, LIEORDIE_BET, "lieordie refund")
            payout_lines.append(f"↩️ <@{uid}> bet refunded (no losers)")
    elif not winners and losers:
        # Everyone wrong — house keeps the coins
        for uid in losers:
            economy.record_loss(uid)
            payout_lines.append(f"❌ <@{uid}> lost **{LIEORDIE_BET}** (no one was right)")

    truth_label = "✅ **TRUE**" if game["is_true"] else "❌ **FALSE**"
    summary = (
        f"# 🤥 LIE OR DIE — RESULTS\n\n"
        f"## > {game['fact']}\n\n"
        f"### The truth: {truth_label}\n\n"
    )
    if payout_lines:
        summary += "**Payouts:**\n" + "\n".join(payout_lines)
    else:
        summary += "_No one voted._"

    if len(summary) > 2000:
        summary = summary[:1990] + "..."

    try:
        await interaction.channel.send(summary, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        pass

    ACTIVE_LIEORDIE.pop(channel_id, None)


# ── 🏆 /achievements ─────────────────────────────────────────────────────────
@tree.command(name="achievements", description="See your earned achievements + perks.")
@discord.app_commands.describe(user="Whose achievements to view (defaults to you)")
async def achievements_command(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user
    earned = _get_achievements(target.id)
    perks = economy._user(target.id).get("perks", {})

    badges = get_user_badges(target.id)
    title = f"🏆 {target.display_name}'s Achievements"
    if badges:
        title = f"{badges} {target.display_name}'s Achievements"

    embed = discord.Embed(
        title=title,
        description=f"**Earned: {len(earned)}/{len(ACHIEVEMENTS)}**",
        color=discord.Color.gold(),
    )

    earned_text = []
    locked_text = []
    for ach_id, ach in ACHIEVEMENTS.items():
        line = f"{ach['emoji']} **{ach['name']}** — _{ach['description']}_"
        if ach.get("perk"):
            line += f"\n   ↳ Perk: +{ach['perk_value']}% {_perk_label(ach['perk'])}"
        if ach_id in earned:
            earned_text.append(line)
        else:
            locked_text.append(f"🔒 **{ach['name']}** — _{ach['description']}_")

    if earned_text:
        # Discord field limit is 1024 chars, so chunk if needed
        chunk = ""
        for line in earned_text:
            if len(chunk) + len(line) + 2 > 1000:
                embed.add_field(name="✅ Earned", value=chunk, inline=False)
                chunk = ""
            chunk += line + "\n\n"
        if chunk:
            embed.add_field(name="✅ Earned", value=chunk.strip(), inline=False)
    else:
        embed.add_field(name="✅ Earned", value="_none yet — start playing!_", inline=False)

    if locked_text and len(earned) < len(ACHIEVEMENTS):
        # Show only first ~10 locked to keep it clean
        locked_show = locked_text[:10]
        more = len(locked_text) - len(locked_show)
        locked_value = "\n".join(locked_show)
        if more > 0:
            locked_value += f"\n_...and {more} more_"
        embed.add_field(name="🔒 Locked", value=locked_value[:1020], inline=False)

    if perks:
        perks_lines = "\n".join(
            f"+{v}% {_perk_label(k)}" for k, v in perks.items()
        )
        embed.add_field(name="✨ Active Perks", value=perks_lines, inline=False)

    await interaction.response.send_message(embed=embed)


# ── 📊 /level ────────────────────────────────────────────────────────────────
@tree.command(name="level", description="See your XP, level, and progress.")
@discord.app_commands.describe(user="Whose level to view (defaults to you)")
async def level_command(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user
    xp = _get_xp(target.id)
    level = level_for_xp(xp)
    title = get_user_title(target.id)
    badges = get_user_badges(target.id)

    # Progress to next level
    current_level_xp = xp_for_level(level)
    next_level_xp = xp_for_level(level + 1)
    progress = xp - current_level_xp
    needed = next_level_xp - current_level_xp
    bar_len = 20
    filled = int((progress / needed) * bar_len) if needed > 0 else bar_len
    bar = "█" * filled + "░" * (bar_len - filled)

    embed = discord.Embed(
        title=f"📊 {target.display_name}'s Level",
        color=discord.Color.green(),
    )
    if badges:
        embed.title = f"📊 {badges} {target.display_name}"
    embed.add_field(name="🏅 Title", value=f"**{title}**", inline=True)
    embed.add_field(name="📈 Level", value=f"**{level}**", inline=True)
    embed.add_field(name="✨ Total XP", value=f"{xp:,}", inline=True)
    embed.add_field(
        name=f"Progress to Level {level + 1}",
        value=f"`{bar}`\n**{progress:,} / {needed:,}** XP",
        inline=False,
    )
    await interaction.response.send_message(embed=embed)


# ── 🏆 /tournament ───────────────────────────────────────────────────────────
@tree.command(name="tournament", description="See the current weekly tournament leaderboard.")
async def tournament_command(interaction: discord.Interaction):
    data = _load_tournament()
    scores = data.get("scores", {})
    season_start = data.get("season_start", "?")

    if not scores:
        embed = discord.Embed(
            title="🏆 WEEKLY TOURNAMENT",
            description="_No participants yet this week._\n\nEarn coins, win games, or use commands to compete!",
            color=discord.Color.gold(),
        )
        embed.set_footer(text=f"Season started: {season_start} • Resets Monday")
        await interaction.response.send_message(embed=embed)
        return

    ranked = sorted(scores.items(), key=lambda x: tournament_score(x[1]), reverse=True)
    medals = ["🥇", "🥈", "🥉"]
    lines = []
    for i, (uid, stats) in enumerate(ranked[:10]):
        try:
            member = interaction.guild.get_member(int(uid)) if interaction.guild else None
            name = member.display_name if member else f"User {uid}"
        except Exception:
            name = f"User {uid}"
        prefix = medals[i] if i < 3 else f"`#{i+1}`"
        prize_text = ""
        if i < 3:
            prize_text = f" • Prize: **{TOURNAMENT_PRIZES[i]:,}**"
        score = tournament_score(stats)
        lines.append(
            f"{prefix} **{name}** — score **{score:,}**{prize_text}\n"
            f"   ↳ {stats['coins_earned']:,} earned • {stats['games_won']} wins • {stats['commands_used']} cmds"
        )

    embed = discord.Embed(
        title="🏆 WEEKLY TOURNAMENT",
        description="\n".join(lines),
        color=discord.Color.gold(),
    )
    embed.set_footer(text=f"Season started: {season_start} • Resets Monday at recap hour")
    await interaction.response.send_message(embed=embed)


# ─────────────────────────────────────────────────────────────────────────────
# 🐶 PETS / COMPANIONS
# Adopt a pet, feed it daily, it earns passive coins, levels up over time.
# ─────────────────────────────────────────────────────────────────────────────
PETS_FILE = MEMORY_DIR / "pets.json"
PET_ADOPT_COST = 1500
PET_FEED_COST = 50
PET_DAILY_INCOME_BASE = 30  # coins per level per day
PET_HUNGER_DECAY_HOURS = 6  # how often hunger ticks down without food

PET_TYPES = {
    "dog":      {"emoji": "🐶", "name": "Dog",      "desc": "Loyal coin earner."},
    "cat":      {"emoji": "🐱", "name": "Cat",      "desc": "Aloof but profitable."},
    "fox":      {"emoji": "🦊", "name": "Fox",      "desc": "Sneaky and sly."},
    "dragon":   {"emoji": "🐲", "name": "Dragon",   "desc": "Hoards treasure."},
    "monkey":   {"emoji": "🐒", "name": "Monkey",   "desc": "Chaotic neutral."},
    "shark":    {"emoji": "🦈", "name": "Shark",    "desc": "Eats your enemies."},
    "raccoon":  {"emoji": "🦝", "name": "Raccoon",  "desc": "Born thief."},
    "owl":      {"emoji": "🦉", "name": "Owl",      "desc": "Wise and calculating."},
}


def _load_pets() -> dict:
    if PETS_FILE.exists():
        try:
            with open(PETS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_pets(data: dict):
    with open(PETS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _pet_xp_for_level(level: int) -> int:
    return 50 * level * level


def _pet_level(xp: int) -> int:
    if xp < 50:
        return 1
    import math
    return max(1, int(math.sqrt(xp / 50)))


def _pet_hunger(pet: dict) -> int:
    """Calculate current hunger (0-100). Decays over time."""
    last_fed = pet.get("last_fed", time.time())
    hours_since = (time.time() - last_fed) / 3600
    decay = int(hours_since / PET_HUNGER_DECAY_HOURS * 25)
    return max(0, 100 - decay)


@tree.command(name="adopt", description="Adopt a pet companion. Pets earn passive coins and level up.")
@discord.app_commands.describe(pet_type="What kind of pet?", name="What to call your pet")
@discord.app_commands.choices(
    pet_type=[
        discord.app_commands.Choice(name=f"{p['emoji']} {p['name']} — {p['desc']}", value=k)
        for k, p in PET_TYPES.items()
    ],
)
async def adopt_command(
    interaction: discord.Interaction,
    pet_type: discord.app_commands.Choice[str],
    name: str,
):
    user = interaction.user
    pets = _load_pets()
    if str(user.id) in pets:
        await interaction.response.send_message(
            f"❌ You already have a pet. Abandon it first with `/abandon`.", ephemeral=True
        )
        return
    if economy.balance(user.id) < PET_ADOPT_COST:
        await interaction.response.send_message(
            f"❌ Adoption fee is **{PET_ADOPT_COST:,}** coins.", ephemeral=True
        )
        return

    name = name.strip()[:30]
    if not name:
        await interaction.response.send_message("❌ Give your pet a name.", ephemeral=True)
        return

    economy.add(user.id, -PET_ADOPT_COST, "pet adoption")
    pet_info = PET_TYPES[pet_type.value]
    pets[str(user.id)] = {
        "type": pet_type.value,
        "name": name,
        "xp": 0,
        "last_fed": time.time(),
        "last_collected": time.time(),
        "adopted_at": time.time(),
    }
    _save_pets(pets)

    await interaction.response.send_message(
        f"# {pet_info['emoji']} You adopted **{name}** the {pet_info['name']}!\n\n"
        f"💰 Earns **{PET_DAILY_INCOME_BASE}** coins per level per day\n"
        f"🍖 Feed daily with `/feed` (or your pet runs away after 4+ days hungry)\n"
        f"📊 Check on them with `/pet`\n"
        f"💎 Use `/collect` to claim earned coins"
    )


@tree.command(name="pet", description="Check on your pet's stats.")
@discord.app_commands.describe(user="Whose pet to view (defaults to you)")
async def pet_command(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user
    pets = _load_pets()
    pet = pets.get(str(target.id))
    if not pet:
        await interaction.response.send_message(
            f"{target.mention} doesn't have a pet. Adopt one with `/adopt`!",
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    info = PET_TYPES.get(pet["type"], {"emoji": "🐾", "name": "Pet"})
    level = _pet_level(pet["xp"])
    next_level_xp = _pet_xp_for_level(level + 1)
    progress = pet["xp"] - _pet_xp_for_level(level)
    needed = next_level_xp - _pet_xp_for_level(level)
    hunger = _pet_hunger(pet)

    bar_len = 15
    filled = int((progress / needed) * bar_len) if needed > 0 else bar_len
    xp_bar = "█" * filled + "░" * (bar_len - filled)

    hunger_emoji = "🍖" if hunger >= 70 else ("😐" if hunger >= 40 else ("😟" if hunger >= 15 else "💀"))
    hunger_filled = int(hunger / 100 * bar_len)
    hunger_bar = "█" * hunger_filled + "░" * (bar_len - hunger_filled)

    # Calculate pending earnings
    hours_since_collect = (time.time() - pet.get("last_collected", time.time())) / 3600
    pending = int(level * PET_DAILY_INCOME_BASE * (hours_since_collect / 24))
    if hunger < 30:
        pending = pending // 2  # half earnings if hungry

    embed = discord.Embed(
        title=f"{info['emoji']} {pet['name']}",
        description=f"_{info['name']}_ — Level **{level}**",
        color=discord.Color.teal(),
    )
    embed.add_field(name=f"📊 XP ({pet['xp']:,})", value=f"`{xp_bar}` {progress}/{needed}", inline=False)
    embed.add_field(name=f"{hunger_emoji} Hunger", value=f"`{hunger_bar}` {hunger}/100", inline=False)
    embed.add_field(name="💰 Pending Earnings", value=f"**{pending:,}** coins", inline=True)
    embed.add_field(name="📈 Income/day", value=f"{level * PET_DAILY_INCOME_BASE:,}", inline=True)
    embed.set_footer(text=f"Owner: {target.display_name}")
    await interaction.response.send_message(embed=embed)


@tree.command(name="feed", description="Feed your pet to keep it happy.")
async def feed_command(interaction: discord.Interaction):
    user = interaction.user
    pets = _load_pets()
    pet = pets.get(str(user.id))
    if not pet:
        await interaction.response.send_message("You don't have a pet.", ephemeral=True)
        return
    if economy.balance(user.id) < PET_FEED_COST:
        await interaction.response.send_message(
            f"❌ Feeding costs **{PET_FEED_COST}** coins.", ephemeral=True
        )
        return
    if _pet_hunger(pet) >= 95:
        await interaction.response.send_message(
            f"🍖 {pet['name']} is already full!", ephemeral=True
        )
        return

    economy.add(user.id, -PET_FEED_COST, "pet feed")
    pet["last_fed"] = time.time()
    pet["xp"] = pet.get("xp", 0) + 10  # Feeding gives small XP
    _save_pets(pets)

    info = PET_TYPES.get(pet["type"], {"emoji": "🐾"})
    new_level = _pet_level(pet["xp"])
    await interaction.response.send_message(
        f"{info['emoji']} You fed **{pet['name']}**! +10 XP\n"
        f"Level: **{new_level}** | Hunger: **{_pet_hunger(pet)}/100**"
    )


@tree.command(name="collect", description="Collect coins your pet has earned.")
async def collect_command(interaction: discord.Interaction):
    user = interaction.user
    pets = _load_pets()
    pet = pets.get(str(user.id))
    if not pet:
        await interaction.response.send_message("You don't have a pet.", ephemeral=True)
        return

    level = _pet_level(pet["xp"])
    hunger = _pet_hunger(pet)
    hours_since = (time.time() - pet.get("last_collected", time.time())) / 3600
    if hours_since < 1:
        await interaction.response.send_message(
            f"⏰ Wait at least an hour between collections.", ephemeral=True
        )
        return

    earnings = int(level * PET_DAILY_INCOME_BASE * (hours_since / 24))
    if hunger < 30:
        earnings = earnings // 2  # hungry pet earns half
    if hunger == 0:
        earnings = 0

    if earnings <= 0:
        await interaction.response.send_message(
            f"💀 {pet['name']} is too hungry to work. Feed them first!", ephemeral=True
        )
        return

    pet["last_collected"] = time.time()
    pet["xp"] = pet.get("xp", 0) + earnings // 10  # Working gives XP
    _save_pets(pets)
    new_bal = economy.add(user.id, earnings, "pet collect")
    track_economy_event("earned", earnings)
    track_feature_use("pet")
    track_activity("pet_collect", user.id, user.display_name, f"collected {earnings:,} from pet")

    info = PET_TYPES.get(pet["type"], {"emoji": "🐾"})
    note = " _(earnings halved — pet is hungry)_" if hunger < 30 else ""
    await interaction.response.send_message(
        f"{info['emoji']} **{pet['name']}** brought you **{earnings:,}** coins!{note}\n"
        f"Balance: **{new_bal:,}**"
    )


@tree.command(name="abandon", description="Give up your pet (no refund).")
async def abandon_command(interaction: discord.Interaction):
    user = interaction.user
    pets = _load_pets()
    if str(user.id) not in pets:
        await interaction.response.send_message("You don't have a pet.", ephemeral=True)
        return
    pet = pets[str(user.id)]
    info = PET_TYPES.get(pet["type"], {"emoji": "🐾"})
    name = pet["name"]
    del pets[str(user.id)]
    _save_pets(pets)
    await interaction.response.send_message(
        f"💔 You abandoned {info['emoji']} **{name}**. They look back at you sadly..."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 👍 REPUTATION SYSTEM
# Users +rep each other for funny/cool posts. Top rep unlocks perks.
# 24h cooldown between giving rep.
# ─────────────────────────────────────────────────────────────────────────────
REP_COOLDOWN_HOURS = 24


@tree.command(name="rep", description="Give another user reputation. 24h cooldown.")
@discord.app_commands.describe(user="Who deserves rep?", reason="Optional reason")
async def rep_command(interaction: discord.Interaction, user: discord.Member, reason: str = None):
    giver = interaction.user
    if user.id == giver.id:
        await interaction.response.send_message("Can't rep yourself.", ephemeral=True)
        return
    if user.bot:
        await interaction.response.send_message("Bots don't need rep.", ephemeral=True)
        return

    counters = _get_counters(giver.id)
    last_rep = counters.get("last_rep_given", 0)
    cd_seconds = REP_COOLDOWN_HOURS * 3600
    if time.time() - last_rep < cd_seconds:
        remaining = int(cd_seconds - (time.time() - last_rep))
        await interaction.response.send_message(
            f"⏰ You can rep again in **{fmt_cooldown(remaining)}**.", ephemeral=True
        )
        return

    # Increment target's rep
    target_data = economy._user(user.id)
    target_data["rep"] = target_data.get("rep", 0) + 1
    economy._save()
    counters["last_rep_given"] = time.time()
    economy._save()

    new_rep = target_data["rep"]
    rep_perks = {
        10:  ("daily_bonus_pct", 5,  "Liked"),
        25:  ("work_bonus_pct", 5,   "Respected"),
        50:  ("rob_protection_pct", 10, "Beloved"),
        100: ("passive_income", 50,  "Server Hero"),
    }
    perk_announcement = ""
    if new_rep in rep_perks:
        pkey, pval, title = rep_perks[new_rep]
        perks = target_data.setdefault("perks", {})
        perks[pkey] = perks.get(pkey, 0) + pval
        economy._save()
        perk_announcement = f"\n🏅 **{title}** title unlocked! +{pval}% {_perk_label(pkey)}"

    msg = (
        f"👍 {giver.mention} gave +1 rep to {user.mention}!\n"
        f"📊 {user.display_name}'s rep: **{new_rep}**"
    )
    if reason:
        msg += f"\n💬 _\"{reason[:100]}\"_"
    if perk_announcement:
        msg += perk_announcement

    await interaction.response.send_message(
        msg,
        allowed_mentions=discord.AllowedMentions(users=[user]),
    )
    # DM the recipient
    try:
        dm_msg = f"👍 **{giver.display_name}** gave you +1 rep!\n📊 Your total rep: **{new_rep}**"
        if reason:
            dm_msg += f"\n💬 _\"{reason[:200]}\"_"
        await send_dm(user.id, "rep", content=dm_msg)
    except Exception:
        pass


@tree.command(name="reputation", description="Check a user's reputation.")
@discord.app_commands.describe(user="Whose rep to check (defaults to you)")
async def reputation_command(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user
    rep = economy._user(target.id).get("rep", 0)
    counters = _get_counters(target.id)
    last_given = counters.get("last_rep_given", 0)
    can_give = time.time() - last_given >= REP_COOLDOWN_HOURS * 3600

    embed = discord.Embed(
        title=f"👍 {target.display_name}'s Reputation",
        color=discord.Color.green(),
    )
    embed.add_field(name="Rep Score", value=f"**{rep}**", inline=False)
    embed.add_field(
        name="Can give rep?",
        value="✅ Ready" if can_give else f"⏰ {fmt_cooldown(int(REP_COOLDOWN_HOURS * 3600 - (time.time() - last_given)))}",
        inline=False,
    )
    await interaction.response.send_message(embed=embed)


# ─────────────────────────────────────────────────────────────────────────────
# 🎉 REACTION REWARDS
# Earn small coin rewards when your messages get reactions.
# Anti-spam: only first 3 reactions per message count, capped per day.
# ─────────────────────────────────────────────────────────────────────────────
REACTION_REWARD = 5         # coins per reaction
REACTION_DAILY_CAP = 200    # max coins from reactions per user per day
REACTIONS_PER_MESSAGE_CAP = 3  # max reactions counted per single message


@client.event
async def on_reaction_add(reaction: discord.Reaction, user):
    # Ignore bots and self-reactions
    if user.bot:
        return
    msg = reaction.message
    if msg.author.bot or msg.author.id == user.id:
        return

    counters = _get_counters(msg.author.id)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_key = f"reactions_today_{today}"
    msg_key = f"reaction_msg_{msg.id}"

    earned_today = counters.get(daily_key, 0)
    if earned_today >= REACTION_DAILY_CAP:
        return

    msg_count = counters.get(msg_key, 0)
    if msg_count >= REACTIONS_PER_MESSAGE_CAP:
        return

    counters[msg_key] = msg_count + 1
    counters[daily_key] = earned_today + REACTION_REWARD
    economy._save()
    economy.add(msg.author.id, REACTION_REWARD, "reaction reward")


# ─────────────────────────────────────────────────────────────────────────────
# 📋 QUESTS / MISSIONS
# Daily and weekly tasks for big bonuses. Resets at recap hour.
# ─────────────────────────────────────────────────────────────────────────────
QUESTS_FILE = MEMORY_DIR / "quests.json"

DAILY_QUEST_POOL = [
    {"id": "play_3_games",     "name": "Play 3 games",                   "target": 3,    "reward": 500,  "track": "games_played"},
    {"id": "win_2_games",      "name": "Win 2 games",                    "target": 2,    "reward": 750,  "track": "games_won"},
    {"id": "earn_1k",          "name": "Earn 1,000 coins",               "target": 1000, "reward": 500,  "track": "coins_earned"},
    {"id": "use_5_cmds",       "name": "Use 5 commands",                 "target": 5,    "reward": 300,  "track": "commands_used"},
    {"id": "give_500",         "name": "Give 500 coins to another user", "target": 500,  "reward": 600,  "track": "coins_given"},
    {"id": "rob_someone",      "name": "Rob another user",               "target": 1,    "reward": 400,  "track": "rob_attempts"},
]

WEEKLY_QUEST_POOL = [
    {"id": "win_10_games",     "name": "Win 10 games this week",         "target": 10,   "reward": 3_000, "track": "games_won"},
    {"id": "earn_10k",         "name": "Earn 10,000 coins this week",    "target": 10_000,"reward": 2_500, "track": "coins_earned"},
    {"id": "play_25_games",    "name": "Play 25 games this week",        "target": 25,   "reward": 2_000, "track": "games_played"},
    {"id": "use_50_cmds",      "name": "Use 50 commands this week",      "target": 50,   "reward": 1_500, "track": "commands_used"},
]


def _load_quests() -> dict:
    if QUESTS_FILE.exists():
        try:
            with open(QUESTS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_quests(data: dict):
    with open(QUESTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _get_user_quests(user_id: int) -> dict:
    """Get quests, generating new ones if needed."""
    quests = _load_quests()
    uid = str(user_id)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    week = datetime.now(timezone.utc).strftime("%Y-W%U")

    if uid not in quests:
        quests[uid] = {}

    user_quests = quests[uid]

    # Daily quests
    if user_quests.get("daily_date") != today:
        user_quests["daily_date"] = today
        chosen = random.sample(DAILY_QUEST_POOL, min(3, len(DAILY_QUEST_POOL)))
        user_quests["daily"] = [
            {**q, "progress": 0, "completed": False, "claimed": False}
            for q in chosen
        ]

    # Weekly quests
    if user_quests.get("weekly_date") != week:
        user_quests["weekly_date"] = week
        chosen = random.sample(WEEKLY_QUEST_POOL, min(2, len(WEEKLY_QUEST_POOL)))
        user_quests["weekly"] = [
            {**q, "progress": 0, "completed": False, "claimed": False}
            for q in chosen
        ]

    quests[uid] = user_quests
    _save_quests(quests)
    return user_quests


def track_quest_progress(user_id: int, track_key: str, amount: int = 1):
    """Increment progress on any quest tracking this key."""
    user_quests = _get_user_quests(user_id)
    quests = _load_quests()
    uid = str(user_id)
    changed = False
    for bucket in ("daily", "weekly"):
        for q in user_quests.get(bucket, []):
            if q["track"] == track_key and not q["completed"]:
                q["progress"] += amount
                if q["progress"] >= q["target"]:
                    q["completed"] = True
                changed = True
    if changed:
        quests[uid] = user_quests
        _save_quests(quests)


@tree.command(name="quests", description="See your daily and weekly quests.")
async def quests_command(interaction: discord.Interaction):
    user = interaction.user
    user_quests = _get_user_quests(user.id)

    embed = discord.Embed(
        title="📋 QUESTS",
        description="_Daily quests reset every 24h. Weekly resets every Monday._",
        color=discord.Color.purple(),
    )

    def fmt_quest(q):
        if q["claimed"]:
            return f"✅ ~~**{q['name']}**~~ — _claimed_"
        elif q["completed"]:
            return f"🎁 **{q['name']}** — Ready to claim! ({q['progress']}/{q['target']}) — **+{q['reward']:,}**"
        else:
            return f"⏳ {q['name']} — {q['progress']}/{q['target']} — _+{q['reward']:,}_"

    daily_text = "\n".join(fmt_quest(q) for q in user_quests.get("daily", []))
    weekly_text = "\n".join(fmt_quest(q) for q in user_quests.get("weekly", []))
    embed.add_field(name="📅 DAILY", value=daily_text or "_No quests_", inline=False)
    embed.add_field(name="📆 WEEKLY", value=weekly_text or "_No quests_", inline=False)
    embed.set_footer(text="Use /claimquest to claim completed quests.")
    await interaction.response.send_message(embed=embed)


@tree.command(name="claimquest", description="Claim rewards from completed quests.")
async def claimquest_command(interaction: discord.Interaction):
    user = interaction.user
    user_quests = _get_user_quests(user.id)
    quests = _load_quests()
    uid = str(user.id)

    claimed = []
    total_reward = 0
    for bucket in ("daily", "weekly"):
        for q in user_quests.get(bucket, []):
            if q["completed"] and not q["claimed"]:
                q["claimed"] = True
                economy.add(user.id, q["reward"], f"quest: {q['id']}")
                claimed.append(f"✅ **{q['name']}** (+{q['reward']:,})")
                total_reward += q["reward"]

    quests[uid] = user_quests
    _save_quests(quests)

    if not claimed:
        await interaction.response.send_message(
            "No completed quests to claim. Run `/quests` to see progress.", ephemeral=True
        )
        return

    new_bal = economy.balance(user.id)
    await interaction.response.send_message(
        f"🎁 **QUESTS CLAIMED!**\n\n" + "\n".join(claimed) +
        f"\n\nTotal: **{total_reward:,}** coins\nBalance: **{new_bal:,}**"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 🏢 BUSINESSES SYSTEM
# Buy multiple businesses, hire employees, collect passive income,
# face risks, sabotage rivals.
# ─────────────────────────────────────────────────────────────────────────────
BUSINESSES_FILE = MEMORY_DIR / "businesses.json"

# Business types: cost, base hourly income, max employees, tier (cost scales with quantity)
BUSINESS_TYPES = {
    "lemonade": {
        "emoji": "🍋", "name": "Lemonade Stand",
        "cost": 2_000, "income_per_hour": 30, "max_employees": 1, "tier": 1,
        "desc": "A humble beginning. Quick to start.",
    },
    "foodtruck": {
        "emoji": "🚚", "name": "Food Truck",
        "cost": 10_000, "income_per_hour": 120, "max_employees": 2, "tier": 2,
        "desc": "Mobile and profitable.",
    },
    "barbershop": {
        "emoji": "💈", "name": "Barbershop",
        "cost": 25_000, "income_per_hour": 250, "max_employees": 3, "tier": 2,
        "desc": "Always in demand.",
    },
    "cafe": {
        "emoji": "☕", "name": "Coffee Shop",
        "cost": 50_000, "income_per_hour": 450, "max_employees": 4, "tier": 3,
        "desc": "Caffeine addicts pay rent.",
    },
    "gym": {
        "emoji": "💪", "name": "Gym",
        "cost": 80_000, "income_per_hour": 650, "max_employees": 5, "tier": 3,
        "desc": "People pay to suffer.",
    },
    "nightclub": {
        "emoji": "🎵", "name": "Nightclub",
        "cost": 150_000, "income_per_hour": 1_200, "max_employees": 6, "tier": 4,
        "desc": "Bottles and bouncers.",
    },
    "casino": {
        "emoji": "🎰", "name": "Casino",
        "cost": 300_000, "income_per_hour": 2_500, "max_employees": 8, "tier": 5,
        "desc": "The house always wins.",
    },
    "techstartup": {
        "emoji": "💻", "name": "Tech Startup",
        "cost": 500_000, "income_per_hour": 4_000, "max_employees": 10, "tier": 5,
        "desc": "Burn cash. Promise the future.",
    },
}

# Tier 1: cost stays. Higher tiers: cost increases 50% per additional business owned.
BUSINESS_QUANTITY_MULTIPLIER = 1.5
# Employee gives 15% income boost per employee (capped by max_employees)
EMPLOYEE_INCOME_BOOST = 0.15
# Employee earns 30% of business income share (per employee)
EMPLOYEE_PAY_SHARE = 0.30
# Maximum hours of unclaimed income (caps idle accumulation)
BUSINESS_MAX_IDLE_HOURS = 24
# Sabotage costs and effects
SABOTAGE_COST = 1_500
SABOTAGE_COOLDOWN_HOURS = 6
SABOTAGE_DAMAGE_HOURS = 6  # business produces 0 for this long after sabotage
SABOTAGE_FAIL_FINE = 800   # pay this if caught sabotaging
SABOTAGE_SUCCESS_CHANCE = 0.55


def _load_businesses() -> dict:
    if BUSINESSES_FILE.exists():
        try:
            with open(BUSINESSES_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"users": {}}


def _save_businesses(data: dict):
    with open(BUSINESSES_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _user_businesses(user_id: int) -> list:
    """Return list of this user's businesses."""
    data = _load_businesses()
    return data["users"].get(str(user_id), [])


def _business_cost(user_id: int, biz_type: str) -> int:
    """Quantity-scaled cost for buying ANOTHER business of this tier+."""
    info = BUSINESS_TYPES[biz_type]
    base = info["cost"]
    same_tier_owned = sum(
        1 for b in _user_businesses(user_id)
        if BUSINESS_TYPES.get(b["type"], {}).get("tier") == info["tier"]
    )
    return int(base * (BUSINESS_QUANTITY_MULTIPLIER ** same_tier_owned))


def _business_income_per_hour(biz: dict) -> int:
    """Compute current per-hour income for a single business (counting employees + upgrades)."""
    info = BUSINESS_TYPES.get(biz["type"])
    if not info:
        return 0
    base = info["income_per_hour"]
    employees = len(biz.get("employees", []))
    upgrade_level = biz.get("upgrade_level", 0)
    boost = 1 + employees * EMPLOYEE_INCOME_BOOST + upgrade_level * BUSINESS_UPGRADE_BOOST
    return int(base * boost)


def _business_pending_income(biz: dict) -> int:
    """How much this business has produced since last collection (capped)."""
    # Sabotage: if damaged_until > now, no income during that window
    now = time.time()
    last_collected = biz.get("last_collected", biz.get("purchased_at", now))
    damaged_until = biz.get("damaged_until", 0)

    # Effective start of earning is max(last_collected, damaged_until_end if it applies during the period)
    effective_start = max(last_collected, damaged_until) if damaged_until > last_collected else last_collected
    hours = (now - effective_start) / 3600
    if hours <= 0:
        return 0
    hours = min(hours, BUSINESS_MAX_IDLE_HOURS)
    return int(_business_income_per_hour(biz) * hours)


# ── /buybusiness ─────────────────────────────────────────────────────────────
@tree.command(name="buybusiness", description="Buy a business. Earn passive income.")
@discord.app_commands.describe(business_type="Which business to buy")
@discord.app_commands.choices(
    business_type=[
        discord.app_commands.Choice(
            name=f"{b['emoji']} {b['name']} — {b['cost']:,} coins ({b['income_per_hour']}/hr)",
            value=k,
        )
        for k, b in BUSINESS_TYPES.items()
    ]
)
async def buybusiness_command(
    interaction: discord.Interaction,
    business_type: discord.app_commands.Choice[str],
):
    user = interaction.user
    biz_type = business_type.value
    if biz_type not in BUSINESS_TYPES:
        await interaction.response.send_message("Invalid business type.", ephemeral=True)
        return

    cost = _business_cost(user.id, biz_type)
    bal = economy.balance(user.id)
    if bal < cost:
        await interaction.response.send_message(
            f"❌ Need **{cost:,}** coins (price scales with quantity owned). You have **{bal:,}**.",
            ephemeral=True,
        )
        return

    info = BUSINESS_TYPES[biz_type]
    economy.add(user.id, -cost, f"bought {biz_type}")

    data = _load_businesses()
    user_bizs = data["users"].setdefault(str(user.id), [])
    new_biz = {
        "id": f"{biz_type}_{int(time.time())}_{random.randint(1000,9999)}",
        "type": biz_type,
        "purchased_at": time.time(),
        "last_collected": time.time(),
        "employees": [],         # list of user_ids
        "damaged_until": 0,      # sabotage damage timer
        "lifetime_earned": 0,
    }
    user_bizs.append(new_biz)
    _save_businesses(data)

    await interaction.response.send_message(
        f"# {info['emoji']} BUSINESS ACQUIRED\n\n"
        f"You bought a **{info['name']}** for **{cost:,}** coins.\n"
        f"💰 Income: **{info['income_per_hour']:,}** coins/hour\n"
        f"👥 Max employees: **{info['max_employees']}**\n"
        f"📋 _{info['desc']}_\n\n"
        f"Use `/businesses` to manage, `/collectbusiness` to claim earnings."
    )


# ── /businesses ──────────────────────────────────────────────────────────────
@tree.command(name="businesses", description="View your business portfolio.")
@discord.app_commands.describe(user="Whose businesses to view (defaults to you)")
async def businesses_command(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user
    user_bizs = _user_businesses(target.id)

    if not user_bizs:
        await interaction.response.send_message(
            f"{target.mention} doesn't own any businesses. Buy one with `/buybusiness`.",
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    embed = discord.Embed(
        title=f"🏢 {target.display_name}'s Portfolio",
        color=discord.Color.dark_green(),
    )

    total_hourly = 0
    total_pending = 0
    total_lifetime = 0
    for biz in user_bizs:
        info = BUSINESS_TYPES.get(biz["type"], {"emoji":"🏢","name":"Unknown","max_employees":0})
        hourly = _business_income_per_hour(biz)
        pending = _business_pending_income(biz)
        employees = biz.get("employees", [])
        damaged = biz.get("damaged_until", 0) > time.time()
        status = "🚨 SABOTAGED" if damaged else "✅ Running"
        emp_str = f"{len(employees)}/{info.get('max_employees',0)}"

        embed.add_field(
            name=f"{info['emoji']} {info['name']}",
            value=(
                f"Status: {status}\n"
                f"Income: **{hourly:,}**/hr\n"
                f"Employees: {emp_str}\n"
                f"Pending: **{pending:,}** coins\n"
                f"Lifetime: {biz.get('lifetime_earned', 0):,}"
            ),
            inline=True,
        )
        total_hourly += hourly
        total_pending += pending
        total_lifetime += biz.get("lifetime_earned", 0)

    embed.add_field(
        name="📊 TOTALS",
        value=(
            f"Hourly: **{total_hourly:,}**\n"
            f"Pending: **{total_pending:,}**\n"
            f"Lifetime: **{total_lifetime:,}**"
        ),
        inline=False,
    )
    embed.set_footer(text="Collect with /collectbusiness")
    await interaction.response.send_message(embed=embed)


# ── /collectbusiness ─────────────────────────────────────────────────────────
@tree.command(name="collectbusiness", description="Collect earnings from all your businesses.")
async def collectbusiness_command(interaction: discord.Interaction):
    user = interaction.user
    data = _load_businesses()
    user_bizs = data["users"].get(str(user.id), [])
    if not user_bizs:
        await interaction.response.send_message("You don't own any businesses.", ephemeral=True)
        return

    total_collected = 0
    employee_payouts: dict[int, int] = {}  # user_id -> total paid
    lines = []

    for biz in user_bizs:
        info = BUSINESS_TYPES.get(biz["type"], {"emoji":"🏢","name":"Unknown"})
        pending = _business_pending_income(biz)
        if pending <= 0:
            continue

        # Pay employees first
        employees = biz.get("employees", [])
        employee_pool = int(pending * EMPLOYEE_PAY_SHARE * (len(employees) / max(1, BUSINESS_TYPES[biz['type']]['max_employees'])))
        per_employee = employee_pool // max(1, len(employees)) if employees else 0
        owner_take = pending - (per_employee * len(employees))

        for emp_id in employees:
            economy.add(emp_id, per_employee, f"wages: {biz['type']}")
            employee_payouts[emp_id] = employee_payouts.get(emp_id, 0) + per_employee

        biz["last_collected"] = time.time()
        biz["lifetime_earned"] = biz.get("lifetime_earned", 0) + pending
        total_collected += owner_take

        lines.append(f"{info['emoji']} **{info['name']}** — owner: {owner_take:,} | employees: {per_employee * len(employees):,}")

    _save_businesses(data)

    if total_collected <= 0 and not employee_payouts:
        await interaction.response.send_message(
            "💤 No earnings to collect yet. Be patient.", ephemeral=True
        )
        return

    new_bal = economy.add(user.id, total_collected, "business collect")
    track_economy_event("earned", total_collected)
    track_feature_use("business")
    track_activity("business_collect", user.id, user.display_name, f"collected {total_collected:,} from businesses")

    # Track achievements/tournament
    track_quest_progress(user.id, "coins_earned", total_collected)
    add_tournament_score(user.id, coins_earned=total_collected)
    await trigger_balance_check(user.id, channel=interaction.channel)

    response = (
        f"# 💰 PAYDAY!\n\n"
        + "\n".join(lines)
        + f"\n\n**Owner take:** {total_collected:,} coins"
    )
    if employee_payouts:
        emp_lines = "\n".join(f"• <@{uid}>: {amt:,}" for uid, amt in employee_payouts.items())
        response += f"\n\n**👥 Employees paid:**\n{emp_lines}"
    response += f"\n\nBalance: **{new_bal:,}**"

    if len(response) > 2000:
        response = response[:1990] + "..."

    await interaction.response.send_message(
        response,
        allowed_mentions=discord.AllowedMentions.none(),
    )


# ── /hire ────────────────────────────────────────────────────────────────────
@tree.command(name="hire", description="Hire another user to work at one of your businesses.")
@discord.app_commands.describe(employee="Who to hire", business_type="Which business")
@discord.app_commands.choices(
    business_type=[
        discord.app_commands.Choice(name=f"{b['emoji']} {b['name']}", value=k)
        for k, b in BUSINESS_TYPES.items()
    ]
)
async def hire_command(
    interaction: discord.Interaction,
    employee: discord.Member,
    business_type: discord.app_commands.Choice[str],
):
    owner = interaction.user
    if employee.id == owner.id:
        await interaction.response.send_message("Can't hire yourself.", ephemeral=True)
        return
    if employee.bot:
        await interaction.response.send_message("Can't hire bots.", ephemeral=True)
        return

    data = _load_businesses()
    user_bizs = data["users"].get(str(owner.id), [])
    biz_type = business_type.value

    # Find first business of this type with room
    matching = [b for b in user_bizs if b["type"] == biz_type]
    if not matching:
        await interaction.response.send_message(
            f"You don't own a {BUSINESS_TYPES[biz_type]['name']}.", ephemeral=True
        )
        return

    info = BUSINESS_TYPES[biz_type]
    biz = None
    for b in matching:
        if len(b.get("employees", [])) < info["max_employees"] and employee.id not in b.get("employees", []):
            biz = b
            break

    if biz is None:
        await interaction.response.send_message(
            f"All your {info['name']}s are fully staffed or already employ {employee.mention}.",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    biz.setdefault("employees", []).append(employee.id)
    _save_businesses(data)

    await interaction.response.send_message(
        f"👥 {employee.mention} was hired at {owner.mention}'s {info['emoji']} **{info['name']}**!\n"
        f"💼 Employees boost income by **15% each**.\n"
        f"💰 They'll earn a share when {owner.display_name} runs `/collectbusiness`.",
        allowed_mentions=discord.AllowedMentions(users=[employee]),
    )


# ── /fire ────────────────────────────────────────────────────────────────────
@tree.command(name="fire", description="Fire an employee from one of your businesses.")
@discord.app_commands.describe(employee="Who to fire")
async def fire_command(interaction: discord.Interaction, employee: discord.Member):
    owner = interaction.user
    data = _load_businesses()
    user_bizs = data["users"].get(str(owner.id), [])

    fired_from = []
    for biz in user_bizs:
        if employee.id in biz.get("employees", []):
            biz["employees"].remove(employee.id)
            fired_from.append(BUSINESS_TYPES.get(biz["type"], {}).get("name", "?"))

    if not fired_from:
        await interaction.response.send_message(
            f"{employee.mention} doesn't work for you.",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    _save_businesses(data)
    await interaction.response.send_message(
        f"🚪 {employee.mention} was fired from {', '.join(fired_from)}.",
        allowed_mentions=discord.AllowedMentions(users=[employee]),
    )


# ── /sabotage ────────────────────────────────────────────────────────────────
@tree.command(name="sabotage", description="Sabotage another user's business. Risky.")
@discord.app_commands.describe(target="Whose business to sabotage", business_type="Which business of theirs")
@discord.app_commands.choices(
    business_type=[
        discord.app_commands.Choice(name=f"{b['emoji']} {b['name']}", value=k)
        for k, b in BUSINESS_TYPES.items()
    ]
)
async def sabotage_command(
    interaction: discord.Interaction,
    target: discord.Member,
    business_type: discord.app_commands.Choice[str],
):
    user = interaction.user
    if target.id == user.id:
        await interaction.response.send_message("Can't sabotage yourself.", ephemeral=True)
        return
    if target.bot:
        await interaction.response.send_message("Bots don't have businesses.", ephemeral=True)
        return

    # Cooldown
    counters = _get_counters(user.id)
    last_sabotage = counters.get("last_sabotage", 0)
    cd_seconds = SABOTAGE_COOLDOWN_HOURS * 3600
    if time.time() - last_sabotage < cd_seconds:
        remaining = int(cd_seconds - (time.time() - last_sabotage))
        await interaction.response.send_message(
            f"⏰ You're laying low. Try again in **{fmt_cooldown(remaining)}**.",
            ephemeral=True,
        )
        return

    if economy.balance(user.id) < SABOTAGE_COST:
        await interaction.response.send_message(
            f"❌ Sabotage costs **{SABOTAGE_COST:,}** coins.", ephemeral=True
        )
        return

    # Find target business
    data = _load_businesses()
    target_bizs = data["users"].get(str(target.id), [])
    biz_type = business_type.value
    matching = [b for b in target_bizs if b["type"] == biz_type and b.get("damaged_until", 0) <= time.time()]
    if not matching:
        info = BUSINESS_TYPES[biz_type]
        await interaction.response.send_message(
            f"❌ {target.mention} doesn't own a healthy {info['name']}.",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    # Insurance protection blocks sabotage entirely
    if has_insurance(target.id):
        # Charge user, give them a useless attempt
        economy.add(user.id, -SABOTAGE_COST, "sabotage attempt (insured)")
        counters = _get_counters(user.id)
        counters["last_sabotage"] = time.time()
        economy._save()
        await interaction.response.send_message(
            f"🛡️ {target.mention}'s business is **INSURED**. Your sabotage attempt failed silently.\n"
            f"You lost **{SABOTAGE_COST:,}** coins on the attempt.",
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    biz = matching[0]
    info = BUSINESS_TYPES[biz_type]

    economy.add(user.id, -SABOTAGE_COST, "sabotage attempt")
    counters["last_sabotage"] = time.time()
    economy._save()

    await interaction.response.defer()

    silent = discord.AllowedMentions.none()
    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    await edit(f"🦝 *{user.mention} sneaks into {target.mention}'s {info['name']}...*")
    await asyncio.sleep(1.5)
    await edit(f"🦝 *planting evidence...*")
    await asyncio.sleep(1.5)
    await edit(f"🦝 *covering tracks...*")
    await asyncio.sleep(1.5)

    if random.random() < SABOTAGE_SUCCESS_CHANCE:
        # Success — damage the business
        biz["damaged_until"] = time.time() + SABOTAGE_DAMAGE_HOURS * 3600
        _save_businesses(data)
        await interaction.edit_original_response(
            content=(
                f"# 💣 SABOTAGE SUCCESSFUL\n\n"
                f"{user.mention} sabotaged {target.mention}'s {info['emoji']} **{info['name']}**!\n"
                f"⏰ It will produce **NOTHING** for the next **{SABOTAGE_DAMAGE_HOURS} hours**."
            ),
            allowed_mentions=discord.AllowedMentions(users=[target]),
        )
        # DM the victim
        try:
            embed = discord.Embed(
                title="🚨 YOUR BUSINESS WAS SABOTAGED",
                description=(
                    f"**{user.display_name}** sabotaged your {info['emoji']} **{info['name']}**.\n"
                    f"⏰ It will produce nothing for the next **{SABOTAGE_DAMAGE_HOURS} hours**.\n\n"
                    f"_Buy `/insurance` to prevent future sabotage attempts._"
                ),
                color=discord.Color.dark_red(),
            )
            await send_dm(target.id, "business", embed=embed)
        except Exception:
            pass
    else:
        # Caught
        fine = min(SABOTAGE_FAIL_FINE, economy.balance(user.id))
        economy.add(user.id, -fine, "sabotage caught")
        # Compensation to target
        comp = fine // 2
        if comp > 0:
            economy.add(target.id, comp, "sabotage compensation")
        await interaction.edit_original_response(
            content=(
                f"# 🚨 YOU GOT CAUGHT\n\n"
                f"{target.mention}'s security caught {user.mention} red-handed!\n"
                f"💸 Paid **{fine:,}** in fines. {target.display_name} got **{comp:,}** in compensation."
            ),
            allowed_mentions=discord.AllowedMentions(users=[target]),
        )


# ── Background task: random business events (IRS, fire, etc.) ────────────────
async def business_events_scheduler():
    """Periodically rolls random events on businesses."""
    await client.wait_until_ready()
    while not client.is_closed():
        # Check every 2 hours
        await asyncio.sleep(2 * 3600)
        try:
            data = _load_businesses()
            cfg = load_config()
            recap_channel_id = get_notification_channel_id(cfg)
            channel = client.get_channel(int(recap_channel_id)) if recap_channel_id else None

            for uid, bizs in data["users"].items():
                if not bizs:
                    continue
                # Insurance blocks events
                if has_insurance(int(uid)):
                    continue
                # 8% chance any user gets an event
                if random.random() > 0.08:
                    continue
                biz = random.choice(bizs)
                if biz.get("damaged_until", 0) > time.time():
                    continue  # already damaged
                info = BUSINESS_TYPES.get(biz["type"], {"emoji":"🏢","name":"?"})

                event = random.choice([
                    ("🔥", "fire broke out", 6, 0.5),    # 6h damage, 50% pending lost
                    ("🚨", "got robbed", 3, 0.3),         # 3h damage, 30% lost
                    ("📋", "IRS audit", 8, 0.0),          # 8h damage no $ loss
                    ("💧", "plumbing burst", 4, 0.2),
                    ("⚡", "electrical failure", 5, 0.1),
                    ("🦠", "health code violation", 6, 0.4),
                ])
                emoji, desc, hours, loss_pct = event
                biz["damaged_until"] = time.time() + hours * 3600

                # Steal/burn some pending income
                pending = _business_pending_income(biz)
                lost = int(pending * loss_pct)
                if lost > 0:
                    biz["last_collected"] = time.time()  # zero out pending

                if channel:
                    try:
                        await channel.send(
                            f"{emoji} **EVENT:** <@{uid}>'s {info['emoji']} **{info['name']}** {desc}!\n"
                            f"⏰ Closed for **{hours} hours**."
                            + (f"\n💸 Lost **{lost:,}** coins of pending income." if lost > 0 else ""),
                            allowed_mentions=discord.AllowedMentions.none(),
                        )
                    except Exception:
                        pass
                # DM the owner
                try:
                    dm_embed = discord.Embed(
                        title=f"{emoji} BUSINESS EVENT",
                        description=(
                            f"Your {info['emoji']} **{info['name']}** {desc}.\n"
                            f"⏰ Closed for **{hours} hours**."
                            + (f"\n💸 Lost **{lost:,}** coins." if lost > 0 else "")
                            + "\n\n_Buy `/insurance` to prevent future events._"
                        ),
                        color=discord.Color.orange(),
                    )
                    await send_dm(int(uid), "business", embed=dm_embed)
                except Exception:
                    pass

            _save_businesses(data)
        except Exception as e:
            log.exception("business_events_scheduler: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# 🛒 EXPANDED SHOP ITEMS
# Insurance, VIP, XP boost, custom titles, lottery multiplier, loans,
# bounties, pet food bundle, business upgrades, heist tools
# (Price constants are defined earlier in the file, above ShopMainView.)
# ─────────────────────────────────────────────────────────────────────────────


def _load_json_file(path, default):
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _save_json_file(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# Helpers for time-based item activations stored in user record
def _user_set_until(user_id: int, key: str, hours: float):
    u = economy._user(user_id)
    u[key] = time.time() + hours * 3600
    economy._save()


def _user_is_active(user_id: int, key: str) -> bool:
    u = economy._user(user_id)
    return u.get(key, 0) > time.time()


def _user_active_remaining(user_id: int, key: str) -> int:
    u = economy._user(user_id)
    return max(0, int(u.get(key, 0) - time.time()))


# ─────────────────────────────────────────────────────────────────────────────
# 🛡️ /insurance — protect business from events & sabotage
# ─────────────────────────────────────────────────────────────────────────────
@tree.command(name="insurance", description="Buy 48h business insurance — blocks sabotage and event damage.")
async def insurance_command(interaction: discord.Interaction):
    user = interaction.user
    if _user_is_active(user.id, "insurance_until"):
        remaining = _user_active_remaining(user.id, "insurance_until")
        await interaction.response.send_message(
            f"🛡️ You already have insurance for **{fmt_cooldown(remaining)}**.", ephemeral=True
        )
        return
    if economy.balance(user.id) < INSURANCE_PRICE:
        await interaction.response.send_message(
            f"❌ Costs **{INSURANCE_PRICE:,}** coins.", ephemeral=True
        )
        return
    economy.add(user.id, -INSURANCE_PRICE, "insurance")
    _user_set_until(user.id, "insurance_until", INSURANCE_DURATION_HOURS)
    await interaction.response.send_message(
        f"# 🛡️ INSURANCE ACTIVE\n\n"
        f"For the next **{INSURANCE_DURATION_HOURS} hours**, your businesses are immune to "
        f"sabotage and random events.\n\n"
        f"Balance: **{economy.balance(user.id):,}**"
    )


def has_insurance(user_id: int) -> bool:
    return _user_is_active(user_id, "insurance_until")


# ─────────────────────────────────────────────────────────────────────────────
# 💎 /vip — weekly cosmetic badge
# ─────────────────────────────────────────────────────────────────────────────
@tree.command(name="vip", description="Buy a VIP badge — glowing 💎 next to your name for 7 days.")
async def vip_command(interaction: discord.Interaction):
    user = interaction.user
    if _user_is_active(user.id, "vip_until"):
        remaining = _user_active_remaining(user.id, "vip_until")
        await interaction.response.send_message(
            f"💎 You're already VIP for **{fmt_cooldown(remaining)}**.", ephemeral=True
        )
        return
    if economy.balance(user.id) < VIP_PRICE:
        await interaction.response.send_message(
            f"❌ Costs **{VIP_PRICE:,}** coins.", ephemeral=True
        )
        return
    economy.add(user.id, -VIP_PRICE, "vip")
    _user_set_until(user.id, "vip_until", VIP_DURATION_DAYS * 24)
    await interaction.response.send_message(
        f"# 💎 VIP ACTIVATED\n\n"
        f"{user.mention} is now VIP for **{VIP_DURATION_DAYS} days**!\n"
        f"💎 badge will appear next to your name everywhere."
    )


def is_vip(user_id: int) -> bool:
    return _user_is_active(user_id, "vip_until")


# ─────────────────────────────────────────────────────────────────────────────
# ⚡ /xpboost — 2x XP for 24h
# ─────────────────────────────────────────────────────────────────────────────
@tree.command(name="xpboost", description="Buy a 2x XP boost for 24 hours.")
async def xpboost_command(interaction: discord.Interaction):
    user = interaction.user
    if _user_is_active(user.id, "xp_boost_until"):
        remaining = _user_active_remaining(user.id, "xp_boost_until")
        await interaction.response.send_message(
            f"⚡ XP boost active for **{fmt_cooldown(remaining)}** more.", ephemeral=True
        )
        return
    if economy.balance(user.id) < XP_BOOST_PRICE:
        await interaction.response.send_message(
            f"❌ Costs **{XP_BOOST_PRICE:,}** coins.", ephemeral=True
        )
        return
    economy.add(user.id, -XP_BOOST_PRICE, "xp boost")
    _user_set_until(user.id, "xp_boost_until", XP_BOOST_DURATION_HOURS)
    await interaction.response.send_message(
        f"# ⚡ XP BOOST ACTIVE\n\n"
        f"For the next **{XP_BOOST_DURATION_HOURS} hours**, you earn **2x XP** from messages and commands."
    )


def xp_multiplier(user_id: int) -> int:
    return 2 if _user_is_active(user_id, "xp_boost_until") else 1


# ─────────────────────────────────────────────────────────────────────────────
# 🏷️ /title — set or buy a custom title
# ─────────────────────────────────────────────────────────────────────────────
@tree.command(name="title", description="Set or buy a custom title (shown on /balance and /leaderboard).")
@discord.app_commands.describe(new_title="Your custom title (max 24 chars). Leave empty to view yours.")
async def title_command(interaction: discord.Interaction, new_title: str = None):
    user = interaction.user
    u = economy._user(user.id)
    current = u.get("custom_title", "")

    if new_title is None:
        if current:
            await interaction.response.send_message(
                f"🏷️ Your title: **{current}**\nUse `/title new_title:...` to change. "
                f"Each new title costs **{CUSTOM_TITLE_PRICE:,}** coins."
            )
        else:
            await interaction.response.send_message(
                f"You have no custom title. Set one with `/title new_title:Your Title Here` "
                f"for **{CUSTOM_TITLE_PRICE:,}** coins."
            )
        return

    title_text = new_title.strip()[:24]
    if not title_text:
        await interaction.response.send_message("Title can't be empty.", ephemeral=True)
        return
    # Block @everyone-style abuse
    title_text = title_text.replace("@everyone", "").replace("@here", "")

    if economy.balance(user.id) < CUSTOM_TITLE_PRICE:
        await interaction.response.send_message(
            f"❌ Costs **{CUSTOM_TITLE_PRICE:,}** coins.", ephemeral=True
        )
        return

    economy.add(user.id, -CUSTOM_TITLE_PRICE, "custom title")
    u["custom_title"] = title_text
    economy._save()
    await interaction.response.send_message(
        f"🏷️ Your title is now: **{title_text}**"
    )


def get_custom_title(user_id: int) -> str:
    return economy._user(user_id).get("custom_title", "")


# ─────────────────────────────────────────────────────────────────────────────
# 🎰 /lotterymult — 2x next lottery win (one-shot)
# ─────────────────────────────────────────────────────────────────────────────
@tree.command(name="lotterymult", description="Buy a 2x multiplier on your next lottery win (one-time use).")
async def lotterymult_command(interaction: discord.Interaction):
    user = interaction.user
    u = economy._user(user.id)
    if u.get("lottery_mult", 1) > 1:
        await interaction.response.send_message(
            "🎰 You already have a lottery multiplier ready for your next win.", ephemeral=True
        )
        return
    if economy.balance(user.id) < LOTTERY_MULT_PRICE:
        await interaction.response.send_message(
            f"❌ Costs **{LOTTERY_MULT_PRICE:,}** coins.", ephemeral=True
        )
        return
    economy.add(user.id, -LOTTERY_MULT_PRICE, "lottery multiplier")
    u["lottery_mult"] = 2
    economy._save()
    await interaction.response.send_message(
        f"# 🎰 LOTTERY MULTIPLIER ACTIVE\n\n"
        f"If you win the next lottery drawing, you'll receive **2x** the jackpot!\n"
        f"Active until you win a lottery."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 💸 /loan — borrow coins now, pay more later
# ─────────────────────────────────────────────────────────────────────────────
@tree.command(name="loan", description="Borrow coins from the loan shark. Owe more later.")
@discord.app_commands.describe(amount="How much to borrow")
@discord.app_commands.choices(
    amount=[
        discord.app_commands.Choice(name=f"{k:,} → owe {v[1]:,} in {v[2]}h", value=k)
        for k, v in LOAN_AMOUNTS.items()
    ]
)
async def loan_command(interaction: discord.Interaction, amount: discord.app_commands.Choice[int]):
    user = interaction.user
    loans = _load_json_file(LOANS_FILE, {"users": {}})
    if str(user.id) in loans["users"]:
        existing = loans["users"][str(user.id)]
        remaining = max(0, existing["due_at"] - time.time())
        await interaction.response.send_message(
            f"❌ You already have a loan of **{existing['owe']:,}** coins. "
            f"Due in **{fmt_cooldown(int(remaining))}**. Repay with `/repay`.",
            ephemeral=True,
        )
        return

    borrow_amount = amount.value
    cost, owe, hours = LOAN_AMOUNTS[borrow_amount]
    economy.add(user.id, borrow_amount, "loan borrow")
    loans["users"][str(user.id)] = {
        "borrowed": borrow_amount,
        "owe": owe,
        "due_at": time.time() + hours * 3600,
        "borrowed_at": time.time(),
    }
    _save_json_file(LOANS_FILE, loans)
    await interaction.response.send_message(
        f"# 💸 LOAN APPROVED\n\n"
        f"You received **{borrow_amount:,}** coins.\n"
        f"You owe the loan shark **{owe:,}** coins in **{hours} hours**.\n"
        f"⚠️ Miss the deadline → lose all coins to interest until paid.\n\n"
        f"Repay anytime with `/repay`."
    )


@tree.command(name="repay", description="Repay your loan.")
async def repay_command(interaction: discord.Interaction):
    user = interaction.user
    loans = _load_json_file(LOANS_FILE, {"users": {}})
    if str(user.id) not in loans["users"]:
        await interaction.response.send_message("You don't have an active loan.", ephemeral=True)
        return
    loan = loans["users"][str(user.id)]
    owe = loan["owe"]
    if economy.balance(user.id) < owe:
        await interaction.response.send_message(
            f"❌ You owe **{owe:,}** coins but only have **{economy.balance(user.id):,}**.",
            ephemeral=True,
        )
        return
    economy.add(user.id, -owe, "loan repay")
    del loans["users"][str(user.id)]
    _save_json_file(LOANS_FILE, loans)
    await interaction.response.send_message(
        f"💸 Loan repaid. **{owe:,}** coins gone. The loan shark nods respectfully.\n"
        f"Balance: **{economy.balance(user.id):,}**"
    )


async def pet_starving_scheduler():
    """DM pet owners when their pet is going hungry (every 4h, max 1 DM/day per user)."""
    await client.wait_until_ready()
    while not client.is_closed():
        await asyncio.sleep(4 * 3600)
        try:
            pets = _load_pets()
            for uid, pet in pets.items():
                hunger = _pet_hunger(pet)
                if hunger >= 30:
                    continue
                # Don't spam — max 1 hunger DM per day per user
                last_dm = pet.get("last_hunger_dm", 0)
                if time.time() - last_dm < 24 * 3600:
                    continue
                info = PET_TYPES.get(pet["type"], {"emoji":"🐾","name":"?"})
                state = "starving" if hunger == 0 else ("very hungry" if hunger < 15 else "hungry")
                try:
                    embed = discord.Embed(
                        title=f"🍖 PET {state.upper()}",
                        description=(
                            f"{info['emoji']} **{pet['name']}** is {state}. Hunger: **{hunger}/100**.\n\n"
                            f"_Run `/feed` or buy `/petfood` for a 7-day bundle._"
                        ),
                        color=discord.Color.orange(),
                    )
                    sent = await send_dm(int(uid), "pet", embed=embed)
                    if sent:
                        pet["last_hunger_dm"] = time.time()
                except Exception:
                    pass
            _save_pets(pets)
        except Exception as e:
            log.exception("pet_starving_scheduler: %s", e)


async def loan_shark_scheduler():
    """Background task: charge defaulted loans periodically."""
    await client.wait_until_ready()
    while not client.is_closed():
        await asyncio.sleep(3600)  # every hour
        try:
            loans = _load_json_file(LOANS_FILE, {"users": {}})
            now = time.time()
            cfg = load_config()
            recap_channel_id = get_notification_channel_id(cfg)
            channel = client.get_channel(int(recap_channel_id)) if recap_channel_id else None
            changed = False
            for uid, loan in list(loans["users"].items()):
                # Send "due soon" warning if loan due within 2h and not already warned
                time_until_due = loan["due_at"] - now
                if 0 < time_until_due < 7200 and not loan.get("warning_sent"):
                    loan["warning_sent"] = True
                    changed = True
                    try:
                        embed = discord.Embed(
                            title="⏰ LOAN DUE SOON",
                            description=(
                                f"Your loan of **{loan['owe']:,}** coins is due in **{fmt_cooldown(int(time_until_due))}**.\n\n"
                                f"_Run `/repay` to pay it off and avoid 10%/hr penalties._"
                            ),
                            color=discord.Color.gold(),
                        )
                        await send_dm(int(uid), "loan", embed=embed)
                    except Exception:
                        pass
                if loan["due_at"] < now:
                    # Default — apply 10% interest per hour overdue, taken from balance or business pending
                    overdue_hours = (now - loan["due_at"]) / 3600
                    if overdue_hours < 1:
                        continue
                    penalty = int(loan["owe"] * 0.1)
                    bal = economy.balance(int(uid))
                    actual = min(penalty, bal)
                    if actual > 0:
                        economy.add(int(uid), -actual, "loan default penalty")
                        if channel:
                            try:
                                await channel.send(
                                    f"💸 **LOAN SHARK COLLECTS**\n"
                                    f"<@{uid}> defaulted on their loan. Lost **{actual:,}** coins.",
                                    allowed_mentions=discord.AllowedMentions.none(),
                                )
                            except Exception:
                                pass
                        # DM the defaulter
                        try:
                            embed = discord.Embed(
                                title="💸 LOAN OVERDUE",
                                description=(
                                    f"You missed your loan deadline. The loan shark took **{actual:,}** coins.\n"
                                    f"You still owe **{loan['owe']:,}** coins total.\n\n"
                                    f"_Run `/repay` to pay it off before more penalties hit._"
                                ),
                                color=discord.Color.dark_red(),
                            )
                            await send_dm(int(uid), "loan", embed=embed)
                        except Exception:
                            pass
                        loan["due_at"] = now + 3600  # next charge in 1h
                        changed = True
                    else:
                        # Broke — wipe loan (loan shark gives up)
                        del loans["users"][uid]
                        changed = True
            if changed:
                _save_json_file(LOANS_FILE, loans)
        except Exception as e:
            log.exception("loan_shark_scheduler: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# 💰 /bounty — put a bounty on a user
# ─────────────────────────────────────────────────────────────────────────────
@tree.command(name="bounty", description="Place a bounty on a user. Whoever beats them in PvP wins it.")
@discord.app_commands.describe(target="Who to put a bounty on", amount="How many coins (min 500)")
async def bounty_command(interaction: discord.Interaction, target: discord.Member, amount: int):
    user = interaction.user
    if target.id == user.id:
        await interaction.response.send_message("Can't bounty yourself.", ephemeral=True)
        return
    if target.bot:
        await interaction.response.send_message("Can't bounty a bot.", ephemeral=True)
        return
    if amount < BOUNTY_MIN:
        await interaction.response.send_message(f"❌ Min bounty is **{BOUNTY_MIN}** coins.", ephemeral=True)
        return
    if economy.balance(user.id) < amount:
        await interaction.response.send_message(
            f"❌ You only have **{economy.balance(user.id):,}** coins.", ephemeral=True
        )
        return

    bounties = _load_json_file(BOUNTIES_FILE, {"targets": {}})
    economy.add(user.id, -amount, "bounty placed")
    targets = bounties["targets"].setdefault(str(target.id), [])
    targets.append({
        "placer_id": user.id,
        "amount": amount,
        "placed_at": time.time(),
    })
    _save_json_file(BOUNTIES_FILE, bounties)

    total_on_target = sum(b["amount"] for b in targets)
    await interaction.response.send_message(
        f"# 💰 BOUNTY PLACED\n\n"
        f"{user.mention} put a **{amount:,}** coin bounty on {target.mention}!\n"
        f"💀 Total bounty on {target.display_name}: **{total_on_target:,}** coins\n\n"
        f"Whoever beats {target.display_name} in `/fight`, `/duel`, or `/rob` claims it.",
        allowed_mentions=discord.AllowedMentions(users=[target]),
    )
    # DM the target
    try:
        embed = discord.Embed(
            title="💀 BOUNTY PLACED ON YOU",
            description=(
                f"**{user.display_name}** put a **{amount:,}** coin bounty on you.\n"
                f"💀 Total bounty on your head: **{total_on_target:,}** coins\n\n"
                f"Anyone who beats you in `/fight`, `/duel`, or `/rob` claims it.\n"
                f"_Watch your back._"
            ),
            color=discord.Color.dark_red(),
        )
        await send_dm(target.id, "bounty", embed=embed)
    except Exception:
        pass


@tree.command(name="bounties", description="See all active bounties.")
async def bounties_command(interaction: discord.Interaction):
    bounties = _load_json_file(BOUNTIES_FILE, {"targets": {}})
    if not bounties["targets"]:
        await interaction.response.send_message("No active bounties.", ephemeral=True)
        return
    lines = []
    sorted_targets = sorted(
        bounties["targets"].items(),
        key=lambda x: sum(b["amount"] for b in x[1]),
        reverse=True,
    )
    for uid, blist in sorted_targets[:15]:
        total = sum(b["amount"] for b in blist)
        lines.append(f"💀 <@{uid}> — **{total:,}** coins ({len(blist)} bounty{'s' if len(blist) != 1 else ''})")

    embed = discord.Embed(
        title="💰 ACTIVE BOUNTIES",
        description="\n".join(lines),
        color=discord.Color.dark_red(),
    )
    embed.set_footer(text="Beat them in /fight, /duel, or /rob to claim.")
    await interaction.response.send_message(embed=embed, allowed_mentions=discord.AllowedMentions.none())


async def claim_bounty(winner_id: int, loser_id: int, channel) -> int:
    """Called by fight/duel/rob handlers. Pays out all bounties on loser to winner. Returns total."""
    bounties = _load_json_file(BOUNTIES_FILE, {"targets": {}})
    targets = bounties["targets"].get(str(loser_id))
    if not targets:
        return 0
    total = sum(b["amount"] for b in targets)
    if total <= 0:
        return 0
    economy.add(winner_id, total, "bounty claim")
    del bounties["targets"][str(loser_id)]
    _save_json_file(BOUNTIES_FILE, bounties)
    try:
        if channel:
            await channel.send(
                f"💀 **BOUNTY CLAIMED!**\n"
                f"<@{winner_id}> collected **{total:,}** coins for taking down <@{loser_id}>.",
                allowed_mentions=discord.AllowedMentions.none(),
            )
    except Exception:
        pass
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 🍖 /petfood — feed pet for 7 days (auto)
# ─────────────────────────────────────────────────────────────────────────────
@tree.command(name="petfood", description="Buy a 7-day food bundle for your pet (auto-feeds, discounted).")
async def petfood_command(interaction: discord.Interaction):
    user = interaction.user
    pets = _load_pets()
    if str(user.id) not in pets:
        await interaction.response.send_message("You don't have a pet.", ephemeral=True)
        return
    if economy.balance(user.id) < PET_FOOD_BUNDLE_PRICE:
        await interaction.response.send_message(
            f"❌ Costs **{PET_FOOD_BUNDLE_PRICE:,}** coins.", ephemeral=True
        )
        return
    economy.add(user.id, -PET_FOOD_BUNDLE_PRICE, "pet food bundle")
    pet = pets[str(user.id)]
    # Push last_fed forward by 7 days
    pet["last_fed"] = time.time() + PET_FOOD_BUNDLE_DAYS * 24 * 3600
    pet["xp"] = pet.get("xp", 0) + 50  # bonus XP for the bundle
    _save_pets(pets)
    info = PET_TYPES.get(pet["type"], {"emoji": "🐾"})
    await interaction.response.send_message(
        f"# 🍖 PET FOOD BUNDLE PURCHASED\n\n"
        f"{info['emoji']} **{pet['name']}** is fed for the next **{PET_FOOD_BUNDLE_DAYS} days**!\n"
        f"Bonus: +50 XP from the great meal."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 📈 /upgradebusiness — permanently boost a single business
# ─────────────────────────────────────────────────────────────────────────────
@tree.command(name="upgradebusiness", description="Permanently upgrade ONE of your businesses (+10% income per level).")
@discord.app_commands.describe(business_type="Which business type to upgrade")
@discord.app_commands.choices(
    business_type=[
        discord.app_commands.Choice(name=f"{b['emoji']} {b['name']}", value=k)
        for k, b in BUSINESS_TYPES.items()
    ]
)
async def upgradebusiness_command(
    interaction: discord.Interaction,
    business_type: discord.app_commands.Choice[str],
):
    user = interaction.user
    data = _load_businesses()
    user_bizs = data["users"].get(str(user.id), [])
    biz_type = business_type.value

    matching = [b for b in user_bizs if b["type"] == biz_type]
    if not matching:
        await interaction.response.send_message(
            f"❌ You don't own a {BUSINESS_TYPES[biz_type]['name']}.", ephemeral=True
        )
        return

    # Pick the lowest-level instance to upgrade
    matching.sort(key=lambda b: b.get("upgrade_level", 0))
    biz = matching[0]
    current_level = biz.get("upgrade_level", 0)
    if current_level >= BUSINESS_UPGRADE_MAX_LEVEL:
        await interaction.response.send_message(
            f"❌ This business is maxed at level {BUSINESS_UPGRADE_MAX_LEVEL}.", ephemeral=True
        )
        return
    cost = BUSINESS_UPGRADE_PRICE_PER_LEVEL * (current_level + 1)
    if economy.balance(user.id) < cost:
        await interaction.response.send_message(
            f"❌ Upgrade level {current_level + 1} costs **{cost:,}** coins.", ephemeral=True
        )
        return

    economy.add(user.id, -cost, "business upgrade")
    biz["upgrade_level"] = current_level + 1
    _save_businesses(data)
    info = BUSINESS_TYPES[biz_type]
    boost_total = (current_level + 1) * BUSINESS_UPGRADE_BOOST * 100
    await interaction.response.send_message(
        f"# 📈 BUSINESS UPGRADED\n\n"
        f"{info['emoji']} **{info['name']}** upgraded to **Level {current_level + 1}**!\n"
        f"💰 Total boost: **+{int(boost_total)}%** income.\n"
        f"Cost: **{cost:,}** coins.\n"
        f"_Max upgrade level: {BUSINESS_UPGRADE_MAX_LEVEL}_"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 🦝 /heisttools — 24h /rob success boost
# ─────────────────────────────────────────────────────────────────────────────
@tree.command(name="heisttools", description="Buy heist tools — +20% rob success rate for 24h.")
async def heisttools_command(interaction: discord.Interaction):
    user = interaction.user
    if _user_is_active(user.id, "heist_tools_until"):
        remaining = _user_active_remaining(user.id, "heist_tools_until")
        await interaction.response.send_message(
            f"🦝 Heist tools active for **{fmt_cooldown(remaining)}** more.", ephemeral=True
        )
        return
    if economy.balance(user.id) < HEIST_TOOLS_PRICE:
        await interaction.response.send_message(
            f"❌ Costs **{HEIST_TOOLS_PRICE:,}** coins.", ephemeral=True
        )
        return
    economy.add(user.id, -HEIST_TOOLS_PRICE, "heist tools")
    _user_set_until(user.id, "heist_tools_until", HEIST_TOOLS_DURATION_HOURS)
    await interaction.response.send_message(
        f"# 🦝 HEIST TOOLS EQUIPPED\n\n"
        f"For the next **{HEIST_TOOLS_DURATION_HOURS} hours**, your `/rob` success rate is boosted by **+{int(HEIST_TOOLS_BOOST*100)}%**."
    )


def has_heist_tools(user_id: int) -> bool:
    return _user_is_active(user_id, "heist_tools_until")


# ─────────────────────────────────────────────────────────────────────────────
# 📬 DM NOTIFICATIONS
# Sends DMs for important per-user events. Users can opt-out per category.
# ─────────────────────────────────────────────────────────────────────────────

# Notification categories with default state (all on by default)
DM_CATEGORIES = {
    "bounty":     "💀 Bounty placed on you",
    "rob":        "💸 You got robbed",
    "business":   "🏢 Your business sabotaged or event hit",
    "loan":       "💸 Loan due / overdue warnings",
    "pet":        "🐶 Pet starving",
    "lottery":    "🎰 You won the lottery",
    "rep":        "👍 Someone gave you rep",
    "tournament": "🏆 Tournament results",
}


def _dm_enabled(user_id: int, category: str) -> bool:
    """Check if user wants DMs for a category. Defaults to ON."""
    u = economy._user(user_id)
    settings = u.get("dm_settings", {})
    # Special "all off" flag overrides
    if settings.get("_all_off"):
        return False
    return settings.get(category, True)  # default True


async def send_dm(user_id: int, category: str, content: str = None, embed: discord.Embed = None) -> bool:
    """Send a DM to a user if they have that category enabled.
    Returns True if sent, False if blocked/disabled/DMs closed."""
    if not _dm_enabled(user_id, category):
        return False
    try:
        user = await client.fetch_user(user_id)
        if user is None:
            return False
        kwargs = {}
        if content:
            kwargs["content"] = content
        if embed:
            kwargs["embed"] = embed
        await user.send(**kwargs)
        return True
    except discord.Forbidden:
        # User has DMs closed — don't crash
        return False
    except Exception as e:
        log.warning("DM to %s failed: %s", user_id, e)
        return False


@tree.command(name="notifications", description="Toggle which DM notifications you receive from Jordan.")
async def notifications_command(interaction: discord.Interaction):
    user = interaction.user
    u = economy._user(user.id)
    settings = u.setdefault("dm_settings", {})

    embed = discord.Embed(
        title="📬 DM Notifications",
        description="Toggle which events DM you. Click a button to flip its state.\n_Default: all enabled._",
        color=discord.Color.blurple(),
    )
    if settings.get("_all_off"):
        embed.add_field(name="⚠️ All notifications", value="**OFF** — click 'Enable all' to receive any DMs.", inline=False)
    else:
        lines = []
        for key, label in DM_CATEGORIES.items():
            state = "✅" if settings.get(key, True) else "❌"
            lines.append(f"{state} {label}")
        embed.add_field(name="Current settings", value="\n".join(lines), inline=False)

    view = NotificationsView(user.id)
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)


class NotificationsView(discord.ui.View):
    def __init__(self, user_id: int):
        super().__init__(timeout=180)
        self.user_id = user_id
        # Make a button for each category
        for i, (key, label) in enumerate(DM_CATEGORIES.items()):
            short = label.split(" ", 1)[1][:30] if " " in label else label
            btn = discord.ui.Button(
                label=short,
                emoji=label.split(" ")[0],
                style=discord.ButtonStyle.secondary,
                row=i // 4,
            )
            btn.callback = self._make_toggle(key, short)
            self.add_item(btn)
        # Master toggles
        all_on = discord.ui.Button(label="Enable all", style=discord.ButtonStyle.success, row=2)
        all_on.callback = self._all_on
        self.add_item(all_on)
        all_off = discord.ui.Button(label="Disable all", style=discord.ButtonStyle.danger, row=2)
        all_off.callback = self._all_off
        self.add_item(all_off)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your settings.", ephemeral=True)
            return False
        return True

    def _make_toggle(self, key: str, label: str):
        async def cb(interaction: discord.Interaction):
            u = economy._user(self.user_id)
            settings = u.setdefault("dm_settings", {})
            # If master "off" was on, un-set that and turn this one on
            if settings.get("_all_off"):
                settings.pop("_all_off", None)
                for k in DM_CATEGORIES:
                    settings[k] = False
                settings[key] = True
            else:
                current = settings.get(key, True)
                settings[key] = not current
            economy._save()
            state = "ON ✅" if settings.get(key, True) else "OFF ❌"
            await interaction.response.send_message(f"{label}: **{state}**", ephemeral=True)
        return cb

    async def _all_on(self, interaction: discord.Interaction):
        u = economy._user(self.user_id)
        u["dm_settings"] = {}  # empty = all defaults (true)
        economy._save()
        await interaction.response.send_message("✅ All DM notifications enabled.", ephemeral=True)

    async def _all_off(self, interaction: discord.Interaction):
        u = economy._user(self.user_id)
        u["dm_settings"] = {"_all_off": True}
        economy._save()
        await interaction.response.send_message("❌ All DM notifications disabled.", ephemeral=True)


# ─────────────────────────────────────────────────────────────────────────────
# 💊 DEALER MINI-GAME
# Buy supply from the plug, sell to NPCs, dodge cops, watch market prices.
# Pure economy mini-game. No /use commands, no effects, no dosing info.
# ─────────────────────────────────────────────────────────────────────────────
DEALER_FILE = MEMORY_DIR / "dealer.json"

# Substance catalog. Prices are PER GRAM in coins.
# heat = chance of getting caught per sale (out of 100)
# base_buy/base_sell are baseline prices — market fluctuates ±30% daily
SUBSTANCES = {
    "weed": {
        "emoji": "🌿", "name": "Weed",
        "base_buy": 50, "base_sell": 90,
        "heat": 3, "stash_max": 200,
        "tier": 1, "desc": "Low risk starter product.",
    },
    "shrooms": {
        "emoji": "🍄", "name": "Shrooms",
        "base_buy": 120, "base_sell": 220,
        "heat": 4, "stash_max": 100,
        "tier": 1, "desc": "Boutique market. Steady demand.",
    },
    "molly": {
        "emoji": "💊", "name": "Molly",
        "base_buy": 180, "base_sell": 380,
        "heat": 8, "stash_max": 80,
        "tier": 2, "desc": "Festival favorite. Good margins.",
    },
    "addy": {
        "emoji": "💉", "name": "Addy",
        "base_buy": 100, "base_sell": 230,
        "heat": 6, "stash_max": 120,
        "tier": 2, "desc": "Steady stream of college clients.",
    },
    "coke": {
        "emoji": "❄️", "name": "Coke",
        "base_buy": 400, "base_sell": 900,
        "heat": 15, "stash_max": 50,
        "tier": 3, "desc": "Wall Street loves it. So do the cops.",
    },
    "meff": {
        "emoji": "🔥", "name": "Meff",
        "base_buy": 250, "base_sell": 700,
        "heat": 20, "stash_max": 40,
        "tier": 3, "desc": "Massive markup. Massive heat.",
    },
    "ket": {
        "emoji": "🐴", "name": "Ket",
        "base_buy": 300, "base_sell": 650,
        "heat": 10, "stash_max": 60,
        "tier": 2, "desc": "Niche but consistent.",
    },
    "fent": {
        "emoji": "☠️", "name": "Fent",
        "base_buy": 1000, "base_sell": 3000,
        "heat": 35, "stash_max": 20,
        "tier": 4, "desc": "Highest profit. Highest heat. Don't.",
    },
}

# Cooldowns
SUPPLY_COOLDOWN_MIN = 30  # minutes between /buysupply runs
SELL_COOLDOWN_MIN = 10    # minutes between /sell runs
BUST_FINE_PCT = 0.2       # lose 20% of balance when caught
HEAT_DECAY_PER_HOUR = 5   # heat drops by 5 per hour of inactivity
HEAT_MAX = 100            # at 100 heat, cops auto-bust next sell


def _load_dealer() -> dict:
    if DEALER_FILE.exists():
        try:
            with open(DEALER_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"users": {}, "market_date": "", "market_prices": {}}


def _save_dealer(data: dict):
    with open(DEALER_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _get_market_prices() -> dict:
    """Returns today's market prices for all substances. Refreshes daily."""
    data = _load_dealer()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if data.get("market_date") != today:
        new_prices = {}
        for key, info in SUBSTANCES.items():
            buy_mult = random.uniform(0.7, 1.3)
            sell_mult = random.uniform(0.7, 1.3)
            new_prices[key] = {
                "buy": int(info["base_buy"] * buy_mult),
                "sell": int(info["base_sell"] * sell_mult),
            }
        data["market_date"] = today
        data["market_prices"] = new_prices
        _save_dealer(data)
    return data["market_prices"]


def _get_dealer_record(user_id: int) -> dict:
    data = _load_dealer()
    uid = str(user_id)
    if uid not in data["users"]:
        data["users"][uid] = {
            "stash": {},          # substance -> grams
            "heat": 0,
            "last_supply": 0,
            "last_sell": 0,
            "last_heat_decay": time.time(),
            "lifetime_sold": 0,
            "lifetime_profit": 0,
            "times_busted": 0,
        }
        _save_dealer(data)
    return data["users"][uid]


def _decay_heat(record: dict):
    """Apply hourly heat decay."""
    last = record.get("last_heat_decay", time.time())
    hours = (time.time() - last) / 3600
    if hours > 0:
        record["heat"] = max(0, record["heat"] - int(hours * HEAT_DECAY_PER_HOUR))
        record["last_heat_decay"] = time.time()


# ── /buysupply ───────────────────────────────────────────────────────────────
@tree.command(name="buysupply", description="Buy product from the plug. Market prices change daily.")
@discord.app_commands.describe(substance="What to buy", grams="How many grams")
@discord.app_commands.choices(
    substance=[
        discord.app_commands.Choice(name=f"{s['emoji']} {s['name']}", value=k)
        for k, s in SUBSTANCES.items()
    ]
)
async def buysupply_command(
    interaction: discord.Interaction,
    substance: discord.app_commands.Choice[str],
    grams: int,
):
    user = interaction.user
    sub_key = substance.value
    if sub_key not in SUBSTANCES:
        await interaction.response.send_message("Invalid substance.", ephemeral=True)
        return
    if grams <= 0 or grams > 500:
        await interaction.response.send_message("❌ Grams must be 1–500.", ephemeral=True)
        return

    info = SUBSTANCES[sub_key]
    data = _load_dealer()
    uid = str(user.id)
    record = data["users"].setdefault(uid, {
        "stash": {}, "heat": 0, "last_supply": 0, "last_sell": 0,
        "last_heat_decay": time.time(), "lifetime_sold": 0,
        "lifetime_profit": 0, "times_busted": 0,
    })

    # Cooldown
    cd_remaining = SUPPLY_COOLDOWN_MIN * 60 - (time.time() - record.get("last_supply", 0))
    if cd_remaining > 0:
        await interaction.response.send_message(
            f"⏰ The plug isn't picking up. Try again in **{fmt_cooldown(int(cd_remaining))}**.",
            ephemeral=True,
        )
        return

    # Stash limit check
    current_grams = record["stash"].get(sub_key, 0)
    if current_grams + grams > info["stash_max"]:
        room = info["stash_max"] - current_grams
        await interaction.response.send_message(
            f"❌ Stash too full. You can hold **{room}g** more {info['name']} "
            f"(max {info['stash_max']}g). Sell or get caught with less.",
            ephemeral=True,
        )
        return

    prices = _get_market_prices()
    buy_price = prices[sub_key]["buy"]
    total_cost = buy_price * grams

    if economy.balance(user.id) < total_cost:
        await interaction.response.send_message(
            f"❌ Need **{total_cost:,}** coins. You have **{economy.balance(user.id):,}**.",
            ephemeral=True,
        )
        return

    economy.add(user.id, -total_cost, f"supply: {sub_key}")
    record["stash"][sub_key] = current_grams + grams
    record["last_supply"] = time.time()
    _save_dealer(data)

    await interaction.response.send_message(
        f"# {info['emoji']} SUPPLY ACQUIRED\n\n"
        f"You bought **{grams}g of {info['name']}** for **{total_cost:,}** coins "
        f"(**{buy_price}/g** today).\n\n"
        f"📦 Stash: **{current_grams + grams}g / {info['stash_max']}g**\n"
        f"💰 Balance: **{economy.balance(user.id):,}**\n\n"
        f"_Use `/sell` to move product. Watch your heat._"
    )


# ── /sell ────────────────────────────────────────────────────────────────────
@tree.command(name="sell", description="Sell product to NPC customers. Builds cop heat.")
@discord.app_commands.describe(substance="What to sell", grams="How many grams")
@discord.app_commands.choices(
    substance=[
        discord.app_commands.Choice(name=f"{s['emoji']} {s['name']}", value=k)
        for k, s in SUBSTANCES.items()
    ]
)
async def sell_command(
    interaction: discord.Interaction,
    substance: discord.app_commands.Choice[str],
    grams: int,
):
    user = interaction.user
    sub_key = substance.value
    if sub_key not in SUBSTANCES:
        await interaction.response.send_message("Invalid substance.", ephemeral=True)
        return
    if grams <= 0:
        await interaction.response.send_message("❌ Grams must be positive.", ephemeral=True)
        return

    info = SUBSTANCES[sub_key]
    data = _load_dealer()
    uid = str(user.id)
    if uid not in data["users"]:
        await interaction.response.send_message(
            "❌ You don't have any product. Buy from `/buysupply` first.", ephemeral=True
        )
        return
    record = data["users"][uid]
    _decay_heat(record)

    # Cooldown
    cd_remaining = SELL_COOLDOWN_MIN * 60 - (time.time() - record.get("last_sell", 0))
    if cd_remaining > 0:
        await interaction.response.send_message(
            f"⏰ Customers haven't lined up yet. Try again in **{fmt_cooldown(int(cd_remaining))}**.",
            ephemeral=True,
        )
        return

    current_grams = record["stash"].get(sub_key, 0)
    if current_grams < grams:
        await interaction.response.send_message(
            f"❌ You only have **{current_grams}g** of {info['name']}.", ephemeral=True
        )
        return

    # Heat at 100 = auto-bust
    if record["heat"] >= HEAT_MAX:
        # Forced bust
        await _bust_player(interaction, record, data, forced=True)
        return

    # Roll for cops based on heat + substance heat
    cop_chance = info["heat"] + (record["heat"] / 5)  # heat scales risk
    if random.random() * 100 < cop_chance:
        await _bust_player(interaction, record, data, forced=False)
        return

    # Successful sale
    prices = _get_market_prices()
    sell_price = prices[sub_key]["sell"]
    total = sell_price * grams
    record["stash"][sub_key] = current_grams - grams
    if record["stash"][sub_key] == 0:
        del record["stash"][sub_key]
    record["last_sell"] = time.time()
    # Add heat (more for hot products + bigger sales)
    heat_gain = max(1, int(info["heat"] * (grams / 10)))
    record["heat"] = min(HEAT_MAX, record["heat"] + heat_gain)
    record["lifetime_sold"] = record.get("lifetime_sold", 0) + grams
    record["lifetime_profit"] = record.get("lifetime_profit", 0) + total
    _save_dealer(data)
    new_bal = economy.add(user.id, total, f"sold {sub_key}")
    track_economy_event("earned", total)
    track_feature_use("dealer")
    track_activity("dealer_sale", user.id, user.display_name, f"sold {grams}g of {info['name']} for {total:,}")

    # Quest tracking
    track_quest_progress(user.id, "coins_earned", total)
    add_tournament_score(user.id, coins_earned=total)
    await trigger_balance_check(user.id, channel=interaction.channel)

    # NPC narration
    npcs = [
        "a college kid", "a Wall Street guy in a suit", "a tired nurse",
        "a sketchy dude in a hoodie", "a soccer mom in a Range Rover",
        "a tweaker on a bike", "an Uber driver on break", "a sad-looking lawyer",
        "a hot girl at the bar", "a guy who said 'my buddy sent me'",
    ]
    await interaction.response.send_message(
        f"# 💰 DEAL DONE\n\n"
        f"Sold **{grams}g of {info['emoji']} {info['name']}** to {random.choice(npcs)} "
        f"for **{total:,}** coins (**{sell_price}/g**).\n\n"
        f"📦 Stash: **{current_grams - grams}g / {info['stash_max']}g**\n"
        f"🔥 Heat: **{record['heat']}/100** _(+{heat_gain})_\n"
        f"💰 Balance: **{new_bal:,}**"
    )


async def _bust_player(interaction, record, data, forced=False):
    """Cops catch the player. Lose all stash + a fine."""
    user = interaction.user
    stash_grams = sum(record["stash"].values())
    record["stash"] = {}
    record["heat"] = 0
    record["times_busted"] = record.get("times_busted", 0) + 1
    record["last_sell"] = time.time()

    bal = economy.balance(user.id)
    fine = int(bal * BUST_FINE_PCT)
    economy.add(user.id, -fine, "cop fine")
    _save_dealer(data)

    title = "🚨 BUSTED — HEAT MAXED OUT" if forced else "🚨 BUSTED"
    flavor = [
        "Undercover cop. Sting operation. They had your face for weeks.",
        "Some snitch ratted you out for a plea deal.",
        "A patrol unit happened to roll up at the wrong moment.",
        "You sold to a narc. Rookie mistake.",
        "Surveillance had been tailing you all week.",
        "Your customer was wearing a wire. Damn.",
    ]
    await interaction.response.send_message(
        f"# {title}\n\n"
        f"{random.choice(flavor)}\n\n"
        f"💸 **Lost {stash_grams}g** of product (entire stash).\n"
        f"⚖️ **Fine:** {fine:,} coins ({int(BUST_FINE_PCT*100)}% of balance)\n"
        f"🔥 Heat reset to **0** — lay low for a while.\n\n"
        f"_Balance: **{economy.balance(user.id):,}**_"
    )


# ── /stash ───────────────────────────────────────────────────────────────────
@tree.command(name="stash", description="View your dealing stash, heat, and stats.")
@discord.app_commands.describe(user="Whose stash (defaults to you)")
async def stash_command(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user
    data = _load_dealer()
    record = data["users"].get(str(target.id))

    if not record or (not record.get("stash") and record.get("lifetime_sold", 0) == 0):
        await interaction.response.send_message(
            f"{target.mention} isn't in the game. `/buysupply` to start.",
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    _decay_heat(record)
    _save_dealer(data)
    prices = _get_market_prices()

    embed = discord.Embed(
        title=f"📦 {target.display_name}'s Stash",
        color=discord.Color.dark_purple(),
    )

    if record.get("stash"):
        lines = []
        total_value = 0
        for sub_key, grams in record["stash"].items():
            info = SUBSTANCES.get(sub_key, {"emoji":"❓","name":"?","stash_max":0})
            value = prices.get(sub_key, {}).get("sell", 0) * grams
            total_value += value
            lines.append(
                f"{info['emoji']} **{info['name']}** — {grams}g / {info['stash_max']}g "
                f"(worth ~{value:,})"
            )
        embed.add_field(name="📦 Inventory", value="\n".join(lines), inline=False)
        embed.add_field(name="💰 Total Value", value=f"~{total_value:,} coins", inline=True)
    else:
        embed.add_field(name="📦 Inventory", value="_Empty. Buy from `/buysupply`._", inline=False)

    heat = record.get("heat", 0)
    heat_emoji = "🟢" if heat < 30 else ("🟡" if heat < 60 else ("🟠" if heat < 90 else "🔴"))
    embed.add_field(name=f"{heat_emoji} Heat", value=f"**{heat}/100**", inline=True)
    embed.add_field(name="💀 Times Busted", value=str(record.get("times_busted", 0)), inline=True)
    embed.add_field(name="📈 Lifetime Profit", value=f"{record.get('lifetime_profit', 0):,}", inline=True)
    embed.add_field(name="📊 Lifetime Sold", value=f"{record.get('lifetime_sold', 0):,}g", inline=True)
    embed.set_footer(text="Use /streetprice to see today's market.")
    await interaction.response.send_message(embed=embed)


# ── /streetprice ─────────────────────────────────────────────────────────────
@tree.command(name="streetprice", description="See today's buy/sell market prices.")
async def streetprice_command(interaction: discord.Interaction):
    prices = _get_market_prices()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = []
    for sub_key, info in SUBSTANCES.items():
        p = prices.get(sub_key, {})
        buy = p.get("buy", info["base_buy"])
        sell = p.get("sell", info["base_sell"])
        margin = sell - buy
        margin_pct = int((margin / buy) * 100) if buy > 0 else 0
        # Indicator: how good today's margin is vs baseline
        base_margin_pct = int(((info["base_sell"] - info["base_buy"]) / info["base_buy"]) * 100)
        if margin_pct > base_margin_pct + 15:
            indicator = "📈"
        elif margin_pct < base_margin_pct - 15:
            indicator = "📉"
        else:
            indicator = "➖"
        lines.append(
            f"{info['emoji']} **{info['name']}** {indicator} — Buy: **{buy}/g** • Sell: **{sell}/g** "
            f"_(+{margin_pct}% margin, heat {info['heat']})_"
        )

    embed = discord.Embed(
        title="📰 STREET PRICES",
        description="\n".join(lines),
        color=discord.Color.dark_gold(),
    )
    embed.set_footer(text=f"Market updates daily • {today} UTC")
    await interaction.response.send_message(embed=embed)


# ── /laylow ──────────────────────────────────────────────────────────────────
@tree.command(name="laylow", description="Lay low for an hour to reduce heat faster.")
async def laylow_command(interaction: discord.Interaction):
    user = interaction.user
    data = _load_dealer()
    record = data["users"].get(str(user.id))
    if not record:
        await interaction.response.send_message("You're not in the game.", ephemeral=True)
        return
    _decay_heat(record)
    if record["heat"] < 20:
        await interaction.response.send_message(
            f"🟢 You're already cool. Heat: **{record['heat']}/100**.",
            ephemeral=True,
        )
        return
    # Costs 500 coins to lay low — reduces heat by 30
    LAYLOW_COST = 500
    if economy.balance(user.id) < LAYLOW_COST:
        await interaction.response.send_message(
            f"❌ Laying low costs **{LAYLOW_COST}** coins (cab fares, burner phone, etc).",
            ephemeral=True,
        )
        return
    economy.add(user.id, -LAYLOW_COST, "laylow")
    reduction = min(30, record["heat"])
    record["heat"] = max(0, record["heat"] - 30)
    _save_dealer(data)
    await interaction.response.send_message(
        f"🤫 You ditched the phone and went off the grid. Heat dropped by **{reduction}** to **{record['heat']}/100**.\n"
        f"💰 Cost: **{LAYLOW_COST}** coins."
    )


# ── /dealers — leaderboard ───────────────────────────────────────────────────
@tree.command(name="dealers", description="Top dealers by lifetime profit.")
async def dealers_command(interaction: discord.Interaction):
    data = _load_dealer()
    users = data.get("users", {})
    if not users:
        await interaction.response.send_message("No dealers in the game yet.", ephemeral=True)
        return

    ranked = sorted(
        users.items(),
        key=lambda x: x[1].get("lifetime_profit", 0),
        reverse=True,
    )

    medals = ["🥇", "🥈", "🥉"]
    lines = []
    for i, (uid, record) in enumerate(ranked[:10]):
        prefix = medals[i] if i < 3 else f"`#{i+1}`"
        try:
            member = interaction.guild.get_member(int(uid)) if interaction.guild else None
            name = member.display_name if member else f"User {uid}"
        except Exception:
            name = f"User {uid}"
        profit = record.get("lifetime_profit", 0)
        sold = record.get("lifetime_sold", 0)
        busts = record.get("times_busted", 0)
        lines.append(
            f"{prefix} **{name}** — {profit:,} profit • {sold:,}g sold • {busts} busts"
        )

    embed = discord.Embed(
        title="💊 TOP DEALERS",
        description="\n".join(lines),
        color=discord.Color.dark_purple(),
    )
    await interaction.response.send_message(embed=embed)


# ─────────────────────────────────────────────────────────────────────────────
# 📊 /dealer DASHBOARD
# Interactive embed with tabs: Overview, Market, Leaderboard, Server Stats.
# Quick action buttons for buy/sell/laylow without typing.
# ─────────────────────────────────────────────────────────────────────────────

def _build_dealer_overview_embed(user_id: int, display_name: str) -> discord.Embed:
    """Personal overview tab — stash, heat, lifetime stats."""
    data = _load_dealer()
    record = data["users"].get(str(user_id))
    prices = _get_market_prices()

    embed = discord.Embed(
        title=f"💊 {display_name}'s Dealer Dashboard — Overview",
        color=discord.Color.dark_purple(),
    )

    if not record:
        embed.description = (
            "_You haven't started dealing yet._\n\n"
            "Hit **Buy Supply** below or `/buysupply` to start.\n"
            "Check **Market** to see today's prices."
        )
        return embed

    _decay_heat(record)
    _save_dealer(data)

    # Stash
    if record.get("stash"):
        stash_lines = []
        total_value = 0
        total_grams = 0
        for sub_key, grams in record["stash"].items():
            info = SUBSTANCES.get(sub_key, {"emoji":"❓","name":"?","stash_max":0})
            value = prices.get(sub_key, {}).get("sell", 0) * grams
            total_value += value
            total_grams += grams
            stash_lines.append(
                f"{info['emoji']} **{info['name']}** — {grams}g / {info['stash_max']}g "
                f"(~{value:,})"
            )
        embed.add_field(name="📦 Stash", value="\n".join(stash_lines), inline=False)
        embed.add_field(name="💰 Stash Value", value=f"~{total_value:,} coins", inline=True)
        embed.add_field(name="⚖️ Total Weight", value=f"{total_grams}g", inline=True)
    else:
        embed.add_field(name="📦 Stash", value="_Empty — buy supply to start._", inline=False)

    # Heat status
    heat = record.get("heat", 0)
    if heat < 30:
        heat_emoji, heat_status = "🟢", "Safe"
    elif heat < 60:
        heat_emoji, heat_status = "🟡", "Watch yourself"
    elif heat < 90:
        heat_emoji, heat_status = "🟠", "Cops are sniffing"
    else:
        heat_emoji, heat_status = "🔴", "BUSTED on next sale"
    bar_len = 10
    filled = int(heat / 100 * bar_len)
    heat_bar = "█" * filled + "░" * (bar_len - filled)
    embed.add_field(
        name=f"{heat_emoji} Heat",
        value=f"`{heat_bar}` **{heat}/100**\n_{heat_status}_",
        inline=False,
    )

    # Cooldowns
    supply_cd = SUPPLY_COOLDOWN_MIN * 60 - (time.time() - record.get("last_supply", 0))
    sell_cd = SELL_COOLDOWN_MIN * 60 - (time.time() - record.get("last_sell", 0))
    cd_text = []
    cd_text.append("✅ Supply ready" if supply_cd <= 0 else f"⏰ Supply in {fmt_cooldown(int(supply_cd))}")
    cd_text.append("✅ Sell ready" if sell_cd <= 0 else f"⏰ Sell in {fmt_cooldown(int(sell_cd))}")
    embed.add_field(name="⏰ Cooldowns", value="\n".join(cd_text), inline=False)

    # Lifetime stats
    embed.add_field(name="📈 Lifetime Profit", value=f"{record.get('lifetime_profit', 0):,}", inline=True)
    embed.add_field(name="📊 Lifetime Sold", value=f"{record.get('lifetime_sold', 0):,}g", inline=True)
    embed.add_field(name="💀 Times Busted", value=str(record.get("times_busted", 0)), inline=True)

    embed.set_footer(text="Use the buttons below for quick actions.")
    return embed


def _build_dealer_market_embed() -> discord.Embed:
    """Market tab — today's buy/sell prices with trend indicators."""
    prices = _get_market_prices()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = []
    for sub_key, info in SUBSTANCES.items():
        p = prices.get(sub_key, {})
        buy = p.get("buy", info["base_buy"])
        sell = p.get("sell", info["base_sell"])
        margin = sell - buy
        margin_pct = int((margin / buy) * 100) if buy > 0 else 0
        base_margin_pct = int(((info["base_sell"] - info["base_buy"]) / info["base_buy"]) * 100)
        if margin_pct > base_margin_pct + 15:
            indicator = "📈 HOT"
        elif margin_pct < base_margin_pct - 15:
            indicator = "📉 COLD"
        else:
            indicator = "➖ avg"
        lines.append(
            f"{info['emoji']} **{info['name']}** {indicator}\n"
            f"   Buy: **{buy}/g** • Sell: **{sell}/g** • Margin: **+{margin_pct}%** • Heat: {info['heat']}"
        )

    embed = discord.Embed(
        title="📰 Market — Today's Street Prices",
        description="\n\n".join(lines),
        color=discord.Color.dark_gold(),
    )
    embed.set_footer(text=f"Updates daily at recap hour • {today} UTC")
    return embed


def _build_dealer_leaderboard_embed(guild) -> discord.Embed:
    """Leaderboard tab — top dealers by profit."""
    data = _load_dealer()
    users = data.get("users", {})

    embed = discord.Embed(
        title="🏆 Top Dealers — Lifetime Profit",
        color=discord.Color.gold(),
    )

    if not users:
        embed.description = "_No dealers in the game yet._"
        return embed

    ranked = sorted(
        users.items(),
        key=lambda x: x[1].get("lifetime_profit", 0),
        reverse=True,
    )

    medals = ["🥇", "🥈", "🥉"]
    lines = []
    for i, (uid, record) in enumerate(ranked[:10]):
        prefix = medals[i] if i < 3 else f"`#{i+1}`"
        try:
            member = guild.get_member(int(uid)) if guild else None
            name = member.display_name if member else f"User {uid}"
        except Exception:
            name = f"User {uid}"
        profit = record.get("lifetime_profit", 0)
        sold = record.get("lifetime_sold", 0)
        busts = record.get("times_busted", 0)
        lines.append(
            f"{prefix} **{name}** — **{profit:,}** profit\n"
            f"   {sold:,}g sold • {busts} busts"
        )
    embed.description = "\n\n".join(lines)
    return embed


def _build_dealer_server_stats_embed() -> discord.Embed:
    """Server-wide dealer stats."""
    data = _load_dealer()
    users = data.get("users", {})

    embed = discord.Embed(
        title="📊 Server Dealer Stats",
        color=discord.Color.dark_teal(),
    )

    if not users:
        embed.description = "_No dealers in the game yet._"
        return embed

    total_dealers = len(users)
    active_dealers = sum(1 for r in users.values() if r.get("stash") or r.get("lifetime_sold", 0) > 0)
    total_profit = sum(r.get("lifetime_profit", 0) for r in users.values())
    total_grams = sum(r.get("lifetime_sold", 0) for r in users.values())
    total_busts = sum(r.get("times_busted", 0) for r in users.values())
    bust_rate = (total_busts / max(1, total_grams)) * 1000  # busts per 1000g sold

    # Calculate which substance is most popular by lifetime sold
    substance_sold = {}
    for r in users.values():
        for sub_key, grams in r.get("stash", {}).items():
            substance_sold[sub_key] = substance_sold.get(sub_key, 0) + grams
    # That's just current stash; better to track lifetime — for now show stash
    most_in_stash = max(substance_sold.items(), key=lambda x: x[1], default=(None, 0))

    # Currently hot product (best market margin today)
    prices = _get_market_prices()
    best_margin = None
    best_margin_pct = -1
    for sub_key, p in prices.items():
        info = SUBSTANCES.get(sub_key, {})
        margin = p.get("sell", 0) - p.get("buy", 0)
        pct = int((margin / max(1, p.get("buy", 1))) * 100)
        if pct > best_margin_pct:
            best_margin_pct = pct
            best_margin = (sub_key, info, pct)

    embed.add_field(name="👥 Total Dealers", value=str(total_dealers), inline=True)
    embed.add_field(name="🎯 Active", value=str(active_dealers), inline=True)
    embed.add_field(name="💀 Total Busts", value=str(total_busts), inline=True)
    embed.add_field(name="📈 Server Lifetime Profit", value=f"{total_profit:,} coins", inline=False)
    embed.add_field(name="⚖️ Server Lifetime Weight", value=f"{total_grams:,}g", inline=True)
    embed.add_field(name="🚨 Bust Rate", value=f"{bust_rate:.1f} per 1000g", inline=True)

    if best_margin:
        sub_key, info, pct = best_margin
        embed.add_field(
            name="🔥 Today's Best Margin",
            value=f"{info.get('emoji','💊')} **{info.get('name', sub_key)}** (+{pct}%)",
            inline=False,
        )
    if most_in_stash[0]:
        info = SUBSTANCES.get(most_in_stash[0], {})
        embed.add_field(
            name="📦 Most In Stash (Server)",
            value=f"{info.get('emoji','💊')} **{info.get('name', most_in_stash[0])}** — {most_in_stash[1]:,}g",
            inline=False,
        )

    return embed


class DealerDashboardView(discord.ui.View):
    """Dashboard view with tab switching + quick action buttons."""

    def __init__(self, user_id: int, tab: str = "overview"):
        super().__init__(timeout=240)
        self.user_id = user_id
        self.tab = tab
        # Tab buttons (row 0)
        for label, key, emoji in [
            ("Overview", "overview", "👤"),
            ("Market", "market", "📰"),
            ("Leaderboard", "leaderboard", "🏆"),
            ("Server", "server", "📊"),
        ]:
            btn = discord.ui.Button(
                label=label,
                emoji=emoji,
                style=discord.ButtonStyle.primary if key == tab else discord.ButtonStyle.secondary,
                row=0,
            )
            btn.callback = self._make_tab_callback(key)
            self.add_item(btn)

        # Quick action buttons (row 1) — only shown on Overview tab
        if tab == "overview":
            buy_btn = discord.ui.Button(label="Buy Supply", emoji="📦", style=discord.ButtonStyle.success, row=1)
            buy_btn.callback = self._buy_supply_callback
            self.add_item(buy_btn)
            sell_btn = discord.ui.Button(label="Sell", emoji="💰", style=discord.ButtonStyle.success, row=1)
            sell_btn.callback = self._sell_callback
            self.add_item(sell_btn)
            laylow_btn = discord.ui.Button(label="Lay Low", emoji="🤫", style=discord.ButtonStyle.secondary, row=1)
            laylow_btn.callback = self._laylow_callback
            self.add_item(laylow_btn)
            refresh_btn = discord.ui.Button(label="Refresh", emoji="🔄", style=discord.ButtonStyle.secondary, row=1)
            refresh_btn.callback = self._refresh_callback
            self.add_item(refresh_btn)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not your dashboard.", ephemeral=True)
            return False
        return True

    def _make_tab_callback(self, tab_key: str):
        async def cb(interaction: discord.Interaction):
            member = interaction.guild.get_member(self.user_id) if interaction.guild else None
            display_name = member.display_name if member else interaction.user.display_name

            if tab_key == "overview":
                embed = _build_dealer_overview_embed(self.user_id, display_name)
            elif tab_key == "market":
                embed = _build_dealer_market_embed()
            elif tab_key == "leaderboard":
                embed = _build_dealer_leaderboard_embed(interaction.guild)
            elif tab_key == "server":
                embed = _build_dealer_server_stats_embed()
            else:
                embed = _build_dealer_overview_embed(self.user_id, display_name)
            view = DealerDashboardView(self.user_id, tab=tab_key)
            await interaction.response.edit_message(embed=embed, view=view)
        return cb

    async def _buy_supply_callback(self, interaction: discord.Interaction):
        # Open a buy modal
        await interaction.response.send_modal(DealerBuyModal(self.user_id))

    async def _sell_callback(self, interaction: discord.Interaction):
        # Check they have something to sell
        data = _load_dealer()
        record = data["users"].get(str(self.user_id))
        if not record or not record.get("stash"):
            await interaction.response.send_message(
                "❌ Your stash is empty. Use **Buy Supply** first.", ephemeral=True
            )
            return
        await interaction.response.send_modal(DealerSellModal(self.user_id, record))

    async def _laylow_callback(self, interaction: discord.Interaction):
        # Inline call to the laylow logic
        user_id = self.user_id
        data = _load_dealer()
        record = data["users"].get(str(user_id))
        if not record:
            await interaction.response.send_message("You're not in the game yet.", ephemeral=True)
            return
        _decay_heat(record)
        if record["heat"] < 20:
            await interaction.response.send_message(
                f"🟢 You're already cool. Heat: **{record['heat']}/100**.", ephemeral=True
            )
            return
        LAYLOW_COST = 500
        if economy.balance(user_id) < LAYLOW_COST:
            await interaction.response.send_message(
                f"❌ Laying low costs **{LAYLOW_COST}** coins.", ephemeral=True
            )
            return
        economy.add(user_id, -LAYLOW_COST, "laylow")
        reduction = min(30, record["heat"])
        record["heat"] = max(0, record["heat"] - 30)
        _save_dealer(data)
        await interaction.response.send_message(
            f"🤫 Heat dropped by **{reduction}** to **{record['heat']}/100**. (-500 coins)",
            ephemeral=True,
        )

    async def _refresh_callback(self, interaction: discord.Interaction):
        member = interaction.guild.get_member(self.user_id) if interaction.guild else None
        display_name = member.display_name if member else interaction.user.display_name
        embed = _build_dealer_overview_embed(self.user_id, display_name)
        view = DealerDashboardView(self.user_id, tab="overview")
        await interaction.response.edit_message(embed=embed, view=view)


class DealerBuyModal(discord.ui.Modal, title="📦 Buy Supply"):
    substance_input = discord.ui.TextInput(
        label="Substance",
        placeholder="weed, shrooms, molly, addy, ket, coke, meff, fent",
        required=True,
        max_length=20,
    )
    grams_input = discord.ui.TextInput(
        label="Grams",
        placeholder="How many grams to buy (1-500)",
        required=True,
        max_length=5,
    )

    def __init__(self, user_id: int):
        super().__init__()
        self.user_id = user_id

    async def on_submit(self, interaction: discord.Interaction):
        sub_key = str(self.substance_input).strip().lower()
        if sub_key not in SUBSTANCES:
            await interaction.response.send_message(
                f"❌ Unknown substance. Try: {', '.join(SUBSTANCES.keys())}",
                ephemeral=True,
            )
            return
        try:
            grams = int(str(self.grams_input).strip())
        except ValueError:
            await interaction.response.send_message("❌ Grams must be a number.", ephemeral=True)
            return
        if grams <= 0 or grams > 500:
            await interaction.response.send_message("❌ Grams must be 1–500.", ephemeral=True)
            return

        info = SUBSTANCES[sub_key]
        data = _load_dealer()
        uid = str(self.user_id)
        record = data["users"].setdefault(uid, {
            "stash": {}, "heat": 0, "last_supply": 0, "last_sell": 0,
            "last_heat_decay": time.time(), "lifetime_sold": 0,
            "lifetime_profit": 0, "times_busted": 0,
        })

        cd_remaining = SUPPLY_COOLDOWN_MIN * 60 - (time.time() - record.get("last_supply", 0))
        if cd_remaining > 0:
            await interaction.response.send_message(
                f"⏰ The plug isn't picking up. Try again in **{fmt_cooldown(int(cd_remaining))}**.",
                ephemeral=True,
            )
            return

        current_grams = record["stash"].get(sub_key, 0)
        if current_grams + grams > info["stash_max"]:
            room = info["stash_max"] - current_grams
            await interaction.response.send_message(
                f"❌ Stash too full. Room for **{room}g** more {info['name']}.", ephemeral=True
            )
            return

        prices = _get_market_prices()
        buy_price = prices[sub_key]["buy"]
        total_cost = buy_price * grams

        if economy.balance(self.user_id) < total_cost:
            await interaction.response.send_message(
                f"❌ Need **{total_cost:,}** coins. You have **{economy.balance(self.user_id):,}**.",
                ephemeral=True,
            )
            return

        economy.add(self.user_id, -total_cost, f"supply: {sub_key}")
        record["stash"][sub_key] = current_grams + grams
        record["last_supply"] = time.time()
        _save_dealer(data)

        await interaction.response.send_message(
            f"📦 Bought **{grams}g of {info['emoji']} {info['name']}** for **{total_cost:,}** coins.\n"
            f"Stash: **{current_grams + grams}g / {info['stash_max']}g** • Balance: **{economy.balance(self.user_id):,}**",
            ephemeral=True,
        )


class DealerSellModal(discord.ui.Modal, title="💰 Sell Product"):
    substance_input = discord.ui.TextInput(
        label="Substance",
        placeholder="weed, shrooms, molly, etc.",
        required=True,
        max_length=20,
    )
    grams_input = discord.ui.TextInput(
        label="Grams to sell",
        placeholder="How many grams",
        required=True,
        max_length=5,
    )

    def __init__(self, user_id: int, record: dict):
        super().__init__()
        self.user_id = user_id
        # Pre-fill substance with the largest item in stash if possible
        if record.get("stash"):
            biggest = max(record["stash"].items(), key=lambda x: x[1])
            self.substance_input.default = biggest[0]

    async def on_submit(self, interaction: discord.Interaction):
        sub_key = str(self.substance_input).strip().lower()
        if sub_key not in SUBSTANCES:
            await interaction.response.send_message(
                f"❌ Unknown substance. Try: {', '.join(SUBSTANCES.keys())}", ephemeral=True
            )
            return
        try:
            grams = int(str(self.grams_input).strip())
        except ValueError:
            await interaction.response.send_message("❌ Grams must be a number.", ephemeral=True)
            return
        if grams <= 0:
            await interaction.response.send_message("❌ Grams must be positive.", ephemeral=True)
            return

        info = SUBSTANCES[sub_key]
        data = _load_dealer()
        uid = str(self.user_id)
        if uid not in data["users"]:
            await interaction.response.send_message("❌ No stash to sell from.", ephemeral=True)
            return
        record = data["users"][uid]
        _decay_heat(record)

        cd_remaining = SELL_COOLDOWN_MIN * 60 - (time.time() - record.get("last_sell", 0))
        if cd_remaining > 0:
            await interaction.response.send_message(
                f"⏰ Customers haven't lined up yet. Try again in **{fmt_cooldown(int(cd_remaining))}**.",
                ephemeral=True,
            )
            return

        current_grams = record["stash"].get(sub_key, 0)
        if current_grams < grams:
            await interaction.response.send_message(
                f"❌ You only have **{current_grams}g** of {info['name']}.", ephemeral=True
            )
            return

        if record["heat"] >= HEAT_MAX:
            # Forced bust
            await self._bust_via_modal(interaction, record, data, forced=True)
            return

        # Roll for cops
        cop_chance = info["heat"] + (record["heat"] / 5)
        if random.random() * 100 < cop_chance:
            await self._bust_via_modal(interaction, record, data, forced=False)
            return

        # Successful sale
        prices = _get_market_prices()
        sell_price = prices[sub_key]["sell"]
        total = sell_price * grams
        record["stash"][sub_key] = current_grams - grams
        if record["stash"][sub_key] == 0:
            del record["stash"][sub_key]
        record["last_sell"] = time.time()
        heat_gain = max(1, int(info["heat"] * (grams / 10)))
        record["heat"] = min(HEAT_MAX, record["heat"] + heat_gain)
        record["lifetime_sold"] = record.get("lifetime_sold", 0) + grams
        record["lifetime_profit"] = record.get("lifetime_profit", 0) + total
        _save_dealer(data)
        new_bal = economy.add(self.user_id, total, f"sold {sub_key}")
        track_quest_progress(self.user_id, "coins_earned", total)
        add_tournament_score(self.user_id, coins_earned=total)
        await trigger_balance_check(self.user_id, channel=interaction.channel)

        npcs = [
            "a college kid", "a Wall Street guy", "a tired nurse",
            "a sketchy dude in a hoodie", "a soccer mom in a Range Rover",
            "an Uber driver on break", "a sad-looking lawyer",
        ]
        await interaction.response.send_message(
            f"💰 Sold **{grams}g of {info['emoji']} {info['name']}** to {random.choice(npcs)} for **{total:,}** coins.\n"
            f"🔥 Heat: **{record['heat']}/100** (+{heat_gain}) • Balance: **{new_bal:,}**",
            ephemeral=True,
        )

    async def _bust_via_modal(self, interaction, record, data, forced=False):
        stash_grams = sum(record["stash"].values())
        record["stash"] = {}
        record["heat"] = 0
        record["times_busted"] = record.get("times_busted", 0) + 1
        record["last_sell"] = time.time()
        bal = economy.balance(self.user_id)
        fine = int(bal * BUST_FINE_PCT)
        economy.add(self.user_id, -fine, "cop fine")
        _save_dealer(data)

        title = "🚨 BUSTED — HEAT MAXED OUT" if forced else "🚨 BUSTED"
        flavor = [
            "Undercover cop.",
            "Some snitch ratted you out.",
            "Patrol unit at the wrong moment.",
            "You sold to a narc. Rookie mistake.",
            "Surveillance had been tailing you.",
        ]
        await interaction.response.send_message(
            f"# {title}\n\n"
            f"{random.choice(flavor)}\n\n"
            f"💸 Lost **{stash_grams}g**. Fine: **{fine:,}** coins ({int(BUST_FINE_PCT*100)}% of balance).\n"
            f"Heat reset to **0**. Balance: **{economy.balance(self.user_id):,}**",
            ephemeral=True,
        )


@tree.command(name="dealer", description="Open the dealer dashboard — full overview, market, leaderboard, server stats.")
async def dealer_command(interaction: discord.Interaction):
    user = interaction.user
    embed = _build_dealer_overview_embed(user.id, user.display_name)
    view = DealerDashboardView(user.id, tab="overview")
    await interaction.response.send_message(embed=embed, view=view)


# ─────────────────────────────────────────────────────────────────────────────
# 🍸 NIGHTLIFE EMPIRE
# Own bars and nightclubs. Stock liquor. Hire staff (NPCs). Throw events.
# Customers are NPCs. Risk events: raids, brawls. Celebrity nights = bonus.
# ─────────────────────────────────────────────────────────────────────────────
NIGHTLIFE_FILE = MEMORY_DIR / "nightlife.json"

VENUE_TYPES = {
    "dive_bar": {
        "emoji": "🍺", "name": "Dive Bar",
        "cost": 5_000, "income_per_hour": 80, "max_staff": 2,
        "tier": 1, "desc": "Low overhead, blue-collar regulars.",
    },
    "sports_bar": {
        "emoji": "📺", "name": "Sports Bar",
        "cost": 20_000, "income_per_hour": 250, "max_staff": 3,
        "tier": 1, "desc": "Game-day crowds. Wings & beer.",
    },
    "cocktail_lounge": {
        "emoji": "🍸", "name": "Cocktail Lounge",
        "cost": 45_000, "income_per_hour": 500, "max_staff": 4,
        "tier": 2, "desc": "Upscale vibe. Bottle service.",
    },
    "speakeasy": {
        "emoji": "🥃", "name": "Speakeasy",
        "cost": 80_000, "income_per_hour": 850, "max_staff": 4,
        "tier": 2, "desc": "Hidden door. Velvet booths.",
    },
    "nightclub": {
        "emoji": "🎵", "name": "Nightclub",
        "cost": 200_000, "income_per_hour": 2_000, "max_staff": 6,
        "tier": 3, "desc": "Lines around the block. Bottle wars.",
    },
    "rooftop": {
        "emoji": "🌃", "name": "Rooftop Lounge",
        "cost": 400_000, "income_per_hour": 3_500, "max_staff": 7,
        "tier": 4, "desc": "City views. Influencers everywhere.",
    },
    "mega_club": {
        "emoji": "💎", "name": "Mega Club",
        "cost": 750_000, "income_per_hour": 6_000, "max_staff": 10,
        "tier": 5, "desc": "Multi-floor. International DJs. Real money.",
    },
}

STAFF_ROLES = {
    "bartender":  {"emoji": "🍹", "name": "Bartender",   "cost": 2_000, "income_boost": 0.15, "desc": "Faster pours, more sales."},
    "bouncer":    {"emoji": "💪", "name": "Bouncer",     "cost": 2_500, "income_boost": 0.10, "desc": "Reduces brawl chance.", "risk_reduction": 0.5},
    "dj":         {"emoji": "🎧", "name": "DJ",          "cost": 5_000, "income_boost": 0.25, "desc": "Big crowds love bangers."},
    "promoter":   {"emoji": "📣", "name": "Promoter",    "cost": 3_500, "income_boost": 0.20, "desc": "Brings the night to life."},
}

LIQUOR_TYPES = {
    "well":    {"emoji": "🍺", "name": "Well Liquor",  "cost_per_bottle": 50,  "revenue_per_bottle": 120,  "tier_min": 1},
    "premium": {"emoji": "🍷", "name": "Premium",      "cost_per_bottle": 200, "revenue_per_bottle": 500,  "tier_min": 1},
    "top_shelf": {"emoji": "🥃", "name": "Top Shelf",  "cost_per_bottle": 500, "revenue_per_bottle": 1_400, "tier_min": 2},
    "bottle_service": {"emoji": "🍾", "name": "Bottle Service", "cost_per_bottle": 1_500, "revenue_per_bottle": 5_000, "tier_min": 3},
}

NIGHTLIFE_COLLECT_COOLDOWN_MIN = 60  # 1 hour cooldown on collecting per venue
NIGHTLIFE_MAX_IDLE_HOURS = 18
NIGHTLIFE_RISK_INTERVAL_HOURS = 3  # how often to roll for events
NIGHTLIFE_RISK_CHANCE = 0.10       # 10% chance per venue per roll


def _load_nightlife() -> dict:
    if NIGHTLIFE_FILE.exists():
        try:
            with open(NIGHTLIFE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"users": {}}


def _save_nightlife(data: dict):
    with open(NIGHTLIFE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _user_venues(user_id: int) -> list:
    return _load_nightlife()["users"].get(str(user_id), [])


def _venue_cost(user_id: int, venue_type: str) -> int:
    info = VENUE_TYPES[venue_type]
    base = info["cost"]
    same_tier_owned = sum(
        1 for v in _user_venues(user_id)
        if VENUE_TYPES.get(v["type"], {}).get("tier") == info["tier"]
    )
    return int(base * (1.5 ** same_tier_owned))


def _venue_income_per_hour(venue: dict) -> int:
    info = VENUE_TYPES.get(venue["type"])
    if not info:
        return 0
    base = info["income_per_hour"]
    boost = 1.0
    for staff_role in venue.get("staff", []):
        role_info = STAFF_ROLES.get(staff_role, {})
        boost += role_info.get("income_boost", 0)
    # Celebrity night triple income
    if venue.get("celebrity_until", 0) > time.time():
        boost *= 3
    return int(base * boost)


def _venue_pending_income(venue: dict) -> int:
    now = time.time()
    last_collected = venue.get("last_collected", venue.get("purchased_at", now))
    closed_until = venue.get("closed_until", 0)
    # Don't earn during closure
    effective_start = max(last_collected, closed_until) if closed_until > last_collected else last_collected
    hours = (now - effective_start) / 3600
    if hours <= 0:
        return 0
    hours = min(hours, NIGHTLIFE_MAX_IDLE_HOURS)
    return int(_venue_income_per_hour(venue) * hours)


# ── /buyvenue ────────────────────────────────────────────────────────────────
@tree.command(name="buyvenue", description="Buy a nightclub or bar. Earn passive income from drinks & cover.")
@discord.app_commands.describe(venue_type="Which venue type")
@discord.app_commands.choices(
    venue_type=[
        discord.app_commands.Choice(
            name=f"{v['emoji']} {v['name']} — {v['cost']:,} coins ({v['income_per_hour']}/hr)",
            value=k,
        )
        for k, v in VENUE_TYPES.items()
    ]
)
async def buyvenue_command(
    interaction: discord.Interaction,
    venue_type: discord.app_commands.Choice[str],
):
    user = interaction.user
    v_type = venue_type.value
    if v_type not in VENUE_TYPES:
        await interaction.response.send_message("Invalid venue type.", ephemeral=True)
        return

    cost = _venue_cost(user.id, v_type)
    if economy.balance(user.id) < cost:
        await interaction.response.send_message(
            f"❌ Need **{cost:,}** coins (scales with same-tier ownership). You have **{economy.balance(user.id):,}**.",
            ephemeral=True,
        )
        return

    info = VENUE_TYPES[v_type]
    economy.add(user.id, -cost, f"venue: {v_type}")

    data = _load_nightlife()
    venues = data["users"].setdefault(str(user.id), [])
    new_venue = {
        "id": f"{v_type}_{int(time.time())}_{random.randint(1000,9999)}",
        "type": v_type,
        "name": f"{info['name']} #{len([v for v in venues if v['type']==v_type])+1}",
        "purchased_at": time.time(),
        "last_collected": time.time(),
        "staff": [],
        "liquor": {},          # liquor_type -> bottles stocked
        "closed_until": 0,
        "celebrity_until": 0,
        "lifetime_revenue": 0,
        "last_risk_roll": time.time(),
    }
    venues.append(new_venue)
    _save_nightlife(data)

    await interaction.response.send_message(
        f"# {info['emoji']} VENUE ACQUIRED\n\n"
        f"You opened **{new_venue['name']}** for **{cost:,}** coins.\n"
        f"💰 Income: **{info['income_per_hour']:,}**/hr\n"
        f"👥 Max staff: **{info['max_staff']}**\n"
        f"📋 _{info['desc']}_\n\n"
        f"`/venues` to manage, `/collectvenue` to claim earnings.\n"
        f"`/hirestaff` to boost income, `/stockliquor` for bigger margins."
    )


# ── /venues ──────────────────────────────────────────────────────────────────
@tree.command(name="venues", description="View your nightlife portfolio.")
@discord.app_commands.describe(user="Whose venues (defaults to you)")
async def venues_command(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user
    venues = _user_venues(target.id)
    if not venues:
        await interaction.response.send_message(
            f"{target.mention} doesn't own any venues. Try `/buyvenue`.",
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    embed = discord.Embed(
        title=f"🌃 {target.display_name}'s Nightlife Empire",
        color=discord.Color.dark_magenta(),
    )

    total_hourly = 0
    total_pending = 0
    for venue in venues:
        info = VENUE_TYPES.get(venue["type"], {"emoji":"🏢","name":"?","max_staff":0})
        hourly = _venue_income_per_hour(venue)
        pending = _venue_pending_income(venue)
        staff = venue.get("staff", [])
        if venue.get("closed_until", 0) > time.time():
            status = "🚨 CLOSED"
        elif venue.get("celebrity_until", 0) > time.time():
            status = "⭐ CELEBRITY NIGHT (3x)"
        else:
            status = "✅ Open"
        embed.add_field(
            name=f"{info['emoji']} {venue['name']}",
            value=(
                f"Status: {status}\n"
                f"Income: **{hourly:,}**/hr\n"
                f"Staff: {len(staff)}/{info['max_staff']}\n"
                f"Pending: **{pending:,}** coins\n"
                f"Lifetime: {venue.get('lifetime_revenue', 0):,}"
            ),
            inline=True,
        )
        total_hourly += hourly
        total_pending += pending

    embed.add_field(
        name="📊 TOTALS",
        value=f"Hourly: **{total_hourly:,}** | Pending: **{total_pending:,}**",
        inline=False,
    )
    await interaction.response.send_message(embed=embed)


# ── /collectvenue ────────────────────────────────────────────────────────────
@tree.command(name="collectvenue", description="Collect revenue from your venues.")
async def collectvenue_command(interaction: discord.Interaction):
    user = interaction.user
    data = _load_nightlife()
    venues = data["users"].get(str(user.id), [])
    if not venues:
        await interaction.response.send_message("You don't own any venues.", ephemeral=True)
        return

    total = 0
    lines = []
    for venue in venues:
        info = VENUE_TYPES.get(venue["type"], {"emoji":"🏢","name":"?"})
        pending = _venue_pending_income(venue)
        if pending <= 0:
            continue
        # Add liquor sales bonus based on stocked bottles
        liquor_bonus = 0
        for liq_key, bottles in list(venue.get("liquor", {}).items()):
            if bottles <= 0:
                continue
            liq_info = LIQUOR_TYPES.get(liq_key)
            if not liq_info:
                continue
            # Sell up to 20% of stock per collection (with a sane cap)
            sold = min(bottles, max(1, int(bottles * 0.2)))
            liquor_bonus += sold * (liq_info["revenue_per_bottle"] - liq_info["cost_per_bottle"])
            venue["liquor"][liq_key] = bottles - sold

        revenue = pending + liquor_bonus
        venue["last_collected"] = time.time()
        venue["lifetime_revenue"] = venue.get("lifetime_revenue", 0) + revenue
        total += revenue
        liquor_text = f" + {liquor_bonus:,} liquor" if liquor_bonus > 0 else ""
        lines.append(f"{info['emoji']} **{venue['name']}** — {pending:,}{liquor_text} = **{revenue:,}**")

    _save_nightlife(data)
    if total <= 0:
        await interaction.response.send_message(
            "💤 No revenue yet. Be patient — venues take time.", ephemeral=True
        )
        return

    new_bal = economy.add(user.id, total, "venue collect")
    track_economy_event("earned", total)
    track_feature_use("venue")
    track_activity("venue_collect", user.id, user.display_name, f"collected {total:,} from venues")
    track_quest_progress(user.id, "coins_earned", total)
    add_tournament_score(user.id, coins_earned=total)
    await trigger_balance_check(user.id, channel=interaction.channel)

    response = (
        f"# 🍾 NIGHT'S TAKE\n\n"
        + "\n".join(lines)
        + f"\n\n💰 **Total: {total:,}**\nBalance: **{new_bal:,}**"
    )
    if len(response) > 2000:
        response = response[:1990] + "..."
    await interaction.response.send_message(response)


# ── /hirestaff ───────────────────────────────────────────────────────────────
@tree.command(name="hirestaff", description="Hire a staff role at one of your venues.")
@discord.app_commands.describe(venue_type="Which venue type to staff", role="Staff role")
@discord.app_commands.choices(
    venue_type=[
        discord.app_commands.Choice(name=f"{v['emoji']} {v['name']}", value=k)
        for k, v in VENUE_TYPES.items()
    ],
    role=[
        discord.app_commands.Choice(
            name=f"{r['emoji']} {r['name']} — {r['cost']:,} (+{int(r['income_boost']*100)}%)",
            value=k,
        )
        for k, r in STAFF_ROLES.items()
    ]
)
async def hirestaff_command(
    interaction: discord.Interaction,
    venue_type: discord.app_commands.Choice[str],
    role: discord.app_commands.Choice[str],
):
    user = interaction.user
    v_type = venue_type.value
    r_key = role.value
    if r_key not in STAFF_ROLES:
        await interaction.response.send_message("Invalid role.", ephemeral=True)
        return

    role_info = STAFF_ROLES[r_key]
    if economy.balance(user.id) < role_info["cost"]:
        await interaction.response.send_message(
            f"❌ Hiring a {role_info['name']} costs **{role_info['cost']:,}** coins.", ephemeral=True
        )
        return

    data = _load_nightlife()
    venues = data["users"].get(str(user.id), [])
    # Find first venue of this type with room
    info = VENUE_TYPES.get(v_type, {})
    biz = None
    for v in venues:
        if v["type"] == v_type and len(v.get("staff", [])) < info.get("max_staff", 0):
            biz = v
            break

    if not biz:
        await interaction.response.send_message(
            f"❌ No {info.get('name', v_type)} with open staff slots.", ephemeral=True
        )
        return

    economy.add(user.id, -role_info["cost"], f"hire {r_key}")
    biz.setdefault("staff", []).append(r_key)
    _save_nightlife(data)
    await interaction.response.send_message(
        f"👥 Hired a {role_info['emoji']} **{role_info['name']}** at **{biz['name']}**!\n"
        f"💰 Income now: **{_venue_income_per_hour(biz):,}/hr**"
    )


# ── /stockliquor ─────────────────────────────────────────────────────────────
@tree.command(name="stockliquor", description="Stock liquor at one of your venues for bonus revenue.")
@discord.app_commands.describe(venue_type="Which venue type", liquor="Which liquor", bottles="How many bottles")
@discord.app_commands.choices(
    venue_type=[
        discord.app_commands.Choice(name=f"{v['emoji']} {v['name']}", value=k)
        for k, v in VENUE_TYPES.items()
    ],
    liquor=[
        discord.app_commands.Choice(
            name=f"{l['emoji']} {l['name']} — {l['cost_per_bottle']:,}/bottle",
            value=k,
        )
        for k, l in LIQUOR_TYPES.items()
    ]
)
async def stockliquor_command(
    interaction: discord.Interaction,
    venue_type: discord.app_commands.Choice[str],
    liquor: discord.app_commands.Choice[str],
    bottles: int,
):
    user = interaction.user
    v_type = venue_type.value
    l_key = liquor.value
    if l_key not in LIQUOR_TYPES:
        await interaction.response.send_message("Invalid liquor.", ephemeral=True)
        return
    if bottles <= 0 or bottles > 200:
        await interaction.response.send_message("❌ Bottles must be 1–200.", ephemeral=True)
        return

    liq_info = LIQUOR_TYPES[l_key]
    venue_info = VENUE_TYPES.get(v_type, {})

    # Check tier requirement
    if liq_info["tier_min"] > venue_info.get("tier", 1):
        await interaction.response.send_message(
            f"❌ **{liq_info['name']}** requires a Tier {liq_info['tier_min']}+ venue. "
            f"This is Tier {venue_info.get('tier', 1)}.",
            ephemeral=True,
        )
        return

    data = _load_nightlife()
    venues = data["users"].get(str(user.id), [])
    biz = next((v for v in venues if v["type"] == v_type), None)
    if not biz:
        await interaction.response.send_message(
            f"❌ You don't own a {venue_info.get('name', v_type)}.", ephemeral=True
        )
        return

    total_cost = liq_info["cost_per_bottle"] * bottles
    if economy.balance(user.id) < total_cost:
        await interaction.response.send_message(
            f"❌ Need **{total_cost:,}** coins. You have **{economy.balance(user.id):,}**.",
            ephemeral=True,
        )
        return

    economy.add(user.id, -total_cost, f"liquor: {l_key}")
    biz.setdefault("liquor", {})[l_key] = biz["liquor"].get(l_key, 0) + bottles
    _save_nightlife(data)

    expected_revenue = bottles * liq_info["revenue_per_bottle"]
    expected_profit = expected_revenue - total_cost
    await interaction.response.send_message(
        f"🍾 Stocked **{bottles} bottles** of {liq_info['emoji']} **{liq_info['name']}** at **{biz['name']}**.\n"
        f"💸 Cost: {total_cost:,} • Expected revenue: ~{expected_revenue:,} (+{expected_profit:,} profit)\n"
        f"_Bottles sell automatically when you `/collectvenue`._"
    )


# ── Risk events scheduler ────────────────────────────────────────────────────
async def nightlife_events_scheduler():
    """Random nightlife events: liquor raid, brawl, celebrity night."""
    await client.wait_until_ready()
    while not client.is_closed():
        await asyncio.sleep(NIGHTLIFE_RISK_INTERVAL_HOURS * 3600)
        try:
            data = _load_nightlife()
            cfg = load_config()
            recap_channel_id = get_notification_channel_id(cfg)
            channel = client.get_channel(int(recap_channel_id)) if recap_channel_id else None

            for uid, venues in data["users"].items():
                for venue in venues:
                    if venue.get("closed_until", 0) > time.time():
                        continue
                    if random.random() > NIGHTLIFE_RISK_CHANCE:
                        continue
                    info = VENUE_TYPES.get(venue["type"], {"emoji":"🏢","name":"?"})

                    # Check for bouncer risk reduction
                    has_bouncer = "bouncer" in venue.get("staff", [])
                    # Pick an event weighted by venue + staff
                    events = [
                        ("🚨", "got raided — liquor license violation", 6, True),  # closes
                        ("🥊", "had a brawl break out", 3, True),                    # closes (bouncer reduces)
                        ("📋", "got a noise complaint visit", 2, False),             # no close, no celebrity
                        ("⭐", "CELEBRITY in the building!", 0, False),              # 3x income for 6h
                    ]
                    event = random.choice(events)
                    emoji, desc, close_hours, is_brawl = event

                    if "CELEBRITY" in desc:
                        venue["celebrity_until"] = time.time() + 6 * 3600
                        if channel:
                            try:
                                await channel.send(
                                    f"{emoji} **<@{uid}>'s {info['emoji']} {venue['name']}** had a {desc}\n"
                                    f"💰 **3x income** for the next **6 hours**!",
                                    allowed_mentions=discord.AllowedMentions.none(),
                                )
                            except Exception:
                                pass
                        # DM owner
                        try:
                            await send_dm(int(uid), "business",
                                content=f"⭐ A celebrity walked into your {venue['name']}! 3x income for 6 hours.")
                        except Exception:
                            pass
                    elif is_brawl and has_bouncer and random.random() < 0.5:
                        # Bouncer prevented it
                        if channel:
                            try:
                                await channel.send(
                                    f"💪 **<@{uid}>'s {info['emoji']} {venue['name']}**: a brawl was about to break out, "
                                    f"but the bouncer shut it down. No damage.",
                                    allowed_mentions=discord.AllowedMentions.none(),
                                )
                            except Exception:
                                pass
                    else:
                        venue["closed_until"] = time.time() + close_hours * 3600
                        if channel:
                            try:
                                await channel.send(
                                    f"{emoji} **<@{uid}>'s {info['emoji']} {venue['name']}** {desc}!\n"
                                    f"⏰ Closed for **{close_hours} hours**.",
                                    allowed_mentions=discord.AllowedMentions.none(),
                                )
                            except Exception:
                                pass
                        try:
                            await send_dm(int(uid), "business",
                                content=f"🚨 Your {venue['name']} {desc}. Closed for {close_hours} hours.")
                        except Exception:
                            pass
                    venue["last_risk_roll"] = time.time()
            _save_nightlife(data)
        except Exception as e:
            log.exception("nightlife_events_scheduler: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# 🛠️ /renamechannels — admin bulk channel rename
# Preview-by-default. Requires confirm:true to actually execute.
# ─────────────────────────────────────────────────────────────────────────────

# Small-caps mapping
SMALL_CAPS_MAP = str.maketrans({
    "a": "ᴀ", "b": "ʙ", "c": "ᴄ", "d": "ᴅ", "e": "ᴇ", "f": "ғ", "g": "ɢ",
    "h": "ʜ", "i": "ɪ", "j": "ᴊ", "k": "ᴋ", "l": "ʟ", "m": "ᴍ", "n": "ɴ",
    "o": "ᴏ", "p": "ᴘ", "q": "ǫ", "r": "ʀ", "s": "s", "t": "ᴛ", "u": "ᴜ",
    "v": "ᴠ", "w": "ᴡ", "x": "x", "y": "ʏ", "z": "ᴢ",
})

# Bold sans-serif (𝗮𝗯𝗰 / 𝗔𝗕𝗖)
BOLD_SANS_MAP = str.maketrans({c: chr(0x1D5EE + i) for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")} |
                              {c: chr(0x1D5D4 + i) for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})

# Italic sans-serif (𝘢𝘣𝘤)
ITALIC_SANS_MAP = str.maketrans({c: chr(0x1D622 + i) for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")})

# Cursive / script (𝒶𝒷𝒸)
SCRIPT_MAP = str.maketrans({c: chr(0x1D4B6 + i) for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")})

# Double-struck (𝕒𝕓𝕔)
DOUBLE_STRUCK_MAP = str.maketrans({c: chr(0x1D552 + i) for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")})

# Monospace (𝚊𝚋𝚌)
MONO_MAP = str.maketrans({c: chr(0x1D68A + i) for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")})

# Fraktur / gothic (𝔞𝔟𝔠)
FRAKTUR_MAP = str.maketrans({c: chr(0x1D51E + i) for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")})

# Circled letters (ⓐⓑⓒ)
CIRCLED_MAP = str.maketrans({c: chr(0x24D0 + i) for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")})

# Squared / inverted box (🅐🅑🅒)
SQUARED_MAP = str.maketrans({c: chr(0x1F170 + i) for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")})

# Fullwidth (ａｂｃ) — wide spaced look
FULLWIDTH_MAP = str.maketrans({c: chr(0xFF21 + i) for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")} |
                              {c: chr(0xFF41 + i) for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")})

# Bubble inverse (🅐🅑🅒 — black background)
# already SQUARED_MAP above

# Superscript-ish small
SMALL_CAPS_MAP_2 = SMALL_CAPS_MAP


def _xform(text: str, mapping: dict) -> str:
    """Apply a translate map, leaving unmapped chars alone."""
    return text.translate(mapping)


def to_small_caps(text: str) -> str:
    return text.lower().translate(SMALL_CAPS_MAP)


def _strip_channel_decoration(name: str) -> str:
    """Strip decoration to alphanumeric base."""
    import re
    cleaned = re.sub(r"[^a-zA-Z0-9 _-]", " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace(" ", "-").strip("-")
    return cleaned or "channel"


def _format_channel_name(base: str, style: str) -> str:
    """Apply a preset style to a channel name."""
    base_clean = _strip_channel_decoration(base)
    base_pretty = base_clean.replace("-", " ").replace("_", " ").lower()

    # Helper to apply font + optional prefix/suffix
    def deco(prefix: str, body_fn, suffix: str = "") -> str:
        return f"{prefix}{body_fn(base_pretty)}{suffix}".strip()

    # Font transforms (build the body)
    small = lambda t: to_small_caps(t)
    bold_sans = lambda t: _xform(t, BOLD_SANS_MAP)
    italic = lambda t: _xform(t, ITALIC_SANS_MAP)
    script = lambda t: _xform(t, SCRIPT_MAP)
    double = lambda t: _xform(t, DOUBLE_STRUCK_MAP)
    mono = lambda t: _xform(t, MONO_MAP)
    fraktur = lambda t: _xform(t, FRAKTUR_MAP)
    circled = lambda t: _xform(t, CIRCLED_MAP)
    squared = lambda t: _xform(t.upper(), SQUARED_MAP)
    fullwidth = lambda t: _xform(t, FULLWIDTH_MAP)
    plain = lambda t: t
    upper = lambda t: t.upper()

    # ─── 40+ Style Presets ──────────────────────────────────────────────
    STYLES = {
        # Symbol prefix + small caps
        "star":          deco("★ ", small),
        "star-arrow":    deco("★彡 ", small),
        "dot":           deco("・", small),
        "diamond":       deco("❖ ", small),
        "flame":         deco("🔥・", small),
        "lightning":     deco("⚡・", small),
        "skull":         deco("💀・", small),
        "crown":         deco("👑・", small),
        "heart":         deco("♡ ", small),
        "spade":         deco("♠ ", small),
        "moon":          deco("☾ ", small),
        "snowflake":     deco("❄ ", small),
        "rose":          deco("✿ ", small),
        "sakura":        deco("✦ ", small),
        "fourstars":     deco("✦°. ", small, " .°✦"),
        "pipe":          deco("┃ ", small),
        "arrow":         deco("➤ ", small),
        "doublearrow":   deco("⪼ ", small),
        "cross":         deco("✟ ", small),
        "yinyang":       deco("☯ ", small),

        # Brackets + small caps
        "brackets":      deco("「", small, "」"),
        "double-brackets": deco("『", small, "』"),
        "angle-brackets":  deco("《", small, "》"),
        "tilde":         deco("~ ", small, " ~"),
        "stars-around":  deco("✧ ", small, " ✧"),

        # Pure font styles (no prefix)
        "small-caps":    small(base_pretty),
        "bold":          bold_sans(base_pretty),
        "italic":        italic(base_pretty),
        "script":        script(base_pretty),
        "double-struck": double(base_pretty),
        "monospace":     mono(base_pretty),
        "fraktur":       fraktur(base_pretty),
        "circled":       circled(base_pretty),
        "squared":       squared(base_pretty),
        "fullwidth":     fullwidth(base_pretty),

        # Themed combos
        "vampire":       deco("🩸 ", fraktur),
        "royal":         deco("♛ ", script),
        "matrix":        deco("⌬ ", mono),
        "y2k":           deco("✿ ", italic, " ✿"),
        "vaporwave":     deco("ﾟ･: ", fullwidth, " :･ﾟ"),
        "gamer":         deco("⌜ ", bold_sans, " ⌟"),
        "cyber":         deco("◤ ", mono, " ◥"),
        "minimal":       deco("· ", small),
        "ornate":        deco("❀ ", script, " ❀"),
        "battle":        deco("⚔️ ", small),

        # Plain
        "lowercase":     base_clean.lower(),
        "uppercase":     upper(base_pretty),
        "plain":         base_clean,
    }

    return STYLES.get(style, base_clean)[:100]


@tree.command(name="renamechannels", description="🛠️ ADMIN: bulk rename channels in this server (preview by default).")
@discord.app_commands.describe(
    style="Naming style preset",
    category="Limit to one category (optional, leave empty for all text channels)",
    confirm="Set True to actually apply changes. Otherwise shows preview only.",
)
@discord.app_commands.choices(
    style=[
        # Top 25 picks — Discord caps choices at 25
        discord.app_commands.Choice(name="★ ᴍᴀɪɴ  (star)", value="star"),
        discord.app_commands.Choice(name="★彡 ᴍᴀɪɴ  (star-arrow)", value="star-arrow"),
        discord.app_commands.Choice(name="・ᴍᴀɪɴ  (dot)", value="dot"),
        discord.app_commands.Choice(name="❖ ᴍᴀɪɴ  (diamond)", value="diamond"),
        discord.app_commands.Choice(name="🔥・ᴍᴀɪɴ  (flame)", value="flame"),
        discord.app_commands.Choice(name="⚡・ᴍᴀɪɴ  (lightning)", value="lightning"),
        discord.app_commands.Choice(name="💀・ᴍᴀɪɴ  (skull)", value="skull"),
        discord.app_commands.Choice(name="👑・ᴍᴀɪɴ  (crown)", value="crown"),
        discord.app_commands.Choice(name="♡ ᴍᴀɪɴ  (heart)", value="heart"),
        discord.app_commands.Choice(name="☾ ᴍᴀɪɴ  (moon)", value="moon"),
        discord.app_commands.Choice(name="✦°. ᴍᴀɪɴ .°✦  (sparkles)", value="fourstars"),
        discord.app_commands.Choice(name="┃ ᴍᴀɪɴ  (pipe)", value="pipe"),
        discord.app_commands.Choice(name="➤ ᴍᴀɪɴ  (arrow)", value="arrow"),
        discord.app_commands.Choice(name="「ᴍᴀɪɴ」  (brackets)", value="brackets"),
        discord.app_commands.Choice(name="《ᴍᴀɪɴ》  (angle-brackets)", value="angle-brackets"),
        discord.app_commands.Choice(name="✧ ᴍᴀɪɴ ✧  (stars-around)", value="stars-around"),
        # Pure fonts
        discord.app_commands.Choice(name="ᴍᴀɪɴ  (small-caps)", value="small-caps"),
        discord.app_commands.Choice(name="𝗺𝗮𝗶𝗻  (bold)", value="bold"),
        discord.app_commands.Choice(name="𝘮𝘢𝘪𝘯  (italic)", value="italic"),
        discord.app_commands.Choice(name="𝓂𝒶𝒾𝓃  (script)", value="script"),
        discord.app_commands.Choice(name="𝕞𝕒𝕚𝕟  (double-struck)", value="double-struck"),
        discord.app_commands.Choice(name="𝚖𝚊𝚒𝚗  (monospace)", value="monospace"),
        discord.app_commands.Choice(name="𝔪𝔞𝔦𝔫  (fraktur)", value="fraktur"),
        discord.app_commands.Choice(name="ⓜⓐⓘⓝ  (circled)", value="circled"),
        discord.app_commands.Choice(name="ｍａｉｎ  (fullwidth)", value="fullwidth"),
    ]
)
async def renamechannels_command(
    interaction: discord.Interaction,
    style: discord.app_commands.Choice[str],
    category: discord.CategoryChannel = None,
    confirm: bool = False,
):
    cfg = load_config()
    is_boss = str(interaction.user.id) in cfg.get("respected_users", [])
    has_perm = interaction.user.guild_permissions.manage_channels if interaction.guild else False
    if not (is_boss or has_perm):
        await interaction.response.send_message(
            "❌ You need **Manage Channels** permission to use this.",
            ephemeral=True,
        )
        return

    if not interaction.guild:
        await interaction.response.send_message("Server-only command.", ephemeral=True)
        return

    style_value = style.value

    targets = []
    if category:
        channels_to_check = category.channels
    else:
        channels_to_check = interaction.guild.text_channels

    for ch in channels_to_check:
        if not isinstance(ch, discord.TextChannel):
            continue
        new_name = _format_channel_name(ch.name, style_value)
        new_name = new_name[:100]
        if new_name != ch.name:
            targets.append((ch, ch.name, new_name))

    if not targets:
        await interaction.response.send_message(
            "✅ Nothing to rename — all channels already match the target format.",
            ephemeral=True,
        )
        return

    preview_lines = [f"`{old}` → `{new}`" for ch, old, new in targets]
    preview_text = "\n".join(preview_lines)

    embed = discord.Embed(
        title="🛠️ Channel Rename " + ("— APPLIED" if confirm else "— PREVIEW"),
        description=preview_text[:4000],
        color=discord.Color.green() if confirm else discord.Color.gold(),
    )
    embed.add_field(
        name="📊 Count",
        value=f"{len(targets)} channel(s) {'renamed' if confirm else 'would be renamed'}",
        inline=False,
    )

    if not confirm:
        embed.add_field(
            name="⚠️ Preview only",
            value=(
                "Re-run with `confirm:True` to apply changes.\n"
                "_Discord rate-limits channel renames to ~2 per 10 min per channel — large batches will pace themselves._"
            ),
            inline=False,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    renamed = 0
    failed = []
    for ch, old, new in targets:
        try:
            await ch.edit(name=new, reason=f"Bulk rename by {interaction.user} ({interaction.user.id})")
            renamed += 1
            await asyncio.sleep(0.5)
        except discord.Forbidden:
            failed.append(f"`{old}` — no permission")
        except discord.HTTPException as e:
            failed.append(f"`{old}` — {e.text[:80] if hasattr(e,'text') else str(e)[:80]}")
        except Exception as e:
            failed.append(f"`{old}` — {str(e)[:80]}")

    result = discord.Embed(
        title="✅ Bulk Rename Complete" if not failed else "⚠️ Bulk Rename Partial",
        description=f"**{renamed}** channel(s) renamed.",
        color=discord.Color.green() if not failed else discord.Color.orange(),
    )
    if failed:
        result.add_field(
            name=f"❌ Failed ({len(failed)})",
            value="\n".join(failed[:10])[:1024],
            inline=False,
        )
    await interaction.followup.send(embed=result, ephemeral=True)


# ── /stylepreview — see all available styles ─────────────────────────────────
@tree.command(name="stylepreview", description="🛠️ See all channel name styles available.")
async def stylepreview_command(interaction: discord.Interaction):
    cfg = load_config()
    is_boss = str(interaction.user.id) in cfg.get("respected_users", [])
    has_perm = interaction.user.guild_permissions.manage_channels if interaction.guild else False
    if not (is_boss or has_perm):
        await interaction.response.send_message(
            "❌ You need **Manage Channels** permission to use this.",
            ephemeral=True,
        )
        return

    sample = "main"
    styles = [
        # Symbol prefixes
        "star", "star-arrow", "dot", "diamond", "flame", "lightning",
        "skull", "crown", "heart", "spade", "moon", "snowflake",
        "rose", "sakura", "fourstars", "pipe", "arrow", "doublearrow",
        "cross", "yinyang",
        # Brackets
        "brackets", "double-brackets", "angle-brackets", "tilde", "stars-around",
        # Pure fonts
        "small-caps", "bold", "italic", "script", "double-struck",
        "monospace", "fraktur", "circled", "squared", "fullwidth",
        # Themed
        "vampire", "royal", "matrix", "y2k", "vaporwave",
        "gamer", "cyber", "minimal", "ornate", "battle",
        # Plain
        "lowercase", "uppercase", "plain",
    ]

    chunks = []
    current_chunk = ""
    for s in styles:
        try:
            rendered = _format_channel_name(sample, s)
        except Exception:
            rendered = "(error)"
        line = f"`{s}` → {rendered}\n"
        if len(current_chunk) + len(line) > 1000:
            chunks.append(current_chunk)
            current_chunk = ""
        current_chunk += line
    if current_chunk:
        chunks.append(current_chunk)

    embed = discord.Embed(
        title="🎨 All Channel Name Styles",
        description="_Sample input: `main`. Pick a style key and use it as `/renamechannels style:<key>`._",
        color=discord.Color.blurple(),
    )
    for i, chunk in enumerate(chunks, 1):
        embed.add_field(name=f"Styles ({i}/{len(chunks)})", value=chunk, inline=False)
    embed.set_footer(text=f"{len(styles)} total styles available")

    await interaction.response.send_message(embed=embed, ephemeral=True)


# ── /setnotifchannel — admin sets where bot notifications post ───────────────
@tree.command(name="setnotifchannel", description="🛠️ ADMIN: set the channel where all bot notifications post.")
@discord.app_commands.describe(channel="The channel for events, recaps, tournaments, etc.")
async def setnotifchannel_command(interaction: discord.Interaction, channel: discord.TextChannel):
    cfg = load_config()
    is_boss = str(interaction.user.id) in cfg.get("respected_users", [])
    has_perm = interaction.user.guild_permissions.manage_guild if interaction.guild else False
    if not (is_boss or has_perm):
        await interaction.response.send_message(
            "❌ You need **Manage Server** permission to use this.", ephemeral=True
        )
        return

    cfg["notifications_channel"] = str(channel.id)
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        await interaction.response.send_message(f"❌ Failed to save: {e}", ephemeral=True)
        return

    await interaction.response.send_message(
        f"✅ All bot notifications will now post in {channel.mention}.\n"
        f"_This covers: daily recaps, tournaments, random events, business/nightlife events, "
        f"lottery draws, and loan shark collections._\n"
        f"Personal DMs still go to users who have them enabled (`/notifications`).",
        ephemeral=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 🎭 BOOSTER SIGNATURE REACTIONS
# Boosters set their OWN signature emoji(s). When anyone REPLIES to the booster,
# the bot auto-reacts to that reply with the booster's emoji(s).
# Boosters self-serve their own emoji. Admins can override/remove anyone's.
# ─────────────────────────────────────────────────────────────────────────────
AUTOREACT_FILE = MEMORY_DIR / "autoreact.json"
BOOSTER_ROLE_ID = 1247868962881146972  # Discord booster role


def _load_autoreact() -> dict:
    if AUTOREACT_FILE.exists():
        try:
            with open(AUTOREACT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    # signatures: booster_user_id -> list of emoji strings
    return {"signatures": {}}


def _save_autoreact(data: dict):
    with open(AUTOREACT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _is_booster(member: discord.Member) -> bool:
    return any(r.id == BOOSTER_ROLE_ID for r in member.roles)


def _is_admin(member: discord.Member) -> bool:
    return (member.guild_permissions.administrator or
            member.guild_permissions.manage_guild)


def _parse_emojis(raw: str) -> list:
    """Parse a string of emojis (unicode + custom <:name:id>) into a list."""
    import re
    emojis = []
    custom = re.findall(r"<a?:\w+:\d+>", raw)
    emojis.extend(custom)
    leftover = re.sub(r"<a?:\w+:\d+>", "", raw)
    unicode_emojis = re.findall(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u2190-\u21FF\u2300-\u23FF\u2B00-\u2BFF]",
        leftover,
    )
    emojis.extend(unicode_emojis)
    return emojis


async def handle_autoreact(message: discord.Message):
    """If this message replies to a booster who has a signature, react to it."""
    if not message.reference or not message.reference.message_id:
        return
    data = _load_autoreact()
    signatures = data.get("signatures", {})
    if not signatures:
        return

    # Resolve who is being replied to
    try:
        replied_to = message.reference.resolved
        if replied_to is None:
            replied_to = await message.channel.fetch_message(message.reference.message_id)
        if not isinstance(replied_to, discord.Message):
            log.info("autoreact: replied_to not a Message (%s)", type(replied_to))
            return
        booster_id = str(replied_to.author.id)
    except Exception as e:
        log.warning("autoreact: failed to resolve reply: %s", e)
        return

    emojis = signatures.get(booster_id)
    if not emojis:
        return

    log.info("autoreact: triggered for booster %s, emojis=%s", booster_id, emojis)

    # Live booster check — if they lost the role while bot was offline, skip + clean up
    try:
        booster_member = message.guild.get_member(int(booster_id)) if message.guild else None
        if booster_member and not _is_booster(booster_member):
            del signatures[booster_id]
            _save_autoreact(data)
            log.info("autoreact: %s no longer booster, cleaned up", booster_id)
            return
    except Exception as e:
        log.warning("autoreact: booster check failed: %s", e)

    # Don't react to the booster replying to themselves
    if str(message.author.id) == booster_id:
        return

    for emoji in emojis:
        try:
            await message.add_reaction(emoji)
            await asyncio.sleep(0.25)
        except Exception as e:
            log.warning("autoreact: add_reaction failed for %r: %s", emoji, e)


# ── /signature — boosters set their own reaction emoji ───────────────────────
@tree.command(name="signature", description="\U0001F3AD BOOSTER PERK: set emoji that react to replies you receive.")
@discord.app_commands.describe(
    emojis="Emojis to use (space-separated, custom emojis OK). Leave empty to view yours.",
)
async def signature_command(interaction: discord.Interaction, emojis: str = None):
    if not interaction.guild:
        await interaction.response.send_message("Server-only command.", ephemeral=True)
        return

    member = interaction.user
    if not (_is_booster(member) or _is_admin(member)):
        await interaction.response.send_message(
            "\U0001F512 This is a **booster perk**! Boost the server to set a signature reaction "
            "that gets added to every reply you receive.",
            ephemeral=True,
        )
        return

    data = _load_autoreact()
    signatures = data.setdefault("signatures", {})

    # View mode
    if emojis is None:
        current = signatures.get(str(member.id))
        if current:
            await interaction.response.send_message(
                f"\U0001F3AD Your signature: {' '.join(current)}\n"
                f"_Anyone who replies to you gets these reactions. "
                f"Set new emojis to change, or `/signatureremove` to clear._",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "You don't have a signature set. Use `/signature emojis:\U0001F525\U0001F480` to set one.",
                ephemeral=True,
            )
        return

    parsed = _parse_emojis(emojis)
    if not parsed:
        await interaction.response.send_message(
            "\u274C No valid emojis found. Use unicode or custom `<:name:id>` emojis.",
            ephemeral=True,
        )
        return
    if len(parsed) > 3:
        await interaction.response.send_message(
            "\u274C Max **3** signature emojis per booster.", ephemeral=True
        )
        return

    # Validate custom emojis bot can use
    valid, invalid = [], []
    import re
    for emoji in parsed:
        if emoji.startswith("<"):
            m = re.match(r"<a?:\w+:(\d+)>", emoji)
            if m and client.get_emoji(int(m.group(1))):
                valid.append(emoji)
            else:
                invalid.append(emoji)
        else:
            valid.append(emoji)

    if not valid:
        await interaction.response.send_message(
            "\u274C The bot can't use those custom emojis (must be from a server it's in).",
            ephemeral=True,
        )
        return

    signatures[str(member.id)] = valid
    _save_autoreact(data)
    msg = f"\u2705 Signature set: {' '.join(valid)}\nNow anyone who replies to you gets these reactions!"
    if invalid:
        msg += f"\n\u26A0\uFE0F Skipped (bot can't use): {' '.join(invalid)}"
    await interaction.response.send_message(msg, ephemeral=True)


# ── /signatureremove — clear your own (or admin clears anyone's) ─────────────
@tree.command(name="signatureremove", description="\U0001F3AD Remove your signature reaction (admins can target anyone).")
@discord.app_commands.describe(user="(Admin only) remove this user's signature instead of yours")
async def signatureremove_command(interaction: discord.Interaction, user: discord.Member = None):
    if not interaction.guild:
        await interaction.response.send_message("Server-only command.", ephemeral=True)
        return

    member = interaction.user
    data = _load_autoreact()
    signatures = data.setdefault("signatures", {})

    # Admin removing someone else's
    if user is not None and user.id != member.id:
        if not _is_admin(member):
            await interaction.response.send_message(
                "\u274C Only admins can remove other people's signatures.", ephemeral=True
            )
            return
        if str(user.id) not in signatures:
            await interaction.response.send_message(
                f"{user.mention} has no signature set.", ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            return
        del signatures[str(user.id)]
        _save_autoreact(data)
        await interaction.response.send_message(
            f"\U0001F5D1\uFE0F Removed {user.mention}'s signature.", ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    # Removing your own
    if str(member.id) not in signatures:
        await interaction.response.send_message(
            "You don't have a signature set.", ephemeral=True
        )
        return
    del signatures[str(member.id)]
    _save_autoreact(data)
    await interaction.response.send_message(
        "\U0001F5D1\uFE0F Your signature reaction has been removed.", ephemeral=True
    )


# ── /signatures — admin: view all booster signatures ─────────────────────────
@tree.command(name="signatures", description="\U0001F3AD ADMIN: view all booster signature reactions.")
async def signatures_command(interaction: discord.Interaction):
    if not interaction.guild:
        await interaction.response.send_message("Server-only command.", ephemeral=True)
        return
    if not _is_admin(interaction.user):
        await interaction.response.send_message(
            "\u274C Admins only.", ephemeral=True
        )
        return

    data = _load_autoreact()
    signatures = data.get("signatures", {})
    if not signatures:
        await interaction.response.send_message("No signatures set.", ephemeral=True)
        return

    lines = [f"<@{uid}> \u2192 {' '.join(emj)}" for uid, emj in signatures.items()]
    embed = discord.Embed(
        title="\U0001F3AD Booster Signature Reactions",
        description="\n".join(lines),
        color=discord.Color.fuchsia() if hasattr(discord.Color, "fuchsia") else discord.Color.purple(),
    )
    embed.set_footer(text="Bot reacts to replies these users receive.")
    await interaction.response.send_message(
        embed=embed, ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
    )


@client.event
async def on_member_update(before: discord.Member, after: discord.Member):
    """When a member loses the booster role, remove their signature perk."""
    try:
        had_role = any(r.id == BOOSTER_ROLE_ID for r in before.roles)
        has_role = any(r.id == BOOSTER_ROLE_ID for r in after.roles)
        if had_role and not has_role:
            data = _load_autoreact()
            signatures = data.get("signatures", {})
            if str(after.id) in signatures:
                del signatures[str(after.id)]
                _save_autoreact(data)
                log.info("Removed signature for %s (lost booster role)", after.id)
    except Exception as e:
        log.warning("on_member_update signature cleanup failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# 🎙️ AUTO-TRANSCRIBE VOICE MESSAGES
# Uses Groq Whisper API (free, fast). Replies to voice messages with transcript.
# ─────────────────────────────────────────────────────────────────────────────
TRANSCRIBE_FILE = MEMORY_DIR / "transcribe.json"
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3"
TRANSCRIBE_MAX_SIZE_MB = 25  # Groq's audio file limit
TRANSCRIBE_MAX_DURATION_SECONDS = 600  # 10 min cap to avoid huge files


def _load_transcribe_config() -> dict:
    if TRANSCRIBE_FILE.exists():
        try:
            with open(TRANSCRIBE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"enabled": True, "disabled_channels": []}


def _save_transcribe_config(data: dict):
    with open(TRANSCRIBE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _is_voice_message(message: discord.Message) -> bool:
    """Detect a Discord voice message attachment."""
    if not message.attachments:
        return False
    for att in message.attachments:
        # Voice messages have content_type audio/ogg and a waveform field
        ct = (att.content_type or "").lower()
        if ct.startswith("audio/"):
            # Voice messages have a duration_secs or waveform
            if getattr(att, "duration", None) is not None or getattr(att, "waveform", None):
                return True
            # Fallback: filename pattern voice-message.ogg
            if "voice-message" in (att.filename or "").lower():
                return True
    return False


async def _transcribe_audio(audio_bytes: bytes, filename: str = "voice.ogg") -> str | None:
    """Send audio to Groq Whisper, return transcript or None on error."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.warning("transcribe: no GROQ_API_KEY")
        return None

    headers = {"Authorization": f"Bearer {api_key}"}
    data = aiohttp.FormData()
    data.add_field("file", audio_bytes, filename=filename, content_type="audio/ogg")
    data.add_field("model", GROQ_WHISPER_MODEL)
    data.add_field("response_format", "json")

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(GROQ_WHISPER_URL, headers=headers, data=data) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.warning("transcribe: Groq returned %s: %s", resp.status, body[:300])
                    return None
                payload = await resp.json()
                return (payload.get("text") or "").strip()
    except asyncio.TimeoutError:
        log.warning("transcribe: timed out")
        return None
    except Exception as e:
        log.warning("transcribe: exception: %s", e)
        return None


async def handle_voice_transcription(message: discord.Message):
    """If message is a voice message, transcribe and reply."""
    if not _is_voice_message(message):
        return

    cfg = _load_transcribe_config()
    if not cfg.get("enabled", True):
        return
    if str(message.channel.id) in cfg.get("disabled_channels", []):
        return

    # Get the audio attachment
    att = None
    for a in message.attachments:
        ct = (a.content_type or "").lower()
        if ct.startswith("audio/"):
            att = a
            break
    if not att:
        return

    # Size check
    size_mb = att.size / (1024 * 1024) if att.size else 0
    if size_mb > TRANSCRIBE_MAX_SIZE_MB:
        log.info("transcribe: skipping (too large %.1fMB)", size_mb)
        return
    # Duration check (if exposed)
    dur = getattr(att, "duration", None)
    if dur and dur > TRANSCRIBE_MAX_DURATION_SECONDS:
        log.info("transcribe: skipping (too long %.0fs)", dur)
        return

    try:
        audio_bytes = await att.read()
    except Exception as e:
        log.warning("transcribe: failed to download audio: %s", e)
        return

    transcript = await _transcribe_audio(audio_bytes, filename=att.filename or "voice.ogg")
    if not transcript:
        return

    # Truncate if absurdly long
    if len(transcript) > 1900:
        transcript = transcript[:1900] + "…"

    try:
        await message.reply(
            f"🎙️ **Transcript:**\n```\n{transcript}\n```",
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )
    except Exception as e:
        log.warning("transcribe: failed to reply: %s", e)


# ── /transcribetoggle — admin toggle on/off, per-channel ─────────────────────
@tree.command(name="transcribetoggle", description="🎙️ ADMIN: toggle voice message transcription (global or per-channel).")
@discord.app_commands.describe(
    scope="What to toggle",
    channel="If scope=channel, which one (defaults to current)",
)
@discord.app_commands.choices(
    scope=[
        discord.app_commands.Choice(name="Global on/off", value="global"),
        discord.app_commands.Choice(name="Disable in one channel", value="disable_channel"),
        discord.app_commands.Choice(name="Enable in one channel", value="enable_channel"),
        discord.app_commands.Choice(name="Show current settings", value="status"),
    ]
)
async def transcribetoggle_command(
    interaction: discord.Interaction,
    scope: discord.app_commands.Choice[str],
    channel: discord.TextChannel = None,
):
    if not interaction.guild:
        await interaction.response.send_message("Server-only.", ephemeral=True)
        return
    if not (interaction.user.guild_permissions.administrator or
            interaction.user.guild_permissions.manage_guild):
        await interaction.response.send_message(
            "❌ Admin only.", ephemeral=True
        )
        return

    cfg = _load_transcribe_config()
    scope_value = scope.value

    if scope_value == "status":
        disabled = cfg.get("disabled_channels", [])
        embed = discord.Embed(
            title="🎙️ Voice Transcription Status",
            color=discord.Color.blurple(),
        )
        embed.add_field(
            name="Globally",
            value="✅ ON" if cfg.get("enabled", True) else "❌ OFF",
            inline=False,
        )
        if disabled:
            ch_list = []
            for cid in disabled:
                try:
                    c = interaction.guild.get_channel(int(cid))
                    ch_list.append(c.mention if c else f"`{cid}`")
                except Exception:
                    ch_list.append(f"`{cid}`")
            embed.add_field(
                name="Disabled in these channels",
                value="\n".join(ch_list),
                inline=False,
            )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return

    if scope_value == "global":
        new_state = not cfg.get("enabled", True)
        cfg["enabled"] = new_state
        _save_transcribe_config(cfg)
        await interaction.response.send_message(
            f"🎙️ Voice transcription is now **{'ON' if new_state else 'OFF'}** globally.",
            ephemeral=True,
        )
        return

    # Per-channel toggles
    target = channel or interaction.channel
    cid = str(target.id)
    disabled = cfg.setdefault("disabled_channels", [])

    if scope_value == "disable_channel":
        if cid not in disabled:
            disabled.append(cid)
        _save_transcribe_config(cfg)
        await interaction.response.send_message(
            f"🔇 Voice transcription **disabled** in {target.mention}.",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
    elif scope_value == "enable_channel":
        if cid in disabled:
            disabled.remove(cid)
        _save_transcribe_config(cfg)
        await interaction.response.send_message(
            f"🔊 Voice transcription **enabled** in {target.mention}.",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 🦹 HEIST CREW — Multi-step heists with hired specialists
# 6 targets, 6 specialist roles, 5-phase live narration, partial-failure payouts
# ─────────────────────────────────────────────────────────────────────────────
HEIST_CREW_FILE = MEMORY_DIR / "heist_crew.json"

# Specialists — each has a base success chance per role
SPECIALISTS = {
    "safecracker": {"emoji": "🔓", "name": "Safecracker", "desc": "Cracks vaults & safes."},
    "hacker":      {"emoji": "💻", "name": "Hacker",      "desc": "Disables alarms & security."},
    "driver":      {"emoji": "🚗", "name": "Driver",      "desc": "Getaway and escape rolls."},
    "demolitions": {"emoji": "💣", "name": "Demolitions", "desc": "Blows open shortcuts & bonus vaults."},
    "lookout":     {"emoji": "👁️", "name": "Lookout",    "desc": "Early warning + casing rolls."},
    "conman":      {"emoji": "🎭", "name": "Conman",      "desc": "Bypasses guards & security checks."},
}

# Tiers — multiplier on hire cost & boost magnitude
SPECIALIST_TIERS = {
    "rookie":  {"name": "Rookie", "cost_mult": 1.0, "boost": 0.0, "emoji": "⚪"},
    "pro":     {"name": "Pro",    "cost_mult": 3.0, "boost": 0.15, "emoji": "🔵"},
    "legend":  {"name": "Legend", "cost_mult": 8.0, "boost": 0.30, "emoji": "🟣"},
}
SPECIALIST_BASE_COST = 500  # rookie cost; multiplied by tier

# Heist targets — required specialists per phase, payout range
HEIST_TARGETS = {
    "convenience": {
        "emoji": "🏪", "name": "Convenience Store",
        "min_crew": 1, "max_crew": 2,
        "required_specialists": [],
        "useful_specialists": ["driver"],
        "payout_min": 500,  "payout_max": 2_000,
        "cooldown_hours": 0.5,
        "fail_fine_pct": 0.05,
        "tier": 1,
        "desc": "Easy in-and-out. Low loot, low risk.",
    },
    "bank": {
        "emoji": "🏦", "name": "Local Bank",
        "min_crew": 2, "max_crew": 4,
        "required_specialists": ["driver"],
        "useful_specialists": ["safecracker", "lookout"],
        "payout_min": 3_000, "payout_max": 10_000,
        "cooldown_hours": 2,
        "fail_fine_pct": 0.10,
        "tier": 2,
        "desc": "Classic bank job. Need a getaway driver.",
    },
    "jewelry": {
        "emoji": "💎", "name": "Jewelry Store",
        "min_crew": 2, "max_crew": 3,
        "required_specialists": ["safecracker"],
        "useful_specialists": ["lookout", "driver"],
        "payout_min": 8_000, "payout_max": 25_000,
        "cooldown_hours": 3,
        "fail_fine_pct": 0.10,
        "tier": 3,
        "desc": "Smash the cases. Need a cracker for the back safe.",
    },
    "casino": {
        "emoji": "🎰", "name": "Casino Vault",
        "min_crew": 4, "max_crew": 6,
        "required_specialists": ["hacker", "driver", "safecracker"],
        "useful_specialists": ["lookout", "conman"],
        "payout_min": 30_000, "payout_max": 100_000,
        "cooldown_hours": 12,
        "fail_fine_pct": 0.15,
        "tier": 4,
        "desc": "Ocean's Eleven energy. Big crew, bigger take.",
    },
    "crypto": {
        "emoji": "💰", "name": "Crypto Exchange",
        "min_crew": 4, "max_crew": 5,
        "required_specialists": ["hacker", "conman"],
        "useful_specialists": ["demolitions", "lookout"],
        "payout_min": 100_000, "payout_max": 300_000,
        "cooldown_hours": 24,
        "fail_fine_pct": 0.20,
        "tier": 5,
        "desc": "Pull the keys off cold storage. Total digital heist.",
    },
    "federal": {
        "emoji": "🏛️", "name": "Federal Reserve",
        "min_crew": 5, "max_crew": 6,
        "required_specialists": ["safecracker", "hacker", "driver", "demolitions", "lookout"],
        "useful_specialists": ["conman"],
        "payout_min": 200_000, "payout_max": 500_000,
        "cooldown_hours": 48,
        "fail_fine_pct": 0.25,
        "tier": 6,
        "desc": "The ultimate score. Don't even THINK about going in light.",
    },
}

# Phase definitions per target (which specialist handles each phase)
HEIST_PHASES_BY_TARGET = {
    "convenience": [
        ("👀", "Casing the joint",   "lookout"),
        ("🚪", "Cracking the safe",  None),  # No specialist needed
        ("🚓", "Cops arrive — escape!", "driver"),
    ],
    "bank": [
        ("👀", "Casing the bank",        "lookout"),
        ("💻", "Disabling silent alarm", "hacker"),
        ("🔓", "Cracking the vault",     "safecracker"),
        ("🚓", "Cops responding — escape!", "driver"),
    ],
    "jewelry": [
        ("👀", "Casing the storefront",     "lookout"),
        ("💎", "Smashing display cases",    None),
        ("🔓", "Opening the back safe",     "safecracker"),
        ("🚓", "Slipping out the back",     "driver"),
    ],
    "casino": [
        ("🎭", "Talking past front security", "conman"),
        ("💻", "Killing the cameras",          "hacker"),
        ("👁️", "Tracking the guards",         "lookout"),
        ("🔓", "Cracking the vault",           "safecracker"),
        ("🚓", "Escape through the parking deck", "driver"),
    ],
    "crypto": [
        ("🎭", "Social engineering the CEO",   "conman"),
        ("💻", "Breaching the network",        "hacker"),
        ("💣", "Blowing the cold storage room","demolitions"),
        ("💻", "Transferring funds",           "hacker"),
        ("👁️", "Watching for feds",           "lookout"),
    ],
    "federal": [
        ("👁️", "Casing the Fed building",   "lookout"),
        ("🎭", "Slipping past badge scanners","conman"),
        ("💻", "Cutting building security",  "hacker"),
        ("💣", "Blowing the inner blast door","demolitions"),
        ("🔓", "Cracking the gold vault",    "safecracker"),
        ("🚓", "Escape from the National Guard", "driver"),
    ],
}

HEIST_CREW_INVITE_TIMEOUT = 90  # seconds to wait for crew accepts
HEIST_PHASE_DELAY = 4  # seconds between phases for drama
ACTIVE_HEIST_CREW: dict[str, dict] = {}  # channel_id -> heist state


def _load_heist_crew() -> dict:
    if HEIST_CREW_FILE.exists():
        try:
            with open(HEIST_CREW_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"users": {}}


def _save_heist_crew(data: dict):
    with open(HEIST_CREW_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _user_heist_record(user_id: int) -> dict:
    data = _load_heist_crew()
    uid = str(user_id)
    if uid not in data["users"]:
        data["users"][uid] = {
            "specialists": {},  # role -> tier (most recent hire)
            "last_heist": {},   # target_key -> timestamp
            "lifetime_heists": 0,
            "lifetime_loot": 0,
            "jail_until": 0,
        }
        _save_heist_crew(data)
    return data["users"][uid]


def _specialist_cost(role: str, tier: str) -> int:
    base = SPECIALIST_BASE_COST
    return int(base * SPECIALIST_TIERS[tier]["cost_mult"])


# ── /hire_specialist ─────────────────────────────────────────────────────────
@tree.command(name="hire_specialist", description="🦹 Hire a specialist for upcoming heists.")
@discord.app_commands.describe(
    role="Which specialist",
    tier="Tier (better = more expensive but more reliable)",
)
@discord.app_commands.choices(
    role=[
        discord.app_commands.Choice(name=f"{s['emoji']} {s['name']} — {s['desc']}", value=k)
        for k, s in SPECIALISTS.items()
    ],
    tier=[
        discord.app_commands.Choice(name=f"{t['emoji']} {t['name']} (+{int(t['boost']*100)}% boost)", value=k)
        for k, t in SPECIALIST_TIERS.items()
    ],
)
async def hire_specialist_command(
    interaction: discord.Interaction,
    role: discord.app_commands.Choice[str],
    tier: discord.app_commands.Choice[str],
):
    user = interaction.user
    role_key = role.value
    tier_key = tier.value
    cost = _specialist_cost(role_key, tier_key)

    if economy.balance(user.id) < cost:
        await interaction.response.send_message(
            f"❌ This {SPECIALIST_TIERS[tier_key]['name']} {SPECIALISTS[role_key]['name']} "
            f"costs **{cost:,}** coins. You have **{economy.balance(user.id):,}**.",
            ephemeral=True,
        )
        return

    data = _load_heist_crew()
    record = data["users"].setdefault(str(user.id), {
        "specialists": {}, "last_heist": {}, "lifetime_heists": 0,
        "lifetime_loot": 0, "jail_until": 0,
    })

    # Upgrading? If current tier is same or better, warn
    current = record["specialists"].get(role_key)
    if current:
        current_idx = list(SPECIALIST_TIERS).index(current)
        new_idx = list(SPECIALIST_TIERS).index(tier_key)
        if new_idx <= current_idx:
            await interaction.response.send_message(
                f"❌ You already have a {SPECIALIST_TIERS[current]['name']} "
                f"{SPECIALISTS[role_key]['name']}. Hiring a lower-tier replacement would downgrade you.",
                ephemeral=True,
            )
            return

    economy.add(user.id, -cost, f"hire {role_key} {tier_key}")
    record["specialists"][role_key] = tier_key
    _save_heist_crew(data)

    info = SPECIALISTS[role_key]
    tier_info = SPECIALIST_TIERS[tier_key]
    await interaction.response.send_message(
        f"# {info['emoji']} {tier_info['name'].upper()} {info['name'].upper()} HIRED\n\n"
        f"💰 Cost: **{cost:,}** coins\n"
        f"📈 Boost: **+{int(tier_info['boost']*100)}%** to {info['name'].lower()} rolls\n"
        f"💡 _Specialists persist between heists. They get fired only if you replace them._"
    )


# ── /crew ────────────────────────────────────────────────────────────────────
@tree.command(name="crew", description="🦹 View your current heist crew (specialists & stats).")
@discord.app_commands.describe(user="Whose crew to view (defaults to you)")
async def crew_command(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user
    record = _user_heist_record(target.id)
    specialists = record.get("specialists", {})

    embed = discord.Embed(
        title=f"🦹 {target.display_name}'s Heist Crew",
        color=discord.Color.dark_grey(),
    )

    if specialists:
        lines = []
        for role_key, tier_key in specialists.items():
            info = SPECIALISTS[role_key]
            tier_info = SPECIALIST_TIERS[tier_key]
            lines.append(f"{info['emoji']} **{info['name']}** — {tier_info['emoji']} {tier_info['name']}")
        embed.add_field(name="👥 Specialists", value="\n".join(lines), inline=False)
    else:
        embed.add_field(
            name="👥 Specialists",
            value="_No crew yet. Hire some with `/hire_specialist`._",
            inline=False,
        )

    # Jail status
    jail = record.get("jail_until", 0)
    if jail > time.time():
        remaining = int(jail - time.time())
        embed.add_field(name="🚔 IN JAIL", value=f"Out in **{fmt_cooldown(remaining)}**", inline=False)

    embed.add_field(name="🎯 Lifetime Heists", value=str(record.get("lifetime_heists", 0)), inline=True)
    embed.add_field(name="💰 Lifetime Loot", value=f"{record.get('lifetime_loot', 0):,}", inline=True)

    await interaction.response.send_message(embed=embed)


# ── /targets ─────────────────────────────────────────────────────────────────
@tree.command(name="targets", description="🦹 View available heist targets and their requirements.")
async def targets_command(interaction: discord.Interaction):
    record = _user_heist_record(interaction.user.id)
    user_specs = set(record.get("specialists", {}).keys())

    embed = discord.Embed(
        title="🎯 HEIST TARGETS",
        description="_Pull these jobs with `/heist target:<name>`. Match specialists to the requirements._",
        color=discord.Color.dark_red(),
    )

    for key, t in HEIST_TARGETS.items():
        # Show what user is missing
        required = set(t["required_specialists"])
        missing = required - user_specs
        if missing:
            missing_str = ", ".join(SPECIALISTS[r]["name"] for r in missing)
            status = f"❌ Missing: {missing_str}"
        else:
            status = "✅ Ready to pull"
        # Cooldown
        last = record.get("last_heist", {}).get(key, 0)
        cd_remaining = (t["cooldown_hours"] * 3600) - (time.time() - last)
        if cd_remaining > 0:
            status += f" • ⏰ {fmt_cooldown(int(cd_remaining))}"

        req_str = ", ".join(SPECIALISTS[r]["name"] for r in t["required_specialists"]) or "_none_"
        useful_str = ", ".join(SPECIALISTS[r]["name"] for r in t["useful_specialists"]) or "_none_"

        embed.add_field(
            name=f"{t['emoji']} {t['name']} (Tier {t['tier']})",
            value=(
                f"💰 Loot: **{t['payout_min']:,}–{t['payout_max']:,}**\n"
                f"👥 Crew: **{t['min_crew']}–{t['max_crew']}**\n"
                f"⚠️ Required: {req_str}\n"
                f"💡 Useful: {useful_str}\n"
                f"{status}"
            ),
            inline=False,
        )

    await interaction.response.send_message(embed=embed)


# ── /heist (the main event) ──────────────────────────────────────────────────
@tree.command(name="heist", description="🦹 Plan a multi-phase heist with your crew. Big risk, big reward.")
@discord.app_commands.describe(
    target="Which target to hit",
    crewmate1="Crewmate (required for most targets)",
    crewmate2="Optional second crewmate",
    crewmate3="Optional third crewmate",
    crewmate4="Optional fourth crewmate",
    crewmate5="Optional fifth crewmate",
)
@discord.app_commands.choices(
    target=[
        discord.app_commands.Choice(name=f"{t['emoji']} {t['name']} (Tier {t['tier']})", value=k)
        for k, t in HEIST_TARGETS.items()
    ]
)
async def heist_command(
    interaction: discord.Interaction,
    target: discord.app_commands.Choice[str],
    crewmate1: discord.Member = None,
    crewmate2: discord.Member = None,
    crewmate3: discord.Member = None,
    crewmate4: discord.Member = None,
    crewmate5: discord.Member = None,
):
    user = interaction.user
    target_key = target.value
    t = HEIST_TARGETS[target_key]
    channel_id = str(interaction.channel_id)

    if channel_id in ACTIVE_HEIST_CREW:
        await interaction.response.send_message(
            "❌ A heist is already running in this channel.", ephemeral=True
        )
        return

    # Build initial crew list (no duplicates, no bots, no self-duplicate)
    crew = [user]
    seen = {user.id}
    for cm in [crewmate1, crewmate2, crewmate3, crewmate4, crewmate5]:
        if cm is None or cm.bot or cm.id in seen:
            continue
        seen.add(cm.id)
        crew.append(cm)

    if len(crew) < t["min_crew"]:
        await interaction.response.send_message(
            f"❌ {t['name']} needs **{t['min_crew']}** crew minimum. You have {len(crew)}.",
            ephemeral=True,
        )
        return
    if len(crew) > t["max_crew"]:
        crew = crew[:t["max_crew"]]

    # Check leader's jail status
    leader_record = _user_heist_record(user.id)
    if leader_record.get("jail_until", 0) > time.time():
        await interaction.response.send_message(
            f"🚔 You're in jail until <t:{int(leader_record['jail_until'])}:R>.",
            ephemeral=True,
        )
        return

    # Cooldown for this target
    last = leader_record.get("last_heist", {}).get(target_key, 0)
    cd_remaining = (t["cooldown_hours"] * 3600) - (time.time() - last)
    if cd_remaining > 0:
        await interaction.response.send_message(
            f"⏰ {t['name']} cooldown: **{fmt_cooldown(int(cd_remaining))}** left.",
            ephemeral=True,
        )
        return

    # Check required specialists across the WHOLE crew (any crewmate's specialists count)
    crew_specialists = {}  # role_key -> best_tier_idx
    crew_specialist_owners = {}  # role_key -> user_id
    for member in crew:
        m_record = _user_heist_record(member.id)
        for role_key, tier_key in m_record.get("specialists", {}).items():
            tier_idx = list(SPECIALIST_TIERS).index(tier_key)
            if role_key not in crew_specialists or tier_idx > crew_specialists[role_key]:
                crew_specialists[role_key] = tier_idx
                crew_specialist_owners[role_key] = member.id

    missing_required = [
        r for r in t["required_specialists"] if r not in crew_specialists
    ]
    if missing_required:
        names = ", ".join(SPECIALISTS[r]["name"] for r in missing_required)
        await interaction.response.send_message(
            f"❌ Crew is missing required specialist(s): **{names}**. "
            f"Hire them with `/hire_specialist` or bring a crewmate who has them.",
            ephemeral=True,
        )
        return

    # Lock channel for heist
    ACTIVE_HEIST_CREW[channel_id] = {
        "leader": user.id,
        "target": target_key,
        "crew_ids": [m.id for m in crew],
        "phase": "running",
        "started_at": time.time(),
    }

    # Initial embed
    await interaction.response.send_message(
        embed=discord.Embed(
            title=f"{t['emoji']} {t['name'].upper()} — INCOMING",
            description=(
                f"**Leader:** {user.mention}\n"
                f"**Crew:** {', '.join(m.mention for m in crew)}\n"
                f"**Specialists on deck:** " +
                ", ".join(f"{SPECIALISTS[r]['emoji']} {SPECIALISTS[r]['name']} ({SPECIALIST_TIERS[list(SPECIALIST_TIERS)[tier_idx]]['emoji']})" for r, tier_idx in crew_specialists.items())
            ),
            color=discord.Color.dark_red(),
        ),
        allowed_mentions=discord.AllowedMentions.none(),
    )
    heist_msg = await interaction.original_response()
    asyncio.create_task(_run_heist(interaction, channel_id, heist_msg, crew, crew_specialists, crew_specialist_owners))


async def _run_heist(interaction, channel_id, heist_msg, crew, crew_specialists, crew_specialist_owners):
    """Run the multi-phase heist animation."""
    state = ACTIVE_HEIST_CREW.get(channel_id)
    if not state:
        return
    target_key = state["target"]
    t = HEIST_TARGETS[target_key]
    phases = HEIST_PHASES_BY_TARGET[target_key]
    leader = crew[0]

    log_lines = []
    phase_results = []  # True/False per phase
    payout_share = 1.0  # multiplier — drops with each failure

    for idx, (emoji, phase_name, required_role) in enumerate(phases):
        await asyncio.sleep(HEIST_PHASE_DELAY)
        # Compute success chance
        base_success = 0.75
        if required_role:
            if required_role in crew_specialists:
                tier_idx = crew_specialists[required_role]
                tier_key = list(SPECIALIST_TIERS)[tier_idx]
                base_success += SPECIALIST_TIERS[tier_key]["boost"]
            else:
                base_success = 0.40  # punishing if missing specialist for a phase
        # Useful (non-required) specialists give small global bonus too
        for useful_role in t["useful_specialists"]:
            if useful_role in crew_specialists:
                tier_idx = crew_specialists[useful_role]
                tier_key = list(SPECIALIST_TIERS)[tier_idx]
                base_success += SPECIALIST_TIERS[tier_key]["boost"] * 0.3
        base_success = min(0.97, base_success)  # cap

        success = random.random() < base_success
        phase_results.append(success)

        if success:
            if required_role and required_role in crew_specialist_owners:
                spec_owner = crew_specialist_owners[required_role]
                log_lines.append(f"{emoji} **{phase_name}** — ✅ Pulled off by <@{spec_owner}>")
            else:
                log_lines.append(f"{emoji} **{phase_name}** — ✅ Smooth")
        else:
            log_lines.append(f"{emoji} **{phase_name}** — ❌ Bungled")
            payout_share *= 0.6  # each failure costs 40% of payout
            # Final phase failure (usually escape) = total bust
            if idx == len(phases) - 1:
                payout_share = 0

        # Update embed
        try:
            await heist_msg.edit(
                embed=discord.Embed(
                    title=f"{t['emoji']} {t['name'].upper()} — IN PROGRESS",
                    description=(
                        f"**Crew:** {', '.join(m.mention for m in crew)}\n\n"
                        + "\n".join(log_lines)
                    ),
                    color=discord.Color.gold(),
                ),
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except Exception:
            pass

    # Calculate final payout
    base_payout = random.randint(t["payout_min"], t["payout_max"])
    final_payout = int(base_payout * payout_share)

    # Last phase determines if cops catch them
    last_succeeded = phase_results[-1]
    failures = phase_results.count(False)
    # Total bust = lose everything + jail time
    total_bust = not last_succeeded

    if total_bust:
        # Leader pays fine + jail time
        bal = economy.balance(leader.id)
        fine = int(bal * t["fail_fine_pct"])
        economy.add(leader.id, -fine, "heist failed")
        # Jail leader for the cooldown
        leader_record = _user_heist_record(leader.id)
        leader_record["jail_until"] = time.time() + t["cooldown_hours"] * 3600
        # Update last_heist so cooldown still works
        leader_record["last_heist"][target_key] = time.time()
        data = _load_heist_crew()
        data["users"][str(leader.id)] = leader_record
        _save_heist_crew(data)

        try:
            await heist_msg.edit(
                embed=discord.Embed(
                    title=f"🚔 BUSTED — {t['name'].upper()} FAILED",
                    description=(
                        f"**Crew:** {', '.join(m.mention for m in crew)}\n\n"
                        + "\n".join(log_lines)
                        + f"\n\n💀 {failures}/{len(phases)} phases failed. Cops caught the crew.\n"
                        f"💸 {leader.mention} paid **{fine:,}** in fines.\n"
                        f"🚔 Leader is in **jail for {t['cooldown_hours']}h**."
                    ),
                    color=discord.Color.dark_red(),
                ),
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except Exception:
            pass
    else:
        # Successful (maybe partial) — distribute payout
        per_member = final_payout // len(crew) if len(crew) > 0 else 0
        leader_record = _user_heist_record(leader.id)
        leader_record["last_heist"][target_key] = time.time()
        leader_record["lifetime_heists"] = leader_record.get("lifetime_heists", 0) + 1
        leader_record["lifetime_loot"] = leader_record.get("lifetime_loot", 0) + final_payout

        payout_lines = []
        for m in crew:
            economy.add(m.id, per_member, f"heist {target_key}")
            track_economy_event("earned", per_member)
            payout_lines.append(f"💰 {m.mention} — +**{per_member:,}**")
            # Quests/tournament tracking
            try:
                track_quest_progress(m.id, "coins_earned", per_member)
                add_tournament_score(m.id, coins_earned=per_member, games_won=1)
                await trigger_balance_check(m.id, channel=interaction.channel)
            except Exception:
                pass

        # One activity feed entry per heist
        track_feature_use("heist")
        track_activity(
            "heist", leader.id, leader.display_name,
            f"pulled {HEIST_TARGETS[target_key]['name']} for {final_payout:,} ({len(crew)} crew)",
        )

        # Save record
        data = _load_heist_crew()
        data["users"][str(leader.id)] = leader_record
        _save_heist_crew(data)

        title = f"💰 {t['name'].upper()} — SCORE!" if final_payout >= base_payout * 0.8 else f"⚠️ {t['name'].upper()} — PARTIAL"
        try:
            await heist_msg.edit(
                embed=discord.Embed(
                    title=title,
                    description=(
                        "\n".join(log_lines)
                        + f"\n\n**Total take:** {final_payout:,} coins ({int(payout_share*100)}% of max)\n"
                        + "\n".join(payout_lines)
                    ),
                    color=discord.Color.green() if final_payout >= base_payout * 0.8 else discord.Color.orange(),
                ),
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except Exception:
            pass

    ACTIVE_HEIST_CREW.pop(channel_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# 💡 SUGGEST NEXT STEP
# Shows contextual "next purchase" tips after earning commands.
# Drives progression: balance → pet → business → nightlife → empire.
# ─────────────────────────────────────────────────────────────────────────────

def suggest_next_step(user_id: int) -> str:
    """Return a contextual next-goal tip based on what the user already owns.
    Returns a short formatted string or empty if user has it all."""
    bal = economy.balance(user_id)
    pets = _load_pets()
    has_pet = str(user_id) in pets
    user_bizs = _user_businesses(user_id)
    has_business = len(user_bizs) > 0
    user_venues = _user_venues(user_id)
    has_venue = len(user_venues) > 0

    # Pet collect available? (passive income waiting)
    if has_pet:
        pet = pets[str(user_id)]
        level = _pet_level(pet["xp"])
        hours_since = (time.time() - pet.get("last_collected", time.time())) / 3600
        pending = int(level * PET_DAILY_INCOME_BASE * (hours_since / 24))
        if _pet_hunger(pet) < 30:
            pending = pending // 2
        if pending >= 100:
            return f"💰 _Your pet has **{pending:,}** coins waiting! Run `/collect`._"

    # Business pending?
    if has_business:
        total_pending = sum(_business_pending_income(b) for b in user_bizs)
        if total_pending >= 500:
            return f"💰 _Your businesses have **{total_pending:,}** coins pending! Run `/collectbusiness`._"

    # Venue pending?
    if has_venue:
        total_pending = sum(_venue_pending_income(v) for v in user_venues)
        if total_pending >= 500:
            return f"💰 _Your venues have **{total_pending:,}** coins pending! Run `/collectvenue`._"

    # Progressive suggestions by balance + ownership
    if not has_pet and bal >= 1_500:
        return f"🐶 _You can afford a pet (1,500 coins)! Use `/adopt` for passive income._"
    if not has_business and bal >= 2_000:
        return f"🏢 _Buy your first business with `/buybusiness`! Lemonade Stand = 2,000._"
    if not has_pet and bal < 1_500:
        return f"🐶 _Save up to **1,500** for your first pet (`/adopt`). Earns coins 24/7._"
    if not has_business and bal < 2_000:
        return f"🏢 _Save up to **2,000** for your first business (`/buybusiness`)._"

    # Has both pet and business — push to nightlife
    if has_pet and has_business and not has_venue:
        if bal >= 5_000:
            return f"🌃 _Open a Dive Bar (5,000) with `/buyvenue` to scale up your empire._"
        else:
            return f"🌃 _Save up to **5,000** to open your first bar (`/buyvenue`)._"

    # Has everything — push to dealer game / heist / shop perks
    if has_pet and has_business and has_venue:
        # Suggest based on balance
        if bal >= 50_000:
            return f"🦹 _Try a heist! `/targets` to see jobs. Or hit `/shop` for boosts._"
        if bal >= 10_000:
            return f"💊 _Try the dealer game with `/dealer` — fast cash if you can dodge cops._"
        return f"🏆 _Run `/quests` for daily bonuses or `/tournament` to chase the weekly prize._"

    return ""


# ─────────────────────────────────────────────────────────────────────────────


@tree.command(name="commands", description="See all bot commands organized by category.")
async def commands_command(interaction: discord.Interaction):
    embed = _build_commands_home_embed()
    view = CommandsNavView(interaction.user.id)
    await interaction.response.send_message(embed=embed, view=view)


def _build_commands_home_embed() -> discord.Embed:
    """The 'start here' landing page."""
    embed = discord.Embed(
        title="\U0001F3AE Welcome to Jordan Belfort",
        description=(
            "_The full degen economy bot. Pick a category below to see commands._\n\n"
            "**Use `/` to type any command, or `_` as a prefix shortcut (e.g. `_balance`)**"
        ),
        color=discord.Color.gold(),
    )

    embed.add_field(
        name="\U0001F195 NEW HERE? START EARNING",
        value=(
            "1\u20E3 `/balance` \u2014 Check your wallet\n"
            "2\u20E3 `/daily` \u2014 Free coins every 24h\n"
            "3\u20E3 `/work` \u2014 Earn 50-250 every 45min\n"
            "4\u20E3 `/weekly` \u2014 Big bonus every 7 days\n\n"
            "_Goal: get to ~2,000 coins so you can adopt a pet or buy a business._"
        ),
        inline=False,
    )

    embed.add_field(
        name="\U0001F4B0 NEXT: PASSIVE INCOME",
        value=(
            "**\U0001F436 Pets** \u2014 `/adopt` then `/pet` `/feed` `/collect`\n"
            "**\U0001F3E2 Businesses** \u2014 `/buybusiness` then `/businesses` `/collectbusiness`\n"
            "**\U0001F303 Nightlife** \u2014 `/buyvenue` then `/venues` `/collectvenue`\n\n"
            "_Earn coins 24/7 even when you're offline._"
        ),
        inline=False,
    )

    embed.add_field(
        name="\U0001F3AF AFTER THAT: PROGRESSION",
        value=(
            "`/level` \u2014 Your XP and rank\n"
            "`/achievements` \u2014 Earn badges & perks\n"
            "`/quests` \u2014 Daily & weekly tasks\n"
            "`/tournament` \u2014 Weekly comp for top 3"
        ),
        inline=False,
    )

    embed.add_field(
        name="\U0001F4D6 BROWSE ALL COMMANDS",
        value="_Use the buttons below to explore every category._",
        inline=False,
    )
    embed.set_footer(text="Click a button \u2192 see commands in that category")
    return embed


def _build_commands_category_embed(category: str) -> discord.Embed:
    """Build an embed for a specific category."""
    categories = {
        "economy": {
            "title": "\U0001F4B0 ECONOMY \u2014 Earn & Spend",
            "color": discord.Color.gold(),
            "fields": [
                ("Earning", "`/daily` Daily reward (24h)\n`/weekly` Weekly reward (7d)\n`/work` Work for coins (45m)\n`/beg` Beg for change (10m)"),
                ("Spending & Transfers", "`/balance` Check wallet\n`/pay` Send coins to a user\n`/leaderboard` Richest users"),
                ("Risky Plays", "`/rob` Try to rob someone (2h cd)\n`/crime` Commit a crime (2h cd)\n`/bet` Double or nothing"),
            ],
        },
        "games": {
            "title": "\U0001F3B2 GAMES \u2014 Play to Win",
            "color": discord.Color.purple(),
            "fields": [
                ("Casino", "`/slots` Spin the slots\n`/blackjack` Beat the dealer\n`/wheel` Wheel of fortune\n`/casino` Browse casino games"),
                ("Multiplayer PvP", "`/duel` Quick 1v1 duel\n`/fight` 60s fight w/ spectator bets\n`/gun` Russian roulette (2-4p)"),
                ("Lobby Games", "`/rs` Race tag (up to 3)\n`/shootout` Lobby + doors\n`/bomb` Hot potato\n`/connect4` Classic Connect 4\n`/rps` Rock-paper-scissors\n`/rps-tournament` Bracket RPS\n`/lieordie` Lie detector"),
                ("Fun & Solo", "`/roll` Dice\n`/flip` Coin flip\n`/8ball` Magic 8-ball\n`/rate` Get rated\n`/ship` Ship two users\n`/quest` AI choose-adventure"),
                ("AI Roleplay", "`/court` Criminal trial (no $)\n`/lawsuit` Civil suit (real $)\n`/tarot` Tarot reading\n`/roast` AI roast\n`/bio` Generate a bio\n`/analyze` Analyze a user\n`/hack` Pretend hack"),
            ],
        },
        "passive": {
            "title": "\U0001F4C8 PASSIVE INCOME EMPIRES",
            "color": discord.Color.dark_green(),
            "fields": [
                ("\U0001F436 Pets", "`/adopt` Get a pet (1,500)\n`/pet` View stats\n`/feed` Feed (50)\n`/collect` Claim earnings\n`/abandon` Give up your pet"),
                ("\U0001F3E2 Businesses", "`/buybusiness` Buy (8 tiers)\n`/businesses` Portfolio\n`/collectbusiness` Claim earnings\n`/hire` Hire employee\n`/fire` Fire employee\n`/sabotage` Sabotage a rival"),
                ("\U0001F303 Nightlife", "`/buyvenue` Open bar/club (7 tiers)\n`/venues` Portfolio\n`/collectvenue` Claim earnings\n`/hirestaff` Hire bartender/bouncer/DJ\n`/stockliquor` Stock liquor"),
            ],
        },
        "dealer": {
            "title": "\U0001F48A DEALER GAME",
            "color": discord.Color.dark_purple(),
            "fields": [
                ("Get Started", "**`/dealer`** \u2014 Full dashboard (recommended)\n`/buysupply` Cop product from the plug\n`/sell` Sell to NPCs (builds heat)"),
                ("Manage", "`/stash` Inventory + heat + stats\n`/streetprice` Today's market prices\n`/laylow` Pay 500 to drop heat\n`/dealers` Top dealers leaderboard"),
            ],
        },
        "heist": {
            "title": "\U0001F979 HEIST CREW",
            "color": discord.Color.dark_red(),
            "fields": [
                ("Build Your Crew", "`/hire_specialist` Hire safecracker/hacker/driver/etc\n`/crew` View your specialists\n`/targets` See heist targets"),
                ("Pull a Job", "`/heist` Plan a multi-phase heist with crew\n_6 targets from convenience stores to the Federal Reserve._"),
            ],
        },
        "progression": {
            "title": "\U0001F3C6 PROGRESSION",
            "color": discord.Color.green(),
            "fields": [
                ("Level Up", "`/level` Your XP and rank\n`/achievements` Earned badges + perks\n`/quests` Daily & weekly tasks\n`/claimquest` Claim completed quests"),
                ("Tournaments & Rep", "`/tournament` Weekly leaderboard\n`/rep` Give someone reputation\n`/reputation` Check rep"),
            ],
        },
        "shop": {
            "title": "\U0001F6D2 SHOP & ITEMS",
            "color": discord.Color.blue(),
            "fields": [
                ("Main Shop", "**`/shop`** \u2014 Interactive shop (3 pages)\n`/buyrole` Colored Discord role\n`/protect` 12h rob immunity\n`/megaphone` @here announcement\n`/lottery` Lottery tickets"),
                ("Boosts & Cosmetics", "`/vip` 7-day VIP badge \u270D\n`/title` Custom title\n`/xpboost` 2x XP for 24h\n`/insurance` Business protection\n`/heisttools` +20% rob success\n`/lotterymult` 2x lottery"),
                ("Misc", "`/petfood` 7-day pet food bundle\n`/upgradebusiness` Permanent boost\n`/loan` Borrow from loan shark\n`/repay` Pay back loan\n`/bounty` Place bounty on user\n`/bounties` See active bounties"),
            ],
        },
        "settings": {
            "title": "\u2699\uFE0F SETTINGS & MODS",
            "color": discord.Color.greyple(),
            "fields": [
                ("Personal Settings", "`/notifications` Toggle DM notifications\n`/signature` \U0001F3AD BOOSTER: signature reactions"),
                ("Admin", "`/setnotifchannel` Set notification channel\n`/renamechannels` Bulk rename channels\n`/stylepreview` See channel name styles\n`/transcribetoggle` Voice transcription"),
            ],
        },
    }

    cat = categories.get(category, categories["economy"])
    embed = discord.Embed(title=cat["title"], color=cat["color"])
    for name, value in cat["fields"]:
        embed.add_field(name=name, value=value, inline=False)
    embed.set_footer(text="\u2190 Use the buttons to switch categories or go home")
    return embed


class CommandsNavView(discord.ui.View):
    def __init__(self, user_id: int, current: str = "home"):
        super().__init__(timeout=300)
        self.user_id = user_id
        self.current = current

        # Row 0 — main categories
        for label, key, emoji in [
            ("Home", "home", "\U0001F3E0"),
            ("Economy", "economy", "\U0001F4B0"),
            ("Games", "games", "\U0001F3B2"),
            ("Passive", "passive", "\U0001F4C8"),
        ]:
            btn = discord.ui.Button(
                label=label,
                emoji=emoji,
                style=discord.ButtonStyle.primary if key == current else discord.ButtonStyle.secondary,
                row=0,
            )
            btn.callback = self._make_cb(key)
            self.add_item(btn)

        # Row 1 — more categories
        for label, key, emoji in [
            ("Dealer", "dealer", "\U0001F48A"),
            ("Heist", "heist", "\U0001F979"),
            ("Progression", "progression", "\U0001F3C6"),
            ("Shop", "shop", "\U0001F6D2"),
            ("Settings", "settings", "\u2699\uFE0F"),
        ]:
            btn = discord.ui.Button(
                label=label,
                emoji=emoji,
                style=discord.ButtonStyle.primary if key == current else discord.ButtonStyle.secondary,
                row=1,
            )
            btn.callback = self._make_cb(key)
            self.add_item(btn)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Run `/commands` to open your own.", ephemeral=True
            )
            return False
        return True

    def _make_cb(self, key: str):
        async def cb(interaction: discord.Interaction):
            if key == "home":
                embed = _build_commands_home_embed()
            else:
                embed = _build_commands_category_embed(key)
            view = CommandsNavView(self.user_id, current=key)
            await interaction.response.edit_message(embed=embed, view=view)
        return cb


@client.event
async def on_app_command_completion(interaction: discord.Interaction, command):
    """Auto-track every successful slash command for analytics."""
    try:
        track_command_use(command.name, interaction.user.id)
    except Exception as e:
        log.warning("auto command tracking failed: %s", e)


@client.event
async def on_ready():
    log.info("Logged in as %s (ID: %s)", client.user, client.user.id)
    available = [p for p in ["groq","cerebras","gemini"] if os.environ.get(f"{p.upper()}_API_KEY")]
    log.info("Available AI providers: %s", available or "NONE")
    try:
        synced = await tree.sync()
        log.info("Synced %d slash commands", len(synced))
    except Exception as e:
        log.error("Slash command sync failed: %s", e)
    client.loop.create_task(rotate_status())
    client.loop.create_task(silence_checker())
    client.loop.create_task(daily_recap_scheduler())
    client.loop.create_task(expire_temp_roles())
    client.loop.create_task(lottery_drawing_scheduler())
    client.loop.create_task(tournament_scheduler())
    client.loop.create_task(random_event_scheduler())
    client.loop.create_task(business_events_scheduler())
    client.loop.create_task(loan_shark_scheduler())
    client.loop.create_task(pet_starving_scheduler())
    client.loop.create_task(nightlife_events_scheduler())


# ─────────────────────────────────────────────────────────────────────────────
# 🪄 _prefix command router
# Allows users to invoke slash commands via "_command args" in chat.
# Supports user mentions, plain numbers, and quoted strings.
# ─────────────────────────────────────────────────────────────────────────────

class _FakeNamespace:
    """Mimics Discord's interaction namespace for slash command params."""
    pass


class _PrefixInteraction:
    """A minimal interaction wrapper for messages so slash command handlers
    can use the same code path. NOT a real Discord interaction — only the
    methods/attributes the existing slash commands actually use are stubbed."""
    def __init__(self, message: discord.Message):
        self._message = message
        self.user = message.author
        self.channel = message.channel
        self.guild = message.guild
        self.channel_id = message.channel.id
        self.guild_id = message.guild.id if message.guild else None
        self._original_response: discord.Message | None = None
        self._deferred = False
        self.response = self._Response(self)

    class _Response:
        def __init__(self, parent):
            self._parent = parent
            self._done = False

        def is_done(self) -> bool:
            return self._done

        async def send_message(self, content=None, *, embed=None, embeds=None, view=None,
                               ephemeral=False, allowed_mentions=None):
            kwargs = {}
            if content is not None: kwargs["content"] = content
            if embed is not None: kwargs["embed"] = embed
            if embeds is not None: kwargs["embeds"] = embeds
            if view is not None: kwargs["view"] = view
            if allowed_mentions is not None: kwargs["allowed_mentions"] = allowed_mentions
            sent = await self._parent.channel.send(**kwargs)
            self._parent._original_response = sent
            self._done = True
            return sent

        async def defer(self, ephemeral=False):
            self._done = True
            self._parent._deferred = True

        async def send_modal(self, modal):
            # Modals can't be sent without a real interaction; reply with a notice
            await self._parent.channel.send(
                f"⚠️ This command needs an interactive form. Use `/{modal.title}` instead."
            )
            self._done = True

    async def edit_original_response(self, **kwargs):
        if self._original_response:
            return await self._original_response.edit(**kwargs)
        # If we deferred but never sent, send a fresh message now
        sent = await self.channel.send(**{k: v for k, v in kwargs.items() if v is not None})
        self._original_response = sent
        return sent

    async def original_response(self):
        return self._original_response

    @property
    def followup(self):
        return self._Followup(self)

    class _Followup:
        def __init__(self, parent):
            self._parent = parent

        async def send(self, content=None, *, embed=None, embeds=None, view=None,
                       ephemeral=False, allowed_mentions=None, wait=False):
            kwargs = {}
            if content is not None: kwargs["content"] = content
            if embed is not None: kwargs["embed"] = embed
            if embeds is not None: kwargs["embeds"] = embeds
            if view is not None: kwargs["view"] = view
            if allowed_mentions is not None: kwargs["allowed_mentions"] = allowed_mentions
            return await self._parent.channel.send(**kwargs)


def _parse_prefix_args(text: str, message: discord.Message) -> list:
    """Split text into argument tokens. Resolves user mentions, integers, and strings."""
    # First handle quoted segments
    tokens = []
    pattern = re.compile(r'"([^"]+)"|(\S+)')
    for m in pattern.finditer(text):
        tokens.append(m.group(1) if m.group(1) is not None else m.group(2))
    return tokens


def _resolve_member(token: str, message: discord.Message) -> discord.Member | None:
    """Try to convert a token to a Discord Member."""
    if not message.guild:
        return None
    # Mention format
    m = re.match(r"<@!?(\d+)>", token)
    if m:
        return message.guild.get_member(int(m.group(1)))
    # Bare ID
    if token.isdigit():
        return message.guild.get_member(int(token))
    # Display name / username (case-insensitive contains)
    lower = token.lstrip("@").lower()
    for member in message.guild.members:
        if lower in (member.display_name.lower(), member.name.lower()):
            return member
    return None


# Map prefix command names to (slash_command_name, [param_specs])
# param_specs = list of dicts: {"name": str, "type": "user"|"int"|"str"|"rest_str", "required": bool, "default": Any}
PREFIX_COMMANDS = {
    # Economy
    "balance":      ("balance",       [{"name":"user","type":"user","required":False,"default":None}]),
    "bal":          ("balance",       [{"name":"user","type":"user","required":False,"default":None}]),
    "daily":        ("daily",         []),
    "weekly":       ("weekly",        []),
    "work":         ("work",          []),
    "beg":          ("beg",           []),
    "rob":          ("rob",           [{"name":"target","type":"user","required":True}]),
    "pay":          ("pay",           [{"name":"user","type":"user","required":True},{"name":"amount","type":"int","required":True}]),
    "leaderboard":  ("leaderboard",   []),
    "lb":           ("leaderboard",   []),
    "bet":          ("bet",           [{"name":"amount","type":"int","required":True}]),
    # Games
    "fight":        ("fight",         [{"name":"opponent","type":"user","required":True},{"name":"wager","type":"int","required":False,"default":200}]),
    "lieordie":     ("lieordie",      [{"name":"target","type":"user","required":True}]),
    "lod":          ("lieordie",      [{"name":"target","type":"user","required":True}]),
    "duel":         ("duel",          [{"name":"opponent","type":"user","required":True}]),
    "blackjack":    ("blackjack",     [{"name":"bet","type":"int","required":True}]),
    "bj":           ("blackjack",     [{"name":"bet","type":"int","required":True}]),
    "slots":        ("slots",         [{"name":"bet","type":"int","required":False,"default":100}]),
    "wheel":        ("wheel",         []),
    "crime":        ("crime",         []),
    "quest":        ("quest",         []),
    "rs":           ("rs",            [{"name":"racer1","type":"user","required":False,"default":None},{"name":"racer2","type":"user","required":False,"default":None},{"name":"racer3","type":"user","required":False,"default":None}]),
    "race":         ("rs",            [{"name":"racer1","type":"user","required":False,"default":None},{"name":"racer2","type":"user","required":False,"default":None},{"name":"racer3","type":"user","required":False,"default":None}]),
    "casino":       ("casino",        []),
    "shop":         ("shop",          []),
    "shootout":     ("shootout",      [{"name":"buy_in","type":"int","required":False,"default":200}]),
    "bomb":         ("bomb",          [{"name":"target","type":"user","required":True},{"name":"stakes","type":"int","required":False,"default":500}]),
    "connect4":     ("connect4",      [{"name":"opponent","type":"user","required":True},{"name":"wager","type":"int","required":False,"default":200}]),
    "c4":           ("connect4",      [{"name":"opponent","type":"user","required":True},{"name":"wager","type":"int","required":False,"default":200}]),
    "gun":          ("gun",           [{"name":"player2","type":"user","required":True},{"name":"player3","type":"user","required":False,"default":None},{"name":"player4","type":"user","required":False,"default":None}]),
    # AI
    "roast":        ("roast",         [{"name":"user","type":"user","required":True}]),
    "bio":          ("bio",           [{"name":"user","type":"user","required":True}]),
    "court":        ("court",         [{"name":"defendant","type":"user","required":True},{"name":"charge","type":"rest_str","required":True}]),
    "lawsuit":      ("lawsuit",       [{"name":"defendant","type":"user","required":True},{"name":"claim","type":"rest_str","required":True}]),
    "sue":          ("lawsuit",       [{"name":"defendant","type":"user","required":True},{"name":"claim","type":"rest_str","required":True}]),
    "tarot":        ("tarot",         []),
    "analyze":      ("analyze",       []),
    "achievements": ("achievements",  [{"name":"user","type":"user","required":False,"default":None}]),
    "ach":          ("achievements",  [{"name":"user","type":"user","required":False,"default":None}]),
    "level":        ("level",         [{"name":"user","type":"user","required":False,"default":None}]),
    "lvl":          ("level",         [{"name":"user","type":"user","required":False,"default":None}]),
    "rank":         ("level",         [{"name":"user","type":"user","required":False,"default":None}]),
    "tournament":   ("tournament",    []),
    "tourney":      ("tournament",    []),
    "adopt":        ("adopt",         [{"name":"pet_type","type":"str","required":True},{"name":"name","type":"rest_str","required":True}]),
    "pet":          ("pet",           [{"name":"user","type":"user","required":False,"default":None}]),
    "feed":         ("feed",          []),
    "collect":      ("collect",       []),
    "abandon":      ("abandon",       []),
    "rep":          ("rep",           [{"name":"user","type":"user","required":True},{"name":"reason","type":"rest_str","required":False,"default":None}]),
    "reputation":   ("reputation",    [{"name":"user","type":"user","required":False,"default":None}]),
    "quests":       ("quests",        []),
    "claimquest":   ("claimquest",    []),
    "claim":        ("claimquest",    []),
    "buybusiness":  ("buybusiness",   [{"name":"business_type","type":"str","required":True}]),
    "buybiz":       ("buybusiness",   [{"name":"business_type","type":"str","required":True}]),
    "businesses":   ("businesses",    [{"name":"user","type":"user","required":False,"default":None}]),
    "biz":          ("businesses",    [{"name":"user","type":"user","required":False,"default":None}]),
    "collectbusiness":("collectbusiness", []),
    "collectbiz":   ("collectbusiness", []),
    "payday":       ("collectbusiness", []),
    "hire":         ("hire",          [{"name":"employee","type":"user","required":True},{"name":"business_type","type":"str","required":True}]),
    "fire":         ("fire",          [{"name":"employee","type":"user","required":True}]),
    "sabotage":     ("sabotage",      [{"name":"target","type":"user","required":True},{"name":"business_type","type":"str","required":True}]),
    "insurance":    ("insurance",     []),
    "vip":          ("vip",           []),
    "xpboost":      ("xpboost",       []),
    "title":        ("title",         [{"name":"new_title","type":"rest_str","required":False,"default":None}]),
    "lotterymult":  ("lotterymult",   []),
    "loan":         ("loan",          [{"name":"amount","type":"int","required":True}]),
    "repay":        ("repay",         []),
    "bounty":       ("bounty",        [{"name":"target","type":"user","required":True},{"name":"amount","type":"int","required":True}]),
    "bounties":     ("bounties",      []),
    "petfood":      ("petfood",       []),
    "upgradebusiness":("upgradebusiness", [{"name":"business_type","type":"str","required":True}]),
    "upgrade":      ("upgradebusiness", [{"name":"business_type","type":"str","required":True}]),
    "heisttools":   ("heisttools",    []),
    "buysupply":    ("buysupply",     [{"name":"substance","type":"str","required":True},{"name":"grams","type":"int","required":True}]),
    "cop":          ("buysupply",     [{"name":"substance","type":"str","required":True},{"name":"grams","type":"int","required":True}]),
    "sell":         ("sell",          [{"name":"substance","type":"str","required":True},{"name":"grams","type":"int","required":True}]),
    "stash":        ("stash",         [{"name":"user","type":"user","required":False,"default":None}]),
    "streetprice":  ("streetprice",   []),
    "prices":       ("streetprice",   []),
    "market":       ("streetprice",   []),
    "laylow":       ("laylow",        []),
    "dealers":      ("dealers",       []),
    "plug":         ("dealers",       []),
    "dealer":       ("dealer",        []),
    "dashboard":    ("dealer",        []),
    "buyvenue":     ("buyvenue",      [{"name":"venue_type","type":"str","required":True}]),
    "venues":       ("venues",        [{"name":"user","type":"user","required":False,"default":None}]),
    "club":         ("venues",        [{"name":"user","type":"user","required":False,"default":None}]),
    "collectvenue": ("collectvenue",  []),
    "nightstake":   ("collectvenue",  []),
    "hirestaff":    ("hirestaff",     [{"name":"venue_type","type":"str","required":True},{"name":"role","type":"str","required":True}]),
    "stockliquor":  ("stockliquor",   [{"name":"venue_type","type":"str","required":True},{"name":"liquor","type":"str","required":True},{"name":"bottles","type":"int","required":True}]),
    "notifications":("notifications", []),
    "notif":        ("notifications", []),
    "dms":          ("notifications", []),
    "hack":         ("hack",          [{"name":"target","type":"user","required":True}]),
    # Fun
    "ship":         ("ship",          [{"name":"user1","type":"str","required":True},{"name":"user2","type":"str","required":True}]),
    "rate":         ("rate",          [{"name":"thing","type":"rest_str","required":True}]),
    "8ball":        ("8ball",         [{"name":"question","type":"rest_str","required":True}]),
    "rps":          ("rps",           [{"name":"choice","type":"str","required":True}]),
    "roll":         ("roll",          [{"name":"sides","type":"int","required":False,"default":100}]),
    "flip":         ("flip",          []),
    "marry":        ("marry",         [{"name":"user","type":"user","required":True}]),
    "divorce":      ("divorce",       []),
    # Shop & redeem
    "protect":      ("protect",       []),
    "lottery":      ("lottery",       [{"name":"tickets","type":"int","required":False,"default":0}]),
    "megaphone":    ("megaphone",     [{"name":"message","type":"rest_str","required":True}]),
    # Meta
    "commands":     ("commands",      []),
    "help":         ("commands",      []),
}


async def handle_prefix_command(message: discord.Message, body: str) -> bool:
    """Parse and dispatch a `_command args` style command. Returns True if handled."""
    # Split into command name + rest
    parts = body.split(maxsplit=1)
    cmd_name = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    spec = PREFIX_COMMANDS.get(cmd_name)
    if not spec:
        return False

    slash_name, param_specs = spec

    # Find the slash command in the tree
    slash_cmd = tree.get_command(slash_name)
    if slash_cmd is None:
        await message.channel.send(f"⚠️ Internal error: `/{slash_name}` not registered.")
        return True

    # Parse arguments based on the spec
    kwargs = {}
    tokens = _parse_prefix_args(rest, message)
    token_idx = 0

    for ps in param_specs:
        ptype = ps["type"]
        if ptype == "rest_str":
            # Consume all remaining tokens as one string
            value = " ".join(tokens[token_idx:]).strip()
            if not value and ps.get("required"):
                await message.channel.send(f"❌ `{cmd_name}` needs a `{ps['name']}` argument.")
                return True
            kwargs[ps["name"]] = value
            token_idx = len(tokens)
            continue

        if token_idx >= len(tokens):
            if ps.get("required"):
                await message.channel.send(
                    f"❌ `{cmd_name}` is missing required arg `{ps['name']}`. Try `_{cmd_name} <{ps['name']}>`."
                )
                return True
            kwargs[ps["name"]] = ps.get("default")
            continue

        tok = tokens[token_idx]
        token_idx += 1

        if ptype == "user":
            member = _resolve_member(tok, message)
            if not member:
                if ps.get("required"):
                    await message.channel.send(f"❌ Couldn't find user `{tok}` for `{cmd_name}`.")
                    return True
                kwargs[ps["name"]] = ps.get("default")
            else:
                kwargs[ps["name"]] = member
        elif ptype == "int":
            try:
                kwargs[ps["name"]] = int(tok)
            except ValueError:
                await message.channel.send(f"❌ `{ps['name']}` must be a number for `{cmd_name}`.")
                return True
        elif ptype == "str":
            kwargs[ps["name"]] = tok
        else:
            kwargs[ps["name"]] = tok

    # Handle Choice-typed slash params (rps and buyrole use Choice)
    if slash_name == "rps":
        # rps takes a Choice; build one
        choice_value = kwargs.get("choice", "").lower()
        if choice_value not in ("rock", "paper", "scissors"):
            await message.channel.send("❌ Pick rock, paper, or scissors.")
            return True
        kwargs["choice"] = discord.app_commands.Choice(name=choice_value, value=choice_value)

    if slash_name == "adopt":
        # adopt takes a pet_type Choice
        pet_type = kwargs.get("pet_type", "").lower()
        if pet_type not in PET_TYPES:
            await message.channel.send(
                f"❌ Pet types: {', '.join(PET_TYPES.keys())}"
            )
            return True
        kwargs["pet_type"] = discord.app_commands.Choice(name=pet_type, value=pet_type)

    if slash_name in ("buybusiness", "hire", "sabotage"):
        # These take a business_type Choice
        biz_type = kwargs.get("business_type", "").lower()
        if biz_type not in BUSINESS_TYPES:
            await message.channel.send(
                f"❌ Business types: {', '.join(BUSINESS_TYPES.keys())}"
            )
            return True
        kwargs["business_type"] = discord.app_commands.Choice(name=biz_type, value=biz_type)

    if slash_name in ("buysupply", "sell"):
        # Substance Choice for dealer commands
        sub_key = kwargs.get("substance", "").lower()
        if sub_key not in SUBSTANCES:
            await message.channel.send(
                f"❌ Substances: {', '.join(SUBSTANCES.keys())}"
            )
            return True
        kwargs["substance"] = discord.app_commands.Choice(name=sub_key, value=sub_key)

    if slash_name in ("buyvenue", "hirestaff", "stockliquor"):
        v_type = kwargs.get("venue_type", "").lower()
        if v_type not in VENUE_TYPES:
            await message.channel.send(
                f"❌ Venue types: {', '.join(VENUE_TYPES.keys())}"
            )
            return True
        kwargs["venue_type"] = discord.app_commands.Choice(name=v_type, value=v_type)

    if slash_name == "hirestaff":
        r_key = kwargs.get("role", "").lower()
        if r_key not in STAFF_ROLES:
            await message.channel.send(
                f"❌ Staff roles: {', '.join(STAFF_ROLES.keys())}"
            )
            return True
        kwargs["role"] = discord.app_commands.Choice(name=r_key, value=r_key)

    if slash_name == "stockliquor":
        l_key = kwargs.get("liquor", "").lower()
        if l_key not in LIQUOR_TYPES:
            await message.channel.send(
                f"❌ Liquor types: {', '.join(LIQUOR_TYPES.keys())}"
            )
            return True
        kwargs["liquor"] = discord.app_commands.Choice(name=l_key, value=l_key)

    if slash_name == "upgradebusiness":
        biz_type = kwargs.get("business_type", "").lower()
        if biz_type not in BUSINESS_TYPES:
            await message.channel.send(
                f"❌ Business types: {', '.join(BUSINESS_TYPES.keys())}"
            )
            return True
        kwargs["business_type"] = discord.app_commands.Choice(name=biz_type, value=biz_type)

    if slash_name == "loan":
        try:
            amt = int(kwargs.get("amount", 0))
        except Exception:
            amt = 0
        if amt not in LOAN_AMOUNTS:
            await message.channel.send(
                f"❌ Loan amounts: {', '.join(str(a) for a in LOAN_AMOUNTS)}"
            )
            return True
        kwargs["amount"] = discord.app_commands.Choice(name=str(amt), value=amt)

    # Build a fake interaction and invoke the underlying callback
    fake = _PrefixInteraction(message)
    try:
        # The callback is the underlying coroutine — call it directly
        await slash_cmd.callback(fake, **kwargs)
    except Exception as e:
        log.exception("prefix command failed")
        try:
            await message.channel.send(f"⚠️ `{cmd_name}` errored: {e}")
        except Exception:
            pass
    return True


@client.event
async def on_message(message: discord.Message):
    if message.author.bot and message.author != client.user:
        last_channel_activity[str(message.channel.id)] = time.time()
        return
    if message.author == client.user:
        return

    cfg = load_config()
    memory.update_maxlen(cfg["max_memory_messages"])
    channel_id = str(message.channel.id)
    last_channel_activity[channel_id] = time.time()

    # ── Auto-react: react to anyone replying to a target user (all channels) ──
    try:
        await handle_autoreact(message)
    except Exception as e:
        log.warning("autoreact failed: %s", e)

    # ── Voice message transcription (all channels) ────────────────────────────
    try:
        if message.attachments:
            asyncio.create_task(handle_voice_transcription(message))
    except Exception as e:
        log.warning("voice transcription failed: %s", e)

    # ── Random events: check if this answers an active event ─────────────────
    try:
        if await check_random_event_answer(message):
            return
    except Exception as e:
        log.warning("random event check failed: %s", e)

    # ── XP for chatting (cooldown to prevent farming) ─────────────────────────
    try:
        now_ts = time.time()
        last = _xp_last_message.get(message.author.id, 0)
        if now_ts - last >= XP_MESSAGE_COOLDOWN and message.content.strip():
            _xp_last_message[message.author.id] = now_ts
            await grant_xp(message.author.id, XP_PER_MESSAGE, channel=message.channel)
    except Exception as e:
        log.warning("xp grant failed: %s", e)

    # ── _prefix commands → invoke slash commands ─────────────────────────────
    content_stripped = message.content.strip()
    if content_stripped.startswith("_") and len(content_stripped) > 1:
        handled = await handle_prefix_command(message, content_stripped[1:])
        if handled:
            # Bonus XP for using a command
            try:
                await grant_xp(message.author.id, XP_PER_COMMAND, channel=message.channel)
                # Track command use for tournament scoring
                add_tournament_score(message.author.id, commands_used=1)
                # Track quest progress
                track_quest_progress(message.author.id, "commands_used")
            except Exception:
                pass
            return

    # Commands
    if message.content.strip() == "!clearhistory":
        memory.clear(channel_id)
        await message.reply("🧹 Conversation history cleared for this channel.")
        return

    if message.content.strip() == "!botinfo":
        embed = discord.Embed(title=f"🤖 {cfg['name']}", color=discord.Color.blurple())
        embed.add_field(name="Providers", value=", ".join(cfg.get("provider_order", [])), inline=False)
        embed.add_field(name="Trigger", value=cfg["trigger_mode"], inline=True)
        embed.add_field(name="Memory", value=f"{cfg['max_memory_messages']} msgs", inline=True)
        chime = f"Every {cfg['chime_in_min']}–{cfg['chime_in_max']} msgs" if cfg.get("chime_in_enabled") else "Off"
        embed.add_field(name="Chime-in", value=chime, inline=True)
        embed.add_field(name="Vision", value="On" if cfg.get("vision_enabled") else "Off", inline=True)
        embed.add_field(name="Daily recap", value="On" if cfg.get("daily_recap_enabled") else "Off", inline=True)
        available = [p for p in ["groq","cerebras","gemini"] if os.environ.get(f"{p.upper()}_API_KEY")]
        embed.add_field(name="Keys loaded", value=", ".join(available) or "none", inline=True)
        await message.reply(embed=embed)
        return

    # Manual recap trigger (admin only — anyone can use, but it'll recap today's messages so far)
    if message.content.strip() == "!recap":
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logs = daily_log.read_day(channel_id, today)
        if len(logs) < 5:
            await message.reply("Not enough messages yet today for a recap.")
            return
        transcript = "\n".join(f"{m['author']}: {m['content']}" for m in logs[-300:])
        system = cfg["system_prompt"] + (
            "\n\nWrite a short recap of the chat so far today. "
            "Pick out the funniest or most ridiculous moments. 3-5 highlights max. "
            "Stay in character. Be ruthless."
        )
        async with message.channel.typing():
            recap = await ask_ai(
                system,
                [{"role": "user", "content": f"Recap this:\n\n{transcript}"}],
                {**cfg, "max_tokens": 400},
            )
        if not recap.startswith("⚠️"):
            await message.reply(f"📰 **Recap so far today:**\n\n{recap}")
        return

    if str(message.author.id) in cfg["blocked_users"]:
        return
    if not channel_allowed(channel_id, cfg):
        return

    # Log for daily recap
    if message.content.strip():
        daily_log.append(channel_id, message.author.display_name, message.content.strip())

    # Passive watching — log every message
    if cfg.get("watch_mode_enabled", True):
        is_respected = str(message.author.id) in cfg.get("respected_users", [])
        tag = "⭐BOSS⭐ " if is_respected else ""
        watch_line = f"[{tag}{message.author.display_name}]: {message.content}"
        memory.append(channel_id, "user", watch_line)

    # Emoji reaction chance
    if cfg.get("reaction_enabled") and random.random() < cfg.get("reaction_chance", 0.05):
        try:
            await message.add_reaction(random.choice(REACTION_POOL))
        except Exception as e:
            log.debug("Reaction failed: %s", e)

    # Chime-in counter
    chime_triggered = False
    if cfg.get("chime_in_enabled"):
        message_counters[channel_id] += 1
        if channel_id not in chime_thresholds:
            chime_thresholds[channel_id] = get_chime_threshold(cfg)
        if message_counters[channel_id] >= chime_thresholds[channel_id]:
            message_counters[channel_id] = 0
            chime_thresholds[channel_id] = get_chime_threshold(cfg)
            chime_triggered = True

    direct = is_direct_trigger(message, cfg)

    if not direct and not chime_triggered:
        return

    # ── Check for image attachments (only on direct triggers) ────────────────
    images = []
    if direct and cfg.get("vision_enabled"):
        images = await download_images(message)

    # Chime-in response
    if chime_triggered and not direct:
        history = memory.get_messages(channel_id)
        chime_system = (
            cfg["system_prompt"] +
            "\n\nYou are chiming into an ongoing conversation uninvited. "
            "You've been silently reading every message. Reference specific things people said, "
            "call users out by name if relevant (you can see their display names in [brackets] before each message). "
            "Arrive like you've been listening the whole time and have already formed a verdict. "
            "Keep it short — 1 to 3 sentences max. Do NOT announce that you're joining. "
            + get_time_context()
        )
        async with message.channel.typing():
            reply = await ask_ai(
                chime_system,
                history if history else [{"role": "user", "content": message.content}],
                {**cfg, "temperature": min(cfg["temperature"] + 0.1, 1.0)},
            )
        if not reply.startswith("⚠️"):
            memory.append(channel_id, "assistant", reply)
            last_bot_activity[channel_id] = time.time()
            await send_reply(message, reply, chimed_in=True)
            log.info("Chimed in on #%s", message.channel)
        return

    # Direct reply
    content = message.content
    if client.user.mentioned_in(message):
        content = content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()

    if not content and not images:
        await message.reply("what")
        return

    if not cfg.get("watch_mode_enabled", True):
        is_respected = str(message.author.id) in cfg.get("respected_users", [])
        tag = "⭐BOSS⭐ " if is_respected else ""
        memory.append(channel_id, "user", f"[{tag}{message.author.display_name}]: {content}")

    async with message.channel.typing():
        history = memory.get_messages(channel_id)
        # If there are images, enrich the system prompt slightly so Jordan knows to roast them
        sys_prompt = cfg["system_prompt"] + "\n\n" + get_time_context()
        if images:
            sys_prompt += (
                "\n\nThe user just posted one or more images. Look at them and react in character. "
                "Roast what you see. Reference details from the image."
            )
            if not content:
                # If no text, replace the last user message with an image prompt
                history = history + [{"role": "user", "content": f"[{message.author.display_name} posted an image]"}]
        reply = await ask_ai(sys_prompt, history, cfg, images=images if images else None)

    if not reply.startswith("⚠️"):
        memory.append(channel_id, "assistant", reply)
        last_bot_activity[channel_id] = time.time()
    await send_reply(message, reply)
    log.info("Replied to %s in #%s %s", message.author, message.channel,
             f"(with {len(images)} image(s))" if images else "")


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set.")
    if not any(os.environ.get(f"{p.upper()}_API_KEY") for p in ["groq","cerebras","gemini"]):
        raise RuntimeError("No AI provider keys set. Add at least GROQ_API_KEY, CEREBRAS_API_KEY, or GEMINI_API_KEY.")
    client.run(DISCORD_TOKEN, log_handler=None)

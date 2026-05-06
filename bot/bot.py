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
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULT_PERSONALITY, **json.load(f)}
    return DEFAULT_PERSONALITY.copy()


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

    def record_win(self, user_id: int):
        u = self._user(user_id)
        u["stats"]["games_won"] = u["stats"].get("games_won", 0) + 1
        self._save()

    def record_loss(self, user_id: int):
        u = self._user(user_id)
        u["stats"]["games_lost"] = u["stats"].get("games_lost", 0) + 1
        self._save()

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
    last_posted_date = None
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

        # Only fire within the configured hour, and only once per day
        if now.hour != target_hour:
            continue
        today_str = now.strftime("%Y-%m-%d")
        if last_posted_date == today_str:
            continue

        # Recap the PREVIOUS day's messages
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        # Gather from all channels that had activity yesterday, but primarily the recap channel
        logs = daily_log.read_day(recap_channel_id, yesterday)
        if len(logs) < 10:
            log.info("Not enough messages for recap (%d), skipping", len(logs))
            last_posted_date = today_str
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
                channel = client.get_channel(int(recap_channel_id))
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
                    last_posted_date = today_str
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
@tree.command(name="court", description="Put a user on trial for a crime. AI judges the case.")
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
@tree.command(name="duel", description="Pistol duel between two users at dawn.")
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
@tree.command(name="gun", description="Russian roulette. 1 in 6 chance per pull. Last person standing wins.")
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

@tree.command(name="heist", description="Pull off a bank heist with your crew.")
@discord.app_commands.describe(crew1="Crew member", crew2="Crew member", crew3="Crew member", crew4="Crew member")
async def heist_command(
    interaction: discord.Interaction,
    crew1: discord.Member,
    crew2: discord.Member = None,
    crew3: discord.Member = None,
    crew4: discord.Member = None,
):
    silent = discord.AllowedMentions.none()
    starter = interaction.user
    candidates = [starter, crew1, crew2, crew3, crew4]
    seen = set()
    crew = []
    for u in candidates:
        if u and u.id not in seen and not u.bot:
            seen.add(u.id)
            crew.append(u)
    if len(crew) < 2:
        await interaction.response.send_message("Need at least 2 unique crew members.", ephemeral=True)
        return

    await interaction.response.defer()

    async def edit(content):
        try:
            await interaction.edit_original_response(content=content, allowed_mentions=silent)
        except Exception:
            pass

    crew_str = ", ".join(p.mention for p in crew)
    await edit(f"🚐 **THE HEIST BEGINS**\n\n{crew_str} pull up to the bank...")
    await asyncio.sleep(2.0)

    stages = [
        "🔍 Scoping the perimeter...",
        "💣 Planting the explosives...",
        "🚨 ALARMS TRIGGERED!",
        "🏃 Crew runs into the vault!",
        "💰 Cracking the safe...",
    ]
    for stage in stages:
        await edit(f"🚐 **HEIST IN PROGRESS**\n\n{crew_str}\n\n{stage}")
        await asyncio.sleep(1.6)

    # 15% chance the heist fails
    if random.random() < 0.15:
        caught = random.choice(crew)
        # Caught crew member pays a fine
        fine = min(300, economy.balance(caught.id))
        economy.add(caught.id, -fine, "heist caught")
        for p in crew:
            economy.record_loss(p.id)
        final = (
            f"🚔 **HEIST FAILED!**\n\n"
            f"The cops showed up. {caught.mention} got pinned to the wall.\n"
            f"The rest of the crew escaped with **NOTHING**.\n"
            f"💀 {caught.display_name} paid a **{fine:,}** coin fine.\n\n"
            f"The crew: {crew_str}"
        )
        try:
            await interaction.edit_original_response(content=final)
        except Exception:
            pass
        return

    # Otherwise everyone gets a random share — REAL coins now
    payouts = []
    total = 0
    for p in crew:
        amount = random.randint(200, 1500)
        loot_emoji, loot_name = random.choice(HEIST_LOOT)
        economy.add(p.id, amount, "heist payout")
        economy.record_win(p.id)
        payouts.append((p, amount, loot_emoji, loot_name))
        total += amount

    # Sort by haul descending
    payouts.sort(key=lambda x: x[1], reverse=True)
    biggest = payouts[0]

    payout_lines = "\n".join(
        f"{['👑','🥈','🥉','🎖️','🎖️'][i] if i < 5 else '•'} {p.mention} — 💰 **{amt:,}** coins {emoji} _{name}_"
        for i, (p, amt, emoji, name) in enumerate(payouts)
    )

    final = (
        f"💰 **HEIST SUCCESSFUL!** 💰\n\n"
        f"The crew got out clean with **{total:,}** coins in loot!\n\n"
        f"**Cuts:**\n{payout_lines}\n\n"
        f"🏆 {biggest[0].mention} took the biggest haul."
    )
    try:
        await interaction.edit_original_response(content=final)
    except Exception:
        pass


# ── 💻 /hack ─────────────────────────────────────────────────────────────────
HACK_FAKE_FINDINGS = [
    "browser history: 7,432 pages of *cat videos*",
    "Spotify wrapped: 'Mr. Brightside' (843 plays)",
    "screen time average: 11h 47m daily 💀",
    "Twitter drafts folder: 23 unsent posts",
    "Photos folder: 89% screenshots",
    "search history: 'how to talk to women'",
    "deleted texts to ex: 47",
    "$3,800 spent on Uber Eats this month",
    "DoorDash account: 'extra ranch' on every order",
    "Notes app: 'business ideas' (empty since 2021)",
    "TikTok For You Page: BRAINROT detected",
    "Apple Wallet: 1 punch card to a closed café",
    "draft text to mom: 'pls send money'",
    "Google search: 'is it normal to-' x 200",
    "iCloud full of blurry concert videos",
    "Steam library: 487 games, 2 played",
    "screenshot of an email from 2019 still saved",
    "$0.32 in checking, $4,200 on a credit card",
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
    embed = discord.Embed(
        title=f"{COIN_EMOJI} {target.display_name}'s Wallet",
        color=discord.Color.gold(),
    )
    embed.add_field(name="Balance", value=f"**{bal:,}** coins", inline=False)
    embed.add_field(name="Wins", value=str(stats.get("games_won", 0)), inline=True)
    embed.add_field(name="Losses", value=str(stats.get("games_lost", 0)), inline=True)
    embed.add_field(name="Total Earned", value=f"{stats.get('total_earned',0):,}", inline=True)
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
    new_bal = economy.add(user.id, reward, "daily")
    await edit(
        f"🎁 **DAILY REWARD!**\n\n"
        f"You found {COIN_EMOJI} **{reward:,}** coins!\n"
        f"Balance: **{new_bal:,}**"
    )


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
    await edit(
        f"💎 **WEEKLY REWARD!**\n\n"
        f"You scored {COIN_EMOJI} **{reward:,}** coins!\n"
        f"Balance: **{new_bal:,}**"
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
    economy.set_cooldown(user.id, "work")

    await edit(f"{emoji} *Clocking in at {job}...*")
    await asyncio.sleep(1.0)
    await edit(f"{emoji} *Working hard at {job}...*")
    await asyncio.sleep(1.2)
    await edit(f"{emoji} *Almost done at {job}...*")
    await asyncio.sleep(1.0)
    new_bal = economy.add(user.id, reward, "work")
    await edit(
        f"{emoji} **JOB COMPLETE!**\n\n"
        f"You worked at *{job}* and earned {COIN_EMOJI} **{reward:,}** coins.\n"
        f"Balance: **{new_bal:,}**"
    )


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

    reward = random.randint(5, 80)
    await edit(f"🥺 *begging on the corner...*")
    await asyncio.sleep(1.0)
    await edit(f"🤲 *someone is approaching...*")
    await asyncio.sleep(1.0)
    new_bal = economy.add(user.id, reward, "beg")
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

    await edit(f"🦝 {user.mention} is sneaking up on {target.mention}...")
    await asyncio.sleep(1.5)
    await edit(f"🦝 *picking the lock...*")
    await asyncio.sleep(1.5)
    await edit(f"🦝 *reaching into the wallet...*")
    await asyncio.sleep(1.5)

    if random.random() < ROB_SUCCESS_CHANCE:
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

    success = economy.transfer(sender.id, user.id, amount)
    if not success:
        bal = economy.balance(sender.id)
        await interaction.response.send_message(
            f"❌ Insufficient funds. You have **{bal:,}** coins.",
            ephemeral=True,
        )
        return

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
        lines.append(f"{prefix} **{name}** — {COIN_EMOJI} {data.get('balance', 0):,}")

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
        await edit(
            f"## 🎉 YOU WON!\n"
            f"{user.mention} doubled up — gained **{amount:,}** coins!\n"
            f"Balance: **{new_bal:,}**"
        )
    else:
        new_bal = economy.add(user.id, -amount, "bet loss")
        economy.record_loss(user.id)
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
        # Pay 2.5x for natural
        winnings = int(bet * 2.5)
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

    remaining = economy.get_cooldown_remaining(user.id, "rob")  # share rob cooldown
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

    economy.set_cooldown(user.id, "rob")

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
    remaining = economy.get_cooldown_remaining(user.id, "work")  # share work cooldown
    if remaining > 0:
        await interaction.response.send_message(
            f"⏰ The wheel needs to cool down. Try again in **{fmt_cooldown(remaining)}**.",
            ephemeral=True,
        )
        return

    economy.set_cooldown(user.id, "work")
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

    # Pick weighted outcome
    weights = [o["weight"] for o in WHEEL_OUTCOMES]
    outcome = random.choices(WHEEL_OUTCOMES, weights=weights, k=1)[0]

    label = outcome["label"]
    result_text = ""
    if outcome["type"] == "gain":
        new_bal = economy.add(user.id, outcome["amount"], "wheel gain")
        result_text = f"You gained **{outcome['amount']:,}** coins!\nBalance: **{new_bal:,}**"
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
    last_drawn_date = None
    while not client.is_closed():
        await asyncio.sleep(300)  # check every 5 min
        cfg = load_config()
        target_hour = cfg.get("daily_recap_hour_utc", 4)
        now = datetime.now(timezone.utc)
        if now.hour != target_hour:
            continue
        today_str = now.strftime("%Y-%m-%d")
        if last_drawn_date == today_str:
            continue

        try:
            data = _load_lottery()
            if not data["tickets"]:
                last_drawn_date = today_str
                continue

            # Build weighted draw — each user weighted by their ticket count
            entries = []
            for uid, count in data["tickets"].items():
                entries.extend([uid] * count)
            if not entries:
                last_drawn_date = today_str
                continue
            winner_uid = random.choice(entries)
            jackpot = data["jackpot"]

            # Pay out
            economy.add(int(winner_uid), jackpot, "lottery jackpot")
            economy.record_win(int(winner_uid))
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
            recap_channel_id = cfg.get("daily_recap_channel", "").strip()
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
            last_drawn_date = today_str
        except Exception as e:
            log.exception("Lottery drawing error: %s", e)


# ── 🛒 /shop — interactive shop hub ──────────────────────────────────────────

class ShopMainView(discord.ui.View):
    """Top-level shop menu — buttons for each category."""
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
        title="🛒 SHOP",
        description=(
            f"Welcome, **{name}**. Pick a category below.\n\n"
            f"🎨 **Colored Role** — {ROLE_PRICES['24h']:,}–{ROLE_PRICES['perm']:,} coins\n"
            f"🛡️ **Rob Immunity (12h)** — {PROTECT_PRICE:,} coins\n"
            f"📢 **Megaphone (@here)** — {MEGAPHONE_PRICE:,} coins\n"
            f"🎰 **Lottery Tickets** — {LOTTERY_TICKET_PRICE:,} coins each"
        ),
        color=discord.Color.gold(),
    )
    embed.set_footer(text=f"💰 Your balance: {bal:,}")
    view = ShopMainView(user_id)
    if edit:
        await interaction.response.edit_message(embed=embed, view=view)
    else:
        await interaction.response.send_message(embed=embed, view=view)


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


@tree.command(name="fight", description="Challenge someone to a 60-second fight. Spectators can bet.")
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

    # Send the initial fight card with betting buttons
    view = FightBetView(channel_id)
    header = (
        f"# 🥊 FIGHT NIGHT 🥊\n\n"
        f"**Betting closes in {FIGHT_BETTING_WINDOW} seconds!**"
    )
    await interaction.response.send_message(
        content=_fight_status_text(fight, header),
        view=view,
        allowed_mentions=discord.AllowedMentions(users=[opponent]),
    )

    # Run the full fight asynchronously so we don't hold the interaction
    asyncio.create_task(_run_fight(interaction, channel_id))


async def _run_fight(interaction: discord.Interaction, channel_id: str):
    fight = ACTIVE_FIGHTS.get(channel_id)
    if not fight:
        return
    channel = interaction.channel

    # ── Betting window countdown ──────────────────────────────────────────────
    countdown_marks = [20, 10, 5]
    elapsed = 0
    next_marks = list(countdown_marks)
    while elapsed < FIGHT_BETTING_WINDOW:
        await asyncio.sleep(5)
        elapsed += 5
        remaining = FIGHT_BETTING_WINDOW - elapsed
        if next_marks and remaining <= next_marks[0]:
            mark = next_marks.pop(0)
            try:
                await channel.send(
                    f"⏰ **{mark} seconds** left to bet on this fight!\n\n"
                    + _fight_status_text(fight)
                )
            except Exception:
                pass

    # Close betting
    fight["phase"] = "fighting"

    # Pre-fight intro
    try:
        await channel.send(
            f"# 🔔 BETTING CLOSED — FIGHT STARTING\n\n"
            + _fight_status_text(fight)
        )
    except Exception:
        pass
    await asyncio.sleep(2.0)

    # ── Fight animation ───────────────────────────────────────────────────────
    a_score = 0
    b_score = 0
    log_lines = [
        f"🔔 **DING DING DING!** {fight['a_mention']} and {fight['b_mention']} step into the ring."
    ]

    fight_msg = await channel.send(
        f"# 🥊 LIVE FIGHT 🥊\n\n"
        f"🔴 {fight['a_name']}: **{a_score}** | 🔵 {fight['b_name']}: **{b_score}**\n\n"
        + "\n".join(log_lines[-6:]),
        allowed_mentions=discord.AllowedMentions.none(),
    )

    fight_start = time.time()
    tick = 0
    while time.time() - fight_start < FIGHT_DURATION:
        await asyncio.sleep(random.uniform(3.5, 5.5))  # variable tempo
        tick += 1

        # Decide attacker — slight randomization, occasional double action
        attacker_id = random.choice([fight["a_id"], fight["b_id"]])
        attacker_name = fight["a_name"] if attacker_id == fight["a_id"] else fight["b_name"]
        defender_name = fight["b_name"] if attacker_id == fight["a_id"] else fight["a_name"]

        emoji, action = random.choice(FIGHT_MOVES)

        # Random outcome: hit, miss, counter
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
            # Miss
            log_lines.append(f"💨 **{attacker_name}** swings — and {defender_name} dodges it.")
        else:
            # Counter — defender hits back
            damage = random.randint(2, 5)
            if attacker_id == fight["a_id"]:
                b_score += damage
            else:
                a_score += damage
            log_lines.append(
                f"🌀 **{defender_name}** counters! **{attacker_name}** eats it. **(+{damage} for {defender_name})**"
            )

        # Occasional trash talk
        if random.random() < 0.18:
            talker = random.choice([fight["a_name"], fight["b_name"]])
            log_lines.append(f"🗯️ {talker} {random.choice(FIGHT_TRASH_TALK)}.")

        # Update message
        try:
            await fight_msg.edit(
                content=(
                    f"# 🥊 LIVE FIGHT 🥊\n\n"
                    f"🔴 **{fight['a_name']}**: **{a_score}** | 🔵 **{fight['b_name']}**: **{b_score}**\n"
                    f"⏰ Time left: **{int(FIGHT_DURATION - (time.time() - fight_start))}s**\n\n"
                    + "\n".join(log_lines[-6:])
                )
            )
        except Exception:
            pass

    # ── Decide winner ─────────────────────────────────────────────────────────
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
        # Coin flip on tie
        if random.choice([True, False]):
            winner_id, loser_id = fight["a_id"], fight["b_id"]
            winner_name = fight["a_name"]
            winning_side = "a"
        else:
            winner_id, loser_id = fight["b_id"], fight["a_id"]
            winner_name = fight["b_name"]
            winning_side = "b"

    # Pay the fighter wager pot to the winner (2x their wager)
    fighter_payout = fight["wager"] * 2
    economy.add(winner_id, fighter_payout, "fight win")
    economy.record_win(winner_id)
    economy.record_loss(loser_id)

    # ── Resolve spectator bets (parimutuel) ───────────────────────────────────
    a_pool = sum(b["amount"] for b in fight["bets"].values() if b["side"] == "a")
    b_pool = sum(b["amount"] for b in fight["bets"].values() if b["side"] == "b")
    total_pool = a_pool + b_pool
    winning_pool = a_pool if winning_side == "a" else b_pool
    losing_pool = b_pool if winning_side == "a" else a_pool

    payout_lines = []
    if winning_pool > 0:
        for uid, b in fight["bets"].items():
            if b["side"] == winning_side:
                # Each winner gets their stake back PLUS proportional share of losing pool
                share = (b["amount"] / winning_pool) * losing_pool
                payout = b["amount"] + int(share)
                economy.add(uid, payout, "fight bet win")
                profit = payout - b["amount"]
                payout_lines.append(f"✅ <@{uid}> bet {b['amount']:,} → won **{payout:,}** (+{profit:,})")
            else:
                payout_lines.append(f"❌ <@{uid}> bet {b['amount']:,} → **LOST**")
    else:
        # No one bet on the winner — refund losers (rare edge case)
        for uid, b in fight["bets"].items():
            economy.add(uid, b["amount"], "fight bet refund")
            payout_lines.append(f"↩️ <@{uid}> bet {b['amount']:,} → **refunded** (no winning side)")

    new_winner_bal = economy.balance(winner_id)
    final = (
        f"# 🏆 FIGHT OVER 🏆\n\n"
        f"## {winner_name} WINS!\n"
        f"Final score: 🔴 **{a_score}** — **{b_score}** 🔵\n\n"
        f"💰 **{winner_name}** earned **{fighter_payout:,}** coins from the wager pool.\n"
        f"Balance: **{new_winner_bal:,}**\n\n"
    )
    if payout_lines:
        final += f"**Spectator payouts ({total_pool:,} coin pool):**\n" + "\n".join(payout_lines)
    else:
        final += "_No spectators placed bets._"

    if len(final) > 2000:
        final = final[:1990] + "..."

    try:
        winner_member = interaction.guild.get_member(winner_id) if interaction.guild else None
        await channel.send(
            content=final,
            allowed_mentions=discord.AllowedMentions(users=[winner_member] if winner_member else False),
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


# ── ⚖️ /lawsuit — sue a user, AI judges, real damages ───────────────────────
LAWSUIT_FILING_FEE = 200


@tree.command(name="lawsuit", description="Sue another user. AI judge awards (or denies) damages.")
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


@tree.command(name="commands", description="List all available bot commands.")
async def commands_command(interaction: discord.Interaction):
    embed = discord.Embed(title="🎮 Bot Commands", color=discord.Color.blurple())
    embed.add_field(
        name="💰 ECONOMY",
        value=(
            "`/balance` Check coins\n"
            "`/daily` Daily reward\n"
            "`/weekly` Weekly reward\n"
            "`/work` Earn from a job\n"
            "`/beg` Beg for change\n"
            "`/rob` Try to rob someone\n"
            "`/pay` Send coins\n"
            "`/bet` Coinflip wager\n"
            "`/leaderboard` Richest users"
        ),
        inline=False
    )
    embed.add_field(
        name="🛒 SHOP & REDEEM",
        value=(
            "**`/shop` — Interactive shop menu (everything in one place)**\n"
            "`/buyrole` Colored Discord role\n"
            "`/protect` 12h rob immunity\n"
            "`/megaphone` Channel-wide announcement\n"
            "`/lottery` Buy tickets / view jackpot"
        ),
        inline=False
    )
    embed.add_field(
        name="🎲 GAMES (earn coins)",
        value=(
            "`/fight` Fight w/ spectator betting\n"
            "`/lieordie` Vote True/False on AI fact\n"
            "`/quest` AI choose-your-adventure\n"
            "`/shootout` Door elimination\n"
            "`/bomb` Hot potato\n"
            "`/connect4` Connect 4 PvP\n"
            "`/blackjack` Beat the dealer\n"
            "`/casino` Casino menu\n"
            "`/wheel` Wheel of fortune\n"
            "`/crime` Random crime\n"
            "`/rs` Race (tag racers)\n"
            "`/duel` Pistol duel\n"
            "`/gun` Russian roulette\n"
            "`/heist` Bank heist crew\n"
            "`/slots` Slot machine\n"
            "`/rps-tournament` 4-player RPS"
        ),
        inline=True
    )
    embed.add_field(
        name="🤖 AI",
        value=(
            "`/roast` Personalized roast\n"
            "`/court` AI courtroom trial\n"
            "`/lawsuit` Sue another user\n"
            "`/bio` AI Tinder bio\n"
            "`/tarot` 3-card tarot reading\n"
            "`/analyze` Analyze your messages"
        ),
        inline=True
    )
    embed.add_field(
        name="🎉 FUN",
        value=(
            "`/marry` Propose to a user\n"
            "`/divorce` End a marriage\n"
            "`/hack` Hack a user\n"
            "`/ship` Ship two users\n"
            "`/rate` Rate something /10\n"
            "`/8ball` Magic 8-ball\n"
            "`/rps` RPS vs bot\n"
            "`/roll` Roll dice\n"
            "`/flip` Coin flip"
        ),
        inline=True
    )
    embed.add_field(
        name="💬 CHAT",
        value="Mention me, reply, or say my name\n`!recap` `!clearhistory` `!botinfo`",
        inline=False
    )
    await interaction.response.send_message(embed=embed)


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
    "heist":        ("heist",         [{"name":"crew1","type":"user","required":True},{"name":"crew2","type":"user","required":False,"default":None},{"name":"crew3","type":"user","required":False,"default":None},{"name":"crew4","type":"user","required":False,"default":None}]),
    # AI
    "roast":        ("roast",         [{"name":"user","type":"user","required":True}]),
    "bio":          ("bio",           [{"name":"user","type":"user","required":True}]),
    "court":        ("court",         [{"name":"defendant","type":"user","required":True},{"name":"charge","type":"rest_str","required":True}]),
    "lawsuit":      ("lawsuit",       [{"name":"defendant","type":"user","required":True},{"name":"claim","type":"rest_str","required":True}]),
    "sue":          ("lawsuit",       [{"name":"defendant","type":"user","required":True},{"name":"claim","type":"rest_str","required":True}]),
    "tarot":        ("tarot",         []),
    "analyze":      ("analyze",       []),
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

    # ── _prefix commands → invoke slash commands ─────────────────────────────
    content_stripped = message.content.strip()
    if content_stripped.startswith("_") and len(content_stripped) > 1:
        handled = await handle_prefix_command(message, content_stripped[1:])
        if handled:
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

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
        for i, u in enumerate(all_users):
            emoji, _vehicle = vehicles[i]
            racers.append((emoji, u.display_name, u.mention))

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
        final = (
            f"🏁 **RACE FINISHED!**\n\n"
            f"{draw(positions, finished)}\n\n"
            f"## 🏆 {winner_emoji} {winner_label} WINS!\n\n"
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


# ── Helper: gather a user's recent messages from channel + logs ──────────────
async def gather_user_messages(interaction: discord.Interaction, user: discord.Member, limit: int = 80) -> list[str]:
    user_messages: list[str] = []
    try:
        async for m in interaction.channel.history(limit=1000):
            if m.author.id == user.id and m.content and m.content.strip():
                user_messages.append(m.content.strip())
                if len(user_messages) >= 50:
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

    return user_messages[:limit]


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
    final = (
        f"🎩 **DUEL RESULT** 🎩\n\n"
        f"{random.choice(flavors)}\n\n"
        f"## 🏆 {winner.mention} stands victorious.\n"
        f"💀 {loser.mention} lies in the dust."
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
    final = (
        f"🎯 **GAME OVER**\n\n"
        + "\n".join(log_lines[-8:])
        + f"\n\n## 🏆 {winner.mention} is the last one standing!"
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
        final = (
            f"🚔 **HEIST FAILED!**\n\n"
            f"The cops showed up. {caught.mention} got pinned to the wall.\n"
            f"The rest of the crew escaped with **NOTHING**.\n\n"
            f"💀 The crew: {crew_str}"
        )
        try:
            await interaction.edit_original_response(content=final)
        except Exception:
            pass
        return

    # Otherwise everyone gets a random share
    payouts = []
    total = 0
    for p in crew:
        amount = random.randint(5_000, 250_000)
        loot_emoji, loot_name = random.choice(HEIST_LOOT)
        payouts.append((p, amount, loot_emoji, loot_name))
        total += amount

    # Sort by haul descending
    payouts.sort(key=lambda x: x[1], reverse=True)
    biggest = payouts[0]

    payout_lines = "\n".join(
        f"{['👑','🥈','🥉','🎖️','🎖️'][i] if i < 5 else '•'} {p.mention} — **${amt:,}** {emoji} _{name}_"
        for i, (p, amt, emoji, name) in enumerate(payouts)
    )

    final = (
        f"💰 **HEIST SUCCESSFUL!** 💰\n\n"
        f"The crew got out clean with **${total:,}** in loot!\n\n"
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

@tree.command(name="slots", description="Pull the lever on the slot machine!")
async def slots_command(interaction: discord.Interaction):
    silent = discord.AllowedMentions.none()
    user = interaction.user
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

    # Result
    if final_reels[0] == final_reels[1] == final_reels[2]:
        symbol = final_reels[0]
        winnings = SLOT_PAYOUT.get(symbol, 50) * 3
        result = f"## 🎉 JACKPOT! 🎉\n\n{user.mention} hit **3x {symbol}** and won **${winnings:,}**!"
    elif final_reels[0] == final_reels[1] or final_reels[1] == final_reels[2] or final_reels[0] == final_reels[2]:
        # Find the pair
        pair_symbol = max(set(final_reels), key=final_reels.count)
        winnings = SLOT_PAYOUT.get(pair_symbol, 25)
        result = f"## ✨ PAIR! ✨\n\n{user.mention} hit **2x {pair_symbol}** and won **${winnings}**."
    else:
        result = f"## 💸 NO MATCH 💸\n\nBetter luck next time, {user.mention}."

    final_display = render(final_reels, "🎰 RESULT 🎰") + "\n\n" + result
    try:
        await interaction.edit_original_response(content=final_display)
    except Exception:
        pass


# ── ⚖️ /court ───────────────────────────────────────────────────────────────
@tree.command(name="commands", description="List all available bot commands.")
async def commands_command(interaction: discord.Interaction):
    embed = discord.Embed(title="🎮 Bot Commands", color=discord.Color.blurple())
    embed.add_field(name="🏎️ /rs", value="Animated race (tag up to 3 racers)", inline=False)
    embed.add_field(name="🔥 /roast", value="AI-roast a user using their messages", inline=False)
    embed.add_field(name="⚖️ /court", value="Put a user on trial. AI judges.", inline=False)
    embed.add_field(name="📱 /bio", value="AI-generated Tinder bio for a user", inline=False)
    embed.add_field(name="🔫 /duel", value="Pistol duel between 2 users", inline=True)
    embed.add_field(name="🎯 /gun", value="Russian roulette (2-4 players)", inline=True)
    embed.add_field(name="💰 /heist", value="Group bank heist (2-5 crew)", inline=True)
    embed.add_field(name="💻 /hack", value="Hack a user's data", inline=True)
    embed.add_field(name="🥊 /rps-tournament", value="4-player RPS bracket", inline=True)
    embed.add_field(name="🎰 /slots", value="Pull the lever", inline=True)
    embed.add_field(name="🎲 /roll", value="Roll a dice", inline=True)
    embed.add_field(name="🪙 /flip", value="Flip a coin", inline=True)
    embed.add_field(name="🎱 /8ball", value="Magic 8-ball", inline=True)
    embed.add_field(name="✂️ /rps", value="RPS vs the bot", inline=True)
    embed.add_field(name="⭐ /rate", value="Rate something /10", inline=True)
    embed.add_field(name="💞 /ship", value="Ship two users", inline=True)
    embed.add_field(name="💬 Chat",
        value="Mention me, reply, or say my name to chat\n`!recap` for a daily recap\n`!clearhistory` to wipe channel memory\n`!botinfo` for info",
        inline=False)
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

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
async def download_images(message: discord.Message) -> list:
    """Return a list of (mime, base64) for all image attachments on a message."""
    images = []
    allowed_mimes = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
    for att in message.attachments:
        if att.content_type and att.content_type.lower() in allowed_mimes:
            # Skip huge files
            if att.size > 4 * 1024 * 1024:  # 4MB cap
                continue
            try:
                data = await att.read()
                images.append((att.content_type, base64.b64encode(data).decode("utf-8")))
            except Exception as e:
                log.warning("Failed to download %s: %s", att.filename, e)
    return images


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
client = discord.Client(intents=intents)

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


@client.event
async def on_ready():
    log.info("Logged in as %s (ID: %s)", client.user, client.user.id)
    available = [p for p in ["groq","cerebras","gemini"] if os.environ.get(f"{p.upper()}_API_KEY")]
    log.info("Available AI providers: %s", available or "NONE")
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
        watch_line = f"[{message.author.display_name}]: {message.content}"
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
        memory.append(channel_id, "user", f"[{message.author.display_name}]: {content}")

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

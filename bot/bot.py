"""
Discord AI Bot - Powered by Groq (free, fast cloud AI)
Full personality customization, persistent memory, chime-in mode,
passive watching, emoji reactions, silence check-ins, rotating status.
"""

import discord
import json
import os
import aiohttp
import asyncio
import logging
import random
import time
from datetime import datetime, timezone
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
MEMORY_DIR.mkdir(exist_ok=True)

DEFAULT_PERSONALITY = {
    "name": "Aria",
    "model": "llama-3.3-70b-versatile",
    "system_prompt": "You are a friendly Discord community assistant.",
    "temperature": 0.9,
    "max_memory_messages": 30,
    "respond_to_bots": False,
    "trigger_mode": "mention_or_reply",
    "allowed_channels": [],
    "blocked_users": [],
    # Chime-in (spontaneous messages)
    "chime_in_enabled": False,
    "chime_in_min": 40,
    "chime_in_max": 80,
    # Passive watching (log all messages for context even if not replying)
    "watch_mode_enabled": True,
    # Emoji reactions
    "reaction_enabled": True,
    "reaction_chance": 0.05,   # 5% chance per message to react
    # Silence check-in
    "silence_checkin_enabled": True,
    "silence_minutes": 45,     # trigger if no bot activity for this long AND channel has activity
    # Rotating status
    "status_messages": [
        "watching",
        "judging quietly",
        "peaked",
        "counting typos",
        "3am thoughts",
        "making bad decisions",
        "up way too late"
    ],
    "status_rotation_minutes": 10,
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULT_PERSONALITY, **json.load(f)}
    return DEFAULT_PERSONALITY.copy()


# ── Chime-in counters ─────────────────────────────────────────────────────────
message_counters: dict[str, int] = defaultdict(int)
chime_thresholds: dict[str, int] = {}

def get_chime_threshold(cfg: dict) -> int:
    return random.randint(cfg["chime_in_min"], cfg["chime_in_max"])


# ── Per-channel last-activity tracking (for silence check-in) ────────────────
last_bot_activity: dict[str, float] = {}
last_channel_activity: dict[str, float] = {}
last_channel_id_used: dict[str, int] = {}  # channel_id -> guild_id


# ── Memory ────────────────────────────────────────────────────────────────────
class MemoryStore:
    """Stores last N messages per channel as a passive log.
    Keeps user+assistant turns, plus 'watched' user messages that weren't replied to."""
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


# ── Groq API ──────────────────────────────────────────────────────────────────
async def ask_groq(system_prompt, messages, model, temperature, api_key) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 401:
                    return "⚠️ Invalid Groq API key."
                if resp.status == 429:
                    return "⚠️ Groq rate limit hit. Try again in a moment."
                if resp.status != 200:
                    text = await resp.text()
                    log.error("Groq error %s: %s", resp.status, text)
                    return "⚠️ AI error. Please try again."
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
    except asyncio.TimeoutError:
        return "⚠️ The AI took too long to respond."
    except Exception as e:
        log.exception("Unexpected Groq error")
        return f"⚠️ Unexpected error: {e}"


# ── Common emojis for reactions ───────────────────────────────────────────────
REACTION_POOL = [
    "💀", "👀", "🔥", "😂", "🤡", "😒", "🤨", "😮‍💨",
    "🙄", "💅", "🧠", "⚡", "💯", "🫠", "🫡", "🤌"
]


# ── Send helper ───────────────────────────────────────────────────────────────
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

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
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
    return mentioned or is_reply


def get_time_context() -> str:
    """Returns natural time context like 'it's 3am' for the AI to optionally reference."""
    now = datetime.now(timezone.utc)
    hour = now.hour
    if 0 <= hour < 5:
        return f"(It is currently {hour}:{now.minute:02d} UTC — late night / early morning)"
    if 5 <= hour < 12:
        return f"(It is currently {hour}:{now.minute:02d} UTC — morning)"
    if 12 <= hour < 18:
        return f"(It is currently {hour}:{now.minute:02d} UTC — afternoon)"
    return f"(It is currently {hour}:{now.minute:02d} UTC — evening/night)"


# ── Background tasks: status rotation + silence check-in ──────────────────────
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
    """Checks every few minutes if a channel has been quiet and drops a message."""
    await client.wait_until_ready()
    while not client.is_closed():
        await asyncio.sleep(180)  # check every 3 min
        cfg = load_config()
        if not cfg.get("silence_checkin_enabled"):
            continue
        if not GROQ_API_KEY:
            continue
        silence_seconds = cfg["silence_minutes"] * 60

        for channel_id, ch_last in list(last_channel_activity.items()):
            if not channel_allowed(channel_id, cfg):
                continue
            bot_last = last_bot_activity.get(channel_id, 0)
            now = time.time()
            # Channel had activity recently (within 2h) but bot has been quiet for threshold
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
                    reply = await ask_groq(
                        system_prompt=system,
                        messages=history if history else [{"role":"user","content":"..."}],
                        model=cfg["model"],
                        temperature=min(cfg["temperature"] + 0.15, 1.0),
                        api_key=GROQ_API_KEY,
                    )
                    if not reply.startswith("⚠️"):
                        await channel.send(reply)
                        last_bot_activity[channel_id] = time.time()
                        memory.append(channel_id, "assistant", reply)
                        log.info("Silence check-in on #%s", channel.name if hasattr(channel,'name') else channel_id)
                except Exception as e:
                    log.warning("Silence check-in failed: %s", e)


@client.event
async def on_ready():
    log.info("Logged in as %s (ID: %s)", client.user, client.user.id)
    client.loop.create_task(rotate_status())
    client.loop.create_task(silence_checker())


@client.event
async def on_message(message: discord.Message):
    if message.author.bot and message.author != client.user:
        # track other bots' activity for channel-alive detection but don't respond
        last_channel_activity[str(message.channel.id)] = time.time()
        return
    if message.author == client.user:
        return

    cfg = load_config()
    memory.update_maxlen(cfg["max_memory_messages"])
    channel_id = str(message.channel.id)
    last_channel_activity[channel_id] = time.time()

    # ── Commands ──────────────────────────────────────────────────────────────
    if message.content.strip() == "!clearhistory":
        memory.clear(channel_id)
        await message.reply("🧹 Conversation history cleared for this channel.")
        return

    if message.content.strip() == "!botinfo":
        embed = discord.Embed(title=f"🤖 {cfg['name']}", color=discord.Color.blurple())
        embed.add_field(name="Model", value=cfg["model"], inline=True)
        embed.add_field(name="Trigger", value=cfg["trigger_mode"], inline=True)
        embed.add_field(name="Memory", value=f"{cfg['max_memory_messages']} msgs", inline=True)
        chime = f"Every {cfg['chime_in_min']}–{cfg['chime_in_max']} msgs" if cfg.get("chime_in_enabled") else "Off"
        embed.add_field(name="Chime-in", value=chime, inline=True)
        embed.add_field(name="Watch mode", value="On" if cfg.get("watch_mode_enabled") else "Off", inline=True)
        embed.add_field(name="Reactions", value=f"{int(cfg.get('reaction_chance',0)*100)}% chance" if cfg.get("reaction_enabled") else "Off", inline=True)
        await message.reply(embed=embed)
        return

    # Skip blocked users and disallowed channels for ALL features
    if str(message.author.id) in cfg["blocked_users"]:
        return
    if not channel_allowed(channel_id, cfg):
        return

    # ── Passive watch mode: log every message so bot sees context later ──────
    if cfg.get("watch_mode_enabled", True):
        watch_line = f"[{message.author.display_name}]: {message.content}"
        memory.append(channel_id, "user", watch_line)

    # ── Emoji reaction chance ────────────────────────────────────────────────
    if cfg.get("reaction_enabled") and random.random() < cfg.get("reaction_chance", 0.05):
        try:
            await message.add_reaction(random.choice(REACTION_POOL))
        except Exception as e:
            log.debug("Reaction failed: %s", e)

    # ── Chime-in counter ──────────────────────────────────────────────────────
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

    if not GROQ_API_KEY:
        await message.reply("⚠️ GROQ_API_KEY is not set.")
        return

    # ── Chime-in response ─────────────────────────────────────────────────────
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
            reply = await ask_groq(
                system_prompt=chime_system,
                messages=history if history else [{"role": "user", "content": message.content}],
                model=cfg["model"],
                temperature=min(cfg["temperature"] + 0.1, 1.0),
                api_key=GROQ_API_KEY,
            )
        if not reply.startswith("⚠️"):
            memory.append(channel_id, "assistant", reply)
            last_bot_activity[channel_id] = time.time()
            await send_reply(message, reply, chimed_in=True)
            log.info("Chimed in on #%s", message.channel)
        return

    # ── Direct reply ──────────────────────────────────────────────────────────
    content = message.content
    if client.user.mentioned_in(message):
        content = content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()

    if not content:
        await message.reply("what")
        return

    # Watch mode already logged the message, but if off we still need to log for context
    if not cfg.get("watch_mode_enabled", True):
        memory.append(channel_id, "user", f"[{message.author.display_name}]: {content}")

    async with message.channel.typing():
        history = memory.get_messages(channel_id)
        reply = await ask_groq(
            system_prompt=cfg["system_prompt"] + "\n\n" + get_time_context(),
            messages=history,
            model=cfg["model"],
            temperature=cfg["temperature"],
            api_key=GROQ_API_KEY,
        )

    if not reply.startswith("⚠️"):
        memory.append(channel_id, "assistant", reply)
        last_bot_activity[channel_id] = time.time()
    await send_reply(message, reply)
    log.info("Replied to %s in #%s", message.author, message.channel)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set.")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    client.run(DISCORD_TOKEN, log_handler=None)

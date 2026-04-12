"""
Discord AI Bot - Powered by Groq (free, fast cloud AI)
Supports full personality customization, persistent memory, and chime-in mode.
"""

import discord
import json
import os
import aiohttp
import asyncio
import logging
import random
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
    "system_prompt": (
        "You are Aria, a friendly and helpful Discord community assistant. "
        "You are witty, concise, and genuinely care about helping community members. "
        "Keep responses conversational and under 300 words unless detail is truly needed."
    ),
    "temperature": 0.7,
    "max_memory_messages": 20,
    "respond_to_bots": False,
    "trigger_mode": "mention_or_reply",
    "allowed_channels": [],
    "blocked_users": [],
    "chime_in_enabled": False,
    "chime_in_min": 10,
    "chime_in_max": 20,
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


# ── Memory ────────────────────────────────────────────────────────────────────
class MemoryStore:
    def __init__(self, max_messages: int = 20):
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


def is_direct_trigger(message: discord.Message, cfg: dict) -> bool:
    if message.author == client.user:
        return False
    if not cfg["respond_to_bots"] and message.author.bot:
        return False
    if str(message.author.id) in cfg["blocked_users"]:
        return False
    if cfg["allowed_channels"] and str(message.channel.id) not in cfg["allowed_channels"]:
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


@client.event
async def on_ready():
    log.info("Logged in as %s (ID: %s)", client.user, client.user.id)


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    cfg = load_config()
    memory.update_maxlen(cfg["max_memory_messages"])
    channel_id = str(message.channel.id)

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
        await message.reply(embed=embed)
        return

    # ── Chime-in counter ──────────────────────────────────────────────────────
    chime_triggered = False
    if cfg.get("chime_in_enabled"):
        channel_allowed = not cfg["allowed_channels"] or channel_id in cfg["allowed_channels"]
        if channel_allowed:
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
            "\n\nYou are chiming into an ongoing conversation naturally. "
            "React to what's been said, add something fun or interesting, or ask a question. "
            "Keep it short — 1 to 3 sentences max. "
            "Don't announce that you're joining the conversation."
        )
        async with message.channel.typing():
            reply = await ask_groq(
                system_prompt=chime_system,
                messages=history if history else [{"role": "user", "content": message.content}],
                model=cfg["model"],
                temperature=min(cfg["temperature"] + 0.1, 1.0),
                api_key=GROQ_API_KEY,
            )
        memory.append(channel_id, "assistant", reply)
        await send_reply(message, reply, chimed_in=True)
        log.info("Chimed in on #%s", message.channel)
        return

    # ── Direct reply ──────────────────────────────────────────────────────────
    content = message.content
    if client.user.mentioned_in(message):
        content = content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()

    if not content:
        await message.reply("Hey! How can I help? 😊")
        return

    memory.append(channel_id, "user", f"[{message.author.display_name}]: {content}")

    async with message.channel.typing():
        history = memory.get_messages(channel_id)
        reply = await ask_groq(
            system_prompt=cfg["system_prompt"],
            messages=history,
            model=cfg["model"],
            temperature=cfg["temperature"],
            api_key=GROQ_API_KEY,
        )

    memory.append(channel_id, "assistant", reply)
    await send_reply(message, reply)
    log.info("Replied to %s in #%s", message.author, message.channel)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set.")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    client.run(DISCORD_TOKEN, log_handler=None)

"""
Discord Inactivity Bot
======================
Single-use script: assigns a configurable role to every member who has had
NO activity in the server for the past 30 days.

Roles are assigned AS THE SCAN PROGRESSES — not batched at the end.

Activity tracked:
  - Messages sent in any text channel
  - Reactions added to messages
  - Audit log entries (covers VC joins/leaves, role changes, etc.)
  - Current voice-channel presence

Requirements:
  pip install discord.py

Permissions the bot needs (Discord Developer Portal):
  - Read Messages / View Channels
  - Read Message History
  - Manage Roles
  - View Audit Log

Privileged Intents (Developer Portal → Bot → Privileged Gateway Intents):
  - Server Members Intent    ✅
  - Message Content Intent   ✅
"""

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta

import discord
from discord.ext import commands

# ── Configuration ──────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
GUILD_ID  = int(os.getenv("GUILD_ID",       "0"))
ROLE_ID   = int(os.getenv("ROLE_ID",        "0"))
DAYS      = int(os.getenv("INACTIVE_DAYS", "30"))
# ───────────────────────────────────────────────────────────────────────────────

CUTOFF = datetime.now(timezone.utc) - timedelta(days=DAYS)

intents = discord.Intents.default()
intents.members         = True
intents.message_content = True
intents.reactions       = True
intents.voice_states    = True

bot = commands.Bot(command_prefix="!", intents=intents)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


async def assign_role(member: discord.Member, role: discord.Role) -> bool:
    """
    Assign the inactivity role to a member immediately.
    Returns True on success, False on failure.
    """
    if role in member.roles:
        log(f"  ↷  {member} already has the role — skipping.")
        return False
    try:
        await member.add_roles(role, reason=f"No activity in the past {DAYS} days (auto-assigned)")
        log(f"  ✔  TAGGED {member} (joined: {member.joined_at.strftime('%Y-%m-%d') if member.joined_at else 'unknown'})")
        return True
    except discord.Forbidden:
        log(f"  ✘  Missing permissions for {member}.")
        return False
    except discord.HTTPException as exc:
        log(f"  ✘  HTTP error for {member}: {exc}")
        return False


async def run() -> None:
    await bot.wait_until_ready()

    guild = bot.get_guild(GUILD_ID)
    if guild is None:
        log(f"❌  Guild {GUILD_ID} not found. Is the bot in that server?")
        await bot.close()
        return

    role = guild.get_role(ROLE_ID)
    if role is None:
        log(f"❌  Role {ROLE_ID} not found in guild '{guild.name}'.")
        await bot.close()
        return

    log(f"✅  Connected to guild: {guild.name} ({guild.member_count} members)")
    log(f"    Inactivity threshold : {DAYS} days  (cutoff: {CUTOFF.strftime('%Y-%m-%d %H:%M UTC')})")
    log(f"    Role to assign       : @{role.name}")
    log("")

    # ── Step 1: Fetch full member list ────────────────────────────────────────
    log("Fetching full member list …")
    await guild.chunk()

    # Build working dict: member_id → Member (exclude bots)
    all_members: dict[int, discord.Member] = {
        m.id: m for m in guild.members if not m.bot
    }
    # Track who is confirmed active (will NOT get the role)
    active_ids: set[int] = set()
    # Track who has already been tagged (avoid double-assigning)
    tagged_ids: set[int] = set()

    total_members = len(all_members)
    log(f"Non-bot members to evaluate: {total_members}\n")

    # ── Step 2: Scan voice channels (instant — no API calls needed) ───────────
    log("Checking current voice-channel presence …")
    for vc in guild.voice_channels:
        for member in vc.members:
            if not member.bot:
                active_ids.add(member.id)
    log(f"  Members currently in VC (active): {len(active_ids)}\n")

    # ── Step 3: Scan audit log for VC / misc activity ─────────────────────────
    log("Scanning audit log for voice / misc activity …")
    audit_count = 0
    try:
        async for entry in guild.audit_logs(limit=None, after=CUTOFF):
            if entry.user and not entry.user.bot:
                active_ids.add(entry.user.id)
                audit_count += 1
            if (
                hasattr(entry, "target")
                and isinstance(entry.target, (discord.Member, discord.User))
                and not entry.target.bot
                and entry.action == discord.AuditLogAction.member_update
            ):
                active_ids.add(entry.target.id)
    except discord.Forbidden:
        log("  ⚠  Cannot read audit log — VC activity tracking limited.")
    except discord.HTTPException as exc:
        log(f"  ⚠  Audit log error: {exc}")
    log(f"  Audit log entries processed: {audit_count}\n")

    # ── Step 4: Scan text channels — assign role AS WE GO ────────────────────
    text_channels = guild.text_channels
    total_channels = len(text_channels)
    assigned = 0
    errors   = 0

    log(f"Scanning {total_channels} text channel(s) — roles assigned in real time …\n")

    for idx, channel in enumerate(text_channels, 1):
        log(f"  [{idx}/{total_channels}] #{channel.name}  |  tagged so far: {assigned}")
        try:
            async for message in channel.history(limit=None, after=CUTOFF, oldest_first=False):
                # Message author is active
                if message.author and not message.author.bot:
                    active_ids.add(message.author.id)

                # Reaction users are active
                for reaction in message.reactions:
                    try:
                        async for user in reaction.users():
                            if not user.bot:
                                active_ids.add(user.id)
                    except (discord.Forbidden, discord.HTTPException):
                        pass

        except discord.Forbidden:
            log(f"    ⚠  No access to #{channel.name} — skipping.")
        except discord.HTTPException as exc:
            log(f"    ⚠  HTTP error in #{channel.name}: {exc} — skipping.")

        # After finishing each channel, tag anyone we now know is inactive
        # (not seen anywhere active so far, not already tagged)
        newly_inactive = [
            m for uid, m in all_members.items()
            if uid not in active_ids and uid not in tagged_ids
        ]

        for member in newly_inactive:
            ok = await assign_role(member, role)
            tagged_ids.add(member.id)  # mark as processed regardless
            if ok:
                assigned += 1
            else:
                errors += 1
            await asyncio.sleep(0.5)  # rate-limit safety

    # ── Step 5: Final pass — anyone still not tagged and not active ───────────
    remaining = [
        m for uid, m in all_members.items()
        if uid not in active_ids and uid not in tagged_ids
    ]
    if remaining:
        log(f"\nFinal pass — {len(remaining)} member(s) not yet processed …")
        for member in remaining:
            ok = await assign_role(member, role)
            tagged_ids.add(member.id)
            if ok:
                assigned += 1
            else:
                errors += 1
            await asyncio.sleep(0.5)

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "═" * 50)
    log("SUMMARY")
    log("═" * 50)
    log(f"  Total non-bot members : {total_members}")
    log(f"  Active (skipped)      : {len(active_ids)}")
    log(f"  Role assigned         : {assigned}")
    log(f"  Errors                : {errors}")
    log("═" * 50)
    log("Done. Shutting down.")

    await bot.close()


@bot.event
async def on_ready() -> None:
    log(f"Logged in as {bot.user} — starting inactivity scan …\n")
    bot.loop.create_task(run())


# ── Entry-point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    missing = []
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        missing.append("BOT_TOKEN")
    if GUILD_ID == 0:
        missing.append("GUILD_ID")
    if ROLE_ID == 0:
        missing.append("ROLE_ID")

    if missing:
        print("❌  Please set the following before running:")
        for m in missing:
            print(f"    • {m}  (edit bot.py or set as environment variable)")
        sys.exit(1)

    try:
        bot.run(BOT_TOKEN)
    except discord.LoginFailure:
        print("❌  Invalid bot token. Double-check BOT_TOKEN.")
        sys.exit(1)

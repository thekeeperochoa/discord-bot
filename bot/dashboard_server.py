"""
Embedded analytics dashboard for the Discord bot.
Runs on a background thread alongside the Discord client.
Reads stats.json from the same volume the bot writes to.

Required env vars:
- DISCORD_CLIENT_ID
- DISCORD_CLIENT_SECRET
- DISCORD_OAUTH_REDIRECT  (e.g. https://your-worker.up.railway.app/auth/callback)
- DASHBOARD_ADMIN_IDS     (comma-separated Discord user IDs)
- DASHBOARD_SESSION_SECRET (random 64-char hex)
- MEMORY_DIR              (optional, defaults to /app/memory)
"""
import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from flask import Flask, jsonify, redirect, request, session

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
CLIENT_ID = os.environ.get("DISCORD_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("DISCORD_CLIENT_SECRET", "")
REDIRECT_URI = os.environ.get("DISCORD_OAUTH_REDIRECT", "")
ADMIN_IDS = set(
    s.strip() for s in os.environ.get("DASHBOARD_ADMIN_IDS", "").split(",") if s.strip()
)
SESSION_SECRET = os.environ.get("DASHBOARD_SESSION_SECRET", secrets.token_hex(32))
MEMORY_DIR = Path(os.environ.get("MEMORY_DIR", "/app/memory"))
STATS_FILE = MEMORY_DIR / "stats.json"
ECONOMY_FILE = MEMORY_DIR / "economy.json"

# For the Custom Messages tab: the bot token (same one the bot logs in with) lets
# the dashboard post messages AS THE BOT — so there's no sender attribution — and
# read the guild's channels/roles/members for the pickers + @ autocomplete.
BOT_TOKEN = os.environ.get("DISCORD_TOKEN", "") or os.environ.get("BOT_TOKEN", "")
GUILD_ID = os.environ.get("DISCORD_GUILD_ID", "") or os.environ.get("GUILD_ID", "")

DISCORD_API = "https://discord.com/api/v10"
OAUTH_SCOPE = "identify"


def _bot_headers() -> dict:
    return {"Authorization": f"Bot {BOT_TOKEN}", "Content-Type": "application/json"}


# Small in-process cache for guild metadata (channels/roles/members) so the
# autocomplete doesn't hammer the Discord API on every keystroke.
_guild_meta_cache = {"ts": 0.0, "data": None}

app = Flask(__name__)
app.secret_key = SESSION_SECRET
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


# ── Helpers ──────────────────────────────────────────────────────────────
def is_logged_in() -> bool:
    return bool(session.get("user_id"))


def is_admin() -> bool:
    return is_logged_in() and str(session.get("user_id")) in ADMIN_IDS


def _load_json(path: Path, default):
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _last_n_days(n: int) -> list:
    today = datetime.now(timezone.utc).date()
    return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n - 1, -1, -1)]


# ── Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if not is_logged_in():
        return redirect("/auth/login")
    if not is_admin():
        return _render_denied()
    return _render_dashboard()


@app.route("/auth/login")
def auth_login():
    if not CLIENT_ID or not REDIRECT_URI:
        return ("Server config error: DISCORD_CLIENT_ID and DISCORD_OAUTH_REDIRECT "
                "must be set in Railway env vars."), 500
    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": OAUTH_SCOPE,
        "state": state,
        "prompt": "consent",
    }
    qs = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
    return redirect(f"{DISCORD_API}/oauth2/authorize?{qs}")


@app.route("/auth/callback")
def auth_callback():
    code = request.args.get("code")
    state = request.args.get("state")
    if not code or state != session.get("oauth_state"):
        return "Invalid OAuth state.", 400
    try:
        token_resp = requests.post(
            f"{DISCORD_API}/oauth2/token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]
        user_resp = requests.get(
            f"{DISCORD_API}/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        user_resp.raise_for_status()
        user = user_resp.json()
    except Exception as e:
        return f"OAuth exchange failed: {e}", 500
    session["user_id"] = user["id"]
    session["username"] = user.get("username", "?")
    session.pop("oauth_state", None)
    return redirect("/")


@app.route("/auth/logout")
def auth_logout():
    session.clear()
    return redirect("/")


@app.route("/health")
def health():
    return jsonify({"ok": True, "stats_file_exists": STATS_FILE.exists()})


VOICE_FILE = MEMORY_DIR / "voice_analytics.json"


@app.route("/api/voice-analytics")
def api_voice_analytics():
    if not is_admin():
        return jsonify({"error": "unauthorized"}), 403

    import time as _t
    from datetime import datetime as _dt
    v = _load_json(VOICE_FILE, {})
    sessions = v.get("sessions", [])
    events = v.get("events", [])
    totals = v.get("totals", {})
    channel_totals = v.get("channel_totals", {})
    pairs = v.get("pairs", {})
    now = _t.time()

    # Name lookups from sessions/events
    name_by_id = {}
    for s in sessions:
        name_by_id[s.get("user_id")] = s.get("user_name")
    for e in events:
        name_by_id.setdefault(e.get("user_id"), e.get("user_name"))
    chname_by_id = {}
    for s in sessions:
        chname_by_id[s.get("channel_id")] = s.get("channel_name")

    # Currently in voice: last event per user is a join/move (not leave)
    live = {}
    seen = set()
    for e in events:  # newest first
        uid = e.get("user_id")
        if uid in seen:
            continue
        seen.add(uid)
        if e.get("type") in ("join", "move"):
            live.setdefault(e.get("channel_name") or "?", []).append({
                "name": e.get("user_name"),
                "state": e.get("state", {}),
            })
    live_list = [{"channel": ch, "members": mem} for ch, mem in live.items()]

    # Summary numbers
    total_seconds = sum(totals.values())
    day_ago = now - 86400
    week_ago = now - 604800
    sessions_today = [s for s in sessions if s.get("left", 0) >= day_ago]
    unique_today = len({s.get("user_id") for s in sessions_today})
    avg_session = int(sum(s.get("duration", 0) for s in sessions) / len(sessions)) if sessions else 0
    longest = max((s.get("duration", 0) for s in sessions), default=0)

    # Leaderboard: top users by total voice time
    top_users = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:12]
    top_users = [{"name": name_by_id.get(uid, uid), "seconds": sec} for uid, sec in top_users]

    # Channel breakdown
    top_channels = sorted(channel_totals.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_channels = [{"name": chname_by_id.get(cid, cid), "seconds": sec} for cid, sec in top_channels]

    # Heatmap: 7 days x 24 hours of join events
    heat = [[0] * 24 for _ in range(7)]
    for e in events:
        if e.get("type") == "join":
            d = _dt.fromtimestamp(e.get("t", 0))
            heat[d.weekday()][d.hour] += 1

    # Activity over last 14 days (sessions/day)
    daily = {}
    for s in sessions:
        if s.get("left"):
            day = _dt.fromtimestamp(s["left"]).strftime("%Y-%m-%d")
            daily[day] = daily.get(day, 0) + 1
    daily_series = sorted(daily.items())[-14:]

    # Social graph pairs (top co-presence)
    top_pairs = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:15]
    pair_list = []
    for pk, sec in top_pairs:
        a, b = pk.split(":")
        pair_list.append({"a": name_by_id.get(a, a), "b": name_by_id.get(b, b), "seconds": sec})

    # Streaming / camera usage counts
    stream_ct = sum(1 for e in events if e.get("state", {}).get("streaming"))
    cam_ct = sum(1 for e in events if e.get("state", {}).get("camera"))

    # Recent sessions table (last 25)
    recent = []
    for s in sessions[:25]:
        recent.append({
            "user": s.get("user_name"),
            "channel": s.get("channel_name"),
            "joined": s.get("joined"),
            "duration": s.get("duration", 0),
            "with": len(s.get("co_present", [])),
        })

    return jsonify({
        "ok": True,
        "summary": {
            "total_seconds": total_seconds,
            "sessions_all": len(sessions),
            "unique_today": unique_today,
            "avg_session": avg_session,
            "longest": longest,
            "live_count": sum(len(x["members"]) for x in live_list),
            "streams": stream_ct,
            "cameras": cam_ct,
        },
        "live": live_list,
        "top_users": top_users,
        "top_channels": top_channels,
        "heatmap": heat,
        "daily": daily_series,
        "pairs": pair_list,
        "recent": recent,
    })


@app.route("/api/stats")
def api_stats():
    if not is_admin():
        return jsonify({"error": "unauthorized"}), 403

    stats = _load_json(STATS_FILE, {})
    economy = _load_json(ECONOMY_FILE, {})

    cmd_uses = dict(stats.get("command_uses", {}))
    # Merge in ALL known commands so even never-used ones appear (count 0).
    known = _load_json(MEMORY_DIR / "known_commands.json", {})
    for name in known.get("slash", []):
        cmd_uses.setdefault(name, 0)
    # Sort by usage desc, then name — show EVERY command, not just top 15.
    top_commands = sorted(cmd_uses.items(), key=lambda x: (-x[1], x[0]))

    days = _last_n_days(14)
    top5_cmds = [c for c, _ in top_commands[:5]]
    command_trend = {"days": days, "series": {}}
    for cmd in top5_cmds:
        per_day = stats.get("command_uses_today", {}).get(cmd, {})
        command_trend["series"][cmd] = [per_day.get(d, 0) for d in days]

    today = _today_str()
    today_econ = stats.get("economy_events_today", {}).get(today, {})
    econ_trend = {"days": days, "earned": [], "spent": [], "transactions": []}
    for d in days:
        e = stats.get("economy_events_today", {}).get(d, {})
        econ_trend["earned"].append(e.get("earned", 0))
        econ_trend["spent"].append(e.get("spent", 0))
        econ_trend["transactions"].append(e.get("transactions", 0))

    total_coins = 0
    user_count = 0
    top_users = []
    if "users" in economy:
        for uid, u in economy["users"].items():
            bal = u.get("balance", 0)
            total_coins += bal
            user_count += 1
            top_users.append({"user_id": uid, "balance": bal})
    top_users.sort(key=lambda x: x["balance"], reverse=True)
    top_users = top_users[:10]

    active_today = stats.get("active_users_today", {}).get(today, [])
    new_today = stats.get("new_users", {}).get(today, [])
    activity_trend = {"days": days, "active": [], "new": []}
    for d in days:
        activity_trend["active"].append(len(stats.get("active_users_today", {}).get(d, [])))
        activity_trend["new"].append(len(stats.get("new_users", {}).get(d, [])))

    games = stats.get("game_outcomes", {})
    game_stats = []
    for name, g in sorted(games.items(), key=lambda x: x[1].get("wins", 0) + x[1].get("losses", 0), reverse=True):
        total_plays = g.get("wins", 0) + g.get("losses", 0)
        win_rate = (g.get("wins", 0) / total_plays * 100) if total_plays > 0 else 0
        game_stats.append({
            "name": name,
            "plays": total_plays,
            "wins": g.get("wins", 0),
            "losses": g.get("losses", 0),
            "win_rate": round(win_rate, 1),
            "total_payout": g.get("total_payout", 0),
            "biggest_payout": g.get("biggest_payout", 0),
        })

    feature_usage = stats.get("feature_usage", {})
    feed = stats.get("activity_feed", [])[:30]

    return jsonify({
        "summary": {
            "total_commands_ever": sum(cmd_uses.values()),
            "unique_commands": sum(1 for v in cmd_uses.values() if v > 0),
            "active_users_today": len(active_today),
            "new_users_today": len(new_today),
            "total_users": user_count,
            "total_coins": total_coins,
            "transactions_today": today_econ.get("transactions", 0),
            "earned_today": today_econ.get("earned", 0),
        },
        "top_commands": [{"name": n, "count": c} for n, c in top_commands],
        "command_trend": command_trend,
        "economy_trend": econ_trend,
        "top_users": top_users,
        "activity_trend": activity_trend,
        "games": game_stats,
        "feature_usage": feature_usage,
        "activity_feed": feed,
        "viewer": {
            "user_id": session.get("user_id"),
            "username": session.get("username"),
        },
    })



# ── Custom Messages: guild metadata + send ────────────────────────────────
@app.route("/api/guild-meta")
def api_guild_meta():
    """Channels, roles, and members for the Custom Messages pickers + @ autocomplete.
    Admin-only. Cached ~60s to avoid hammering the Discord API."""
    if not is_admin():
        return jsonify({"error": "unauthorized"}), 403
    if not BOT_TOKEN or not GUILD_ID:
        return jsonify({"error": "Bot token or guild ID not configured on the server."}), 500

    import time as _time
    now = _time.time()
    if _guild_meta_cache["data"] is not None and now - _guild_meta_cache["ts"] < 60:
        return jsonify(_guild_meta_cache["data"])

    try:
        # Channels
        ch_resp = requests.get(
            f"{DISCORD_API}/guilds/{GUILD_ID}/channels",
            headers=_bot_headers(), timeout=15,
        )
        ch_resp.raise_for_status()
        channels = [
            {"id": c["id"], "name": c.get("name", "?"), "type": c.get("type")}
            for c in ch_resp.json()
            # 0 = text, 5 = announcement, 15 = forum — text-capable channels
            if c.get("type") in (0, 5, 15)
        ]

        # Roles
        role_resp = requests.get(
            f"{DISCORD_API}/guilds/{GUILD_ID}/roles",
            headers=_bot_headers(), timeout=15,
        )
        role_resp.raise_for_status()
        roles = [
            {"id": r["id"], "name": r.get("name", "?")}
            for r in role_resp.json()
            if r.get("name") != "@everyone"
        ]

        # Members (first 1000 — Discord caps a single call at 1000)
        mem_resp = requests.get(
            f"{DISCORD_API}/guilds/{GUILD_ID}/members?limit=1000",
            headers=_bot_headers(), timeout=20,
        )
        members = []
        if mem_resp.status_code == 200:
            for m in mem_resp.json():
                u = m.get("user", {})
                if u.get("bot"):
                    continue
                members.append({
                    "id": u.get("id"),
                    "name": m.get("nick") or u.get("global_name") or u.get("username", "?"),
                    "username": u.get("username", ""),
                })

        data = {"channels": channels, "roles": roles, "members": members}
        _guild_meta_cache["data"] = data
        _guild_meta_cache["ts"] = now
        return jsonify(data)
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        msg = "Missing GUILD MEMBERS INTENT for member list — enable it in the Developer Portal." if code == 403 else f"Discord API error ({code})."
        log.warning("guild-meta failed: %s", e)
        return jsonify({"error": msg}), 502
    except Exception as e:
        log.warning("guild-meta failed: %s", e)
        return jsonify({"error": "Failed to load server data."}), 502


def _build_embed(data: dict):
    """Turn the dashboard's embed form data into a Discord embed dict.
    Returns the embed dict on success, or an error string on validation failure.
    Enforces Discord's field limits so the API doesn't reject the whole send."""
    embed = {}

    def _clip(s, n):
        s = str(s or "").strip()
        return s[:n]

    title = _clip(data.get("title"), 256)
    if title:
        embed["title"] = title

    desc = _clip(data.get("description"), 4096)
    if desc:
        embed["description"] = desc

    url = str(data.get("url") or "").strip()
    if url:
        if not url.startswith(("http://", "https://")):
            return "Embed title URL must start with http:// or https://"
        embed["url"] = url

    # Color: accept "#RRGGBB", "RRGGBB", or an int
    color = data.get("color")
    if color not in (None, "", "#"):
        try:
            if isinstance(color, str):
                color = int(color.lstrip("#"), 16)
            embed["color"] = int(color) & 0xFFFFFF
        except (ValueError, TypeError):
            return "Embed color must be a hex code like #f1c40f."

    author_name = _clip(data.get("author_name"), 256)
    if author_name:
        author = {"name": author_name}
        a_icon = str(data.get("author_icon") or "").strip()
        if a_icon.startswith(("http://", "https://")):
            author["icon_url"] = a_icon
        embed["author"] = author

    footer_text = _clip(data.get("footer_text"), 2048)
    if footer_text:
        footer = {"text": footer_text}
        f_icon = str(data.get("footer_icon") or "").strip()
        if f_icon.startswith(("http://", "https://")):
            footer["icon_url"] = f_icon
        embed["footer"] = footer

    image = str(data.get("image") or "").strip()
    if image:
        if not image.startswith(("http://", "https://")):
            return "Image URL must start with http:// or https://"
        embed["image"] = {"url": image}

    thumb = str(data.get("thumbnail") or "").strip()
    if thumb:
        if not thumb.startswith(("http://", "https://")):
            return "Thumbnail URL must start with http:// or https://"
        embed["thumbnail"] = {"url": thumb}

    if data.get("timestamp"):
        from datetime import datetime, timezone
        embed["timestamp"] = datetime.now(timezone.utc).isoformat()

    fields_in = data.get("fields")
    if isinstance(fields_in, list):
        fields = []
        for f in fields_in[:25]:  # Discord caps at 25 fields
            if not isinstance(f, dict):
                continue
            name = _clip(f.get("name"), 256)
            value = _clip(f.get("value"), 1024)
            if not name or not value:
                continue  # skip incomplete rows silently
            fields.append({"name": name, "value": value, "inline": bool(f.get("inline"))})
        if fields:
            embed["fields"] = fields

    # An embed with nothing in it is meaningless
    if not embed:
        return "Your embed is empty — add a title, description, or at least one field."

    # Discord's 6000-char total budget across an embed
    total = len(embed.get("title", "")) + len(embed.get("description", "")) \
        + len(embed.get("author", {}).get("name", "")) \
        + len(embed.get("footer", {}).get("text", "")) \
        + sum(len(f["name"]) + len(f["value"]) for f in embed.get("fields", []))
    if total > 6000:
        return f"Embed is too long ({total}/6000 characters total). Trim it down."

    return embed


@app.route("/api/send-message", methods=["POST"])
def api_send_message():
    """Send a message to a channel AS THE BOT (no sender attribution). Admin-only.
    Supports plain content, a rich embed, or both. Mentions in the content
    (<@id>, <@&roleid>) are allowed to ping."""
    if not is_admin():
        return jsonify({"error": "unauthorized"}), 403
    if not BOT_TOKEN:
        return jsonify({"error": "Bot token not configured on the server."}), 500

    body = request.get_json(silent=True) or {}
    channel_id = str(body.get("channel_id", "")).strip()
    message = body.get("message", "")
    embed_in = body.get("embed") if isinstance(body.get("embed"), dict) else None

    if not channel_id.isdigit():
        return jsonify({"error": "Enter a valid channel ID (numbers only)."}), 400

    # Build the embed (if provided and non-empty) BEFORE requiring message text,
    # since an embed-only message is valid.
    embed = None
    if embed_in:
        embed = _build_embed(embed_in)
        if isinstance(embed, str):  # _build_embed returns an error string on failure
            return jsonify({"error": embed}), 400

    has_text = isinstance(message, str) and message.strip()
    if not has_text and not embed:
        return jsonify({"error": "Add a message, an embed, or both."}), 400
    if isinstance(message, str) and len(message) > 2000:
        return jsonify({"error": "Message is over Discord's 2000-character limit."}), 400

    payload = {
        # Allow user + role pings to actually fire; block @everyone/@here for safety.
        "allowed_mentions": {"parse": ["users", "roles"]},
    }
    if has_text:
        payload["content"] = message
    if embed:
        payload["embeds"] = [embed]
    try:
        resp = requests.post(
            f"{DISCORD_API}/channels/{channel_id}/messages",
            headers=_bot_headers(), json=payload, timeout=15,
        )
        if resp.status_code in (200, 201):
            log.info("dashboard message sent to channel %s by admin %s",
                     channel_id, session.get("user_id"))
            return jsonify({"ok": True})
        if resp.status_code == 403:
            return jsonify({"error": "The bot can't send to that channel (missing permission)."}), 502
        if resp.status_code == 404:
            return jsonify({"error": "Channel not found — check the ID."}), 502
        return jsonify({"error": f"Discord rejected the message ({resp.status_code})."}), 502
    except Exception as e:
        log.warning("send-message failed: %s", e)
        return jsonify({"error": "Failed to reach Discord."}), 502


def _parse_message_link(link: str):
    """Pull (channel_id, message_id) from a Discord message link or a
    'channel-message' / 'channel/message' shorthand. Returns (ch, msg) or (None, None)."""
    import re
    link = (link or "").strip()
    # Full link: https://discord.com/channels/<guild>/<channel>/<message>
    m = re.search(r"/channels/\d+/(\d+)/(\d+)", link)
    if m:
        return m.group(1), m.group(2)
    # Shorthand: "channelid-messageid" (Discord's "Copy ID" pair) or "channelid/messageid"
    m = re.match(r"^\s*(\d+)[-/](\d+)\s*$", link)
    if m:
        return m.group(1), m.group(2)
    return None, None


@app.route("/api/load-message", methods=["POST"])
def api_load_message():
    """Fetch an existing message (by link) so the composer can edit it.
    Only messages the BOT authored are editable — flagged in the response."""
    if not is_admin():
        return jsonify({"error": "unauthorized"}), 403
    if not BOT_TOKEN:
        return jsonify({"error": "Bot token not configured."}), 500

    body = request.get_json(silent=True) or {}
    ch_id, msg_id = _parse_message_link(body.get("link", ""))
    if not ch_id or not msg_id:
        return jsonify({"error": "Paste a valid message link (right-click a message → Copy Message Link)."}), 400

    try:
        resp = requests.get(
            f"{DISCORD_API}/channels/{ch_id}/messages/{msg_id}",
            headers=_bot_headers(), timeout=15,
        )
        if resp.status_code == 403:
            return jsonify({"error": "The bot can't read that channel (missing permission)."}), 502
        if resp.status_code == 404:
            return jsonify({"error": "Message not found — check the link, or the bot may not be in that channel."}), 502
        if resp.status_code != 200:
            return jsonify({"error": f"Discord rejected the request ({resp.status_code})."}), 502
        msg = resp.json()
    except Exception as e:
        log.warning("load-message failed: %s", e)
        return jsonify({"error": "Failed to reach Discord."}), 502

    # Is this message authored by the bot? (only then can we edit it)
    bot_id = None
    try:
        me = requests.get(f"{DISCORD_API}/users/@me", headers=_bot_headers(), timeout=10)
        if me.status_code == 200:
            bot_id = me.json().get("id")
    except Exception:
        pass
    author_id = (msg.get("author") or {}).get("id")
    editable = bool(bot_id and author_id == bot_id)

    # Convert the first embed (if any) back into the dashboard's form shape
    embed_form = None
    embeds = msg.get("embeds") or []
    if embeds:
        e = embeds[0]
        embed_form = {
            "title": e.get("title", ""),
            "description": e.get("description", ""),
            "url": e.get("url", ""),
            "color": ("#%06x" % e["color"]) if isinstance(e.get("color"), int) else "",
            "author_name": (e.get("author") or {}).get("name", ""),
            "author_icon": (e.get("author") or {}).get("icon_url", ""),
            "footer_text": (e.get("footer") or {}).get("text", ""),
            "footer_icon": (e.get("footer") or {}).get("icon_url", ""),
            "image": (e.get("image") or {}).get("url", ""),
            "thumbnail": (e.get("thumbnail") or {}).get("url", ""),
            "timestamp": bool(e.get("timestamp")),
            "fields": [{"name": f.get("name", ""), "value": f.get("value", ""),
                        "inline": bool(f.get("inline"))} for f in (e.get("fields") or [])],
        }

    return jsonify({
        "ok": True,
        "channel_id": ch_id,
        "message_id": msg_id,
        "content": msg.get("content", ""),
        "embed": embed_form,
        "editable": editable,
        "author_name": (msg.get("author") or {}).get("username", "someone"),
    })


@app.route("/api/edit-message", methods=["POST"])
def api_edit_message():
    """Edit an existing bot message (content and/or embed) via its link."""
    if not is_admin():
        return jsonify({"error": "unauthorized"}), 403
    if not BOT_TOKEN:
        return jsonify({"error": "Bot token not configured."}), 500

    body = request.get_json(silent=True) or {}
    ch_id, msg_id = _parse_message_link(body.get("link", ""))
    if not ch_id or not msg_id:
        return jsonify({"error": "Missing or invalid message link."}), 400

    message = body.get("message", "")
    embed_in = body.get("embed") if isinstance(body.get("embed"), dict) else None
    embed = None
    if embed_in:
        embed = _build_embed(embed_in)
        if isinstance(embed, str):
            return jsonify({"error": embed}), 400

    has_text = isinstance(message, str) and message.strip()
    if not has_text and not embed:
        return jsonify({"error": "The edited message can't be empty — keep some text or an embed."}), 400
    if isinstance(message, str) and len(message) > 2000:
        return jsonify({"error": "Message is over Discord's 2000-character limit."}), 400

    # PATCH: set content and embeds explicitly. Empty content clears text;
    # empty embeds list clears the embed.
    payload = {
        "content": message if has_text else "",
        "embeds": [embed] if embed else [],
        "allowed_mentions": {"parse": ["users", "roles"]},
    }
    try:
        resp = requests.patch(
            f"{DISCORD_API}/channels/{ch_id}/messages/{msg_id}",
            headers=_bot_headers(), json=payload, timeout=15,
        )
        if resp.status_code == 200:
            log.info("dashboard message %s edited by admin %s", msg_id, session.get("user_id"))
            return jsonify({"ok": True})
        if resp.status_code == 403:
            return jsonify({"error": "Can't edit that message — the bot can only edit messages IT sent."}), 502
        if resp.status_code == 404:
            return jsonify({"error": "Message not found — check the link."}), 502
        return jsonify({"error": f"Discord rejected the edit ({resp.status_code})."}), 502
    except Exception as e:
        log.warning("edit-message failed: %s", e)
        return jsonify({"error": "Failed to reach Discord."}), 502



# ── Pages ────────────────────────────────────────────────────────────────
def _render_denied():
    return f"""
<!DOCTYPE html>
<html><head><title>Access Denied</title><style>
body {{ font-family: -apple-system, sans-serif; background:#1a1a1a; color:#eee; display:flex; align-items:center; justify-content:center; height:100vh; margin:0; }}
.box {{ background:#252525; padding:40px; border-radius:12px; text-align:center; max-width:400px; }}
a {{ color:#5865f2; }}
</style></head><body>
<div class="box">
<h1>🚫 Access Denied</h1>
<p>Your Discord account ({session.get('username')}) isn't in the admin allowlist.</p>
<p><a href="/auth/logout">Logout</a></p>
</div></body></html>
"""


def _render_dashboard():
    return DASHBOARD_HTML.replace("__USERNAME__", session.get("username", "?"))


DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Jordan Belfort — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0e0e10; --card: #1a1a1d; --border: #2a2a2f; --text: #e6e6e6;
  --muted: #8a8a92; --accent: #5865f2; --green: #43b581; --red: #f04747;
  --gold: #f1c40f; --purple: #9b59b6;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; flex-wrap: wrap; gap: 16px; }
h1 { font-size: 28px; }
.user-info { color: var(--muted); font-size: 14px; }
.user-info a { color: var(--accent); text-decoration: none; margin-left: 12px; }
nav.tabs { display: flex; gap: 4px; margin-bottom: 24px; border-bottom: 1px solid var(--border); overflow-x: auto; }
.tab-btn { background: transparent; color: var(--muted); border: none; padding: 12px 20px; font-size: 14px; font-weight: 500; cursor: pointer; border-bottom: 2px solid transparent; white-space: nowrap; }
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-content { display: none; }
.tab-content.active { display: block; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
.kpi-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.kpi-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-value { font-size: 28px; font-weight: 600; margin-top: 6px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }
.card h2 { font-size: 16px; margin-bottom: 16px; }
.card .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
canvas { width: 100% !important; max-height: 320px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { text-align: left; color: var(--muted); font-weight: 500; padding: 8px 4px; border-bottom: 1px solid var(--border); }
td { padding: 10px 4px; border-bottom: 1px solid var(--border); }
.num { font-variant-numeric: tabular-nums; }
.right { text-align: right; }
.feed-item { padding: 10px 0; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; gap: 12px; }
.feed-user { font-weight: 500; color: var(--accent); }
.feed-time { color: var(--muted); font-size: 12px; white-space: nowrap; }
.loading { color: var(--muted); padding: 40px; text-align: center; }
.error { color: var(--red); padding: 20px; }
/* Custom Messages composer */
.cm-label { display:block; color: var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.5px; margin-bottom:6px; }
.cm-field { position: relative; }
.cm-input { width:100%; background:#0e0e10; color:var(--text); border:1px solid var(--border); border-radius:8px; padding:11px 12px; font-size:14px; font-family:inherit; resize:vertical; }
.cm-input:focus { outline:none; border-color:var(--accent); }
textarea.cm-input { line-height:1.5; }
.cm-dropdown { position:absolute; left:0; right:0; top:calc(100% + 4px); background:var(--card); border:1px solid var(--border); border-radius:8px; max-height:240px; overflow-y:auto; z-index:50; display:none; box-shadow:0 8px 24px rgba(0,0,0,.4); }
.cm-dropdown.show { display:block; }
.cm-mention-dropdown { top:auto; }
.cm-item { padding:9px 12px; cursor:pointer; font-size:14px; display:flex; align-items:center; gap:8px; }
.cm-item:hover, .cm-item.active { background:var(--accent); color:#fff; }
.cm-item .cm-tag { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }
.cm-item.active .cm-tag, .cm-item:hover .cm-tag { color:#dfe0ff; }
.cm-chosen { margin-top:8px; font-size:13px; color:var(--green); min-height:18px; }
.cm-send { background:var(--accent); color:#fff; border:none; padding:10px 22px; border-radius:8px; font-size:14px; font-weight:600; cursor:pointer; }
.cm-send:hover { filter:brightness(1.1); }
.cm-send:disabled { opacity:.5; cursor:not-allowed; }
.cm-status { margin-top:12px; font-size:14px; min-height:20px; }
.cm-editbox { margin:14px 0 4px; padding:12px 14px; background:#0e0e10; border:1px solid var(--border); border-radius:10px; }
/* Voice analytics tab */
.vx-live-banner { margin-bottom:16px; }
.vx-live-card { background:linear-gradient(135deg,rgba(0,168,252,.12),rgba(61,255,154,.06)); border:1px solid rgba(0,168,252,.3); border-radius:12px; padding:14px 18px; margin-bottom:10px; }
.vx-live-card .vx-ch { font-weight:700; color:#00a8fc; font-size:15px; margin-bottom:8px; display:flex; align-items:center; gap:8px; }
.vx-live-card .vx-ch .dot { width:8px; height:8px; border-radius:50%; background:#3dff9a; box-shadow:0 0 8px #3dff9a; animation:vxpulse 1.5s infinite; }
@keyframes vxpulse { 0%,100%{opacity:1;} 50%{opacity:.4;} }
.vx-live-members { display:flex; flex-wrap:wrap; gap:8px; }
.vx-chip { background:#0e0e10; border:1px solid var(--border); border-radius:20px; padding:4px 12px; font-size:13px; display:flex; align-items:center; gap:6px; }
.vx-empty-live { color:var(--muted); font-size:13px; padding:10px; }
.vx-heatmap { display:grid; grid-template-columns:auto repeat(24,1fr); gap:2px; font-size:10px; }
.vx-heatmap .hlabel { color:var(--muted); font-size:10px; display:flex; align-items:center; justify-content:flex-end; padding-right:6px; }
.vx-heatmap .hhour { color:var(--muted); font-size:9px; text-align:center; }
.vx-heatmap .cell { aspect-ratio:1; border-radius:2px; background:#151517; transition:transform .1s; cursor:default; }
.vx-heatmap .cell:hover { transform:scale(1.4); z-index:2; position:relative; }
.vx-pairs { display:flex; flex-direction:column; gap:8px; }
.vx-pair { display:flex; align-items:center; gap:10px; font-size:13px; }
.vx-pair .bar { height:8px; border-radius:4px; background:linear-gradient(90deg,#00a8fc,#3dff9a); }
.vx-pair .names { flex:0 0 auto; min-width:180px; }
.vx-pair .amt { color:var(--muted); font-size:12px; margin-left:auto; }
.vx-recent { display:flex; flex-direction:column; gap:6px; max-height:360px; overflow-y:auto; }
.vx-recent .row { display:grid; grid-template-columns:1.5fr 1.2fr 1fr .8fr; gap:10px; padding:8px 10px; background:#0e0e10; border-radius:6px; font-size:13px; align-items:center; }
.vx-recent .row .u { font-weight:600; }
.vx-recent .row .c { color:#00a8fc; }
.vx-recent .row .d { color:var(--muted); }
.vx-recent .head { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.5px; background:transparent; }
/* Embed builder */
.eb-toggle { display:flex; align-items:center; gap:8px; margin-top:20px; padding-top:16px; border-top:1px solid var(--border); cursor:pointer; user-select:none; }
.eb-toggle input { width:16px; height:16px; accent-color:var(--accent); cursor:pointer; }
.eb-toggle span { font-size:14px; font-weight:600; }
.eb-body { display:none; margin-top:14px; gap:20px; }
.eb-body.open { display:grid; grid-template-columns:1fr 300px; }
@media (max-width:800px){ .eb-body.open { grid-template-columns:1fr; } }
.eb-form { display:flex; flex-direction:column; gap:12px; }
.eb-row { display:flex; gap:10px; }
.eb-row > * { flex:1; }
.eb-mini { font-size:11px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted); margin-bottom:4px; display:block; }
.eb-in { width:100%; background:#0e0e10; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:8px 10px; font-size:13px; font-family:inherit; resize:vertical; }
.eb-color { display:flex; align-items:center; gap:8px; }
.eb-color input[type=color] { width:38px; height:34px; padding:0; border:1px solid var(--border); border-radius:6px; background:#0e0e10; cursor:pointer; }
.eb-field { border:1px solid var(--border); border-radius:8px; padding:10px; background:#0e0e10; }
.eb-field .eb-row { align-items:flex-end; }
.eb-inline-lbl { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); white-space:nowrap; flex:0 0 auto; }
.eb-fieldbtns { display:flex; gap:8px; margin-top:4px; }
.eb-btn { background:transparent; border:1px solid var(--border); color:var(--text); border-radius:6px; padding:6px 12px; font-size:12px; cursor:pointer; }
.eb-btn:hover { border-color:var(--accent); }
.eb-del { color:var(--red); border-color:transparent; padding:4px 8px; }
/* Live preview — mimics a Discord embed */
.eb-preview-wrap { position:sticky; top:20px; align-self:start; }
.eb-preview-lbl { font-size:11px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted); margin-bottom:8px; }
.eb-preview { background:#313338; border-radius:4px; padding:2px 0; }
.eb-embed { border-left:4px solid #4f545c; background:#2b2d31; border-radius:4px; padding:12px 16px 16px; margin:2px 0; max-width:100%; position:relative; }
.eb-embed .pv-author { display:flex; align-items:center; gap:8px; font-size:13px; font-weight:600; color:#fff; margin-bottom:6px; }
.eb-embed .pv-author img { width:20px; height:20px; border-radius:50%; }
.eb-embed .pv-title { color:#00a8fc; font-size:15px; font-weight:600; margin-bottom:6px; word-wrap:break-word; }
.eb-embed .pv-desc { color:#dbdee1; font-size:13px; line-height:1.4; white-space:pre-wrap; word-wrap:break-word; margin-bottom:6px; }
.eb-embed .pv-fields { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0; }
.eb-embed .pv-field { min-width:100%; }
.eb-embed .pv-field.inline { min-width:30%; flex:1; }
.eb-embed .pv-fname { color:#fff; font-size:13px; font-weight:600; margin-bottom:2px; }
.eb-embed .pv-fval { color:#dbdee1; font-size:13px; white-space:pre-wrap; }
.eb-embed .pv-thumb { position:absolute; top:12px; right:16px; width:60px; height:60px; border-radius:6px; object-fit:cover; }
.eb-embed .pv-image { width:100%; border-radius:6px; margin-top:8px; }
.eb-embed .pv-footer { display:flex; align-items:center; gap:8px; font-size:11px; color:#949ba4; margin-top:8px; }
.eb-embed .pv-footer img { width:18px; height:18px; border-radius:50%; }
.eb-empty { color:#949ba4; font-size:12px; padding:16px; text-align:center; }
.cm-status.ok { color:var(--green); }
.cm-status.err { color:var(--red); }
</style>
</head>
<body>
<header>
  <div>
    <h1>📊 Jordan Belfort — Analytics Dashboard</h1>
    <div class="user-info" style="margin-top: 6px;">Logged in as <strong>__USERNAME__</strong><a href="/auth/logout">Logout</a></div>
  </div>
  <button onclick="loadStats()" style="background:var(--card);color:var(--text);border:1px solid var(--border);padding:8px 16px;border-radius:6px;cursor:pointer;">🔄 Refresh</button>
</header>
<div id="loading" class="loading">Loading stats...</div>
<div id="error" class="error" style="display:none;"></div>
<div id="main" style="display:none;">
  <div class="kpi-grid" id="kpi-grid"></div>
  <nav class="tabs" id="tabs">
    <button class="tab-btn active" data-tab="overview">Overview</button>
    <button class="tab-btn" data-tab="commands">Commands</button>
    <button class="tab-btn" data-tab="economy">Economy</button>
    <button class="tab-btn" data-tab="users">Users</button>
    <button class="tab-btn" data-tab="games">Games</button>
    <button class="tab-btn" data-tab="features">Features</button>
    <button class="tab-btn" data-tab="feed">Live Feed</button>
    <button class="tab-btn" data-tab="voice">🎙️ Voice</button>
    <button class="tab-btn" data-tab="messages">Custom Messages</button>
  </nav>
  <div id="tab-overview" class="tab-content active">
    <div class="grid">
      <div class="card"><h2>📈 Activity — Last 14 Days</h2><canvas id="overview-activity"></canvas></div>
      <div class="card"><h2>💰 Economy — Earned vs Spent</h2><canvas id="overview-economy"></canvas></div>
      <div class="card"><h2>🏆 Top Commands</h2><canvas id="overview-commands"></canvas></div>
      <div class="card"><h2>🎮 Feature Usage</h2><canvas id="overview-features"></canvas></div>
    </div>
  </div>
  <div id="tab-commands" class="tab-content">
    <div class="grid">
      <div class="card"><h2>🔥 Top 15 Commands (All Time)</h2><canvas id="commands-top"></canvas></div>
      <div class="card"><h2>📊 Top 5 Commands — Last 14 Days</h2><canvas id="commands-trend"></canvas></div>
    </div>
    <div class="card" style="grid-column: 1 / -1;">
      <h2>📋 All Commands — Full Usage</h2>
      <input id="cmd-search" placeholder="Filter commands…" style="width:100%;padding:8px;margin-bottom:10px;background:#1a1a2e;border:1px solid #333;border-radius:6px;color:#eee;">
      <div id="all-commands-table"></div>
    </div>
  </div>
  <div id="tab-economy" class="tab-content">
    <div class="grid">
      <div class="card"><h2>💵 Daily Economy Flow</h2><canvas id="economy-flow"></canvas></div>
      <div class="card"><h2>📜 Transactions Per Day</h2><canvas id="economy-tx"></canvas></div>
      <div class="card" style="grid-column: 1 / -1;"><h2>🏆 Top 10 Richest Users</h2><div id="top-users-table"></div></div>
    </div>
  </div>
  <div id="tab-users" class="tab-content">
    <div class="grid"><div class="card"><h2>👥 Active vs New Users — Last 14 Days</h2><canvas id="users-activity"></canvas></div></div>
  </div>
  <div id="tab-games" class="tab-content">
    <div class="card"><h2>🎲 Game Stats</h2><div id="games-table"></div></div>
  </div>
  <div id="tab-features" class="tab-content">
    <div class="grid"><div class="card"><h2>🎮 Feature Usage Breakdown</h2><canvas id="features-chart"></canvas></div></div>
  </div>
  <div id="tab-feed" class="tab-content">
    <div class="card"><h2>📰 Live Activity Feed</h2><div class="subtitle">Last 30 events. Refresh to update.</div><div id="activity-feed"></div></div>
  </div>

  <div id="tab-voice" class="tab-content">
    <div class="vx-live-banner" id="vx-live-banner"></div>
    <div class="stat-grid" id="vx-cards"></div>
    <div class="card-row">
      <div class="chart-card">
        <h3>🔥 Activity Heatmap <span class="hint">joins by day &amp; hour</span></h3>
        <div id="vx-heatmap" class="vx-heatmap"></div>
      </div>
    </div>
    <div class="card-row">
      <div class="chart-card"><h3>🏆 Top Talkers <span class="hint">total time in voice</span></h3><canvas id="vx-top-users"></canvas></div>
      <div class="chart-card"><h3>📊 Busiest Channels</h3><canvas id="vx-channels"></canvas></div>
    </div>
    <div class="card-row">
      <div class="chart-card"><h3>📈 Sessions / Day <span class="hint">last 14 days</span></h3><canvas id="vx-daily"></canvas></div>
      <div class="chart-card"><h3>🤝 Who Hangs Together <span class="hint">shared voice time</span></h3><div id="vx-pairs" class="vx-pairs"></div></div>
    </div>
    <div class="card-row">
      <div class="chart-card" style="flex:1"><h3>🕐 Recent Sessions</h3><div id="vx-recent" class="vx-recent"></div></div>
    </div>
  </div>

  <div id="tab-messages" class="tab-content">
    <div class="card" style="max-width:720px;">
      <h2>✉️ Send a Custom Message</h2>
      <div class="subtitle">Posts as the bot — no sender shown. Type <strong>@</strong> to mention a role or user (searchable). Markdown works.</div>

      <!-- EDIT MODE -->
      <div class="cm-editbox">
        <label class="eb-toggle" for="cm-edit-enable" style="margin-top:0;border-top:none;padding-top:0;">
          <input type="checkbox" id="cm-edit-enable" />
          <span>✏️ Edit an existing message instead</span>
        </label>
        <div id="cm-edit-panel" style="display:none; margin-top:10px;">
          <label class="cm-label">Message link</label>
          <div class="cm-field" style="display:flex; gap:8px;">
            <input id="cm-edit-link" class="cm-input" type="text" placeholder="Paste message link (right-click message → Copy Message Link)" style="flex:1;" />
            <button id="cm-load-btn" class="eb-btn" style="white-space:nowrap;">Load</button>
          </div>
          <div id="cm-edit-status" class="cm-status" style="min-height:0;"></div>
        </div>
      </div>

      <label class="cm-label">Channel</label>
      <div class="cm-field">
        <input id="cm-channel-search" class="cm-input" type="text" placeholder="Search channels or paste a channel ID…" autocomplete="off" />
        <div id="cm-channel-list" class="cm-dropdown"></div>
      </div>
      <input id="cm-channel-id" type="hidden" />
      <div id="cm-channel-chosen" class="cm-chosen"></div>

      <label class="cm-label" style="margin-top:16px;">Message</label>
      <div class="cm-field">
        <textarea id="cm-message" class="cm-input" rows="7" placeholder="Type your message… use @ to mention. **bold**, *italic*, `code`, > quotes all work."></textarea>
        <div id="cm-mention-list" class="cm-dropdown cm-mention-dropdown"></div>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
        <span id="cm-charcount" class="subtitle" style="margin:0;">0 / 2000</span>
      </div>

      <!-- EMBED BUILDER -->
      <label class="eb-toggle" for="eb-enable">
        <input type="checkbox" id="eb-enable" />
        <span>📐 Add an embed</span>
      </label>
      <div id="eb-body" class="eb-body">
        <div class="eb-form">
          <div>
            <label class="eb-mini">Color</label>
            <div class="eb-color">
              <input type="color" id="eb-color" value="#f1c40f" />
              <input type="text" id="eb-color-hex" class="eb-in" value="#f1c40f" style="max-width:110px;" />
              <span class="subtitle" style="margin:0;">left bar color</span>
            </div>
          </div>
          <div class="eb-row">
            <div><label class="eb-mini">Author name</label><input type="text" id="eb-author" class="eb-in" placeholder="e.g. Jordan Belfort" /></div>
            <div><label class="eb-mini">Author icon URL</label><input type="text" id="eb-author-icon" class="eb-in" placeholder="https://…" /></div>
          </div>
          <div class="eb-row">
            <div><label class="eb-mini">Title</label><input type="text" id="eb-title" class="eb-in" placeholder="Embed title" /></div>
            <div><label class="eb-mini">Title link URL</label><input type="text" id="eb-url" class="eb-in" placeholder="https://… (optional)" /></div>
          </div>
          <div>
            <label class="eb-mini">Description</label>
            <textarea id="eb-desc" class="eb-in" rows="4" placeholder="Main body text. Markdown works — **bold**, *italic*, links, etc."></textarea>
          </div>
          <div class="eb-row">
            <div><label class="eb-mini">Thumbnail URL</label><input type="text" id="eb-thumb" class="eb-in" placeholder="https://… (small, top-right)" /></div>
            <div><label class="eb-mini">Big image URL</label><input type="text" id="eb-image" class="eb-in" placeholder="https://… (full width)" /></div>
          </div>
          <div class="eb-row">
            <div><label class="eb-mini">Footer text</label><input type="text" id="eb-footer" class="eb-in" placeholder="Footer line" /></div>
            <div><label class="eb-mini">Footer icon URL</label><input type="text" id="eb-footer-icon" class="eb-in" placeholder="https://…" /></div>
          </div>
          <label class="eb-inline-lbl"><input type="checkbox" id="eb-timestamp" /> Add current timestamp</label>

          <div>
            <label class="eb-mini">Fields</label>
            <div id="eb-fields"></div>
            <div class="eb-fieldbtns">
              <button type="button" class="eb-btn" id="eb-add-field">+ Add field</button>
            </div>
          </div>
        </div>

        <div class="eb-preview-wrap">
          <div class="eb-preview-lbl">Live preview</div>
          <div class="eb-preview"><div id="eb-preview-inner"><div class="eb-empty">Your embed preview shows here as you type.</div></div></div>
        </div>
      </div>

      <div style="display:flex; justify-content:flex-end; align-items:center; margin-top:16px;">
        <button id="cm-send-btn" class="cm-send">Send Message</button>
      </div>
      <div id="cm-status" class="cm-status"></div>
    </div>
  </div>
</div>
<script>
let charts = {};
const palette = ['#5865f2','#43b581','#f04747','#f1c40f','#9b59b6','#1abc9c','#e67e22','#3498db'];
function destroyChart(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }
function fmt(n) { return (n || 0).toLocaleString(); }
function timeAgo(ts) {
  const diff = Math.floor(Date.now()/1000 - ts);
  if (diff < 60) return diff + 's ago';
  if (diff < 3600) return Math.floor(diff/60) + 'm ago';
  if (diff < 86400) return Math.floor(diff/3600) + 'h ago';
  return Math.floor(diff/86400) + 'd ago';
}
async function loadStats() {
  document.getElementById('loading').style.display = 'block';
  document.getElementById('error').style.display = 'none';
  document.getElementById('main').style.display = 'none';
  try {
    const resp = await fetch('/api/stats');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    renderDashboard(data);
    document.getElementById('loading').style.display = 'none';
    document.getElementById('main').style.display = 'block';
  } catch (e) {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('error').textContent = 'Failed to load: ' + e.message;
    document.getElementById('error').style.display = 'block';
  }
}
function renderDashboard(data) {
  renderKPIs(data.summary);
  renderOverview(data); renderCommands(data); renderEconomy(data);
  renderUsers(data); renderGames(data); renderFeatures(data); renderFeed(data);
}
function renderKPIs(s) {
  const cards = [
    ['Active Today', fmt(s.active_users_today)],
    ['New Today', fmt(s.new_users_today)],
    ['Total Users', fmt(s.total_users)],
    ['Coins In Circulation', fmt(s.total_coins)],
    ['Commands Today', fmt(s.transactions_today)],
    ['Total Commands', fmt(s.total_commands_ever)],
  ];
  document.getElementById('kpi-grid').innerHTML = cards.map(([k,v]) => `<div class="kpi-card"><div class="kpi-label">${k}</div><div class="kpi-value">${v}</div></div>`).join('');
}
function renderOverview(d) {
  destroyChart('overview-activity');
  charts['overview-activity'] = new Chart(document.getElementById('overview-activity'), {
    type: 'line',
    data: { labels: d.activity_trend.days, datasets: [
      { label: 'Active Users', data: d.activity_trend.active, borderColor: palette[0], backgroundColor: 'transparent', tension: 0.3 },
      { label: 'New Users', data: d.activity_trend.new, borderColor: palette[1], backgroundColor: 'transparent', tension: 0.3 },
    ]}, options: chartOpts(),
  });
  destroyChart('overview-economy');
  charts['overview-economy'] = new Chart(document.getElementById('overview-economy'), {
    type: 'bar',
    data: { labels: d.economy_trend.days, datasets: [
      { label: 'Earned', data: d.economy_trend.earned, backgroundColor: palette[1] },
      { label: 'Spent', data: d.economy_trend.spent, backgroundColor: palette[2] },
    ]}, options: chartOpts(),
  });
  destroyChart('overview-commands');
  const top10 = d.top_commands.slice(0, 10);
  charts['overview-commands'] = new Chart(document.getElementById('overview-commands'), {
    type: 'bar',
    data: { labels: top10.map(c => c.name), datasets: [{ label: 'Uses', data: top10.map(c => c.count), backgroundColor: palette[0] }] },
    options: { ...chartOpts(), indexAxis: 'y' },
  });
  destroyChart('overview-features');
  const feats = Object.entries(d.feature_usage || {});
  charts['overview-features'] = new Chart(document.getElementById('overview-features'), {
    type: 'doughnut',
    data: { labels: feats.map(([k,_]) => k), datasets: [{ data: feats.map(([_,v]) => v), backgroundColor: palette }] },
    options: chartOpts(true),
  });
}
function renderCommands(d) {
  const top15 = d.top_commands.slice(0, 15);
  destroyChart('commands-top');
  charts['commands-top'] = new Chart(document.getElementById('commands-top'), {
    type: 'bar',
    data: { labels: top15.map(c => c.name), datasets: [{ label: 'Uses', data: top15.map(c => c.count), backgroundColor: palette[0] }] },
    options: { ...chartOpts(), indexAxis: 'y' },
  });
  destroyChart('commands-trend');
  charts['commands-trend'] = new Chart(document.getElementById('commands-trend'), {
    type: 'line',
    data: { labels: d.command_trend.days, datasets: Object.entries(d.command_trend.series).map(([name, series], i) => ({
      label: name, data: series, borderColor: palette[i % palette.length], backgroundColor: 'transparent', tension: 0.3,
    }))},
    options: chartOpts(),
  });
  // Full commands table — EVERY command, with usage counts
  window._allCommands = d.top_commands;
  renderCommandsTable('');
  const search = document.getElementById('cmd-search');
  if (search && !search._wired) {
    search._wired = true;
    search.addEventListener('input', e => renderCommandsTable(e.target.value));
  }
}
function renderCommandsTable(filter) {
  const cmds = (window._allCommands || []).filter(c => c.name.toLowerCase().includes((filter||'').toLowerCase()));
  const total = cmds.reduce((a, c) => a + c.count, 0);
  const rows = cmds.map((c, i) => `<tr><td>${i+1}</td><td>${c.name}</td><td class="right">${c.count.toLocaleString()}</td></tr>`).join('');
  document.getElementById('all-commands-table').innerHTML =
    `<p style="color:var(--muted);margin-bottom:8px;">${cmds.length} commands shown • ${total.toLocaleString()} total uses</p>`
    + `<table><thead><tr><th>#</th><th>Command</th><th class="right">Total Uses</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function renderEconomy(d) {
  destroyChart('economy-flow');
  charts['economy-flow'] = new Chart(document.getElementById('economy-flow'), {
    type: 'line',
    data: { labels: d.economy_trend.days, datasets: [
      { label: 'Earned', data: d.economy_trend.earned, borderColor: palette[1], backgroundColor: 'rgba(67,181,129,0.1)', tension: 0.3, fill: true },
      { label: 'Spent', data: d.economy_trend.spent, borderColor: palette[2], backgroundColor: 'rgba(240,71,71,0.1)', tension: 0.3, fill: true },
    ]}, options: chartOpts(),
  });
  destroyChart('economy-tx');
  charts['economy-tx'] = new Chart(document.getElementById('economy-tx'), {
    type: 'bar',
    data: { labels: d.economy_trend.days, datasets: [{ label: 'Transactions', data: d.economy_trend.transactions, backgroundColor: palette[3] }] },
    options: chartOpts(),
  });
  document.getElementById('top-users-table').innerHTML = '<table><thead><tr><th>#</th><th>User ID</th><th class="right">Balance</th></tr></thead><tbody>'
    + d.top_users.map((u,i) => `<tr><td>${i+1}</td><td>${u.user_id}</td><td class="right num">${fmt(u.balance)}</td></tr>`).join('') + '</tbody></table>';
}
function renderUsers(d) {
  destroyChart('users-activity');
  charts['users-activity'] = new Chart(document.getElementById('users-activity'), {
    type: 'line',
    data: { labels: d.activity_trend.days, datasets: [
      { label: 'Active', data: d.activity_trend.active, borderColor: palette[0], backgroundColor: 'rgba(88,101,242,0.1)', fill: true, tension: 0.3 },
      { label: 'New', data: d.activity_trend.new, borderColor: palette[1], backgroundColor: 'rgba(67,181,129,0.1)', fill: true, tension: 0.3 },
    ]}, options: chartOpts(),
  });
}
function renderGames(d) {
  if (!d.games.length) { document.getElementById('games-table').innerHTML = '<p style="color:var(--muted);padding:20px;">No game data yet.</p>'; return; }
  const rows = d.games.map(g => `<tr><td>${g.name}</td><td class="num">${fmt(g.plays)}</td><td class="num" style="color:var(--green)">${fmt(g.wins)}</td><td class="num" style="color:var(--red)">${fmt(g.losses)}</td><td class="num">${g.win_rate}%</td><td class="num right">${fmt(g.total_payout)}</td><td class="num right">${fmt(g.biggest_payout)}</td></tr>`).join('');
  document.getElementById('games-table').innerHTML = `<table><thead><tr><th>Game</th><th>Plays</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th class="right">Total Payout</th><th class="right">Biggest</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function renderFeatures(d) {
  destroyChart('features-chart');
  const feats = Object.entries(d.feature_usage || {});
  charts['features-chart'] = new Chart(document.getElementById('features-chart'), {
    type: 'bar',
    data: { labels: feats.map(([k,_]) => k), datasets: [{ label: 'Uses', data: feats.map(([_,v]) => v), backgroundColor: palette }] },
    options: chartOpts(),
  });
}
function renderFeed(d) {
  if (!d.activity_feed.length) { document.getElementById('activity-feed').innerHTML = '<p style="color:var(--muted);padding:20px;">No activity yet.</p>'; return; }
  document.getElementById('activity-feed').innerHTML = d.activity_feed.map(e => `<div class="feed-item"><div><span class="feed-user">${e.user_name}</span> ${e.detail}</div><div class="feed-time">${timeAgo(e.ts)}</div></div>`).join('');
}
function chartOpts(forDoughnut) {
  return {
    responsive: true, maintainAspectRatio: true,
    plugins: { legend: { labels: { color: '#e6e6e6' } } },
    scales: forDoughnut ? {} : {
      y: { beginAtZero: true, ticks: { color: '#8a8a92' }, grid: { color: '#2a2a2f' } },
      x: { ticks: { color: '#8a8a92' }, grid: { color: '#2a2a2f' } },
    },
  };
}

// ── Custom Messages tab ──────────────────────────────────────────────────
let cmMeta = { channels: [], roles: [], members: [] };
let cmMetaLoaded = false;

async function loadGuildMeta() {
  if (cmMetaLoaded) return;
  try {
    const r = await fetch('/api/guild-meta');
    const d = await r.json();
    if (d.error) { setCmStatus(d.error, false); return; }
    cmMeta = d;
    cmMetaLoaded = true;
  } catch (e) { setCmStatus('Could not load server data.', false); }
}

function setCmStatus(msg, ok) {
  const el = document.getElementById('cm-status');
  el.textContent = msg || '';
  el.className = 'cm-status ' + (msg ? (ok ? 'ok' : 'err') : '');
}

// ---- Channel picker ----
function initChannelPicker() {
  const search = document.getElementById('cm-channel-search');
  const list = document.getElementById('cm-channel-list');
  const hidden = document.getElementById('cm-channel-id');
  const chosen = document.getElementById('cm-channel-chosen');

  function render(filter) {
    const f = (filter || '').toLowerCase().replace(/^#/, '');
    let items = cmMeta.channels.filter(c => c.name.toLowerCase().includes(f)).slice(0, 50);
    if (!items.length) { list.classList.remove('show'); return; }
    list.innerHTML = items.map(c =>
      `<div class="cm-item" data-id="${c.id}" data-name="${c.name}">#${c.name} <span class="cm-tag">${c.id}</span></div>`
    ).join('');
    list.classList.add('show');
    list.querySelectorAll('.cm-item').forEach(it => {
      it.onclick = () => {
        hidden.value = it.dataset.id;
        search.value = '#' + it.dataset.name;
        chosen.textContent = '✓ Sending to #' + it.dataset.name;
        list.classList.remove('show');
      };
    });
  }
  search.addEventListener('input', e => {
    const v = e.target.value.trim();
    // Allow pasting a raw numeric ID directly
    if (/^\d{5,}$/.test(v)) { hidden.value = v; chosen.textContent = '✓ Using channel ID ' + v; list.classList.remove('show'); return; }
    hidden.value = '';
    chosen.textContent = '';
    render(v);
  });
  search.addEventListener('focus', () => render(search.value));
  document.addEventListener('click', e => { if (!list.contains(e.target) && e.target !== search) list.classList.remove('show'); });
}

// ---- @ mention autocomplete in the textarea ----
function initMentionAutocomplete() {
  const ta = document.getElementById('cm-message');
  const menu = document.getElementById('cm-mention-list');
  const charcount = document.getElementById('cm-charcount');
  let activeIdx = 0;
  let matches = [];
  let triggerPos = -1;

  function updateCount() {
    const n = ta.value.length;
    charcount.textContent = n + ' / 2000';
    charcount.style.color = n > 2000 ? 'var(--red)' : 'var(--muted)';
  }

  function closeMenu() { menu.classList.remove('show'); matches = []; triggerPos = -1; }

  function currentQuery() {
    // Find an @ before the caret with no whitespace after it
    const pos = ta.selectionStart;
    const text = ta.value.slice(0, pos);
    const at = text.lastIndexOf('@');
    if (at === -1) return null;
    const between = text.slice(at + 1);
    if (/\s/.test(between)) return null;      // whitespace ends a mention query
    return { at, query: between };
  }

  function renderMenu() {
    const q = currentQuery();
    if (q === null) { closeMenu(); return; }
    triggerPos = q.at;
    const query = q.query.toLowerCase();
    const roles = cmMeta.roles
      .filter(r => r.name.toLowerCase().includes(query))
      .map(r => ({ type: 'role', id: r.id, name: r.name }));
    const users = cmMeta.members
      .filter(m => m.name.toLowerCase().includes(query) || (m.username || '').toLowerCase().includes(query))
      .map(m => ({ type: 'user', id: m.id, name: m.name }));
    // Roles first, then users, capped
    matches = roles.concat(users).slice(0, 40);
    if (!matches.length) { closeMenu(); return; }
    activeIdx = 0;
    menu.innerHTML = matches.map((m, i) =>
      `<div class="cm-item ${i===0?'active':''}" data-i="${i}">` +
      `${m.type==='role' ? '🏷️' : '👤'} ${m.name} <span class="cm-tag">${m.type}</span></div>`
    ).join('');
    menu.classList.add('show');
    menu.querySelectorAll('.cm-item').forEach(it => {
      it.onmousedown = (e) => { e.preventDefault(); choose(parseInt(it.dataset.i)); };
    });
  }

  function choose(i) {
    const m = matches[i];
    if (!m) return;
    const pos = ta.selectionStart;
    const before = ta.value.slice(0, triggerPos);
    const after = ta.value.slice(pos);
    // Insert the real mention token so Discord actually pings
    const token = m.type === 'role' ? `<@&${m.id}>` : `<@${m.id}>`;
    ta.value = before + token + ' ' + after;
    const newPos = (before + token + ' ').length;
    ta.setSelectionRange(newPos, newPos);
    ta.focus();
    closeMenu();
    updateCount();
  }

  ta.addEventListener('input', () => { updateCount(); renderMenu(); });
  ta.addEventListener('keydown', (e) => {
    if (!menu.classList.contains('show')) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); activeIdx = Math.min(activeIdx+1, matches.length-1); highlight(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); activeIdx = Math.max(activeIdx-1, 0); highlight(); }
    else if (e.key === 'Enter' || e.key === 'Tab') { e.preventDefault(); choose(activeIdx); }
    else if (e.key === 'Escape') { closeMenu(); }
  });
  function highlight() {
    menu.querySelectorAll('.cm-item').forEach((it, i) => it.classList.toggle('active', i === activeIdx));
    const active = menu.querySelector('.cm-item.active');
    if (active) active.scrollIntoView({ block: 'nearest' });
  }
  ta.addEventListener('blur', () => setTimeout(closeMenu, 150));
  updateCount();
}

function ebCollect() {
  // Returns an embed object, or null if the builder is off / empty.
  if (!document.getElementById('eb-enable').checked) return null;
  const v = id => (document.getElementById(id).value || '').trim();
  const fields = [];
  document.querySelectorAll('#eb-fields .eb-field').forEach(row => {
    const name = row.querySelector('.eb-fname-in').value.trim();
    const value = row.querySelector('.eb-fval-in').value.trim();
    const inline = row.querySelector('.eb-finline').checked;
    if (name && value) fields.push({ name, value, inline });
  });
  const e = {
    title: v('eb-title'), description: v('eb-desc'), url: v('eb-url'),
    color: v('eb-color-hex'),
    author_name: v('eb-author'), author_icon: v('eb-author-icon'),
    footer_text: v('eb-footer'), footer_icon: v('eb-footer-icon'),
    image: v('eb-image'), thumbnail: v('eb-thumb'),
    timestamp: document.getElementById('eb-timestamp').checked,
    fields
  };
  // If nothing meaningful was entered, treat as no embed
  const hasContent = e.title || e.description || fields.length || e.author_name || e.image;
  return hasContent ? e : null;
}

let cmEditMode = false;   // false = send new, true = edit existing
let cmEditLink = '';

function initEditMode() {
  const enable = document.getElementById('cm-edit-enable');
  const panel = document.getElementById('cm-edit-panel');
  const btn = document.getElementById('cm-send-btn');
  enable.addEventListener('change', () => {
    cmEditMode = enable.checked;
    panel.style.display = cmEditMode ? 'block' : 'none';
    btn.textContent = cmEditMode ? 'Save Changes' : 'Send Message';
    if (!cmEditMode) { cmEditLink = ''; }
  });
  document.getElementById('cm-load-btn').addEventListener('click', loadMessageForEdit);
}

function setEditStatus(msg, ok) {
  const el = document.getElementById('cm-edit-status');
  el.textContent = msg;
  el.style.color = ok ? 'var(--green, #3dff9a)' : 'var(--red, #ff6b6b)';
}

async function loadMessageForEdit() {
  const link = document.getElementById('cm-edit-link').value.trim();
  if (!link) { setEditStatus('Paste a message link first.', false); return; }
  setEditStatus('Loading…', true);
  try {
    const r = await fetch('/api/load-message', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ link })
    });
    const d = await r.json();
    if (!d.ok) { setEditStatus(d.error || 'Failed to load.', false); return; }
    if (!d.editable) {
      setEditStatus('⚠️ That message was sent by ' + d.author_name + ', not the bot — it can\\'t be edited. The bot can only edit its own messages.', false);
      return;
    }
    cmEditLink = link;

    // Populate the message box
    document.getElementById('cm-message').value = d.content || '';
    document.getElementById('cm-charcount').textContent = (d.content || '').length + ' / 2000';

    // Populate the embed builder
    const ebEnable = document.getElementById('eb-enable');
    document.getElementById('eb-fields').innerHTML = '';
    if (d.embed) {
      ebEnable.checked = true;
      document.getElementById('eb-body').classList.add('open');
      const e = d.embed;
      const set = (id, val) => { document.getElementById(id).value = val || ''; };
      set('eb-title', e.title); set('eb-desc', e.description); set('eb-url', e.url);
      set('eb-author', e.author_name); set('eb-author-icon', e.author_icon);
      set('eb-footer', e.footer_text); set('eb-footer-icon', e.footer_icon);
      set('eb-image', e.image); set('eb-thumb', e.thumbnail);
      const hex = e.color || '#f1c40f';
      document.getElementById('eb-color-hex').value = hex;
      if (/^#[0-9a-fA-F]{6}$/.test(hex)) document.getElementById('eb-color').value = hex;
      document.getElementById('eb-timestamp').checked = !!e.timestamp;
      (e.fields || []).forEach(f => ebAddField(f.name, f.value, f.inline));
      ebRenderPreview();
    } else {
      ebEnable.checked = false;
      document.getElementById('eb-body').classList.remove('open');
    }
    setEditStatus('✓ Loaded. Make your changes and hit Save Changes.', true);
  } catch (e) { setEditStatus('Failed to reach the server.', false); }
}

function sendOrEditMessage() {
  if (cmEditMode) { editCustomMessage(); }
  else { sendCustomMessage(); }
}

async function editCustomMessage() {
  const btn = document.getElementById('cm-send-btn');
  if (!cmEditLink) { setCmStatus('Load a message first (paste its link and hit Load).', false); return; }
  const message = document.getElementById('cm-message').value;
  const embed = ebCollect();
  if (!message.trim() && !embed) { setCmStatus('The edited message needs text or an embed.', false); return; }
  btn.disabled = true; setCmStatus('Saving…', true);
  try {
    const r = await fetch('/api/edit-message', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ link: cmEditLink, message, embed })
    });
    const d = await r.json();
    if (d.ok) { setCmStatus('✓ Saved! The message has been updated.', true); }
    else { setCmStatus(d.error || 'Failed to save.', false); }
  } catch (e) { setCmStatus('Failed to reach the server.', false); }
  btn.disabled = false;
}

async function sendCustomMessage() {
  const btn = document.getElementById('cm-send-btn');
  const channelId = document.getElementById('cm-channel-id').value.trim();
  const message = document.getElementById('cm-message').value;
  const embed = ebCollect();
  if (!channelId) { setCmStatus('Pick a channel first.', false); return; }
  if (!message.trim() && !embed) { setCmStatus('Write a message or build an embed first.', false); return; }
  if (message.length > 2000) { setCmStatus('Message is over 2000 characters.', false); return; }
  btn.disabled = true; setCmStatus('Sending…', true);
  try {
    const r = await fetch('/api/send-message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel_id: channelId, message, embed })
    });
    const d = await r.json();
    if (d.ok) {
      setCmStatus('✓ Sent!', true);
      document.getElementById('cm-message').value = '';
      document.getElementById('cm-charcount').textContent = '0 / 2000';
    } else {
      setCmStatus(d.error || 'Failed to send.', false);
    }
  } catch (e) { setCmStatus('Failed to reach the server.', false); }
  btn.disabled = false;
}

function ebAddField(name, value, inline) {
  const wrap = document.getElementById('eb-fields');
  const row = document.createElement('div');
  row.className = 'eb-field';
  row.style.marginBottom = '8px';
  row.innerHTML =
    '<div class="eb-row">'
    + '<div style="flex:2"><label class="eb-mini">Field name</label><input type="text" class="eb-in eb-fname-in" placeholder="Name" /></div>'
    + '<label class="eb-inline-lbl"><input type="checkbox" class="eb-finline" /> Inline</label>'
    + '<button type="button" class="eb-btn eb-del" title="Remove">✕</button>'
    + '</div>'
    + '<div style="margin-top:6px"><label class="eb-mini">Field value</label><textarea class="eb-in eb-fval-in" rows="2" placeholder="Value"></textarea></div>';
  wrap.appendChild(row);
  if (name) row.querySelector('.eb-fname-in').value = name;
  if (value) row.querySelector('.eb-fval-in').value = value;
  if (inline) row.querySelector('.eb-finline').checked = true;
  row.querySelector('.eb-del').addEventListener('click', () => { row.remove(); ebRenderPreview(); });
  row.querySelectorAll('input, textarea').forEach(el => el.addEventListener('input', ebRenderPreview));
  ebRenderPreview();
}

function ebEsc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function ebRenderPreview() {
  const e = ebCollect();
  const box = document.getElementById('eb-preview-inner');
  if (!e) { box.innerHTML = '<div class="eb-empty">Your embed preview shows here as you type.</div>'; return; }
  let hex = (document.getElementById('eb-color-hex').value || '#4f545c').trim();
  if (!/^#?[0-9a-fA-F]{6}$/.test(hex)) hex = '#4f545c';
  if (hex[0] !== '#') hex = '#' + hex;
  let h = '<div class="eb-embed" style="border-left-color:' + hex + '">';
  if (e.thumbnail) h += '<img class="pv-thumb" src="' + ebEsc(e.thumbnail) + '" onerror="this.style.display=\'none\'" />';
  if (e.author_name) {
    h += '<div class="pv-author">';
    if (e.author_icon) h += '<img src="' + ebEsc(e.author_icon) + '" onerror="this.style.display=\'none\'" />';
    h += ebEsc(e.author_name) + '</div>';
  }
  if (e.title) h += '<div class="pv-title">' + ebEsc(e.title) + '</div>';
  if (e.description) h += '<div class="pv-desc">' + ebEsc(e.description) + '</div>';
  if (e.fields && e.fields.length) {
    h += '<div class="pv-fields">';
    e.fields.forEach(f => {
      h += '<div class="pv-field' + (f.inline ? ' inline' : '') + '">'
        + '<div class="pv-fname">' + ebEsc(f.name) + '</div>'
        + '<div class="pv-fval">' + ebEsc(f.value) + '</div></div>';
    });
    h += '</div>';
  }
  if (e.image) h += '<img class="pv-image" src="' + ebEsc(e.image) + '" onerror="this.style.display=\'none\'" />';
  if (e.footer_text || e.timestamp) {
    h += '<div class="pv-footer">';
    if (e.footer_icon) h += '<img src="' + ebEsc(e.footer_icon) + '" onerror="this.style.display=\'none\'" />';
    let ft = ebEsc(e.footer_text);
    if (e.timestamp) ft += (ft ? ' • ' : '') + new Date().toLocaleString();
    h += ft + '</div>';
  }
  h += '</div>';
  box.innerHTML = h;
}

function initEmbedBuilder() {
  const enable = document.getElementById('eb-enable');
  const body = document.getElementById('eb-body');
  enable.addEventListener('change', () => {
    body.classList.toggle('open', enable.checked);
    ebRenderPreview();
  });
  // keep the two color inputs in sync
  const colr = document.getElementById('eb-color');
  const hex = document.getElementById('eb-color-hex');
  colr.addEventListener('input', () => { hex.value = colr.value; ebRenderPreview(); });
  hex.addEventListener('input', () => {
    let val = hex.value.trim(); if (val[0] !== '#') val = '#' + val;
    if (/^#[0-9a-fA-F]{6}$/.test(val)) colr.value = val;
    ebRenderPreview();
  });
  document.getElementById('eb-add-field').addEventListener('click', () => ebAddField());
  ['eb-title','eb-desc','eb-url','eb-author','eb-author-icon','eb-footer','eb-footer-icon','eb-image','eb-thumb']
    .forEach(id => document.getElementById(id).addEventListener('input', ebRenderPreview));
  document.getElementById('eb-timestamp').addEventListener('change', ebRenderPreview);
}

let cmInited = false;
function initCustomMessages() {
  if (cmInited) return;
  cmInited = true;
  initChannelPicker();
  initMentionAutocomplete();
  initEmbedBuilder();
  initEditMode();
  document.getElementById('cm-send-btn').addEventListener('click', sendOrEditMessage);
}


document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    // Custom Messages tab: lazy-init the composer + load server data
    if (btn.dataset.tab === 'messages') {
      initCustomMessages();
      loadGuildMeta();
    }
    if (btn.dataset.tab === 'voice') {
      loadVoiceAnalytics();
    }
  });
});
loadStats();
setInterval(loadStats, 60_000);

// ── Voice Analytics ──────────────────────────────────────────────────────────
function vxFmt(sec) {
  sec = Math.round(sec || 0);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h) return h + 'h ' + m + 'm';
  if (m) return m + 'm ' + s + 's';
  return s + 's';
}
function vxAgo(ts) {
  const d = Math.round(Date.now() / 1000 - ts);
  if (d < 60) return d + 's ago';
  if (d < 3600) return Math.floor(d / 60) + 'm ago';
  if (d < 86400) return Math.floor(d / 3600) + 'h ago';
  return Math.floor(d / 86400) + 'd ago';
}
async function loadVoiceAnalytics() {
  let d;
  try {
    const r = await fetch('/api/voice-analytics');
    d = await r.json();
    if (!d.ok) throw new Error(d.error || 'failed');
  } catch (e) {
    document.getElementById('vx-cards').innerHTML = '<div class="vx-empty-live">Couldn\\'t load voice data yet. Once people use voice channels, stats show here.</div>';
    return;
  }
  const s = d.summary;

  // Summary cards
  document.getElementById('vx-cards').innerHTML = [
    ['🔴 In Voice Now', s.live_count],
    ['⏱️ Total Voice Time', vxFmt(s.total_seconds)],
    ['📞 Total Sessions', s.sessions_all.toLocaleString()],
    ['👥 Unique Today', s.unique_today],
    ['📊 Avg Session', vxFmt(s.avg_session)],
    ['🏅 Longest Session', vxFmt(s.longest)],
    ['📺 Streams Logged', s.streams],
    ['📹 Cameras Logged', s.cameras],
  ].map(([label, val]) =>
    '<div class="stat-box"><div class="stat-label">' + label + '</div><div class="stat-value">' + val + '</div></div>'
  ).join('');

  // Live banner
  const banner = document.getElementById('vx-live-banner');
  if (d.live && d.live.length) {
    banner.innerHTML = d.live.map(ch => {
      const chips = ch.members.map(m => {
        const st = m.state || {};
        let icons = '';
        if (st.streaming) icons += '📺'; if (st.camera) icons += '📹';
        if (st.self_mute) icons += '🔇'; if (st.self_deaf) icons += '🔕';
        return '<span class="vx-chip">' + (m.name || '?') + (icons ? ' ' + icons : '') + '</span>';
      }).join('');
      return '<div class="vx-live-card"><div class="vx-ch"><span class="dot"></span>' +
        (ch.channel || '?') + ' — ' + ch.members.length + ' in call</div>' +
        '<div class="vx-live-members">' + chips + '</div></div>';
    }).join('');
  } else {
    banner.innerHTML = '<div class="vx-live-card"><div class="vx-empty-live">🔇 Nobody in voice right now.</div></div>';
  }

  // Heatmap (7x24)
  const days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  let maxH = 1;
  d.heatmap.forEach(row => row.forEach(v => { if (v > maxH) maxH = v; }));
  let hm = '<div class="hlabel"></div>';
  for (let h = 0; h < 24; h++) hm += '<div class="hhour">' + (h % 6 === 0 ? h : '') + '</div>';
  d.heatmap.forEach((row, di) => {
    hm += '<div class="hlabel">' + days[di] + '</div>';
    row.forEach(v => {
      const intensity = v / maxH;
      const bg = v === 0 ? '#151517'
        : 'rgba(0,168,252,' + (0.15 + intensity * 0.85).toFixed(2) + ')';
      hm += '<div class="cell" style="background:' + bg + '" title="' + days[di] + ' ' + '"' + '></div>';
    });
  });
  document.getElementById('vx-heatmap').innerHTML = hm;

  // Top users bar
  destroyChart('vx-top-users');
  charts['vx-top-users'] = new Chart(document.getElementById('vx-top-users'), {
    type: 'bar',
    data: { labels: d.top_users.map(u => u.name), datasets: [{ data: d.top_users.map(u => Math.round(u.seconds / 60)), backgroundColor: '#00a8fc', borderRadius: 4 }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => c.parsed.x + ' min' } } }, scales: { x: { title: { display: true, text: 'minutes' } } } }
  });

  // Channels doughnut
  destroyChart('vx-channels');
  charts['vx-channels'] = new Chart(document.getElementById('vx-channels'), {
    type: 'doughnut',
    data: { labels: d.top_channels.map(c => c.name), datasets: [{ data: d.top_channels.map(c => Math.round(c.seconds / 60)), backgroundColor: ['#00a8fc','#3dff9a','#f1c40f','#e74c3c','#9b59b6','#1abc9c','#e67e22','#34495e','#fd79a8','#00cec9'] }] },
    options: { plugins: { legend: { position: 'right', labels: { boxWidth: 12, font: { size: 11 } } } } }
  });

  // Daily line
  destroyChart('vx-daily');
  charts['vx-daily'] = new Chart(document.getElementById('vx-daily'), {
    type: 'line',
    data: { labels: d.daily.map(x => x[0].slice(5)), datasets: [{ data: d.daily.map(x => x[1]), borderColor: '#3dff9a', backgroundColor: 'rgba(61,255,154,.1)', fill: true, tension: .3 }] },
    options: { plugins: { legend: { display: false } } }
  });

  // Pairs
  const maxP = d.pairs.length ? d.pairs[0].seconds : 1;
  document.getElementById('vx-pairs').innerHTML = d.pairs.length
    ? d.pairs.map(p =>
        '<div class="vx-pair"><span class="names">' + p.a + ' × ' + p.b + '</span>' +
        '<div class="bar" style="width:' + Math.max(6, (p.seconds / maxP) * 120) + 'px"></div>' +
        '<span class="amt">' + vxFmt(p.seconds) + '</span></div>'
      ).join('')
    : '<div class="vx-empty-live">No shared sessions logged yet.</div>';

  // Recent sessions
  const rec = document.getElementById('vx-recent');
  let rh = '<div class="row head"><div>User</div><div>Channel</div><div>Duration</div><div>With</div></div>';
  rh += d.recent.map(r =>
    '<div class="row"><div class="u">' + (r.user || '?') + '</div>' +
    '<div class="c">' + (r.channel || '?') + '</div>' +
    '<div class="d">' + vxFmt(r.duration) + ' · ' + vxAgo(r.joined) + '</div>' +
    '<div class="d">' + (r.with ? r.with + ' others' : 'alone') + '</div></div>'
  ).join('');
  rec.innerHTML = d.recent.length ? rh : '<div class="vx-empty-live">No completed sessions yet.</div>';
}
</script>
</body>
</html>
"""


def start_dashboard(port: int = 8080):
    """Run the Flask server. Called from bot.py on a background thread."""
    log.info("Starting dashboard on port %s", port)
    # Use waitress if available (production-quality), else Flask dev server
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port, threads=4)
    except ImportError:
        # Suppress Flask dev-server "WARNING: This is a development server" noise
        import logging as logging_mod
        logging_mod.getLogger("werkzeug").setLevel(logging_mod.ERROR)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

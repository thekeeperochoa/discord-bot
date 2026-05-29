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

DISCORD_API = "https://discord.com/api"
OAUTH_SCOPE = "identify"

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


@app.route("/api/stats")
def api_stats():
    if not is_admin():
        return jsonify({"error": "unauthorized"}), 403

    stats = _load_json(STATS_FILE, {})
    economy = _load_json(ECONOMY_FILE, {})

    cmd_uses = stats.get("command_uses", {})
    top_commands = sorted(cmd_uses.items(), key=lambda x: x[1], reverse=True)[:15]

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
            "unique_commands": len(cmd_uses),
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
  destroyChart('commands-top');
  charts['commands-top'] = new Chart(document.getElementById('commands-top'), {
    type: 'bar',
    data: { labels: d.top_commands.map(c => c.name), datasets: [{ label: 'Uses', data: d.top_commands.map(c => c.count), backgroundColor: palette[0] }] },
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
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});
loadStats();
setInterval(loadStats, 60_000);
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

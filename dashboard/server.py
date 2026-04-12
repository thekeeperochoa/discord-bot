"""
Dashboard API server for Discord AI Bot (Groq edition)
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import os
import logging
from pathlib import Path

log = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT        = Path(__file__).parent.parent
CONFIG_FILE = ROOT / "config" / "personality.json"
MEMORY_DIR  = ROOT / "memory"
STATIC_DIR  = Path(__file__).parent / "static"

GROQ_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

DEFAULT_PERSONALITY = {
    "name": "Aria",
    "model": "llama3-8b-8192",
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
}

app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULT_PERSONALITY, **json.load(f)}
    return DEFAULT_PERSONALITY.copy()


def save_config(data: dict):
    CONFIG_FILE.parent.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(load_config())


@app.route("/api/config", methods=["POST"])
def set_config():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    cfg = load_config()
    cfg.update(data)
    save_config(cfg)
    return jsonify({"ok": True, "config": cfg})


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({"models": GROQ_MODELS, "connected": True})


@app.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    channels = []
    if MEMORY_DIR.exists():
        for f in MEMORY_DIR.glob("*.json"):
            with open(f) as fp:
                msgs = json.load(fp)
            channels.append({
                "channel_id": f.stem,
                "message_count": len(msgs),
                "last_updated": f.stat().st_mtime,
            })
    channels.sort(key=lambda x: x["last_updated"], reverse=True)
    return jsonify({"channels": channels})


@app.route("/api/memory/clear", methods=["POST"])
def clear_memory():
    channel_id = request.json.get("channel_id")
    if channel_id:
        p = MEMORY_DIR / f"{channel_id}.json"
        if p.exists():
            p.unlink()
        return jsonify({"ok": True, "cleared": channel_id})
    cleared = 0
    for f in MEMORY_DIR.glob("*.json"):
        f.unlink()
        cleared += 1
    return jsonify({"ok": True, "cleared": cleared})


@app.route("/api/memory/<channel_id>", methods=["GET"])
def get_channel_memory(channel_id):
    p = MEMORY_DIR / f"{channel_id}.json"
    if not p.exists():
        return jsonify({"messages": []})
    with open(p) as f:
        return jsonify({"messages": json.load(f)})


@app.route("/api/logs", methods=["GET"])
def get_logs():
    log_file = ROOT / "logs" / "bot.log"
    if not log_file.exists():
        return jsonify({"lines": []})
    lines = log_file.read_text().splitlines()
    return jsonify({"lines": lines[-100:]})


@app.route("/api/reset", methods=["POST"])
def reset_config():
    save_config(DEFAULT_PERSONALITY)
    return jsonify({"ok": True, "config": DEFAULT_PERSONALITY})


@app.route("/api/status", methods=["GET"])
def status():
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    has_discord = bool(os.environ.get("DISCORD_TOKEN"))
    return jsonify({
        "groq_key_set": has_groq,
        "discord_token_set": has_discord,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"╔══════════════════════════════════════╗")
    print(f"║  Discord AI Bot Dashboard            ║")
    print(f"║  http://localhost:{port}               ║")
    print(f"╚══════════════════════════════════════╝")
    app.run(host="0.0.0.0", port=port, debug=False)

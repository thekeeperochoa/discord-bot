# 🤖 Discord AI Bot (Groq + Railway Edition)

Free, 24/7 Discord AI bot powered by **Groq** (free cloud AI) hosted on **Railway** (free hosting).
No GPU, no credit card for AI, no local machine needed.

---

## 🚀 Deploy in 7 Steps

### 1. Get a free Groq API key
- Go to https://console.groq.com
- Sign up (free, no credit card)
- API Keys → Create API Key → copy it

### 2. Create a Discord Bot
- https://discord.com/developers/applications
- New Application → Bot → Add Bot
- Enable **Message Content Intent**
- Reset Token → copy it

### 3. Push this folder to GitHub
```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 4. Deploy to Railway
- Go to https://railway.app
- New Project → Deploy from GitHub Repo
- Select your repository

### 5. Add environment variables in Railway
In your project → **Variables** tab:
```
DISCORD_TOKEN = your-discord-bot-token-here
GROQ_API_KEY  = your-groq-api-key-here
```

### 6. Set start command
Railway → Settings → Deploy → Start Command:
```
python bot/bot.py
```

### 7. Invite your bot
Discord Developer Portal → OAuth2 → URL Generator
- Scope: `bot`
- Permissions: Send Messages, Read Message History, Read Messages/View Channels

---

## 🎛️ Dashboard (local use)

To run the personality dashboard locally:
```bash
pip install -r requirements.txt
python dashboard/server.py
```
Then open http://localhost:5001

To use the dashboard on Railway, add a second service pointing to:
```
python dashboard/server.py
```

---

## 💬 Discord Commands

| Command | Description |
|---|---|
| `@BotName message` | Chat with the AI |
| Reply to bot | Continues conversation |
| `!clearhistory` | Clear this channel's memory |
| `!botinfo` | Show bot config |

---

## 🧠 Free Groq Models

| Model | Speed | Quality |
|---|---|---|
| llama3-8b-8192 | ⚡ Fast | ⭐⭐⭐ |
| llama3-70b-8192 | 🐢 Slower | ⭐⭐⭐⭐⭐ |
| llama-3.1-8b-instant | ⚡⚡ Fastest | ⭐⭐⭐ |
| llama-3.3-70b-versatile | 🐢 Slower | ⭐⭐⭐⭐⭐ |
| mixtral-8x7b-32768 | ⚡ Fast | ⭐⭐⭐⭐ |
| gemma2-9b-it | ⚡ Fast | ⭐⭐⭐ |

---

## 📁 Structure

```
discord-bot-groq/
├── bot/
│   └── bot.py              # Discord bot (uses Groq API)
├── dashboard/
│   ├── server.py           # Flask dashboard API
│   └── static/
│       └── index.html      # Web dashboard UI
├── config/
│   └── personality.json    # Bot personality settings
├── memory/                 # Per-channel conversation history
├── logs/                   # Bot logs
├── requirements.txt
├── Procfile                # Railway process config
├── railway.json            # Railway deployment config
└── README.md
```

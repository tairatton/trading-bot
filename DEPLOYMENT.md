# 🚀 Trading Bot Deployment Guide (Windows VPS)

## 1. Prerequisites (สิ่งที่ต้องมี)
Log in to your VPS and install these programs:

1.  **MetaTrader 5 (MT5)**
    *   Download from your broker (Exness, etc.)
    *   Install and **Log in** to your trading account.
    *   **IMPORTANT:** Go to `Tools` > `Options` > `Expert Advisors`:
        *   ✅ Allow automated trading
        *   ✅ Allow WebRequest for listed URL: `http://localhost:8000` (optional but good practice)

2.  **Python 3.11+**
    *   Download: [https://www.python.org/downloads/](https://www.python.org/downloads/)
    *   **IMPORTANT:** During installation, check the box **"Add Python to PATH"** (Very important!)

3.  **Git (Optional but recommended)**
    *   Download: [https://git-scm.com/download/win](https://git-scm.com/download/win)
    *   Install with default settings.

---

## 2. Installation (ติดตั้งบอท)

Open **PowerShell** or **Command Prompt** on VPS and run:

```powershell
# 1. Clone the repository (Use your own URL)
git clone https://github.com/tairatton/trading-bot.git
cd trading-bot

# 2. Create Virtual Environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt
```

---

## 3. Configuration (ตั้งค่าสิ่งสำคัญ)

You need to create a `.env` file because it was not uploaded to GitHub (for security).

1.  Create a new file named `.env` in the `trading_bot` folder.
2.  Paste your settings (Change `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` to match your account):

```env
# MT5 Credentials
MT5_LOGIN=YOUR_LOGIN_ID
MT5_PASSWORD=YOUR_PASSWORD
MT5_SERVER=Exness-MT5Real
MT5_PATH="C:\Program Files\MetaTrader 5\terminal64.exe"

# Trading Config
SYMBOL=EURUSDm
TIMEFRAME=M30
RISK_PERCENT=1.5
MAX_SPREAD_PIPS=3.0

# 24H Trading Session
SESSION_START_UTC=0
SESSION_END_UTC=24

# Web Dashboard
WEB_HOST=0.0.0.0
WEB_PORT=80
SECRET_KEY=mysecretkey123
```

---

## 4. Run the Bot (เริ่มรัน)

In the same PowerShell window (make sure `(venv)` is shown):

```powershell
python main.py
```

✅ **Done!** Access the dashboard at `http://localhost` (or your VPS IP address).

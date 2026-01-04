"""
Centralized Credentials Configuration
"""

# Common Settings
MT5_PATH = r"C:\Users\TAIRAT.N\AppData\Roaming\MetaTrader 5\terminal64.exe"
TELEGRAM_BOT_TOKEN = "7638375632:AAHMhOaB1EvYr5A2YbJqjg7pO0QMnYj1rdY"
TELEGRAM_CHAT_ID = "-4636616968"

# Multi-Account List (Main Config)
ACCOUNTS_LIST = [
    {
        "name": "Non", 
        "login": 413124491, 
        "password": "Non0618944274.", 
        "server": "Exness-MT5Trial6"
    }
]

# Compatibility Mapping (Auto-generated from first account)
# DO NOT EDIT BELOW THIS LINE
if ACCOUNTS_LIST:
    MT5_LOGIN = ACCOUNTS_LIST[0]["login"]
    MT5_PASSWORD = ACCOUNTS_LIST[0]["password"]
    MT5_SERVER = ACCOUNTS_LIST[0]["server"]
else:
    MT5_LOGIN = 0
    MT5_PASSWORD = ""
    MT5_SERVER = ""

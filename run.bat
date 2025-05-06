@echo off
REM Launch Git Bash and run the bot
set REPO_DIR=C:\Users\kylet\OneDrive\Trading Bots\GitHub\ml-trade-bot
cd "%REPO_DIR%"
REM start Git Bash here and run python
"C:\Program Files\Git\git-bash.exe" -c "python trade_bot.py; exec bash"

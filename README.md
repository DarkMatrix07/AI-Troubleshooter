# AI Troubleshooter (Hackathon MVP)

AI Troubleshooter is a local Flask web app that matches a user's Windows issue to a curated database of fixes, optionally augments them with Gemini suggestions, and lets the user execute selected commands with confirmation. It includes system status insights, activity logging, and a robust fallback matcher so the demo keeps working even if Gemini fails.

## Features
- Issue analysis: Gemini-based selection of the best matching problem statement.
- Extra fixes: Gemini can suggest additional cmd/PowerShell commands.
- Safe execution flow: user confirmation required; admin required for privileged actions.
- Fallback matcher: local lightweight matcher when Gemini is unavailable.
- System status: RAM, disk, battery, network, Wi-Fi, top processes.
- Logs: console logs + UI activity log + execution history file.

## Requirements
- Python 3.10+
- Dependencies:
  - `flask`
  - `numpy`
  - `scikit-learn`
  - `google-generativeai`
  - `python-dotenv`
  - (Optional) `psutil`

## Quick Start
```powershell
cd "D:\GDG On Campus Hackathon\ai_troubleshooter"
pip install flask numpy scikit-learn google-generativeai python-dotenv psutil
python app.py
```
Open `http://0.0.0.0:5050` in your browser.

## Configuration (.env)
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
AI_TS_DEBUG_GEMINI=1
```

Optional settings:
```
AI_TS_ALLOW_TFIDF_FALLBACK=1
GEMINI_TIMEOUT_SECONDS=20
```

## How It Works
1) User submits a problem description.
2) Gemini picks the best matching problem statement from `issues.json`.
3) Gemini suggests extra commands (or returns "I don't know").
4) The app merges Gemini suggestions with database solutions and returns them.
5) User confirms a fix to execute it; output is logged and displayed.

## Project Structure
- `app.py` - Flask backend, Gemini integration, execution, logging.
- `templates/enhanced_index.html` - UI.
- `issues.json` - Problem statements and fixes.
- `execution_log.json` - Command execution history.
- `DOCUMENTATION.md` - Additional project documentation.

## Safety Notes
- Commands run locally via `subprocess` and require admin for privileged actions.
- This is an MVP; add allowlists or approval workflows before production use.

## Troubleshooting
- If Gemini fails, the app falls back to the local matcher and logs a warning.
- If you see timeouts, increase `GEMINI_TIMEOUT_SECONDS`.
- If system info is limited, install `psutil`.

## Demo Tips
- Use quick examples in the UI for instant matches.
- Show the Activity Log panel to highlight AI decisions and execution output.


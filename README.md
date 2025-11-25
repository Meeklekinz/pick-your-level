# Pick Your Level (MVP)

**Pick. Score. Level up.**
A tiny local web app that “gamifies” fingerpicking practice.
You pick a Level, upload a short phone clip (audio or video), and the app scores it:
- onset detection (did you play enough plucks?)
- pitch target matching (wrong-string swaps)
- muted-note heuristic
- timing stability

**Privacy**: Uploads are stored only in a temporary directory during analysis and deleted immediately after scoring.

## What this version (Lane Lock) expects you to play
Count: **1 & 2 & 3 & 4 &**

- Beats (1,2,3,4): **low E open** (E2)
- & of 1 and 3: **B string open** (B3)
- & of 2 and 4: **high E open** (E4)

## Requirements
- Python 3.10+
- **ffmpeg** installed and available on your PATH

### Install ffmpeg
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Fedora/RHEL/CentOS: `sudo dnf install ffmpeg`

## Setup (VS Code friendly)
1) Unzip this project.
2) Open the folder in VS Code.
3) Create a venv + install deps:

```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows powershell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run locally
```bash
uvicorn main:app --reload
```

Open:
- http://127.0.0.1:8000

## Upload directly from your phone (optional)
Run:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then visit:
- `http://<your-laptop-LAN-IP>:8000`

**Tip**: only do this on a trusted network and stop the server when you’re done.

## Next improvements you can add
- More Levels (patterns, chord changes)
- A metronome / click track for better timing scoring
- Better mute detection (spectral decay curves)
- User profiles + XP saved in a local SQLite DB

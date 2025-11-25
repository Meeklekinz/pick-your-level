# Pick Your Level

Practice helper with built‑in metronome, step guide, and TAB follow‑along. Pick a level, use the reference player, record a short clip (audio or video), and get quick feedback.

**Privacy:** Uploads live only in a temp directory during analysis and are deleted right after.

## Current Level (Lane Lock)
- Count: **1 & 2 & 3 & 4 &** (8 plucks per loop, two bars of 8th notes)
- Targets: low E open on 1/2/3/4, B open on & of 1/3, high e open on & of 2/4
- Views: **8-Step Pattern** tiles or **2-Bar TAB** with moving highlight
- Play Reference: toggle button runs a continuous metronome + guide tones; TAB highlight tracks 16 ticks (two loops) when TAB view is active

## How scoring works (lenient)
- Needs at least **one clean loop** (no misses) to pass; pitch/string choice and muting are *not* graded.
- Timing CV is reported for coaching, but timing wobble does **not** fail you.
- KPIs show detected plucks, clean loops, timing CV, and misses; pitch accuracy shows “Not scored.”

## Requirements
- Python 3.10+
- `ffmpeg` on your PATH
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - Fedora/RHEL/CentOS: `sudo dnf install ffmpeg`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run locally
```bash
uvicorn main:app --reload
```
Open http://127.0.0.1:8000

To upload from your phone on LAN:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Then visit `http://<your-laptop-LAN-IP>:8000` (trusted network only).

## Using the app
1) Pick a level (Lane Lock). Beginner View shows steps/TAB and tips; turning it off shows a compact summary.
2) Use **Play Reference** to hear/metronome the pattern; TAB view highlights the active subdivision on the right string.
3) Set tempo with the slider.
4) Record and upload a ~8–15s clip (one or two loops is fine).
5) Read the scoreboard and coaching note; timing advice appears if wobble is high.

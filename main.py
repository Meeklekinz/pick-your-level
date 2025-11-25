from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

APP_TITLE = "Pick Your Level"
app = FastAPI(title=APP_TITLE)

# ===========================
# Levels (data-driven)
# ===========================

@dataclass(frozen=True)
class LevelUI:
    title: str
    count_text: str
    bullets: List[str]
    default_bpm: int
    steps: List[Dict[str, Any]]   # count/finger/target/label/display/hz/kind
    tab_text: str                 # 2-bar practice view (monospace)
    legend: str
    recording_tip: str


@dataclass(frozen=True)
class LevelSpec:
    id: str
    name: str
    description: str

    # analyzer type (so you can add different scoring strategies later)
    analyzer: str

    # audio/scoring (for analyzer == "pitch_targets_v1")
    sr: int
    expected_hz: List[float]        # expected pitch targets per pluck (one "loop")
    cents_tolerance: float
    min_loops_to_pass: int
    max_timing_cv: float            # onset interval stability (lower is better)
    max_mute_rate: float            # heuristic
    min_pitch_acc: float            # accuracy among non-misses

    # UI
    ui: LevelUI


def _level_to_api(s: LevelSpec) -> Dict[str, Any]:
    return {
        "id": s.id,
        "name": s.name,
        "description": s.description,
        "analyzer": s.analyzer,
        "ui": asdict(s.ui),
    }


# Lane Lock drill: 8 events per loop (1 & 2 & 3 & 4 &)
E2 = 82.41
B3 = 246.94
E4 = 329.63

LANE_LOCK_STEPS: List[Dict[str, Any]] = [
    {"count": "1", "finger": "Pick",   "target": "Low E string (6th) open",  "label": "E2", "display": "E2 (low E)",    "hz": E2, "kind": "bass"},
    {"count": "&", "finger": "Middle", "target": "B string (2nd) open",      "label": "B3", "display": "B3 (B string)",  "hz": B3, "kind": "b"},
    {"count": "2", "finger": "Pick",   "target": "Low E string (6th) open",  "label": "E2", "display": "E2 (low E)",    "hz": E2, "kind": "bass"},
    {"count": "&", "finger": "Middle", "target": "High e string (1st) open", "label": "E4", "display": "E4 (high e)",    "hz": E4, "kind": "e"},
    {"count": "3", "finger": "Pick",   "target": "Low E string (6th) open",  "label": "E2", "display": "E2 (low E)",    "hz": E2, "kind": "bass"},
    {"count": "&", "finger": "Middle", "target": "B string (2nd) open",      "label": "B3", "display": "B3 (B string)",  "hz": B3, "kind": "b"},
    {"count": "4", "finger": "Pick",   "target": "Low E string (6th) open",  "label": "E2", "display": "E2 (low E)",    "hz": E2, "kind": "bass"},
    {"count": "&", "finger": "Middle", "target": "High e string (1st) open", "label": "E4", "display": "E4 (high e)",    "hz": E4, "kind": "e"},
]

# NOTE: Header/ruler line removed per request.
LANE_LOCK_TAB = """Level 1 — Lane Lock (2 bars, 8th-notes)

e|--------------0---------------0---------------0---------------0---|
B|------0---------------0---------------0---------------0-----------|
G|------------------------------------------------------------------|
D|------------------------------------------------------------------|
A|------------------------------------------------------------------|
E|--0-------0-------0-------0-------0-------0-------0-------0-------|"""


LEVELS: Dict[str, LevelSpec] = {
    "lane_lock": LevelSpec(
        id="lane_lock",
        name="Level 1 — Lane Lock",
        description="Open strings drill: E2 on beats (1,2,3,4), B3 on & of 1 and 3, E4 on & of 2 and 4.",
        analyzer="pitch_targets_v1",
        sr=22050,
        expected_hz=[E2, B3, E2, E4, E2, B3, E2, E4],
        cents_tolerance=60.0,     # generous for phone mics/noise
        min_loops_to_pass=2,
        max_timing_cv=0.18,
        max_mute_rate=0.18,
        min_pitch_acc=0.82,
        ui=LevelUI(
            title="What you are practicing",
            count_text="Count: 1 & 2 & 3 & 4 & (8 plucks per loop)",
            bullets=[
                "Pick (P) plays low E open on 1,2,3,4",
                "Middle (M) plucks B open on & after 1 and 3",
                "Middle (M) plucks high e open on & after 2 and 4",
                "Plant → pluck: touch the target string first, then pluck",
                "No palm mute: keep your hand heel floating off the strings",
            ],
            default_bpm=70,
            steps=LANE_LOCK_STEPS,
            tab_text=LANE_LOCK_TAB,
            legend="P = pick low E, M = middle finger on B/high e",
            recording_tip="Phone 1–2 feet from bridge/soundhole, play a bit louder than normal, and count out loud.",
        ),
    ),
}

# ===========================
# Audio utilities
# ===========================

def cents_diff(f: float, target: float) -> float:
    if f <= 0 or target <= 0:
        return float("inf")
    return 1200.0 * math.log2(f / target)


def run_ffmpeg_to_wav(input_path: str, output_path: str, sr: int) -> None:
    """
    Convert almost anything (m4a/mov/mp4/wav/...) to a mono WAV at the requested sample rate.
    Requires ffmpeg installed and on PATH.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",               # strip video if present
        "-ac", "1",          # mono
        "-ar", str(sr),      # sample rate
        "-f", "wav",
        output_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed. Is it installed?\n\n{err[:1200]}")


def detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Onset detection tuned for plucked strings.
    Returns onset times in seconds.
    """
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        backtrack=False,
        pre_max=10,
        post_max=10,
        pre_avg=20,
        post_avg=20,
        delta=0.18,
        wait=2,
        units="frames",
    )
    times = librosa.frames_to_time(onset_frames, sr=sr)

    # Remove double-triggers that happen too close together
    if len(times) == 0:
        return times
    filtered = [times[0]]
    for t in times[1:]:
        if t - filtered[-1] >= 0.08:
            filtered.append(t)
    return np.array(filtered, dtype=float)


def estimate_pitch_at_onset(y: np.ndarray, sr: int, t: float) -> float:
    """
    Estimate f0 in a short window after an onset using librosa.yin.
    Returns 0.0 if no reliable pitch is found.
    """
    start = int(max(0, (t - 0.01) * sr))
    end = int(min(len(y), (t + 0.18) * sr))
    seg = y[start:end]
    if len(seg) < int(0.05 * sr):
        return 0.0

    f0 = librosa.yin(seg, fmin=60, fmax=420, sr=sr)
    f0 = f0[np.isfinite(f0)]
    if len(f0) == 0:
        return 0.0
    return float(np.median(f0))


def onset_energy_features(y: np.ndarray, sr: int, t: float) -> Tuple[float, float]:
    """
    Return:
      - attack_rms: RMS in [0ms..120ms) after onset
      - sustain_ratio: RMS in [120ms..260ms) / attack_rms
    sustain_ratio low => likely muted/choked.
    """
    def rms(seg: np.ndarray) -> float:
        return float(np.sqrt(np.mean(seg * seg))) if len(seg) else 0.0

    a0 = int(max(0, (t + 0.00) * sr))
    a1 = int(min(len(y), (t + 0.12) * sr))
    s0 = int(min(len(y), (t + 0.12) * sr))
    s1 = int(min(len(y), (t + 0.26) * sr))

    attack = rms(y[a0:a1])
    sustain = rms(y[s0:s1])
    sustain_ratio = (sustain / attack) if attack > 1e-9 else 0.0
    return attack, sustain_ratio


# ===========================
# Scoring (pitch_targets_v1)
# ===========================

def score_pitch_targets_v1(
    spec: LevelSpec,
    pattern_expected: List[float],
    pitches: List[float],
    attacks: List[float],
    sustain_ratios: List[float],
    onset_times: np.ndarray,
) -> Dict[str, Any]:
    expected = pattern_expected
    n = len(pitches)

    if n < len(expected):
        return {
            "passed": False,
            "reason": "Not enough detected plucks. Record closer to the guitar and play a bit louder.",
            "detected_onsets": n,
        }

    loops = n // len(expected)
    usable = loops * len(expected)

    # Robust "miss" threshold based on median onset attack
    med_attack = float(np.median([a for a in attacks[:usable] if np.isfinite(a)])) if usable > 0 else 0.0
    miss_thresh = med_attack * 0.20  # much lower than typical => likely miss or too quiet

    per_event: List[Dict[str, Any]] = []
    correct_pitch = 0
    muted = 0
    missed = 0

    for i in range(usable):
        target = expected[i % len(expected)]
        f = pitches[i]
        attack = attacks[i]
        sustain_ratio = sustain_ratios[i]

        is_miss = (attack < miss_thresh) or (f <= 0.0)
        if is_miss:
            missed += 1
            per_event.append({"i": i, "expected_hz": target, "f0_hz": f, "result": "miss"})
            continue

        cd = abs(cents_diff(f, target))
        ok_pitch = cd <= spec.cents_tolerance

        # Muted heuristic: sustain dies fast
        is_muted = sustain_ratio < 0.22
        if is_muted:
            muted += 1

        if ok_pitch:
            correct_pitch += 1
            res = "ok_muted" if is_muted else "ok"
        else:
            res = "wrong_muted" if is_muted else "wrong"

        per_event.append({
            "i": i,
            "expected_hz": target,
            "f0_hz": f,
            "cents_off": float(cd),
            "attack": float(attack),
            "sustain_ratio": float(sustain_ratio),
            "result": res,
        })

    pitch_acc = correct_pitch / max(1, (usable - missed))
    mute_rate = muted / max(1, usable)

    # Timing stability = coefficient of variation of inter-onset-intervals
    usable_times = onset_times[:usable]
    diffs = np.diff(usable_times)
    diffs = diffs[diffs > 1e-6]
    timing_cv = float(np.std(diffs) / np.mean(diffs)) if len(diffs) >= 4 else 999.0

    # Count "clean loops": no wrong/miss; allow <=1 muted per loop
    clean_loops = 0
    loop_len = len(expected)
    for L in range(loops):
        chunk = per_event[L * loop_len:(L + 1) * loop_len]
        if all(ev["result"] in ("ok", "ok_muted") for ev in chunk):
            mute_in_loop = sum(1 for ev in chunk if ev["result"] == "ok_muted")
            if mute_in_loop <= 1:
                clean_loops += 1

    passed = (
        clean_loops >= spec.min_loops_to_pass
        and pitch_acc >= spec.min_pitch_acc
        and mute_rate <= spec.max_mute_rate
        and timing_cv <= spec.max_timing_cv
    )

    reason = None
    if not passed:
        bits = []
        if clean_loops < spec.min_loops_to_pass:
            bits.append(f"Need {spec.min_loops_to_pass} clean loops (you got {clean_loops}).")
        if pitch_acc < spec.min_pitch_acc:
            bits.append(f"Pitch accuracy low ({pitch_acc:.0%}). Likely wrong-string hits or swaps.")
        if mute_rate > spec.max_mute_rate:
            bits.append(f"Muted-note rate high ({mute_rate:.0%}). Watch palm/hand contact.")
        if timing_cv > spec.max_timing_cv:
            bits.append(f"Timing unstable (CV {timing_cv:.2f}). Slow down and count out loud.")
        reason = " ".join(bits) if bits else "Try again with a cleaner recording."

    xp = 10 if passed else 2

    return {
        "passed": passed,
        "xp_awarded": xp,
        "loops_total": loops,
        "clean_loops": clean_loops,
        "pitch_accuracy": pitch_acc,
        "mute_rate": mute_rate,
        "timing_cv": timing_cv,
        "missed_hits": missed,
        "evaluated_events": usable,
        "events": per_event[:min(len(per_event), 64)],
        "reason": reason,
    }


def analyze(level_id: str, wav_path: str) -> Dict[str, Any]:
    spec = LEVELS.get(level_id)
    if not spec:
        return {"error": f"Unknown level: {level_id}"}

    y, sr = librosa.load(wav_path, sr=spec.sr, mono=True)
    if y is None or len(y) < sr * 1.0:
        return {"error": "Audio too short. Record ~10 seconds."}

    # Trim quiet parts & normalize
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < sr * 1.0:
        return {"error": "Too much silence detected. Record closer to the guitar."}
    y = y / (np.max(np.abs(y)) + 1e-9)

    onset_times = detect_onsets(y, sr)
    pitches: List[float] = []
    attacks: List[float] = []
    sustains: List[float] = []

    for t in onset_times:
        pitches.append(estimate_pitch_at_onset(y, sr, float(t)))
        a, s = onset_energy_features(y, sr, float(t))
        attacks.append(a)
        sustains.append(s)

    if spec.analyzer == "pitch_targets_v1":
        scored = score_pitch_targets_v1(spec, spec.expected_hz, pitches, attacks, sustains, onset_times)
        scored["level"] = {"id": spec.id, "name": spec.name, "description": spec.description}
        scored["detected_onsets"] = int(len(onset_times))
        return scored

    return {"error": f"Analyzer not implemented: {spec.analyzer}"}


# ===========================
# Web UI (single-file)
# ===========================

INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Pick Your Level</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system; background:#0b1220; color:#e5e7eb; margin:0; }
    .wrap { max-width: 1060px; margin: 0 auto; padding: 24px; }
    .card { background: #111a2e; border: 1px solid #243055; border-radius: 18px; padding: 16px; }
    .grid { display:grid; grid-template-columns: 1fr; gap: 16px; }
    @media (min-width: 920px) { .grid { grid-template-columns: 1.2fr 0.8fr; } }

    .btn { background:#3b82f6; border:none; color:white; padding:10px 14px; border-radius:12px; cursor:pointer; font-weight:900; }
    .btn:disabled { opacity: 0.5; cursor:not-allowed; }
    .btn2 { background:#22c55e; }
    .btn3 { background:#a855f7; }

    .segRow { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }
    .segBtn {
      background:#0b1220; color:#e5e7eb; border:1px solid #243055;
      padding:10px 14px; border-radius:12px; cursor:pointer; font-weight:900;
    }
    .segBtn.on { outline: 2px solid #3b82f6; box-shadow: 0 0 0 4px rgba(59,130,246,0.15); }

    select, input[type=file], input[type=range] {
      width:100%; padding:10px; border-radius:12px; border:1px solid #243055;
      background:#0b1220; color:#e5e7eb;
    }
    .small { opacity: 0.92; font-size: 0.95rem; line-height: 1.35; }
    .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .toggle { display:flex; gap:10px; align-items:center; }

    .kpi { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }
    .pill { background:#0b1220; border:1px solid #243055; border-radius:999px; padding:8px 12px; }
    .ok { color:#34d399; font-weight:900; }
    .bad { color:#fb7185; font-weight:900; }

    .coach { margin-top: 12px; border:1px solid #243055; background:#0b1220; border-radius:14px; padding:12px; }

    .steps { display:grid; grid-template-columns: repeat(8, 1fr); gap:10px; margin-top:10px; }
    .step {
      border:1px solid #243055; background:#0b1220; border-radius:14px; padding:10px;
      min-height: 96px; display:flex; flex-direction:column; gap:6px; justify-content:space-between;
    }
    .step .count { font-weight:900; font-size: 1.05rem; }
    .step .finger { font-size: 0.9rem; opacity:0.95; }
    .step .target { font-size: 0.92rem; }
    .step.active { outline: 2px solid #3b82f6; box-shadow: 0 0 0 4px rgba(59,130,246,0.15); }

    pre {
      white-space: pre; overflow:auto;
      background:#0b1220; padding:12px; border-radius:12px; border:1px solid #243055;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 0.92rem;
    }

    .events { margin-top: 10px; border: 1px solid #243055; border-radius: 14px; overflow:hidden; }
    .eventsHeader, .eventsRow {
      display:grid; grid-template-columns: 60px 1fr 1fr 100px; gap: 10px;
      padding: 10px 12px; align-items:center;
    }
    .eventsHeader { background:#0b1220; font-weight:900; border-bottom:1px solid #243055; }
    .eventsRow { background:#0f1730; border-bottom:1px solid #243055; }
    .eventsRow:last-child { border-bottom:none; }
    .tag { font-family: ui-monospace, SFMono-Regular; font-size: 0.9rem; opacity: 0.95; }
    .tabHi { background:#fbbf24; color:#0b1220; font-weight:900; border-radius:4px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 style="margin:0 0 6px 0;">Pick Your Level</h1>
    <p class="small" style="margin:0 0 6px 0;"><strong>Pick. Score. Level up.</strong></p>
    <p class="small" style="margin:0 0 14px 0;">
      Upload a short clip (audio or video). It gets analyzed, and the file is deleted after scoring.
    </p>

    <div class="grid">
      <div class="card">
        <div class="row" style="justify-content:space-between;">
          <h2 style="margin:0;">Choose a Level</h2>
          <label class="toggle small">
            <input id="beginner" type="checkbox" checked />
            Beginner View
          </label>
        </div>

        <select id="level" style="margin-top:10px;">
          <option value="">Loading…</option>
        </select>

        <div id="beginnerPanel" class="coach">
          <div class="small">
            <div style="font-weight:900; margin-bottom:6px;" id="uiTitle">What you are practicing</div>
            <div id="uiCount">—</div>
            <ul id="uiBullets" style="margin:8px 0 0 18px;"></ul>
          </div>

          <!-- VIEW TOGGLE -->
          <div class="segRow">
            <button id="viewSteps" class="segBtn on" type="button">8-Step Pattern</button>
            <button id="viewTab" class="segBtn" type="button">2-Bar Practice TAB</button>
          </div>

          <!-- STEPS VIEW -->
          <div id="stepsWrap">
            <div class="small" style="margin-top:12px; font-weight:900;">The 8-step pattern</div>
            <div class="steps" id="steps"></div>
          </div>

          <!-- TAB VIEW -->
          <div id="tabWrap" style="display:none;">
            <div class="small" style="margin-top:12px; font-weight:900;">2-Bar practice view</div>
            <pre id="tabPre">—</pre>
            <div class="small" style="margin-top:8px;">
              Legend: <span id="uiLegend"></span>
            </div>
          </div>

          <!-- REFERENCE PLAYER (shared for BOTH views) -->
          <div class="row" id="refWrap" style="margin-top:12px;">
            <button id="playRef" class="btn btn3" type="button">Play Reference (metronome + guide)</button>
            <div style="flex:1; min-width:220px;">
              <div class="small" style="margin-bottom:6px;"><b>Tempo</b>: <span id="bpmVal">70</span> BPM</div>
              <input id="bpm" type="range" min="40" max="120" value="70" />
            </div>
          </div>
          <div class="small" id="nowPlay" style="margin-top:8px;">Now: —</div>

          <div class="small" style="margin-top:10px;">
            <b>Recording tip:</b> <span id="uiRecordingTip">—</span>
          </div>
        </div>

        <div id="summaryPanel" class="coach small" style="display:none;">
          <div style="font-weight:900; margin-bottom:6px;" id="summaryTitle">Level overview</div>
          <div><b>Count:</b> <span id="summaryCount">—</span></div>
          <div><b>Legend:</b> <span id="summaryLegend">—</span></div>
          <div><b>Recording tip:</b> <span id="summaryRecordingTip">—</span></div>
          <div style="margin-top:8px; opacity:0.9;">Toggle Beginner View back on for guided steps, TAB highlight, and the reference player.</div>
        </div>
        
        <h3 style="margin:16px 0 8px 0;">Upload a clip</h3>
        <input id="file" type="file" accept="audio/*,video/*"/>

        <div class="row" style="margin-top:12px;">
          <button id="analyze" class="btn" disabled>Analyze</button>
          <button id="tipsBtn" class="btn btn2" type="button">Quick Tips</button>
        </div>

        <div id="tips" class="small" style="display:none; margin-top:12px;">
          <ul style="margin:0 0 0 18px;">
            <li>Record 8–15 seconds (multiple loops).</li>
            <li>One pluck per count (“1 & 2 & 3 & 4 &”).</li>
            <li>If “not enough detected plucks”: move closer + play louder.</li>
          </ul>
        </div>
      </div>

      <div class="card">
        <h2 style="margin:0 0 10px 0;">Scoreboard</h2>
        <div id="status" class="small">Upload a clip to begin.</div>

        <div class="coach" id="coachBox" style="display:none;"></div>
        <div class="kpi" id="kpis"></div>

        <div id="eventsWrap" style="display:none;">
          <h3 style="margin:14px 0 8px 0;">Event feedback (first ~24 plucks)</h3>
          <div class="events">
            <div class="eventsHeader">
              <div>#</div><div>Expected</div><div>Detected</div><div>Result</div>
            </div>
            <div id="events"></div>
          </div>
        </div>

        <h3 style="margin:14px 0 6px 0;">Raw details (debug)</h3>
        <pre id="details">—</pre>
      </div>
    </div>
  </div>

<script>
  // ===================
  // Levels (loaded from backend)
  // ===================
  let levelsById = {};
  let currentLevelId = null;
  let currentSteps = [];
  let pitchTargets = []; // [{hz, name}]
  let currentView = "steps";
  let rawTabText = "";

  function buildPitchTargets() {
    const map = new Map();
    for (const s of currentSteps) {
      if (s && s.hz && (s.display || s.label)) {
        map.set(String(s.hz), { hz: Number(s.hz), name: (s.display || s.label) });
      }
    }
    pitchTargets = Array.from(map.values());
  }

  function renderBullets(items) {
    const ul = document.getElementById("uiBullets");
    ul.innerHTML = (items || []).map(t => `<li>${t}</li>`).join("");
  }

  function applyLevel(levelId) {
    const lvl = levelsById[levelId];
    if (!lvl) return;

    currentLevelId = levelId;

    document.getElementById("uiTitle").textContent = `What you are practicing (${lvl.name})`;
    document.getElementById("uiCount").textContent = (lvl.ui.count_text || "");
    renderBullets(lvl.ui.bullets || []);
    document.getElementById("uiLegend").textContent = (lvl.ui.legend || "");
    document.getElementById("uiRecordingTip").textContent = (lvl.ui.recording_tip || "");
    summaryTitle.textContent = lvl.name || "Level overview";
    summaryCount.textContent = (lvl.ui.count_text || "");
    summaryLegend.textContent = (lvl.ui.legend || "");
    summaryRecordingTip.textContent = (lvl.ui.recording_tip || "");

    currentSteps = lvl.ui.steps || [];
    buildPitchTargets();

    rawTabText = String(lvl.ui.tab_text || "—");
    tabPre.textContent = rawTabText;

    const bpm = (lvl.ui.default_bpm || 70);
    bpmEl.value = String(bpm);
    bpmValEl.textContent = String(bpm);

    renderSteps(-1);
  }

  async function loadLevels() {
    const res = await fetch("/api/levels");
    const levels = await res.json();

    levelsById = Object.fromEntries(levels.map(l => [l.id, l]));

    const sel = document.getElementById("level");
    sel.innerHTML = levels.map(l => `<option value="${l.id}">${l.name}</option>`).join("");
    sel.addEventListener("change", (e) => applyLevel(e.target.value));

    if (levels.length) {
      sel.value = levels[0].id;
      applyLevel(levels[0].id);
    }
  }

  // ===================
  // Beginner view controls
  // ===================
  const fileEl = document.getElementById('file');
  const analyzeBtn = document.getElementById('analyze');
  const statusEl = document.getElementById('status');
  const detailsEl = document.getElementById('details');
  const kpisEl = document.getElementById('kpis');
  const tipsBtn = document.getElementById('tipsBtn');
  const tipsEl = document.getElementById('tips');
  const beginnerToggle = document.getElementById('beginner');
  const beginnerPanel = document.getElementById('beginnerPanel');
  const summaryPanel = document.getElementById('summaryPanel');
  const summaryTitle = document.getElementById('summaryTitle');
  const summaryCount = document.getElementById('summaryCount');
  const summaryLegend = document.getElementById('summaryLegend');
  const summaryRecordingTip = document.getElementById('summaryRecordingTip');

  const stepsEl = document.getElementById('steps');
  const playRefBtn = document.getElementById('playRef');
  const bpmEl = document.getElementById('bpm');
  const bpmValEl = document.getElementById('bpmVal');
  const nowPlayEl = document.getElementById('nowPlay');

  const coachBox = document.getElementById('coachBox');
  const eventsWrap = document.getElementById('eventsWrap');
  const eventsEl = document.getElementById('events');

  const viewStepsBtn = document.getElementById('viewSteps');
  const viewTabBtn = document.getElementById('viewTab');
  const stepsWrap = document.getElementById('stepsWrap');
  const tabWrap = document.getElementById('tabWrap');
  const tabPre = document.getElementById('tabPre');

  bpmValEl.textContent = bpmEl.value;
  bpmEl.addEventListener('input', () => bpmValEl.textContent = bpmEl.value);

  tipsBtn.onclick = () => { tipsEl.style.display = tipsEl.style.display === 'none' ? 'block' : 'none'; };
  beginnerToggle.addEventListener('change', () => {
    const showBeginner = beginnerToggle.checked;
    beginnerPanel.style.display = showBeginner ? 'block' : 'none';
    summaryPanel.style.display = showBeginner ? 'none' : 'block';
  });

  fileEl.addEventListener('change', () => {
    analyzeBtn.disabled = !(fileEl.files && fileEl.files.length === 1);
  });

  function setView(mode) {
    currentView = mode;
    const stepsOn = (mode === "steps");
    stepsWrap.style.display = stepsOn ? "block" : "none";
    tabWrap.style.display = stepsOn ? "none" : "block";
    viewStepsBtn.classList.toggle("on", stepsOn);
    viewTabBtn.classList.toggle("on", !stepsOn);
  }
  viewStepsBtn.onclick = () => setView("steps");
  viewTabBtn.onclick = () => setView("tab");

  function pill(label, value, ok=null) {
    const div = document.createElement('div');
    div.className = 'pill';
    let cls = '';
    if (ok === true) cls='ok';
    if (ok === false) cls='bad';
    div.innerHTML = `<strong>${label}:</strong> <span class="${cls}">${value}</span>`;
    return div;
  }

  function renderSteps(activeIdx = -1) {
    stepsEl.innerHTML = '';
    currentSteps.forEach((s, i) => {
      const div = document.createElement('div');
      div.className = 'step' + (i === activeIdx ? ' active' : '');
      div.innerHTML = `
        <div class="count">${s.count}</div>
        <div class="target">${s.target}</div>
        <div class="finger"><b>${s.finger}</b> • ${s.label}</div>
      `;
      stepsEl.appendChild(div);
    });
  }

  function setNow(text) {
    nowPlayEl.textContent = `Now: ${text}`;
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (ch) => {
      if (ch === '&') return '&amp;';
      if (ch === '<') return '&lt;';
      if (ch === '>') return '&gt;';
      if (ch === '"') return '&quot;';
      return '&#39;';
    });
  }

  function renderTabHighlight(tick, stepKind = null) {
    const kindToLinePrefix = { bass: "E|", b: "B|", e: "e|" };
    const targetPrefix = kindToLinePrefix[stepKind] || null;

    if (!Number.isFinite(tick) || tick < 0 || !rawTabText || !targetPrefix) {
      tabPre.textContent = rawTabText;
      return;
    }

    const beat = Math.floor(tick / 2);
    const within = (tick % 2 === 0) ? 0 : 4;
    const interiorOffset = 2 + (beat * 8) + within;
    const absoluteIndexInLine = 2 + interiorOffset;

    const prefixes = new Set(["e|", "B|", "G|", "D|", "A|", "E|"]);
    const highlighted = rawTabText.split("\n").map((line) => {
      const prefix = line.slice(0, 2);
      if (!prefixes.has(prefix) || prefix !== targetPrefix) {
        return escapeHtml(line);
      }
      if (absoluteIndexInLine < 0 || absoluteIndexInLine >= line.length) {
        return escapeHtml(line);
      }
      const before = escapeHtml(line.slice(0, absoluteIndexInLine));
      const hi = escapeHtml(line.charAt(absoluteIndexInLine));
      const after = escapeHtml(line.slice(absoluteIndexInLine + 1));
      return `${before}<span class="tabHi">${hi}</span>${after}`;
    });

    tabPre.innerHTML = highlighted.join("\n");
  }

  // ===================
  // Reference player (shared for BOTH views)
  // ===================
  let audioCtx = null;
  let refPlaying = false;
  let refTimer = null;
  let refTick = 0;
  
  function playTone(freq, time, dur=0.07) {
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.type = 'triangle';
    o.frequency.value = freq;
    g.gain.setValueAtTime(0.0001, time);
    g.gain.exponentialRampToValueAtTime(0.2, time + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, time + dur);
    o.connect(g).connect(audioCtx.destination);
    o.start(time);
    o.stop(time + dur + 0.02);
  }

  function guideFreqFor(kind) {
    if (kind === 'bass') return 220;
    if (kind === 'b') return 523.25;
    return 659.25;
  }

  function stopReference() {
    refPlaying = false;
    if (refTimer) {
      clearTimeout(refTimer);
      refTimer = null;
    }
    renderSteps(-1);
    renderTabHighlight(-1);
    setNow("—");
    playRefBtn.textContent = "Play Reference (metronome + guide)";
  }

  function scheduleTick() {
    if (!refPlaying || !currentSteps.length) return;

    const bpm = parseInt(bpmEl.value, 10) || 60;
    const stepDur = (60 / bpm) / 2; // 8th notes
    const showSteps = (currentView === "steps");
    const showTab = (currentView === "tab");

    const tick = refTick;
    const step = currentSteps[tick % currentSteps.length];
    const when = audioCtx.currentTime + 0.04;

    playTone(guideFreqFor(step.kind), when);

    const uiDelay = Math.max(0, (when - audioCtx.currentTime) * 1000);
    setTimeout(() => {
      if (!refPlaying) return;
      const nowText = `${step.count} — ${step.finger} on ${step.target}`;
      setNow(nowText);

      if (showSteps) {
        renderSteps(tick % currentSteps.length);
      } else if (showTab) {
        renderTabHighlight(tick % 16, step.kind);
      }
    }, uiDelay);

    refTick += 1;
    const delayMs = Math.max(20, stepDur * 1000);
    refTimer = setTimeout(scheduleTick, delayMs);
  }

  playRefBtn.onclick = async () => {
    if (!currentSteps.length) return;
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    if (refPlaying) {
      stopReference();
      return;
    }

    refPlaying = true;
    refTick = 0;
    setNow("starting…");
    playRefBtn.textContent = "Stop Reference";
    scheduleTick();
  };

  // ===================
  // Helpers for scoring display
  // ===================
  function nearestLabel(hz) {
    if (!hz || hz <= 0) return "—";
    const cents = (f, t) => 1200 * Math.log2(f / t);

    let best = null, bestAbs = Infinity;
    for (const t of pitchTargets) {
      const d = Math.abs(cents(hz, t.hz));
      if (d < bestAbs) { bestAbs = d; best = t; }
    }
    if (!best || bestAbs > 250) return `${hz.toFixed(1)} Hz (unknown)`;
    return best.name;
  }

  function coachMessage(data) {
    const miss = data.missed_hits || 0;
    const muteRate = data.mute_rate || 0;
    const pitchAcc = data.pitch_accuracy || 0;
    const timingCV = data.timing_cv || 999;

    if (miss >= 3) return { title: "Main issue: missed strings", text: "Plant → pluck: touch the target (B/e) before plucking. Also record closer + play a bit louder." };
    if (muteRate > 0.22) return { title: "Main issue: muted notes", text: "Your palm/hand is touching strings after pluck. Float the heel of your hand; don’t anchor on the bridge." };
    if (pitchAcc < 0.82) return { title: "Main issue: wrong-string swaps", text: "Slow down. Keep pick assigned to low E only; middle assigned to B/high e only. Plant on target string." };
    if (timingCV > 0.18) return { title: "Main issue: timing wobble", text: "Drop tempo and count out loud. Use Play Reference at 60–70 BPM, then match it." };
    return { title: "Nice work", text: "Keep going—aim for more clean loops in a row." };
  }

  // ===================
  // Analyze upload
  // ===================
  analyzeBtn.onclick = async () => {
    const level = document.getElementById('level').value;
    const f = fileEl.files[0];
    const form = new FormData();
    form.append('file', f);

    statusEl.textContent = 'Analyzing…';
    detailsEl.textContent = '…';
    kpisEl.innerHTML = '';
    coachBox.style.display = 'none';
    eventsWrap.style.display = 'none';
    eventsEl.innerHTML = '';
    analyzeBtn.disabled = true;

    try {
      const res = await fetch(`/analyze?level=${encodeURIComponent(level)}`, { method: 'POST', body: form });
      const data = await res.json();

      if (data.error) {
        statusEl.innerHTML = `<span class="bad">Error:</span> ${data.error}`;
        detailsEl.textContent = JSON.stringify(data, null, 2);
        return;
      }

      const passed = !!data.passed;
      statusEl.innerHTML = passed
        ? `<span class="ok">PASS</span> — XP +${data.xp_awarded}`
        : `<span class="bad">NOT YET</span> — XP +${data.xp_awarded}<br/><span class="small">${data.reason || ''}</span>`;

      const msg = coachMessage(data);
      coachBox.style.display = 'block';
      coachBox.innerHTML = `<div style="font-weight:900; margin-bottom:6px;">${msg.title}</div><div class="small">${msg.text}</div>`;

      kpisEl.appendChild(pill('Detected plucks', data.detected_onsets ?? '—', null));
      kpisEl.appendChild(pill('Clean loops', `${data.clean_loops}/${data.loops_total}`, passed ? true : false));
      kpisEl.appendChild(pill('Pitch accuracy', `${Math.round((data.pitch_accuracy||0)*100)}%`, (data.pitch_accuracy||0) >= 0.82));
      kpisEl.appendChild(pill('Mute rate', `${Math.round((data.mute_rate||0)*100)}%`, (data.mute_rate||0) <= 0.18));
      kpisEl.appendChild(pill('Timing CV', (data.timing_cv||0).toFixed(2), (data.timing_cv||9) <= 0.18));
      kpisEl.appendChild(pill('Missed hits', data.missed_hits ?? '—', (data.missed_hits||0) <= 2));

      const evs = (data.events || []).slice(0, 24);
      if (evs.length) {
        eventsWrap.style.display = 'block';
        evs.forEach((ev, idx) => {
          const expected = nearestLabel(ev.expected_hz);
          const detected = nearestLabel(ev.f0_hz);
          const result = ev.result || '—';
          const resClass = (result.startsWith('ok')) ? 'ok' : 'bad';
          const row = document.createElement('div');
          row.className = 'eventsRow';
          row.innerHTML = `
            <div class="tag">${idx+1}</div>
            <div>${expected}</div>
            <div>${detected}</div>
            <div class="${resClass}">${result}</div>
          `;
          eventsEl.appendChild(row);
        });
      }

      detailsEl.textContent = JSON.stringify({
        level: data.level,
        passed: data.passed,
        reason: data.reason,
        xp_awarded: data.xp_awarded,
        summary: {
          detected_onsets: data.detected_onsets,
          clean_loops: data.clean_loops,
          loops_total: data.loops_total,
          pitch_accuracy: data.pitch_accuracy,
          mute_rate: data.mute_rate,
          timing_cv: data.timing_cv,
          missed_hits: data.missed_hits
        }
      }, null, 2);

    } catch (e) {
      statusEl.innerHTML = `<span class="bad">Error:</span> ${e}`;
      detailsEl.textContent = String(e);
    } finally {
      analyzeBtn.disabled = false;
    }
  };

  // boot
  window.addEventListener("DOMContentLoaded", async () => {
    await loadLevels();
    setView("steps");
    setNow("—");
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/api/levels")
def api_levels():
    return [_level_to_api(s) for s in LEVELS.values()]


@app.post("/analyze")
async def analyze_endpoint(level: str = Query("lane_lock"), file: UploadFile = File(...)):
    spec = LEVELS.get(level)
    if not spec:
        return JSONResponse({"error": f"Unknown level '{level}'"}, status_code=400)

    # Save upload to a temp directory, convert to wav, analyze, then delete everything.
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "upload.bin")
        with open(in_path, "wb") as f:
            f.write(await file.read())

        wav_path = os.path.join(td, "audio.wav")
        try:
            run_ffmpeg_to_wav(in_path, wav_path, sr=spec.sr)
            result = analyze(level, wav_path)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

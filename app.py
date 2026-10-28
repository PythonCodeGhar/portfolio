import html
import io
import os
import re
import shutil
import threading
import time
import zlib
from pathlib import Path

import joblib
import numpy as np
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Auto-capture (live camera) ke liye:  pip install streamlit-webrtc av
try:
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    WEBRTC_OK = True
except Exception:   # package install nahi hai -> baaki app chalta rahe
    av = None
    VideoProcessorBase = object
    WEBRTC_OK = False

# =====================================================================
# 1. SETTINGS  (yahan apni details badlein)
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent      # jis folder me app.py hai (GitHub / Streamlit Cloud par bhi)
MODELS_DIR = BASE_DIR / "models"


LEGACY_DIR = Path(r"D:\ai_agentic\ai_conda")

PROFILE = {
    "name": "Ankit Prajapati",
    "role": "AI Developer",
    "about": (
        "I build Machine Learning, NLP, speech and computer-vision projects that turn raw text, "
        "audio and images into useful insights. Explore my live demos below: type "
        "something, upload your voice or take a photo and watch the models work."
    ),
    "skills": [
        "Python", "Machine Learning", "NLP", "Speech / Audio", "Computer Vision",
        "Scikit-learn", "OpenCV", "Librosa", "Pandas", "NumPy", "Streamlit",
    ],
    "links": {
        "GitHub": "https://github.com/PythonCodeGhar",
        "LinkedIn": "https://www.linkedin.com/in/ankit-kumar-b9235b326",
        "Email": "ankitgole579@gmail.com",
    },
}

# Mental-health page helpline numbers.
HELPLINES = [
    ("Health helpline (24x7, free)", "108"),
    
    ("Emergency", "112"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# NOTE: st.set_page_config() hamesha pehla Streamlit command hona chahiye,
# isliye sidebar image ko iske baad rakha gaya hai.
st.set_page_config(
    page_title=f"{PROFILE['name']} | AI Portfolio",
    layout="wide",
)

st.sidebar.image(
    "https://images.nightcafe.studio/galleries/GyzsoaOdxRKFcwbJD0sj/an_abstract_representation_of_artificial--1--gr9zw.jpg"
)

# =====================================================================
# 2. HELPERS
# =====================================================================


def H(s: str) -> str:
    """Multi-line HTML ko ek line me badalta hai (markdown code-block bug se bachne ke liye)."""
    return " ".join(line.strip() for line in s.strip().splitlines())


def chips(items):
    return "".join(f'<span class="chip">{html.escape(str(t))}</span>' for t in items)


_SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "env", "site-packages"}


def find_by_name(name: str):
    """app.py wale folder (aur uske andar ke sab folders) me file naam se dhundo."""
    direct = [BASE_DIR / name, MODELS_DIR / name]
    for c in direct:
        if c.is_file():
            return c
    for hit in BASE_DIR.rglob(name):
        if hit.is_file() and not (set(hit.relative_to(BASE_DIR).parts) & _SKIP_DIRS):
            return hit
    return None


def pick(*paths) -> Path:
    """
    Diye gaye paths me se pehla jo exist kare wo lo. Nahi mila to app.py ke folder me
    (subfolders me bhi) wahi file naam se dhundo. Phir bhi na mile to pehla path (error me dikhane ke liye).
    """
    for p in paths:
        if Path(p).exists():
            return Path(p)
    found = find_by_name(Path(paths[0]).name)
    return found if found else Path(paths[0])


@st.cache_resource(show_spinner="Loading model...")
def load_pickle(path_str: str):
    return joblib.load(path_str)


def files_ready(paths) -> bool:
    return all(Path(p).exists() for p in paths)


def require_files(paths) -> bool:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        st.error("Model file not found. Put it in the same folder as app.py "
                 "(or inside its `models/` folder) and make sure it is committed to GitHub.")
        st.code("app.py folder:  " + str(BASE_DIR) + "\n\nLooked for:\n" + "\n".join(missing),
                language="text")
        return False
    return True


def go(label: str):
    st.session_state.page = label


def set_text(key: str, value: str):
    st.session_state[key] = value


def theme(c1: str, c2: str):
    st.markdown(f"<style>:root{{--c1:{c1};--c2:{c2};}}</style>", unsafe_allow_html=True)


def neural_svg() -> str:
    """Chhota animated neural-network drawing (hero ke liye)."""
    layers, xs, h = [4, 6, 6, 3], [30, 120, 210, 300], 220
    pts = [[(xs[li], h / (n + 1) * (j + 1)) for j in range(n)] for li, n in enumerate(layers)]
    lines_svg = "".join(
        f'<line x1="{x1}" y1="{y1:.0f}" x2="{x2}" y2="{y2:.0f}"/>'
        for a, b in zip(pts, pts[1:]) for (x1, y1) in a for (x2, y2) in b
    )
    dots, k = "", 0
    for layer in pts:
        for (x, y) in layer:
            dots += f'<circle cx="{x}" cy="{y:.0f}" r="5" style="animation-delay:{(k % 7) * 0.35:.2f}s"/>'
            k += 1
    return (f'<svg class="neural" viewBox="0 0 330 {h}" xmlns="http://www.w3.org/2000/svg">'
            f'<g class="n-lines">{lines_svg}</g><g class="n-dots">{dots}</g></svg>')


def page_hero(p: dict):
    st.markdown(
        H(f"""
        <div class="page-hero">
            <div class="scan"></div>
            <div class="page-icon">{p['icon']}</div>
            <div class="page-hero-text">
                <div class="crumb">~/projects/{html.escape(p['key'])} <span class="cursor">▍</span></div>
                <h1>{html.escape(p['title'])}</h1>
                <p>{html.escape(p['tagline'])}</p>
                <div class="page-tags">{chips(p['tags'])}</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def how_panel(p: dict):
    steps = "".join(
        f'<div class="pipe-step"><div class="pipe-num">{i:02d}</div>'
        f'<div class="pipe-text">{html.escape(s)}</div></div>'
        for i, s in enumerate(p["how"], start=1)
    )
    st.markdown(
        H(f"""
        <div class="panel">
            <div class="panel-title"><span class="dot"></span>PIPELINE · HOW IT WORKS</div>
            <div class="pipe">{steps}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def result_card(emoji: str, color: str, title: str, sub: str):
    st.markdown(
        H(f"""
        <div class="result-card" style="--rc:{color}">
            <div class="result-emoji">{emoji}</div>
            <div>
                <div class="result-kicker">▸ MODEL OUTPUT</div>
                <div class="result-label">{html.escape(title)}</div>
                <div class="result-sub">{html.escape(sub)}</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def prob_bars(scores):
    if not scores:
        return
    bars = "".join(
        H(f"""
        <div class="bar-row">
            <span class="bar-name">{html.escape(n.capitalize())}</span>
            <div class="bar"><div class="bar-fill" style="width:{s * 100:.1f}%"></div></div>
            <span class="bar-pct">{s * 100:.1f}%</span>
        </div>
        """)
        for n, s in scores[:6]
    )
    st.markdown(
        f"<div class='panel'><div class='panel-title'><span class='dot'></span>"
        f"PROBABILITY DISTRIBUTION</div>{bars}</div>",
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        f"<div class='footer'><span class='prompt'>&gt;</span> built by "
        f"{html.escape(PROFILE['name'])} · python · scikit-learn · streamlit"
        "<span class='cursor'>▍</span></div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# 3. CSS
# =====================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --c1:#8b5cf6; --c2:#22d3ee;
    --bg:#06070d; --panel:rgba(255,255,255,.035); --line:rgba(255,255,255,.09);
    --text:#eceefb; --muted:#9aa0bd;
    --mono:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    --head:'Space Grotesk', 'Inter', sans-serif;
}

.stApp, .stApp p, .stApp label, .stApp button, .stApp textarea,
.stApp input, .stApp li { font-family: 'Inter', sans-serif; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: var(--head); }
.stApp { background: var(--bg); color: var(--text); }
::selection { background: color-mix(in srgb, var(--c1) 55%, transparent); color:#fff; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { position: relative; z-index: 1; padding-top: 2rem; max-width: 1200px; }
hr { border-color: var(--line) !important; }

/* ---------- AI background: glow orbs + faint grid ---------- */
.stApp::before {
    content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
    background:
        radial-gradient(700px 480px at 8% -5%, color-mix(in srgb, var(--c1) 28%, transparent), transparent 65%),
        radial-gradient(620px 460px at 98% 105%, color-mix(in srgb, var(--c2) 22%, transparent), transparent 65%);
}
.stApp::after {
    content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
    background-image:
        linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
    background-size: 44px 44px;
    -webkit-mask-image: radial-gradient(ellipse at 50% 25%, #000 15%, transparent 75%);
    mask-image: radial-gradient(ellipse at 50% 25%, #000 15%, transparent 75%);
}

.mono { font-family: var(--mono); }
.cursor { color: var(--c2); animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
@keyframes pulse { 0%,100% { opacity:.35; transform: scale(.9); } 50% { opacity:1; transform: scale(1.25); } }
@keyframes dash { to { stroke-dashoffset: -40; } }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes scan { 0% { transform: translateX(-120%); } 100% { transform: translateX(520%); } }
@keyframes typing { to { width: 100%; } }
@keyframes caret { 50% { border-color: transparent; } }
@keyframes spin { to { transform: rotate(360deg); } }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0d18, #080a12);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: #ffffff; }
.side-logo { text-align:center; padding: 10px 0 6px 0; }
.avatar {
    width: 74px; height: 74px; margin: 0 auto 10px auto; border-radius: 50%; padding: 3px;
    background: conic-gradient(from 0deg, var(--c1), var(--c2), var(--c1));
    animation: spin 6s linear infinite;
}
.avatar span {
    width:100%; height:100%; border-radius:50%; background:#0b0d18;
    display:flex; align-items:center; justify-content:center;
    font: 700 24px var(--head); animation: spin 6s linear infinite reverse;
}
.side-logo .name { font: 700 19px var(--head); }
.side-logo .role { font: 500 12px var(--mono); color: var(--c2) !important; margin-top: 2px; }
.side-label {
    font: 600 10.5px var(--mono); letter-spacing: .18em; color: var(--muted) !important;
    margin: 14px 0 6px 2px;
}
[data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 8px 12px; border-radius: 12px; border: 1px solid transparent; transition: .2s; width:100%;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--panel); border-color: var(--line); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, color-mix(in srgb, var(--c1) 30%, transparent), transparent);
    border-color: color-mix(in srgb, var(--c1) 55%, transparent);
}
.side-status {
    display:flex; align-items:center; gap:9px; font-size: 12.5px; padding: 5px 2px; color:#c9cde4;
}
.side-state { margin-left:auto; font: 500 9.5px var(--mono); letter-spacing:.08em; color: var(--muted); }
.led { width:8px; height:8px; border-radius:50%; flex:none; }
.led.on  { background:#34d399; box-shadow:0 0 0 0 rgba(52,211,153,.7); animation: ledpulse 2s infinite; }
.led.off { background:#f87171; box-shadow:0 0 8px rgba(248,113,113,.6); }
@keyframes ledpulse { 70% { box-shadow: 0 0 0 7px rgba(52,211,153,0); } 100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); } }

/* ---------- Home hero (terminal window) ---------- */
.hero {
    position: relative; overflow: hidden; border-radius: 22px; margin-bottom: 20px;
    border: 1px solid var(--line);
    background: linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.015));
    box-shadow: 0 24px 70px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.07);
}
.term-bar {
    display:flex; align-items:center; gap:7px; padding: 11px 18px;
    background: rgba(0,0,0,.28); border-bottom: 1px solid var(--line);
    font: 500 12px var(--mono); color: var(--muted);
}
.term-bar i { width:11px; height:11px; border-radius:50%; display:inline-block; }
.term-bar i:nth-child(1){ background:#ff5f57; } .term-bar i:nth-child(2){ background:#febc2e; }
.term-bar i:nth-child(3){ background:#28c840; }
.term-bar b { margin-left: 10px; font-weight: 500; }
.hero-body { display:flex; flex-wrap:wrap; align-items:center; gap: 24px; padding: 34px 40px; }
.hero-left { flex: 1 1 440px; min-width: 0; }
.hero-right { flex: 0 1 330px; }
.hero-cmd { font: 500 14px var(--mono); color: var(--c2); margin-bottom: 14px; }
.typing {
    display:inline-block; overflow:hidden; white-space:nowrap; vertical-align:bottom; width:0;
    border-right: 2px solid var(--c2);
    animation: typing 2.6s steps(30) .3s forwards, caret .8s step-end infinite;
}
.hero-hi { font: 500 15px var(--mono); color: var(--muted); }
.hero h1 {
    font: 700 58px/1.05 var(--head); margin: 4px 0 0 0; letter-spacing: -.02em;
    background: linear-gradient(90deg, #ffffff 0%, color-mix(in srgb, var(--c1) 70%, #fff) 45%, var(--c2) 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-role { font: 600 18px var(--mono); color: var(--c2); margin: 10px 0 14px 0; }
.hero p { font-size: 15.5px; color: #cfd2e8; max-width: 640px; line-height: 1.75; }

.neural { width:100%; height:auto; filter: drop-shadow(0 0 18px color-mix(in srgb, var(--c1) 45%, transparent)); }
.neural .n-lines line { stroke: rgba(170,180,255,.16); stroke-width: 1; }
.neural .n-lines line:nth-child(7n) {
    stroke: var(--c2); opacity:.6; stroke-dasharray: 4 6; animation: dash 3s linear infinite;
}
.neural .n-dots circle { fill: var(--c1); transform-box: fill-box; transform-origin: center; animation: pulse 2.4s ease-in-out infinite; }
.neural .n-dots circle:nth-child(3n) { fill: var(--c2); }

.chip {
    display:inline-block; padding: 4px 12px; margin: 4px 6px 0 0; border-radius: 8px;
    font: 500 11.5px var(--mono); letter-spacing: .01em; color: #dfe2f7;
    background: rgba(255,255,255,.055); border: 1px solid var(--line);
}

/* ---------- Stat strip ---------- */
.stat-row { display:grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 8px; }
.stat {
    padding: 16px 20px; border-radius: 16px; background: var(--panel); border: 1px solid var(--line);
}
.stat .v {
    font: 700 30px var(--mono);
    background: linear-gradient(90deg, var(--c1), var(--c2));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.stat .l { font: 500 11px var(--mono); letter-spacing: .14em; color: var(--muted); text-transform: uppercase; }
@media (max-width: 800px) { .stat-row { grid-template-columns: repeat(2, 1fr); } .hero h1 { font-size: 40px; } }

.section-title {
    display:flex; align-items:center; gap:10px; font: 700 25px var(--head); margin: 30px 0 4px 0;
}
.section-title .mono { color: var(--c2); font-size: 20px; }
.section-title::after { content:""; flex:1; height:1px; background: linear-gradient(90deg, var(--line), transparent); }
.section-sub { color: var(--muted); font-size: 14px; margin-bottom: 16px; }

/* ---------- Project cards (home) ---------- */
.proj-card {
    position: relative; overflow: hidden; min-height: 330px; padding: 22px;
    background: linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
    border: 1px solid var(--line); border-radius: 20px; transition: .3s; backdrop-filter: blur(6px);
}
.proj-card::before {
    content:""; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, var(--c1), var(--c2));
}
.proj-card::after {
    content:""; position:absolute; width:220px; height:220px; right:-90px; top:-90px; border-radius:50%;
    background: radial-gradient(circle, color-mix(in srgb, var(--c1) 35%, transparent), transparent 70%);
    opacity:.5; transition:.3s;
}
.proj-card:hover {
    transform: translateY(-6px); border-color: color-mix(in srgb, var(--c1) 70%, transparent);
    box-shadow: 0 18px 44px rgba(0,0,0,.55), 0 0 32px color-mix(in srgb, var(--c1) 28%, transparent);
}
.proj-card:hover::after { opacity: 1; }
.proj-top { position:relative; z-index:1; display:flex; justify-content:space-between; align-items:center; }
.proj-icon {
    width: 54px; height: 54px; border-radius: 16px; font-size: 27px;
    display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg, var(--c1), var(--c2));
    box-shadow: 0 8px 22px color-mix(in srgb, var(--c1) 40%, transparent);
}
.proj-id { position:relative; z-index:1; font: 500 11px var(--mono); letter-spacing:.12em; color: var(--muted); margin-top: 16px; }
.proj-card h3 { position:relative; z-index:1; margin: 6px 0 6px 0; font-size: 20px; font-weight: 700; }
.proj-card p { position:relative; z-index:1; color:#aeb3cf; font-size: 13.5px; line-height: 1.65; }
.proj-card div { position:relative; z-index:1; }
.status { font: 600 10.5px var(--mono); letter-spacing:.06em; padding: 4px 10px; border-radius: 999px; }
.status.ok  { background: rgba(52,211,153,.14); color:#4ade9a; border:1px solid rgba(52,211,153,.35); }
.status.bad { background: rgba(248,113,113,.12); color:#ff8f8f; border:1px solid rgba(248,113,113,.35); }

.soon-card {
    min-height: 330px; border: 2px dashed #2a2d45; border-radius: 20px; padding: 22px;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    text-align:center; color:#7f84a3; font-family: var(--mono);
}
.soon-card .big { font-size: 40px; margin-bottom: 8px; }

/* ---------- Workflow band ---------- */
.flow { display:grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 6px 0 10px 0; }
.flow-node {
    position:relative; padding: 16px 16px 14px 16px; border-radius: 16px;
    background: var(--panel); border: 1px solid var(--line);
}
.flow-node:not(:last-child)::after {
    content:"→"; position:absolute; right:-11px; top:50%; transform: translateY(-50%);
    color: var(--c2); font: 700 16px var(--mono); z-index:2;
}
.flow-node .n { font: 700 12px var(--mono); color: var(--c2); }
.flow-node .t { font: 700 16px var(--head); margin: 4px 0 2px 0; }
.flow-node .s { font-size: 12.5px; color: var(--muted); }
@media (max-width: 800px) { .flow { grid-template-columns: 1fr 1fr; } .flow-node::after { display:none; } }

/* ---------- Project page hero ---------- */
.page-hero {
    position:relative; overflow:hidden; display:flex; align-items:center; gap: 22px;
    padding: 28px 32px; border-radius: 22px; margin-bottom: 22px;
    background:
        linear-gradient(120deg, color-mix(in srgb, var(--c1) 38%, #0a0c16), color-mix(in srgb, var(--c2) 30%, #0a0c16));
    border: 1px solid color-mix(in srgb, var(--c1) 55%, transparent);
    box-shadow: 0 18px 50px rgba(0,0,0,.5), 0 0 40px color-mix(in srgb, var(--c1) 22%, transparent);
}
.page-hero .scan {
    position:absolute; top:0; bottom:0; left:0; width:18%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.10), transparent);
    animation: scan 5s linear infinite; pointer-events:none;
}
.page-icon {
    position:relative; font-size: 44px; width: 78px; height: 78px; border-radius: 20px; flex:none;
    display:flex; align-items:center; justify-content:center;
    background: rgba(0,0,0,.35); border: 1px solid rgba(255,255,255,.18);
}
.page-hero-text { position:relative; }
.crumb { font: 500 12.5px var(--mono); color: rgba(255,255,255,.75); }
.page-hero h1 { margin: 2px 0 0 0; font: 700 34px var(--head); color:#fff; letter-spacing:-.01em; }
.page-hero p  { margin: 4px 0 0 0; color: rgba(255,255,255,.88); font-size: 15px; }
.page-tags .chip { background: rgba(0,0,0,.28); border-color: rgba(255,255,255,.2); }

/* ---------- Panels ---------- */
.panel {
    background: linear-gradient(160deg, rgba(255,255,255,.045), rgba(255,255,255,.015));
    border: 1px solid var(--line); border-radius: 18px; padding: 18px 20px; margin-bottom: 14px;
}
.panel h4 { margin: 0 0 10px 0; font-size: 16px; }
.panel li, .panel p { color:#c3c7e0; font-size: 14px; line-height: 1.7; }
.panel-title {
    display:flex; align-items:center; gap:9px; font: 600 11px var(--mono);
    letter-spacing:.16em; color: var(--muted); margin-bottom: 12px;
}
.panel-title .dot { width:7px; height:7px; border-radius:50%; background: var(--c2); box-shadow: 0 0 10px var(--c2); }
.pipe { position:relative; }
.pipe-step { position:relative; display:flex; gap:14px; align-items:flex-start; padding-bottom: 14px; }
.pipe-step:not(:last-child)::before {
    content:""; position:absolute; left:15px; top:32px; bottom:2px; width:1px;
    background: linear-gradient(var(--c1), transparent);
}
.pipe-num {
    flex:none; width:31px; height:31px; border-radius:10px; display:flex; align-items:center; justify-content:center;
    font: 700 11.5px var(--mono); color:#fff;
    background: linear-gradient(135deg, var(--c1), var(--c2));
}
.pipe-text { font-size: 13.8px; line-height: 1.55; color:#cdd1e8; padding-top: 5px; }

/* ---------- Result card (HUD) ---------- */
.result-card {
    display:flex; align-items:center; gap: 20px;
    padding: 22px 26px; border-radius: 20px; margin: 10px 0 14px 0;
    background: linear-gradient(120deg, color-mix(in srgb, var(--rc) 24%, #0b0d17), #0b0d17);
    border: 1px solid var(--rc);
    box-shadow: 0 0 34px color-mix(in srgb, var(--rc) 24%, transparent);
}
.result-emoji { font-size: 54px; }
.result-kicker { font: 600 10.5px var(--mono); letter-spacing:.18em; color: var(--muted); margin-bottom: 4px; }
.result-label { font: 800 30px/1.1 var(--head); color: var(--rc); }
.result-sub { color:#c3c7e0; font: 500 13px var(--mono); margin-top: 6px; }

.bar-row { display:flex; align-items:center; gap: 12px; margin: 10px 0; }
.bar-name { width: 150px; font-size: 13.5px; color:#dcdfef; }
.bar { flex: 1; height: 10px; background:#1a1c2c; border-radius: 99px; overflow:hidden; }
.bar-fill {
    height:100%; border-radius: 99px; transform-origin:left; animation: grow .9s ease-out;
    background: linear-gradient(90deg, var(--c1), var(--c2)); box-shadow: 0 0 14px color-mix(in srgb, var(--c2) 60%, transparent);
}
.bar-pct { width: 56px; text-align:right; font: 500 12.5px var(--mono); color:#b9bed8; }

/* ---------- Support boxes ---------- */
.support-box {
    padding: 16px 20px; border-radius: 16px; margin-top: 10px;
    background: rgba(94,150,255,.09); border: 1px solid rgba(94,150,255,.35);
    color:#dbe6ff; font-size: 14.5px; line-height: 1.7;
}

/* ---------- Student list (attendance) ---------- */
.student-row {
    display:flex; align-items:center; gap: 12px; margin: 8px 0; padding: 10px 14px; border-radius: 12px;
    background: rgba(255,255,255,.04); border: 1px solid var(--line);
}
.student-avatar {
    width: 34px; height: 34px; border-radius: 10px; font: 700 14px var(--mono);
    display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg, var(--c1), var(--c2)); color:#fff;
}
.student-name { flex:1; font-size: 14.5px; font-weight: 600; color:#f2f2f7; }
.student-count { font: 500 12px var(--mono); color:#b4b9d4; }
.student-count.low { color:#ff9c9c; }

/* ---------- Movie cards ---------- */
.poster-card {
    position: relative; border-radius: 18px; padding: 10px; margin-bottom: 14px;
    background: linear-gradient(145deg, var(--m1), var(--m2));
    box-shadow: 0 8px 25px rgba(0,0,0,.5); transition: .3s;
}
.poster-card:hover { transform: translateY(-6px) scale(1.02); box-shadow: 0 14px 35px var(--m1); }
.poster-card img { width:100%; aspect-ratio: 2/3; object-fit: cover; border-radius: 12px; display:block; }
.rank-badge {
    position:absolute; top:18px; left:18px; width:38px; height:38px; border-radius:50%;
    background:#fff; color: var(--m1); font: 800 19px var(--mono);
    display:flex; align-items:center; justify-content:center;
    box-shadow: 0 3px 10px rgba(0,0,0,.5); z-index:2;
}
.poster-title {
    margin-top:10px; min-height:60px; padding:8px 10px; border-radius:10px;
    background: rgba(0,0,0,.45); display:flex; align-items:center; justify-content:center;
    text-align:center; font-size:16px; font-weight:700; color:#fff; line-height:1.3;
    text-shadow: 0 1px 4px rgba(0,0,0,.8);
}

/* ---------- Streamlit widgets ---------- */
.stButton > button, .stLinkButton > a {
    width:100%; height:46px; border:none; border-radius:12px; font-weight:600; font-size:14.5px; transition:.25s;
    background: linear-gradient(90deg, var(--c1), var(--c2)); color:#fff;
    box-shadow: 0 6px 20px color-mix(in srgb, var(--c1) 30%, transparent);
}
.stButton > button:hover, .stLinkButton > a:hover {
    filter: brightness(1.12); transform: translateY(-2px); color:#fff;
    box-shadow: 0 10px 28px color-mix(in srgb, var(--c2) 38%, transparent);
}
.stButton > button:disabled { filter: grayscale(.8); opacity:.45; box-shadow:none; }
.stTextArea textarea, .stTextInput input, .stNumberInput input {
    background: rgba(255,255,255,.04); color:#fff; border:1px solid var(--line); border-radius:12px; font-size:15px;
    transition: .2s;
}
.stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--c2); box-shadow: 0 0 0 3px color-mix(in srgb, var(--c2) 22%, transparent);
}
div[data-baseweb="select"] > div { background: rgba(255,255,255,.04); color:#fff; border-radius:12px; border-color: var(--line); }
[data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.03); border:1px dashed #3a3e63; border-radius:14px; }
[data-testid="stMetric"] {
    background: var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px 16px;
}
[data-testid="stMetricValue"] { font-family: var(--mono); }
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--line); }
[data-testid="stExpander"] details { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, var(--c1), var(--c2)); }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 14px; color: var(--muted); }
.stTabs [aria-selected="true"] { color:#fff; }
.stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--c1), var(--c2)); height: 3px; }

/* segmented control look for horizontal radios in the main area */
[data-testid="stMain"] [role="radiogroup"], section.main [role="radiogroup"] { gap: 8px; flex-wrap: wrap; }
[data-testid="stMain"] [role="radiogroup"] label, section.main [role="radiogroup"] label {
    padding: 7px 16px; border-radius: 999px; background: rgba(255,255,255,.04);
    border: 1px solid var(--line); transition: .2s;
}
[data-testid="stMain"] [role="radiogroup"] label:hover, section.main [role="radiogroup"] label:hover { border-color: var(--c2); }
[data-testid="stMain"] [role="radiogroup"] label:has(input:checked), section.main [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, color-mix(in srgb, var(--c1) 55%, transparent), color-mix(in srgb, var(--c2) 40%, transparent));
    border-color: var(--c2);
}

.footer { text-align:center; color:#6f7494; font: 500 12.5px var(--mono); margin: 50px 0 10px 0; }
.footer .prompt { color: var(--c2); }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: #272a44; border-radius: 9px; }
::-webkit-scrollbar-thumb:hover { background: #3a3e63; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
    .typing { width: 100%; }
}
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_STYLE = ("🔹", "#6c8cff")

# =====================================================================
# 4. FOOD REVIEW PAGE  (text -> sentiment)
# =====================================================================

RED_BALLOONS = """
<style>
.balloon-container { position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9999; }
.red-balloon { position: absolute; bottom: -150px; width: 55px; height: 70px;
    background: red; border-radius: 50%; animation: floatUp 5s linear infinite; }
.red-balloon::after { content: ""; position: absolute; bottom: -80px; left: 27px;
    width: 2px; height: 80px; background: #555; }
.b1 { left: 10%; animation-delay: 0s; }
.b2 { left: 30%; animation-delay: 1s; }
.b3 { left: 50%; animation-delay: 2s; }
.b4 { left: 70%; animation-delay: 0.5s; }
.b5 { left: 90%; animation-delay: 1.5s; }
@keyframes floatUp {
    0%   { transform: translateY(0); opacity: 1; }
    100% { transform: translateY(-110vh); opacity: 0; }
}
</style>
<div class="balloon-container">
    <div class="red-balloon b1"></div><div class="red-balloon b2"></div>
    <div class="red-balloon b3"></div><div class="red-balloon b4"></div>
    <div class="red-balloon b5"></div>
</div>
"""


def food_celebration(label: str):
    """Positive -> Streamlit balloons, Negative -> red balloons, Neutral -> kuch nahi."""
    if label == "positive":
        st.balloons()
    elif label == "negative":
        st.markdown(H(RED_BALLOONS), unsafe_allow_html=True)


def news_extras(label: str):
    """Fake / Real news ke baad ek zimmedaar note. Category labels par kuch nahi dikhata."""
    if label in {"fake", "false"}:
        st.markdown(
            "<div class='support-box'>⚠️ <b>The model thinks this article looks fake.</b><br>"
            "This is an automated estimate, not proof. Check the claim on trusted news sites "
            "or fact-checkers before sharing it.</div>",
            unsafe_allow_html=True,
        )
    elif label in {"real", "true"}:
        st.markdown(
            "<div class='support-box'>✅ <b>The model thinks this article looks real.</b><br>"
            "No model is perfect, so it's still a good habit to cross-check important news "
            "with a second trusted source.</div>",
            unsafe_allow_html=True,
        )


def classify_text(text, model):
    """Pipeline model: model.predict([text]) (bilkul aapke original code jaisa)."""
    label = str(model.predict([text])[0])

    scores = []
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba([text])[0]
            scores = sorted(
                ((str(c), float(pr)) for c, pr in zip(model.classes_, probs)),
                key=lambda x: -x[1],
            )
        except Exception:
            scores = []
    return label, scores


def render_classifier(p: dict):
    page_hero(p)

    if not require_files([p["files"]["model"]]):
        return

    model = load_pickle(str(p["files"]["model"]))

    key = p["key"]
    text_key = f"{key}_text"
    if text_key not in st.session_state:
        st.session_state[text_key] = ""

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(f"##### ✍️ {p['input_label']}")
        text = st.text_area(
            "input", key=text_key, height=170,
            placeholder=p["placeholder"], label_visibility="collapsed",
        )
        st.caption(f"{len(text)} characters · {len(text.split())} words")

        if p.get("examples"):
            st.caption("Try an example:")
            ex_cols = st.columns(len(p["examples"]))
            for i, (c, ex) in enumerate(zip(ex_cols, p["examples"])):
                with c:
                    st.button(
                        f"Example {i + 1}", key=f"{key}_ex{i}",
                        on_click=set_text, args=(text_key, ex),
                    )

        analyze = st.button(p["button_text"], key=f"{key}_go")

    with right:
        how_panel(p)

    if analyze:
        if text.strip() == "":
            st.warning(p.get("empty_msg", "⚠️ Please enter some text first."))
            return

        with st.spinner("Analyzing..."):
            label, scores = classify_text(text, model)

        label_l = label.lower()
        emoji, color = p["label_style"].get(label_l, DEFAULT_STYLE)
        conf = next((s for n, s in scores if n == label), None)
        sub = f"Confidence: {conf * 100:.1f}%" if conf is not None else "Prediction by your trained model"

        display = p.get("label_alias", {}).get(label_l, label.capitalize())
        title = f"{display} {p.get('result_suffix', 'Review')}".strip()

        st.markdown("---")
        result_card(emoji, color, title, sub)
        prob_bars(scores)

        if p.get("on_result"):
            p["on_result"](label_l)

    footer()


# =====================================================================
# 5. MOVIE RECOMMENDATION PAGE
# =====================================================================

MOVIE_NAME_COL = "name"     # dataframe me movie ke naam ka column
MOVIE_YEAR_COL = None       # agar year column hai to e.g. "year" likhein

MOVIE_GRADIENTS = [
    ("#ff512f", "#dd2476"),   # red-pink
    ("#00c6ff", "#0072ff"),   # blue
    ("#f7971e", "#ffd200"),   # orange-yellow
    ("#11998e", "#38ef7d"),   # green
]

NO_POSTER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='500' height='750'>"
    "<rect width='100%' height='100%' fill='%2317171d'/>"
    "<text x='50%' y='50%' fill='white' font-size='36' "
    "text-anchor='middle'>No Poster</text></svg>"
)


@st.cache_data(show_spinner=False)
def get_movie_poster(movie_name, year=None):
    query = re.sub(r"[^a-z0-9]+", "_", str(movie_name).lower()).strip("_")
    if not query:
        return None

    url = f"https://v2.sg.media-imdb.com/suggestion/{query[0]}/{query}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        results = r.json().get("d", [])
    except Exception:
        return None

    movies = [m for m in results if m.get("qid") in ("movie", "tvMovie") and m.get("i")]
    if year:
        movies = [m for m in movies if str(m.get("y")) == str(year)]
    if not movies:
        return None

    exact = [m for m in movies if m["l"].lower().strip() == str(movie_name).lower().strip()]
    best = (exact or movies)[0]
    return re.sub(r"\._V1_.*\.jpg$", "._V1_UX500.jpg", best["i"]["imageUrl"])


def render_movie(p: dict):
    page_hero(p)

    f = p["files"]
    if not require_files([f["model"], f["vectors"], f["dataframe"]]):
        return

    model = load_pickle(str(f["model"]))
    X = load_pickle(str(f["vectors"]))
    df = load_pickle(str(f["dataframe"]))

    c1, c2 = st.columns([3, 1], gap="large")
    with c1:
        selected = st.selectbox("🎬 Select a movie to get recommendations:", df[MOVIE_NAME_COL].values)
    with c2:
        n = st.slider("How many?", 4, 12, 4, step=4)

    if st.button("🔍 Get Recommendations", key="movie_go"):
        st.session_state.movie_shown = selected

    shown = st.session_state.get("movie_shown")
    if shown is not None and shown in set(df[MOVIE_NAME_COL].values):
        pos = int(np.flatnonzero(df[MOVIE_NAME_COL].values == shown)[0])
        row = X[pos]
        row = row.toarray() if hasattr(row, "toarray") else np.asarray(row).reshape(1, -1)

        _, idx = model.kneighbors(row, n_neighbors=n + 1)
        names = df[MOVIE_NAME_COL].iloc[idx[0]].values[1:n + 1]   # [0] = selected movie khud
        years = (
            df[MOVIE_YEAR_COL].iloc[idx[0]].values[1:n + 1]
            if MOVIE_YEAR_COL and MOVIE_YEAR_COL in df.columns
            else [None] * len(names)
        )

        st.markdown(
            f"<div class='section-title'>Because you liked "
            f"<span style='color:var(--c1)'>{html.escape(str(shown))}</span></div>",
            unsafe_allow_html=True,
        )

        for start in range(0, len(names), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                i = start + j
                if i >= len(names):
                    break
                m1, m2 = MOVIE_GRADIENTS[i % 4]
                poster = get_movie_poster(names[i], years[i]) or NO_POSTER
                with col:
                    st.markdown(
                        H(f"""
                        <div class="poster-card" style="--m1:{m1}; --m2:{m2};">
                            <div class="rank-badge">{i + 1}</div>
                            <img src="{poster}">
                            <div class="poster-title">{html.escape(str(names[i]))}</div>
                        </div>
                        """),
                        unsafe_allow_html=True,
                    )

    footer()


# =====================================================================
# 6. SPEECH MENTAL HEALTH PAGE  (audio -> Normal / Anxious / Depressed)
# =====================================================================

SPEECH_LABELS = ["Normal", "Anxious", "Depressed"]   # LabelEncoder inhi par fit hota hai
EXPECTED_FEATURES = 33


def stable_hash(text: str) -> int:
    """
    Aapke original code me hash(text) % 1000 tha. Python ka hash() har run par badal jaata hai
    (hash randomization), isliye same audio par bhi alag prediction aa sakti thi.
    crc32 hamesha same value deta hai.
    """
    return zlib.crc32(text.encode("utf-8")) % 1000


@st.cache_data(show_spinner=False, max_entries=20)
def extract_features(audio_bytes: bytes, segment: str, transcript: str, user_age: int):
    """
    Aapke original function jaisi hi 33 features:
    1 segment + 4 spectral + 6 (loudness, pitch, hnr, f1, f2, f3) + 6 (duration, pauses,
    sentiment, complexity, word_frequency, keywords) + 2 (emotions, transcript) + age + 13 MFCC
    """
    import librosa

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    if y.size == 0:
        raise ValueError("The audio file is empty.")

    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))

    rms = librosa.feature.rms(y=y)
    loudness = np.mean(rms)

    # Pitch (agar koi voiced frame na mile to 0, warna NaN aa jaata tha)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    voiced = [pitches[:, i].max() for i in range(pitches.shape[1]) if magnitudes[:, i].max() > 0]
    pitch = float(np.mean(voiced)) if voiced else 0.0

    # Harmonic / percussive ratio (zero se divide hone par inf na aaye)
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    hnr = float(np.mean(harmonic / np.where(percussive == 0, 1e-10, percussive)))

    # Placeholders (aapke original code jaise, formants ke liye alag tools chahiye)
    formant_f1, formant_f2, formant_f3 = 0, 0, 0

    speech_duration = librosa.get_duration(y=y, sr=sr)
    pauses = int(len(np.where(rms[0] < 0.01)[0]))

    sentiment = complexity = word_frequency = keywords = emotions = 0

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_features = [float(np.mean(mfccs[i])) for i in range(13)]

    feature_vector = [
        stable_hash(segment),
        zcr_mean, spectral_centroid_mean, spectral_rolloff_mean, spectral_flux,
        loudness, pitch, hnr, formant_f1, formant_f2, formant_f3,
        speech_duration, pauses, sentiment, complexity, word_frequency, keywords,
        emotions, stable_hash(transcript),
        user_age,
    ] + mfcc_features

    feature_vector = [float(v) for v in feature_vector]
    feature_vector = list(np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0))

    summary = {
        "duration": float(speech_duration),
        "loudness": float(loudness),
        "pitch": float(pitch),
        "pauses": pauses,
    }
    return feature_vector, summary


@st.cache_resource(show_spinner=False)
def get_label_encoder(path_str: str):
    """label_encoder.pkl ho to load, warna default labels se bana lo (aapke original jaisa)."""
    path = Path(path_str)
    if path.exists():
        return joblib.load(path), False

    encoder = LabelEncoder()
    encoder.fit(SPEECH_LABELS)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, path)
    except Exception:
        pass
    return encoder, True


def decode_label(value, encoder) -> str:
    if isinstance(value, (int, np.integer)):
        return str(encoder.inverse_transform([int(value)])[0])
    return str(value)


def speech_extras(label: str):
    helplines = "<br>".join(f"📞 <b>{html.escape(n)}</b>: {html.escape(num)}" for n, num in HELPLINES)

    if label in {"anxious", "depressed"}:
        st.markdown(
            H(f"""
            <div class="support-box">
                <b>🌱 Taking care of your mind is important.</b><br>
                This is only an automated estimate from your voice and not a diagnosis.
                Talking to a friend, family member or a mental-health professional can really help.
                If things feel heavy right now, you can reach out here:<br><br>{helplines}
            </div>
            """),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='support-box'>😊 Your voice doesn't show strong signs of distress. "
            "Keep taking care of yourself!</div>",
            unsafe_allow_html=True,
        )


def render_speech(p: dict):
    page_hero(p)

    if not require_files([p["files"]["model"]]):
        return

    try:
        import librosa  # noqa: F401
    except ImportError:
        st.error("The `librosa` package is not installed. Run this in your terminal:")
        st.code("pip install librosa soundfile", language="bash")
        return

    model = load_pickle(str(p["files"]["model"]))
    encoder, recreated = get_label_encoder(str(p["files"]["label_encoder"]))
    if recreated:
        st.warning("Label encoder file not found, so it was recreated with the default labels: "
                   + ", ".join(SPEECH_LABELS))

    st.info(p["disclaimer"])

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("##### 👤 About you")
        c1, c2 = st.columns([2, 1])
        with c1:
            name = st.text_input("Your name", key="speech_name", placeholder="e.g. Ankit")
        with c2:
            age = st.number_input("Your age", min_value=0, max_value=100, value=25, step=1, key="speech_age")

        st.markdown("##### 🎤 Your voice")
        uploaded = None
        recorded = None
        if hasattr(st, "audio_input"):
            tab_up, tab_rec = st.tabs(["📁 Upload audio", "🎙️ Record now"])
            with tab_up:
                uploaded = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"],
                                            key="speech_file", label_visibility="collapsed")
            with tab_rec:
                recorded = st.audio_input("Record your voice", key="speech_rec",
                                          label_visibility="collapsed")
        else:
            uploaded = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"],
                                        key="speech_file", label_visibility="collapsed")

        audio_bytes = None
        if uploaded is not None:
            audio_bytes = uploaded.getvalue()
        elif recorded is not None:
            audio_bytes = recorded.getvalue()

        if audio_bytes:
            st.audio(audio_bytes)

        analyze = st.button(p["button_text"], key="speech_go")

    with right:
        how_panel(p)

    if analyze:
        if not audio_bytes:
            st.warning("⚠️ Please upload or record an audio clip first.")
            return

        try:
            with st.spinner("Listening to your voice..."):
                features, summary = extract_features(
                    audio_bytes, "Unknown", "No transcript provided", int(age)
                )

                if len(features) != EXPECTED_FEATURES:
                    st.error(f"Feature vector size mismatch! Extracted {len(features)} features, "
                             f"but the model needs {EXPECTED_FEATURES}.")
                    return

                arr = np.array(features).reshape(1, -1)
                raw_pred = model.predict(arr)[0]
                label = decode_label(raw_pred, encoder)

                scores = []
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(arr)[0]
                        scores = sorted(
                            ((decode_label(c, encoder), float(pr)) for c, pr in zip(model.classes_, probs)),
                            key=lambda x: -x[1],
                        )
                    except Exception:
                        scores = []
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            return

        label_l = label.lower()
        emoji, color = p["label_style"].get(label_l, DEFAULT_STYLE)
        conf = next((s for n, s in scores if n == label), None)
        sub = f"Confidence: {conf * 100:.1f}%" if conf is not None else "Prediction by your trained model"

        st.markdown("---")
        if name.strip():
            st.markdown(f"#### Hello, {html.escape(name.strip())}! 👋")
        result_card(emoji, color, label, sub)
        prob_bars(scores)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Duration", f"{summary['duration']:.1f}s")
        m2.metric("Loudness", f"{summary['loudness']:.3f}")
        m3.metric("Avg pitch", f"{summary['pitch']:.0f} Hz")
        m4.metric("Pause frames", summary["pauses"])

        speech_extras(label_l)

    with st.expander("🔧 Model info"):
        st.write("Labels known by the label encoder:")
        st.write(list(encoder.classes_))
        st.caption(f"The model expects {EXPECTED_FEATURES} audio features per clip.")

    footer()


# =====================================================================
# 6B. FACE GENDER PAGE  (photo -> detect faces -> Female / Male)
# =====================================================================

FACE_SIZE = 200        # aapka model 200x200 grayscale face par train hua hai
FACE_MAX_SIDE = 1280   # bahut badi photo ko itne pixel tak chhota karo (detection tez hogi)


def hex_to_bgr(hex_color: str):
    """'#ec4899' -> (b, g, r). OpenCV colors BGR me leta hai."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def face_label(pred, label_map) -> str:
    """Model ka output (0/1 ya text) -> 'Female' / 'Male'."""
    try:
        return label_map[int(pred)]
    except (ValueError, TypeError, KeyError):
        return str(pred).capitalize()


def decode_image(data: bytes):
    """Uploaded / camera image bytes -> BGR image (OpenCV format), zarurat ho to chhoti."""
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read this image. Please use a JPG or PNG file.")

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > FACE_MAX_SIDE:
        s = FACE_MAX_SIDE / longest
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


@st.cache_resource(show_spinner=False)
def get_face_cascade():
    import cv2

    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_faces(frame_bgr):
    """
    Pehle aapka original detector (face_recognition / dlib). Wo install na ho
    (online deploy par dlib build karna mushkil hota hai) to OpenCV Haar cascade.
    Return: ([(top, right, bottom, left), ...], detector_name)
    """
    import cv2

    reason = ""
    try:
        import face_recognition

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return face_recognition.face_locations(rgb), "face_recognition (dlib)"
    except (ImportError, SystemExit) as e:
        reason = f"{type(e).__name__}: {e}"

    if not hasattr(cv2, "CascadeClassifier"):
        # OpenCV 5.x me Haar cascade detector nahi hai
        raise RuntimeError(
            f"No face detector available. OpenCV {cv2.__version__} has no CascadeClassifier "
            f"(removed in OpenCV 5) and face_recognition could not be loaded ({reason or 'unknown reason'}). "
            "FIX: in the same Python environment that runs Streamlit, run "
            "`pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python` and then "
            "`pip install \"opencv-python-headless<5\"`, and restart Streamlit."
        )

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    boxes = get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return [(int(y), int(x + w), int(y + h), int(x)) for x, y, w, h in boxes], "OpenCV Haar cascade"


def predict_faces(frame, model, p: dict):
    """
    Aapke original loop jaisa hi: face crop -> gray -> 200x200 -> /255 -> flatten -> model.predict.
    Return: (annotated RGB image, faces list, detector name, kitne boxes mile)
    """
    import cv2

    boxes, detector = detect_faces(frame)
    h, w = frame.shape[:2]
    out = frame.copy()
    scale = max(0.6, min(h, w) / 700)
    thick = max(2, int(round(scale * 2)))
    faces = []

    for top, right, bottom, left in boxes:
        top, left = max(top, 0), max(left, 0)
        bottom, right = min(bottom, h), min(right, w)
        crop = frame[top:bottom, left:right]
        if crop.size == 0 or min(crop.shape[:2]) < 20:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (FACE_SIZE, FACE_SIZE)) / 255
        x = gray.flatten()

        pred = model.predict([x])[0]
        label = face_label(pred, p["label_map"])

        scores = []
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba([x])[0]
                scores = sorted(
                    ((face_label(c, p["label_map"]), float(pr)) for c, pr in zip(model.classes_, probs)),
                    key=lambda s: -s[1],
                )
            except Exception:
                scores = []
        conf = next((s for n, s in scores if n == label), None)

        emoji, color = p["label_style"].get(label.lower(), DEFAULT_STYLE)
        bgr = hex_to_bgr(color)

        # Face par box + upar naam ka tag
        cv2.rectangle(out, (left, top), (right, bottom), bgr, thick)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        y0 = max(top - th - base - 8, 0)
        cv2.rectangle(out, (left, y0), (left + tw + 12, y0 + th + base + 8), bgr, -1)
        cv2.putText(out, label, (left + 6, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thick, cv2.LINE_AA)

        faces.append({
            "label": label, "emoji": emoji, "color": color, "conf": conf, "scores": scores,
            "thumb": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        })

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), faces, detector, len(boxes)


def render_face(p: dict):
    page_hero(p)

    if not require_files([p["files"]["model"]]):
        return

    try:
        import cv2  # noqa: F401
    except ImportError:
        st.error("The `opencv-python` package is not installed. Run this in your terminal:")
        st.code("pip install opencv-python-headless", language="bash")
        return

    model = load_pickle(str(p["files"]["model"]))

    st.info(p["disclaimer"])

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("##### 📸 Your photo")
        tab_cam, tab_up = st.tabs(["📷 Take a photo", "📁 Upload a photo"])
        with tab_cam:
            shot = st.camera_input("Take a photo", key="face_cam", label_visibility="collapsed")
        with tab_up:
            uploaded = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"],
                                        key="face_file", label_visibility="collapsed")

        data = None
        if uploaded is not None:
            data = uploaded.getvalue()
        elif shot is not None:
            data = shot.getvalue()

        analyze = st.button(p["button_text"], key="face_go")

    with right:
        how_panel(p)

    if analyze:
        if not data:
            st.warning("⚠️ Please take a photo or upload an image first.")
            return

        try:
            with st.spinner("Looking for faces..."):
                frame = decode_image(data)
                annotated, faces, detector, n_boxes = predict_faces(frame, model, p)
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            return

        st.markdown("---")

        if not faces:
            st.image(annotated, use_container_width=True)
            st.warning("😕 No face was found. Try a clear, front-facing photo with good light.")
        else:
            st.image(annotated, use_container_width=True)

            m1, m2 = st.columns(2)
            m1.metric("Faces found", len(faces))
            m2.metric("Face detector", detector)

            for i, f in enumerate(faces[:6], start=1):
                c_img, c_res = st.columns([1, 4], gap="medium")
                with c_img:
                    st.image(f["thumb"], use_container_width=True)
                with c_res:
                    sub = (f"Confidence: {f['conf'] * 100:.1f}%" if f["conf"] is not None
                           else "Prediction by your trained model")
                    result_card(f["emoji"], f["color"], f"Face {i}: {f['label']}", sub)
                    prob_bars(f["scores"])
            if len(faces) > 6:
                st.caption(f"Showing details for the first 6 of {len(faces)} faces. "
                           "All faces are marked on the photo above.")

    footer()


# =====================================================================
# 6C. FACE RECOGNITION ATTENDANCE PAGE  (add student -> train -> recognize)
# =====================================================================

ATT_DATA_DIR = Path(__file__).parent / "Addan_data"   # student photos yahan save hoti hain
ATT_MODEL_PATH = MODELS_DIR / "face_attendance" / "model1.pkl"
ATT_IMG_SIZE = (200, 200)
ATT_MIN_IMAGES = 10   # isse kam photos par accuracy kharab hoti hai
ATT_AUTO_COUNT = 100     # auto-capture me default kitni photos
ATT_AUTO_INTERVAL = 0.2  # do photos ke beech ka gap (seconds), taaki angles alag-alag aayein
ATT_STEPS = ["➕ Add Student", "🗑️ Remove Student", "🧠 Train Model", "🎯 Recognize"]
ATT_MODES = ["🎥 Auto capture", "📷 One by one", "🖼️ Upload photos"]
ATT_RTC = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def att_clean_name(name: str) -> str:
    """Folder name safe banao (path tricks jaise ../ hata do)."""
    return re.sub(r"[^\w\- ]", "", name, flags=re.UNICODE).strip()


def att_students():
    if not ATT_DATA_DIR.exists():
        return []
    return sorted(d.name for d in ATT_DATA_DIR.iterdir() if d.is_dir())


def att_count(name: str) -> int:
    folder = ATT_DATA_DIR / name
    return len(list(folder.glob("*.jpg"))) if folder.is_dir() else 0


def att_get_faces(frame_bgr):
    """
    Haar cascade se faces (largest pehle). Return: [(top, right, bottom, left), ...]
    """
    import cv2

    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            f"OpenCV {cv2.__version__} has no CascadeClassifier (removed in OpenCV 5). "
            "FIX: run `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python` "
            "and then `pip install \"opencv-python-headless<5\"`, and restart Streamlit."
        )

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes = get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    return [(int(y), int(x + w), int(y + h), int(x)) for (x, y, w, h) in boxes]


def att_preprocess(face_bgr):
    import cv2

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, ATT_IMG_SIZE) / 255.0
    return gray.flatten()


def att_save_face(frame, name: str) -> bool:
    """Frame ka sabse bada face crop karke save karo. Face mila to True."""
    import cv2

    faces = att_get_faces(frame)
    if not faces:
        return False
    t, r, b, l = faces[0]
    t, l = max(t, 0), max(l, 0)
    folder = ATT_DATA_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    # Naya naam: purani photo kabhi overwrite na ho (delete ke baad bhi)
    idx = att_count(name)
    while (folder / f"img-{idx}.jpg").exists():
        idx += 1
    cv2.imwrite(str(folder / f"img-{idx}.jpg"), frame[t:b, l:r])
    return True


def att_train():
    """Saari student photos par LogisticRegression train karo. Return: (message, level)."""
    import cv2

    X, y = [], []
    for s in att_students():
        for f in (ATT_DATA_DIR / s).glob("*.jpg"):
            img = cv2.imread(str(f))
            if img is None:
                continue
            X.append(att_preprocess(img))
            y.append(s)

    if len(set(y)) < 2:
        return "Need at least 2 students with images to train.", "error"

    X, y = np.array(X), np.array(y)
    names, counts = np.unique(y, return_counts=True)
    few = [f"{n} ({c})" for n, c in zip(names, counts) if c < ATT_MIN_IMAGES]

    acc = None
    # Validation split tabhi mumkin hai jab har student ki kam se kam 2 photos hon
    if counts.min() >= 2 and len(X) >= 10:
        n_test = max(len(names), int(len(X) * 0.2))
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=n_test, stratify=y, random_state=42)
        check = LogisticRegression(max_iter=1000, class_weight="balanced")
        check.fit(X_tr, y_tr)
        acc = check.score(X_te, y_te)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)   # final model saare data par
    ATT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ATT_MODEL_PATH)

    msg = f"Model trained on {len(X)} images of {len(names)} students."
    msg += (f" Validation accuracy: {acc:.2%}" if acc is not None
            else " (Validation skipped: not enough images per student.)")
    if few:
        msg += "  ⚠️ Very few images for: " + ", ".join(few) + ". Add more for better accuracy."
    return msg, "success"


def att_students_panel(p: dict):
    students = att_students()
    if not students:
        return
    rows = "".join(
        H(f"""
        <div class="student-row">
            <div class="student-avatar">{html.escape(s[:1].upper())}</div>
            <div class="student-name">{html.escape(s)}</div>
            <div class="student-count {'low' if att_count(s) < ATT_MIN_IMAGES else ''}">
                {att_count(s)} photos
            </div>
        </div>
        """)
        for s in students
    )
    st.markdown(f"<div class='panel'><h4>👥 Registered students ({len(students)})</h4>{rows}</div>",
                unsafe_allow_html=True)


class AttCaptureProcessor(VideoProcessorBase):
    """
    Live camera frames par chalta hai (alag thread me, isliye yahan koi st.* call nahi).
    Naam set ho aur face dikhe to har ATT_AUTO_INTERVAL second me ek face crop save karta hai,
    jab tak target photos poori na ho jayein.
    """

    def __init__(self):
        import cv2

        self.lock = threading.Lock()
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.name = ""
        self.target = ATT_AUTO_COUNT
        self.saved = 0
        self.last_save = 0.0

    def recv(self, frame):
        import cv2

        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            name, target, saved = self.name, self.target, self.saved

        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        box = max(boxes, key=lambda b: b[2] * b[3]) if len(boxes) else None

        done = saved >= target
        if box is not None and name and not done:
            now = time.time()
            if now - self.last_save >= ATT_AUTO_INTERVAL:
                x, y, w, h = (int(v) for v in box)
                crop = img[max(y, 0):y + h, max(x, 0):x + w]
                if crop.size:
                    folder = ATT_DATA_DIR / name
                    folder.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(folder / f"img-auto-{int(now * 1000)}.jpg"), crop)
                    with self.lock:
                        self.saved += 1
                        saved = self.saved
                    self.last_save = now

        # Overlay: box + status text
        if not name:
            status, color = "Enter a name first", (0, 165, 255)
        elif saved >= target:
            status, color = f"Done! {saved}/{target} - press STOP", (80, 200, 80)
        elif box is None:
            status, color = f"No face found  {saved}/{target}", (60, 60, 255)
        else:
            status, color = f"Capturing {saved}/{target}", (255, 120, 60)
        if box is not None:
            x, y, w, h = (int(v) for v in box)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


@st.fragment(run_every=1)
def att_auto_progress(ctx, target: int, name: str):
    """Har second progress bar update karta hai (sirf ye hissa dobara chalta hai)."""
    notified_key = "att_auto_notified"
    proc = ctx.video_processor if ctx.state.playing else None
    if proc is None:
        st.session_state[notified_key] = False
        st.caption("Press **START** above, allow camera access and slowly turn your face "
                   "left, right, up and down. Photos are captured automatically.")
        return

    saved = proc.saved
    st.progress(min(saved / max(target, 1), 1.0),
                text=f"{saved} / {target} photos captured for {name}")
    if saved >= target and not st.session_state.get(notified_key):
        st.session_state[notified_key] = True
        st.session_state.att_flash = (
            "success", f"✅ {saved} photos captured for {name}. Press STOP, then train the model.")
        st.rerun()   # poori app refresh: student list me nayi count dikhe


def att_mode_auto(name: str):
    if not WEBRTC_OK:
        st.error("Auto capture needs two extra packages. Run this in your terminal and restart Streamlit:")
        st.code("pip install streamlit-webrtc av", language="bash")
        return

    target = int(st.number_input("Photos to capture", min_value=20, max_value=300,
                                 value=ATT_AUTO_COUNT, step=10, key="att_auto_target"))
    ctx = webrtc_streamer(
        key="att_auto_cam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=AttCaptureProcessor,
        rtc_configuration=ATT_RTC,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if ctx.video_processor:
        with ctx.video_processor.lock:   # naam / target live update (rerun par)
            ctx.video_processor.name = name
            ctx.video_processor.target = target
    att_auto_progress(ctx, target, name)
    st.caption("Tip: good light, and move your head slowly. To capture another batch, press STOP and START again.")


def att_mode_single(name: str):
    shot = st.camera_input("Take a photo", key=f"att_cam_{st.session_state.att_cam_key}",
                           label_visibility="collapsed")
    if shot is not None:
        try:
            saved = att_save_face(decode_image(shot.getvalue()), name)
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            saved = None
        if saved:
            st.session_state.att_flash = ("success", "✅ Face saved! Take the next photo.")
            st.session_state.att_cam_key += 1   # camera widget reset
            st.rerun()
        elif saved is False:
            st.error("😕 No face found. Try again with a clear, front-facing photo.")


def att_mode_upload(name: str):
    files = st.file_uploader("Upload multiple photos", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="att_files",
                             label_visibility="collapsed")
    if files and st.button(f"💾 Save {len(files)} uploaded photos", key="att_save_up"):
        ok = 0
        try:
            for f in files:
                ok += bool(att_save_face(decode_image(f.getvalue()), name))
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
        else:
            st.session_state.att_flash = (
                "success", f"✅ Saved {ok}/{len(files)} photos (the rest had no face).")
            st.rerun()


def att_step_add(p: dict):
    st.markdown("##### 👤 Student details")
    raw = st.text_input("Student name", key="att_name", placeholder="e.g. Rahul Sharma")
    name = att_clean_name(raw)
    if raw.strip() and name != raw.strip():
        st.caption(f"Special characters removed. Saving as: **{name}**")

    if not name:
        st.warning("Enter a name to begin.")
        return

    st.info(f"Photos saved for **{name}**: {att_count(name)}. "
            "More photos with different angles and lighting give better accuracy.")

    mode = st.radio("Capture mode", ATT_MODES, horizontal=True, key="att_mode",
                    label_visibility="collapsed")
    if mode == ATT_MODES[0]:
        att_mode_auto(name)
    elif mode == ATT_MODES[1]:
        att_mode_single(name)
    else:
        att_mode_upload(name)


def att_step_remove(p: dict):
    st.markdown("##### 🗑️ Remove a student")
    students = att_students()
    if not students:
        st.info("No students registered yet.")
        return

    sel = st.selectbox("Select student", students, key="att_remove_sel")
    n = att_count(sel)
    st.warning(f"This will permanently delete **{sel}** and all {n} saved photos.")
    confirm = st.checkbox("Yes, I want to delete this student", key=f"att_confirm_{sel}")

    if st.button("🗑️ Remove student", key="att_remove_go", disabled=not confirm):
        # Safety: sirf wahi naam hatao jo sach me Addan_data ke andar registered hai
        if sel in att_students():
            shutil.rmtree(ATT_DATA_DIR / sel, ignore_errors=True)
            msg = f"🗑️ Removed {sel} ({n} photos)."
            if ATT_MODEL_PATH.exists():
                msg += " Train the model again so this student is no longer recognized."
            st.session_state.att_flash = ("success", msg)
            st.rerun()


def att_step_train(p: dict):
    students = att_students()
    st.markdown("##### 🧠 Train the recognizer")
    m1, m2 = st.columns(2)
    m1.metric("Students found", len(students))
    m2.metric("Total photos", sum(att_count(s) for s in students))

    if ATT_MODEL_PATH.exists():
        st.caption("A trained model already exists. Training again will replace it.")

    if st.button("🚀 Train now", key="att_train_go"):
        try:
            with st.spinner("Training..."):
                msg, level = att_train()
        except Exception as e:
            st.error(f"An error occurred during training: {e}")
            return
        getattr(st, level)(msg)


def att_step_predict(p: dict):
    import cv2

    st.markdown("##### 🎯 Who is this?")
    if not ATT_MODEL_PATH.exists():
        st.warning("Train the model first (step 2).")
        return

    model = joblib.load(ATT_MODEL_PATH)   # cache nahi: retrain ke baad hamesha naya model mile
    current = set(att_students())
    stale = [str(c) for c in model.classes_ if str(c) not in current]
    if stale:
        st.warning("The model still knows removed students (" + ", ".join(stale)
                   + "). Train the model again to forget them.")
    shot = st.camera_input("Take a photo to recognize", key="att_pred_cam",
                           label_visibility="collapsed")
    if shot is None:
        return

    try:
        frame = decode_image(shot.getvalue())
        faces = att_get_faces(frame)
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        return

    if not faces:
        st.error("😕 No face detected. Try a clear, front-facing photo with good light.")
        return

    h, w = frame.shape[:2]
    scale = max(0.6, min(h, w) / 700)
    thick = max(2, int(round(scale * 2)))
    c1, c2 = p["colors"]
    bgr = hex_to_bgr(c1)
    results = []

    for t, r, b, l in faces:
        t, l = max(t, 0), max(l, 0)
        b, r = min(b, h), min(r, w)
        crop = frame[t:b, l:r]
        if crop.size == 0:
            continue
        vec = att_preprocess(crop)
        probs = model.predict_proba([vec])[0]
        best = int(np.argmax(probs))
        name, conf = str(model.classes_[best]), float(probs[best])
        scores = sorted(((str(c), float(pr)) for c, pr in zip(model.classes_, probs)),
                        key=lambda s: -s[1])
        results.append((name, conf, scores, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

        label = f"{name} ({conf:.0%})"
        cv2.rectangle(frame, (l, t), (r, b), bgr, thick)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        y0 = max(t - th - base - 8, 0)
        cv2.rectangle(frame, (l, y0), (l + tw + 12, y0 + th + base + 8), bgr, -1)
        cv2.putText(frame, label, (l + 6, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thick, cv2.LINE_AA)

    st.markdown("---")
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    for i, (name, conf, scores, thumb) in enumerate(results[:6], start=1):
        c_img, c_res = st.columns([1, 4], gap="medium")
        with c_img:
            st.image(thumb, use_container_width=True)
        with c_res:
            title = name if len(results) == 1 else f"Face {i}: {name}"
            result_card("🎓", c1, title, f"Confidence: {conf * 100:.1f}%")
            prob_bars(scores)
    if len(results) > 6:
        st.caption(f"Showing details for the first 6 of {len(results)} faces.")


def render_attendance(p: dict):
    page_hero(p)

    try:
        import cv2  # noqa: F401
    except ImportError:
        st.error("The `opencv-python` package is not installed. Run this in your terminal:")
        st.code("pip install opencv-python-headless", language="bash")
        return

    if "att_cam_key" not in st.session_state:
        st.session_state.att_cam_key = 0

    st.info(p["disclaimer"])

    left, right = st.columns([3, 2], gap="large")

    with left:
        # tabs ki jagah radio: ek time par sirf ek camera widget chalta hai
        step = st.radio("Step", ATT_STEPS, horizontal=True, key="att_step",
                        label_visibility="collapsed")
        flash = st.session_state.pop("att_flash", None)
        if flash:
            getattr(st, flash[0])(flash[1])

        if step == ATT_STEPS[0]:
            att_step_add(p)
        elif step == ATT_STEPS[1]:
            att_step_remove(p)
        elif step == ATT_STEPS[2]:
            att_step_train(p)
        else:
            att_step_predict(p)

    with right:
        how_panel(p)
        att_students_panel(p)

    footer()


# =====================================================================
# 6D. OBJECT DETECTION PAGE  (YOLOv4 + OpenCV DNN: photo / live webcam)
# =====================================================================

OBJ_INPUT_SIZES = [320, 416, 512, 608]   # YOLO input 32 ka multiple hona chahiye
OBJ_LOCK = threading.Lock()              # cached net ko ek time par ek hi thread use kare

# Darknet "coco.names": agar objects_list.txt na mile to ye 80 naam use hote hain
COCO_NAMES = (
    "person,bicycle,car,motorbike,aeroplane,bus,train,truck,boat,traffic light,fire hydrant,"
    "stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,"
    "backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,"
    "baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,"
    "fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,"
    "cake,chair,sofa,pottedplant,bed,diningtable,toilet,tvmonitor,laptop,mouse,remote,"
    "keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,"
    "teddy bear,hair drier,toothbrush"
).split(",")


def load_class_names(path) -> list:
    """objects_list.txt (har line me ek naam). File na ho to built-in COCO names."""
    path = Path(path)
    if path.is_file():
        names = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
        names = [n for n in names if n]
        if names:
            return names
    return COCO_NAMES


def obj_color(class_id: int):
    """Har class ka apna fixed rang (BGR)."""
    import colorsys

    r, g, b = colorsys.hsv_to_rgb((class_id * 0.618034) % 1.0, 0.7, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def obj_bgr_to_hex(bgr) -> str:
    return "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])


def build_yolo(weights: str, cfg: str, size: int):
    import cv2

    net = cv2.dnn_DetectionModel(weights, cfg)
    net.setInputSize(size, size)
    net.setInputScale(1 / 255)
    net.setInputSwapRB(True)   # OpenCV BGR deta hai, YOLO RGB par train hua hai (original code me nahi tha)
    return net


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo(weights: str, cfg: str, size: int):
    return build_yolo(weights, cfg, size)


def yolo_detect(net, img, conf: float, nms: float):
    """Return: [(class_id, score, (x, y, w, h)), ...]  (khaali list agar kuch na mile)."""
    ids, scores, boxes = net.detect(img, confThreshold=conf, nmsThreshold=nms)
    if len(ids) == 0:
        return []
    ids = np.array(ids).flatten()
    scores = np.array(scores).flatten()
    return [(int(i), float(sc), tuple(int(v) for v in b)) for i, sc, b in zip(ids, scores, boxes)]


def draw_detections(img, dets, names, allow=None):
    """Boxes + labels banao. Return: (annotated BGR image, [(name, score, bgr), ...])."""
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(h, w) / 800)
    thick = max(2, int(round(scale * 2)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    shown = []

    for cid, score, (x, y, bw, bh) in dets:
        name = names[cid] if 0 <= cid < len(names) else f"class {cid}"
        if allow and name not in allow:
            continue
        col = obj_color(cid)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), col, thick)

        label = f"{name} {score:.0%}"
        fs, ft = scale * 0.7, max(1, thick - 1)
        (tw, th), base = cv2.getTextSize(label, font, fs, ft)
        y0 = max(y - th - base - 6, 0)
        cv2.rectangle(out, (x, y0), (x + tw + 10, y0 + th + base + 6), col, -1)
        cv2.putText(out, label, (x + 5, y0 + th + 2), font, fs, (0, 0, 0), ft, cv2.LINE_AA)
        shown.append((name, score, col))
    return out, shown


class YoloProcessor(VideoProcessorBase):
    """Live webcam: har frame par YOLO chalata hai (alag thread me, isliye yahan koi st.* call nahi)."""

    def __init__(self, weights, cfg, names, size):
        self.lock = threading.Lock()
        self.net = build_yolo(weights, cfg, size)   # apna net: cached wale se alag
        self.names = names
        self.size = size
        self.conf, self.nms, self.allow = 0.5, 0.5, set()
        self.found = 0
        self.counts = {}
        self.fps = 0.0
        self._t = time.time()

    def recv(self, frame):
        import cv2

        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            conf, nms, allow, size = self.conf, self.nms, set(self.allow), self.size

        try:
            self.net.setInputSize(size, size)
            dets = yolo_detect(self.net, img, conf, nms)
        except Exception:
            dets = []
        out, shown = draw_detections(img, dets, self.names, allow)

        now = time.time()
        inst = 1.0 / max(now - self._t, 1e-3)
        self._t = now
        self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst

        counts = {}
        for name, _, _ in shown:
            counts[name] = counts.get(name, 0) + 1
        with self.lock:
            self.found, self.counts = len(shown), counts

        cv2.putText(out, f"{self.fps:4.1f} FPS | {len(shown)} objects", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(out, format="bgr24")


def obj_count_panel(shown):
    """Class-wise count bars + chips."""
    counts = {}
    colors = {}
    for name, _, col in shown:
        counts[name] = counts.get(name, 0) + 1
        colors[name] = col
    if not counts:
        return
    top = max(counts.values())
    rows = "".join(
        H(f"""
        <div class="bar-row">
            <span class="bar-name">{html.escape(n)}</span>
            <div class="bar"><div class="bar-fill" style="width:{c / top * 100:.0f}%;
                background:{obj_bgr_to_hex(colors[n])}"></div></div>
            <span class="bar-pct">×{c}</span>
        </div>
        """)
        for n, c in sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    )
    st.markdown(
        f"<div class='panel'><div class='panel-title'><span class='dot'></span>OBJECTS DETECTED</div>{rows}</div>",
        unsafe_allow_html=True,
    )


@st.fragment(run_every=1)
def obj_live_stats(ctx):
    """Har second live numbers (sirf ye hissa dobara chalta hai)."""
    proc = ctx.video_processor if ctx.state.playing else None
    if proc is None:
        st.caption("Press **START** above and allow camera access. Detection runs on every frame.")
        return
    with proc.lock:
        found, counts = proc.found, dict(proc.counts)
    m1, m2 = st.columns(2)
    m1.metric("Objects in frame", found)
    m2.metric("Speed", f"{proc.fps:.1f} FPS")
    if counts:
        st.markdown(chips(f"{n} ×{c}" for n, c in sorted(counts.items(), key=lambda kv: -kv[1])),
                    unsafe_allow_html=True)


OBJ_MODES = ["📷 Take photo", "📁 Upload image", "🎥 Live webcam"]


def render_objects(p: dict):
    page_hero(p)

    f = p["files"]
    if not require_files([f["weights"], f["cfg"]]):
        return

    # Git LFS ke bina upload karne par .weights ki jagah chhoti text file aa jaati hai
    if Path(f["weights"]).stat().st_size < 1_000_000:
        st.error("The weights file is very small, so it is probably not the real model "
                 "(a Git LFS pointer, or a failed upload). Re-upload the full .weights file.")
        return

    try:
        import cv2  # noqa: F401
    except ImportError:
        st.error("The `opencv-python` package is not installed. Run this in your terminal:")
        st.code("pip install opencv-python-headless", language="bash")
        return

    if int(cv2.__version__.split(".")[0]) >= 5:
        # OpenCV 5 ne Darknet (YOLOv4 .cfg/.weights) importer hata diya hai
        st.error(f"OpenCV {cv2.__version__} cannot load YOLOv4 .weights files (Darknet support was "
                 "removed in OpenCV 5). In the same Python environment that runs Streamlit, run:")
        st.code('pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python\n'
                'pip install "opencv-python-headless<5"', language="bash")
        return

    names = load_class_names(f["names"])
    st.info(p["disclaimer"])

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("##### ⚙️ Detection settings")
        c1, c2, c3 = st.columns(3)
        conf = c1.slider("Confidence", 0.10, 0.95, 0.50, 0.05, key="obj_conf",
                         help="Higher = fewer but surer detections.")
        nms = c2.slider("Overlap (NMS)", 0.10, 0.95, 0.50, 0.05, key="obj_nms",
                        help="Lower = removes duplicate boxes more strongly.")
        size = c3.select_slider("Input size", OBJ_INPUT_SIZES, value=416, key="obj_size",
                                help="Bigger = more accurate but slower.")
        allow = st.multiselect("Only show these objects (empty = show all)", names, key="obj_allow")

        mode = st.radio("Input", OBJ_MODES, horizontal=True, key="obj_mode",
                        label_visibility="collapsed")

        data = None
        analyze = False
        if mode == OBJ_MODES[0]:
            shot = st.camera_input("Take a photo", key="obj_cam", label_visibility="collapsed")
            data = shot.getvalue() if shot is not None else None
            analyze = st.button(p["button_text"], key="obj_go_cam")
        elif mode == OBJ_MODES[1]:
            up = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"],
                                  key="obj_file", label_visibility="collapsed")
            data = up.getvalue() if up is not None else None
            analyze = st.button(p["button_text"], key="obj_go_up")

    with right:
        how_panel(p)

    # ---------------- Live webcam ----------------
    if mode == OBJ_MODES[2]:
        st.markdown("---")
        if not WEBRTC_OK:
            st.error("Live webcam needs two extra packages. Run this and restart Streamlit:")
            st.code("pip install streamlit-webrtc av", language="bash")
        else:
            ctx = webrtc_streamer(
                key="obj_live",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: YoloProcessor(str(f["weights"]), str(f["cfg"]), names, size),
                rtc_configuration=ATT_RTC,
                media_stream_constraints={"video": {"width": {"ideal": 640}}, "audio": False},
                async_processing=True,
            )
            if ctx.video_processor:
                with ctx.video_processor.lock:   # sliders live apply hote hain
                    ctx.video_processor.conf = conf
                    ctx.video_processor.nms = nms
                    ctx.video_processor.allow = set(allow)
                    ctx.video_processor.size = size
            obj_live_stats(ctx)
            st.caption("Tip: the full YOLOv4 is heavy on a CPU. For smoother video use the "
                       "yolov4-tiny files, or a smaller input size.")
        footer()
        return

    # ---------------- Photo / upload ----------------
    if analyze:
        if not data:
            st.warning("⚠️ Please take a photo or upload an image first.")
            return
        try:
            with st.spinner("Detecting objects..."):
                img = decode_image(data)
                net = load_yolo(str(f["weights"]), str(f["cfg"]), size)
                t0 = time.perf_counter()
                with OBJ_LOCK:
                    dets = yolo_detect(net, img, conf, nms)
                ms = (time.perf_counter() - t0) * 1000
                annotated, shown = draw_detections(img, dets, names, set(allow))
        except Exception as e:
            st.error(f"An error occurred during processing: {e}  "
                     "(Check that the .cfg and .weights files belong to the same model.)")
            return

        st.markdown("---")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Objects found", len(shown))
        m2.metric("Different types", len({n for n, _, _ in shown}))
        m3.metric("Inference time", f"{ms:.0f} ms")

        if not shown:
            st.warning("😕 No objects found. Try a lower confidence value or a clearer photo.")
        else:
            obj_count_panel(shown)
            with st.expander("📋 All detections"):
                st.dataframe(
                    [{"object": n, "confidence": f"{sc:.1%}"} for n, sc, _ in
                     sorted(shown, key=lambda t: -t[1])],
                    use_container_width=True,
                )
            ok, buf = cv2.imencode(".png", annotated)
            if ok:
                st.download_button("⬇️ Download result image", buf.tobytes(),
                                   file_name="detected.png", mime="image/png")

    footer()


# =====================================================================
# 7. PROJECTS REGISTRY  (naya project yahan add karein)
# =====================================================================

PROJECTS = {
    # ----------------------------------------------------------------
    "food": {
        "key": "food",
        "icon": "🍽️",
        "title": "Food Review System",
        "tagline": "Instantly understand what customers feel about the food.",
        "desc": "Sentiment analysis on restaurant reviews written in Hinglish or English. "
                "Positive reviews bring balloons, negative ones bring red balloons.",
        "colors": ("#ff7a18", "#e63946"),
        "tags": ["NLP", "Sentiment Analysis", "Hinglish & English", "Scikit-learn"],
        "render": render_classifier,
        "files": {
            "model": pick(MODELS_DIR / "food_review" / "final_rev_data.pkl",
                          LEGACY_DIR / "final_rev_data.pkl"),
        },
        "required": ["model"],
        "label_style": {
            "positive": ("😊", "#2ecc71"),
            "negative": ("😞", "#ff4d4d"),
            "neutral": ("😐", "#f4b400"),
        },
        "input_label": "Enter a restaurant review (Hinglish or English)",
        "placeholder": "e.g. The food was amazing and the service was excellent!",
        "button_text": "🔍 Analyze Review",
        "examples": [
            "Khana bahut tasty tha aur service bhi zabardast thi!",
            "Bilkul bakwaas food tha, ek ghanta wait karwaya aur thanda khana diya.",
            "Food theek-thaak tha, kuch khaas nahi.",
        ],
        "how": [
            "Write a review in Hinglish or English.",
            "Your trained model processes the text.",
            "It predicts positive, negative or neutral.",
            "Positive gets balloons 🎈, negative gets red balloons.",
        ],
        "on_result": food_celebration,
    },
    # ----------------------------------------------------------------
    "movie": {
        "key": "movie",
        "icon": "🎬",
        "title": "Movie Recommendation System",
        "tagline": "Pick a movie you love and discover similar ones.",
        "desc": "Content-based recommender using NLP vectors and nearest neighbours, "
                "with live movie posters and movie poster aceess without API.",
        "colors": ("#e50914", "#7f1d9d"),
        "tags": ["NLP", "Recommender", "Nearest Neighbors" ,"API", "Pandas"],
        "render": render_movie,
        "files": {
            "model": pick(MODELS_DIR / "movie_recommender" / "model.pkl", LEGACY_DIR / "model.pkl"),
            "vectors": pick(MODELS_DIR / "movie_recommender" / "vectors.pkl", LEGACY_DIR / "vectors.pkl"),
            "dataframe": pick(MODELS_DIR / "movie_recommender" / "dataframe.pkl", LEGACY_DIR / "dataframe.pkl"),
        },
        "required": ["model", "vectors", "dataframe"],
    },
    # ----------------------------------------------------------------
    "speech": {
        "key": "speech",
        "icon": "🎤",
        "title": "Speech Mental Health Detection",
        "tagline": "Your voice carries signals. Let the model listen.",
        "desc": "Analyzes a voice clip (tone, pitch, loudness, pauses, MFCCs) to estimate whether "
                "the speaker sounds Normal, Anxious or Depressed. For awareness only.",
        "colors": ("#06b6d4", "#8b5cf6"),
        "tags": ["Speech", "Audio ML", "Librosa", "MFCC", "Scikit-learn"],
        "render": render_speech,
        "files": {
            "model": pick(MODELS_DIR / "speech_mental_health" / "best_model.pkl",
                          MODELS_DIR / "best_model.pkl"),
            "label_encoder": pick(MODELS_DIR / "speech_mental_health" / "label_encoder.pkl",
                                  MODELS_DIR / "label_encoder.pkl"),
        },
        "required": ["model"],   # label_encoder na ho to app khud bana leta hai
        "label_style": {
            "normal": ("😊", "#2ecc71"),
            "anxious": ("😟", "#f4b400"),
            "depressed": ("😔", "#6c8cff"),
        },
        "disclaimer": "This tool is for awareness and learning only. It is NOT a medical diagnosis. "
                      "If you are struggling, please talk to a professional or someone you trust. "
                      "Your audio is processed only for this prediction.",
        "button_text": "🎤 Analyze Speech",
        "how": [
            "Enter your age and upload or record a short voice clip.",
            "Audio features are extracted (pitch, loudness, pauses, MFCCs).",
            "The trained model predicts Normal, Anxious or Depressed.",
            "You get the result with supportive guidance.",
        ],
    },
    # ----------------------------------------------------------------
    "news": {
        "key": "news",
        "icon": "📰",
        "title": "Fake News Detection",
        "tagline": "Paste a news article and find out if it looks real or fake.",
        "desc": "NLP text classifier that reads a news article and predicts whether it is "
                "real or fake. Built to fight misinformation, not to replace fact-checking.",
        "colors": ("#f59e0b", "#2563eb"),
        "tags": ["NLP", "Fake News", "Text Classification", "Scikit-learn"],
        "render": render_classifier,
        "files": {
            "model": pick(MODELS_DIR /  "Fakenews.pkl",
                          LEGACY_DIR / "Fakenews.pkl"),
        },
        "required": ["model"],
        "label_style": {
            "fake": ("🚨", "#ff4d4d"),
            "false": ("🚨", "#ff4d4d"),
            "real": ("✅", "#2ecc71"),
            "true": ("✅", "#2ecc71"),
        },
        "label_alias": {"false": "Fake", "true": "Real"},
        "result_suffix": "News",
        "empty_msg": "⚠️ Please enter a news article first.",
        "input_label": "Paste a news article (English)",
        "placeholder": "e.g. The government has announced new policies to boost the economy.",
        "button_text": "🔍 Classify Article",
        "examples": [
            "The government has announced new policies to boost the economy and create jobs.",
            "SHOCKING: Scientists confirm that drinking hot water cures every disease overnight!",
        ],
        "how": [
            "Paste a news article in English.",
            "The text is converted into numerical features.",
            "The trained model predicts real or fake.",
            "You see the result with a confidence score.",
        ],
        "on_result": news_extras,
    },
    # ----------------------------------------------------------------
    "face": {
        "key": "face",
        "icon": "👤",
        "title": "Face Gender Detection",
        "tagline": "Take a photo and let the model find faces and predict gender.",
        "desc": "Computer-vision project: detects every face in a photo, converts it to a 200x200 "
                "grayscale vector and predicts Female or Male with your trained model.",
        "colors": ("#14b8a6", "#ec4899"),
        "tags": ["Computer Vision", "OpenCV", "Face Detection", "Scikit-learn"],
        "render": render_face,
        "files": {
            "model": pick(MODELS_DIR / "face_gender_model.pkl",
                          MODELS_DIR / "face_gender" / "model.pkl",
                          LEGACY_DIR / "face_gender_model.pkl"),
        },
        "required": ["model"],
        "label_map": {0: "Female", 1: "Male"},   # aapke original code jaisa: 0 = Female, baaki = Male
        "label_style": {
            "female": ("👩", "#ec4899"),
            "male": ("👨", "#3b82f6"),
        },
        "disclaimer": "This is a learning demo. Predictions can be wrong, especially in low light, "
                      "with side views or partly hidden faces. Your photo is used only for this "
                      "prediction and is not stored.",
        "button_text": "🔍 Detect Faces",
        "how": [
            "Take a photo with your camera or upload a clear image.",
            "Faces are found in the photo and cropped.",
            "Each face becomes a 200x200 grayscale vector for the model.",
            "You see boxes and Female / Male labels on the photo.",
        ],
    },
    # ----------------------------------------------------------------
    "attendance": {
        "key": "attendance",
        "icon": "🎓",
        "title": "Face Recognition Attendance",
        "tagline": "Add students, train the model and recognize who is in front of the camera.",
        "desc": "End-to-end face recognition: register students with camera or uploaded photos, "
                "train a Logistic Regression model in one click and identify faces with confidence scores.",
        "colors": ("#6366f1", "#f43f5e"),
        "tags": ["Computer Vision", "OpenCV", "Face Recognition", "Logistic Regression"],
        "render": render_attendance,
        "files": {"model": ATT_MODEL_PATH},
        "required": [],   # model app ke andar train hota hai, isliye file ki zarurat nahi
        "disclaimer": "This is a learning demo. Face photos you add are stored in the 'Addan_data' "
                      "folder next to this app, so only register people who have agreed to it. "
                      "Accuracy depends on how many clear photos each student has.",
        "how": [
            "Add Student: enter a name, press START and 100 face photos are captured automatically.",
            "Remove Student: delete a student and all their photos anytime.",
            "Train Model: one click trains a Logistic Regression on all photos.",
            "Recognize: take a photo and see the name with a confidence score.",
            "Faces are marked with a box and name on the photo.",
        ],
    },
    # ----------------------------------------------------------------
    "objects": {
        "key": "objects",
        "icon": "🎯",
        "title": "Object Detection (YOLOv4)",
        "tagline": "Point the camera or upload a photo and watch YOLO find everything in it.",
        "desc": "Real-time object detection with YOLOv4 and OpenCV DNN. Detects 80 everyday object "
                "types (person, car, phone, dog...) in a photo or live webcam, with adjustable "
                "confidence, class filter and per-object counts.",
        "colors": ("#f43f5e", "#f59e0b"),
        "tags": ["Computer Vision", "YOLOv4", "OpenCV DNN", "Real-time"],
        "render": render_objects,
        "files": {
            "weights": pick(MODELS_DIR / "object_detection" / "yolov4.weights",
                            MODELS_DIR / "object_detection" / "yolov4-tiny.weights"),
            "cfg": pick(MODELS_DIR / "object_detection" / "yolov4.cfg",
                        MODELS_DIR / "object_detection" / "yolov4-tiny.cfg"),
            "names": pick(MODELS_DIR / "object_detection" / "objects_list.txt",
                          MODELS_DIR / "object_detection" / "coco.names"),
        },
        "required": ["weights", "cfg"],   # names na ho to 80 COCO naam built-in hain
        "disclaimer": "This is a learning demo. YOLO can miss small or blurry objects and sometimes "
                      "mislabels things. Your photo is used only for this prediction and is not stored. "
                      "The live webcam needs an HTTPS address when deployed.",
        "button_text": "🔍 Detect Objects",
        "how": [
            "Take a photo, upload an image or start the live webcam.",
            "The image is resized to 416x416 and scaled to 0-1 for YOLOv4.",
            "The network predicts boxes, and overlapping ones are merged (NMS).",
            "You see coloured boxes, labels and a count of every object type.",
        ],
    },
    # ----------------------------------------------------------------
    # NAYA PROJECT ADD KARNA HO TO IS TEMPLATE KO COPY KARO:
    #
    # "spam": {
    #     "key": "spam", "icon": "📧", "title": "Spam Detector",
    #     "tagline": "Is this message spam?", "desc": "Short description.",
    #     "colors": ("#00c853", "#00838f"),
    #     "tags": ["NLP", "Classification"],
    #     "render": render_classifier,
    #     "files": {"model": MODELS_DIR / "spam" / "model.pkl"},
    #     "required": ["model"],
    #     "label_style": {"spam": ("🚫", "#ff4d4d"), "ham": ("✅", "#2ecc71")},
    #     "input_label": "Paste a message", "placeholder": "Congratulations! You won...",
    #     "button_text": "🔍 Check", "examples": [], "how": ["Step 1", "Step 2"],
    # },
}


def project_ready(p: dict) -> bool:
    return files_ready([p["files"][k] for k in p.get("required", ["model"])])


# =====================================================================
# 8. NAVIGATION
# =====================================================================

HOME = "🏠 Home"
LABELS = {HOME: None}
for k, p in PROJECTS.items():
    LABELS[f"{p['icon']} {p['title']}"] = k

if "page" not in st.session_state:
    st.session_state.page = HOME

with st.sidebar:
    initials = "".join(w[0] for w in PROFILE["name"].split()[:2]).upper()
    st.markdown(
        H(f"""
        <div class="side-logo">
            <div class="avatar"><span>{html.escape(initials)}</span></div>
            <div class="name">{html.escape(PROFILE['name'])}</div>
            <div class="role">&gt; {html.escape(PROFILE['role'])}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )
    st.markdown("<div class='side-label'>NAVIGATION</div>", unsafe_allow_html=True)
    st.radio("Navigation", list(LABELS), key="page", label_visibility="collapsed")
    st.markdown("<div class='side-label'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    for p in PROJECTS.values():
        ok = project_ready(p)
        st.markdown(
            f"<div class='side-status'><span class='led {'on' if ok else 'off'}'></span>"
            f"{html.escape(p['title'])}"
            f"<span class='side-state'>{'READY' if ok else 'NO MODEL'}</span></div>",
            unsafe_allow_html=True,
        )

# =====================================================================
# 9. HOME PAGE
# =====================================================================


def render_home():
    theme("#8b5cf6", "#22d3ee")

    st.markdown(
        H(f"""
        <div class="hero">
            <div class="term-bar"><i></i><i></i><i></i><b>{html.escape(PROFILE['name'].split()[0].lower())}@ai-lab: ~/portfolio</b></div>
            <div class="hero-body">
                <div class="hero-left">
                    <div class="hero-cmd">$ <span class="typing">python run_portfolio.py --live</span></div>
                    <div class="hero-hi">Hello world, I'm</div>
                    <h1>{html.escape(PROFILE['name'])}</h1>
                    <div class="hero-role">&gt; {html.escape(PROFILE['role'])}</div>
                    <p>{html.escape(PROFILE['about'])}</p>
                    <div style="margin-top:14px">{chips(PROFILE['skills'])}</div>
                </div>
                <div class="hero-right">{neural_svg()}</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    ready = sum(project_ready(p) for p in PROJECTS.values())
    stats = [
        (f"{len(PROJECTS):02d}", "AI projects"),
        (f"{ready:02d}", "Live demos"),
        ("03", "Text · Audio · Vision"),
        ("100%", "Python stack"),
    ]
    st.markdown(
        H("<div class='stat-row'>" + "".join(
            f"<div class='stat'><div class='v'>{v}</div><div class='l'>{html.escape(l)}</div></div>"
            for v, l in stats) + "</div>"),
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'><span class='mono'>//</span> Featured Projects</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Open a project to try the live model demo.</div>",
                unsafe_allow_html=True)

    items = list(PROJECTS.items())
    for start in range(0, len(items), 3):
        cols = st.columns(3, gap="large")
        chunk = items[start:start + 3]
        for col, (k, p) in zip(cols, chunk):
            c1, c2 = p["colors"]
            ok = project_ready(p)
            num = list(PROJECTS).index(k) + 1
            badge = "<span class='status ok'>● LIVE</span>" if ok else \
                    "<span class='status bad'>● NO MODEL</span>"
            with col:
                st.markdown(
                    H(f"""
                    <div class="proj-card" style="--c1:{c1}; --c2:{c2};">
                        <div class="proj-top"><div class="proj-icon">{p['icon']}</div>{badge}</div>
                        <div class="proj-id">MODEL_{num:02d} · {html.escape(p['tags'][0].upper())}</div>
                        <h3>{html.escape(p['title'])}</h3>
                        <p>{html.escape(p['desc'])}</p>
                        <div>{chips(p['tags'])}</div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )
                st.button(
                    "Open project →", key=f"open_{k}",
                    on_click=go, args=(f"{p['icon']} {p['title']}",),
                )

        # Last row me khaali jagah par "coming soon" card
        if len(chunk) < 3:
            with cols[len(chunk)]:
                st.markdown(
                    "<div class='soon-card'><div class='big'>✨</div>"
                    "<b>More projects coming soon</b>"
                    "<span style='font-size:13px;margin-top:4px'>Stay tuned!</span></div>",
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    flow = [("01", "Data", "Collect & clean"), ("02", "Features", "Vectorize / extract"),
            ("03", "Model", "Train & evaluate"), ("04", "Serve", "Streamlit interface"),
            ("05", "Demo", "Try it live")]
    st.markdown("<div class='section-title'><span class='mono'>//</span> How I ship a model</div>",
                unsafe_allow_html=True)
    st.markdown(
        H("<div class='flow'>" + "".join(
            f"<div class='flow-node'><div class='n'>{n}</div><div class='t'>{t}</div>"
            f"<div class='s'>{s}</div></div>" for n, t, s in flow) + "</div>"),
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'><span class='mono'>//</span> Get in touch</div>",
                unsafe_allow_html=True)
    link_cols = st.columns(len(PROFILE["links"]))
    for col, (name, url) in zip(link_cols, PROFILE["links"].items()):
        with col:
            st.link_button(name, url, use_container_width=True)

    footer()


# =====================================================================
# 10. ROUTER
# =====================================================================

page = LABELS[st.session_state.page]

if page is None:
    render_home()
else:
    proj = PROJECTS[page]
    theme(*proj["colors"])
    proj["render"](proj)

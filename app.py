"""
AI / NLP PORTFOLIO  -  Streamlit

Projects:
  1. Food Review System              (text  -> sentiment)
  2. Movie Recommendation System     (movie -> similar movies + posters)
  3. Speech Mental Health Detection  (audio -> Normal / Anxious / Depressed)
  4. Fake News Detection             (text  -> Real / Fake)
  5. Face Gender Detection           (photo -> detect faces -> Female / Male)

Naya project add karna ho to file ke neeche PROJECTS dict me entry add karein.

Folder structure:
    Project/
    ├── app.py
    ├── requirements.txt
    └── models/
        ├── food_review/
        │   └── final_rev_data.pkl
        ├── movie_recommender/
        │   ├── model.pkl
        │   ├── vectors.pkl
        │   └── dataframe.pkl
        ├── speech_mental_health/
        │   ├── best_model.pkl
        │   └── label_encoder.pkl      (optional, na ho to app khud bana lega)
        ├── news_classifier/
        │   └── Fakenews.pkl
        └── face_gender_model.pkl      (ya face_gender/model.pkl)
"""

import html
import io
import re
import zlib
from pathlib import Path

import joblib
import numpy as np
import requests
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# =====================================================================
# 1. SETTINGS  (yahan apni details badlein)
# =====================================================================

MODELS_DIR = Path(__file__).parent / "models"


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
st.sidebar.image(
    "https://images.nightcafe.studio/galleries/GyzsoaOdxRKFcwbJD0sj/an_abstract_representation_of_artificial--1--gr9zw.jpg"
)

st.set_page_config(
    page_title=f"{PROFILE['name']} | AI Portfolio",
    layout="wide",
)

# =====================================================================
# 2. HELPERS
# =====================================================================


def H(s: str) -> str:
    """Multi-line HTML ko ek line me badalta hai (markdown code-block bug se bachne ke liye)."""
    return " ".join(line.strip() for line in s.strip().splitlines())


def chips(items):
    return "".join(f'<span class="chip">{html.escape(str(t))}</span>' for t in items)


def pick(*paths) -> Path:
    """Diye gaye paths me se pehla jo exist kare wo lo, warna pehla path (error me dikhane ke liye)."""
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return Path(paths[0])


@st.cache_resource(show_spinner="Loading model...")
def load_pickle(path_str: str):
    return joblib.load(path_str)


def files_ready(paths) -> bool:
    return all(Path(p).exists() for p in paths)


def require_files(paths) -> bool:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        st.error("Model file not found. Please place it at this path:")
        st.code("\n".join(missing), language="text")
        return False
    return True


def go(label: str):
    st.session_state.page = label


def set_text(key: str, value: str):
    st.session_state[key] = value


def theme(c1: str, c2: str):
    st.markdown(f"<style>:root{{--c1:{c1};--c2:{c2};}}</style>", unsafe_allow_html=True)


def page_hero(p: dict):
    st.markdown(
        H(f"""
        <div class="page-hero">
            <div class="page-icon">{p['icon']}</div>
            <div>
                <h1>{html.escape(p['title'])}</h1>
                <p>{html.escape(p['tagline'])}</p>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def how_panel(p: dict):
    steps = "".join(f"<li>{html.escape(s)}</li>" for s in p["how"])
    st.markdown(
        H(f"""
        <div class="panel">
            <h4>⚙️ How it works</h4>
            <ol>{steps}</ol>
            <div style="margin-top:8px">{chips(p['tags'])}</div>
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
    st.markdown(f"<div class='panel'><h4>📊 Probability breakdown</h4>{bars}</div>",
                unsafe_allow_html=True)


def footer():
    st.markdown(
        f"<div class='footer'>Built with ❤️ by {html.escape(PROFILE['name'])} · "
        "Python · Scikit-learn · Streamlit</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# 3. CSS
# =====================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

:root { --c1:#7f5af0; --c2:#2cb67d; }

.stApp, .stApp p, .stApp label, .stApp button, .stApp textarea,
.stApp input, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li {
    font-family: 'Poppins', sans-serif;
}
.stApp { background: #0b0b12; color: #f2f2f7; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: #10101a; border-right: 1px solid #1f1f2e; }
[data-testid="stSidebar"] * { color: #ffffff; }
.side-logo { text-align:center; padding: 8px 0 4px 0; }
.side-logo .emoji { font-size: 46px; }
.side-logo .name { font-size: 20px; font-weight: 700; }
.side-logo .role { font-size: 12px; color: #9a9ab0 !important; }
.side-status { font-size: 13px; padding: 3px 0; color:#c9c9d8; }

/* ---------- Home hero ---------- */
.hero {
    padding: 50px 40px; border-radius: 24px; margin-bottom: 28px;
    background:
        radial-gradient(circle at 15% 20%, rgba(127,90,240,0.45), transparent 45%),
        radial-gradient(circle at 85% 80%, rgba(44,182,125,0.35), transparent 45%),
        #14141f;
    border: 1px solid #25253a;
}
.hero-hi { font-size: 18px; color: #b9b9d0; }
.hero h1 {
    font-size: 58px; font-weight: 800; margin: 0; line-height: 1.1;
    background: linear-gradient(90deg, #ffffff, #b69cff, #5eeaa8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-role { font-size: 22px; font-weight: 600; color: #5eeaa8; margin: 6px 0 14px 0; }
.hero p { font-size: 16px; color: #d3d3e3; max-width: 720px; line-height: 1.7; }

.chip {
    display: inline-block; padding: 5px 13px; margin: 4px 6px 0 0;
    border-radius: 999px; font-size: 12.5px; font-weight: 500;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
    color: #f2f2f7;
}

.section-title { font-size: 26px; font-weight: 700; margin: 26px 0 4px 0; }
.section-sub { color: #9a9ab0; margin-bottom: 14px; }

/* ---------- Project cards (home) ---------- */
.proj-card {
    position: relative; overflow: hidden;
    background: #14141f; border: 1px solid #25253a; border-radius: 20px;
    padding: 22px; min-height: 300px; transition: .3s;
}
.proj-card::before {
    content:""; position:absolute; top:0; left:0; right:0; height:5px;
    background: linear-gradient(90deg, var(--c1), var(--c2));
}
.proj-card:hover { transform: translateY(-5px); border-color: var(--c1); box-shadow: 0 14px 34px rgba(0,0,0,.5); }
.proj-top { display:flex; justify-content:space-between; align-items:center; }
.proj-icon {
    width: 56px; height: 56px; border-radius: 16px; font-size: 28px;
    display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg, var(--c1), var(--c2));
}
.proj-card h3 { margin: 14px 0 6px 0; font-size: 20px; font-weight: 700; }
.proj-card p { color:#b4b4c8; font-size: 14px; line-height: 1.6; }
.status { font-size: 11.5px; font-weight: 600; padding: 4px 10px; border-radius: 999px; }
.status.ok  { background: rgba(44,182,125,.18); color:#4ade9a; }
.status.bad { background: rgba(255,90,90,.16);  color:#ff8080; }

.soon-card {
    min-height: 300px; border: 2px dashed #2f2f48; border-radius: 20px; padding: 22px;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    text-align:center; color:#8d8da6;
}
.soon-card .big { font-size: 40px; margin-bottom: 8px; }

/* ---------- Project page hero ---------- */
.page-hero {
    display:flex; align-items:center; gap: 20px;
    padding: 30px 32px; border-radius: 22px; margin-bottom: 22px;
    background: linear-gradient(120deg, var(--c1), var(--c2));
    box-shadow: 0 10px 30px rgba(0,0,0,.4);
}
.page-icon {
    font-size: 48px; width: 82px; height: 82px; border-radius: 22px;
    display:flex; align-items:center; justify-content:center;
    background: rgba(0,0,0,.25);
}
.page-hero h1 { margin:0; font-size: 34px; font-weight: 800; color:#fff; }
.page-hero p  { margin:4px 0 0 0; color: rgba(255,255,255,.9); font-size: 15px; }

/* ---------- Panels ---------- */
.panel {
    background: #14141f; border: 1px solid #25253a; border-radius: 18px;
    padding: 20px 22px; margin-bottom: 14px;
}
.panel h4 { margin: 0 0 10px 0; font-size: 17px; }
.panel li, .panel p { color:#c3c3d6; font-size: 14px; line-height: 1.7; }

/* ---------- Result card ---------- */
.result-card {
    display:flex; align-items:center; gap: 18px;
    padding: 22px 26px; border-radius: 20px; margin: 10px 0 14px 0;
    background: linear-gradient(120deg, color-mix(in srgb, var(--rc) 28%, #14141f), #14141f);
    border: 1px solid var(--rc);
}
.result-emoji { font-size: 54px; }
.result-label { font-size: 30px; font-weight: 800; color: var(--rc); line-height: 1.1; }
.result-sub { color:#c3c3d6; font-size: 14px; margin-top: 4px; }

.bar-row { display:flex; align-items:center; gap: 12px; margin: 9px 0; }
.bar-name { width: 150px; font-size: 14px; color:#dcdce8; }
.bar { flex: 1; height: 11px; background:#232336; border-radius: 99px; overflow:hidden; }
.bar-fill { height:100%; border-radius: 99px; background: linear-gradient(90deg, var(--c1), var(--c2)); }
.bar-pct { width: 52px; text-align:right; font-size: 13px; color:#b4b4c8; }

/* ---------- Support boxes ---------- */
.support-box {
    padding: 16px 20px; border-radius: 16px; margin-top: 10px;
    background: rgba(94,150,255,.10); border: 1px solid rgba(94,150,255,.35);
    color:#dbe6ff; font-size: 14.5px; line-height: 1.7;
}

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
    background:#fff; color: var(--m1); font-size:20px; font-weight:800;
    display:flex; align-items:center; justify-content:center;
    box-shadow: 0 3px 10px rgba(0,0,0,.5); z-index:2;
}
.poster-title {
    margin-top:10px; min-height:60px; padding:8px 10px; border-radius:10px;
    background: rgba(0,0,0,.45); display:flex; align-items:center; justify-content:center;
    text-align:center; font-size:16px; font-weight:700; color:#fff; line-height:1.3;
    text-shadow: 0 1px 4px rgba(0,0,0,.8);
}

/* ---------- Inputs & buttons ---------- */
.stButton > button {
    width:100%; height:46px; border:none; border-radius:12px;
    background: linear-gradient(90deg, var(--c1), var(--c2));
    color:#fff; font-weight:600; font-size:15px; transition:.25s;
}
.stButton > button:hover { filter: brightness(1.12); transform: translateY(-1px); color:#fff; }
.stTextArea textarea, .stTextInput input, .stNumberInput input {
    background:#14141f; color:#fff; border:1px solid #2a2a40; border-radius:12px; font-size:15px;
}
.stTextArea textarea:focus { border-color: var(--c1); }
div[data-baseweb="select"] > div { background:#14141f; color:#fff; border-radius:12px; }
[data-testid="stFileUploaderDropzone"] { background:#14141f; border:1px dashed #3a3a58; border-radius:14px; }
[data-testid="stMetric"] {
    background:#14141f; border:1px solid #25253a; border-radius:14px; padding:12px 16px;
}

.footer { text-align:center; color:#7c7c92; font-size:13px; margin: 50px 0 10px 0; }
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
            "model": MODELS_DIR / "best_model.pkl",
            "label_encoder": MODELS_DIR /  "label_encoder.pkl",
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
    st.markdown(
        H(f"""
        <div class="side-logo">
            
            <div class="name">{html.escape(PROFILE['name'])}</div>
            <div class="role">{html.escape(PROFILE['role'])}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )
    st.divider()
    st.radio("Navigation", list(LABELS), key="page", label_visibility="collapsed")
    st.divider()
    st.caption("MODEL STATUS")
    for p in PROJECTS.values():
        dot = "🟢" if project_ready(p) else "🔴"
        st.markdown(f"<div class='side-status'>{dot} {html.escape(p['title'])}</div>",
                    unsafe_allow_html=True)

# =====================================================================
# 9. HOME PAGE
# =====================================================================


def render_home():
    theme("#7f5af0", "#2cb67d")

    st.markdown(
        H(f"""
        <div class="hero">
            <div class="hero-hi">👋 Hello, I'm</div>
            <h1>{html.escape(PROFILE['name'])}</h1>
            <div class="hero-role">{html.escape(PROFILE['role'])}</div>
            <p>{html.escape(PROFILE['about'])}</p>
            <div style="margin-top:14px">{chips(PROFILE['skills'])}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    ready = sum(project_ready(p) for p in PROJECTS.values())
    m1, m2, m3 = st.columns(3)
    m1.metric("Projects", len(PROJECTS))
    m2.metric("Live demos ready", ready)
    m3.metric("Domain", "NLP, ML, CV")

    st.markdown("<div class='section-title'>🧪 Featured Projects</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Click a project to try the live demo.</div>",
                unsafe_allow_html=True)

    items = list(PROJECTS.items())
    for start in range(0, len(items), 3):
        cols = st.columns(3, gap="large")
        chunk = items[start:start + 3]
        for col, (k, p) in zip(cols, chunk):
            c1, c2 = p["colors"]
            ok = project_ready(p)
            badge = "<span class='status ok'>● Live</span>" if ok else \
                    "<span class='status bad'>● Model missing</span>"
            with col:
                st.markdown(
                    H(f"""
                    <div class="proj-card" style="--c1:{c1}; --c2:{c2};">
                        <div class="proj-top"><div class="proj-icon">{p['icon']}</div>{badge}</div>
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

    st.markdown("<div class='section-title'>📬 Get in touch</div>", unsafe_allow_html=True)
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
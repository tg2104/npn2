# app (3).py — Streamlit app with AI Helper (LLM) + Detoxify moderation + hotel-only guard
import io
import os
import re
import json
import contextlib
from typing import Dict, Any, List, Optional
import urllib.parse

import streamlit as st
import pandas as pd
import numpy as np

# ------- Your engine (unchanged import) -------
import recommender_engine as eng  # uses df1, mlb, SUB_TO_PARENT, hybrid_recommender

# ---------------- UI Setup ----------------
st.set_page_config(page_title="Hotel Recommender", layout="wide")
st.title("🏨 Hotel Recommendation Engine")

# ---------------- Helpers ----------------
def safe_float(x, default=None):
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default

def _flatten_city_values(series) -> list:
    cities = []
    if series is None:
        return cities
    for v in series.dropna():
        if isinstance(v, (set, list, tuple)):
            for x in v:
                x_str = str(x).strip()
                if x_str:
                    cities.append(x_str)
        else:
            v_str = str(v).strip()
            if v_str:
                cities.append(v_str)
    seen = set()
    out = []
    for c in cities:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            out.append(c)
    return out

def city_options():
    if "city" in eng.df1.columns:
        vals = _flatten_city_values(eng.df1["city"])
        return sorted(set(vals))
    return []

def get_default_city(opts):
    if opts:
        return opts[0]
    return "New York"

def _google_search_url(name: str, city: Optional[str] = None) -> str:
    # Always add country "US" as requested
    q = f"{name} {city} US" if city else f"{name} US"
    return f"https://www.google.com/search?q={urllib.parse.quote_plus(q)}"

@st.cache_data(show_spinner=False)
def get_recs_cached(user_city, min_rating, min_sentiment, min_recency,
                    desired_categories, category_logic, desired_segments,
                    user_lat, user_lon, within_km, top_k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        recs = eng.hybrid_recommender(
            user_city=user_city,
            min_rating=min_rating,
            min_sentiment=min_sentiment,
            min_recency=min_recency,
            desired_categories=desired_categories,
            category_logic=category_logic,
            desired_segments=desired_segments,
            user_lat=user_lat,
            user_lon=user_lon,
            within_km=within_km,
            top_k=top_k
        )
    return recs

# ----- Similarity helper -----
from sklearn.metrics.pairwise import cosine_similarity

def compute_top_similar_with_fallback(selected_name: str,
                                      pool_df: pd.DataFrame,
                                      k: int = 5,
                                      include_selected: bool = True) -> pd.DataFrame:
    if pool_df is None or pool_df.empty or "name" not in pool_df.columns:
        return pd.DataFrame()

    pool_df = pool_df.copy()
    sel_row = pool_df.loc[pool_df["name"] == selected_name]
    sel_city = sel_row["city"].iloc[0] if not sel_row.empty and "city" in sel_row.columns else None

    picked_names = []
    parts = []

    if include_selected and not sel_row.empty:
        parts.append(sel_row.head(1))
        picked_names.append(selected_name)

    base_cols = ["reviews.rating", "sentiment_score", "recency_score",
                 "segment_match_score", "distance_km"]
    try:
        mlb_classes = list(getattr(eng, "mlb").classes_)
    except Exception:
        mlb_classes = []
    cat_cols = [c for c in pool_df.columns if c in mlb_classes]
    sim_cols = [c for c in (base_cols + cat_cols) if c in pool_df.columns]

    if sim_cols and (selected_name in set(pool_df["name"])) and pool_df.shape[0] > 1:
        feats = pool_df.set_index("name")[sim_cols].fillna(0.0)
        if selected_name in feats.index and feats.shape[1] > 0:
            v = feats.loc[[selected_name]].values
            sims = cosine_similarity(v, feats.values).ravel()
            sim_series = pd.Series(sims, index=feats.index)
            if selected_name in sim_series.index:
                sim_series = sim_series.drop(index=selected_name)
            sim_top = sim_series.sort_values(ascending=False)
            need = max(k - len(picked_names), 0)
            sim_top_names = sim_top.head(need).index.tolist()
            if sim_top_names:
                part = (pool_df[pool_df["name"].isin(sim_top_names)]
                        .set_index("name")
                        .loc[sim_top_names]
                        .reset_index())
                parts.append(part)
                picked_names.extend(sim_top_names)

    remaining = max(k - len(picked_names), 0)
    if remaining > 0 and sel_city is not None and "city" in pool_df.columns:
        same_city_pool = pool_df[
            (pool_df["city"].astype(str).str.strip().str.lower() == str(sel_city).strip().lower())
            & (~pool_df["name"].isin(picked_names))
        ]
        if not same_city_pool.empty:
            same_city_fill = same_city_pool.sort_values("hybrid_score", ascending=False).head(remaining)
            parts.append(same_city_fill)
            picked_names.extend(same_city_fill["name"].tolist())

    remaining = max(k - len(picked_names), 0)
    if remaining > 0:
        global_pool = pool_df[~pool_df["name"].isin(picked_names)]
        if not global_pool.empty:
            global_fill = global_pool.sort_values("hybrid_score", ascending=False).head(remaining)
            parts.append(global_fill)
            picked_names.extend(global_fill["name"].tolist())

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["name"], keep="first").head(k).reset_index(drop=True)
    return combined

# =========================
#           AI HELPER
# =========================

# --- Hotel-only guard ---
_HOTEL_WORDS = [
    "hotel","hotels","resort","accommodation","stay","room","suite","lodging","motel","inn","hostel"
]
def is_hotel_related(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in _HOTEL_WORDS)

# --- Simple profanity / abuse list (fast, local) ---
_PROFANITY = [
    "fuck","shit","bitch","bastard","asshole","slut","whore",
    "retard","idiot","moron","dumbass","cunt"
]
def contains_abuse(text: str) -> bool:
    t = (text or "").lower()
    return any(bad in t for bad in _PROFANITY)

# --- Detoxify moderation (local, free, no API key) ---
try:
    from detoxify import Detoxify
    _detox_available = True
except Exception:
    Detoxify = None
    _detox_available = False

@st.cache_resource(show_spinner=False)
def _load_detox_model():
    if not _detox_available:
        return None
    try:
        return Detoxify('original')  # English fast model
    except Exception:
        return None

_DETOX_MODEL = _load_detox_model()

# Thresholds you can tune
_DETOX_THRESHOLDS = {
    "toxicity": 0.80,
    "severe_toxicity": 0.70,
    "obscene": 0.80,
    "insult": 0.80,
    "identity_attack": 0.60,
    "threat": 0.50,
}

def detoxify_block(text: str) -> Optional[str]:
    if not _DETOX_MODEL:
        return None
    try:
        scores = _DETOX_MODEL.predict(text or "")
        for k, th in _DETOX_THRESHOLDS.items():
            if k in scores and float(scores[k]) >= th:
                return "Your message seems abusive/toxic. Please rephrase."
    except Exception:
        return None
    return None

def moderate_prompt(prompt: str) -> (bool, str):
    """
    Order: 1) Profanity  2) Detoxify  3) Hotel-related
    """
    if contains_abuse(prompt):
        return False, "Your message appears to include abusive language. Please rephrase and try again."
    m = detoxify_block(prompt)
    if m:
        return False, m
    if not is_hotel_related(prompt):
        return False, "Please write related to hotel recommendation. I am only trained for it."
    return True, ""

# --- Normalizers & location-only detection ---
def _normalize_text_for_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return f" {s} "

def _guess_city_from_prompt(prompt: str) -> Optional[str]:
    m = re.search(r'\b(?:in|at|near|around)\s+([a-z][a-z\s\-]+)\b', prompt, flags=re.I)
    if m:
        cand = m.group(1).strip(" ,.;:!?").strip()
        return " ".join(w.capitalize() for w in re.split(r"\s+", cand))
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]*", prompt)
    preps = {"in","at","near","around","for","to","of","the"}
    hotel_words = set(_HOTEL_WORDS)
    candidates = [t for t in tokens if t.lower() not in preps and t.lower() not in hotel_words]
    if candidates:
        return " ".join(candidates[-2:]) if len(candidates) >= 2 else candidates[-1]
    return None

_LOCATION_HINT_KEYWORDS = {
    "star","rating","ratings","sentiment","review","reviews","positive","good","excellent",
    "recent","new","latest","within","km","distance",
    "lat","lon","longitude","latitude",
    "family","business","couple","friends","pet","solo","group","honeymoon",
    "nightlife","club","bar","shopping","mall","market","restaurant","cafe","food","airport","tour","attraction"
}
_PREPOSITIONS = {"in","at","near","around","for","to","of","the"}

def _looks_like_location_only_pattern(prompt: str) -> bool:
    lo = (prompt or "").strip().lower()
    simple_pairs = [
        r"^[a-z\s\-]+?\s+(hotels?|resorts?|accommodation|lodging)$",
        r"^(hotels?|resorts?|accommodation|lodging)\s+[a-z\s\-]+?$",
    ]
    for pat in simple_pairs:
        if re.match(pat, lo):
            return True
    if re.match(r"^(hotels?|resorts?|accommodation|lodging)\s+(in|at|near|around)\s+[a-z\s\-]+$", lo):
        return True
    return False

def _is_only_location_query(prompt: str, city: Optional[str]) -> bool:
    if _looks_like_location_only_pattern(prompt):
        return True
    t = _normalize_text_for_match(prompt).strip()
    for w in _HOTEL_WORDS:
        t = t.replace(f" {w} ", " ")
    for w in _PREPOSITIONS:
        t = t.replace(f" {w} ", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if re.search(r"\d", t):
        return False
    for kw in _LOCATION_HINT_KEYWORDS:
        if f" {kw} " in f" {t} ":
            return False
    target_city = city or _guess_city_from_prompt(prompt) or ""
    if target_city:
        c_norm = _normalize_text_for_match(target_city).strip()
        t = f" {t} ".replace(c_norm, " ").strip()
        t = re.sub(r"\s+", " ", t)
    return len(t) == 0

# --- LLM extractors (Gemini/OpenAI) with graceful fallback ---
def _extract_with_gemini(prompt: str) -> Optional[Dict[str, Any]]:
    api_key = "AIzaSyAYYJS-pSg0yOvXgNXEITO3yCUnRhHDWRY"  # hardcoded
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        schema = (
            "Return ONLY a compact JSON with keys: "
            "city, min_rating (0-5), min_sentiment (0-1), min_recency (0-1), "
            "segments (array from [Family,Couple,Business,Friends,Pet,Solo] or 'All'), "
            "categories (array; include 'Hotels' if applicable), within_km (number or null), "
            "lat (number or null), lon (number or null), top_k (int, default 5)."
        )
        resp = model.generate_content(
            f"Extract hotel search filters from: {prompt}\n{schema}\nRespond with JSON only."
        )
        text = (resp.text or "").strip()
        j = json.loads(text)
        return j
    except Exception:
        return None

def _extract_with_openai(prompt: str) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        SYS = (
            "You convert natural language hotel requests into filters. "
            "Return ONLY JSON keys: city, min_rating, min_sentiment, min_recency, "
            "segments (array or 'All'), categories (array; include 'Hotels' if applicable), "
            "within_km, lat, lon, top_k."
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYS},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        txt = resp.choices[0].message.content.strip()
        j = json.loads(txt)
        return j
    except Exception:
        return None

def _normalize_num(v, lo=None, hi=None, default=None):
    try:
        x = float(v)
        if lo is not None: x = max(lo, x)
        if hi is not None: x = min(hi, x)
        return x
    except Exception:
        return default

def _match_existing_city(prompt: str) -> Optional[str]:
    if "city" in eng.df1.columns:
        candidates = _flatten_city_values(eng.df1["city"])
    else:
        candidates = []
    t = _normalize_text_for_match(prompt)
    for c in sorted(candidates, key=lambda s: len(s), reverse=True):
        c_norm = _normalize_text_for_match(c)
        if c_norm in t:
            return c
    return _guess_city_from_prompt(prompt)

def _apply_minimums(params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params or {})
    params["min_rating"] = 0.0
    params["min_sentiment"] = 0.0
    params["min_recency"] = 0.0
    params["segments"] = ["All"]
    params["categories"] = ["Hotels"]
    params["within_km"] = None
    return params

def _extract_local(prompt: str) -> Dict[str, Any]:
    t = str(prompt or "")
    lo = t.lower()
    city = _match_existing_city(t)

    if _is_only_location_query(prompt, city):
        return _apply_minimums({
            "city": city,
            "lat": None,
            "lon": None,
            "top_k": 5
        })

    m = re.search(r'(\d(?:\.\d)?)\s*star', lo) or re.search(r'rating\s*(?:>=|at\s*least|minimum|above)\s*(\d(?:\.\d)?)', lo)
    min_rating = _normalize_num(m.group(1), 0, 5, 4.0) if m else (4.0 if "high" in lo or "best" in lo else 3.5)

    if any(w in lo for w in ["very positive", "excellent reviews", "amazing reviews", "top reviews"]):
        min_sentiment = 0.85
    elif any(w in lo for w in ["good reviews", "positive reviews", "nice reviews"]):
        min_sentiment = 0.7
    else:
        min_sentiment = 0.6

    if any(w in lo for w in ["recent", "new", "latest"]):
        min_recency = 0.6
    elif any(w in lo for w in ["not older than", "last year"]):
        min_recency = 0.4
    else:
        min_recency = 0.3

    segs = []
    if "family" in lo: segs.append("Family")
    if "business" in lo: segs.append("Business")
    if "couple" in lo or "honeymoon" in lo: segs.append("Couple")
    if "friends" in lo or "group" in lo: segs.append("Friends")
    if "pet" in lo or "dog" in lo or "cat" in lo: segs.append("Pet")
    if "solo" in lo or "single" in lo: segs.append("Solo")
    if not segs: segs = ["All"]

    cats = ["Hotels"]
    if "nightlife" in lo or "club" in lo or "bar" in lo: cats.append("Entertainment")
    if "shopping" in lo or "mall" in lo or "market" in lo: cats.append("Shopping")
    if "restaurant" in lo or "cafe" in lo or "food" in lo: cats.append("Restaurants")
    if "airport" in lo or "tour" in lo or "attraction" in lo: cats.append("Travel")

    wkm = None
    m = re.search(r'within\s+(\d+(?:\.\d+)?)\s*km', lo)
    if m:
        wkm = _normalize_num(m.group(1), 0, 200, None)
    elif any(w in lo for w in ["near", "close", "walk"]):
        wkm = 5.0

    lat = None
    lon = None
    mlat = re.search(r'lat(?:itude)?\s*[:=]?\s*(-?\d+(?:\.\d+)?)', lo)
    mlon = re.search(r'lon(?:gitude)?\s*[:=]?\s*(-?\d+(?:\.\d+)?)', lo)
    if mlat and mlon:
        lat = _normalize_num(mlat.group(1), -90, 90, None)
        lon = _normalize_num(mlon.group(1), -180, 180, None)

    return {
        "city": city,
        "min_rating": float(min_rating),
        "min_sentiment": float(min_sentiment),
        "min_recency": float(min_recency),
        "segments": segs,
        "categories": list(dict.fromkeys(cats)),
        "within_km": wkm,
        "lat": lat,
        "lon": lon,
        "top_k": 5
    }

def extract_filters_with_llm(prompt: str) -> Dict[str, Any]:
    j = _extract_with_gemini(prompt) or _extract_with_openai(prompt)
    if not j:
        return _extract_local(prompt)

    out = {
        "city": j.get("city") or _match_existing_city(prompt),
        "min_rating": _normalize_num(j.get("min_rating"), 0, 5, None),
        "min_sentiment": _normalize_num(j.get("min_sentiment"), 0, 1, None),
        "min_recency": _normalize_num(j.get("min_recency"), 0, 1, None),
        "segments": (j.get("segments") if isinstance(j.get("segments"), list) else None),
        "categories": j.get("categories"),
        "within_km": _normalize_num(j.get("within_km"), 0, 200, None),
        "lat": _normalize_num(j.get("lat"), -90, 90, None),
        "lon": _normalize_num(j.get("lon"), -180, 180, None),
        "top_k": int(j.get("top_k") or 5)
    }

    if _is_only_location_query(prompt, out["city"]):
        out = _apply_minimums(out)
    else:
        if out["min_rating"] is None:
            out["min_rating"] = 4.0
        if out["min_sentiment"] is None:
            out["min_sentiment"] = 0.6
        if out["min_recency"] is None:
            out["min_recency"] = 0.3
        if not out.get("segments"):
            out["segments"] = ["All"]
        if not out.get("categories"):
            out["categories"] = ["Hotels"]

    return out

def ai_helper_to_recs(prompt: str):
    ok, msg = moderate_prompt(prompt)
    if not ok:
        st.warning(msg)
        return

    # Extract filters
    params = extract_filters_with_llm(prompt)

    # FINAL safety: enforce location-only minimums again just before using
    if _is_only_location_query(prompt, params.get("city")):
        params = _apply_minimums(params)

    # Resolve city with a sensible default
    opts = city_options()
    default_city = get_default_city(opts) if opts else "New York"
    user_city = params["city"] or default_city

    cats = params["categories"] or ["Hotels"]
    segs = params["segments"] or ["All"]

    # final call to engine — always 5 for AI helper
    recs = get_recs_cached(
        user_city=user_city,
        min_rating=params["min_rating"],
        min_sentiment=params["min_sentiment"],
        min_recency=params["min_recency"],
        desired_categories=cats,
        category_logic="any",
        desired_segments=segs,
        user_lat=params["lat"],
        user_lon=params["lon"],
        within_km=params["within_km"],
        top_k=5
    )

    st.session_state["recs"] = recs

    # Build a larger pool for similarities (cap)
    total_names = 200
    try:
        total_names = int(eng.df1["name"].nunique())
    except Exception:
        pass
    pool_size = min(500, total_names if total_names else 200)

    st.session_state["pool"] = get_recs_cached(
        user_city=user_city,
        min_rating=params["min_rating"],
        min_sentiment=params["min_sentiment"],
        min_recency=params["min_recency"],
        desired_categories=cats,
        category_logic="any",
        desired_segments=segs,
        user_lat=params["lat"],
        user_lon=params["lon"],
        within_km=params["within_km"],
        top_k=pool_size
    )

    if recs is not None and not recs.empty:
        st.session_state["selected_hotel"] = recs["name"].iloc[0]

    with st.expander("🔎 AI Helper • extracted filters", expanded=False):
        st.json(params)

# =========================
#     Sidebar Filters
# =========================
st.sidebar.header("Filters (optional)")

opts = city_options()
default_city = get_default_city(opts) if opts else "New York"
city_choice = st.sidebar.selectbox(
    "City",
    options=opts + ["Other…"],
    index=(opts + ["Other…"]).index(default_city) if default_city in (opts + ["Other…"]) else 0
)
if city_choice == "Other…":
    user_city = st.sidebar.text_input("Type a city name", value=default_city)
else:
    user_city = city_choice

min_rating = st.sidebar.slider("Minimum Avg Rating", 0.0, 5.0, 4.0, 0.1)
min_sentiment = st.sidebar.slider("Minimum Sentiment Score", 0.0, 1.0, 0.6, 0.05)
min_recency = st.sidebar.slider("Minimum Recency Score", 0.0, 1.0, 0.3, 0.05)

try:
    base_classes = list(getattr(eng, "mlb").classes_)
except Exception:
    base_classes = []
try:
    sub_to_parent_keys = sorted(list(getattr(eng, "SUB_TO_PARENT").keys()))
except Exception:
    sub_to_parent_keys = []

category_options = ["All"] + base_classes + sub_to_parent_keys
desired_categories = st.sidebar.multiselect("Desired Categories", category_options, default=["Hotels", "nightlife"])
category_logic = st.sidebar.radio("Category Logic", ["any", "all"], index=0)

segment_options = ["All", "Family", "Couple", "Business", "Friends", "Pet", "Solo"]
desired_segments = st.sidebar.multiselect("Audience Segments", segment_options, default=["Family", "Business"])

with st.sidebar.expander("Advanced: Location (optional)"):
    user_lat_txt = st.text_input("User Latitude", value="")
    user_lon_txt = st.text_input("User Longitude", value="")
    within_km_txt = st.text_input("Within km", value="10")

user_lat = safe_float(user_lat_txt, None)
user_lon = safe_float(user_lon_txt, None)
within_km = safe_float(within_km_txt, None)
top_k = st.sidebar.slider("Number of recommendations", 1, 50, 5, 1)

# =========================
#     State + Actions
# =========================
if "recs" not in st.session_state:
    st.session_state["recs"] = None
if "pool" not in st.session_state:
    st.session_state["pool"] = None
if "selected_hotel" not in st.session_state:
    st.session_state["selected_hotel"] = None

# --- Row: manual vs AI actions
colA, colB = st.columns([1, 1])

with colA:
    if st.button("Recommend", type="primary", use_container_width=True):
        uc = (user_city or default_city).strip()
        cats = desired_categories if desired_categories else ["All"]
        segs = desired_segments if desired_segments else ["All"]

        st.session_state["recs"] = get_recs_cached(
            uc, min_rating, min_sentiment, min_recency,
            cats, category_logic, segs, user_lat, user_lon, within_km, top_k
        )

        try:
            total_names = int(eng.df1["name"].nunique())
        except Exception:
            total_names = 200
        pool_size = min(500, total_names if total_names else 200)

        st.session_state["pool"] = get_recs_cached(
            uc, min_rating, min_sentiment, min_recency,
            cats, category_logic, segs, user_lat, user_lon, within_km, pool_size
        )

with colB:
    st.markdown("**🤖 AI Helper**")
    prompt = st.text_input(
        "Describe what you want (e.g., “family-friendly 4⭐ hotels near the Strip in Las Vegas within 5 km with great reviews”).",
        value="",
        placeholder="Type your request for hotels…"
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🔎 Search with AI Helper", use_container_width=True):
            ai_helper_to_recs(prompt)
    with c2:
        if st.button("Clear", use_container_width=True):
            st.session_state["recs"] = None
            st.session_state["pool"] = None
            st.session_state["selected_hotel"] = None

# =========================
#        Display
# =========================
def _with_search_links(df: pd.DataFrame) -> pd.DataFrame:
    """Add a clickable 🔎 US web search link per hotel."""
    if df is None or df.empty or "name" not in df.columns:
        return df
    df = df.copy()
    # Try to include city if present to improve search relevance
    city_col = "city" if "city" in df.columns else None
    df["🔎 Web Search (US)"] = [
        _google_search_url(row["name"], row[city_col] if city_col else None)
        for _, row in df.iterrows()
    ]
    return df

if st.session_state.get("recs") is None or st.session_state["recs"].empty:
    st.info("Set filters and click **Recommend**, or try **Search with AI Helper**.")
else:
    st.subheader("Recommended Hotels (Top 5 from your request)" if top_k == 5 else "Recommended Hotels")

    # Add a link column for US web search
    recs_disp = _with_search_links(st.session_state["recs"])

    # If Streamlit supports LinkColumn in your version, this will render as clickable links:
    try:
        st.dataframe(
            recs_disp,
            use_container_width=True,
            column_config={
                "🔎 Web Search (US)": st.column_config.LinkColumn("🔎 Web Search (US)", display_text="Open")
            }
        )
    except Exception:
        # Fallback: just show the dataframe; links may still be clickable in many environments
        st.dataframe(recs_disp, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 5 Similar (with fallback)")

    # Always show a selector so you can change the anchor hotel
    rec_names = st.session_state["recs"]["name"].tolist() if st.session_state["recs"] is not None else []
    current = st.session_state.get("selected_hotel")
    if rec_names:
        try:
            default_idx = rec_names.index(current) if current in rec_names else 0
        except Exception:
            default_idx = 0
        selected_now = st.selectbox(
            "Pick a hotel to find similar (change anytime)",
            rec_names,
            index=default_idx,
            key="similar_picker"
        )
        # Update the session_state anchor when user changes selection
        st.session_state["selected_hotel"] = selected_now

    if st.session_state.get("selected_hotel"):
        similar = compute_top_similar_with_fallback(
            st.session_state["selected_hotel"],
            st.session_state.get("pool"),
            k=5,
            include_selected=True
        )
        if similar is None or similar.empty:
            st.info("No similar hotels found and no fallback candidates available.")
        else:
            st.write(
                f"Showing up to {len(similar)} hotels starting with **{st.session_state['selected_hotel']}**, "
                "then same-city and best-overall to ensure at least 5 when possible."
            )

            # Add US web search links to Similar table too
            similar_disp = _with_search_links(similar)
            try:
                st.dataframe(
                    similar_disp,
                    use_container_width=True,
                    column_config={
                        "🔎 Web Search (US)": st.column_config.LinkColumn("🔎 Web Search (US)", display_text="Open")
                    }
                )
            except Exception:
                st.dataframe(similar_disp, use_container_width=True)

    # Download
    csv = st.session_state["recs"].to_csv(index=False).encode("utf-8")
    st.download_button("Download recommendations as CSV", csv, "recommendations.csv", "text/csv")
